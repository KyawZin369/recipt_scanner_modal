"""
AI Scanner API – model-only backend for use from Flutter (or any client).
Exposes receipt scanning and text scanning via REST. Run this server and call it from your app.
"""
import io
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from PIL import Image
import numpy as np

from supabase_storage import (
    upload_image_to_bucket,
    upload_receipt_image_bytes,
    upload_receipt_image_to_bucket,
)

# Load .env early so imported modules can read env vars.
if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from scanner import extract_text, generate_output_from_text
from auth_users import (
    _get_users_collection,
    authenticate_user,
    create_user,
    logout_user,
    user_public_dict,
    verify_session_token,
)
from receipt_scanner import (
    close_mongo_client,
    create_mart_in_mongo,
    extract_receipt,
    add_receipt_to_expenses,
    get_last_mongo_error,
    get_mart_for_user,
    list_marts_from_mongo,
    load_expenses,
    load_expenses_from_mongo_only,
    _get_mongo_collection,
    ReceiptData,
)


def _raise_auth_error(err: str, *, login: bool = False) -> None:
    """Map auth layer errors to HTTP status; surface Mongo diagnostics for DB down."""
    if err == "Database unavailable":
        detail = get_last_mongo_error() or err
        raise HTTPException(status_code=503, detail=detail)
    if login:
        raise HTTPException(status_code=401, detail=err)
    raise HTTPException(status_code=400, detail=err)

_auth_bearer = HTTPBearer(auto_error=False)


def _require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_auth_bearer),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = verify_session_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Reload .env so uvicorn inherits MONGO_URI etc. even if shell env was empty at import time.
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
    yield
    close_mongo_client()


app = FastAPI(
    title="AI Scanner API",
    description="Receipt and text scanning for Flutter / mobile or web clients.",
    version="1.0.0",
    lifespan=_lifespan,
)

# Allow Flutter web/mobile to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(file: UploadFile) -> tuple[Image.Image, bytes]:
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return img, contents


# --- Response models (so Flutter knows the JSON shape) ---

class ReceiptItemOut(BaseModel):
    product: str
    price: str
    price_with_currency: str = ""  # e.g. "1600.00 Ks"; set by API on scan, optional when posting to /expenses
    needs_review: bool = False


class ReceiptResponse(BaseModel):
    merchant: str
    date: str
    currency: str  # Detected from receipt (e.g. Ks, USD, EUR)
    items: List[ReceiptItemOut]
    subtotal: str
    tax: str
    image_url: str = ""
    warnings: List[str]
    raw_text: str = ""


class TextScanResponse(BaseModel):
    text: str
    summary: str


class ExpenseEntry(BaseModel):
    merchant: str
    date: str
    items: List[ReceiptItemOut]
    subtotal: str
    tax: str
    currency: str = ""
    image_url: str = ""
    mart_id: Optional[str] = None  # optional; must belong to the authenticated user


class ReceiptImageUploadResponse(BaseModel):
    """Response after uploading a file to the Supabase `receipts` bucket."""
    image_url: str
    bucket: str
    path: str


class MartOut(BaseModel):
    """Mart / store entry (saved in MongoDB, logos in Supabase Storage under `marts/`)."""
    id: str
    name: str
    description: str = ""
    logo_url: str = ""
    created_at: str = ""


class UserOut(BaseModel):
    id: str
    username: str
    email: str


class RegisterBody(BaseModel):
    username: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str
    token_type: str = "bearer"


# --- Endpoints ---

@app.get("/health")
def health():
    """Check if the API and OCR are available."""
    return {"status": "ok", "service": "ai-scanner"}


@app.get("/health/db")
def health_db():
    """Mongo connectivity for receipts vs users (diagnostic; no secrets)."""
    receipts_ok = _get_mongo_collection() is not None
    users_ok = _get_users_collection() is not None
    err = get_last_mongo_error()
    return {
        "receipts_collection": receipts_ok,
        "users_collection": users_ok,
        "last_mongo_error": err if err else None,
    }


@app.post("/auth/register", response_model=AuthResponse)
def auth_register(body: RegisterBody):
    """Create a user (password stored as bcrypt hash; `user_token` holds the active JWT)."""
    result, err = create_user(
        username=body.username.strip(),
        email=body.email.strip(),
        password=body.password,
    )
    if err:
        _raise_auth_error(err, login=False)
    if not result:
        raise HTTPException(500, "Registration failed")
    u = result["user"]
    return AuthResponse(
        user=UserOut(id=u["id"], username=u["username"], email=u["email"]),
        access_token=result["access_token"],
    )


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(body: LoginBody):
    result, err = authenticate_user(email=body.email, password=body.password)
    if err:
        _raise_auth_error(err, login=True)
    if not result:
        raise HTTPException(500, "Login failed")
    u = result["user"]
    return AuthResponse(
        user=UserOut(id=u["id"], username=u["username"], email=u["email"]),
        access_token=result["access_token"],
    )


@app.post("/auth/logout")
def auth_logout(user: dict = Depends(_require_user)):
    """Invalidate server-side session (`user_token` cleared). Client should delete local token."""
    uid = user.get("id")
    if uid:
        logout_user(uid)
    return {"status": "ok"}


@app.get("/auth/me", response_model=UserOut)
def auth_me(user: dict = Depends(_require_user)):
    u = user_public_dict(user)
    return UserOut(id=u["id"], username=u["username"], email=u["email"])


@app.get("/marts")
def get_marts(user: dict = Depends(_require_user)):
    """List marts created by the authenticated user only."""
    marts = list_marts_from_mongo(user_id=user["id"])
    if marts is None:
        raise HTTPException(
            status_code=503,
            detail=get_last_mongo_error()
            or "MongoDB is unavailable. Check MONGO_URI and credentials.",
        )
    return {"marts": marts}


@app.post("/marts", response_model=MartOut)
async def create_mart(
    user: dict = Depends(_require_user),
    name: str = Form(..., min_length=1),
    description: str = Form(""),
    logo: Optional[UploadFile] = File(None),
):
    """
    Create a mart for the authenticated user: **name** (required), **description** (optional), **logo** (optional image).
    Multipart field names: `name`, `description`, `logo`.
    Logo is uploaded to Supabase under `marts/` in the configured bucket.
    """
    logo_url = ""
    if logo is not None and (logo.filename or "").strip():
        if not logo.content_type or not logo.content_type.startswith("image/"):
            raise HTTPException(
                400, "Logo must be an image (e.g. image/jpeg, image/png, image/webp)"
            )
        try:
            contents = await logo.read()
        except Exception as e:
            raise HTTPException(400, f"Could not read logo: {e}") from e
        if contents:
            fn = logo.filename or "logo.png"
            logo_url, _, _ = upload_image_to_bucket(contents, fn, folder="marts")
            if not logo_url:
                raise HTTPException(
                    503,
                    "Logo upload failed. Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, "
                    "and ensure the Storage bucket exists.",
                )

    doc = create_mart_in_mongo(
        name=name,
        description=description,
        logo_url=logo_url,
        user_id=user["id"],
    )
    if doc is None:
        raise HTTPException(
            status_code=503,
            detail=get_last_mongo_error()
            or "Could not save mart to MongoDB. Check MONGO_URI and credentials.",
        )
    return MartOut(
        id=doc["id"],
        name=doc["name"],
        description=doc.get("description", "") or "",
        logo_url=doc.get("logo_url", "") or "",
        created_at=doc.get("created_at", "") or "",
    )


@app.post("/storage/receipts/upload", response_model=ReceiptImageUploadResponse)
async def upload_receipt_image_to_storage(
    file: UploadFile = File(..., description="Receipt image (JPEG, PNG, WebP, etc.)"),
):
    """
    Upload an image to Supabase Storage (bucket from `SUPABASE_STORAGE_BUCKET`, default `receipts`).
    Returns a public URL (or signed URL if `SUPABASE_USE_SIGNED_URLS=1`).

    Multipart field name: **`file`** (same as Swagger UI / Flutter `MultipartFile`).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            400, "File must be an image (e.g. image/jpeg, image/png, image/webp)"
        )
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}") from e
    if not contents:
        raise HTTPException(400, "Empty file")

    name = file.filename or "receipt.jpg"
    image_url, object_path, bucket = upload_receipt_image_to_bucket(contents, name)
    if not image_url:
        raise HTTPException(
            503,
            "Supabase upload failed. Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY "
            "(service_role, not anon), create the Storage bucket, and restart the API.",
        )
    return ReceiptImageUploadResponse(
        image_url=image_url,
        bucket=bucket,
        path=object_path,
    )


@app.post("/scan/receipt", response_model=ReceiptResponse)
async def scan_receipt(image: UploadFile = File(...)):
    """
    Upload a receipt image; get structured data (merchant, date, line items with product name and price).
    Prices are strict—only exact OCR values, no guessing.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (e.g. image/jpeg, image/png)")
    try:
        img, contents = _read_upload(image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    receipt = extract_receipt(np.array(img))
    uploaded_name = image.filename or "receipt.jpg"
    receipt_image_url = upload_receipt_image_bytes(contents, uploaded_name)
    currency = getattr(receipt, "currency", "") or ""
    items_out = []
    for it in receipt.items:
        price = it.get("price", "")
        price_with_currency = f"{price} {currency}".strip() if currency else price
        items_out.append(ReceiptItemOut(
            product=it.get("product", ""),
            price=price,
            price_with_currency=price_with_currency,
            needs_review=it.get("needs_review", False),
        ))
    return ReceiptResponse(
        merchant=receipt.merchant,
        date=receipt.date,
        currency=currency,
        items=items_out,
        subtotal=receipt.subtotal,
        tax=receipt.tax,
        image_url=receipt_image_url,
        warnings=receipt.warnings,
        raw_text=receipt.raw_text,
    )


@app.post("/scan/text", response_model=TextScanResponse)
async def scan_text(image: UploadFile = File(...)):
    """Upload an image; get extracted text and a short summary."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        img, _ = _read_upload(image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    text = extract_text(np.array(img))
    summary = generate_output_from_text(text)
    return TextScanResponse(text=text, summary=summary)


@app.post("/expenses")
async def add_expense(
    payload: ExpenseEntry,
    user: dict = Depends(_require_user),
):
    """Add a receipt for the authenticated user (1 user : many receipts)."""
    mart_id = (payload.mart_id or "").strip()
    mart_name = ""
    if mart_id:
        doc = get_mart_for_user(user["id"], mart_id)
        if doc is None:
            raise HTTPException(
                status_code=400,
                detail="Unknown mart or mart does not belong to your account.",
            )
        mart_name = (doc.get("name") or "").strip()
    receipt = ReceiptData(
        merchant=payload.merchant,
        date=payload.date,
        items=[{"product": it.product, "price": it.price, "needs_review": it.needs_review} for it in payload.items],
        subtotal=payload.subtotal,
        tax=payload.tax,
        total="",
        currency=getattr(payload, "currency", ""),
        image_url=getattr(payload, "image_url", "") or "",
        raw_text="",
        warnings=[],
        mart_id=mart_id,
        mart_name=mart_name,
    )
    add_receipt_to_expenses(receipt, user_id=user["id"])
    return {"status": "added"}


@app.get("/expenses")
async def get_expenses(
    mart_id: Optional[str] = None,
    user: dict = Depends(_require_user),
):
    """Return saved expenses for the current user (MongoDB). Optional `mart_id` filters by mart."""
    mid = (mart_id or "").strip() or None
    receipts = load_expenses_from_mongo_only(user_id=user["id"], mart_id=mid)
    if receipts is None:
        raise HTTPException(
            status_code=503,
            detail=get_last_mongo_error()
            or "MongoDB is unavailable. Check MONGO_URI, Atlas IP allowlist, and credentials.",
        )
    return {"expenses": receipts}


@app.get("/receipts")
async def get_receipts(
    mart_id: Optional[str] = None,
    user: dict = Depends(_require_user),
):
    """Return receipts from MongoDB for the authenticated user. Optional `mart_id` filters (1 mart : many receipts)."""
    mid = (mart_id or "").strip() or None
    receipts = load_expenses_from_mongo_only(user_id=user["id"], mart_id=mid)
    if receipts is None:
        raise HTTPException(
            status_code=503,
            detail=get_last_mongo_error()
            or "MongoDB is unavailable. Check MONGO_URI, Atlas IP allowlist, and credentials.",
        )
    return {"receipts": receipts, "source": "mongodb"}
