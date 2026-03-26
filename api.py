"""
AI Scanner API – model-only backend for use from Flutter (or any client).
Exposes receipt scanning and text scanning via REST. Run this server and call it from your app.
"""
import io
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import numpy as np

from scanner import extract_text, generate_output_from_text
from receipt_scanner import extract_receipt, add_receipt_to_expenses, load_expenses, ReceiptData

app = FastAPI(
    title="AI Scanner API",
    description="Receipt and text scanning for Flutter / mobile or web clients.",
    version="1.0.0",
)

# Allow Flutter web/mobile to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _image_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return img


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


# --- Endpoints ---

@app.get("/health")
def health():
    """Check if the API and OCR are available."""
    return {"status": "ok", "service": "ai-scanner"}


@app.post("/scan/receipt", response_model=ReceiptResponse)
async def scan_receipt(image: UploadFile = File(...)):
    """
    Upload a receipt image; get structured data (merchant, date, line items with product name and price).
    Prices are strict—only exact OCR values, no guessing.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (e.g. image/jpeg, image/png)")
    try:
        img = _image_from_upload(image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    receipt = extract_receipt(np.array(img))
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
        warnings=receipt.warnings,
        raw_text=receipt.raw_text,
    )


@app.post("/scan/text", response_model=TextScanResponse)
async def scan_text(image: UploadFile = File(...)):
    """Upload an image; get extracted text and a short summary."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        img = _image_from_upload(image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    text = extract_text(np.array(img))
    summary = generate_output_from_text(text)
    return TextScanResponse(text=text, summary=summary)


@app.post("/expenses")
async def add_expense(payload: ExpenseEntry):
    """Add a receipt to expenses (e.g. after scanning and confirming in Flutter)."""
    receipt = ReceiptData(
        merchant=payload.merchant,
        date=payload.date,
        items=[{"product": it.product, "price": it.price, "needs_review": it.needs_review} for it in payload.items],
        subtotal=payload.subtotal,
        tax=payload.tax,
        total="",
        currency=getattr(payload, "currency", ""),
        raw_text="",
        warnings=[],
    )
    add_receipt_to_expenses(receipt)
    return {"status": "added"}


@app.get("/expenses")
async def get_expenses():
    """Return all saved expenses (list of receipts)."""
    return {"expenses": load_expenses()}
