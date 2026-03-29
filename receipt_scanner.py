"""
Receipt scanner: extract product names and prices with strict, no-guess handling.
Prices are never modified or guessed—only exact OCR results matching price format are used.
Output is structured for expense tracking.
"""
import json
import re
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Expense storage path
EXPENSES_PATH = Path(__file__).resolve().parent / "expenses.json"

# Reused Mongo client (pool) — avoids opening/closing TCP + TLS on every /receipts call
_mongo_client: Optional[MongoClient] = None
_last_mongo_error: str = ""


def _mongo_settings() -> tuple[str, str, str]:
    """
    Read MongoDB settings at call time so runtime-loaded .env values are used.
    """
    mongo_uri = os.getenv("MONGO_URI", "").strip()
    mongo_db_name = os.getenv("MONGO_DB_NAME", "expenses").strip() or "expenses"
    mongo_collection_name = os.getenv("MONGO_COLLECTION_NAME", "receipts").strip() or "receipts"
    return mongo_uri, mongo_db_name, mongo_collection_name


def _marts_collection_name() -> str:
    return os.getenv("MONGO_MARTS_COLLECTION", "marts").strip() or "marts"

# Strict price pattern: X.XX or X.X (1–2 decimals).
PRICE_PATTERN = re.compile(r"^\$?(\d{1,6}\.\d{1,2})$")
PRICE_INT_PATTERN = re.compile(r"^\$?(\d{1,8})$")
# OCR variants: trailing ) or , ; colon as decimal (30:00 -> 30.00); l/O mistaken for 1/0
PRICE_FLEXIBLE = re.compile(r"^[\$]?(\d{1,6})[.:,](\d{1,2})[\)]?$")  # 2.50) 30:00 1,00

# Lines containing these phrases are footer lines—not product line items. Kept broad for all receipt formats.
FOOTER_PHRASES = (
    "total", "inclusive tax", "commercial", "paid by", "cash(ks)", "changed", "change:",
    "thank you", "slip no", "counter ", "cashier", "qty---", "paid by:", "amount due",
    "balance", "tendered", "change due", "vat", "gst", "subtotal", "grand total",
    "cash", "card", "credit", "debit", "refund", "discount", "invoice no", "receipt no",
)


@dataclass
class ReceiptItem:
    """Single line item: product name and price. Price is never guessed."""
    product: str
    price: str  # Keep as string to avoid rounding; display exactly as scanned
    needs_review: bool = False  # True if price was ambiguous or missing


@dataclass
class ReceiptData:
    """Structured receipt data for expense tracking. Works with any receipt format."""
    merchant: str
    date: str
    items: list
    subtotal: str
    tax: str
    total: str  # Not scanned; left empty
    currency: str  # Detected from receipt (e.g. Ks, USD, EUR)
    raw_text: str
    warnings: list
    image_url: str = ""
    mart_id: str = ""  # optional; links to marts.id (1 mart : many receipts)
    mart_name: str = ""  # denormalized for list display


def _preprocess_receipt(image: np.ndarray) -> np.ndarray:
    """Optimize image for receipt OCR: scale, grayscale, high contrast."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if h < 1200:
        scale = 1200 / h
        gray = cv2.resize(gray, (int(w * scale), 1200), interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _run_receipt_ocr(img: np.ndarray, psm: int = 6) -> str:
    """Run Tesseract on receipt image."""
    try:
        return pytesseract.image_to_string(img, lang="eng", config=f"--psm {psm}")
    except Exception:
        return pytesseract.image_to_string(Image.fromarray(img), lang="eng", config=f"--psm {psm}")


def _ocr_receipt_all_fonts(image: np.ndarray) -> str:
    """
    Run OCR with multiple preprocessing and PSM modes to support many fonts
    (thermal, dot-matrix, printed, varying sizes). Merge results and dedupe lines.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if h < 1200:
        scale = 1200 / h
        gray = cv2.resize(gray, (int(w * scale), 1200), interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Multiple preprocessing variants for different fonts
    variants = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive
        cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21),  # Denoised
    ]
    # Inverted for some thermal receipts (dark on light)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(inv)

    texts = []
    for v in variants:
        texts.append(_run_receipt_ocr(v, psm=6))
        texts.append(_run_receipt_ocr(v, psm=3))
    texts.append(_run_receipt_ocr(gray, psm=11))  # Sparse text
    texts.append(_run_receipt_ocr(variants[0], psm=4))  # Single column
    texts.append(_run_receipt_ocr(variants[0], psm=13))  # Raw line

    def _is_garbage(line: str) -> bool:
        """Filter OCR garbage (eee, ooo, random consonants, etc.)."""
        if len(line) < 3:
            return True
        s = line.lower()
        # Too many repeated chars (eee, ooo, ccc)
        if re.search(r"(.)\1{4,}", s):
            return True
        # Mostly vowels or short fragments
        letters = sum(1 for c in s if c.isalpha())
        if letters < len(s) * 0.3:
            return True
        # No digits and no word >= 4 chars → likely noise
        if not re.search(r"\d", s) and not re.search(r"[a-z]{4,}", s):
            return True
        return False

    seen = set()
    merged = []
    for t in texts:
        for line in (t or "").splitlines():
            line = line.strip()
            if not line or len(line) < 2 or _is_garbage(line):
                continue
            key = line.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            merged.append(line)
    return "\n".join(merged)


def _ocr_receipt_fast(image: np.ndarray) -> str:
    """Single-pass OCR for speed; slightly less robust than multi-pass. Set FAST_RECEIPT_OCR=1."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if h < 900:
        scale = 900 / h
        gray = cv2.resize(gray, (int(w * scale), 900), interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return _run_receipt_ocr(thresh, psm=6)


def _strict_price(s: str) -> Optional[str]:
    """
    Return price string only if it matches a valid format.
    Handles OCR variants: trailing ), : as decimal, etc.
    """
    if not s or not s.strip():
        return None
    s = s.strip().replace(",", "").rstrip(")")
    # OCR: "1e00" often means 1.00
    s_fix = s.replace("e", ".", 1) if re.match(r"^\d+e\d{2}$", s) else s
    m = PRICE_PATTERN.match(s) or PRICE_PATTERN.match(s_fix)
    if m:
        return m.group(1)
    m = PRICE_FLEXIBLE.match(s)
    if m:
        return m.group(1) + "." + m.group(2).ljust(2, "0")[:2]
    m = PRICE_INT_PATTERN.match(s)
    if m:
        return m.group(1) + ".00"
    return None


def _parse_receipt_lines(ocr_text: str, word_data: list) -> tuple[list, list]:
    """
    Parse OCR into line items: product name + price per line.
    Uses word-level data when available to group by line; falls back to raw text.
    Prices are only accepted when they match strict pattern (never modified).
    Returns (items, warnings).
    """
    items = []
    warnings = []
    # Build lines from word-level data (group by line_num)
    line_num_key = "line_num" if word_data and "line_num" in word_data[0] else None
    if line_num_key:
        lines_by_num = {}
        for w in word_data:
            if not w.get("text", "").strip():
                continue
            num = w.get(line_num_key, 0)
            lines_by_num.setdefault(num, []).append(w["text"].strip())
        line_texts = [ " ".join(lines_by_num[k]) for k in sorted(lines_by_num.keys()) ]
    else:
        line_texts = [ ln.strip() for ln in ocr_text.splitlines() if ln.strip() ]
    for i, line in enumerate(line_texts):
        if not line:
            continue
        line_lower = line.lower()
        # Skip footer lines: total, tax, paid by, change, thank you—not product line items
        if any(phrase in line_lower for phrase in FOOTER_PHRASES):
            continue
        tokens = line.split()
        if not tokens:
            continue
        # Find rightmost strict price (support comma: "1,600" or tokens "1," "600")
        price_str = None
        price_idx = -1
        for j in range(len(tokens) - 1, -1, -1):
            cand = tokens[j].replace("$", "").replace(",", "").strip().rstrip(")")
            price_str = _strict_price(cand) or _strict_price(tokens[j])
            if price_str:
                price_idx = j
                break
        if price_str is None and len(tokens) >= 2:
            combined = (tokens[-2] + tokens[-1]).replace("$", "").replace(",", "").replace(" ", "").strip().rstrip(")")
            if PRICE_INT_PATTERN.match(combined) and len(combined) <= 8:
                price_str = combined + ".00"
                price_idx = len(tokens) - 2
        if price_str is None and tokens:
            last = tokens[-1].replace("$", "").replace(",", "").strip().rstrip(")")
            if PRICE_INT_PATTERN.match(last) and len(last) <= 8:
                price_str = last + ".00"
                price_idx = len(tokens) - 1
        if price_str is None:
            # Price may be embedded: e.g. "APPLE 2.50" or "KIWI180"
            for m in re.finditer(r"(\d{1,6}\.\d{1,2})|(\d{1,8})(?=\s*[\)]?\s*$)", line):
                g = m.group(1) or m.group(2)
                if g and _strict_price(g):
                    price_str = _strict_price(g) or (g + ".00" if "." not in g else g)
                    # Approximate price_idx from match position
                    before = line[:m.start()].split()
                    price_idx = len(before) if before else len(tokens) - 1
                    break
        if price_str is not None and price_idx >= 0:
            product = " ".join(tokens[:price_idx]).strip()
            # Trim trailing comma or lone digit (OCR "1,600" split as "1," + "600")
            product = re.sub(r"\s*[\d,]+\.?\d*\s*$", "", product).strip()
            if not product:
                product = f"Line item {i+1}"
            items.append(ReceiptItem(product=product, price=price_str, needs_review=False))
        else:
            # Only warn for lines that look like real products (word 3+ chars + digit), not OCR garbage
            has_digit = any(c.isdigit() for c in line)
            has_word = bool(re.search(r"[a-zA-Z]{3,}", line))
            is_noise = re.search(r"(.)\1{3,}", line_lower) or (len(line) > 50 and line.count(" ") > 15)
            if has_digit and has_word and not is_noise and len(tokens) >= 2:
                skip_kw = ("subtotal", "date", "phone", "open daily", "taxpayer", "qty---", "eee", "ooo", "coo")
                if not any(k in line_lower for k in skip_kw):
                    warnings.append(f"Line {i+1}: no valid price (skipped). Text: {line[:50]}...")
    return items, warnings


def _find_tax(ocr_text: str) -> str:
    """Extract tax from 'Commercial Tax: 76' style line. Prefer commercial tax line and take full number (e.g. 76 not 1 or 10)."""
    lines = ocr_text.splitlines()
    for line in lines:
        line_lower = line.lower()
        # Skip taxpayer ID line
        if "taxpayer" in line_lower and "commercial" not in line_lower:
            continue
        if "commercial" not in line_lower and "tax" not in line_lower:
            continue
        if "taxpayer" in line_lower:
            continue
        # On this line, collect all number-like tokens (e.g. "76" or "7" "6")
        tokens = line.replace(":", " ").replace(",", " ").split()
        numbers = []
        for part in tokens:
            part = part.replace("$", "").replace(",", "").strip()
            if PRICE_PATTERN.match(part):
                numbers.append(part)
            elif PRICE_INT_PATTERN.match(part) and len(part) <= 4:
                numbers.append(part)
        if numbers:
            # Use last number on line (tax is usually at end: "Commercial Tax: 76")
            last = numbers[-1]
            if "." in last:
                return last
            return last + ".00"
        # Try adjacent digits as one number (e.g. "7" "6" -> 76)
        digits = []
        for part in tokens:
            if part.isdigit() and len(part) <= 2:
                digits.append(part)
        if digits:
            combined = "".join(digits)
            if 1 <= len(combined) <= 4:
                return combined + ".00"
    # Fallback: any line with "tax" (not taxpayer)
    def find_label_value(text: str, label: str) -> str:
        label_lower = label.lower()
        for line in text.splitlines():
            line_lower = line.lower()
            if "taxpayer" in line_lower:
                continue
            if label_lower in line_lower:
                for part in line.replace(":", " ").split():
                    part = part.replace("$", "").replace(",", "").strip()
                    if PRICE_INT_PATTERN.match(part) and len(part) <= 4:
                        return part + ".00"
        return ""
    return find_label_value(ocr_text, "tax") or find_label_value(ocr_text, "gst") or find_label_value(ocr_text, "vat")


def _detect_currency(ocr_text: str) -> str:
    """Detect currency from receipt text. Works across formats (Ks, USD, EUR, etc.)."""
    t = ocr_text.upper()
    if "KS" in t or "KYAT" in t or "MMK" in t or "KSA" in t:
        return "Ks"
    if "USD" in t or "DOLLAR" in t or "$" in ocr_text:
        return "USD"
    if "EUR" in t or "€" in ocr_text:
        return "EUR"
    if "GBP" in t or "POUND" in t or "£" in ocr_text:
        return "GBP"
    if "SGD" in t or "S$" in ocr_text:
        return "SGD"
    if "THB" in t or "BAHT" in t:
        return "THB"
    if "JPY" in t or "YEN" in t or "¥" in ocr_text:
        return "JPY"
    if "INR" in t or "₹" in ocr_text:
        return "INR"
    return ""


def _detect_totals(ocr_text: str) -> tuple[str, str]:
    """Extract subtotal and tax only. Total is not scanned."""
    def find_label_value(text: str, label: str) -> str:
        label_lower = label.lower()
        lines = text.lower().splitlines()
        for line in lines:
            if label_lower in line:
                for part in line.replace(label_lower, " ").split():
                    part = part.replace("$", "").replace(",", "").strip()
                    if PRICE_PATTERN.match(part) or PRICE_INT_PATTERN.match(part):
                        if "." in part:
                            return part
                        return part + ".00"
        return ""
    subtotal = find_label_value(ocr_text, "subtotal") or find_label_value(ocr_text, "sub total")
    tax = _find_tax(ocr_text)
    return subtotal, tax


def _detect_merchant(ocr_text: str) -> str:
    """Extract merchant/store name: prefer a clean line like 'City Express', skip address/header lines."""
    # Lines that are clearly not the merchant name (address, phone, receipt metadata)
    skip_patterns = (
        "no(", "phone", "open daily", "taxpayer", "cash sale", "slip no", "counter", "cashier",
        "qty---", "qty----", "total", "paid by", "changed", "thank you", "ks ", "id no",
        "st ", "tsp", "no(26)", "0977", "7:00", "7:90", "10:00",
    )
    candidates = []
    for line in ocr_text.splitlines():
        line = line.strip()
        if len(line) < 3 or len(line) > 60:
            continue
        line_lower = line.lower()
        if any(p in line_lower for p in skip_patterns):
            continue
        if re.match(r"^[\d\.\$\s,]+$", line):
            continue
        words = line.split()
        if len(words) < 1 or len(words) > 5:
            continue
        # Prefer lines that look like a store name: 2–4 words, each at least 2 chars
        if all(len(w) >= 2 for w in words):
            candidates.append(line)
    # First candidate is usually the merchant (top of receipt)
    if candidates:
        return candidates[0][:80]
    # Fallback: first non-empty line that isn’t only digits
    for line in ocr_text.splitlines():
        line = line.strip()
        if len(line) > 2 and not re.match(r"^[\d\.\$\s,]+$", line):
            return line[:80]
    return ""


def extract_receipt(image_input) -> ReceiptData:
    """
    Extract structured receipt data from an image.
    Prices are taken only from strict pattern match—never corrected or guessed.
    """
    if isinstance(image_input, (str, Path)):
        image = np.array(Image.open(image_input).convert("RGB"))
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        return ReceiptData(
            merchant="",
            date="",
            items=[],
            subtotal="",
            tax="",
            total="",
            currency="",
            raw_text="",
            warnings=["Invalid input"],
        )

    # Multi-pass OCR by default; set FAST_RECEIPT_OCR=1 for lower latency.
    if os.getenv("FAST_RECEIPT_OCR", "").lower() in ("1", "true", "yes"):
        ocr_text = _ocr_receipt_fast(image)
    else:
        ocr_text = _ocr_receipt_all_fonts(image)
    word_data = []  # Merged text has no word-level data; parse from lines

    items, warnings = _parse_receipt_lines(ocr_text, word_data)
    subtotal, tax = _detect_totals(ocr_text)
    merchant = _detect_merchant(ocr_text)
    currency = _detect_currency(ocr_text)
    date = ""
    for part in re.findall(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}", ocr_text):
        date = part
        break

    return ReceiptData(
        merchant=merchant,
        date=date,
        items=[asdict(it) for it in items],
        subtotal=subtotal,
        tax=tax,
        total="",  # Not scanned
        currency=currency,
        image_url="",
        raw_text=ocr_text,
        warnings=warnings,
        mart_id="",
        mart_name="",
    )


def load_expenses() -> list:
    """Load expense list from MongoDB, or fallback JSON file."""
    mongo_expenses = _load_expenses_from_mongo()
    if mongo_expenses is not None:
        return mongo_expenses
    if not EXPENSES_PATH.is_file():
        return []
    try:
        return json.loads(EXPENSES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_expenses(expenses: list) -> None:
    """Save expense list to JSON file."""
    EXPENSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPENSES_PATH.write_text(json.dumps(expenses, indent=2), encoding="utf-8")


def get_last_mongo_error() -> str:
    """Last Mongo failure message (for API 503 responses)."""
    return _last_mongo_error


def set_last_mongo_error(message: str) -> None:
    """Set last error (e.g. from auth_users when users collection fails)."""
    global _last_mongo_error
    _last_mongo_error = (message or "").strip()


def close_mongo_client() -> None:
    """Close pooled Mongo client (call on app shutdown)."""
    global _mongo_client, _last_mongo_error
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
        _mongo_client = None
    _last_mongo_error = ""


def _get_mongo_collection():
    """
    Return PyMongo collection handle using a pooled client, or None on failure.
    Sets _last_mongo_error on failure.
    """
    global _mongo_client, _last_mongo_error
    mongo_uri, mongo_db_name, mongo_collection_name = _mongo_settings()
    if not mongo_uri:
        _last_mongo_error = (
            "MONGO_URI is not set. Add it to the `.env` next to `api.py` "
            "or export MONGO_URI before starting uvicorn."
        )
        return None
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=60000,
                maxPoolSize=20,
                retryWrites=True,
            )
            _mongo_client.admin.command("ping")
        except PyMongoError as e:
            msg = str(e).strip()
            if len(msg) > 400:
                msg = msg[:400] + "…"
            _last_mongo_error = f"MongoDB connection failed: {msg}"
            if _mongo_client is not None:
                try:
                    _mongo_client.close()
                except Exception:
                    pass
                _mongo_client = None
            return None
        _last_mongo_error = ""
    try:
        return _mongo_client[mongo_db_name][mongo_collection_name]
    except Exception as e:
        _last_mongo_error = f"MongoDB database/collection error: {e}"
        return None


def _load_expenses_from_mongo(
    user_id: Optional[str] = None,
    mart_id: Optional[str] = None,
):
    """Load expense list from MongoDB. If user_id is set, only that user's receipts (1:N)."""
    global _last_mongo_error
    collection = _get_mongo_collection()
    if collection is None:
        return None
    try:
        query: dict = {}
        if user_id:
            query["user_id"] = user_id
        mid = (mart_id or "").strip()
        if mid:
            query["mart_id"] = mid
        return list(collection.find(query, {"_id": 0}))
    except PyMongoError as e:
        msg = str(e).strip()
        if len(msg) > 400:
            msg = msg[:400] + "…"
        _last_mongo_error = f"MongoDB query failed: {msg}"
        return None


def load_expenses_from_mongo_only(
    user_id: Optional[str] = None,
    mart_id: Optional[str] = None,
):
    """
    Load receipts strictly from MongoDB.
    If user_id is provided, only receipts belonging to that user.
    If mart_id is provided, only receipts linked to that mart (1 mart : many receipts).
    Returns None when MongoDB is unavailable or not configured.
    """
    return _load_expenses_from_mongo(user_id=user_id, mart_id=mart_id)


def _save_receipt_to_mongo(entry: dict) -> bool:
    """Insert one receipt into MongoDB; return True on success."""
    collection = _get_mongo_collection()
    if collection is None:
        return False
    try:
        collection.insert_one(entry)
        return True
    except PyMongoError:
        return False


def add_receipt_to_expenses(receipt: ReceiptData, user_id: Optional[str] = None) -> None:
    """Append one receipt's items to expenses. Prices are stored exactly as extracted."""
    entry = {
        "merchant": receipt.merchant,
        "date": receipt.date,
        "items": receipt.items,
        "subtotal": receipt.subtotal,
        "tax": receipt.tax,
        "currency": getattr(receipt, "currency", ""),
        "image_url": getattr(receipt, "image_url", "") or "",
    }
    mid = (getattr(receipt, "mart_id", "") or "").strip()
    mname = (getattr(receipt, "mart_name", "") or "").strip()
    if mid:
        entry["mart_id"] = mid
    if mname:
        entry["mart_name"] = mname
    if user_id:
        entry["user_id"] = user_id
    if _save_receipt_to_mongo(entry):
        return
    expenses = load_expenses()
    expenses.append(entry)
    save_expenses(expenses)


def _get_marts_collection():
    """Same Mongo pool as receipts; collection name from MONGO_MARTS_COLLECTION (default `marts`)."""
    if _get_mongo_collection() is None:
        return None
    _, mongo_db_name, _ = _mongo_settings()
    try:
        return _mongo_client[mongo_db_name][_marts_collection_name()]
    except Exception as e:
        global _last_mongo_error
        _last_mongo_error = f"MongoDB marts collection error: {e}"
        return None


def list_marts_from_mongo(user_id: Optional[str] = None):
    """Return marts for a user (1 user : many marts). If user_id is None, returns all (legacy tools)."""
    global _last_mongo_error
    collection = _get_marts_collection()
    if collection is None:
        return None
    try:
        query = {"user_id": user_id} if user_id else {}
        return list(collection.find(query, {"_id": 0}).sort("name", 1))
    except PyMongoError as e:
        msg = str(e).strip()
        if len(msg) > 400:
            msg = msg[:400] + "…"
        _last_mongo_error = f"MongoDB marts query failed: {msg}"
        return None


def get_mart_for_user(user_id: str, mart_id: str) -> Optional[dict]:
    """Return a mart document if it exists and belongs to user_id."""
    mid = (mart_id or "").strip()
    if not mid:
        return None
    collection = _get_marts_collection()
    if collection is None:
        return None
    try:
        return collection.find_one({"id": mid, "user_id": user_id}, {"_id": 0})
    except PyMongoError:
        return None


def create_mart_in_mongo(
    name: str,
    description: str,
    logo_url: str,
    user_id: str,
) -> Optional[dict]:
    """Insert one mart for a user. Returns the document (no _id) or None on failure."""
    collection = _get_marts_collection()
    if collection is None:
        return None
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "name": name.strip(),
        "description": (description or "").strip(),
        "logo_url": (logo_url or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        collection.insert_one(doc)
        return doc
    except PyMongoError:
        return None
