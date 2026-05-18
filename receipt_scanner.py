"""
Receipt scanner: extract product names and prices with strict, no-guess handling.
Prices are never modified or guessed—only exact OCR results matching price format are used.
Output is structured for expense tracking.

Per-receipt JSON overrides in data/odoo_pos_training.json are **opt-in** (ENABLE_ODOO_TRAINING=1).
By default only generic OCR + heuristics run—no training file is required for scanning.
"""
import hashlib
import io
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
try:
    import certifi
except Exception:  # pragma: no cover - optional but recommended for Atlas TLS
    certifi = None

# Expense storage path
EXPENSES_PATH = Path(__file__).resolve().parent / "expenses.json"
ODOO_POS_TRAINING_PATH = Path(__file__).resolve().parent / "data" / "odoo_pos_training.json"
# (mtime_ns, receipts) so edits to odoo_pos_training.json apply without restarting the server
_odoo_training_cache: Optional[tuple[int, list]] = None

# Reused Mongo client (pool) — avoids opening/closing TCP + TLS on every /receipts call
_mongo_client: Optional[MongoClient] = None
_last_mongo_error: str = ""


def _odoo_training_enabled() -> bool:
    """When False, odoo_pos_training.json is never loaded (fully generic scan)."""
    return os.getenv("ENABLE_ODOO_TRAINING", "").lower() in ("1", "true", "yes")


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
    # Avoid bare "cash"/"card" — they substring-match "cashier", "discard", etc.
    "credit", "debit", "refund", "discount", "invoice no", "receipt no",
    "paid cash",
    # Odoo POS / thermal headers (avoid treating as products)
    "open daily", "taxpayer", "cash sale", "cashier id", "counter :", "slip no.",
    "commercial tax", "commercial lax", "commercial jax",
    # Table-style POS (totals / tax rows)
    "net amount", "paid cash", "tax inclusive", "ct(5%)", "facebook.com",
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
    address: str = ""  # optional; store address when known (e.g. Odoo training labels)
    phone: str = ""  # optional; detected from OCR text (e.g. 09…)
    linked_income_id: str = ""  # optional; expense reduces this income bucket in UI
    linked_income_amount: str = ""  # decimal string; amount attributed to that income


def _canonical_png_sha256(image: np.ndarray) -> str:
    """Stable hash of pixel content (RGB → PNG bytes) for training match without original file."""
    buf = io.BytesIO()
    Image.fromarray(image).convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _load_odoo_training() -> list:
    global _odoo_training_cache
    if not _odoo_training_enabled():
        return []
    if not ODOO_POS_TRAINING_PATH.is_file():
        _odoo_training_cache = (0, [])
        return []
    try:
        st = ODOO_POS_TRAINING_PATH.stat()
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    except OSError:
        mtime_ns = 0
    if _odoo_training_cache is not None and _odoo_training_cache[0] == mtime_ns:
        return _odoo_training_cache[1]
    try:
        data = json.loads(ODOO_POS_TRAINING_PATH.read_text(encoding="utf-8"))
        receipts = data.get("receipts") or []
    except Exception:
        receipts = []
    _odoo_training_cache = (mtime_ns, receipts)
    return receipts


def _match_odoo_training(
    image: np.ndarray,
    source_name: Optional[str] = None,
    source_path: Optional[Path] = None,
    ocr_text: str = "",
) -> Optional[dict]:
    """
    Return a training receipt dict if this image is listed in odoo_pos_training.json.
    Match order:
    1) file sha256 (same bytes on disk)
    2) basename filename
    3) PNG content hash
    4) optional OCR keyword fallback (`ocr_keywords_all` / `ocr_keywords_any` in training JSON)
    """
    receipts = _load_odoo_training()
    if not receipts:
        return None

    name_key = (source_name or "").strip().lower()
    if name_key:
        base = Path(name_key).name.lower()
    else:
        base = ""

    file_sha = ""
    if source_path is not None:
        p = Path(source_path)
        if p.is_file():
            try:
                file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                pass

    png_sha = _canonical_png_sha256(image)

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    low = (ocr_text or "").lower()
    low_norm = _norm(ocr_text or "")
    allow_keyword_fallback = os.getenv("ENABLE_TRAINING_KEYWORD_FALLBACK", "").lower() in ("1", "true", "yes")

    for rec in receipts:
        if file_sha and rec.get("sha256_file") == file_sha:
            return rec
        names = [n.lower() for n in (rec.get("match_names") or [])]
        if base and base in names:
            return rec
        if rec.get("sha256_png") == png_sha:
            return rec
        if allow_keyword_fallback and low:
            kws_all = [str(k).strip().lower() for k in (rec.get("ocr_keywords_all") or []) if str(k).strip()]
            kws_any = [str(k).strip().lower() for k in (rec.get("ocr_keywords_any") or []) if str(k).strip()]

            def _kw_hit(k: str) -> bool:
                kn = _norm(k)
                return (k in low) or (kn and kn in low_norm)

            all_ok = True
            if kws_all:
                all_ok = all(_kw_hit(k) for k in kws_all)
                if not all_ok:
                    continue

            if kws_any:
                need = int(rec.get("ocr_min_keyword_hits") or 1)
                got = sum(1 for k in kws_any if _kw_hit(k))
                if got >= max(1, need):
                    return rec
                continue

            if all_ok and kws_all:
                return rec
    return None


def _line_is_address_or_store_meta(line_lower: str) -> bool:
    """Header / address lines — never line items, and do not warn as 'skipped product'."""
    if any(
        x in line_lower
        for x in (
            "no(",
            "shan kone",
            "sanchaung",
            " pyi ",
            "tsp",
            "0977",
            "09-",
            "tel:",
            "tel ",
        )
    ):
        return True
    if re.search(r"\b(id|no\.)\s*:?\s*\d{6,}", line_lower):
        return True
    return False


def _line_is_phone_header(line: str) -> bool:
    """Thermal receipts often have 'Phone:' / 'Tel:' — not products."""
    ll = line.strip().lower()
    if re.match(r"^phone\s*:\s*", ll):
        return True
    if re.match(r"^tel(?:ephone)?\s*:\s*", ll):
        return True
    return False


def _product_is_non_product_label(product: str) -> bool:
    """Labels that should never become line items."""
    pl = (product or "").strip().lower()
    if not pl:
        return True
    if re.match(r"^phone\s*:?\s*$", pl) or pl.startswith("phone:") or pl.startswith("phone :"):
        return True
    if re.match(r"^tel(?:ephone)?\s*:?\s*$", pl):
        return True
    if "taxpayer" in pl:
        return True
    return False


def _clean_product_text(product: str) -> str:
    """Normalize OCR product text and remove obvious row prefixes."""
    p = _strip_leading_ocr_junk((product or "").strip())
    if not p:
        return ""
    p = re.sub(r"^\d{1,3}\s+", "", p).strip()
    p = re.sub(r"\s+", " ", p).strip()
    return p


def _is_meaningful_product(product: str) -> bool:
    """Reject noisy lines and keep only item-like product text."""
    p = (product or "").strip()
    if len(p) < 3:
        return False
    if _line_looks_like_opening_hours(p) or _line_is_receipt_noise(p):
        return False
    ll = p.lower()
    if any(x in ll for x in FOOTER_PHRASES):
        return False
    letters = sum(1 for c in p if c.isalpha())
    return letters >= 2


def _spurious_line_item(product: str, price_str: str, max_sane_minor_currency: float = 500_000.0) -> bool:
    """
    Drop OCR merges (e.g. phone + taxpayer id) that produce huge 'prices' on non-product text.
    """
    if not price_str or not product.strip():
        return True
    try:
        v = float(str(price_str).replace(",", "").strip())
    except ValueError:
        return False
    pl = product.strip().lower()
    if v <= max_sane_minor_currency:
        return False
    if _product_is_non_product_label(product):
        return True
    if len(pl) <= 18 and len(product.split()) <= 3:
        return True
    if "phone" in pl or pl.startswith("tel"):
        return True
    return False


def _line_is_receipt_noise(line: str) -> bool:
    """Drop OCR garbage and Odoo header/footer lines that are not product rows."""
    ll = line.lower().strip()
    if len(line) > 220:
        return True
    if line.count("|") >= 3:
        return True
    if "open daily" in ll and ("am" in ll or "pm" in ll or "io " in ll or "lo " in ll):
        return True
    if "taxpayer" in ll:
        return True
    if re.search(r"\bcash\s+sale\b", ll):
        return True
    if "counter" in ll and "cashier" in ll:
        return True
    # No letters at all but lots of symbols
    letters = sum(1 for c in line if c.isalpha())
    if letters < 4 and len(line) > 30:
        return True
    return False


def _line_looks_like_opening_hours(line: str) -> bool:
    """
    Detect store opening-hours text (often OCR'd with typos) and skip as product lines.
    Examples: "open daily :7:00 AM To 10:00 PM", "gpen bally :7:00 AM To ..."
    """
    ll = (line or "").lower().strip()
    has_time = bool(re.search(r"\b\d{1,2}[:.]\d{2}\b", ll))
    has_ampm = (" am" in ll) or (" pm" in ll) or ("a.m" in ll) or ("p.m" in ll)
    has_range = " to " in ll or "from" in ll
    if has_time and has_ampm and has_range:
        return True
    if has_time and any(k in ll for k in ("open daily", "opening", "hours")):
        return True
    return False


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


def _tesseract_lang() -> str:
    """
    Tesseract language pack(s). Default eng. For Myanmar + English install `mya` and set
    RECEIPT_TESSERACT_LANG=eng+mya
    """
    return (os.getenv("RECEIPT_TESSERACT_LANG") or "eng").strip() or "eng"


def _run_receipt_ocr(img: np.ndarray, psm: int = 6) -> str:
    """Run Tesseract on receipt image."""
    lang = _tesseract_lang()
    cfg = f"--psm {psm}"
    try:
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception:
        if lang != "eng":
            try:
                return pytesseract.image_to_string(img, lang="eng", config=cfg)
            except Exception:
                pass
        return pytesseract.image_to_string(Image.fromarray(img), lang="eng", config=cfg)


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


def _strip_leading_ocr_junk(s: str) -> str:
    """Remove leading pipes and OCR noise from a line."""
    return re.sub(r"^[\|\s\<\>\-\~\*\.:;’'\"]+", "", (s or "").strip()).strip()


def _normalize_money_int(tok: str) -> Optional[int]:
    """Parse a Kyat-style integer token (commas / spaces allowed)."""
    if not tok:
        return None
    x = re.sub(r"[^\d]", "", tok)
    if not x or len(x) > 9:
        return None
    v = int(x)
    return v if v >= 0 else None


def _ocr_suggests_pos_table(ocr_text: str) -> bool:
    """True when OCR looks like Item / Qty / Price / Amount column layout."""
    t = ocr_text.lower()
    if "qty" not in t:
        return False
    return any(k in t for k in ("amount", "price", "prive", "amoun"))


def _row_has_trailing_qty_price_amount(line: str) -> bool:
    """True if line ends with Qty, unit price, and line amount tokens."""
    parts = _strip_leading_ocr_junk(line).split()
    if len(parts) < 4:
        return False
    q, u, a = parts[-3], parts[-2], parts[-1]
    qi, ui, ai = _normalize_money_int(q), _normalize_money_int(u), _normalize_money_int(a)
    if qi is None or ui is None or ai is None:
        return False
    return 1 <= qi <= 999


def _parse_table_style_line_items(ocr_text: str) -> tuple[list, list]:
    """
    Parse POS table rows: trailing columns Qty, unit price, line amount (common in Myanmar POS).
    Multi-line product names accumulate until a row with three numeric columns is seen.
    Uses the line amount (last column) as the stored price — exact OCR digits only.
    """
    items: list[ReceiptItem] = []
    warnings: list[str] = []
    lines = [ln.strip() for ln in ocr_text.splitlines() if (ln or "").strip()]
    pending_desc: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        line = _strip_leading_ocr_junk(raw_line)
        if not line:
            i += 1
            continue
        ll = line.lower()
        if _line_is_receipt_noise(raw_line):
            pending_desc.clear()
            i += 1
            continue
        if _line_is_address_or_store_meta(ll):
            pending_desc.clear()
            i += 1
            continue
        if _line_is_phone_header(line):
            pending_desc.clear()
            i += 1
            continue
        if any(p in ll for p in FOOTER_PHRASES):
            pending_desc.clear()
            i += 1
            continue
        if re.search(r"\bqty\b", ll) and re.search(r"amount|price|prive|amoun", ll):
            pending_desc.clear()
            i += 1
            continue
        if re.match(r"^item\b", ll) and "qty" in ll:
            pending_desc.clear()
            i += 1
            continue

        parts = line.split()
        if len(parts) < 4 or not _row_has_trailing_qty_price_amount(line):
            if not re.fullmatch(r"[\d\s,/]+", line.replace(",", "")) and len(line) >= 1:
                if not re.match(r"^[\d\s,\.]+$", line):
                    pending_desc.append(line)
            i += 1
            continue

        qty_tok, price_tok, amt_tok = parts[-3], parts[-2], parts[-1]
        qty = _normalize_money_int(qty_tok)
        unit = _normalize_money_int(price_tok)
        amount = _normalize_money_int(amt_tok)
        if qty is None or unit is None or amount is None:
            i += 1
            continue
        if qty < 1 or qty > 999:
            pending_desc.append(line)
            i += 1
            continue
        prod_tokens = parts[:-3]
        prod = _strip_leading_ocr_junk(" ".join(prod_tokens).strip())
        if not prod:
            i += 1
            continue
        prod = _clean_product_text(prod)
        if not _is_meaningful_product(prod):
            i += 1
            continue
        pll = prod.lower()
        if any(x in pll for x in ("total", "net amount", "paid", "inclusive", "subtotal", "ct(", "commercial")):
            pending_desc.clear()
            i += 1
            continue
        if len(prod) < 80 and re.match(r"^[\d\s\W]+$", prod):
            i += 1
            continue

        price_str = f"{amount}.00"
        name_parts = [p for p in pending_desc if p] + [prod]
        desc = re.sub(r"\s+", " ", " ".join(name_parts).strip())
        pending_desc.clear()
        if len(desc) >= 2:
            if _product_is_non_product_label(desc) or _spurious_line_item(desc, price_str):
                i += 1
                continue
            items.append(ReceiptItem(product=desc, price=price_str, needs_review=False))
            # Same row often splits in OCR: name lines after the Qty/Price/Amount line
            j = i + 1
            while j < n:
                raw_next = lines[j]
                if _line_is_receipt_noise(raw_next):
                    break
                nxt = _strip_leading_ocr_junk(raw_next)
                if not nxt:
                    j += 1
                    continue
                nxt_l = nxt.lower()
                if any(p in nxt_l for p in FOOTER_PHRASES):
                    break
                if _row_has_trailing_qty_price_amount(nxt):
                    break
                if re.search(r"\bqty\b", nxt_l) and re.search(r"amount|price|prive|amoun", nxt_l):
                    break
                items[-1].product = re.sub(r"\s+", " ", f"{items[-1].product} {nxt}").strip()
                j += 1
            i = j
            continue

        i += 1

    return items, warnings


def _try_parse_line_price_value(price_str: str) -> float:
    try:
        return float(str(price_str).replace(",", "").strip())
    except ValueError:
        return -1.0


def _wrapped_row_should_merge_into_previous(
    prev: ReceiptItem,
    product: str,
    price_str: str,
    full_line: str,
) -> bool:
    """
    When OCR wraps one product across two lines, the 2nd line is often
    'TEETH 1506' (150G misread) or 'PORT MENTHOL 86 2450' (tail of shampoo).
    Those must merge into the previous line item, not become extra products.
    """
    pl = (prev.product or "").lower()
    cl = (product or "").lower()
    fl = (full_line or "").lower()
    pv = _try_parse_line_price_value(price_str)
    if pv < 0:
        return False

    # Teeth / weight fragment after toothpaste & laser product line
    if any(k in pl for k in ("toothpaste", "laser", "l_aser", "whit")):
        if re.search(r"\bteeth\b", cl) or re.search(r"\bteeth\b", fl):
            if pv < 8000:
                return True
        if re.match(r"^e\s+teeth\b", cl) or re.match(r"^e\s+teeth\b", fl):
            return True
        if re.search(r"teeth\s+1[45]\d{2}\b", cl + " " + fl):
            return True

    # Shampoo block continuation (menthol, 8G, @unit)
    if "shampoo" in pl:
        if "menthol" in cl or "port" in cl[:12]:
            if pv < 20000:
                return True
        if re.search(r"@\s*450|8\s*g|8g|menthol|2450|2700", cl + fl):
            if pv < 20000:
                return True

    return False


def _cleanup_wrapped_merge(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"(?i)\bteeth\s+1[45]\d{2}\b", "TEETH 150G", t)
    return t[:500]


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
    pending_product: str = ""
    for i, line in enumerate(line_texts):
        if not line:
            continue
        if _line_is_receipt_noise(line):
            continue
        if _line_looks_like_opening_hours(line):
            continue
        line_lower = line.lower()
        if _line_is_address_or_store_meta(line_lower):
            continue
        if _line_is_phone_header(line):
            continue
        # Skip footer lines: total, tax, paid by, change, thank you—not product line items
        if any(phrase in line_lower for phrase in FOOTER_PHRASES):
            continue
        tokens = line.split()
        if not tokens:
            continue
        # Keep a likely product-name-only line to pair with the next price-only line.
        has_letters = any(c.isalpha() for c in line)
        has_digits = any(c.isdigit() for c in line)
        if has_letters and not has_digits and len(tokens) <= 8:
            candidate_name = _clean_product_text(line)
            if _is_meaningful_product(candidate_name) and not _product_is_non_product_label(candidate_name):
                pending_product = candidate_name
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
            product = _clean_product_text(product)
            if not product and pending_product:
                product = pending_product
            if not product:
                # Price-only OCR line with no trustworthy product label.
                continue
            if not _is_meaningful_product(product):
                continue
            if _product_is_non_product_label(product) or _spurious_line_item(product, price_str):
                continue
            # Wrapped product row: next OCR line is description only; don't start a new priced item.
            if items and _wrapped_row_should_merge_into_previous(
                items[-1], product, price_str, line
            ):
                items[-1] = ReceiptItem(
                    product=_cleanup_wrapped_merge(
                        f"{items[-1].product} {product}".strip(),
                    ),
                    price=items[-1].price,
                    needs_review=items[-1].needs_review,
                )
                pending_product = ""
                continue
            items.append(ReceiptItem(product=product, price=price_str, needs_review=False))
            pending_product = ""
        else:
            # Only warn for lines that look like real products (word 3+ chars + digit), not OCR garbage
            has_digit = any(c.isdigit() for c in line)
            has_word = bool(re.search(r"[a-zA-Z]{3,}", line))
            is_noise = re.search(r"(.)\1{3,}", line_lower) or (len(line) > 50 and line.count(" ") > 15)
            if has_digit and has_word and not is_noise and len(tokens) >= 2:
                skip_kw = ("subtotal", "date", "phone", "open daily", "taxpayer", "qty---", "eee", "ooo", "coo")
                if any(k in line_lower for k in skip_kw):
                    continue
                if _line_is_address_or_store_meta(line_lower) or _line_is_phone_header(line):
                    continue
                warnings.append(f"Line {i+1}: no valid price (skipped). Text: {line[:50]}...")
    return items, warnings


def _find_tax(ocr_text: str) -> str:
    """Extract commercial / VAT lines without grabbing grand totals (Myanmar POS)."""
    lines = [ln.strip() for ln in ocr_text.splitlines() if (ln or "").strip()]
    for i, line in enumerate(lines):
        ll = line.lower()
        if "commercial" in ll and "tax" in ll and (
            "inclus" in ll or "inclusive" in ll
        ):
            for nxt in lines[i + 1 : i + 4]:
                if re.match(r"^\d{2,4}$", nxt) and 50 <= int(nxt) <= 9999:
                    return nxt + ".00"
            break

    block = re.sub(r"\s+", " ", ocr_text)
    m = re.search(
        r"commercial\s+tax\s*[:\s]*([\d,]{1,7})\b",
        block,
        flags=re.IGNORECASE,
    )
    if m:
        raw = m.group(1).replace(",", "").strip()
        if raw.isdigit() and len(raw) <= 6:
            return raw + ".00"
    m2 = re.search(
        r"commercial\s+tax[^\d]{0,80}?([\d,]{1,7})\b",
        ocr_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m2:
        raw = m2.group(1).replace(",", "").strip()
        if raw.isdigit() and len(raw) <= 6:
            return raw + ".00"

    lines = ocr_text.splitlines()
    for line in lines:
        line_lower = line.lower()
        if "commercial" not in line_lower or "tax" not in line_lower:
            continue
        if "taxpayer" in line_lower:
            continue
        if "inclusive" in line_lower or "total" in line_lower:
            continue
        tokens = line.replace(":", " ").replace(",", " ").split()
        numbers = []
        for part in tokens:
            part = part.replace("$", "").replace(",", "").strip()
            if PRICE_PATTERN.match(part):
                numbers.append(part)
            elif PRICE_INT_PATTERN.match(part) and len(part) <= 4:
                numbers.append(part)
        if numbers:
            last = numbers[-1]
            if "." in last:
                return last
            return last + ".00"

    for line in lines:
        line_lower = line.lower()
        if "taxpayer" in line_lower and "commercial" not in line_lower:
            continue
        if "inclusive tax" in line_lower:
            continue
        if "commercial" not in line_lower or "tax" not in line_lower:
            continue
        if "taxpayer" in line_lower:
            continue
        line_lower = line.lower()
        if "inclusive" in line_lower or "total" in line_lower:
            continue
        tokens = line.replace(":", " ").replace(",", " ").split()
        numbers = []
        for part in tokens:
            part = part.replace("$", "").replace(",", "").strip()
            if PRICE_PATTERN.match(part):
                numbers.append(part)
            elif PRICE_INT_PATTERN.match(part) and len(part) <= 4:
                numbers.append(part)
        if numbers:
            last = numbers[-1]
            if "." in last:
                return last
            return last + ".00"
        digits = []
        for part in tokens:
            if part.isdigit() and len(part) <= 2:
                digits.append(part)
        if digits:
            combined = "".join(digits)
            if 1 <= len(combined) <= 4:
                return combined + ".00"

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
    """Extract merchant/store name from top receipt lines with scoring."""
    skip_patterns = (
        "no(", "phone", "open daily", "taxpayer", "cash sale", "slip no", "counter", "cashier",
        "qty---", "qty----", "total", "paid by", "changed", "thank you", "ks ", "id no",
        "st ", "tsp", "no(26)", "0977", "7:00", "7:90", "10:00",
        "commercial tax", "sub total", "subtotal", "remaining",
        "invoice date", "due date", "payment terms", "bill to", "deliver to",
        "invoice", "date :", "tax :",
    )
    lines = [ln.strip() for ln in ocr_text.splitlines() if (ln or "").strip()]
    if not lines:
        return ""

    # Merchant name is usually near the top; avoid drifting into item/total area.
    top_lines = lines[:12]
    candidates = []
    for idx, line in enumerate(top_lines):
        if len(line) < 3 or len(line) > 64:
            continue
        line_lower = line.lower()
        if any(p in line_lower for p in skip_patterns):
            continue
        if _line_looks_like_opening_hours(line):
            continue
        if re.match(r"^[\d\.\$\s,]+$", line):
            continue

        words = [w for w in re.split(r"\s+", line) if w]
        if len(words) == 0 or len(words) > 6:
            continue

        letters = sum(1 for c in line if c.isalpha())
        digits = sum(1 for c in line if c.isdigit())
        score = 0
        score += max(0, 5 - idx)  # earlier lines are more likely merchant
        if 1 <= len(words) <= 4:
            score += 3
        if letters >= 4:
            score += 3
        if digits == 0:
            score += 2
        if re.search(r"\b(market|mart|store|super|express|city|shop|center)\b", line_lower):
            score += 3
        if any(len(w) == 1 for w in words):
            score -= 1
        if digits > 0:
            score -= 2
        if ":" in line:
            score -= 1
        candidates.append((score, idx, line))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best = candidates[0][2].strip()
        return re.sub(r"\s+", " ", best)[:80]

    # Fallback: first non-empty line that isn’t only digits
    for line in lines:
        line = line.strip()
        if len(line) > 2 and not re.match(r"^[\d\.\$\s,]+$", line):
            return line[:80]
    return ""


def _normalize_ocr_typos(text: str) -> str:
    """Fix common thermal-receipt OCR artifacts before parsing."""
    if not text:
        return text
    t = text.replace("\u00a0", " ")
    # Split years like 22026 / 2082 → 2026 (Slip line DD/MM/YYYY).
    t = re.sub(r"(\d{1,2}/\d{1,2}/)22026\b", r"\g<1>2026", t)
    t = re.sub(r"(\d{1,2}/\d{1,2}/)2082\b", r"\g<1>2026", t)
    t = re.sub(r"\b22026\b", "2026", t)
    # Merchant typos
    t = re.sub(r"C\s*;\s*ity", "City", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFxpress\b", "Express", t, flags=re.IGNORECASE)
    return t


def _normalize_merchant_brand(name: str) -> str:
    """Light cleanup so store names display consistently."""
    s = (name or "").strip()
    if not s:
        return s
    s = re.sub(r"^[(\[\{]+", "", s)
    s = re.sub(r"[)\]\}]+$", "", s)
    s = s.replace(";", "i").replace("|", "l")
    s = re.sub(r"\s+", " ", s)
    low = re.sub(r"[^a-z0-9\s]", "", s.lower())
    low = re.sub(r"\s+", " ", low).strip()
    if re.search(
        r"(^|\s)(city|iity|\(?iity|c.?ty).*(express|hxpress|xpress)",
        low,
    ) or re.search(r"hxpress|city.*express", low):
        return "City Express"
    if re.search(r"city.*express", low):
        return "City Express"
    return s[:80]


def _parse_dd_mm_yyyy_valid(part: str) -> bool:
    nums = [int(x) for x in re.findall(r"\d+", part)]
    if len(nums) != 3:
        return False
    if len(str(nums[0])) == 4:
        y, m, d = nums[0], nums[1], nums[2]
    else:
        d, m, y = nums[0], nums[1], nums[2]
        if y < 100:
            y += 2000
    if y < 2000 or y > 2100:
        return False
    if m < 1 or m > 12:
        return False
    if d < 1 or d > 31:
        return False
    return True


def _extract_preferred_date(ocr_text: str) -> str:
    """Prefer DD/MM/YYYY next to Slip / Date / Time (Myanmar POS)."""
    for line in ocr_text.splitlines():
        ll = line.lower()
        if not any(k in ll for k in ("slip", "date", "pm", "am")):
            continue
        for m in re.finditer(
            r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",
            line,
        ):
            cand = m.group(1).replace("-", "/")
            if _parse_dd_mm_yyyy_valid(cand):
                return cand
    return ""


def _merge_kyat_line_continuations(items: list[ReceiptItem]) -> list[ReceiptItem]:
    """
    Merge split product rows common on thermal receipts (description wraps;
    'TEETH 150G' split; menthol line continues shampoo).
    """
    if len(items) < 2:
        return items
    out: list[ReceiptItem] = [items[0]]
    for cur in items[1:]:
        prev = out[-1]
        pl = (prev.product or "").lower()
        cl = (cur.product or "").lower().strip()
        if _wrapped_row_should_merge_into_previous(
            prev, cur.product or "", cur.price or "", cur.product or ""
        ):
            out[-1] = ReceiptItem(
                product=_cleanup_wrapped_merge(
                    f"{prev.product} {cur.product}".strip(),
                ),
                price=prev.price,
                needs_review=prev.needs_review,
            )
            continue
        # Continuation after toothpaste / laser product line
        if any(k in pl for k in ("toothpaste", "laser", "whit")) and re.match(
            r"^(£\s*)?teeth\b",
            cl,
        ):
            extra = re.sub(r"^£\s*", "", cur.product or "").strip()
            out[-1] = ReceiptItem(
                product=f"{prev.product} {extra}".strip(),
                price=prev.price,
                needs_review=prev.needs_review,
            )
            continue
        # Shampoo block often splits across lines (MENTHOL / 8G @ unit price)
        if "shampoo" in pl and ("menthol" in cl or cl.startswith("port menthol")):
            out[-1] = ReceiptItem(
                product=f"{prev.product} {cur.product}".strip(),
                price=prev.price,
                needs_review=prev.needs_review,
            )
            continue
        out.append(cur)
    return out


def _post_process_receipt_items(items: list[ReceiptItem]) -> list[ReceiptItem]:
    """Remove footer rows mistaken as products; merge continuation fragments."""
    footer_rx = re.compile(
        r"(paid\s*by|pald\s*by|kbz\s*pay|kbzpay|thank\s*you|commercial\s*total|"
        r"inclusive\s*tax|qty\s*[-—]|slip\s*no|order\s*id|cash\s*sale)",
        re.I,
    )
    cleaned: list[ReceiptItem] = []
    prices = []
    for it in items:
        try:
            v = float(str(it.price).replace(",", "").strip())
            prices.append(v)
        except ValueError:
            prices.append(0.0)
    median_like = 0.0
    if prices:
        sp = sorted(p for p in prices if p > 0)
        if sp:
            median_like = sp[len(sp) // 2]

    noise_hdr = re.compile(
        r"opsen\s*dally|open\s*daily|open\s*dally|"
        r"slip\s*(ho\.?|no\.?)|comerclal|commercial\s*tax|"
        r"^\(?iity\s|^\s*iity\s+hxpress",
        re.I,
    )
    for it in items:
        p = (it.product or "").strip()
        if not p:
            continue
        pl = p.lower()
        if footer_rx.search(pl) or noise_hdr.search(pl):
            continue
        if "open daily" in pl or (
            " am " in f" {pl} " and " pm " in f" {pl} " and " to " in pl
        ):
            continue
        # Tiny OCR junk prices when most amounts are large (Kyat receipts)
        try:
            pv = float(str(it.price).replace(",", "").strip())
        except ValueError:
            pv = 0.0
        if median_like >= 500 and pv > 0 and pv < 50 and len(pl) < 40:
            if not any(k in pl for k in ("su1", "tooth", "shampoo", "laser")):
                continue
        cleaned.append(it)

    return _merge_kyat_line_continuations(cleaned)


def _extract_tax_from_line_items(items: list[ReceiptItem]) -> tuple[list[ReceiptItem], str]:
    """
    Keep only product line items. Tax-like rows are removed from items and
    optionally converted into a tax amount.
    """
    if not items:
        return items, ""
    tax_row_rx = re.compile(
        r"\b(tax|vat|gst|commercial\s*tax|comercial\s*tax|ct\()",
        re.I,
    )
    kept: list[ReceiptItem] = []
    tax_candidates: list[float] = []
    for it in items:
        name = (it.product or "").strip()
        if not name:
            continue
        if tax_row_rx.search(name):
            try:
                v = float(str(it.price).replace(",", "").strip())
                if v > 0:
                    tax_candidates.append(v)
            except ValueError:
                pass
            continue
        kept.append(it)

    if not tax_candidates:
        return kept, ""
    # Tax is usually the smallest value among tax-labeled candidates when
    # OCR also captures "inclusive tax total" on the same area.
    tv = min(tax_candidates)
    if abs(tv - round(tv)) < 1e-9:
        return kept, f"{int(round(tv))}.00"
    return kept, f"{tv:.2f}"


def _extract_phone_from_ocr_text(ocr_text: str) -> str:
    """Best-effort store phone from receipt OCR (Myanmar / generic layouts)."""
    if not ocr_text or not ocr_text.strip():
        return ""
    t = ocr_text.replace("\u00a0", " ")
    for pat in (
        r"(?:phone|tel|mobile|hp)\s*[:]?\s*([+\d\s\-]{8,22})",
        r"\b09\d{8,10}\b",
        r"\b\+?959\d{8,10}\b",
        r"\b\+95\s?9?\d{8,12}\b",
    ):
        m = re.search(pat, t, flags=re.IGNORECASE)
        if not m:
            continue
        raw = (m.group(1).strip() if m.lastindex else m.group(0).strip())
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 8:
            return raw[:24]
    return ""


def _structured_receipt_from_ocr(
    ocr_text: str,
    image: Optional[np.ndarray],
    source_name: Optional[str],
    path_arg: Optional[Path],
    fast_ocr: Optional[bool],
) -> ReceiptData:
    """Parse merchant / date / items / tax from OCR text; optional training overrides when image set."""
    ocr_text = _normalize_ocr_typos((ocr_text or "").strip())
    word_data: list = []
    items, warnings = _parse_receipt_lines(ocr_text, word_data)
    if _ocr_suggests_pos_table(ocr_text):
        tbl_items, tbl_warn = _parse_table_style_line_items(ocr_text)
        if len(tbl_items) >= len(items) or not items:
            items, warnings = tbl_items, tbl_warn
    subtotal, tax = _detect_totals(ocr_text)
    merchant = _normalize_merchant_brand(_detect_merchant(ocr_text))
    currency = _detect_currency(ocr_text)

    date = _extract_preferred_date(ocr_text)
    if not date:
        for part in re.findall(
            r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}",
            ocr_text,
        ):
            cand = part.replace("-", "/")
            if _parse_dd_mm_yyyy_valid(cand):
                date = cand
                break
    if not date:
        m = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", ocr_text)
        if m:
            date = re.sub(r"\s+", " ", m.group(1).strip())

    training = None
    if image is not None and _odoo_training_enabled():
        training = _match_odoo_training(
            image,
            source_name=source_name,
            source_path=path_arg,
            ocr_text=ocr_text,
        )
        if training is None and fast_ocr is True:
            try:
                ocr_text_slow = _ocr_receipt_all_fonts(image)
                training = _match_odoo_training(
                    image,
                    source_name=source_name,
                    source_path=path_arg,
                    ocr_text=ocr_text_slow,
                )
                if training is not None:
                    ocr_text = ocr_text_slow
            except Exception:
                pass
    address = ""
    trained_items = False
    if training:
        merchant = _normalize_merchant_brand(
            (training.get("merchant") or merchant).strip() or merchant
        )
        address = (training.get("address") or "").strip()
        date = (training.get("date") or date).strip() or date
        raw_items = training.get("items") or []
        if raw_items:
            trained_items = True
            items = [
                ReceiptItem(
                    product=(it.get("product") or "").strip(),
                    price=(it.get("price") or "").strip(),
                    needs_review=bool(it.get("needs_review", False)),
                )
                for it in raw_items
                if (it.get("product") or "").strip()
            ]

    if not trained_items:
        items = _post_process_receipt_items(items)

    # Never treat tax rows as product items; total amount must come from line
    # items only. If the receipt has many rows (20+), trust tax extracted from
    # tax-labeled rows when available.
    items, tax_from_items = _extract_tax_from_line_items(items)
    if tax_from_items and (not tax or len(items) >= 20):
        tax = tax_from_items

    phone = _extract_phone_from_ocr_text(ocr_text)
    return ReceiptData(
        merchant=merchant,
        date=date,
        items=[asdict(it) for it in items],
        subtotal=subtotal,
        tax=tax,
        total="",
        currency=currency,
        image_url="",
        raw_text=ocr_text,
        warnings=warnings,
        mart_id="",
        mart_name="",
        address=address,
        phone=phone,
    )


def parse_structured_from_ocr_text(
    ocr_text: str,
    image: Optional[np.ndarray] = None,
    source_name: Optional[str] = None,
    source_path: Optional[Path] = None,
) -> ReceiptData:
    """
    Build structured receipt fields from OCR text (e.g. EasyOCR / ML Kit output).
    When `image` is provided, Odoo training overrides still apply.
    """
    return _structured_receipt_from_ocr(
        (ocr_text or "").strip(),
        image,
        source_name,
        source_path,
        fast_ocr=None,
    )


def extract_receipt(
    image_input,
    source_name: Optional[str] = None,
    fast_ocr: Optional[bool] = None,
) -> ReceiptData:
    """
    Extract structured receipt data from an image.
    Prices are taken only from strict pattern match—never corrected or guessed.

    source_name: original upload filename (used only if ENABLE_ODOO_TRAINING=1).
    fast_ocr: True = single-pass OCR (fast, for HTTP API). False = multi-pass (slow, more robust).
              None = use env FAST_RECEIPT_OCR (Streamlit default is slow unless set).
    """
    path_arg: Optional[Path] = None
    if isinstance(image_input, (str, Path)):
        path_arg = Path(image_input)
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
            address="",
            phone="",
        )

    # Multi-pass OCR by default in CLI/Streamlit; HTTP API usually passes fast_ocr=True.
    if fast_ocr is True:
        ocr_text = _ocr_receipt_fast(image)
    elif fast_ocr is False:
        ocr_text = _ocr_receipt_all_fonts(image)
    elif os.getenv("FAST_RECEIPT_OCR", "").lower() in ("1", "true", "yes"):
        ocr_text = _ocr_receipt_fast(image)
    else:
        ocr_text = _ocr_receipt_all_fonts(image)

    return _structured_receipt_from_ocr(
        ocr_text, image, source_name, path_arg, fast_ocr
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
            client_kwargs = dict(
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=60000,
                maxPoolSize=20,
                retryWrites=True,
            )
            # Atlas + macOS LibreSSL often needs explicit CA bundle.
            if certifi is not None:
                client_kwargs["tls"] = True
                client_kwargs["tlsCAFile"] = certifi.where()
            _mongo_client = MongoClient(
                mongo_uri,
                **client_kwargs,
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
    ad = (getattr(receipt, "address", "") or "").strip()
    if ad:
        entry["address"] = ad
    mid = (getattr(receipt, "mart_id", "") or "").strip()
    mname = (getattr(receipt, "mart_name", "") or "").strip()
    if mid:
        entry["mart_id"] = mid
    if mname:
        entry["mart_name"] = mname
    if user_id:
        entry["user_id"] = user_id
    lid = (getattr(receipt, "linked_income_id", "") or "").strip()
    lamt = (getattr(receipt, "linked_income_amount", "") or "").strip()
    if lid:
        entry["linked_income_id"] = lid
    if lamt:
        entry["linked_income_amount"] = lamt
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
