"""
Receipt image analysis using EasyOCR (CNN-based; strong on thermal / POS receipts),
then structured extraction (merchant, phone, products) via receipt_scanner.parse_structured_from_ocr_text.

Env:
  EASYOCR_LANGS   Comma or plus-separated list, default "en" (e.g. "en,mya" if models installed).
  EASYOCR_GPU     Set to 1/true to use GPU when available.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

_reader: Optional[object] = None


def _easyocr_reader():
    global _reader
    if _reader is None:
        import easyocr

        raw = (os.getenv("EASYOCR_LANGS") or "en").strip()
        if "+" in raw:
            langs = [x.strip() for x in raw.split("+") if x.strip()]
        else:
            langs = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
        if not langs:
            langs = ["en"]
        gpu = os.getenv("EASYOCR_GPU", "").lower() in ("1", "true", "yes")
        _reader = easyocr.Reader(langs, gpu=gpu)
    return _reader


def preprocess_receipt_image(image_rgb: np.ndarray) -> np.ndarray:
    """Resize + denoise + binarize for faster, more stable receipt OCR."""
    if len(image_rgb.shape) != 3:
        return image_rgb
    rgb = image_rgb
    h0, w0 = rgb.shape[:2]
    max_side = max(h0, w0)
    # Downscale very large photos to reduce OCR latency while preserving detail.
    if max_side > 1800:
        scale = 1800.0 / float(max_side)
        rgb = cv2.resize(
            rgb,
            (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    m = min(h, w)
    if m < 800:
        scale = 800 / m
        gray = cv2.resize(
            gray,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)


def run_easyocr_as_lines(image_rgb: np.ndarray) -> str:
    """Run EasyOCR and return grouped lines in reading order."""
    reader = _easyocr_reader()
    arr = np.asarray(image_rgb)
    proc = preprocess_receipt_image(arr)
    try:
        detected = reader.readtext(proc, detail=1, paragraph=False)
    except Exception:
        detected = reader.readtext(arr, detail=1, paragraph=False)
    if not detected:
        return ""

    min_conf = 0.15
    rows = []
    heights: List[float] = []
    for box, text, conf in detected:
        t = (text or "").strip()
        if not t:
            continue
        try:
            c = float(conf)
        except Exception:
            c = 0.0
        if c < min_conf:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x_min = min(xs) if xs else 0.0
        y_min = min(ys) if ys else 0.0
        y_max = max(ys) if ys else y_min
        y_center = (y_min + y_max) / 2.0
        h = max(1.0, y_max - y_min)
        heights.append(h)
        rows.append((y_center, x_min, t))

    if not rows:
        return ""

    rows.sort(key=lambda r: (r[0], r[1]))
    median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 16.0
    join_thresh = max(10.0, median_h * 0.65)

    grouped: List[List[Tuple[float, float, str]]] = []
    current: List[Tuple[float, float, str]] = [rows[0]]
    current_y = rows[0][0]
    for y, x, t in rows[1:]:
        if abs(y - current_y) <= join_thresh:
            current.append((y, x, t))
            current_y = (current_y + y) / 2.0
        else:
            grouped.append(current)
            current = [(y, x, t)]
            current_y = y
    grouped.append(current)

    out: List[str] = []
    for grp in grouped:
        parts = [p[2] for p in sorted(grp, key=lambda z: z[1])]
        line = " ".join(parts).strip()
        line = " ".join(line.split())
        if not line:
            continue
        # Drop repeated duplicate lines from overlapping detection boxes.
        if out and out[-1].lower() == line.lower():
            continue
        out.append(line)
    return "\n".join(out)


def analyze_receipt_from_image(
    image_rgb: np.ndarray, source_name: str = ""
) -> Tuple[object, str]:
    """
    OCR the receipt image, then parse merchant / phone / date / currency / line items.

    Returns (ReceiptData, ocr_plain_text).
    """
    from receipt_scanner import parse_structured_from_ocr_text

    arr = np.asarray(image_rgb)
    ocr_text = run_easyocr_as_lines(arr)
    receipt = parse_structured_from_ocr_text(
        ocr_text,
        image=arr,
        source_name=source_name or "",
        source_path=None,
    )
    return receipt, ocr_text
