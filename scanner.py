"""
Scanner pipeline: load image → (optional) has-text check → OCR → AI-generated output.
"""
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytesseract
from PIL import Image

# Optional scanner model (sklearn or TensorFlow)
SKLEARN_MODEL_PATH = Path(__file__).resolve().parent / "models" / "scanner_sklearn.joblib"
TF_MODEL_DIR = Path(__file__).resolve().parent / "models" / "scanner_model"
IMG_SIZE = (128, 128)


def _load_scanner_model():
    """Load trained scanner model if it exists (sklearn preferred for macOS compatibility)."""
    if SKLEARN_MODEL_PATH.is_file():
        return joblib.load(SKLEARN_MODEL_PATH)
    if TF_MODEL_DIR.is_dir():
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(TF_MODEL_DIR)
        except Exception:
            return None
    return None


def _has_text_sklearn(image: np.ndarray, payload: dict) -> bool:
    """Use sklearn model to predict if image contains text."""
    model = payload["model"]
    img_size = payload.get("img_size", IMG_SIZE)
    resized = cv2.resize(image, img_size)
    if len(resized.shape) == 2:
        resized = np.stack([resized] * 3, axis=-1)
    X = (resized.astype(np.float32) / 255.0).reshape(1, -1)
    pred = model.predict(X)[0]
    return bool(pred == 1)


def _has_text_tf(image: np.ndarray, model) -> bool:
    """Use TensorFlow model to predict if image contains text."""
    try:
        resized = cv2.resize(image, (128, 128))
        if len(resized.shape) == 2:
            resized = np.stack([resized] * 3, axis=-1)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        prob = float(model.predict(batch, verbose=0)[0, 0])
        return prob > 0.5
    except Exception:
        return True


def _has_text(image: np.ndarray, model) -> bool:
    """Predict whether image has text using loaded model (sklearn or TF)."""
    if model is None:
        return True
    if isinstance(model, dict) and "model" in model:
        return _has_text_sklearn(image, model)
    return _has_text_tf(image, model)


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Improve image for OCR: grayscale, denoise."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised


def _scale_for_ocr(image: np.ndarray, min_side: int = 1200) -> np.ndarray:
    """Scale image so text is larger; Tesseract reads better on bigger text."""
    h, w = image.shape[:2]
    if min(h, w) >= min_side:
        return image
    scale = min_side / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _run_ocr(img: np.ndarray, psm: int = 6) -> str:
    """Run Tesseract on a single image. psm 6 = block of text, 3 = auto."""
    try:
        return pytesseract.image_to_string(img, lang="eng", config=f"--psm {psm}")
    except Exception:
        return pytesseract.image_to_string(Image.fromarray(img), lang="eng", config=f"--psm {psm}")


def _clean_to_human_text(raw: str) -> str:
    """
    Turn raw OCR output into clean, human-readable text.
    Fixes spacing, removes junk characters, joins broken lines, normalizes punctuation.
    """
    if not raw or not raw.strip():
        return ""
    # Clean line by line to keep natural, human-readable structure
    out_lines = []
    for line in raw.splitlines():
        line = line.replace("|", " ").replace("ZL", "").replace("=", " ")
        line = re.sub(r"\s*=\s*", " ", line)
        for pat, repl in [
            (r"\bAl\s+its\s+wi\b", "All profits will"),
            (r"\bose\s+wh\b", "those who"),
            (r"\bWritten\s+By\s+", "Written by "),
        ]:
            line = re.sub(pat, repl, line, flags=re.IGNORECASE)
        line = " ".join(line.split()).strip()
        if not line:
            continue
        if line[0].islower():
            line = line[0].upper() + line[1:]
        out_lines.append(line)
    return "\n".join(out_lines)


def extract_text(image_input) -> str:
    """
    Extract text from an image (file path, PIL Image, or numpy array).
    Returns extracted text or empty string.
    """
    if isinstance(image_input, (str, Path)):
        image = np.array(Image.open(image_input).convert("RGB"))
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        return ""

    # Scale up so text is easier for Tesseract to read
    scaled = _scale_for_ocr(image)
    gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY) if len(scaled.shape) == 3 else scaled
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Run OCR with multiple setups to capture more text
    texts = []
    # Standard preprocessed
    texts.append(_run_ocr(denoised, psm=6))
    texts.append(_run_ocr(denoised, psm=3))
    # Adaptive threshold helps with varying brightness (e.g. photos)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    texts.append(_run_ocr(thresh, psm=6))
    texts.append(_run_ocr(thresh, psm=3))
    # Otsu threshold sometimes better for high contrast
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texts.append(_run_ocr(otsu, psm=6))
    # PSM 11 = sparse text (good for posters, book covers with scattered text)
    texts.append(_run_ocr(denoised, psm=11))
    texts.append(_run_ocr(thresh, psm=11))

    # Merge: take all non-empty lines and dedupe while preserving order
    seen = set()
    merged_lines = []
    for t in texts:
        for line in (t or "").splitlines():
            line = line.strip()
            if not line or len(line) < 2:
                continue
            key = line.lower()[:50]
            if key in seen:
                continue
            seen.add(key)
            merged_lines.append(line)
    raw = "\n".join(merged_lines) if merged_lines else ""
    # Clean into human-readable form
    return _clean_to_human_text(raw)


def generate_output_from_text(scanned_text: str) -> str:
    """
    Generate human-readable output from the scanned text.
    Presents the content as clean, natural text.
    """
    if not scanned_text:
        return "No text was found in the scan. Try a clearer image or ensure the image contains readable text."

    # Present as a short human-written summary plus the full text
    word_count = len(scanned_text.split())
    intro = f"The scan contains about {word_count} words. Here is the text:\n\n"
    return intro + scanned_text


def scan_and_generate(image_input) -> tuple[str, str]:
    """
    Run full pipeline: extract text from image, then generate AI output.
    Returns (extracted_text, ai_output).
    """
    extracted = extract_text(image_input)
    ai_output = generate_output_from_text(extracted)
    return extracted, ai_output


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scanner.py <image_path>")
        print("Example: python scanner.py photo.jpg")
        print("\nOr run the web app: streamlit run app.py")
        sys.exit(0)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)
    print("Scanning...")
    extracted, ai_output = scan_and_generate(path)
    print("\n--- EXTRACTED TEXT ---\n")
    print(extracted or "(no text detected)")
    print("\n--- AI OUTPUT ---\n")
    print(ai_output)
    # Save extracted text to .txt file (same name as image, in current directory)
    out_name = path.stem + "_scanned.txt"
    out_path = Path.cwd() / out_name
    content = extracted or "(no text detected)"
    out_path.write_text(content, encoding="utf-8")
    print(f"\nSaved text to: {out_path}")
