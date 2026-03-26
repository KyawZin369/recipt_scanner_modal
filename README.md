# AI Scanner Project

A Python **AI module** for scanning receipts and text. Use it from a **Flutter app** (or any client) by running the REST API—no Flutter code in this repo.

## What it does

- **Receipt scan** – Send a receipt image → get structured data: merchant, date, line items (product name + price). Prices are strict (no guessing).
- **Text scan** – Send an image → get extracted text and a short summary.
- **Expenses** – Add scanned receipts to expenses via API; list or export them.

## Using this module from Flutter

This project is **model/API only**. Run the API server, then call it from your Flutter app.

### 1. Start the API server

```bash
cd "AI project with python"
source venv/bin/activate   # or venv\Scripts\activate on Windows
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Docs: **http://localhost:8000/docs**
- From a device/emulator use your machine’s IP, e.g. `http://192.168.1.10:8000`.

### 2. Endpoints your Flutter app can call

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check if API is up |
| `POST` | `/scan/receipt` | Upload image (multipart) → receipt JSON (merchant, date, items, total) |
| `POST` | `/scan/text` | Upload image (multipart) → `{ "text", "summary" }` |
| `POST` | `/expenses` | Add a receipt to expenses (JSON body) |
| `GET` | `/expenses` | List all saved expenses |

### 3. Example: scan receipt from Flutter

- **Request:** `POST /scan/receipt` with `multipart/form-data`, field name **`image`**, file = receipt image (e.g. JPEG/PNG).
- **Response (200):**
```json
{
  "merchant": "Store Name",
  "date": "01/15/2025",
  "items": [
    { "product": "Item A", "price": "12.99", "needs_review": false },
    { "product": "Item B", "price": "3.50", "needs_review": false }
  ],
  "subtotal": "16.49",
  "tax": "1.65",
  "total": "18.14",
  "warnings": [],
  "raw_text": "..."
}
```

In Flutter: use `http` or `dio` with `MultipartRequest` / `FormData`, attach the image file as `image`, then parse this JSON.

### 4. Example: scan text from Flutter

- **Request:** `POST /scan/text` with `multipart/form-data`, field **`image`**.
- **Response (200):**
```json
{
  "text": "Extracted text from the image...",
  "summary": "The scan contains about 24 words. Here is the text: ..."
}
```

## Setup

### 1. Install Tesseract OCR (required for text extraction)

- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **Windows:** Download from [GitHub – tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Create a virtual environment and install dependencies

```bash
cd "AI project with python"
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Train the scanner model

Train a small scikit-learn classifier to predict “has text” vs “no text”:

```bash
python train_scanner.py
```

Uses synthetic data by default. For better results, add your own images to `data/with_text/` and `data/no_text/`.

### 4. Run the API (for Flutter)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

### 5. (Optional) Run the Streamlit demo UI

```bash
streamlit run app.py
```

## Project structure

- **`api.py`** – **REST API (use this from Flutter):** `/scan/receipt`, `/scan/text`, `/expenses`
- `receipt_scanner.py` – Receipt OCR and parsing; strict prices; expense storage (expenses.json)
- `scanner.py` – Text OCR and summary
- `app.py` – Streamlit demo UI (optional)
- `train_scanner.py` – Train “has text” classifier
- `models/` – Saved model (optional)
- `requirements.txt` – Python dependencies

## Customization

- **AI output:** Edit `scanner.py` and the `generate_output_from_text()` function to call your own API (e.g. OpenAI, Claude) or a local model for summaries or Q&A on the scanned text.
- **OCR language:** In `scanner.py`, set `lang='eng'` (or e.g. `'eng+fra'`) in the `pytesseract.image_to_string()` call to match your documents.
