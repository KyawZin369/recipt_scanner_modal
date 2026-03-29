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
| `POST` | `/storage/receipts/upload` | Upload image to Supabase bucket → `{ image_url, bucket, path }` (multipart field **`file`**) |
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
  "image_url": "https://....supabase.co/storage/v1/object/public/receipts/...",
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

### 3. Configure MongoDB for expense storage

The app can store expenses in MongoDB. Set these environment variables before starting the API:

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB_NAME="expenses"          # optional (default: expenses)
export MONGO_COLLECTION_NAME="receipts"  # optional (default: receipts)
```

If `MONGO_URI` is not set or MongoDB is unavailable, the app falls back to `expenses.json`.

### 3.1 Optional: Store receipt images in Supabase Storage

Receipt images are uploaded to **Supabase Storage**; the public (or signed) URL is returned as `image_url` and saved with expenses in MongoDB.

1. In Supabase: **Storage → New bucket** (e.g. `receipts`). For public URLs in the app, mark the bucket **public** or use signed URLs (see below).
2. In **Project Settings → API**, copy **Project URL** and **service_role** key (server-side only — never ship this in a client app).

```bash
export SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
export SUPABASE_STORAGE_BUCKET="receipts"
```

If the bucket is **private**, use long-lived signed URLs:

```bash
export SUPABASE_USE_SIGNED_URLS=1
```

**Performance:** the API keeps a **pooled MongoDB connection** (no new TCP/TLS handshake on every `GET /receipts`). Restart `uvicorn` after changing `.env`.

**Faster receipt OCR (optional):** trade a bit of accuracy for speed:

```bash
export FAST_RECEIPT_OCR=1
```

Then start `uvicorn` as usual.

**`GET /receipts` returns 503:** MongoDB is required for that route. Open the response `detail` for the exact error (common: Atlas **Network Access** must allow your IP, wrong password, or `MONGO_URI` not loaded — use `.env` next to `api.py` or export vars before `uvicorn`).

### 4. (Optional) Train the scanner model

Train a small scikit-learn classifier to predict “has text” vs “no text”:

```bash
python train_scanner.py
```

Uses synthetic data by default. For better results, add your own images to `data/with_text/` and `data/no_text/`.

### 5. Run the API (for Flutter)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

### 6. (Optional) Run the Streamlit demo UI

```bash
streamlit run app.py
```

## Project structure

- **`api.py`** – **REST API (use this from Flutter):** `/scan/receipt`, `/scan/text`, `/expenses`
- `receipt_scanner.py` – Receipt OCR and parsing; strict prices; expense storage (MongoDB with JSON fallback)
- `scanner.py` – Text OCR and summary
- `app.py` – Streamlit demo UI (optional)
- `train_scanner.py` – Train “has text” classifier
- `models/` – Saved model (optional)
- `requirements.txt` – Python dependencies

## Customization

- **AI output:** Edit `scanner.py` and the `generate_output_from_text()` function to call your own API (e.g. OpenAI, Claude) or a local model for summaries or Q&A on the scanned text.
- **OCR language:** In `scanner.py`, set `lang='eng'` (or e.g. `'eng+fra'`) in the `pytesseract.image_to_string()` call to match your documents.
