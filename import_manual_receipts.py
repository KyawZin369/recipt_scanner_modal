import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from supabase_storage import upload_receipt_image_file


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main():
    base = Path(__file__).resolve().parent
    load_dotenv(base / ".env")

    mongo_uri = os.getenv("MONGO_URI", "").strip()
    db_name = os.getenv("MONGO_DB_NAME", "expenses").strip() or "expenses"
    coll_name = os.getenv("MONGO_COLLECTION_NAME", "receipts").strip() or "receipts"

    if not mongo_uri:
        raise RuntimeError("MONGO_URI is required in .env")

    src_json = base / "mongodb_receipts_manual.json"
    if not src_json.is_file():
        raise RuntimeError(f"Missing file: {src_json}")
    receipts = json.loads(src_json.read_text(encoding="utf-8"))

    image_dir = base / "manual_receipt_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Also check asset/ for one-off images, e.g. asset/citymart.png
    asset_dir = base / "asset"

    # Naming convention:
    # manual_receipt_images/<date>_<merchant_slug>.jpg
    uploaded = 0
    updated = 0
    missing_images = 0

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    try:
        coll = client[db_name][coll_name]
        for r in receipts:
            merchant = (r.get("merchant") or "").strip()
            date = (r.get("date") or "").strip()
            key = f"{date}_{_slug(merchant)}"

            image_url = (r.get("image_url") or "").strip()
            if not image_url:
                candidates = [
                    image_dir / f"{key}.jpg",
                    image_dir / f"{key}.jpeg",
                    image_dir / f"{key}.png",
                    image_dir / f"{key}.webp",
                ]
                # City Mart example asset
                if _slug(merchant) == "city_mart_junction":
                    candidates.insert(0, asset_dir / "citymart.png")

                img_path = next((p for p in candidates if p.is_file()), None)
                if img_path is not None:
                    try:
                        image_url = upload_receipt_image_file(img_path)
                        if image_url:
                            uploaded += 1
                    except Exception:
                        image_url = ""
                if not image_url:
                    missing_images += 1

            doc = {
                "merchant": merchant,
                "date": date,
                "currency": r.get("currency", ""),
                "items": r.get("items", []),
                "subtotal": r.get("subtotal", ""),
                "tax": r.get("tax", ""),
                "total": r.get("total", ""),
                "warnings": r.get("warnings", []),
                "raw_text": r.get("raw_text", ""),
                "image_url": image_url,
            }

            coll.update_one(
                {"merchant": merchant, "date": date},
                {"$set": doc},
                upsert=True,
            )
            updated += 1

        print(f"Upserted {updated} receipts into {db_name}.{coll_name}")
        print(f"Uploaded {uploaded} images to Supabase Storage")
        print(f"Missing/empty image_url for {missing_images} receipts")
        print(f'Put images in: "{image_dir}" or "asset/" (see script for City Mart)')
    except PyMongoError as e:
        raise RuntimeError(f"MongoDB error: {e}") from e
    finally:
        client.close()


if __name__ == "__main__":
    main()
