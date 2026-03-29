"""
Upload receipt images to Supabase Storage and return a public (or signed) URL.
Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY + SUPABASE_STORAGE_BUCKET in .env.
"""
from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


def _object_path_for_filename(filename: str, folder: str = "receipts") -> tuple[str, str]:
    """Returns (extension, storage path inside bucket)."""
    ext = Path(filename or "receipt.jpg").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        ext = ".jpg"
    folder = (folder or "receipts").strip().strip("/")
    object_path = f"{folder}/{uuid.uuid4().hex}{ext}"
    return ext, object_path


def upload_image_to_bucket(
    contents: bytes, filename: str, folder: str = "receipts"
) -> tuple[str, str, str]:
    """
    Upload bytes under `folder/` in the configured Supabase bucket.

    Returns (image_url, object_path, bucket_name). On failure returns ("", "", "").
    """
    if create_client is None:
        return "", "", ""

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv(
        "SUPABASE_KEY", ""
    ).strip()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "receipts").strip() or "receipts"

    if not url or not key:
        return "", "", ""

    _, object_path = _object_path_for_filename(filename, folder=folder)
    content_type = mimetypes.guess_type(filename or "receipt.jpg")[0] or "image/jpeg"

    try:
        client = create_client(url, key)
        storage = client.storage.from_(bucket)
        storage.upload(
            object_path,
            contents,
            file_options={
                "content-type": content_type,
                "upsert": "true",
            },
        )
    except Exception:
        return "", "", ""

    return _public_url_for_path(storage, object_path, bucket)


def _public_url_for_path(storage, object_path: str, bucket: str) -> tuple[str, str, str]:
    use_signed = os.getenv("SUPABASE_USE_SIGNED_URLS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_signed:
        try:
            signed = storage.create_signed_url(object_path, 60 * 60 * 24 * 365)
            if isinstance(signed, dict):
                u = signed.get("signedURL") or signed.get("signedUrl") or ""
                return u, object_path, bucket
            return str(signed), object_path, bucket
        except Exception:
            return "", "", ""

    try:
        public = storage.get_public_url(object_path) or ""
        return public, object_path, bucket
    except Exception:
        return "", "", ""


def upload_receipt_image_to_bucket(contents: bytes, filename: str) -> tuple[str, str, str]:
    """Upload receipt image under `receipts/` in the bucket."""
    return upload_image_to_bucket(contents, filename, folder="receipts")


def upload_receipt_image_bytes(contents: bytes, filename: str) -> str:
    """Upload bytes; returns public/signed URL or "" on failure."""
    url, _, _ = upload_receipt_image_to_bucket(contents, filename)
    return url


def upload_receipt_image_file(path: Path) -> str:
    """Upload a local image file; same URL rules as upload_receipt_image_bytes."""
    if not path.is_file():
        return ""
    return upload_receipt_image_bytes(path.read_bytes(), path.name)
