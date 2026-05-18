"""
Users collection in MongoDB + JWT sessions. Passwords stored as bcrypt hashes.
Env: JWT_SECRET (required in production), MONGO_USERS_COLLECTION (default: users).
"""
from __future__ import annotations

import os
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from pathlib import Path

try:
    import bcrypt
except Exception:  # pragma: no cover
    bcrypt = None

try:
    import jwt
except Exception:  # pragma: no cover
    jwt = None

from pymongo.errors import PyMongoError

import receipt_scanner as _rs

from receipt_scanner import _get_mongo_collection, _mongo_settings, set_last_mongo_error

JWT_ALG = "HS256"
JWT_EXPIRE_DAYS = 30
LOCAL_USERS_PATH = Path(__file__).resolve().parent / "users_local.json"


def _users_collection_name() -> str:
    return os.getenv("MONGO_USERS_COLLECTION", "users").strip() or "users"


def _allow_local_auth_fallback() -> bool:
    """
    Local JSON users are a development fallback only.
    Default OFF to avoid "registered but not in MongoDB" confusion.
    """
    return False


def _load_local_users() -> list[dict]:
    if not LOCAL_USERS_PATH.is_file():
        return []
    try:
        data = json.loads(LOCAL_USERS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_local_users(users: list[dict]) -> None:
    LOCAL_USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _get_users_collection():
    if _get_mongo_collection() is None:
        return None
    _, mongo_db_name, _ = _mongo_settings()
    try:
        # Use _rs._mongo_client — importing _mongo_client by name would stay stuck at None
        # after the pooled client is created in _get_mongo_collection().
        return _rs._mongo_client[mongo_db_name][_users_collection_name()]
    except Exception as e:
        set_last_mongo_error(f"MongoDB users collection error: {e}")
        return None


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "dev-only-change-me").strip()


def hash_password(plain: str) -> str:
    if bcrypt is None:
        raise RuntimeError("bcrypt is not installed")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    if bcrypt is None:
        return False
    try:
        return bcrypt.checkpw(
            plain.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(user_id: str) -> str:
    if jwt is None:
        raise RuntimeError("PyJWT is not installed")
    exp = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": exp}
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALG)


def decode_token(token: str) -> Optional[dict]:
    if jwt is None:
        return None
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALG])
    except jwt.InvalidTokenError:
        return None


def user_public_dict(doc: dict) -> dict:
    return {
        "id": doc.get("id", ""),
        "username": doc.get("username", ""),
        "email": doc.get("email", ""),
    }


def create_user(username: str, email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Insert user. Returns (user_doc_public, error_message).
    On success: (dict with id, username, email, access_token), None
    """
    email_norm = (email or "").strip().lower()
    uname = (username or "").strip()
    if not uname or not email_norm or not password:
        return None, "username, email, and password are required"
    if len(password) < 6:
        return None, "Password must be at least 6 characters"

    col = _get_users_collection()
    if col is None and not _allow_local_auth_fallback():
        return None, "Database unavailable"
    if col is not None:
        try:
            if col.find_one({"email": email_norm}):
                return None, "Email already registered"
            if col.find_one({"username": uname}):
                return None, "Username already taken"
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users query failed: {e}")
            return None, "Database unavailable"
    else:
        users = _load_local_users()
        if any((u.get("email") or "").lower() == email_norm for u in users):
            return None, "Email already registered"
        if any((u.get("username") or "") == uname for u in users):
            return None, "Username already taken"

    uid = str(uuid.uuid4())
    pw_hash = hash_password(password)
    token = create_access_token(uid)
    doc: dict[str, Any] = {
        "id": uid,
        "username": uname,
        "email": email_norm,
        "password_hash": pw_hash,
        "user_token": token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if col is not None:
        try:
            col.insert_one(doc)
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users insert failed: {e}")
            return None, "Database unavailable"
    else:
        users = _load_local_users()
        users.append(doc)
        _save_local_users(users)

    return {"user": user_public_dict(doc), "access_token": token}, None


def authenticate_user(email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    """Returns ({user, access_token}, None) or (None, error)."""
    email_norm = (email or "").strip().lower()
    if not email_norm or not password:
        return None, "email and password are required"

    col = _get_users_collection()
    if col is None and not _allow_local_auth_fallback():
        return None, "Database unavailable"
    if col is not None:
        try:
            doc = col.find_one({"email": email_norm})
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users query failed: {e}")
            return None, "Database unavailable"
    else:
        users = _load_local_users()
        doc = next((u for u in users if (u.get("email") or "").lower() == email_norm), None)

    if not doc or not verify_password(password, doc.get("password_hash", "")):
        return None, "Invalid email or password"

    uid = doc.get("id")
    if not uid:
        return None, "Invalid user record"

    token = create_access_token(uid)
    if col is not None:
        try:
            col.update_one({"id": uid}, {"$set": {"user_token": token}})
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users update failed: {e}")
    else:
        users = _load_local_users()
        for u in users:
            if u.get("id") == uid:
                u["user_token"] = token
                break
        _save_local_users(users)

    doc["user_token"] = token
    return {"user": user_public_dict(doc), "access_token": token}, None


def get_user_by_id(user_id: str) -> Optional[dict]:
    col = _get_users_collection()
    if col is None and not _allow_local_auth_fallback():
        return None
    if col is not None:
        try:
            return col.find_one({"id": user_id}, {"_id": 0})
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users query failed: {e}")
            return None
    users = _load_local_users()
    return next((u for u in users if u.get("id") == user_id), None)


def verify_session_token(token: str) -> Optional[dict]:
    """Validate JWT and optional DB match on user_token."""
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        return None
    uid = payload["sub"]
    doc = get_user_by_id(uid)
    if not doc:
        return None
    if doc.get("user_token") != token:
        return None
    return doc


def logout_user(user_id: str) -> bool:
    col = _get_users_collection()
    if col is None and not _allow_local_auth_fallback():
        return False
    if col is not None:
        try:
            col.update_one({"id": user_id}, {"$set": {"user_token": ""}})
            return True
        except PyMongoError as e:
            set_last_mongo_error(f"MongoDB users update failed: {e}")
            return False
    users = _load_local_users()
    changed = False
    for u in users:
        if u.get("id") == user_id:
            u["user_token"] = ""
            changed = True
            break
    if changed:
        _save_local_users(users)
    return changed
