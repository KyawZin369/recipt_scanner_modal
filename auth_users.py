"""
Users collection in MongoDB + JWT sessions. Passwords stored as bcrypt hashes.
Env: JWT_SECRET (required in production), MONGO_USERS_COLLECTION (default: users).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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


def _users_collection_name() -> str:
    return os.getenv("MONGO_USERS_COLLECTION", "users").strip() or "users"


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
    col = _get_users_collection()
    if col is None:
        return None, "Database unavailable"

    email_norm = (email or "").strip().lower()
    uname = (username or "").strip()
    if not uname or not email_norm or not password:
        return None, "username, email, and password are required"
    if len(password) < 6:
        return None, "Password must be at least 6 characters"

    if col.find_one({"email": email_norm}):
        return None, "Email already registered"
    if col.find_one({"username": uname}):
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
    try:
        col.insert_one(doc)
    except PyMongoError as e:
        return None, str(e)

    return {"user": user_public_dict(doc), "access_token": token}, None


def authenticate_user(email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    """Returns ({user, access_token}, None) or (None, error)."""
    col = _get_users_collection()
    if col is None:
        return None, "Database unavailable"

    email_norm = (email or "").strip().lower()
    if not email_norm or not password:
        return None, "email and password are required"

    try:
        doc = col.find_one({"email": email_norm})
    except PyMongoError as e:
        return None, str(e)

    if not doc or not verify_password(password, doc.get("password_hash", "")):
        return None, "Invalid email or password"

    uid = doc.get("id")
    if not uid:
        return None, "Invalid user record"

    token = create_access_token(uid)
    try:
        col.update_one({"id": uid}, {"$set": {"user_token": token}})
    except PyMongoError:
        pass

    doc["user_token"] = token
    return {"user": user_public_dict(doc), "access_token": token}, None


def get_user_by_id(user_id: str) -> Optional[dict]:
    col = _get_users_collection()
    if col is None:
        return None
    try:
        return col.find_one({"id": user_id}, {"_id": 0})
    except PyMongoError:
        return None


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
    if col is None:
        return False
    try:
        col.update_one({"id": user_id}, {"$set": {"user_token": ""}})
        return True
    except PyMongoError:
        return False
