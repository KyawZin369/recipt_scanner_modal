"""
Per-user income entries in MongoDB (separate from receipts / expenses).

Env: MONGO_INCOMES_COLLECTION (default: incomes)
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo.errors import PyMongoError

import receipt_scanner as _rs
from receipt_scanner import (
    _get_mongo_collection,
    _mongo_settings,
    get_last_mongo_error,
    set_last_mongo_error,
)


def _incomes_collection_name() -> str:
    return os.getenv("MONGO_INCOMES_COLLECTION", "incomes").strip() or "incomes"


def _get_incomes_collection():
    if _get_mongo_collection() is None:
        return None
    _, mongo_db_name, _ = _mongo_settings()
    try:
        return _rs._mongo_client[mongo_db_name][_incomes_collection_name()]
    except Exception as e:
        set_last_mongo_error(f"MongoDB incomes collection error: {e}")
        return None


def list_incomes_for_user(user_id: str) -> Optional[list[dict[str, Any]]]:
    """Return income docs for user, newest first. None if DB unavailable."""
    col = _get_incomes_collection()
    if col is None:
        return None
    try:
        return list(
            col.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
        )
    except PyMongoError as e:
        msg = str(e).strip()
        if len(msg) > 400:
            msg = msg[:400] + "…"
        set_last_mongo_error(f"MongoDB incomes query failed: {msg}")
        return None


def _parse_amount(value: str) -> float:
    if not value or not str(value).strip():
        return 0.0
    import re

    cleaned = re.sub(r"[^0-9.]", "", str(value))
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def incomes_summary_for_user(user_id: str) -> Optional[dict[str, Any]]:
    """
    Aggregate stats for dashboard cards.
    - total_all: sum of all entry amounts
    - count: number of entries
    - monthly_recurring: sum of amounts where recurrence == monthly
    - weekly_recurring: sum where recurrence == weekly (shown separately)
    """
    rows = list_incomes_for_user(user_id)
    if rows is None:
        return None
    total_all = 0.0
    monthly = 0.0
    weekly = 0.0
    yearly = 0.0
    spent_map = linked_spent_totals_by_income_id(user_id)
    total_spent_linked = round(sum(spent_map.values()), 2)
    remaining_across = 0.0
    for r in rows:
        amt = _parse_amount(str(r.get("amount", "") or ""))
        total_all += amt
        rec = (r.get("recurrence") or "one_time").strip().lower()
        if rec == "monthly":
            monthly += amt
        elif rec == "weekly":
            weekly += amt
        elif rec == "yearly":
            yearly += amt
        iid = (r.get("id") or "").strip()
        spent = float(spent_map.get(iid, 0.0))
        remaining_across += max(0.0, amt - spent)
    return {
        "count": len(rows),
        "total_all": round(total_all, 2),
        "monthly_recurring": round(monthly, 2),
        "weekly_recurring": round(weekly, 2),
        "yearly_recurring": round(yearly, 2),
        "total_spent_linked": total_spent_linked,
        "remaining_across_incomes": round(remaining_across, 2),
    }


def create_income_for_user(
    user_id: str,
    name: str,
    amount: str,
    recurrence: str,
    description: str,
) -> Optional[dict[str, Any]]:
    """Insert one income row. Returns document without _id, or None on failure."""
    col = _get_incomes_collection()
    if col is None:
        return None
    allowed = {"one_time", "monthly", "weekly", "yearly"}
    rec = (recurrence or "one_time").strip().lower()
    if rec not in allowed:
        rec = "one_time"
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "name": (name or "").strip(),
        "amount": (amount or "").strip(),
        "recurrence": rec,
        "description": (description or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not doc["name"]:
        return None
    try:
        col.insert_one(doc)
        return {k: v for k, v in doc.items() if k != "_id"}
    except PyMongoError:
        return None


def get_income_by_id_for_user(user_id: str, income_id: str) -> Optional[dict[str, Any]]:
    """Return one income doc for this user, or None if missing / DB down."""
    iid = (income_id or "").strip()
    if not iid:
        return None
    col = _get_incomes_collection()
    if col is None:
        return None
    try:
        return col.find_one({"user_id": user_id, "id": iid}, {"_id": 0})
    except PyMongoError:
        return None


def linked_spent_totals_by_income_id(user_id: str) -> dict[str, float]:
    """
    Sum `linked_income_amount` on expense/receipt docs per `linked_income_id`
    for this user (expenses live in the receipts Mongo collection).
    """
    col = _get_mongo_collection()
    if col is None:
        return {}
    out: dict[str, float] = {}
    try:
        for doc in col.find(
            {"user_id": user_id},
            {"linked_income_id": 1, "linked_income_amount": 1},
        ):
            lid = (doc.get("linked_income_id") or "").strip()
            if not lid:
                continue
            raw = doc.get("linked_income_amount")
            if raw is None or str(raw).strip() == "":
                continue
            amt = _parse_amount(str(raw))
            out[lid] = out.get(lid, 0.0) + amt
        return out
    except PyMongoError as e:
        msg = str(e).strip()
        if len(msg) > 400:
            msg = msg[:400] + "…"
        set_last_mongo_error(f"MongoDB linked expense aggregation failed: {msg}")
        return {}


def list_incomes_with_spending(user_id: str) -> Optional[list[dict[str, Any]]]:
    """Like [list_incomes_for_user] but each row includes spent_linked and remaining_after_expenses."""
    rows = list_incomes_for_user(user_id)
    if rows is None:
        return None
    spent_map = linked_spent_totals_by_income_id(user_id)
    enriched: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        iid = (d.get("id") or "").strip()
        spent = float(spent_map.get(iid, 0.0))
        principal = _parse_amount(str(d.get("amount", "") or ""))
        remaining = max(0.0, principal - spent)
        d["spent_linked"] = round(spent, 2)
        d["remaining_after_expenses"] = round(remaining, 2)
        enriched.append(d)
    return enriched
