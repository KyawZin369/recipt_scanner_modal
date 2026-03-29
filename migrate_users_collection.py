#!/usr/bin/env python3
"""
Migrate / prepare the MongoDB `users` collection for the auth API.

Creates unique indexes on `email` and `username` (idempotent — safe to re-run).

Usage (from this directory, with `.env` containing MONGO_URI):

  python migrate_users_collection.py
  python migrate_users_collection.py --dry-run
  python migrate_users_collection.py --verify-only

Environment (same as `api.py`):
  MONGO_URI            — required
  MONGO_DB_NAME        — default: expenses
  MONGO_USERS_COLLECTION — default: users

If index creation fails with duplicate key error, remove or fix duplicate
documents first (same email or username must not appear twice).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, PyMongoError
except ImportError:
    print("Install pymongo: pip install pymongo", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate MongoDB users collection (indexes for auth)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions and exit without connecting.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only ping MongoDB and print collection stats (no index changes).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if load_dotenv:
        load_dotenv(root / ".env")

    uri = os.getenv("MONGO_URI", "").strip()
    db_name = os.getenv("MONGO_DB_NAME", "expenses").strip() or "expenses"
    coll_name = os.getenv("MONGO_USERS_COLLECTION", "users").strip() or "users"

    print(f"Target: db={db_name!r}, collection={coll_name!r}")

    if args.dry_run:
        if not uri:
            print("MONGO_URI would be required (set in .env).", file=sys.stderr)
            return 1
        print("[dry-run] Would: ping cluster, ensure indexes uniq_email, uniq_username")
        return 0

    if not uri:
        print("MONGO_URI is not set in .env", file=sys.stderr)
        return 1

    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
        )
        client.admin.command("ping")
    except PyMongoError as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1

    coll = client[db_name][coll_name]
    doc_count = coll.count_documents({})
    print(f"Connected. Documents in `{coll_name}`: {doc_count}")

    if args.verify_only:
        for idx in coll.list_indexes():
            print(f"  index: {idx['name']} keys={dict(idx['key'])}")
        client.close()
        print("Verify OK.")
        return 0

    # Unique indexes (match how auth looks up users)
    def _ensure_unique_index(field: str, index_name: str) -> bool:
        try:
            coll.create_index(field, unique=True, name=index_name)
            print(f"OK: unique index on `{field}` ({index_name})")
            return True
        except OperationFailure as e:
            # 11000 = duplicate key in existing data; 85/86 = index exists / conflict
            if getattr(e, "code", None) == 11000:
                print(
                    f"Failed: duplicate `{field}` values exist. "
                    "Remove or merge duplicate documents, then re-run.\n"
                    f"  {e}",
                    file=sys.stderr,
                )
                return False
            if getattr(e, "code", None) in (85, 86):
                print(f"Note ({field} index): {e}")
                return True
            print(f"Warning ({field} index): {e}", file=sys.stderr)
            return True

    if not _ensure_unique_index("email", "uniq_email"):
        client.close()
        return 1
    if not _ensure_unique_index("username", "uniq_username"):
        client.close()
        return 1

    print("\nCurrent indexes:")
    for idx in coll.list_indexes():
        keys = dict(idx["key"])
        uniq = idx.get("unique", False)
        print(f"  - {idx['name']}: {keys} unique={uniq}")

    client.close()
    print("\nMigration finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
