"""
Start the FastAPI receipt scanner API from this folder.

Use this instead of `uvicorn api:app` when your shell cwd might be wrong,
which can load a different `api` module and yield 404 on /incomes/* while
/auth and /receipts still work on another process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(_ROOT)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    import uvicorn

    print(f"[run_server] cwd={_ROOT}", flush=True)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
