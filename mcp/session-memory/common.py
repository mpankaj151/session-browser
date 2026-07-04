"""Shared helpers for the session-memory MCP server."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the repo root importable so we reuse indexer / semsearch / sbconfig.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402

DB_PATH = os.environ.get("SESSION_MEMORY_DB", str(indexer.DB_PATH))
MAX_BYTES = 2048


def connect():
    # Friendly failure pre-install: FastMCP surfaces the exception message as
    # the tool error, so make it actionable instead of a raw SQLite error.
    if not Path(DB_PATH).exists():
        raise RuntimeError(
            "Session index not built yet — run ./install.sh (or scripts/migrate-db.py "
            "+ scripts/backfill.py) in the session-browser repo first.")
    conn = indexer.connect(DB_PATH)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
    if row is None:
        conn.close()
        raise RuntimeError(
            "Session database has no schema — run scripts/migrate-db.py first.")
    return conn


def clamp(items: list, serialize=json.dumps) -> list:
    """Trim a list so its serialized form stays under MAX_BYTES; note what was cut."""
    out: list = []
    for i, item in enumerate(items):
        trial = serialize(out + [item])
        if len(trial.encode()) > MAX_BYTES and out:
            out.append({"_truncated": True, "skipped": len(items) - i})
            break
        out.append(item)
    return out


def semantic_or_keyword(query: str, limit: int):
    """Try semantic search; fall back to keyword LIKE. Returns list of session_id."""
    try:
        import semsearch
        hits = semsearch.search(query, limit=limit)
        if hits:
            return [sid for sid, _ in hits], {sid: round(sc, 3) for sid, sc in hits}
    except Exception:  # noqa: BLE001
        pass
    conn = connect()
    try:
        like = f"%{query}%"
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE archived = 0 AND "
            "(first_message LIKE ? OR summary LIKE ? OR title LIKE ? OR topics LIKE ?) "
            "ORDER BY last_activity DESC LIMIT ?",
            (like, like, like, like, limit),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows], {}
