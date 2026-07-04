#!/usr/bin/env python3
"""Idempotent schema migration for registry.db.

Base tables via executescript; additive columns via a PRAGMA-guarded helper
(SQLite has no ADD COLUMN IF NOT EXISTS). Safe to run any number of times.
Sets WAL + busy_timeout so the hook, watcher, Flask app, and nightly enrich can
share the DB without lock errors.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sbconfig  # noqa: E402

BASE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, project_path TEXT, cwd TEXT, folder_name TEXT,
    start_time TEXT, last_activity TEXT, first_message TEXT, summary TEXT,
    topics TEXT, session_type TEXT, outcome TEXT, turn_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS session_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    type TEXT NOT NULL, content TEXT NOT NULL, turn_index INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON session_artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type    ON session_artifacts(type);
CREATE TABLE IF NOT EXISTS session_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    turn_start INTEGER NOT NULL, turn_end INTEGER NOT NULL, summary TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_cp_session ON session_checkpoints(session_id);
CREATE TABLE IF NOT EXISTS session_snapshots (
    session_id TEXT PRIMARY KEY, goal TEXT, decisions TEXT, artifacts TEXT,
    unresolved TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
-- Embeddings stored as float32 BLOBs; queried via numpy brute-force cosine.
-- (No native extension required; fast for up to a few thousand sessions.)
CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, embedding BLOB NOT NULL,
    source_text TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""

# Additive columns on `sessions` — (column_definition) applied only if missing.
ADDITIVE_COLUMNS = [
    "title TEXT",
    "input_tokens INTEGER",
    "output_tokens INTEGER",
    "cache_read_tokens INTEGER",
    "cache_write_tokens INTEGER",
    "model_used TEXT",
    "models_used TEXT",
    "cli_source TEXT NOT NULL DEFAULT 'claude'",
    "cli_version TEXT",
    "archived INTEGER NOT NULL DEFAULT 0",
    "notes TEXT",
    "end_logged_at TIMESTAMP",
    "cost_usd REAL",
    "enriched_at TIMESTAMP",
    "reasoning_path TEXT",
]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str) -> bool:
    col = col_def.split()[0]
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col in have:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    return True


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(BASE_DDL)
    for col_def in ADDITIVE_COLUMNS:
        _add_column_if_missing(conn, "sessions", col_def)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_source "
        "ON sessions(cli_source, last_activity DESC)"
    )
    # Full-text index over transcript turn text (best-effort: needs FTS5).
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts "
            "USING fts5(session_id UNINDEXED, body)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[migrate] FTS5 unavailable ({e}); full-text search disabled.", file=sys.stderr)
    # Best-effort native vec table — only if an extension-capable sqlite3 is present.
    try:
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(True)
            import sqlite_vec  # type: ignore
            sqlite_vec.load(conn)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_vec "
                "USING vec0(session_id TEXT PRIMARY KEY, embedding FLOAT[384])"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[migrate] sqlite-vec unavailable ({e}); using numpy backend.", file=sys.stderr)
    conn.commit()


def main() -> None:
    sbconfig.ensure_dirs()
    conn = sqlite3.connect(str(sbconfig.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    migrate(conn)
    conn.close()
    print(f"[migrate] OK -> {sbconfig.DB_PATH}")


if __name__ == "__main__":
    main()
