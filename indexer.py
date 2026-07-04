"""Thin DB layer shared by the hook, watcher, and backfill.

upsert() preserves enrichment columns with COALESCE so re-indexing a session
never clobbers its summary/topics/title/etc. archive() flips a flag — rows are
never deleted, preserving history.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sbconfig
from sources.base import SessionHeader

DB_PATH = sbconfig.DB_PATH


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


_UPSERT_SQL = """
INSERT INTO sessions
  (session_id, project_path, cwd, folder_name, start_time, last_activity,
   first_message, title, topics, turn_count, cli_source, cli_version, model_used)
VALUES (:session_id, :project_path, :cwd, :folder_name, :start_time, :last_activity,
        :first_message, :title, :topics, :turn_count, :cli_source, :cli_version, :model_used)
ON CONFLICT(session_id) DO UPDATE SET
  last_activity = excluded.last_activity,
  turn_count    = excluded.turn_count,
  cwd           = COALESCE(sessions.cwd, excluded.cwd),
  folder_name   = COALESCE(sessions.folder_name, excluded.folder_name),
  first_message = COALESCE(sessions.first_message, excluded.first_message),
  start_time    = COALESCE(sessions.start_time, excluded.start_time),
  title         = COALESCE(excluded.title, sessions.title),
  topics        = COALESCE(sessions.topics, excluded.topics),
  cli_source    = excluded.cli_source,
  cli_version   = COALESCE(sessions.cli_version, excluded.cli_version),
  model_used    = COALESCE(excluded.model_used, sessions.model_used);
"""


def _params(h: SessionHeader) -> dict:
    return {
        "session_id": h.session_id,
        "project_path": h.project_path,
        "cwd": h.cwd,
        "folder_name": h.folder_name,
        "start_time": h.start_time,
        "last_activity": h.last_activity,
        "first_message": h.first_message,
        "title": h.title,
        "topics": h.topics,
        "turn_count": h.turn_count,
        "cli_source": h.cli_source,
        "cli_version": h.cli_version,
        "model_used": h.model_used,
    }


def upsert(header: SessionHeader, conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(_UPSERT_SQL, _params(header))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def archive(session_id: str, conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute("UPDATE sessions SET archived = 1 WHERE session_id = ?", (session_id,))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
