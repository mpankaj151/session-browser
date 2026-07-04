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
  -- Monotonic: a re-parse of a SHORTER view of the transcript (partial cloud
  -- sync, a second codex rollout file for the same id) must not walk the row
  -- backwards. MAX() returns NULL if either arg is NULL, hence the COALESCEs.
  last_activity = MAX(COALESCE(sessions.last_activity, ''), COALESCE(excluded.last_activity, '')),
  turn_count    = MAX(COALESCE(sessions.turn_count, 0), COALESCE(excluded.turn_count, 0)),
  -- The canonical transcript dir follows the newest activity (codex `resume`
  -- can write a second rollout file in a different date dir).
  project_path  = CASE WHEN COALESCE(excluded.last_activity, '') >= COALESCE(sessions.last_activity, '')
                       THEN excluded.project_path ELSE sessions.project_path END,
  -- NULLIF treats '' as absent: adapters emit '' for not-yet-known fields (e.g. the
  -- watcher fires before the first user turn is flushed), and a '' must neither
  -- stick nor overwrite a real value.
  cwd           = COALESCE(NULLIF(sessions.cwd, ''), NULLIF(excluded.cwd, '')),
  folder_name   = COALESCE(NULLIF(sessions.folder_name, ''), NULLIF(excluded.folder_name, '')),
  first_message = COALESCE(NULLIF(sessions.first_message, ''), NULLIF(excluded.first_message, '')),
  start_time    = COALESCE(NULLIF(sessions.start_time, ''), NULLIF(excluded.start_time, '')),
  title         = COALESCE(NULLIF(excluded.title, ''), sessions.title),
  topics        = COALESCE(sessions.topics, excluded.topics),
  cli_source    = excluded.cli_source,
  cli_version   = COALESCE(NULLIF(sessions.cli_version, ''), NULLIF(excluded.cli_version, '')),
  model_used    = COALESCE(NULLIF(excluded.model_used, ''), sessions.model_used),
  -- an upsert only ever comes from parsing a file that exists on disk, so the
  -- session is alive: resurrect it if it was (possibly wrongly) archived.
  archived      = 0;
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
