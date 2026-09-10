"""Thin DB layer shared by the hook, watcher, and backfill.

upsert() preserves enrichment columns with COALESCE so re-indexing a session
never clobbers its summary/topics/title/etc. archive() flips a flag — rows are
never deleted, preserving history.
"""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import sbconfig
from sources.base import SessionHeader

DB_PATH = sbconfig.DB_PATH

# Bumped whenever migrate-db.py changes the schema. migrate() stamps it into
# PRAGMA user_version; connect() compares the two so a registry the running
# code is ahead of (a `git pull` before the nightly refresh) heals itself.
#   1: everything up to reasoning_path
#   2: archived_reason / archived_at
SCHEMA_VERSION = 2


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """One integer read in steady state. Migrates only when the registry is
    behind — the Stop hook and watcher upsert straight after a pull, long
    before refresh-all runs migrate-db, and the upsert SQL names new columns."""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
        return
    spec = importlib.util.spec_from_file_location(
        "migrate_db", Path(__file__).resolve().parent / "scripts" / "migrate-db.py")
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)  # type: ignore[union-attr]
    try:
        mig.migrate(conn)
    except sqlite3.OperationalError:
        # Two processes (hook + watcher) racing the same upgrade: the loser's
        # ADD COLUMN sees "duplicate column". Fine if the winner finished.
        if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            raise


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
  -- session is alive: resurrect it if it was (possibly wrongly) archived, and
  -- leave no stale reason behind.
  archived        = 0,
  archived_reason = NULL,
  archived_at     = NULL;
"""

# Closed vocabulary for sessions.archived_reason.
#   TRANSCRIPT_MISSING — the canonical transcript was deleted (Claude Code's
#       cleanupPeriodDays, a manual rm). The row is a real session: still shown
#       in the UI's Archived view and still counted in usage stats.
#   NOT_A_SESSION — the row never mapped to a real transcript (subagent
#       sidechains, workflow journals indexed before the adapter gate was
#       tightened). Hidden everywhere, counted nowhere.
TRANSCRIPT_MISSING = "transcript-missing"
NOT_A_SESSION = "not-a-session"

# SQL predicates on `sessions`. Use these instead of spelling `archived = 0`:
#   LIVE    — a transcript exists on disk right now. For anything that must
#             open the file (enrichment, reasoning extraction, prune).
#   VISIBLE — what the user should see and what usage stats should count:
#             live rows plus real sessions whose transcript aged out. Rows
#             archived without a recorded reason (mid-upgrade) stay hidden.
LIVE = "archived = 0"
ARCHIVED_VISIBLE = f"(archived = 1 AND archived_reason = '{TRANSCRIPT_MISSING}')"
VISIBLE = f"({LIVE} OR {ARCHIVED_VISIBLE})"


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


def infer_archive_reason(row) -> str:
    """Classify an archived row from its own content (for rows archived before
    the reason was recorded, and for prune-sessions, which sees both kinds).

    A row with no typed turns and no first message never held a conversation —
    that's a subagent sidechain or workflow journal, never a session. Anything
    with content was a real session whose transcript is now missing. Derived
    from what the row holds, never from the shape of its id.
    """
    def get(key):
        try:
            return row[key]
        except (KeyError, IndexError):
            return None
    turns = get("turn_count") or 0
    first = (get("first_message") or "").strip()
    if turns or first:
        return TRANSCRIPT_MISSING
    # No typed turns, but a slash-command-only session (/init, /model …) still
    # did real work: tokens, a model, a summary or a rendered trail all prove
    # a conversation happened. Only a row with none of these is sidechain noise.
    worked = any(get(k) for k in ("output_tokens", "input_tokens", "cost_usd",
                                  "model_used", "summary", "reasoning_path"))
    return TRANSCRIPT_MISSING if worked else NOT_A_SESSION


def archive(session_id: str, reason: str, conn: sqlite3.Connection | None = None) -> None:
    """Flip a row to archived=1 and record why (TRANSCRIPT_MISSING / NOT_A_SESSION
    — required, so no caller can archive without saying which). Never deletes:
    history is kept, and a later upsert from a real file resurrects the row."""
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(
            "UPDATE sessions SET archived = 1, archived_reason = ?, "
            "archived_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE session_id = ?",
            (reason, session_id),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
