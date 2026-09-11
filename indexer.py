"""Thin DB layer shared by the hook, watcher, and backfill.

upsert() preserves enrichment columns with COALESCE so re-indexing a session
never clobbers its summary/topics/title/etc. archive() flips a flag — rows are
never deleted, preserving history.

Where this sits in the pipeline (docs/ARCHITECTURE.md has the picture): the source
adapters under sources/ read a *transcript* — the file a coding CLI writes for one
conversation, one *session* — and hand back a SessionHeader of cheap facts. Everything
that produces a header (the Claude Stop hook, the filesystem watcher, backfill.py,
reconcile) funnels through upsert() here, and everything that notices a transcript has
gone funnels through archive(). This module is the only place that writes the `sessions`
table's *indexed* columns. The derived ones — summary, topics, cost, reasoning_path,
embeddings — are written by the nightly enrichment scripts and must survive re-indexing;
that is the whole point of the COALESCE upsert below. Terms are defined in
docs/GLOSSARY.md.

Two rules a reader should take away:

1. Indexing is idempotent and monotonic. Running the hook, the watcher and a backfill
   over the same transcript in any order must converge to the same row, and a re-parse
   that happens to see *less* of a transcript than a previous one (a partially synced
   file, a second Codex rollout for the same id) must never walk the row backwards.

2. Rows are never deleted. When a transcript disappears — Claude Code's own
   `cleanupPeriodDays` cleanup deletes transcripts after 30 days — the row is flipped to
   archived with a recorded reason, so the history, the token totals and the ability to
   restore the session all survive. Consumers never spell the flag themselves; they
   compose the LIVE / VISIBLE / ARCHIVED_VISIBLE predicates defined here (a test greps
   the tree to enforce that).

Connections: every function takes an optional `conn`. Passed one, it neither commits nor
closes — the caller owns the transaction, which is how backfill batches hundreds of rows
and how the tests run against a temporary database. Given none, it opens, commits and
closes its own.
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
    """Open the registry, ready to use, and make sure its schema is current.

    Every reader and writer in the project goes through here, so the three settings below
    are applied exactly once and in one place:

      * row_factory = sqlite3.Row — rows behave like dicts (`row["session_id"]`), which
        is what callers such as infer_archive_reason() and the UI expect.
      * WAL (write-ahead logging): readers are never blocked by a writer. The UI can page
        through sessions while the watcher indexes, and the nightly jobs can overlap.
      * busy_timeout 5000 — if another process does hold the write lock, wait up to five
        seconds instead of failing instantly with "database is locked".

    `db_path` defaults to the configured registry; the tests and `sb demo` pass a
    temporary file. Side effect: creates the file if missing and may run migrations
    (see _ensure_schema). Errors propagate — a registry we cannot open is fatal.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """One integer read in steady state. Migrates only when the registry is
    behind — the Stop hook and watcher upsert straight after a pull, long
    before refresh-all runs migrate-db, and the upsert SQL names new columns.

    In other words this is the self-heal: `git pull` can bring code whose SQL mentions a
    column the user's database does not have yet, and the very next session end would
    fail. Rather than making every caller remember to migrate, the check rides along on
    every connect() and costs a single PRAGMA read once the database is current.

    scripts/migrate-db.py is loaded by file path (its name has a hyphen, so it cannot be
    imported normally) and its migrate() both applies the changes and stamps the new
    PRAGMA user_version.
    """
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


# The one statement that writes an indexed session row. Read it as: "insert this
# session; if a row with that id already exists, merge the new facts into it".
#
# `excluded` is SQLite's name for the row we tried to insert, i.e. the freshly parsed
# values; `sessions.<col>` is what is already stored. Three different merge rules are
# used below, and which one a column gets is a deliberate choice:
#
#   MAX(old, new)                       — monotonic columns that must never shrink.
#   COALESCE(NULLIF(old,''), new)       — first non-empty value wins and then sticks.
#   COALESCE(NULLIF(new,''), old)       — newest non-empty value wins (title, model).
#
# NULLIF(x, '') turns an empty string into NULL so COALESCE skips over it: adapters emit
# '' for "not known yet", and an empty string must neither stick nor overwrite a real
# value. MAX() would return NULL if either side were NULL, hence the COALESCE inside it.
#
# Columns this statement does not name — summary, topics beyond the first parse, cost,
# token counts, reasoning_path, embeddings — are written only by the enrichment scripts
# and are therefore untouched by any amount of re-indexing. That is the invariant the
# whole tool depends on: re-reading a transcript is always free of consequence.
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
#   ARCHIVED_VISIBLE — exactly the Archived tab: real sessions whose transcript
#             aged out, and nothing else. Sidechain noise never appears.
#
# These are SQL fragments, not values, so callers interpolate them into their own
# queries: f"SELECT ... FROM sessions WHERE {indexer.VISIBLE} ORDER BY last_activity".
# Safe to interpolate because they are built here from module constants, never from
# user input. Spelling the flag out by hand anywhere else is what the grep test forbids —
# it is how the Archived tab and the usage stats drifted apart in the first place.
LIVE = "archived = 0"
ARCHIVED_VISIBLE = f"(archived = 1 AND archived_reason = '{TRANSCRIPT_MISSING}')"
VISIBLE = f"({LIVE} OR {ARCHIVED_VISIBLE})"


def _params(h: SessionHeader) -> dict:
    """Flatten a SessionHeader into the named parameters _UPSERT_SQL binds.

    Written out field by field rather than via dataclasses.asdict() on purpose: `metadata`
    (a per-source dict of extras) has no column, and a future header field must not
    silently start or stop being persisted just because the dataclass changed.
    """
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
    """Insert or merge one parsed session into the registry (see _UPSERT_SQL).

    This is THE way a session gets into the database — the hook, the watcher and the
    backfill all end here. Calling it repeatedly for the same session is expected and
    cheap; derived/enrichment columns are never touched.

    Side effect beyond the row itself: because an upsert can only come from a transcript
    that exists on disk, it also clears the archived flag and reason, resurrecting a row
    that was archived earlier (a restored session, or one archived by mistake).

    `conn`: pass one to join the caller's transaction — nothing is committed or closed,
    which is how backfill writes hundreds of rows in a single burst and how the tests
    work against a temporary database. Omit it and this opens, commits and closes its own
    connection. Errors are not caught: a failed write must be visible to the caller.
    """
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

    `row` is anything subscriptable by column name — a sqlite3.Row from connect(), or a
    plain dict in the tests. Returns one of the two reason constants; never raises, never
    writes. Both callers (the watcher's delete handler and scripts/prune-sessions.py) use
    it so that WHICH process happens to notice a deletion cannot change whether the
    session stays browsable — they used to disagree, and sessions vanished accordingly.
    """
    def get(key):
        """Read one column, tolerating rows that simply do not have it.

        A sqlite3.Row raises IndexError for an unknown name and a dict raises KeyError;
        prune passes a narrow SELECT while the watcher passes `SELECT *`, so missing
        columns are normal here and mean "no evidence", not an error.
        """
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
    history is kept, and a later upsert from a real file resurrects the row.

    `reason` is positional and has no default deliberately: the two kinds of archived row
    are indistinguishable afterwards, and an Archived view built on the flag alone drowns
    real sessions under subagent noise. `archived_at` is stamped by SQLite itself in the
    same canonical UTC spelling the adapters emit, so it sorts against the other
    timestamp columns.

    Archiving an id that is not in the registry is a silent no-op (UPDATE matching no
    rows), which is what the watcher wants when a file it never indexed is deleted.
    `conn` follows the same borrow-or-own rule as upsert()."""
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
