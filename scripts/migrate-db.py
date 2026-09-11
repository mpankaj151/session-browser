#!/usr/bin/env python3
"""Idempotent schema migration for registry.db.

Base tables via executescript; additive columns via a PRAGMA-guarded helper
(SQLite has no ADD COLUMN IF NOT EXISTS). Safe to run any number of times.
Sets WAL + busy_timeout so the hook, watcher, Flask app, and nightly enrich can
share the DB without lock errors.

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI and one
row in `sessions`; *indexing* fills the cheap header columns from the transcript; *
enrichment* is the later LLM pass that fills title/summary/topics; a *token* is the unit
models read and write text in (roughly four characters) and is what usage is billed by.

Who runs this: install.sh, the first step of scripts/refresh-all.py, and — automatically —
indexer.connect(), which compares PRAGMA user_version against indexer.SCHEMA_VERSION and
migrates when the running code is ahead of the database on disk. That self-heal exists
because a `git pull` can land new columns hours before the nightly refresh runs, and the
Stop hook will upsert into the new schema in the meantime.

MIGRATIONS ARE ADDITIVE ONLY. Nothing here drops a column, renames one, or rewrites data
that is already there. Three reasons:
  * The registry is the only copy of everything derived (cost, summaries, reasoning paths);
    a destructive step that goes wrong has nothing to restore from.
  * Several processes share the database — the hook, the watcher, the Flask UI, the MCP
    server — and an older one may still be running against the previous shape.
  * Every step must be safe to repeat, because this runs on every connect that finds the
    registry behind. `CREATE ... IF NOT EXISTS`, the "add the column only if it is missing"
    helper, and the reason backfill's "only where it is still empty" filter all exist to
    make a second run a no-op.
Retiring a column therefore means leaving it in place and ignoring it.

PRAGMA user_version is SQLite's one built-in per-database integer, stored in the file
header. This file stamps indexer.SCHEMA_VERSION into it as the LAST step of migrate(), so
the stamp means "every statement above succeeded"; a crash halfway leaves the old number
and the next connection simply migrates again.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import indexer  # noqa: E402
import sbconfig  # noqa: E402

# The tables that have existed since the first version. Every statement is IF NOT EXISTS,
# so running this against a populated registry changes nothing. It is one string executed
# with executescript() rather than separate execute() calls simply because that is the only
# API that accepts several statements at once.
#
# `sessions` — one row per session; the spine of the whole tool. Columns, and which pass
# fills each one:
#   session_id     the CLI's own id for the conversation; primary key. Indexing pass.
#   project_path   where the transcript lives (for Claude, ~/.claude/projects/<slug>).
#   cwd            the directory the session was working in. Indexing pass.
#   folder_name    last component of cwd ("api-server") — what the UI shows as the project.
#   start_time     first timestamp in the transcript, normalised to UTC. Indexing pass.
#   last_activity  last timestamp; this is what every list is sorted by. Indexing pass.
#   first_message  the first thing the user typed; the fallback label before enrichment.
#   summary        a few sentences describing what happened. Enrichment pass only.
#   topics         JSON array of short tags, e.g. ["python","testing"]. Enrichment, or the
#                  keyword rules in scripts/classify-topics.py when there is no LLM.
#   session_type   one of feature / bugfix / refactor / planning / research. Enrichment.
#   outcome        completed / partial / abandoned. Enrichment.
#   turn_count     how many user messages carried real text. Indexing pass.
# (The remaining columns were added later — see ADDITIVE_COLUMNS below.)
#
# `session_artifacts` — many rows per session, each one extracted item: type='decision'
#   (a decision the assistant recorded, with the turn it happened in), type='journal' (the
#   rendered work-journal Markdown), type='reasoning' (one step of the thinking trail).
#   Written by scripts/enrich-sessions.py and reasoning.py; read by the UI, the MCP server,
#   the reports and the daily digest. turn_index is the position in the conversation, or
#   NULL for whole-session items.
#
# `session_checkpoints` — turn-range summaries ("turns 1-40 were about X"). Legacy: kept
#   because migrations are additive; nothing writes it today.
#
# `session_snapshots` — at most one row per session: the enrichment pass's structured
#   output (goal, decisions, artifacts, unresolved threads) that the Bridge feature turns
#   into a briefing for another CLI.
#
# `session_embeddings` — at most one row per session: the vector (a list of numbers a small
#   local model derives from text, arranged so similar meanings get similar numbers) used
#   for search-by-meaning. Written by scripts/embed-sessions.py.
#
# The two indexes on session_artifacts exist because both hot lookups are "all rows for
# this session" and "all rows of this type" — without them SQLite scans the whole table.
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
# In rough chronological order of when each was introduced; the order also matters at
# runtime, because ALTER TABLE appends columns in this sequence on a brand-new database.
ADDITIVE_COLUMNS = [
    # A short human label for the session ("Fix flaky checkout tests"). Claude Code and
    # OpenCode record one themselves, so the indexing pass can fill it; enrichment
    # overwrites it with a better one. Kept separate from `summary` so the UI can show a
    # title immediately and a summary only once it exists.
    "title TEXT",
    # Token counters. A token is roughly four characters of English, and is the unit
    # vendors bill by. All four are summed from the transcript by
    # scripts/compute-costs.py (the indexing pass leaves them NULL).
    #   input  — text sent to the model: the prompt plus the history so far.
    "input_tokens INTEGER",
    #   output — text the model generated. Costs several times more per token than input.
    "output_tokens INTEGER",
    #   cache_read  — history the vendor had already cached on their side; about a tenth
    #                 the price of a fresh input token, which is why long sessions cost far
    #                 less than their raw size suggests.
    "cache_read_tokens INTEGER",
    #   cache_write — putting that history into the cache; slightly dearer than input.
    "cache_write_tokens INTEGER",
    # The model that did most of the work, e.g. 'claude-opus-5'. Indexing pass. Used to
    # pick a pricing tier in pricing.json.
    "model_used TEXT",
    # JSON array of every model seen in the session, for sessions that switched mid-way.
    # Written by the cost pass, which has to read them all anyway.
    "models_used TEXT",
    # Which CLI the session came from: 'claude', 'copilot', 'codex', 'opencode'. Set by the
    # indexing pass from the adapter's name. NOT NULL with a default so rows written before
    # this tool supported several CLIs keep working.
    "cli_source TEXT NOT NULL DEFAULT 'claude'",
    # The CLI's own version string when the transcript records one. Indexing pass.
    "cli_version TEXT",
    # The lifecycle flag: 0 while a transcript exists for this row, 1 once it is gone.
    # Rows are never deleted, so this is how history survives a CLI's own cleanup. Set by
    # indexer.archive() (watcher, prune-sessions) and cleared by any successful upsert.
    "archived INTEGER NOT NULL DEFAULT 0",
    # Free-text notes the user types in the UI. Never touched by any automated pass.
    "notes TEXT",
    # When the SessionEnd hook last saw this session finish. Used to tell "ended and
    # settled" from "still being written to".
    "end_logged_at TIMESTAMP",
    # List-price equivalent in US dollars for the token counts above. Written by
    # scripts/compute-costs.py. Under a flat subscription nothing is actually billed per
    # session, so the UI shows this with '≈' as an intensity signal.
    "cost_usd REAL",
    # When the LLM enrichment pass last ran for this session. scripts/enrich-sessions.py
    # compares it with last_activity to skip sessions that have not changed since.
    "enriched_at TIMESTAMP",
    # Path to the readable Markdown reasoning trail under the archive's readable/ tree.
    # Written by scripts/extract-reasoning.py; the UI links to it.
    "reasoning_path TEXT",
    # Why a row is archived=1. The flag alone conflated "transcript aged out"
    # (a real session the UI should keep showing) with "never was a session".
    # Values are indexer.TRANSCRIPT_MISSING ('transcript-missing') and
    # indexer.NOT_A_SESSION ('not-a-session'); NULL only on rows archived by a build that
    # predates this column, which _backfill_archive_reason() below classifies once.
    "archived_reason TEXT",
    # When the row was archived. Purely informational, shown in the Archived tab.
    "archived_at TIMESTAMP",
]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str) -> bool:
    """Add one column unless it is already there. Returns True if it was added.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, so the check is done by hand:
    `PRAGMA table_info(<table>)` returns one row per column and field 1 of each row is the
    column name. `col_def` is a whole definition such as "cost_usd REAL" or
    "archived INTEGER NOT NULL DEFAULT 0", so the name is its first word.

    The table and definition are interpolated into the SQL rather than bound as parameters
    because SQLite only allows parameters where a VALUE is expected — never for identifiers
    or DDL. Both come from the constant list in this file, never from user input.
    """
    col = col_def.split()[0]
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col in have:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    return True


def _backfill_archive_reason(conn: sqlite3.Connection) -> int:
    """Rows archived before archived_reason existed carry NULL. Classify them
    once with the shared derived rule; a reason recorded at the source (watcher,
    prune) is authoritative and never overwritten. Idempotent by construction.

    The rule itself is indexer.infer_archive_reason(), the same function prune-sessions.py
    and the watcher use: no turns and no first message (and nothing else proving work
    happened) means the row was subagent noise, otherwise it was a real session whose
    transcript is missing. Returns how many rows were classified — 0 on every run after the
    first, which is what makes re-running free.
    """
    # The WHOLE row: infer_archive_reason weighs tokens/cost/model/summary as
    # "this session did work" — a 3-column projection hid all of them, so a
    # slash-command-only session was filed as not-a-session, permanently.
    # The filter is what makes this both additive and idempotent: only rows that are
    # archived AND still have no reason are touched, so a reason written by the watcher or
    # by prune-sessions is never overwritten, and live rows are never touched at all.
    cur = conn.execute("SELECT * FROM sessions WHERE archived = 1 AND archived_reason IS NULL")
    # This connection may not have sqlite3.Row set up (migrate() is also called with a
    # plain connection), so build dicts by hand from cursor.description — infer_archive_
    # reason() looks columns up by name.
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for row in rows:
        reason = indexer.infer_archive_reason(row)
        conn.execute("UPDATE sessions SET archived_reason = ? WHERE session_id = ?",
                     (reason, row["session_id"]))
    return len(rows)


def migrate(conn: sqlite3.Connection) -> None:
    """Bring `conn`'s database up to the current schema, then stamp the version.

    Steps in order: base tables -> additive columns -> one-off reason backfill -> indexes
    -> optional FTS5 and vector tables -> PRAGMA user_version. Commits at the end.

    Takes a connection rather than a path so callers can migrate a database they already
    hold open: indexer.connect() self-heals through here, and the tests hand it a temporary
    database. Every step is safe to repeat.
    """
    conn.executescript(BASE_DDL)
    for col_def in ADDITIVE_COLUMNS:
        _add_column_if_missing(conn, "sessions", col_def)
    _backfill_archive_reason(conn)
    # Covers the tool's most common query shape: "the newest sessions of one CLI".
    # Putting last_activity DESC in the index lets SQLite read the rows already in order
    # instead of sorting the whole table.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_source "
        "ON sessions(cli_source, last_activity DESC)"
    )
    # Full-text index over transcript turn text (best-effort: needs FTS5).
    # FTS5 is SQLite's built-in word index — it makes "find every session mentioning
    # 'webhook'" fast. It is a compile-time option, so some Python builds lack it; search
    # then degrades to the semantic/vector path rather than the whole tool failing. The
    # session_id column is UNINDEXED because it is a key to join back on, not a search term.
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts "
            "USING fts5(session_id UNINDEXED, body)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[migrate] FTS5 unavailable ({e}); full-text search disabled.", file=sys.stderr)
    # Best-effort native vec table — only if an extension-capable sqlite3 is present.
    # sqlite-vec searches vectors inside SQLite; pyenv's sqlite3 usually cannot load
    # extensions at all, so the default path is the numpy brute-force cosine in
    # semsearch.py (fast enough for thousands of sessions). FLOAT[384] matches the default
    # embedding model's vector length.
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
    # Stamp last: indexer.connect() treats this as "schema is current".
    conn.execute(f"PRAGMA user_version = {int(indexer.SCHEMA_VERSION)}")
    conn.commit()


def main() -> None:
    """Command-line entry point: open (or create) the real registry and migrate it.

    Deliberately does NOT use indexer.connect(), which would call back into migrate() —
    this is the bottom of that stack.
    """
    sbconfig.ensure_dirs()
    conn = sqlite3.connect(str(sbconfig.DB_PATH))
    # WAL ("write-ahead log") sends writes to a side file, so readers are never blocked by
    # a writer — the UI stays responsive while the nightly pipeline writes.
    conn.execute("PRAGMA journal_mode=WAL")
    # ...but there is still only ONE writer at a time. busy_timeout makes a blocked writer
    # wait up to 5 seconds instead of failing immediately with "database is locked".
    conn.execute("PRAGMA busy_timeout=5000")
    migrate(conn)
    conn.close()
    print(f"[migrate] OK -> {sbconfig.DB_PATH}")


if __name__ == "__main__":
    main()
