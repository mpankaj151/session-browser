"""Shared helpers for the session-memory MCP server.

MCP (Model Context Protocol) is a standard way for an AI assistant to call external tools.
server.py next door is such a tool server; this module holds the four things all six of
its tools need — opening the registry, redacting results, keeping a result small enough to
be worth sending, and the search that backs `search_sessions`. See docs/GLOSSARY.md for
"session", "registry", "token" and the rest of the vocabulary.

Two rules run through everything here and are worth reading before the functions:

1. **Everything is redacted on the way out.** A tool result is not shown to a human — it
   is inserted into the assistant's conversation and therefore sent to the model provider
   over the network. An API key that a session recorded years ago must not ride along, so
   both exits from this module (`sanitize` directly, `clamp` on the way in) pass the
   result through redact.py.

2. **Every result is capped.** Everything the assistant reads costs it tokens (the word
   pieces a model reads and writes; usage is billed per token) and crowds out the
   conversation itself. `clamp` trims a list to MAX_BYTES and says how much it dropped
   rather than sending everything and hoping.

Nothing here writes: the registry is opened read-write by SQLite's default, but no tool
ever issues an UPDATE.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the repo root importable so we reuse indexer / semsearch / sbconfig.
# parents[2] climbs mcp/session-memory/ -> mcp/ -> the repo root. The `# noqa: E402`
# markers below tell the linter that these imports sit after code on purpose.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402
import redact as _redact  # noqa: E402

# Which registry to serve. Normally the one indexer.py uses
# (~/.session-browser/registry.db); SESSION_MEMORY_DB overrides it, which is how the test
# suite points the server at a throwaway database and how a second registry (a copy from
# another laptop) can be served without touching config.
DB_PATH = os.environ.get("SESSION_MEMORY_DB", str(indexer.DB_PATH))
# Budget for one tool result, in bytes of serialized JSON. Small on purpose: the reply is
# spent out of the assistant's context window, and a tool that returns a wall of text
# pushes out the conversation it was meant to help with. ~2 KB is a comfortable handful of
# session descriptors — enough to pick one and ask a follow-up question about it.
MAX_BYTES = 2048


def sanitize(obj):
    """Tool results leave the machine (they're sent to the model provider), so
    every string field is redacted on the way out — including rows enriched
    before redaction-at-persist existed.

    Takes any JSON-shaped value (dict, list, string, number) and returns the same shape
    with secrets masked; redact_obj walks nested structures, so callers hand it the whole
    result rather than field by field. Used by the dict-returning tools; the
    list-returning ones get it for free through clamp().
    """
    return _redact.redact_obj(obj)


def connect():
    """Open the registry this server serves, or raise with an instruction on how to fix it.

    Returns an open sqlite3 connection with `row_factory` set, so rows are accessed by
    column name (`r["session_id"]`). Every caller closes it in a `finally`.

    Raises RuntimeError — never returns a broken connection — in the two states a fresh
    checkout can genuinely be in: no database file at all, or a file with no schema yet.
    """
    # Friendly failure pre-install: FastMCP surfaces the exception message as
    # the tool error, so make it actionable instead of a raw SQLite error.
    # The user never sees a traceback here; the assistant sees the message and repeats it,
    # so the sentence has to be the actual next step.
    if not Path(DB_PATH).exists():
        raise RuntimeError(
            "Session index not built yet — run ./install.sh (or scripts/migrate-db.py "
            "+ scripts/backfill.py) in the session-browser repo first.")
    conn = indexer.connect(DB_PATH)
    # sqlite_master is SQLite's own catalogue of what exists in the file. An empty file is
    # a perfectly valid database with no tables, so "the file is there" does not mean "the
    # schema is there" — this asks directly whether the `sessions` table exists.
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
    if row is None:
        # Close before raising: nothing else holds this handle, and leaking one per failed
        # tool call would keep a WAL file open for as long as the server runs.
        conn.close()
        raise RuntimeError(
            "Session database has no schema — run scripts/migrate-db.py first.")
    return conn


def clamp(items: list, serialize=json.dumps) -> list:
    """Trim a list so its serialized form stays under MAX_BYTES; note what was cut.
    Redacts first — every list-returning tool funnels through here.

    Returns a prefix of `items`, and — if anything was dropped — one extra entry
    `{"_truncated": True, "skipped": <n>}` so the assistant can see that there was more
    and ask for a narrower query, instead of silently concluding it saw everything.

    `serialize` is a seam for tests: it decides how "size" is measured.
    """
    # Redact BEFORE measuring as well as before returning: masking can change a string's
    # length, so measuring the raw text could let the redacted result exceed the budget.
    items = sanitize(items)
    out: list = []
    for i, item in enumerate(items):
        # Measure the whole list as it would actually be sent, not the item alone: JSON
        # brackets, commas and quoting are part of what the assistant has to read.
        trial = serialize(out + [item])
        # `and out` guarantees at least one real item is always returned. A single huge
        # session is better answered with "here it is, over budget" than with nothing but
        # a truncation marker.
        if len(trial.encode()) > MAX_BYTES and out:
            out.append({"_truncated": True, "skipped": len(items) - i})
            break
        out.append(item)
    return out


def semantic_or_keyword(query: str, limit: int):
    """Try semantic search; fall back to keyword LIKE. Returns list of session_id.

    Two-tier because the good tier is optional. *Semantic* search compares meaning: each
    session's text was turned into a list of numbers (an embedding) by a small local model,
    and the query is scored against them, so "the time the checkout tests went flaky" finds
    the session with no word in common. That needs scripts/embed-sessions.py to have run
    and the sentence-transformers library to be installed. When either is missing — or the
    query simply matches nothing — this drops to a plain substring match over the columns a
    person would search: the first message, the summary, the title and the topic tags.

    Returns a 2-tuple `(ids, scores)`: ids in best-first order, and a dict of
    id -> similarity (rounded, 1.0 = identical meaning) which is EMPTY on the keyword path,
    since a substring match has no score. search_sessions only attaches `_score` for ids
    that appear in it, so the absence is self-describing.

    Never raises for a search failure; the worst case is an empty list.
    """
    try:
        import semsearch
        # The registry THIS server serves (SESSION_MEMORY_DB), not indexer's
        # default: the two can differ, and semantic hits from another DB would
        # be looked up here and silently come back empty.
        sconn = connect()
        try:
            hits = semsearch.search(query, limit=limit, conn=sconn)
        finally:
            sconn.close()
        # No hits is not a failure — it just means nothing was similar enough. Fall
        # through to the keyword pass, which may still find a literal match.
        if hits:
            return [sid for sid, _ in hits], {sid: round(sc, 3) for sid, sc in hits}
    except Exception:  # noqa: BLE001
        # Deliberately broad, and deliberately silent: no embeddings built yet, the
        # library not installed, a stale vector dimension after the model was changed.
        # Search quietly degrading to keyword matching beats a tool that errors.
        pass
    conn = connect()
    try:
        # LIKE '%text%' is a plain case-insensitive substring match. It cannot use an
        # index, but at a few thousand rows that is irrelevant, and it needs nothing built
        # in advance. indexer.VISIBLE keeps the search to real sessions — live ones plus
        # those whose transcript aged out — and out of subagent noise rows.
        like = f"%{query}%"
        rows = conn.execute(
            f"SELECT session_id FROM sessions WHERE {indexer.VISIBLE} AND "
            "(first_message LIKE ? OR summary LIKE ? OR title LIKE ? OR topics LIKE ?) "
            "ORDER BY last_activity DESC LIMIT ?",
            (like, like, like, like, limit),
        ).fetchall()
    finally:
        conn.close()
    # No scores on this path: a substring either matched or it did not.
    return [r[0] for r in rows], {}
