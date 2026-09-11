#!/usr/bin/env python3
"""Build/refresh the full-text index over transcript turn text.

**FTS5** is SQLite's built-in full-text search: a virtual table that keeps a word index
over the text you insert, so `WHERE sessions_fts MATCH 'flaky checkout'` finds the rows
containing those words in milliseconds instead of scanning every transcript on disk. It
is exact-word matching — grep with an index — which is why it sits alongside, not
instead of, the meaning-based search in semsearch.py.

For each session, parse the full transcript via its adapter, join the turn text,
and (re)insert into the sessions_fts FTS5 table. Lets you search by what was
actually discussed, not just the summary/title. Re-runnable per session.

What gets indexed: every turn's role and text ("user: why is this failing", "assistant:
because the fixture…"), plus the one-line summary of each tool call's input, so searching
for a filename or a command finds the session that touched it. Tool OUTPUT is not indexed
— it is the bulk of a transcript and mostly noise.

Redaction happens BEFORE indexing, in _body(). The index is a queryable copy of the
transcript text and it is read back by the UI and the MCP server, so it is an egress
point like any other: an API key that scrolled past in a session must not become
searchable (docs/ARCHITECTURE.md, "Redaction at every egress").

Two passes, and the second is the interesting one: the main loop indexes sessions whose
transcript still exists, then index_archived() re-indexes aged-out sessions from the
byte-identical copy in the reasoning archive — so full-text search keeps working for
sessions whose CLI already deleted the original.

Pipeline position: run by the nightly `refresh-all`. Reads transcripts and the raw
archive; writes only the sessions_fts table. Exits 1 only when this Python's SQLite has
no FTS5 support at all (nothing else can be done); individual bad files are reported and
skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts run directly, not as a package: put the repo root on the import path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import redact as _redact  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# 200 000 characters ≈ the first few hundred turns of a long session. Beyond that the
# index grows faster than its usefulness: the words that identify a session almost always
# appear early, and one enormous session should not dominate the whole table.
MAX_BODY = 200_000  # cap per session to keep the index lean


def _body(parsed) -> str:
    """The searchable text for one parsed session: turns, then tool-call inputs.

    Each turn contributes "<role>: <text>" so a search can tell who said what; each tool
    call contributes its already-summarised input line (a command, a file path) so
    searching for `conftest.py` finds the session that edited it.

    Truncated to MAX_BODY characters and then redacted — in that order, deliberately:
    redacting the full megabyte and throwing most of it away would be wasted work, and
    truncation cannot expose a secret that redaction would have caught, because
    redaction runs last over exactly the text that will be stored.
    """
    parts = []
    for t in parsed.turns:
        if t.content:
            parts.append(f"{t.role}: {t.content}")
        for tc in t.tool_calls:
            if tc.get("input"):
                parts.append(tc["input"])
    text = "\n".join(parts)
    return _redact.redact(text[:MAX_BODY])


def index_archived(conn, registry) -> int:
    """Aged-out sessions have no live transcript for discover() to find — but
    the raw copy refresh-all put in the reasoning archive is byte-identical.
    Index their body from there so full-text search keeps finding them.

    The raw copy is parsed by the SAME adapter as a live transcript — it is the original
    file's bytes, filename and all, so nothing special is needed to read it. Returns how
    many archived sessions were indexed; commits before returning.

    A row is skipped when the vault has no copy of it (raw.get is None) or when this
    machine has no adapter for its CLI (registry.get is None) — both are ordinary states
    on a laptop that only has some of the CLIs installed.
    """
    # One walk of the vault for all archived rows, not one per row.
    raw = reasoning.archived_raw_index()
    # indexer.ARCHIVED_VISIBLE is the shared predicate for "archived because the
    # transcript went missing" — real conversations the user lost, excluding the
    # subagent-noise rows, which have no transcript worth indexing.
    rows = conn.execute(
        f"SELECT session_id, cli_source FROM sessions WHERE {indexer.ARCHIVED_VISIBLE}"
    ).fetchall()
    n = 0
    for sid, source in rows:
        path, adapter = raw.get(sid), registry.get(source)
        if path is None or adapter is None:
            continue
        try:
            parsed = adapter.parse_full(path)
            if parsed is None or not parsed.turns:
                continue
            # Delete-then-insert, because an FTS5 table has no upsert: without the delete
            # a re-run would leave two copies of the session and double every hit.
            conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
            conn.execute("INSERT INTO sessions_fts (session_id, body) VALUES (?, ?)",
                         (sid, _body(parsed)))
            n += 1
        except Exception as e:  # noqa: BLE001
            # One unparseable archive copy is reported and skipped; the rest still index.
            print(f"  ! {path.name}: {e}", file=sys.stderr)
    conn.commit()
    return n


def main() -> None:
    """Index every live transcript, then every archived one, and report the counts.

    --source NAME  limit both passes to one CLI (claude|copilot|codex|opencode)
    --rebuild      clear the index first — for --source, only that CLI's rows

    Exits 1 only when FTS5 itself is unavailable in this Python's SQLite build.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole index")
    args = ap.parse_args()

    conn = indexer.connect()
    try:
        # A "virtual table" is SQLite's plug-in table type; fts5(...) declares the
        # columns of the word index. session_id is UNINDEXED — it is stored so results
        # can be joined back to the sessions table, but its characters are not added to
        # the word index, where a uuid would only be noise.
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(session_id UNINDEXED, body)")
    except Exception as e:  # noqa: BLE001
        # Some Python builds ship a SQLite compiled without FTS5. Nothing here can work
        # in that case, and the UI falls back to keyword matching, so exit loudly once.
        print(f"FTS5 unavailable: {e}", file=sys.stderr)
        sys.exit(1)
    if args.rebuild:
        if args.source:   # only THIS source's rows — never the other CLIs' full-text
            conn.execute("DELETE FROM sessions_fts WHERE session_id IN "
                         "(SELECT session_id FROM sessions WHERE cli_source = ?)", (args.source,))
        else:
            conn.execute("DELETE FROM sessions_fts")

    # only_available=True for the live pass: a CLI with no transcript directory on this
    # machine has nothing to discover.
    registry = build_source_registry(only_available=True)
    if args.source:
        registry = {k: v for k, v in registry.items() if k == args.source}

    n = 0
    for name, adapter in registry.items():
        files = list(adapter.discover())
        print(f"[{name}] {len(files)} files")
        for i, path in enumerate(files, 1):
            try:
                header = adapter.parse_header(path)
                if header is None:
                    continue
                parsed = adapter.parse_full(path)
                if parsed is None or not parsed.turns:
                    continue
                sid = header.session_id
                # Delete-then-insert: FTS5 has no upsert, and a plain insert on a re-run
                # would leave the session in the index twice.
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
                conn.execute("INSERT INTO sessions_fts (session_id, body) VALUES (?, ?)",
                             (sid, _body(parsed)))
                n += 1
            except Exception as e:  # noqa: BLE001
                # Report and continue: one bad transcript must not abort the sweep.
                print(f"  ! {path.name}: {e}", file=sys.stderr)
            # Commit every 20 files so the write lock is released regularly — the Stop
            # hook may be trying to upsert a just-finished session at the same moment.
            if i % 20 == 0:
                conn.commit()
        conn.commit()
    # Archived rows are read from the vault copy, so the CLI's transcript tree
    # being gone (the very state the archive exists for) must not drop them.
    # Hence a SECOND registry built WITHOUT only_available: the archived pass needs the
    # adapter only to parse a file we already hold, so "this CLI is not installed here"
    # is irrelevant — and filtering on it would hide exactly the sessions the archive was
    # built to preserve.
    archived_registry = build_source_registry()
    if args.source:
        archived_registry = {k: v for k, v in archived_registry.items() if k == args.source}
    m = index_archived(conn, archived_registry)
    conn.close()
    print(f"Indexed full text for {n} sessions (+{m} archived, from the raw archive).")


if __name__ == "__main__":
    main()
