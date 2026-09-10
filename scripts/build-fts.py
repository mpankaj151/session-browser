#!/usr/bin/env python3
"""Build/refresh the full-text index over transcript turn text.

For each session, parse the full transcript via its adapter, join the turn text,
and (re)insert into the sessions_fts FTS5 table. Lets you search by what was
actually discussed, not just the summary/title. Re-runnable per session.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import redact as _redact  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

MAX_BODY = 200_000  # cap per session to keep the index lean


def _body(parsed) -> str:
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
    Index their body from there so full-text search keeps finding them."""
    raw = reasoning.archived_raw_index()
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
            conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
            conn.execute("INSERT INTO sessions_fts (session_id, body) VALUES (?, ?)",
                         (sid, _body(parsed)))
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {path.name}: {e}", file=sys.stderr)
    conn.commit()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole index")
    args = ap.parse_args()

    conn = indexer.connect()
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(session_id UNINDEXED, body)")
    except Exception as e:  # noqa: BLE001
        print(f"FTS5 unavailable: {e}", file=sys.stderr)
        sys.exit(1)
    if args.rebuild:
        conn.execute("DELETE FROM sessions_fts")

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
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
                conn.execute("INSERT INTO sessions_fts (session_id, body) VALUES (?, ?)",
                             (sid, _body(parsed)))
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ! {path.name}: {e}", file=sys.stderr)
            if i % 20 == 0:
                conn.commit()
        conn.commit()
    m = index_archived(conn, registry)
    conn.close()
    print(f"Indexed full text for {n} sessions (+{m} archived, from the raw archive).")


if __name__ == "__main__":
    main()
