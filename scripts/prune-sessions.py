#!/usr/bin/env python3
"""Archive registry rows that no live transcript maps to any more.

The predicate is adapter-agnostic and derived, never a hardcoded id pattern: build
the set of session ids every enabled adapter currently claims (discover() +
session_id_for_path()), then archive any non-archived row not in that set.

Two things produce such rows:

  * A path that used to pass an adapter's session_id_for_path() gate and no longer
    does. Multi-agent runs write sidechain transcripts to
    <project>/<session>/subagents/**, and the claude adapter accepted those until
    the gate was tightened — one permanently-empty row per subagent, plus every
    workflow journal.jsonl colliding onto a single row keyed "journal".
  * A transcript deleted while the watcher was down, so its archive-on-delete
    event was never seen.

Rows are archived (archived=1), never deleted — same invariant as indexer.archive():
history is preserved, and a later upsert from a real file resurrects the row.

Safe by default: prints a plan and changes nothing unless you pass --apply.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402


def _live_ids(registry) -> set[str]:
    """Every session id currently backed by a real transcript on disk."""
    live: set[str] = set()
    for adapter in registry.values():
        for path in adapter.discover():
            sid = adapter.session_id_for_path(path)
            if sid:
                live.add(sid)
    return live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry-run)")
    ap.add_argument("--limit", type=int, default=15, help="sample rows to print per source")
    args = ap.parse_args()

    registry = build_source_registry(only_available=True)
    if not registry:
        print("No available sources — refusing to prune (every row would look stale).")
        return

    live = _live_ids(registry)
    print(f"{len(live)} live transcript(s) across {', '.join(registry)}\n")

    conn = indexer.connect()
    try:
        rows = conn.execute(
            "SELECT session_id, cli_source, folder_name, turn_count, last_activity "
            "FROM sessions WHERE archived = 0"
        ).fetchall()

        # Only prune rows belonging to a source we can actually see right now; a
        # disabled/unavailable CLI must never have its history archived wholesale.
        stale = [r for r in rows if r["cli_source"] in registry and r["session_id"] not in live]
        if not stale:
            print(f"✓ All {len(rows)} active row(s) are backed by a live transcript.")
            return

        by_source = collections.Counter(r["cli_source"] for r in stale)
        print(f"{len(stale)} of {len(rows)} active row(s) have no live transcript "
              f"({'APPLYING' if args.apply else 'dry-run — pass --apply to act'}):\n")
        for src, n in by_source.most_common():
            print(f"  [{src}] {n}")
            sample = [r for r in stale if r["cli_source"] == src][: args.limit]
            for r in sample:
                print(f"      {r['session_id'][:40]:42} {str(r['folder_name'] or '-')[:28]:30} "
                      f"turns={r['turn_count'] or 0:<4} {r['last_activity'] or '-'}")
            if n > len(sample):
                print(f"      ... and {n - len(sample)} more")
            print()

        if args.apply:
            for r in stale:
                indexer.archive(r["session_id"], conn=conn)
            conn.commit()
            print(f"Done. Archived {len(stale)} row(s) (archived=1 — nothing deleted).")
        else:
            print("Nothing changed. Re-run with --apply to archive these rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
