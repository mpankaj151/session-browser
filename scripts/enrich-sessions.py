#!/usr/bin/env python3
"""Nightly enrichment driver.

Selects sessions lacking a summary (or all with --force), parses the full
transcript via the owning adapter, calls the configured EnrichmentProvider, writes
a facet JSON under facets/, and updates summary/topics/session_type/outcome.
Rate-limited; resilient to per-session failures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402
from enrichment.provider import get_provider  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402


def _find_transcript(adapters, session) -> tuple[object, Path] | tuple[None, None]:
    src = session["cli_source"]
    adapter = adapters.get(src)
    if adapter is None:
        return None, None
    for path in adapter.discover():
        if path.stem == session["session_id"] or path.parent.name == session["session_id"]:
            return adapter, path
    return adapter, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-enrich all sessions")
    ap.add_argument("--limit", type=int, default=0, help="max sessions this run (0=all)")
    ap.add_argument("--rate-limit", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()

    sbconfig.ensure_dirs()
    provider = get_provider(sbconfig.CONFIG)
    print(f"provider: {provider.name} (available={provider.is_available()})")
    if not provider.is_available():
        print("provider unavailable; aborting", file=sys.stderr)
        sys.exit(1)

    adapters = build_source_registry(only_available=True)
    conn = indexer.connect()
    sel = "SELECT * FROM sessions WHERE archived = 0"
    if not args.force:
        sel += " AND summary IS NULL"
    sel += " ORDER BY last_activity DESC"
    sessions = conn.execute(sel).fetchall()
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{len(sessions)} sessions to enrich")

    done = 0
    for s in sessions:
        adapter, path = _find_transcript(adapters, s)
        if path is None:
            print(f"  skip {s['session_id'][:8]} (transcript not found)")
            continue
        try:
            parsed = adapter.parse_full(path)
            if parsed is None or not parsed.turns:
                continue
            facet = provider.summarize(parsed.turns, s["cli_source"],
                                       s["model_used"] or "", s["cwd"] or "")
            (sbconfig.FACETS_DIR / f"{s['session_id']}.json").write_text(
                json.dumps(facet, indent=2))
            topics = json.dumps(list(facet.get("goal_categories", {}).keys()))
            conn.execute(
                "UPDATE sessions SET summary=?, topics=COALESCE(topics, ?), session_type=?, "
                "outcome=?, enriched_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (facet["brief_summary"], topics, facet["session_type"],
                 facet["outcome"], s["session_id"]),
            )
            # store key decisions as artifacts
            conn.execute("DELETE FROM session_artifacts WHERE session_id=? AND type='decision'",
                         (s["session_id"],))
            for i, dec in enumerate(facet.get("key_decisions", [])):
                conn.execute(
                    "INSERT INTO session_artifacts (session_id, type, content, turn_index) "
                    "VALUES (?, 'decision', ?, ?)", (s["session_id"], str(dec)[:1000], i))
            conn.commit()
            done += 1
            print(f"  ✓ {s['session_id'][:8]} [{s['cli_source']}] {facet['brief_summary'][:70]}")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {s['session_id'][:8]}: {e}", file=sys.stderr)
        time.sleep(args.rate_limit)

    conn.close()
    print(f"Enriched {done}/{len(sessions)} sessions.")


if __name__ == "__main__":
    main()
