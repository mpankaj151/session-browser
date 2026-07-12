#!/usr/bin/env python3
"""Journal-grade enrichment driver (nightly sweep + per-session hook tier).

Selects sessions that were never enriched OR have activity newer than their
last enrichment (resumed sessions), parses the transcript via the owning
adapter, calls the configured EnrichmentProvider, and persists:

  - facet JSON under facets/ (full structured record, incl. _meta.turns_seen)
  - sessions.summary/topics/session_type/outcome/enriched_at
  - session_snapshots (goal / decisions / artifacts / unresolved)
  - session_artifacts type='journal' (rendered markdown journal)
  - session_artifacts type='decision' rows

Re-enrichment is incremental: the provider receives the prior facet plus only
the turns added since it was written (tracked via _meta.turns_seen), so a
resumed session updates its journal without re-paying for the whole transcript.

Rate-limited; resilient to per-session failures. `--session <id>` enriches one
session (the hook's fast path); `--force` re-enriches everything from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import redact as _redact  # noqa: E402
import sbconfig  # noqa: E402
from enrichment.provider import get_provider, render_journal_markdown  # noqa: E402
from sources.base import to_iso_utc  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# Never-enriched, or touched since last enrichment. enriched_at was historically
# written by CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS' UTC) and is now written in
# the canonical 'YYYY-MM-DDTHH:MM:SS.mmmZ' form; replace() normalizes the legacy
# spelling so the lexicographic comparison is valid for both. (Migrating legacy
# one-line summaries to journal grade is a one-time `--force` run, not this
# predicate's job.)
STALE_PREDICATE = ("(summary IS NULL "
                   "OR last_activity > COALESCE(replace(enriched_at, ' ', 'T'), ''))")


def _select_sessions(conn, session_id: str | None, force: bool) -> list:
    """Which sessions this run enriches. --session is the hook fast path and
    still honors staleness (a SessionEnd with no new activity must cost $0);
    --force bypasses it either way."""
    if session_id:
        pred = "" if force else f" AND {STALE_PREDICATE}"
        return conn.execute(
            "SELECT * FROM sessions WHERE archived = 0 AND session_id = ?" + pred,
            (session_id,)).fetchall()
    sel = "SELECT * FROM sessions WHERE archived = 0"
    if not force:
        sel += f" AND {STALE_PREDICATE}"
    sel += " ORDER BY last_activity DESC"
    return conn.execute(sel).fetchall()


def _find_transcript(adapters, session) -> tuple[object, Path] | tuple[None, None]:
    src = session["cli_source"]
    adapter = adapters.get(src)
    if adapter is None:
        return None, None
    for path in adapter.discover():
        if path.stem == session["session_id"] or path.parent.name == session["session_id"]:
            return adapter, path
    return adapter, None


def _load_prior(session, force: bool) -> tuple[dict | None, int]:
    """The prior facet and its turns_seen, if this is a re-enrichment."""
    if force or not session["summary"]:
        return None, 0
    facet_path = sbconfig.FACETS_DIR / f"{session['session_id']}.json"
    if not facet_path.exists():
        return None, 0
    try:
        prior = json.loads(facet_path.read_text(encoding="utf-8"))
        seen = int(prior.get("_meta", {}).get("turns_seen", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None, 0
    if seen <= 0:
        return None, 0
    return prior, seen


def _slice_turns(turns: list, prior: dict | None, prior_seen: int) -> tuple[list, dict | None]:
    """What the provider sees. Full enrich of a long session gets head+tail (the
    tail carries the outcome); re-enrich gets only the turns added since."""
    if prior is not None:
        if prior_seen < len(turns):
            return turns[prior_seen:], prior
        # Activity advanced but no new substantive turns parsed (tool-only noise,
        # replayed history) — give the model the tail to verify/adjust cheaply.
        return turns[-30:], prior
    if len(turns) > 60:
        return turns[:20] + turns[-40:], None
    return turns, None


def _persist(conn, session_id: str, facet: dict, now_iso: str) -> None:
    topics_list = list(facet.get("goal_categories", {}).keys())
    if topics_list:
        # LLM topics are authoritative — they must override the cheap
        # keyword-classifier fallbacks written earlier in the pipeline.
        conn.execute(
            "UPDATE sessions SET summary=?, topics=?, session_type=?, "
            "outcome=?, enriched_at=? WHERE session_id=?",
            (facet["brief_summary"], json.dumps(topics_list),
             facet["session_type"], facet["outcome"], now_iso, session_id),
        )
    else:  # empty facet must not wipe existing topics
        conn.execute(
            "UPDATE sessions SET summary=?, session_type=?, outcome=?, "
            "enriched_at=? WHERE session_id=?",
            (facet["brief_summary"], facet["session_type"],
             facet["outcome"], now_iso, session_id),
        )
    # snapshot row: the structured goal/decisions/artifacts/unresolved view the
    # MCP server and UI surface (the automated equivalent of skills/snapshot)
    conn.execute(
        "INSERT INTO session_snapshots (session_id, goal, decisions, artifacts, "
        "unresolved, created_at) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET goal=excluded.goal, "
        "decisions=excluded.decisions, artifacts=excluded.artifacts, "
        "unresolved=excluded.unresolved, created_at=excluded.created_at",
        (session_id, facet.get("goal") or facet["brief_summary"],
         json.dumps(facet.get("key_decisions", [])),
         json.dumps(facet.get("files_touched", [])),
         json.dumps(facet.get("open_threads", [])), now_iso),
    )
    conn.execute("DELETE FROM session_artifacts WHERE session_id=? AND type='journal'",
                 (session_id,))
    journal = render_journal_markdown(facet)
    if journal:
        conn.execute(
            "INSERT INTO session_artifacts (session_id, type, content) "
            "VALUES (?, 'journal', ?)", (session_id, journal))
    conn.execute("DELETE FROM session_artifacts WHERE session_id=? AND type='decision'",
                 (session_id,))
    for i, dec in enumerate(facet.get("key_decisions", [])):
        conn.execute(
            "INSERT INTO session_artifacts (session_id, type, content, turn_index) "
            "VALUES (?, 'decision', ?, ?)", (session_id, str(dec)[:1000], i))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-enrich all sessions from scratch")
    ap.add_argument("--session", help="enrich exactly this session id (hook fast path)")
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
    sessions = _select_sessions(conn, args.session, args.force)
    if args.session and not sessions:
        # Fresh, unindexed, or archived — all fine outcomes for the hook path,
        # which must never surface an error into session shutdown.
        print(f"session {args.session}: up to date (or not indexed); nothing to do")
        conn.close()
        return
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{len(sessions)} sessions to enrich")

    done = 0
    consecutive_failures = 0
    for s in sessions:
        adapter, path = _find_transcript(adapters, s)
        if path is None:
            print(f"  skip {s['session_id'][:8]} (transcript not found)")
            continue
        try:
            parsed = adapter.parse_full(path)
            if parsed is None or not parsed.turns:
                continue
            prior, prior_seen = _load_prior(s, args.force)
            turns_for_llm, prior = _slice_turns(parsed.turns, prior, prior_seen)
            facet = provider.summarize(turns_for_llm, s["cli_source"],
                                       s["model_used"] or "", s["cwd"] or "",
                                       prior=prior)
            # Belt and suspenders: the prompt is already redacted, but the LLM
            # could still reconstruct a secret-looking string. Nothing derived
            # from a transcript is persisted or egressed unredacted.
            facet = _redact.redact_obj(facet)
            # turns_seen drives the next incremental slice for this session
            facet.setdefault("_meta", {})["turns_seen"] = len(parsed.turns)
            (sbconfig.FACETS_DIR / f"{s['session_id']}.json").write_text(
                json.dumps(facet, indent=2))
            now_iso = to_iso_utc(datetime.now(timezone.utc))
            _persist(conn, s["session_id"], facet, now_iso)
            conn.commit()
            done += 1
            consecutive_failures = 0
            mode = "update" if prior else "new"
            print(f"  ✓ {s['session_id'][:8]} [{s['cli_source']}/{mode}] "
                  f"{facet['brief_summary'][:70]}")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {s['session_id'][:8]}: {e}", file=sys.stderr)
            consecutive_failures += 1
            if consecutive_failures >= 5:
                # circuit breaker: exhausted quota / broken provider would
                # otherwise burn one failing LLM call per remaining session
                print("  !! 5 consecutive failures — provider likely down or "
                      "quota exhausted; aborting this run", file=sys.stderr)
                break
        time.sleep(args.rate_limit)

    conn.close()
    print(f"Enriched {done}/{len(sessions)} sessions.")


if __name__ == "__main__":
    main()
