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

For a reader new to AI tooling (docs/GLOSSARY.md defines the rest): a **model**
is the AI system that writes text, a **prompt** is what we send it, and **tokens**
are the word-pieces it reads and writes — every vendor bills per token.
**Enrichment** is the only part of Session Browser that calls a model at all, and
it does so by running a coding CLI the user already has installed and is already
paying for (a **headless run**: the CLI driven non-interactively, prompt in,
answer out). That means every session this script processes spends a slice of the
user's own quota, which is why scripts/refresh-all.py only runs it behind an
explicit `--enrich` flag and why so much of the code below exists to avoid paying
twice for the same turns.

Two entry points:

  * The nightly sweep (`refresh-all.py --enrich`), which walks every stale session.
  * The per-session hook tier: Claude Code's SessionEnd hook fires
    `enrich-sessions.py --session <id>` the moment a conversation closes, so the
    journal is ready before the user asks for it.

Exit codes matter to both. 0 means the run is healthy — including the case where
no summariser CLI is installed at all, which is a configuration state rather than
a failure. 1 means at least one session failed, so a broken credential or an
exhausted quota shows up as a red nightly run instead of a silent night with zero
summaries.

Everything here is idempotent: re-running never duplicates a row (snapshots
upsert, journal and decision artifacts are deleted and rewritten) and never
re-pays for turns already described.
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

# SQL fragment, composed into the WHERE clause of _select_sessions below.
# Reads as: "this session has no summary yet, OR it has been used since the last
# time we summarised it" — the second half is what catches a RESUMED session, one
# the user reopened and added turns to after it was already journaled.
# Never-enriched, or touched since last enrichment. enriched_at was historically
# written by CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS' UTC) and is now written in
# the canonical 'YYYY-MM-DDTHH:MM:SS.mmmZ' form; replace() normalizes the legacy
# spelling so the lexicographic comparison is valid for both. (Migrating legacy
# one-line summaries to journal grade is a one-time `--force` run, not this
# predicate's job.)
STALE_PREDICATE = ("(TRIM(COALESCE(summary, '')) = '' "
                   "OR last_activity > COALESCE(replace(enriched_at, ' ', 'T'), ''))")

# Both comparisons are LEXICOGRAPHIC — plain string ordering, which only works
# because every timestamp in the registry is normalised to the same spelling by
# sources.base.to_iso_utc ('2026-06-01T09:30:00.000Z'). COALESCE(..., '') makes a
# NULL enriched_at sort below every real timestamp, so a row that was never
# enriched matches; TRIM(COALESCE(summary, '')) = '' catches a facet whose
# brief_summary came back empty, which would otherwise look enriched forever.

# SQL fragment: does this session contain anything worth describing? Three ways of
# saying yes, because no single column covers them all:
#   turn_count > 0        the user typed at least one message
#   first_message <> ''   there is an opening message even if turns were not counted
#   output_tokens > 0     the model produced something, so work happened
# "Nothing to summarise" in the indexer's own vocabulary (infer_archive_reason):
# typed turns, a first message, or output tokens. turn_count alone counts only
# TYPED user turns — a /init-only session has none yet did real work.
# (A slash command such as `/init` is a shortcut the CLI expands into a full
# prompt; it never appears as a typed user turn, but the session it produces is
# real work and must be summarised. Covered by the journal suite.)
HAS_CONTENT = ("(turn_count > 0 OR TRIM(COALESCE(first_message, '')) <> '' "
               "OR COALESCE(output_tokens, 0) > 0)")


def _select_sessions(conn, session_id: str | None, force: bool) -> list:
    """Which sessions this run enriches. --session is the hook fast path and
    still honors staleness (a SessionEnd with no new activity must cost $0);
    --force bypasses it either way.

    The query is assembled from three parts: indexer.LIVE (only sessions whose
    transcript still exists on disk — there is nothing to read for the others),
    HAS_CONTENT and, unless `force`, STALE_PREDICATE. Newest first, so a run cut
    short by --limit or by the circuit breaker has done the most useful work.

    `conn` is an open registry connection; tests pass a temporary one. Returns a
    list of sqlite3.Row, possibly empty. Read-only.
    """
    # turn_count > 0: a session opened and closed at once has nothing to
    # summarise; selecting it every night and skipping it in the loop made the
    # log read "N sessions to enrich ... Enriched 0/N" with no reason, forever.
    base = f"SELECT * FROM sessions WHERE {indexer.LIVE} AND {HAS_CONTENT}"
    if session_id:
        pred = "" if force else f" AND {STALE_PREDICATE}"
        return conn.execute(base + " AND session_id = ?" + pred, (session_id,)).fetchall()
    sel = base
    if not force:
        sel += f" AND {STALE_PREDICATE}"
    sel += " ORDER BY last_activity DESC"
    return conn.execute(sel).fetchall()


def _find_transcript(adapters, session) -> tuple[object, Path] | tuple[None, None]:
    """Locate the file holding this session's conversation.

    `adapters` maps a CLI name ("claude", "codex", ...) to the sources/*.py class
    that understands that CLI's on-disk layout. The registry stores which CLI a
    session came from but not a stable path, so the owning adapter is asked to
    list its transcripts and each one is matched back to the session id.

    Returns (adapter, path). (None, None) when no adapter for that CLI is
    available on this machine — installing only some CLIs is expected, see
    docs/ARCHITECTURE.md — and (adapter, None) when the CLI is present but the
    file is gone (deleted between indexing and now). The caller skips both.
    Read-only: it only globs directories.
    """
    src = session["cli_source"]
    adapter = adapters.get(src)
    if adapter is None:
        return None, None
    sid = session["session_id"]
    for path in adapter.discover():
        # The adapter owns filename → session-id mapping (codex names files
        # rollout-<ts>-<uuid>.jsonl, so a bare stem match never hits them).
        try:
            mapped = adapter.session_id_for_path(path)
        except Exception:  # noqa: BLE001 — fall back to path heuristics
            # Swallowing is right here: a single unreadable or malformed file must
            # not abort the search for a session that may be the next path along.
            mapped = None
        # Three chances, cheapest-correct first: the adapter's own mapping, then
        # two layouts it covers anyway — a file named <session-id>.jsonl (Claude,
        # OpenCode's mirror) and a directory named after the session (Copilot's
        # `<session-id>/events.jsonl`).
        if mapped == sid or path.stem == sid or path.parent.name == sid:
            return adapter, path
    return adapter, None


def _load_prior(session, force: bool) -> tuple[dict | None, int]:
    """The prior facet and its turns_seen, if this is a re-enrichment.

    Reads `~/.session-browser/facets/<session-id>.json`, written by _write_facet
    on the previous run. `turns_seen` is how many turns that facet described; it
    is the pointer that makes re-enrichment incremental — everything after it is
    what the model has not been shown yet.

    Returns (None, 0) — meaning "treat this as a first enrichment" — for every
    reason the incremental path cannot be trusted: --force, no summary on the row,
    no facet file, unreadable or truncated JSON, or a missing/zero turns_seen
    (facets written before that key existed). Falling back costs one full
    enrichment; getting it wrong would silently drop turns from the journal.
    Never raises.
    """
    if force or not session["summary"]:
        return None, 0
    facet_path = sbconfig.FACETS_DIR / f"{session['session_id']}.json"
    if not facet_path.exists():
        return None, 0
    try:
        prior = json.loads(facet_path.read_text(encoding="utf-8"))
        seen = int(prior.get("_meta", {}).get("turns_seen", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        # Truncated file, unreadable file, or turns_seen that is not a number.
        return None, 0
    if seen <= 0:
        return None, 0
    return prior, seen


def _slice_turns(turns: list, prior: dict | None, prior_seen: int) -> tuple[list, dict | None]:
    """What the provider sees. Full enrich of a long session gets head+tail (the
    tail carries the outcome); re-enrich gets only the turns added since.

    Prompts are billed by length, so this is the cost-control seam. Three cases:

      first enrichment, <= 60 turns   everything
      first enrichment, longer        first 20 + last 40. The head holds the goal
                                      ("I need to fix the flaky checkout tests")
                                      and the tail holds how it ended; the middle
                                      is mostly tool calls and is the cheapest
                                      thing to drop.
      re-enrichment                   only the turns after prior_seen, bounded the
                                      same way, plus the prior facet so the model
                                      updates the existing entry.

    Returns (turns to send, prior facet or None). The second element is what the
    caller passes to the provider; it comes back from here rather than being
    reused directly because this function may decide the prior is unusable.
    """
    if prior is not None:
        if prior_seen < len(turns):
            new = turns[prior_seen:]
            # Bounded like the full slice: render_prompt caps at 60, and an
            # unbounded tail let it drop the OUTCOME of a long resumed session
            # while turns_seen then marked every dropped turn as seen.
            return (new[:20] + new[-40:] if len(new) > 60 else new), prior
        # Activity advanced but no new substantive turns parsed (tool-only noise,
        # replayed history) — give the model the tail to verify/adjust cheaply.
        return turns[-30:], prior
    if len(turns) > 60:
        return turns[:20] + turns[-40:], None
    return turns, None


def _write_facet(path: Path, facet: dict) -> None:
    """Atomic: a nightly killed mid-write left truncated JSON that _load_prior
    swallowed, silently re-paying for a full enrichment.

    "Atomic" here means a reader sees either the old file or the complete new one,
    never a half-written mixture. The trick is to write a temporary file in the
    SAME directory (os.replace is only atomic within one filesystem) and then
    rename it over the target — rename is a single filesystem operation.

    Side effect: writes `~/.session-browser/facets/<session-id>.json`. On any
    failure the temporary file is removed and the exception re-raised, so a failed
    write never leaves litter next to the real facets.
    """
    # Imported here rather than at module scope: these are needed on the one write
    # path only, and the module is also imported by tests purely for its helpers.
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(facet, indent=2))
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a Ctrl-C or a SIGTERM from launchd during
        # the nightly run is exactly the case this cleanup exists for.
        try:
            os.unlink(tmp)
        except OSError:
            # Already gone, or the directory is unwritable. Either way there is
            # nothing useful left to do, and the original error is the real news.
            pass
        raise


def _persist(conn, session_id: str, facet: dict, now_iso: str) -> None:
    """Write one facet into the registry. Does NOT commit — the caller does.

    Four destinations, because different consumers need different shapes:
      sessions             the columns the UI list and search read directly
      session_snapshots    one structured row per session (goal / decisions /
                           artifacts / unresolved), read by the MCP server and UI
      session_artifacts 'journal'   the rendered markdown entry, read by
                                    scripts/daily-digest.py
      session_artifacts 'decision'  one row per key decision, so decisions are
                                    searchable and orderable on their own

    Idempotent by construction: the snapshot upserts, and both artifact kinds are
    deleted before being re-inserted, so re-enriching a session replaces its
    records instead of accumulating duplicates.

    `now_iso` is the canonical UTC timestamp for this run, stored as enriched_at —
    which is exactly what STALE_PREDICATE compares against next time.
    """
    topics_list = list(facet.get("goal_categories", {}).keys())
    if topics_list:
        # LLM topics are authoritative — they must override the cheap
        # keyword-classifier fallbacks written earlier in the pipeline.
        # (scripts/classify-topics.py runs first in the nightly order and tags
        # sessions from keywords alone; a model that read the transcript knows
        # better, so this UPDATE sets topics outright rather than COALESCEing.)
        conn.execute(
            "UPDATE sessions SET summary=?, topics=?, session_type=?, "
            "outcome=?, enriched_at=? WHERE session_id=?",
            (facet["brief_summary"], json.dumps(topics_list),
             facet["session_type"], facet["outcome"], now_iso, session_id),
        )
    else:  # empty facet must not wipe existing topics
        # Same UPDATE minus the topics column: the model returned no categories,
        # so whatever the keyword classifier found stays put.
        conn.execute(
            "UPDATE sessions SET summary=?, session_type=?, outcome=?, "
            "enriched_at=? WHERE session_id=?",
            (facet["brief_summary"], facet["session_type"],
             facet["outcome"], now_iso, session_id),
        )
    # snapshot row: the structured goal/decisions/artifacts/unresolved view the
    # MCP server and UI surface (the automated equivalent of skills/snapshot)
    # ON CONFLICT(session_id) DO UPDATE = "insert, or overwrite the existing row":
    # exactly one snapshot per session, no matter how often it is re-enriched.
    # The list columns are stored as JSON text, SQLite having no array type.
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
    # Delete-then-insert rather than an upsert: there is no unique key on
    # (session_id, type) for artifacts — a session has many decision rows — so
    # clearing first is what keeps re-enrichment from stacking duplicates.
    conn.execute("DELETE FROM session_artifacts WHERE session_id=? AND type='journal'",
                 (session_id,))
    journal = render_journal_markdown(facet)
    # Empty journal -> no row at all, so the digest never renders an empty shell.
    if journal:
        conn.execute(
            "INSERT INTO session_artifacts (session_id, type, content) "
            "VALUES (?, 'journal', ?)", (session_id, journal))
    conn.execute("DELETE FROM session_artifacts WHERE session_id=? AND type='decision'",
                 (session_id,))
    # turn_index carries the decision's position in the list, not a real turn
    # number — it exists so the decisions come back in the order the model wrote
    # them (see _artifacts in scripts/report-data.py, which ORDER BYs it).
    # 1000 characters is a sanity cap on a runaway "decision".
    for i, dec in enumerate(facet.get("key_decisions", [])):
        conn.execute(
            "INSERT INTO session_artifacts (session_id, type, content, turn_index) "
            "VALUES (?, 'decision', ?, ?)", (session_id, str(dec)[:1000], i))


def main() -> None:
    """Run one enrichment pass. The only function here with side effects at scale.

    Sequence: resolve the provider -> select sessions -> for each, find and parse
    the transcript, call the model, redact, write the facet file, write the DB
    rows, commit, pause.

    Exit codes (both the nightly job and the SessionEnd hook depend on them):
      0  everything enriched, or nothing needed enriching, or no summariser CLI is
         installed on this machine at all (a configuration state, not a failure —
         `refresh-all` stays green on a laptop with no coding CLI)
      1  a provider was configured but cannot run, or at least one session failed

    Failure handling is per session: one bad transcript must not cost the other
    two hundred their summaries. But five failures in a row abort the run — see
    the circuit breaker below.
    """
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
        if provider.name == "auto":
            # No summariser CLI on this machine at all: a configuration state,
            # not a failed run — say so once and let refresh-all stay green.
            print(f"enrichment skipped: {provider.reason}")
            return
        # The other branch: a provider was named in config.toml but its binary is
        # missing. The user asked for summaries and is not getting any, so this is
        # a hard failure rather than a quiet skip.
        print(f"provider {provider.name} unavailable (binary "
              f"{getattr(provider, 'binary', '?')!r} not on PATH); aborting", file=sys.stderr)
        sys.exit(1)

    # only_available=True: adapters for CLIs with no transcripts on this machine
    # are left out entirely, so _find_transcript never walks an empty tree.
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

    # done / skipped / failed are reported in the closing line and, for `failed`,
    # decide the exit code. consecutive_failures drives the circuit breaker: it is
    # reset to 0 after every success, so only an unbroken run of failures trips it.
    done = failed = skipped = 0
    consecutive_failures = 0
    for s in sessions:
        adapter, path = _find_transcript(adapters, s)
        if path is None:
            print(f"  skip {s['session_id'][:8]} (transcript not found)")
            skipped += 1
            continue
        try:
            parsed = adapter.parse_full(path)
            if parsed is None or not parsed.turns:
                print(f"  skip {s['session_id'][:8]} (no turns in transcript)")
                skipped += 1
                continue
            prior, prior_seen = _load_prior(s, args.force)
            turns_for_llm, prior = _slice_turns(parsed.turns, prior, prior_seen)
            # The single model call. Everything above decided what to send;
            # everything below stores what came back. This is the only line in
            # Session Browser that spends the user's CLI quota.
            facet = provider.summarize(turns_for_llm, s["cli_source"],
                                       s["model_used"] or "", s["cwd"] or "",
                                       prior=prior)
            # Belt and suspenders: the prompt is already redacted, but the LLM
            # could still reconstruct a secret-looking string. Nothing derived
            # from a transcript is persisted or egressed unredacted.
            facet = _redact.redact_obj(facet)
            # turns_seen drives the next incremental slice for this session
            # len(parsed.turns), NOT len(turns_for_llm): it records how far along
            # the transcript this journal is current, including the middle turns
            # _slice_turns dropped. A resumed session then continues from the true
            # end rather than re-sending turns that were never worth sending.
            facet.setdefault("_meta", {})["turns_seen"] = len(parsed.turns)
            _write_facet(sbconfig.FACETS_DIR / f"{s['session_id']}.json", facet)
            now_iso = to_iso_utc(datetime.now(timezone.utc))
            _persist(conn, s["session_id"], facet, now_iso)
            conn.commit()
            done += 1
            consecutive_failures = 0
            mode = "update" if prior else "new"
            print(f"  ✓ {s['session_id'][:8]} [{s['cli_source']}/{mode}] "
                  f"{facet['brief_summary'][:70]}")
        except Exception as e:  # noqa: BLE001
            # Deliberately broad: an unreadable transcript, a provider error, a
            # malformed facet and a DB hiccup are all "this one session failed".
            # The message is truncated so a stack trace in a provider's stderr
            # cannot flood the nightly log.
            print(f"  ! {s['session_id'][:8]}: {str(e)[:300]}", file=sys.stderr)
            failed += 1
            consecutive_failures += 1
            if consecutive_failures >= 5:
                # circuit breaker: exhausted quota / broken provider would
                # otherwise burn one failing LLM call per remaining session
                # Five in a row is the signal that the fault is systemic (expired
                # credential, rate limit, provider outage) rather than one odd
                # transcript, since a run of five unrelated bad sessions is
                # vanishingly unlikely. Stopping turns a long, futile, possibly
                # billable sweep into one short red run. Nothing is lost: the
                # remaining sessions are still stale and get picked up next time.
                print("  !! 5 consecutive failures — provider likely down or "
                      "quota exhausted; aborting this run", file=sys.stderr)
                break
        # Deliberate pause between calls, outside the try so it also applies after
        # a failure: a nightly sweep of hundreds of sessions firing back-to-back
        # requests is exactly what a provider's rate limiter exists to stop.
        time.sleep(args.rate_limit)

    conn.close()
    print(f"Enriched {done}/{len(sessions)} sessions ({skipped} skipped, {failed} failed).")
    if failed:
        # A broken provider (expired credential, exhausted quota) used to end a
        # green night with zero summaries; refresh-all must see it.
        sys.exit(1)


if __name__ == "__main__":
    main()
