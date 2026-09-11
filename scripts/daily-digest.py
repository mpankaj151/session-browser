#!/usr/bin/env python3
"""Assemble daily-logs/YYYY-MM-DD.md from enriched sessions — no LLM call.

Runs as the final nightly-refresh step. Self-healing: every PAST local day that
has sessions but whose file is missing — or stale, because a session on that
day was (re-)enriched after the file was written — is (re)written. Today is
skipped by default (still accumulating); `--date YYYY-MM-DD` forces one day,
including today (the work-journal skill's "what did I do today" path).

Sessions group under the LOCAL date of their start_time; a multi-day session
appears once, on its start date. All sessions are listed — enriched ones with
their journal, unenriched ones with a title/first-message fallback marked
(unsummarized). Session ids are unique in the registry, so a day never lists
duplicates.

"No LLM call" is the headline property. Every other way of producing a daily
summary would send the day's work to an AI model and spend the user's quota (see
docs/GLOSSARY.md; that is what scripts/enrich-sessions.py does). This script only
re-arranges text that is already in the database: the per-session journals written
by enrichment, grouped by day and project. So it is pure assembly — deterministic,
free, and safe to re-run as often as you like. It is therefore NOT gated behind
`refresh-all.py --enrich`; it runs last on every nightly pass so it always sees
that night's enrichment.

Output: one Markdown file per day at `<daily_dir>/YYYY-MM-DD.md`, by default
`~/.session-browser/daily-logs/`, overridable with `[digest].daily_dir` in
config.toml or `--daily-dir`. Consumed by the work-journal skill ("what did I do
yesterday") and readable directly.

Self-healing means the script does not track what it has already written: it
recomputes which days are missing or out of date on every run. A laptop that was
shut for a week catches up on the next pass, and a session resumed and
re-journaled today updates the daily file of the day it STARTED on.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone, tzinfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402


def _parse_ts(s: str | None) -> datetime | None:
    """Mixed-format tolerant: canonical '...T...Z', legacy 'YYYY-MM-DD HH:MM:SS'
    (UTC, from CURRENT_TIMESTAMP), or offset ISO. None for garbage.

    The two replace() calls translate the stored spelling into the one
    datetime.fromisoformat accepts on older Pythons: a space between date and time
    becomes 'T', and a trailing 'Z' (meaning UTC) becomes '+00:00'. A value with no
    zone at all is assumed UTC, which is correct here because every naive timestamp
    in this registry came from SQLite's CURRENT_TIMESTAMP.

    Returns an aware datetime, or None — never raises, because one malformed
    timestamp must not take down a whole digest run.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _local_date(iso: str | None, tz: tzinfo | None = None) -> str:
    """A stored UTC timestamp as a local calendar date, 'YYYY-MM-DD' ('' if unparseable).

    Local, not UTC, is the whole point: a session at 9pm on Monday in a UTC+X zone
    is stored as Tuesday and must still appear in Monday's log. `tz=None` means the
    machine's own zone; tests pass a fixed zone to pin the behaviour.
    """
    dt = _parse_ts(iso)
    if dt is None:
        return ""
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def _duration_min(row) -> int | None:
    """How long a session lasted, in whole minutes, or None if that is not knowable.

    None rather than 0 for a missing or nonsensical span (last_activity before
    start_time, which a clock change can produce), so render_day can omit the badge
    entirely instead of claiming a 0-minute session.
    """
    a, b = _parse_ts(row["start_time"]), _parse_ts(row["last_activity"])
    if a is None or b is None or b < a:
        return None
    return round((b - a).total_seconds() / 60)


def collect_days(conn, tz: tzinfo | None = None) -> dict[str, list]:
    """Local date -> sessions started that day (skips rows with no timestamp).

    One query for the whole registry, bucketed in Python rather than grouped in
    SQL: the bucket is a LOCAL date and SQLite has no notion of the machine's time
    zone, so the conversion has to happen here. At this scale (thousands of rows)
    reading everything once is cheaper than a query per day anyway.

    indexer.VISIBLE selects what the user should see — sessions whose transcript is
    still on disk, plus real sessions whose transcript has since aged out — and
    excludes subagent sidechain noise. ORDER BY start_time makes each day's list
    chronological, which render_day relies on.

    `conn` is an open registry connection (tests pass a temporary one); `tz`
    defaults to the machine's zone. Read-only.
    """
    days: dict[str, list] = defaultdict(list)
    for row in conn.execute(
            f"SELECT * FROM sessions WHERE {indexer.VISIBLE} ORDER BY start_time"):
        day = _local_date(row["start_time"], tz)
        if day:
            days[day].append(row)
    return dict(days)


def _journals(conn, session_ids: list[str]) -> dict[str, str]:
    """session_id -> its rendered markdown journal, for the ids that have one.

    The journal is the per-session record enrichment produced (accomplishments,
    key decisions, explorations, open threads), stored as a `journal` artifact row
    by scripts/enrich-sessions.py. Sessions that were never enriched simply have no
    entry in the returned dict.

    One query for a whole day rather than one per session. `marks` builds the
    matching number of `?` placeholders — '?,?,?' for three ids — so the ids go
    through SQLite's parameter binding instead of being formatted into the SQL.
    Read-only.
    """
    if not session_ids:
        return {}
    marks = ",".join("?" * len(session_ids))
    return {r["session_id"]: r["content"] for r in conn.execute(
        f"SELECT session_id, content FROM session_artifacts "
        f"WHERE type='journal' AND session_id IN ({marks})", session_ids)}


def render_day(day: str, rows: list, journals: dict[str, str],
               tz: tzinfo | None = None) -> str:
    """Render one day's Markdown page. Pure function: same inputs, same bytes out.

    `day` is 'YYYY-MM-DD', `rows` that day's session rows (chronological),
    `journals` the map from _journals(). Shape of the page:

        # Work log — 2026-06-01 (Monday)
        *5 sessions · 2 projects · claude ×4 · codex ×1*
        ## <project with the most sessions>
        ### 09:12 — <title>  `feature · completed · claude · 42 min`
        <summary, then the journal's sections one heading level lower>

    Projects are ordered by session count (busiest first) so the day's main thread
    of work leads; sessions within a project stay chronological. Returning a string
    rather than writing the file is what lets tests assert on the output and lets
    main() decide whether the file needs replacing at all.
    """
    weekday = datetime.strptime(day, "%Y-%m-%d").strftime("%A")
    by_source: dict[str, int] = defaultdict(int)
    by_project: dict[str, list] = defaultdict(list)
    for r in rows:
        by_source[r["cli_source"]] += 1
        by_project[r["folder_name"] or "(unknown project)"].append(r)
    sources = " · ".join(f"{s} ×{n}" for s, n in sorted(by_source.items()))
    out = [f"# Work log — {day} ({weekday})", "",
           f"*{len(rows)} session{'s' if len(rows) != 1 else ''} · "
           f"{len(by_project)} project{'s' if len(by_project) != 1 else ''} · {sources}*", ""]
    # Negative length as the sort key = descending by session count: the project
    # the day was mostly about comes first.
    for project in sorted(by_project, key=lambda p: -len(by_project[p])):
        out.append(f"## {project}")
        out.append("")
        for r in sorted(by_project[project], key=lambda r: r["start_time"]):
            # Three fallbacks, best first: the title (written by a model, or by
            # the CLI), else the first line of what the user typed, else the id's
            # first 8 characters so the entry is at least identifiable.
            title = (r["title"] or "").strip() or \
                (r["first_message"] or "").strip().split("\n")[0][:80] or r["session_id"][:8]
            # Badges skip empty values, so an unenriched session shows just its CLI
            # rather than "· · claude".
            badges = [b for b in (r["session_type"], r["outcome"], r["cli_source"]) if b]
            mins = _duration_min(r)
            if mins is not None:
                badges.append(f"{mins} min")
            start = _parse_ts(r["start_time"])
            clock = start.astimezone(tz).strftime("%H:%M") if start else "--:--"
            out.append(f"### {clock} — {title}  `{' · '.join(badges)}`")
            summary = (r["summary"] or "").strip()
            # Unenriched sessions are listed too, clearly marked. A day's log that
            # silently omitted them would misrepresent the day — and the marker is
            # also the user's cue that enrichment has not run (or has no CLI).
            out.append(summary if summary
                       else f"_(unsummarized)_ {(r['first_message'] or '').strip()[:200]}")
            journal = journals.get(r["session_id"], "").strip()
            if journal:
                # demote the journal's H2 headings under this session's H3
                # The regex rewrites every line that STARTS with "## " (the (?m)
                # flag makes ^ mean "start of line", not "start of string") into
                # "#### ": "## Accomplishments" becomes "#### Accomplishments".
                # Without this the journal's own headings would outrank the "###"
                # session heading they belong under and break the page outline.
                out.append("")
                out.append(re.sub(r"(?m)^## ", "#### ", journal))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def needs_write(path: Path, rows: list) -> bool:
    """Missing, or any session on the day changed after the file was written —
    that is how a resumed session's re-journal propagates into its daily.

    This is the self-healing rule, and it deliberately keeps no state of its own:
    the file's own modification time IS the record of when it was last built.
    Comparing that against each session's timestamps answers "has anything on this
    day changed since?" without a marker file that could drift out of sync.

    Two columns are checked because either can move: `enriched_at` when a session
    is (re-)summarised, `last_activity` when it is resumed and gains new turns.

    Returns True if the day should be rewritten. Only stats the file; no writes.
    """
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    for r in rows:
        for col in ("enriched_at", "last_activity"):
            ts = _parse_ts(r[col])
            if ts is not None and ts > mtime:
                return True
    return False


def main() -> None:
    """Write every day that needs writing. No model call, so nothing is spent.

    Default pass: every PAST local day that has sessions and whose file is missing
    or out of date. Today is skipped because it is still accumulating — a file
    written at 2pm would claim to be the whole day. `--date` overrides that and is
    how the work-journal skill answers "what did I do today".

    Side effects: creates the daily directory if needed and writes
    `<daily_dir>/YYYY-MM-DD.md` per target day. Always exits 0 — a day with no
    sessions is not an error, just nothing to say.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="(re)write exactly this local day, incl. today")
    ap.add_argument("--force", action="store_true", help="rewrite even if fresh")
    ap.add_argument("--daily-dir", help="override [digest].daily_dir")
    args = ap.parse_args()

    daily_dir = Path(args.daily_dir).expanduser() if args.daily_dir else sbconfig.DAILY_DIR
    daily_dir.mkdir(parents=True, exist_ok=True)

    conn = indexer.connect()
    try:
        days = collect_days(conn)
        # .astimezone() attaches the machine's zone to a naive now(), so "today"
        # is the user's today — the same local-date rule collect_days used.
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        if args.date:
            targets = [args.date] if args.date in days else []
            if not targets:
                print(f"no sessions on {args.date}; nothing to write")
                return
        else:
            # `d < today` is a string comparison, which is correct for
            # 'YYYY-MM-DD' — the format sorts chronologically as text. Excludes
            # today: it is still in progress.
            targets = [d for d in sorted(days) if d < today]

        written = skipped = 0
        for day in targets:
            rows = days[day]
            path = daily_dir / f"{day}.md"
            # An explicit --date is itself a request to rewrite, so the freshness
            # check is skipped for it as well as for --force.
            if not args.force and not args.date and not needs_write(path, rows):
                skipped += 1
                continue
            journals = _journals(conn, [r["session_id"] for r in rows])
            path.write_text(render_day(day, rows, journals), encoding="utf-8")
            written += 1
            print(f"  ✓ {path.name} ({len(rows)} sessions)")
        print(f"daily-digest: {written} written, {skipped} up to date "
              f"({len(days)} active days total)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
