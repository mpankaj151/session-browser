#!/usr/bin/env python3
"""Windowed work-journal data for reports — read-only over the registry.

Emits one JSON object the work-journal skill turns into weekly/monthly/review
summaries and the HTML timeline. Never touches transcripts; the journal-grade
rows written by enrichment are the source of truth.

    report-data.py --window last-week|last-month|last-quarter|last-6-months|Nd
    report-data.py --from 2026-04-01 --to 2026-06-30
    report-data.py --check-only --window last-quarter   # coverage block only

Named windows resolve deterministically in LOCAL time: last-week is the
previous Mon-Sun, last-month/quarter the previous calendar month/quarter,
last-6-months the six full months ending with the previous one. Sessions fall
in a window by the local date of their start_time (same rule as daily-digest).

No AI model is called here and nothing is spent: this script only reads and
re-shapes rows that scripts/enrich-sessions.py already wrote. (Enrichment is the
one part of Session Browser that talks to a model, and it costs the user's own CLI
quota — see docs/GLOSSARY.md.) The division of labour is deliberate: this script
produces facts, and the skill that consumes the JSON writes the prose.

"Deterministic" matters because these reports back real-world claims — a
performance review, a stand-up. Running the same window twice must return the same
sessions, so "last week" is a fixed calendar range, never "the last seven days
from right now".

The `coverage` block in the output is the honesty check. It reports how many
sessions in the window have no summary yet (`unenriched_ids`) or have been used
since they were last summarised (`stale_ids`), plus an all-time count of
transcripts on disk versus rows indexed. A consumer that ignores it can silently
write a review of half the quarter's work.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402

# Pulls URLs out of summaries, journals and decisions so a report can link to the
# PR or issue a session was about. Matches http:// or https:// followed by
# everything up to whitespace or a closing bracket/quote — the characters that
# normally END a URL in prose or Markdown: `)`, `]`, `>`, `'`, `"`.
# So "see [the PR](https://github.com/o/r/pull/12)." yields the URL without the
# trailing ")." that a naive \S+ would swallow.
_URL = re.compile(r"https?://[^\s)\]>'\"]+")


def _parse_ts(s: str | None) -> datetime | None:
    """Stored timestamp -> aware datetime, or None if unparseable.

    Same tolerance as scripts/daily-digest.py: the canonical
    '2026-06-01T09:30:00.000Z' spelling, the legacy 'YYYY-MM-DD HH:MM:SS' that
    SQLite's CURRENT_TIMESTAMP produced, or an ISO string with an offset. The two
    replace() calls normalise the first two into what fromisoformat accepts; a
    value with no zone is taken as UTC, which is what the legacy form meant.
    Never raises — one bad row must not sink a quarterly report.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _local(dt_iso: str | None, tz: tzinfo | None):
    """A stored UTC timestamp in local time (None if unparseable).

    Windows are calendar ranges in the user's own zone, so every comparison has to
    happen after this conversion. `tz=None` means the machine's zone; tests pass a
    fixed one.
    """
    dt = _parse_ts(dt_iso)
    return dt.astimezone(tz) if dt else None


def resolve_window(window: str | None, date_from: str | None, date_to: str | None,
                   today: date | None = None) -> tuple[date, date, str]:
    """(from, to, label) — inclusive local dates.

    Turns a window name into a concrete calendar range. `today` is injectable so
    tests can pin "now" and assert, for instance, that last-week really is the
    previous Monday to Sunday regardless of when the suite runs.

    Explicit --from/--to wins over --window. An open end fills in: no --from means
    2000-01-01 (effectively "everything"), no --to means today.

    "this-*" windows end today and are therefore partial by nature; the "last-*"
    ones are the closed, reviewable periods. Raises SystemExit with the list of
    valid names for anything unrecognised — a typo must not silently produce an
    empty report that reads as "you did nothing last quarter".
    """
    today = today or datetime.now().astimezone().date()
    if date_from or date_to:
        lo = date.fromisoformat(date_from) if date_from else date(2000, 1, 1)
        hi = date.fromisoformat(date_to) if date_to else today
        return lo, hi, f"{lo} to {hi}"
    w = (window or "last-week").lower()
    # "30d" / "7d": a rolling window of N days ENDING TODAY. fullmatch means the
    # whole string must be digits followed by "d", so "30days" is rejected rather
    # than half-understood.
    if m := re.fullmatch(r"(\d+)d", w):
        n = int(m.group(1))
        # n - 1 because both ends are inclusive: "7d" is today plus the 6 days
        # before it, not 8 days.
        return today - timedelta(days=n - 1), today, f"last {n} days"
    if w == "today":
        return today, today, "today"
    if w == "yesterday":
        y = today - timedelta(days=1)
        return y, y, "yesterday"
    # date.weekday() is 0 for Monday, so subtracting it lands on this Monday.
    if w == "this-week":
        return today - timedelta(days=today.weekday()), today, "this week"
    if w == "last-week":
        # ... and another 7 days back is the PREVIOUS Monday; +6 makes it Sunday.
        mon = today - timedelta(days=today.weekday() + 7)
        return mon, mon + timedelta(days=6), "last week"
    if w == "this-month":
        return today.replace(day=1), today, "this month"
    if w == "last-month":
        # Day 1 of this month minus one day = the last day of the previous month,
        # whatever its length and without any month-length arithmetic.
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end, "last month"
    if w in ("this-quarter", "last-quarter"):
        # Month -> the first month of its quarter: (month-1)//3 gives the quarter
        # index 0-3, times 3 plus 1 gives 1, 4, 7 or 10 (Jan/Apr/Jul/Oct).
        q_start_month = 3 * ((today.month - 1) // 3) + 1
        this_q = today.replace(month=q_start_month, day=1)
        if w == "this-quarter":
            return this_q, today, "this quarter"
        # One day before this quarter starts is the last day of the previous one;
        # the same month formula then finds that quarter's first month. Works
        # across a year boundary because `end` already carries the right year.
        end = this_q - timedelta(days=1)
        return end.replace(month=3 * ((end.month - 1) // 3) + 1, day=1), end, "last quarter"
    if w == "last-6-months":
        end = today.replace(day=1) - timedelta(days=1)  # last day of prev month
        start = end.replace(day=1)
        # Step back five more months by repeatedly jumping to the day before the
        # 1st (i.e. the end of the previous month) and snapping to its 1st. Six
        # FULL months in total, ending with the previous one — the current, partial
        # month is excluded on purpose.
        for _ in range(5):
            start = (start - timedelta(days=1)).replace(day=1)
        return start, end, "last 6 months"
    if w == "all":
        return date(2000, 1, 1), today, "all time"
    raise SystemExit(f"unknown --window {window!r} (try last-week, last-month, "
                     f"last-quarter, last-6-months, Nd, or --from/--to)")


def _is_stale(row) -> bool:
    """Python twin of enrich-sessions' STALE_PREDICATE.

    True when a session has been used since it was last summarised — it was
    resumed, and its journal no longer describes all of it. Reported as
    `coverage.stale_ids` so a report can say so rather than quietly presenting a
    half-current summary as the whole story.

    Deliberately duplicated from the SQL predicate rather than shared: that one
    has to be a SQL string to compose into a WHERE clause, this one runs over rows
    already loaded. The two must be changed together.
    """
    if not row["summary"]:
        return True
    last, enriched = _parse_ts(row["last_activity"]), _parse_ts(row["enriched_at"])
    return bool(last and (enriched is None or last > enriched))


def _artifacts(conn, sids: list[str], kind: str) -> dict[str, list[str]]:
    """session_id -> its artifacts of one `kind`, in stored order.

    `kind` is 'journal' (the rendered Markdown entry, one per session) or
    'decision' (one row per key decision). Both are written by
    scripts/enrich-sessions.py.

    The SQL selects the rows for this batch of session ids and orders them by
    turn_index, which for decisions is their position in the model's list — so
    decisions come back in the order they were written rather than in whatever
    order SQLite happens to return.

    `marks` builds the right number of `?` placeholders ('?,?,?' for three ids) so
    the ids go through parameter binding rather than string formatting.
    Read-only; returns {} for an empty id list.
    """
    if not sids:
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    # Chunked: older libsqlite3 (Ubuntu 20.04) caps bind parameters at 999.
    # A quarter can easily hold more sessions than that, and the whole query
    # would fail rather than degrade — hence 500 ids at a time (plus `kind`),
    # comfortably under the limit on every SQLite build.
    for i in range(0, len(sids), 500):
        chunk = sids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
                f"SELECT session_id, content FROM session_artifacts WHERE type=? "
                f"AND session_id IN ({marks}) ORDER BY turn_index", [kind, *chunk]):
            out[r["session_id"]].append(r["content"])
    return out


def build_report(conn, lo: date, hi: date, label: str,
                 tz: tzinfo | None = None, on_disk: dict[str, int] | None = None) -> dict:
    """Build the whole report object for one window. Read-only; no model call.

    `lo`/`hi` are inclusive local dates from resolve_window, `label` the phrase to
    print ("last quarter"). `on_disk` is the optional per-CLI transcript census
    from _on_disk_counts(), folded into the coverage block.

    Returns the JSON-serialisable dict documented at the bottom of this function:
    `window`, `generated_at`, `coverage`, `stats`, `projects` (each with its
    sessions) and `weeks`. Sessions are nested under projects because that is how
    the reports read — per project, then chronologically.

    `conn` is an open registry connection; tests pass a temporary one.
    """
    rows = []
    # One pass over every visible session. indexer.VISIBLE means the rows the user
    # should see and that usage stats should count: sessions whose transcript is
    # still on disk, plus real sessions whose transcript has since aged out (their
    # summary survives, which is the entire point of keeping the row). Subagent
    # sidechain noise is excluded. ORDER BY start_time keeps each project's list
    # chronological with no further sorting.
    #
    # Filtering by date in Python, not in SQL: the window is a range of LOCAL
    # calendar dates and the column is UTC text, so the comparison can only be made
    # after converting each row — and SQLite does not know the machine's zone.
    for r in conn.execute(f"SELECT * FROM sessions WHERE {indexer.VISIBLE} ORDER BY start_time"):
        local = _local(r["start_time"], tz)
        if local and lo <= local.date() <= hi:
            rows.append((local, r))

    # Fetch journals and decisions for the whole window in two queries rather than
    # two per session.
    sids = [r["session_id"] for _, r in rows]
    journals = _artifacts(conn, sids, "journal")
    decisions = _artifacts(conn, sids, "decision")

    by_source: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    by_outcome: dict[str, int] = defaultdict(int)
    projects: dict[str, dict] = {}
    weeks: dict[str, int] = defaultdict(int)
    active_days: set[str] = set()
    cost = 0.0
    unenriched_ids, stale_ids = [], []

    # One pass builds every aggregate at once: the histograms, the per-project
    # nesting, the week buckets, the set of days with any activity, and the
    # coverage lists.
    for local, r in rows:
        sid = r["session_id"]
        # Two distinct kinds of gap, kept apart because they mean different things:
        # no summary at all, versus a summary that has fallen behind the session.
        if not r["summary"]:
            unenriched_ids.append(sid)
        elif _is_stale(r):
            stale_ids.append(sid)
        by_source[r["cli_source"]] += 1
        by_type[r["session_type"] or "unclassified"] += 1
        by_outcome[r["outcome"] or "unknown"] += 1
        # Notional under a flat subscription: the public list-price equivalent of
        # the session's tokens, not money actually billed. `cost_is_notional` in
        # the output says which reading applies.
        cost += r["cost_usd"] or 0.0
        day = local.strftime("%Y-%m-%d")
        active_days.add(day)
        # ISO week numbering: isocalendar() returns (year, week, weekday), and the
        # ISO YEAR is not always the calendar year — 1 January can belong to week
        # 52 of the year before. Using iso[0] rather than local.year is what keeps
        # a new-year week from being filed under the wrong year. :02d zero-pads so
        # '2026-W07' sorts correctly against '2026-W12' as plain text.
        iso = local.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        weeks[week] += 1
        # There is at most one journal per session; [""] is the default so the [0]
        # is safe for a session that was never enriched.
        journal = journals.get(sid, [""])[0]
        # Everything a link could be hiding in, concatenated once, so _URL runs
        # over summary + journal + decisions in a single pass.
        text_blob = " ".join([r["summary"] or "", journal, *decisions.get(sid, [])])
        end = _parse_ts(r["last_activity"])
        start = _parse_ts(r["start_time"])
        # None rather than 0 when the span is unknown or negative (a clock change),
        # so a consumer can omit the duration instead of reporting a 0-minute session.
        mins = round((end - start).total_seconds() / 60) if start and end and end >= start else None
        # setdefault creates the project bucket the first time it is seen, so the
        # projects dict grows as the loop discovers them.
        proj = projects.setdefault(r["folder_name"] or "(unknown project)", {
            "name": r["folder_name"] or "(unknown project)",
            "cwd": r["cwd"] or "", "sessions": []})
        proj["sessions"].append({
            "session_id": sid,
            "date": day,
            "week": week,
            "time": local.strftime("%H:%M"),
            "title": (r["title"] or "").strip() or None,
            "summary": (r["summary"] or "").strip() or None,
            "journal": journal or None,
            "decisions": decisions.get(sid, []),
            "topics": json.loads(r["topics"]) if r["topics"] else [],
            "type": r["session_type"],
            "outcome": r["outcome"],
            "source": r["cli_source"],
            "duration_min": mins,
            "turns": r["turn_count"],
            # "trivial" flags a one-question session — a quick lookup, not a piece
            # of work. Reports use it to separate "sessions" from "real sessions"
            # rather than dropping the row, so the count stays honest.
            "trivial": (r["turn_count"] or 0) < 2,
            # set() de-duplicates a URL mentioned in both the summary and the
            # journal; sorted() makes the output stable between runs.
            "links": sorted(set(_URL.findall(text_blob))),
        })

    # The honesty block: how much of this window the report can actually speak for.
    # A consumer that skips it can present a quarter with half its sessions missing
    # as a complete account.
    coverage = {
        "indexed": len(rows),
        "enriched": len(rows) - len(unenriched_ids),
        "unenriched_ids": unenriched_ids,
        "stale_ids": stale_ids,
    }
    if on_disk is not None:
        coverage["on_disk_all_time"] = on_disk
    return {
        "window": {"from": lo.isoformat(), "to": hi.isoformat(), "label": label},
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "coverage": coverage,
        "stats": {
            "sessions": len(rows),
            "non_trivial": sum(1 for p in projects.values()
                               for s in p["sessions"] if not s["trivial"]),
            "active_days": len(active_days),
            # by_source sorts by NAME (a stable list of CLIs reads better);
            # by_type and by_outcome sort by COUNT descending — negative count as
            # the key — so the dominant kind of work leads.
            "by_source": dict(sorted(by_source.items())),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "by_outcome": dict(sorted(by_outcome.items(), key=lambda kv: -kv[1])),
            "cost_usd": round(cost, 2),
            # True under a flat subscription: the figure is a list-price
            # equivalent, an intensity signal, not money billed. The consumer must
            # label it accordingly.
            "cost_is_notional": sbconfig.COST_IS_NOTIONAL,
        },
        # Busiest project first; weeks in chronological order (the 'YYYY-Www'
        # spelling sorts correctly as text).
        "projects": sorted(projects.values(), key=lambda p: -len(p["sessions"])),
        "weeks": [{"week": w, "sessions": n} for w, n in sorted(weeks.items())],
    }


def _on_disk_counts() -> dict[str, int]:
    """All-time transcript counts per source — cheap globs; flags an index that
    has fallen behind disk (watcher down, fresh machine).

    Counts FILES, not sessions in the window: the comparison it enables is
    "3400 transcripts on disk, 2900 rows indexed", which is the quickest way to
    notice that the background watcher has been dead for a week. Deliberately
    all-time, because a per-window count would need every file parsed.

    Returns {} on any failure — this is a nice-to-have annotation on the coverage
    block, and a missing adapter or an unreadable directory must never take a
    quarterly report down with it.
    """
    try:
        from sources.registry import build_source_registry
        adapters = build_source_registry(only_available=True)
        # sum(1 for _ in ...) counts a generator without building a list;
        # discover() only lists paths, it never opens the files.
        return {name: sum(1 for _ in a.discover()) for name, a in adapters.items()}
    except Exception:  # noqa: BLE001 — coverage estimate is best-effort
        return {}


def main() -> None:
    """Resolve the window, build the report, print it as JSON on stdout.

    Read-only: nothing is written to the database or the filesystem, and no model
    is called. The caller (the work-journal skill, or a person) pipes the JSON
    wherever it is needed. `--check-only` prints just the coverage block and the
    session count — the cheap "is my index complete enough to report on this?"
    question, without the full payload.

    ensure_ascii=False keeps non-English characters readable rather than escaped.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", help="last-week|last-month|last-quarter|last-6-months|"
                                     "this-*|today|yesterday|Nd|all")
    ap.add_argument("--from", dest="date_from", help="YYYY-MM-DD (local)")
    ap.add_argument("--to", dest="date_to", help="YYYY-MM-DD (local)")
    ap.add_argument("--check-only", action="store_true", help="coverage block only")
    args = ap.parse_args()

    lo, hi, label = resolve_window(args.window, args.date_from, args.date_to)
    conn = indexer.connect()
    try:
        report = build_report(conn, lo, hi, label, on_disk=_on_disk_counts())
    finally:
        conn.close()
    if args.check_only:
        report = {"window": report["window"], "coverage": report["coverage"],
                  "stats": {"sessions": report["stats"]["sessions"]}}
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
