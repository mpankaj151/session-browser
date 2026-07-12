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

_URL = re.compile(r"https?://[^\s)\]>'\"]+")


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _local(dt_iso: str | None, tz: tzinfo | None):
    dt = _parse_ts(dt_iso)
    return dt.astimezone(tz) if dt else None


def resolve_window(window: str | None, date_from: str | None, date_to: str | None,
                   today: date | None = None) -> tuple[date, date, str]:
    """(from, to, label) — inclusive local dates."""
    today = today or datetime.now().astimezone().date()
    if date_from or date_to:
        lo = date.fromisoformat(date_from) if date_from else date(2000, 1, 1)
        hi = date.fromisoformat(date_to) if date_to else today
        return lo, hi, f"{lo} to {hi}"
    w = (window or "last-week").lower()
    if m := re.fullmatch(r"(\d+)d", w):
        n = int(m.group(1))
        return today - timedelta(days=n - 1), today, f"last {n} days"
    if w == "today":
        return today, today, "today"
    if w == "yesterday":
        y = today - timedelta(days=1)
        return y, y, "yesterday"
    if w == "this-week":
        return today - timedelta(days=today.weekday()), today, "this week"
    if w == "last-week":
        mon = today - timedelta(days=today.weekday() + 7)
        return mon, mon + timedelta(days=6), "last week"
    if w == "this-month":
        return today.replace(day=1), today, "this month"
    if w == "last-month":
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end, "last month"
    if w in ("this-quarter", "last-quarter"):
        q_start_month = 3 * ((today.month - 1) // 3) + 1
        this_q = today.replace(month=q_start_month, day=1)
        if w == "this-quarter":
            return this_q, today, "this quarter"
        end = this_q - timedelta(days=1)
        return end.replace(month=3 * ((end.month - 1) // 3) + 1, day=1), end, "last quarter"
    if w == "last-6-months":
        end = today.replace(day=1) - timedelta(days=1)  # last day of prev month
        start = end.replace(day=1)
        for _ in range(5):
            start = (start - timedelta(days=1)).replace(day=1)
        return start, end, "last 6 months"
    if w == "all":
        return date(2000, 1, 1), today, "all time"
    raise SystemExit(f"unknown --window {window!r} (try last-week, last-month, "
                     f"last-quarter, last-6-months, Nd, or --from/--to)")


def _is_stale(row) -> bool:
    """Python twin of enrich-sessions' STALE_PREDICATE."""
    if not row["summary"]:
        return True
    last, enriched = _parse_ts(row["last_activity"]), _parse_ts(row["enriched_at"])
    return bool(last and (enriched is None or last > enriched))


def _artifacts(conn, sids: list[str], kind: str) -> dict[str, list[str]]:
    if not sids:
        return {}
    marks = ",".join("?" * len(sids))
    out: dict[str, list[str]] = defaultdict(list)
    for r in conn.execute(
            f"SELECT session_id, content FROM session_artifacts WHERE type=? "
            f"AND session_id IN ({marks}) ORDER BY turn_index", [kind, *sids]):
        out[r["session_id"]].append(r["content"])
    return out


def build_report(conn, lo: date, hi: date, label: str,
                 tz: tzinfo | None = None, on_disk: dict[str, int] | None = None) -> dict:
    rows = []
    for r in conn.execute("SELECT * FROM sessions WHERE archived=0 ORDER BY start_time"):
        local = _local(r["start_time"], tz)
        if local and lo <= local.date() <= hi:
            rows.append((local, r))

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

    for local, r in rows:
        sid = r["session_id"]
        if not r["summary"]:
            unenriched_ids.append(sid)
        elif _is_stale(r):
            stale_ids.append(sid)
        by_source[r["cli_source"]] += 1
        by_type[r["session_type"] or "unclassified"] += 1
        by_outcome[r["outcome"] or "unknown"] += 1
        cost += r["cost_usd"] or 0.0
        day = local.strftime("%Y-%m-%d")
        active_days.add(day)
        iso = local.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        weeks[week] += 1
        journal = journals.get(sid, [""])[0]
        text_blob = " ".join([r["summary"] or "", journal, *decisions.get(sid, [])])
        end = _parse_ts(r["last_activity"])
        start = _parse_ts(r["start_time"])
        mins = round((end - start).total_seconds() / 60) if start and end and end >= start else None
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
            "trivial": (r["turn_count"] or 0) < 2,
            "links": sorted(set(_URL.findall(text_blob))),
        })

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
            "by_source": dict(sorted(by_source.items())),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "by_outcome": dict(sorted(by_outcome.items(), key=lambda kv: -kv[1])),
            "cost_usd": round(cost, 2),
            "cost_is_notional": sbconfig.COST_IS_NOTIONAL,
        },
        "projects": sorted(projects.values(), key=lambda p: -len(p["sessions"])),
        "weeks": [{"week": w, "sessions": n} for w, n in sorted(weeks.items())],
    }


def _on_disk_counts() -> dict[str, int]:
    """All-time transcript counts per source — cheap globs; flags an index that
    has fallen behind disk (watcher down, fresh machine)."""
    try:
        from sources.registry import build_source_registry
        adapters = build_source_registry(only_available=True)
        return {name: sum(1 for _ in a.discover()) for name, a in adapters.items()}
    except Exception:  # noqa: BLE001 — coverage estimate is best-effort
        return {}


def main() -> None:
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
