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
    (UTC, from CURRENT_TIMESTAMP), or offset ISO. None for garbage."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _local_date(iso: str | None, tz: tzinfo | None = None) -> str:
    dt = _parse_ts(iso)
    if dt is None:
        return ""
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def _duration_min(row) -> int | None:
    a, b = _parse_ts(row["start_time"]), _parse_ts(row["last_activity"])
    if a is None or b is None or b < a:
        return None
    return round((b - a).total_seconds() / 60)


def collect_days(conn, tz: tzinfo | None = None) -> dict[str, list]:
    """Local date -> sessions started that day (skips rows with no timestamp)."""
    days: dict[str, list] = defaultdict(list)
    for row in conn.execute(
            "SELECT * FROM sessions WHERE archived = 0 ORDER BY start_time"):
        day = _local_date(row["start_time"], tz)
        if day:
            days[day].append(row)
    return dict(days)


def _journals(conn, session_ids: list[str]) -> dict[str, str]:
    if not session_ids:
        return {}
    marks = ",".join("?" * len(session_ids))
    return {r["session_id"]: r["content"] for r in conn.execute(
        f"SELECT session_id, content FROM session_artifacts "
        f"WHERE type='journal' AND session_id IN ({marks})", session_ids)}


def render_day(day: str, rows: list, journals: dict[str, str],
               tz: tzinfo | None = None) -> str:
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
    for project in sorted(by_project, key=lambda p: -len(by_project[p])):
        out.append(f"## {project}")
        out.append("")
        for r in sorted(by_project[project], key=lambda r: r["start_time"]):
            title = (r["title"] or "").strip() or \
                (r["first_message"] or "").strip().split("\n")[0][:80] or r["session_id"][:8]
            badges = [b for b in (r["session_type"], r["outcome"], r["cli_source"]) if b]
            mins = _duration_min(r)
            if mins is not None:
                badges.append(f"{mins} min")
            start = _parse_ts(r["start_time"])
            clock = start.astimezone(tz).strftime("%H:%M") if start else "--:--"
            out.append(f"### {clock} — {title}  `{' · '.join(badges)}`")
            summary = (r["summary"] or "").strip()
            out.append(summary if summary
                       else f"_(unsummarized)_ {(r['first_message'] or '').strip()[:200]}")
            journal = journals.get(r["session_id"], "").strip()
            if journal:
                # demote the journal's H2 headings under this session's H3
                out.append("")
                out.append(re.sub(r"(?m)^## ", "#### ", journal))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def needs_write(path: Path, rows: list) -> bool:
    """Missing, or any session on the day changed after the file was written —
    that is how a resumed session's re-journal propagates into its daily."""
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
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        if args.date:
            targets = [args.date] if args.date in days else []
            if not targets:
                print(f"no sessions on {args.date}; nothing to write")
                return
        else:
            targets = [d for d in sorted(days) if d < today]

        written = skipped = 0
        for day in targets:
            rows = days[day]
            path = daily_dir / f"{day}.md"
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
