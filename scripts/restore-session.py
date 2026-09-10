#!/usr/bin/env python3
"""Restore aged-out sessions from the reasoning archive's raw transcript copies.

    restore-session.py <session_id>      restore one session now
    restore-session.py --all             plan: which archived rows CAN come back
    restore-session.py --all --apply     restore every restorable row

Claude Code's cleanupPeriodDays deletes old transcripts; refresh-all had already
copied them to <archive>/raw/. This puts the copy back where `claude --resume`
and `cr` look, and re-indexes so the row leaves the Archived view.

`--all` is a dry-run by default (same convention as prune/reconcile).
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import restore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id", nargs="?", help="restore this one session")
    ap.add_argument("--all", action="store_true", help="every archived (transcript-missing) row")
    ap.add_argument("--apply", action="store_true", help="with --all: perform the restores")
    args = ap.parse_args()
    if bool(args.session_id) == args.all:
        ap.error("give exactly one of <session_id> or --all")

    if args.session_id:
        r = restore.restore_session(args.session_id)
        flag = "" if r.reimported is None else ("  [re-imported]" if r.reimported else "  [re-import FAILED]")
        print(f"{r.status:14} {r.session_id}  {r.path or ''}  {r.detail}{flag}")
        sys.exit(0 if r.status in ("restored", "already-live") else 1)

    rows = restore.plan()
    if not rows:
        print("✓ No archived sessions with a missing transcript.")
        return
    can = [r for r in rows if r["restorable"]]
    print(f"{len(rows)} archived session(s) with a missing transcript; "
          f"{len(can)} restorable from the raw archive "
          f"({'APPLYING' if args.apply else 'dry-run — pass --apply to restore'}):\n")
    for r in rows:
        mark = "✓" if r["restorable"] else ("·" if r["supported"] else "✗")
        why = "" if r["restorable"] else ("no raw copy" if r["supported"] else f"{r['cli_source']}: unsupported")
        print(f"  {mark} {r['session_id'][:36]:38} {str(r['folder_name'] or '-')[:24]:26} "
              f"{str(r['last_activity'] or '-')[:10]:12} {str(r['title'] or '')[:40]:42} {why}")
    if not args.apply:
        print("\nNothing changed. Re-run with --apply to restore the ✓ rows.")
        return
    outcome: collections.Counter = collections.Counter()
    for r in can:
        res = restore.restore_session(r["session_id"])
        outcome[res.status] += 1
        if res.status != "restored" or res.reimported is False:
            print(f"  ! {r['session_id'][:36]} -> {res.status} {res.detail}")
    print("\nDone: " + ", ".join(f"{n} {k}" for k, n in outcome.most_common()))
    print("Re-run `sb refresh` so reasoning trails / full-text pick the restored files up.")


if __name__ == "__main__":
    main()
