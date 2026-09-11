#!/usr/bin/env python3
"""Restore aged-out sessions from the reasoning archive's raw transcript copies.

    restore-session.py <session_id>      restore one session now
    restore-session.py --all             plan: which archived rows CAN come back
    restore-session.py --all --apply     restore every restorable row

Claude Code's cleanupPeriodDays deletes old transcripts; refresh-all had already
copied them to <archive>/raw/. This puts the copy back where `claude --resume`
and `cr` look, and re-indexes so the row leaves the Archived view.

`--all` is a dry-run by default (same convention as prune/reconcile): it prints the plan
and changes nothing, because the first thing anyone does on a laptop that lost sessions
is find out what is actually recoverable. Only `--apply` writes.

Exit codes (a caller can script against these):
  0   the single restore succeeded or the session was already live; a plan printed; an
      --apply run in which every attempted restore worked; or nothing to do at all
  1   the single restore failed for any reason (not-found / not-a-session / no-raw-copy /
      unsupported), or at least one row failed during --apply
  2   argparse's own error — neither or both of <session_id> and --all were given

Output legend for the plan listing, one line per archived row:
  ✓  restorable now (this machine supports it and a raw copy exists)
  ·  supported, but no raw copy was ever taken — nothing to put back
  ✗  this row's CLI cannot accept a transcript back on this machine

All the real work lives in restore.py; this file is flags, formatting and exit codes.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

# Scripts are run directly rather than imported as a package, so the repo root has to go
# on the import path before `import restore` can work (hence the noqa on that import).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import restore  # noqa: E402


def main() -> None:
    """Parse the flags, run one restore or the whole plan, print it, set the exit code."""
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id", nargs="?", help="restore this one session")
    ap.add_argument("--all", action="store_true", help="every archived (transcript-missing) row")
    ap.add_argument("--apply", action="store_true", help="with --all: perform the restores")
    args = ap.parse_args()
    # Exactly one mode: `==` on the two booleans is true when BOTH were given and when
    # NEITHER was, which are the two ways to be wrong. ap.error() exits 2.
    if bool(args.session_id) == args.all:
        ap.error("give exactly one of <session_id> or --all")

    if args.session_id:
        r = restore.restore_session(args.session_id)
        # reimported is three-valued: None means this CLI needs no re-import step at all
        # (the file on disk IS the session), so stay silent; True/False mean one was
        # attempted, and a failure is worth flagging because the row is live here while
        # the CLI itself may still be unable to resume it.
        flag = "" if r.reimported is None else ("  [re-imported]" if r.reimported else "  [re-import FAILED]")
        print(f"{r.status:14} {r.session_id}  {r.path or ''}  {r.detail}{flag}")
        # "already-live" counts as success: the caller asked for the session to be
        # available, and it is.
        sys.exit(0 if r.status in ("restored", "already-live") else 1)

    # --- --all: plan, then optionally apply --------------------------------------
    # plan() reads the registry and walks the raw vault once; it writes nothing.
    rows = restore.plan()
    if not rows:
        print("✓ No archived sessions with a missing transcript.")
        return
    can = [r for r in rows if r["restorable"]]
    print(f"{len(rows)} archived session(s) with a missing transcript; "
          f"{len(can)} restorable from the raw archive "
          f"({'APPLYING' if args.apply else 'dry-run — pass --apply to restore'}):\n")
    # Every archived row is listed, not just the restorable ones: "this one is gone for
    # good" is exactly the information someone auditing a data loss needs.
    for r in rows:
        mark = "✓" if r["restorable"] else ("·" if r["supported"] else "✗")
        why = "" if r["restorable"] else ("no raw copy" if r["supported"] else f"{r['cli_source']}: unsupported")
        # Fixed-width columns (id, folder, date, title) so the listing stays scannable;
        # each field is truncated first, then padded to its column width.
        print(f"  {mark} {r['session_id'][:36]:38} {str(r['folder_name'] or '-')[:24]:26} "
              f"{str(r['last_activity'] or '-')[:10]:12} {str(r['title'] or '')[:40]:42} {why}")
    if not args.apply:
        print("\nNothing changed. Re-run with --apply to restore the ✓ rows.")
        return
    # Tally statuses so the summary line reads "12 restored, 1 already-live, 1 error"
    # rather than scrolling one line per session.
    outcome: collections.Counter = collections.Counter()
    # Only the ✓ rows are attempted; the others were already shown with their reason.
    for r in can:
        try:
            res = restore.restore_session(r["session_id"])
        except Exception as e:  # noqa: BLE001 — one locked DB / corrupt copy must not abort the batch
            outcome["error"] += 1
            print(f"  ! {r['session_id'][:36]} -> error: {e}", file=sys.stderr)
            continue
        outcome[res.status] += 1
        # Print only what deserves attention: anything that is not a clean restore, plus
        # restores whose CLI-side re-import failed (the file is back, resume may not be).
        if res.status != "restored" or res.reimported is False:
            print(f"  ! {r['session_id'][:36]} -> {res.status} {res.detail}")
    print("\nDone: " + ", ".join(f"{n} {k}" for k, n in outcome.most_common()))
    # The restored transcripts are new files as far as the rest of the pipeline is
    # concerned: their reasoning trails and full-text entries are only rebuilt on the
    # next refresh.
    print("Re-run `sb refresh` so reasoning trails / full-text pick the restored files up.")
    # Anything other than restored/already-live — including the "error" bucket above —
    # makes the whole run nonzero so a scripted caller notices.
    failed = sum(n for k, n in outcome.items() if k not in ("restored", "already-live"))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
