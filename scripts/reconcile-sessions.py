#!/usr/bin/env python3
"""Reconcile diverged Claude session copies into the in-sync symlink model.

Going forward, `cr` symlinks (so sessions never diverge). But a session copied in
the brief cp era — or via the cp fallback — can exist as multiple REAL files in
different project dirs, each a fork with possibly-unique turns. This tool makes the
live state coherent WITHOUT losing anything:

  * picks the most-recently-modified copy as canonical (it stays in place)
  * ARCHIVES every other copy to <archive>/superseded/<id>@<dir>-<ts>.jsonl
    (preserved for manual review — forks are never silently deleted)
  * replaces each archived location with a symlink to the canonical file, so
    resume from those dirs still works and stays in sync afterwards

Safe by default: prints a plan and changes nothing unless you pass --apply.
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sbconfig  # noqa: E402

PROJECTS = Path.home() / ".claude" / "projects"


def _real_copies() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = collections.defaultdict(list)
    if not PROJECTS.exists():
        return groups
    for d in PROJECTS.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("*.jsonl"):
            if not f.is_symlink():
                groups[f.stem].append(f)
    return {k: v for k, v in groups.items() if len(v) > 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry-run)")
    args = ap.parse_args()

    dupes = _real_copies()
    if not dupes:
        print("✓ No diverged copies — every session has a single canonical transcript.")
        return

    superseded = sbconfig.REASONING_ARCHIVE / "superseded"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"{len(dupes)} session(s) with diverged real copies "
          f"({'APPLYING' if args.apply else 'dry-run — pass --apply to act'}):\n")

    for sid, files in dupes.items():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        canonical = files[0]
        print(f"  {sid[:8]}  canonical = {canonical.parent.name}  "
              f"({canonical.stat().st_size//1024} KB, mtime {datetime.fromtimestamp(canonical.stat().st_mtime):%Y-%m-%d %H:%M})")
        for fork in files[1:]:
            tag = f"{sid}@{fork.parent.name}-{stamp}.jsonl"
            print(f"      fork    = {fork.parent.name}  ({fork.stat().st_size//1024} KB) "
                  f"-> archive {tag} + symlink to canonical")
            if args.apply:
                superseded.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fork, superseded / tag)   # preserve the fork
                fork.unlink()
                os.symlink(canonical.resolve(), fork)   # relink to canonical
        print()

    if args.apply:
        print(f"Done. Forks preserved under {superseded}")
        print("Re-run backfill to refresh the index:  scripts/backfill.py --source claude")
    else:
        print("Nothing changed. Re-run with --apply to reconcile.")


if __name__ == "__main__":
    main()
