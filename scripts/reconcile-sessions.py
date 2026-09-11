#!/usr/bin/env python3
"""Reconcile diverged Claude session copies into the in-sync symlink model.

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI; its
*transcript* is the JSONL file the CLI writes as it happens; *resuming* is reopening an old
session so the conversation continues with its history intact.

Background, because the problem is not obvious. Claude Code files each transcript under the
directory it was started in: ~/.claude/projects/<slugified-cwd>/<session-id>.jsonl. To
resume a session from a DIFFERENT directory, the file has to exist under that directory's
slug too. `cr` (bin/resume-here.sh) does that with a SYMLINK, so both locations are the
same file and stay identical. An earlier version copied the file instead — and once there
are two real files, resuming from either one appends turns only to that copy. They diverge.

This is a one-off repair tool, run by hand, never from the nightly pipeline:

    .venv/bin/python scripts/reconcile-sessions.py            # dry run: prints a plan
    .venv/bin/python scripts/reconcile-sessions.py --apply

Claude-only by design: no other adapter has this directory-per-cwd layout.

Going forward, `cr` symlinks (so sessions never diverge). But a session copied in
the brief cp era — or via the cp fallback — can exist as multiple REAL files in
different project dirs, each a fork with possibly-unique turns. This tool makes the
live state coherent WITHOUT losing anything:

  * picks the most-recently-modified copy as canonical (it stays in place)
  * ARCHIVES every other copy to <archive>/superseded/<id>@<dir>-<ts>.jsonl
    (preserved for manual review — forks are never silently deleted)
  * replaces each archived location with a symlink to the canonical file, so
    resume from those dirs still works and stays in sync afterwards

Same invariant as the rest of the tool: preserve, never destroy. A fork may hold turns that
exist nowhere else, so it is copied into <archive>/superseded/ before its location is
replaced by a symlink. Nothing here touches registry.db — re-run backfill afterwards (the
script prints the command) so the rows match the files again.

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

# The same tree the claude adapter indexes (honours $CLAUDE_CONFIG_DIR).
from sources.registry import _make_claude  # noqa: E402
PROJECTS = _make_claude().projects_dir


def _real_copies() -> dict[str, list[Path]]:
    """Group transcripts by session id, keeping only ids that exist as SEVERAL real files.

    The layout walked here is ~/.claude/projects/<slug>/<session-id>.jsonl, so `f.stem` IS
    the session id. Symlinks are skipped because they are the healthy case — a symlink and
    its target are one file, not a fork. An id with a single real file is dropped by the
    final comprehension; what remains is exactly the set that needs repair.

    Returns {session_id: [path, ...]} and never raises: a missing projects directory (Claude
    Code not installed, or $CLAUDE_CONFIG_DIR pointing elsewhere) yields an empty mapping.
    """
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
    """Print the repair plan (default) or carry it out (--apply).

    With --apply, for each diverged session: copy every non-canonical fork to
    <archive>/superseded/<id>@<dir>-<timestamp>.jsonl, delete the fork in place, and put a
    symlink to the canonical file there instead. Touches the filesystem only — the registry
    is refreshed by the backfill command printed at the end.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry-run)")
    args = ap.parse_args()

    dupes = _real_copies()
    if not dupes:
        print("✓ No diverged copies — every session has a single canonical transcript.")
        return

    superseded = sbconfig.REASONING_ARCHIVE / "superseded"
    # One UTC stamp for the whole run, e.g. 20260911T081204Z, so every fork saved by this
    # invocation shares a suffix and a later run can never overwrite an earlier one.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"{len(dupes)} session(s) with diverged real copies "
          f"({'APPLYING' if args.apply else 'dry-run — pass --apply to act'}):\n")

    for sid, files in dupes.items():
        # Newest modification time wins: the copy you last resumed from has the most turns,
        # so it is the one to keep in place. Everything after index 0 is a fork.
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
                # Order matters: copy FIRST, then unlink. If the copy fails (disk full,
                # unwritable archive) the exception leaves the fork untouched.
                shutil.copy2(fork, superseded / tag)   # preserve the fork
                fork.unlink()
                # resolve() so the link stores an absolute target: the two directories are
                # siblings under projects/, and a relative link would be fragile.
                os.symlink(canonical.resolve(), fork)   # relink to canonical
        print()

    if args.apply:
        print(f"Done. Forks preserved under {superseded}")
        print("Re-run backfill to refresh the index:  scripts/backfill.py --source claude")
    else:
        print("Nothing changed. Re-run with --apply to reconcile.")


if __name__ == "__main__":
    main()
