#!/usr/bin/env python3
"""Weekly whole-database snapshots of OpenCode's opencode.db.

    backup-opencode.py [--if-due DAYS] [--keep N] [--db PATH] [--out DIR]

`VACUUM INTO` from a read-only connection writes a consistent single-file copy
(WAL contents included) without touching the source; snapshots rotate to
`keep`. Belt-and-braces beside the per-session mirror: the mirror is what
Restore uses, the snapshot is for a corrupted or migrated-away DB. Exits 0 when
the source is disabled, the DB is missing, or nothing is due — refresh-all
runs it nightly with --if-due 7.

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI. Unlike
Claude Code, Copilot and Codex, OpenCode does not write one transcript file per session —
it keeps every session in a single SQLite database, opencode.db.

Why `VACUUM INTO` from a read-only connection, and not `cp`:
  * OpenCode may be running and writing while this executes. SQLite in WAL mode keeps
    recent writes in a side file (opencode.db-wal), so copying just opencode.db with `cp`
    can produce a file that is missing the latest transactions, or is torn mid-write and
    will not open at all.
  * `VACUUM INTO '<path>'` asks SQLite itself to write a fresh, complete, defragmented
    database containing a consistent snapshot — WAL contents folded in — without changing
    the source in any way.
  * The connection is opened with `mode=ro` (`file:<path>?mode=ro`) as a hard guarantee
    that this tool can never modify the database another program owns.

Rotation: snapshots are named opencode-<UTC timestamp>.db in one directory and the oldest
are unlinked so at most `keep` (default 8) remain — otherwise a weekly whole-database copy
would grow without bound. `--if-due DAYS` makes the nightly invocation cheap: it looks at
the newest snapshot's timestamp and returns immediately unless it is older than DAYS.

This is belt-and-braces beside the per-session mirror: Restore works from the mirror, the
snapshot is for the day opencode.db is corrupted or migrated away.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sbconfig  # noqa: E402

# The timestamp format used in snapshot file names, e.g. 20260911T081204Z. Sorts correctly
# as plain text, and has no characters that need quoting in a shell.
_STAMP = "%Y%m%dT%H%M%SZ"
# Matches exactly the names this script writes: "opencode-", 8 digits, "T", 6 digits, "Z",
# ".db" — e.g. "opencode-20260911T081204Z.db". Anchored at both ends so an unrelated file
# in the same directory (opencode-notes.db, a half-written opencode-x.db.tmp) is ignored
# rather than mistaken for a snapshot and rotated away.
_NAME = re.compile(r"^opencode-(\d{8}T\d{6}Z)\.db$")


def _stamp_of(path: Path) -> datetime | None:
    """The UTC time encoded in a snapshot's file name, or None if it is not one of ours."""
    m = _NAME.match(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), _STAMP).replace(tzinfo=timezone.utc)


def snapshots(out_dir: Path) -> list[Path]:
    """Existing snapshots, oldest first. Empty list when the directory does not exist yet.

    Sorted by the time in the NAME rather than by mtime, so a `cp -p`, a restore from
    backup, or a filesystem that rewrites timestamps cannot reorder the history.
    """
    if not out_dir.is_dir():
        return []
    return sorted((p for p in out_dir.glob("opencode-*.db") if _stamp_of(p)), key=lambda p: _stamp_of(p))


def is_due(out_dir: Path, days: int, now: datetime | None = None) -> bool:
    """True when there is no snapshot yet, or the newest is at least `days` days old.

    `now` is injectable so tests can walk a calendar; production passes nothing.
    """
    now = now or datetime.now(timezone.utc)
    have = snapshots(out_dir)
    return not have or now - _stamp_of(have[-1]) >= timedelta(days=days)


def rotate(out_dir: Path, keep: int) -> list[Path]:
    """Delete the oldest snapshots until at most `keep` remain. Returns what was removed.

    `snapshots()` is oldest-first, so the slice takes exactly the excess from the front and
    the newest `keep` survive. `max(0, ...)` keeps the slice empty when there is nothing to
    remove (a negative bound would silently delete from the wrong end).
    """
    removed = []
    have = snapshots(out_dir)
    for old in have[:max(0, len(have) - keep)]:
        old.unlink()
        removed.append(old)
    return removed


def snapshot(db: Path, out_dir: Path, keep: int = 8, now: datetime | None = None) -> Path:
    """Write one consistent copy of `db` into `out_dir` and rotate. Returns the new path.

    Raises sqlite3.OperationalError when OpenCode holds the database locked — deliberately
    NOT swallowed here, so the caller can decide (main() treats it as "try again tomorrow",
    the tests assert it is reported). On ANY failure the half-written destination is
    removed; see the comment on the except block for why that matters.
    """
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"opencode-{now.strftime(_STAMP)}.db"
    if dest.exists():
        dest.unlink()
    # mode=ro + uri=True: a genuinely read-only handle on a database another program owns.
    # timeout=10 gives SQLite ten seconds to wait out a brief lock before giving up.
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        try:
            conn.execute("VACUUM INTO ?", (str(dest),))
        except sqlite3.OperationalError as e:
            # Two very different failures arrive as the same exception type, so they are
            # told apart by the message: a lock means "OpenCode is mid-write, nothing is
            # wrong" and must propagate untouched; anything else is most likely a SQLite
            # older than 3.27, which has no VACUUM INTO at all.
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise                      # OpenCode is writing: try again next run
            # older SQLite without VACUUM INTO: the backup API, same guarantee
            # conn.backup() copies page by page and restarts if the source changes, so the
            # result is just as consistent — it is only slower and less compact.
            copy = sqlite3.connect(str(dest))
            try:
                with copy:
                    conn.backup(copy)
            finally:
                copy.close()
    except BaseException:
        # Never leave a stub: a 0-byte opencode-<stamp>.db would satisfy
        # is_due() for another week and pass as "the backup".
        # BaseException so an interrupted run cleans up too. `missing_ok` because the
        # failure may have happened before the file was ever created.
        dest.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
    # Only after a successful write: rotating first could delete the last good snapshot
    # just before failing to produce a new one.
    rotate(out_dir, keep)
    return dest


def main() -> None:
    """Resolve config, decide whether a snapshot is due, take one, and report.

    Returns quietly (exit 0) in every "nothing to do" case — the OpenCode source disabled,
    no database at the resolved path, not due yet, or the database locked right now. A
    nightly job must not report a failure for a machine that simply does not use OpenCode.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-due", type=int, default=0, metavar="DAYS",
                    help="only snapshot if the newest one is older than DAYS (0 = always)")
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--db", help="override the resolved opencode.db path")
    ap.add_argument("--out", help="override the snapshot directory")
    args = ap.parse_args()

    cfg = sbconfig.source_config("opencode")
    if not cfg.get("enabled"):
        print("opencode source disabled — no snapshot")
        return
    # Imported here rather than at module scope: building the registry touches config and
    # the filesystem, and there is no reason to pay that when the source is disabled above.
    from sources.registry import build_source_registry
    adapter = build_source_registry().get("opencode")
    # The adapter resolves the database location the same way OpenCode itself does
    # ($OPENCODE_DB, else $XDG_DATA_HOME/opencode/opencode.db), so a shell-overridden home
    # and a background job agree on which file to snapshot. --db overrides for testing.
    db = Path(args.db) if args.db else (adapter.db_path if adapter else None)
    if db is None or not db.exists():
        print(f"no OpenCode DB at {db} — no snapshot")
        return
    out = Path(args.out) if args.out else Path(
        cfg.get("db_snapshot_dir", str(sbconfig.REASONING_ARCHIVE / "opencode-db"))).expanduser()
    keep = args.keep if args.keep is not None else int(cfg.get("db_snapshot_keep", 8))
    if args.if_due and not is_due(out, args.if_due):
        print(f"snapshot not due (newest < {args.if_due} days old) — {out}")
        return
    try:
        dest = snapshot(db, out, keep)
    except sqlite3.OperationalError as e:
        # A lock is normal (the user has OpenCode open) and must not make the nightly
        # pipeline report a failure; is_due() will still be true tomorrow, so nothing is
        # lost. Any other database error is real and propagates to a nonzero exit.
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            print(f"snapshot skipped: {e} (OpenCode is writing; next run retries)")
            return
        raise
    print(f"snapshot {dest} ({dest.stat().st_size // 1024} KB); keeping {keep} in {out}")


if __name__ == "__main__":
    main()
