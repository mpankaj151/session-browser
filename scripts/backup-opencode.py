#!/usr/bin/env python3
"""Weekly whole-database snapshots of OpenCode's opencode.db.

    backup-opencode.py [--if-due DAYS] [--keep N] [--db PATH] [--out DIR]

`VACUUM INTO` from a read-only connection writes a consistent single-file copy
(WAL contents included) without touching the source; snapshots rotate to
`keep`. Belt-and-braces beside the per-session mirror: the mirror is what
Restore uses, the snapshot is for a corrupted or migrated-away DB. Exits 0 when
the source is disabled, the DB is missing, or nothing is due — refresh-all
runs it nightly with --if-due 7.
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

_STAMP = "%Y%m%dT%H%M%SZ"
_NAME = re.compile(r"^opencode-(\d{8}T\d{6}Z)\.db$")


def _stamp_of(path: Path) -> datetime | None:
    m = _NAME.match(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), _STAMP).replace(tzinfo=timezone.utc)


def snapshots(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    return sorted((p for p in out_dir.glob("opencode-*.db") if _stamp_of(p)), key=lambda p: _stamp_of(p))


def is_due(out_dir: Path, days: int, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    have = snapshots(out_dir)
    return not have or now - _stamp_of(have[-1]) >= timedelta(days=days)


def rotate(out_dir: Path, keep: int) -> list[Path]:
    removed = []
    have = snapshots(out_dir)
    for old in have[:max(0, len(have) - keep)]:
        old.unlink()
        removed.append(old)
    return removed


def snapshot(db: Path, out_dir: Path, keep: int = 8, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"opencode-{now.strftime(_STAMP)}.db"
    if dest.exists():
        dest.unlink()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        try:
            conn.execute("VACUUM INTO ?", (str(dest),))
        except sqlite3.OperationalError:
            # older SQLite without VACUUM INTO: the backup API, same guarantee
            copy = sqlite3.connect(str(dest))
            with copy:
                conn.backup(copy)
            copy.close()
    finally:
        conn.close()
    rotate(out_dir, keep)
    return dest


def main() -> None:
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
    from sources.registry import build_source_registry
    adapter = build_source_registry().get("opencode")
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
    dest = snapshot(db, out, keep)
    print(f"snapshot {dest} ({dest.stat().st_size // 1024} KB); keeping {keep} in {out}")


if __name__ == "__main__":
    main()
