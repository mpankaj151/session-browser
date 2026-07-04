#!/usr/bin/env python3
"""Driver for reasoning extraction.

Modes:
  --session PATH        process one transcript file
  --session-id ID       process by session_id (looked up in registry.db)
  --backfill            process every discovered Claude session
  --archive             also copy the raw transcript into the archive

Designed to be safe to run detached from the Stop hook (idempotent, skip-if-fresh).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import sbconfig  # noqa: E402
from sources.claude import ClaudeSource  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

_EXTRACTORS = {"claude": reasoning.extract, "copilot": reasoning.extract_copilot,
               "codex": reasoning.extract_codex}


def _header_dict(adapter, path: Path) -> dict | None:
    h = adapter.parse_header(path)
    if h is None:
        return None
    return asdict(h)


def process_one(adapter, path: Path, do_archive: bool, conn=None, force=False) -> bool:
    header = _header_dict(adapter, path)
    if header is None:
        return False
    sid = header["session_id"]
    extractor = _EXTRACTORS.get(adapter.name)
    if extractor is None:
        return False
    steps = extractor(path)
    # Archive the raw transcript FIRST — even a session with no reasoning steps
    # (user-only, aborted) deserves its durable raw copy when --archive was asked.
    if do_archive and sbconfig.REASONING_ENABLED:
        reasoning.archive_raw(path, header)
    if not steps:
        return False
    readable = reasoning.write_readable(steps, header)
    reasoning.persist(sid, steps, readable, conn=conn)
    return True


def _is_fresh(path: Path, header_last: str) -> bool:
    return False  # placeholder; the hook always re-renders the just-finished session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="transcript file path")
    ap.add_argument("--session-id", help="session_id to look up")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--archive", action="store_true")
    args = ap.parse_args()

    sbconfig.ensure_dirs()
    adapter = ClaudeSource()

    if args.session:
        ok = process_one(adapter, Path(args.session), args.archive)
        print("reasoning:", "ok" if ok else "no-steps", args.session)
        return

    if args.session_id:
        conn = indexer.connect()
        row = conn.execute(
            "SELECT project_path FROM sessions WHERE session_id = ?", (args.session_id,)
        ).fetchone()
        conn.close()
        if not row:
            print("unknown session_id", file=sys.stderr)
            sys.exit(1)
        path = Path(row["project_path"]) / f"{args.session_id}.jsonl"
        ok = process_one(adapter, path, args.archive)
        print("reasoning:", "ok" if ok else "no-steps", args.session_id)
        return

    if args.backfill:
        conn = indexer.connect()
        registry = build_source_registry(only_available=True)
        total = done = 0
        for name, adp in registry.items():
            if name not in _EXTRACTORS:
                continue
            files = list(adp.discover())
            total += len(files)
            print(f"[{name}] {len(files)} files")
            for i, path in enumerate(files, 1):
                try:
                    if process_one(adp, path, args.archive, conn=conn):
                        done += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {path.name}: {e}", file=sys.stderr)
                if i % 20 == 0:
                    conn.commit()
            conn.commit()
        conn.close()
        print(f"Reasoning backfill complete: {done}/{total} sessions had reasoning.")
        return

    ap.error("one of --session / --session-id / --backfill is required")


if __name__ == "__main__":
    main()
