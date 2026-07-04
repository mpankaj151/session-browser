#!/usr/bin/env python3
"""One-shot backfill: discover all sessions from enabled sources and upsert
header metadata into registry.db. Batched in a single connection, committing
every N rows. Safe to re-run (COALESCE upsert preserves enrichment).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# Ensure schema exists first.
import importlib.util  # noqa: E402
_mig_path = Path(__file__).resolve().parent / "migrate-db.py"
_spec = importlib.util.spec_from_file_location("migrate_db", _mig_path)
_mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mig)  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="only backfill this source")
    ap.add_argument("--commit-every", type=int, default=200)
    args = ap.parse_args()

    sbconfig.ensure_dirs()
    _mig.main()  # idempotent migrate

    registry = build_source_registry(only_available=True)
    if args.source:
        registry = {k: v for k, v in registry.items() if k == args.source}
    if not registry:
        print("No available sources to backfill.")
        return

    conn = indexer.connect()
    total = 0
    for name, adapter in registry.items():
        files = list(adapter.discover())
        print(f"[{name}] {len(files)} files")
        ok = err = skipped = 0
        t0 = time.time()
        for i, path in enumerate(files, 1):
            try:
                header = adapter.parse_header(path)
                if header is None:
                    skipped += 1  # e.g. headless sdk-cli sessions — intentionally not indexed
                    continue
                # Seed summary from a free title when there's no summary yet.
                indexer.upsert(header, conn=conn)
                # Note: we intentionally do NOT seed `summary` from the title — the
                # title is its own column and the UI falls back to it, so leaving
                # summary NULL lets nightly enrichment populate a real summary.
                ok += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  ! {path.name}: {e}", file=sys.stderr)
            if i % args.commit_every == 0:
                conn.commit()
                print(f"  ...{i}/{len(files)}")
        conn.commit()
        total += ok
        print(f"[{name}] indexed {ok}, skipped {skipped}, errors {err}, {time.time()-t0:.1f}s")
    conn.close()
    print(f"Backfill complete: {total} sessions.")


if __name__ == "__main__":
    main()
