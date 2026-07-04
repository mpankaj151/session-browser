#!/usr/bin/env python3
"""Run the whole processing pipeline so every session is fully up to date.

Order: migrate -> backfill -> classify-topics -> compute-costs -> reasoning
-> full-text -> embeddings -> (optional) enrich. Each step is idempotent and
isolated, so one failing step doesn't abort the rest — but failures are
COLLECTED and reported, and the run exits nonzero if any step failed.
Run nightly (launchd) or on demand: `refresh-all.py [--enrich]`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Whatever interpreter launched this script runs the steps too — never assume
# a .venv at a fixed path (conda/system/venv-elsewhere installs all exist).
PY = sys.executable
SCRIPTS = REPO / "scripts"


def run(label: str, args: list[str]) -> tuple[str, int]:
    print(f"\n=== {label} ===", flush=True)
    try:
        proc = subprocess.run([PY, *args], cwd=str(REPO), check=False)
        return label, proc.returncode
    except Exception as e:  # noqa: BLE001
        print(f"  ! {label} failed to launch: {e}", file=sys.stderr)
        return label, -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true", help="also run LLM enrichment (uses quota)")
    args = ap.parse_args()

    steps = [
        ("migrate schema", [str(SCRIPTS / "migrate-db.py")]),
        ("backfill sessions", [str(SCRIPTS / "backfill.py")]),
        ("classify topics", [str(SCRIPTS / "classify-topics.py")]),
        ("compute costs", [str(SCRIPTS / "compute-costs.py")]),
        ("reasoning trails", [str(SCRIPTS / "extract-reasoning.py"), "--backfill", "--archive"]),
        ("full-text index", [str(SCRIPTS / "build-fts.py")]),
        ("embeddings", [str(SCRIPTS / "embed-sessions.py")]),
    ]
    if args.enrich:
        steps.append(("enrichment", [str(SCRIPTS / "enrich-sessions.py")]))

    failed: list[tuple[str, int]] = []
    for label, a in steps:
        label, rc = run(label, a)
        if rc != 0:
            failed.append((label, rc))

    if failed:
        print("\nrefresh-all finished WITH FAILURES:", file=sys.stderr)
        for label, rc in failed:
            print(f"  ✗ {label} (exit {rc})", file=sys.stderr)
        sys.exit(1)
    print("\nrefresh-all complete.")


if __name__ == "__main__":
    main()
