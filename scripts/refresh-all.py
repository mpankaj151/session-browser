#!/usr/bin/env python3
"""Run the whole processing pipeline so every session is fully up to date.

Order: migrate -> backfill -> classify-topics -> compute-costs -> reasoning
-> full-text -> embeddings -> (optional) enrich. Each step is idempotent and
isolated, so one failing step doesn't abort the rest. Run nightly (launchd) or
on demand: `refresh-all.py [--enrich]`.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python")
SCRIPTS = REPO / "scripts"


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    try:
        subprocess.run([PY, *args], cwd=str(REPO), check=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {label} failed: {e}", file=sys.stderr)


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

    for label, a in steps:
        run(label, a)
    print("\nrefresh-all complete.")


if __name__ == "__main__":
    main()
