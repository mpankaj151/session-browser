#!/usr/bin/env python3
"""Run the whole processing pipeline so every session is fully up to date.

Order: migrate -> backfill -> classify-topics -> compute-costs -> reasoning
-> full-text -> embeddings -> (optional) enrich. Each step is idempotent and
isolated, so one failing step doesn't abort the rest — but failures are
COLLECTED and reported, and the run exits nonzero if any step failed.
Run nightly (launchd) or on demand: `refresh-all.py [--enrich]`.

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI;
*indexing* copies a transcript's cheap header facts into registry.db; *enrichment* asks a
model, through a non-interactive CLI run, to summarise a finished session; a *token* is the
unit models read and write text in, and what usage is billed by.

This is the batch half of the system. The hook and the watcher keep the registry current
minute to minute; this script does everything that is too slow or too expensive for that,
once a night, driven by a launchd job on macOS or a systemd --user timer on Linux (both
rendered by scripts/render-job.py). It runs each step as a SEPARATE PROCESS rather than
importing it, so one step crashing — or leaking memory, or wedging on a network call —
cannot take the others with it.

WHY THE ORDER IS WHAT IT IS. Each step consumes what an earlier one produced:

  1. migrate schema      — must be first: every later step writes columns that may not
                           exist yet on a registry that predates this build.
  2. backfill sessions   — indexes every transcript, so the rows below exist. Everything
                           downstream works from rows, not from files.
  3. classify topics     — keyword tags for rows enrichment has not reached (or never will,
                           on a machine with no summariser CLI).
  4. compute costs       — sums tokens per session and prices them; needs the rows from (2)
                           and reads the transcripts again itself.
  5. reasoning trails    — renders the readable decision trail AND, with --archive, copies
                           every indexable transcript byte-for-byte into the raw vault.
                           That copy is what makes a session restorable after its CLI
                           deletes the original, so it must happen before anything that
                           depends on the raw text still being on disk.
  6. opencode db snapshot— a whole-database copy of OpenCode's store, at most weekly.
  7. full-text index     — builds the word index from transcript text, including sessions
                           whose original file is gone but whose raw copy (5) survives.
  8. enrichment          — the only LLM step, only with --enrich; needs rows (2) and is the
                           slowest and the only one that spends the user's CLI quota.
  9. embeddings          — vectors are built from title/summary/first_message, so running
                           them BEFORE enrichment left semantic search a night behind.
 10. daily digest        — assembles the day's Markdown page from what (8) wrote. No model
                           call, so it is never gated on --enrich.

STEPS THAT MAY EXIT 0 WITHOUT DOING ANYTHING. "This machine does not have that" is a
supported state, not a failure — a missing piece must not mark the whole nightly run as
broken (tests/test_portability.py boots one-CLI laptops and asserts exit 0):
  * backfill exits 0 with "No available sources" when no CLI's transcripts are present.
  * opencode db snapshot exits 0 when the OpenCode source is disabled, its database is
    missing, or the snapshot is not due yet (--if-due 7).
  * embeddings exit 0 on a --lite install with no sentence-transformers, and when the
    embedding model is not cached locally and downloads are not allowed. Only an ATTEMPTED
    download that then fails is a real error.
  * enrichment exits 0 when `[enrichment].provider = "auto"` finds no summariser CLI.
  * migrate prints a warning and continues when FTS5 or sqlite-vec is unavailable.

AGGREGATE EXIT CODE: every step runs regardless of what came before; failures are collected
and printed at the end, and the process exits 1 if ANY step failed, 0 otherwise. That single
number is what launchd/systemd and install.sh read — one broken step must be visible, but
must not stop the other nine from bringing the registry up to date.
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
    """Run one pipeline step to completion and report (label, exit code).

    The child inherits this process's stdout/stderr, so its output lands in the same log
    the job writes — that is the only record of a nightly run. `check=False` because a
    nonzero exit is data here, not an exception: the caller decides what to do with it.
    A step that cannot even be launched (file missing after a bad upgrade) is reported as
    -1 rather than raising, so the remaining steps still run.
    """
    print(f"\n=== {label} ===", flush=True)
    try:
        proc = subprocess.run([PY, *args], cwd=str(REPO), check=False)
        return label, proc.returncode
    except Exception as e:  # noqa: BLE001
        print(f"  ! {label} failed to launch: {e}", file=sys.stderr)
        return label, -1


def steps(enrich: bool) -> list[tuple[str, list[str]]]:
    """The nightly pipeline, in dependency order.

    Returns (label, argv-without-the-interpreter) pairs. Built as data rather than inlined
    so the order is readable in one place and the tests can inspect it. `enrich` inserts the
    single LLM step; see the module docstring for why each step sits where it does.
    """
    out = [
        ("migrate schema", [str(SCRIPTS / "migrate-db.py")]),
        ("backfill sessions", [str(SCRIPTS / "backfill.py")]),
        ("classify topics", [str(SCRIPTS / "classify-topics.py")]),
        ("compute costs", [str(SCRIPTS / "compute-costs.py")]),
        # --backfill: every session, not just one. --archive: also copy each transcript
        # into the raw vault, which is what makes sessions restorable once a CLI's own
        # cleanup deletes the original file.
        ("reasoning trails", [str(SCRIPTS / "extract-reasoning.py"), "--backfill", "--archive"]),
        # weekly whole-DB copy of OpenCode's store; exits 0 when disabled / not due
        ("opencode db snapshot", [str(SCRIPTS / "backup-opencode.py"), "--if-due", "7"]),
        ("full-text index", [str(SCRIPTS / "build-fts.py")]),
    ]
    # Opt-in: this is the only step that calls a model, so it is the only one that costs
    # the user quota (and minutes). The scheduled nightly job passes --enrich; a manual
    # `refresh-all.py` does not.
    if enrich:
        out.append(("enrichment", [str(SCRIPTS / "enrich-sessions.py")]))
    # After enrichment: embeddings are built from title/summary/first_message,
    # so running them earlier left semantic search one night behind.
    out.append(("embeddings", [str(SCRIPTS / "embed-sessions.py")]))
    # Last so it sees tonight's enrichment; no LLM cost, so never gated on --enrich.
    out.append(("daily digest", [str(SCRIPTS / "daily-digest.py")]))
    return out


def main() -> None:
    """Run every step in order, then exit 1 if any of them failed.

    Never stops early: a failure is recorded and the loop continues, because the steps are
    independent enough that nine successes are worth having even when one breaks.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true", help="also run LLM enrichment (uses quota)")
    args = ap.parse_args()

    failed: list[tuple[str, int]] = []
    for label, a in steps(args.enrich):
        label, rc = run(label, a)
        if rc != 0:
            failed.append((label, rc))

    # The summary goes to stderr and the exit code is 1, so a scheduled run's failure is
    # visible in the job log rather than buried in a hundred lines of progress output.
    if failed:
        print("\nrefresh-all finished WITH FAILURES:", file=sys.stderr)
        for label, rc in failed:
            print(f"  ✗ {label} (exit {rc})", file=sys.stderr)
        sys.exit(1)
    print("\nrefresh-all complete.")


if __name__ == "__main__":
    main()
