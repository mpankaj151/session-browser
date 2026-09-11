#!/usr/bin/env python3
"""One-shot backfill: discover all sessions from enabled sources and upsert
header metadata into registry.db. Batched in a single connection, committing
every N rows. Safe to re-run (COALESCE upsert preserves enrichment).

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI; its
*transcript* is the file the CLI wrote while it happened; *indexing* means parsing the
transcript's cheap header facts (id, working directory, first and last timestamp, turn
count, first message, model) and writing them as one row; *enrichment* is the separate,
later pass that asks a model for a summary.

Where it sits in the pipeline (docs/ARCHITECTURE.md): the hook indexes ONE session the
instant it ends and the watcher daemon reacts to file changes, but neither can see sessions
that already existed when you installed this tool, or ones written while the machine was
asleep. Backfill is the sweep that walks EVERY transcript of every enabled source. It runs
at install time, nightly as the second step of refresh-all.py, and on demand:

    .venv/bin/python scripts/backfill.py [--source claude] [--commit-every 200]

Reads: each adapter's transcript tree (~/.claude/projects/**.jsonl, ~/.codex/sessions/**,
~/.copilot/session-state/**, and the OpenCode mirror). Writes: rows in registry.db.
Prints a per-source tally and a total; it never deletes or archives anything — that is
prune-sessions.py's job.

Concurrency note, and the reason index_source() looks the way it does: the registry is
SQLite in WAL mode, which allows many readers but only ONE writer at a time. A Claude Stop
hook can fire at any moment during a nightly backfill, and its writer waits at most
`busy_timeout` (5 s) before failing with "database is locked". So the expensive part
(parsing transcripts) is kept strictly OUTSIDE the write transaction, and the transaction
itself is a short burst of inserts followed by an immediate commit.
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
# Loaded by file path rather than `import migrate_db`: the file name contains a hyphen, so
# it is not a legal Python module name. indexer.py and the tests load it the same way.
import importlib.util  # noqa: E402
_mig_path = Path(__file__).resolve().parent / "migrate-db.py"
_spec = importlib.util.spec_from_file_location("migrate_db", _mig_path)
_mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mig)  # type: ignore


def main() -> None:
    """Index every transcript of every enabled, available source into registry.db.

    `--source NAME` narrows the run to one adapter (claude / copilot / codex / opencode);
    `--commit-every N` is the write-burst size (see index_source). Exits 0 with a message
    when no source is available — a laptop with none of these CLIs installed is a supported
    state, not an error (see tests/test_portability.py, which boots one-CLI machines).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="only backfill this source")
    ap.add_argument("--commit-every", type=int, default=200)
    args = ap.parse_args()

    sbconfig.ensure_dirs()
    _mig.main()  # idempotent migrate

    # only_available=True: an adapter is "available" when there are transcripts on disk for
    # it, NOT when its binary is on PATH — the whole point is to keep browsing a CLI's
    # history after the CLI is gone.
    registry = build_source_registry(only_available=True)
    if args.source:
        registry = {k: v for k, v in registry.items() if k == args.source}
    if not registry:
        print("No available sources to backfill.")
        return

    # One connection reused across sources: each index_source() opens and closes its own
    # short write transactions inside it.
    conn = indexer.connect()
    total = 0
    for name, adapter in registry.items():
        total += index_source(conn, name, adapter, args.commit_every)
    conn.close()
    print(f"Backfill complete: {total} sessions.")


def index_source(conn, name: str, adapter, commit_every: int = 200) -> int:
    """Index every transcript of one source. Headers are parsed OUTSIDE the
    write transaction and written in one short burst: parsing inside it held
    the SQLite write lock for the whole batch (200 multi-MB transcripts ≈ 5+ s),
    and the Stop hook's extract-reasoning died with 'database is locked'.

    Returns the number of sessions successfully indexed. Prints a progress line every
    `commit_every` files and a one-line summary. Exceptions from a single transcript are
    caught and counted: one unreadable file must not abort the other few thousand.

    `conn` is passed in (the caller owns it) and, in tests, is a temporary database —
    tests/test_smoke.py drives this function with a deliberately slow fake adapter and
    asserts that another writer can still get through mid-run.
    """
    files = list(adapter.discover())
    print(f"[{name}] {len(files)} files")
    ok = err = skipped = 0
    t0 = time.time()
    pending = []

    def flush():
        """Write the queued headers and commit — the ONLY place the write lock is taken.

        Kept deliberately tiny: N inserts with no parsing in between, then commit, so the
        lock is held for milliseconds instead of the minutes a full parse would take.
        """
        for h in pending:
            indexer.upsert(h, conn=conn)
        conn.commit()
        pending.clear()
    for i, path in enumerate(files, 1):
        try:
            # Parsing happens HERE, outside flush() — i.e. outside the write transaction.
            header = adapter.parse_header(path)
            if header is None:
                skipped += 1  # e.g. headless sdk-cli sessions — intentionally not indexed
                continue
            # Note: we intentionally do NOT seed `summary` from the title — the
            # title is its own column and the UI falls back to it, so leaving
            # summary NULL lets nightly enrichment populate a real summary.
            pending.append(header)
            ok += 1
        except Exception as e:  # noqa: BLE001
            # Counted and reported, never re-raised: a truncated JSONL or a transcript in a
            # shape no adapter expects must not take the whole sweep down with it.
            err += 1
            print(f"  ! {path.name}: {e}", file=sys.stderr)
        if len(pending) >= commit_every:
            flush()
            print(f"  ...{i}/{len(files)}")
    flush()
    print(f"[{name}] indexed {ok}, skipped {skipped}, errors {err}, {time.time()-t0:.1f}s")
    return ok


if __name__ == "__main__":
    main()
