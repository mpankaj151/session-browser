#!/usr/bin/env python3
"""Driver for reasoning extraction.

Command-line front end for reasoning.py: pick transcripts, run the right per-CLI
extractor over each, write the Markdown trail under the reasoning archive and record it
in the registry. A "reasoning trail" is the readable reconstruction of how the assistant
reached its decisions in a session — see reasoning.py's docstring and docs/GLOSSARY.md.

Modes (exactly one of the first three is required):
  --session PATH        process one transcript file
  --session-id ID       process by session_id (looked up in registry.db)
  --backfill            process every discovered Claude session
  --archive             also copy the raw transcript into the archive
  --source NAME         which CLI adapter to use for --session / --session-id;
                        one of claude (default) | copilot | codex | opencode. It decides
                        both how to parse the header and which extractor runs, so
                        pointing --session at a Codex rollout without --source codex
                        simply produces no trail.

`--backfill` ignores --source and sweeps every available adapter instead.

Exit codes: 0 on success, including "this session had no reasoning steps" (a user-only or
aborted session is a normal outcome, not a failure). 1 only for a genuinely wrong
invocation — an unknown --source, or a --session-id that is not in the registry. Argparse
itself exits 2 when no mode is given. This matters because the nightly refresh treats a
nonzero step as a failed run, and because the Claude Stop hook spawns this detached: a
crash here must never look like a broken session end.

Designed to be safe to run detached from the Stop hook (idempotent, skip-if-fresh).
Idempotent in practice: archive_raw() re-copies nothing when the transcript has not
grown, write_readable() replaces the trail atomically, and persist() deletes this
session's previous artifact rows before inserting the new ones.

Pipeline position: run by the nightly `refresh-all` (with --backfill --archive, which is
what fills the raw vault restore.py depends on) and by the Stop hook for the session that
just ended. Writes to ~/claude-reasoning-archive/{raw,readable}/ and to registry.db.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

# Scripts live one level below the repo root and are run directly (not as a package), so
# put the repo root on the import path before importing any of its modules. The
# `# noqa: E402` markers below exist only because those imports must follow this line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import sbconfig  # noqa: E402
from sources.claude import ClaudeSource  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# Adapter name -> the reasoning extractor that understands that CLI's transcript format.
# Keyed on `adapter.name`, so a source with no entry here (a future CLI whose reasoning
# format nobody has taught us) is simply skipped rather than mis-parsed. All four
# extractors return the same list[ReasoningStep].
_EXTRACTORS = {"claude": reasoning.extract, "copilot": reasoning.extract_copilot,
               "codex": reasoning.extract_codex, "opencode": reasoning.extract_opencode}


def _adapter_for(name: str):
    """The adapter for --session/--session-id (was hardwired to Claude).

    Claude is special-cased so the common path works even when Claude's transcript
    directory is momentarily unavailable; every other name comes from the source
    registry. An unknown name exits 1 with a message listing the valid ones — better than
    silently producing no trail and reporting success.
    """
    if name == "claude":
        return ClaudeSource()
    adapter = build_source_registry().get(name)
    if adapter is None:
        sys.exit(f"unknown --source {name!r}; one of {sorted(_EXTRACTORS)}")
    return adapter


def _header_dict(adapter, path: Path) -> dict | None:
    """The adapter's parsed header as a plain dict, or None if the file isn't parseable.

    reasoning.py's renderer and archive helpers take a dict (they are shared by callers
    that never built a SessionHeader), so the dataclass is flattened here with asdict().
    """
    h = adapter.parse_header(path)
    if h is None:
        return None
    return asdict(h)


def process_one(adapter, path: Path, do_archive: bool, conn=None, force=False) -> bool:
    """Extract, archive and persist one transcript. True if a trail was written.

    Returns False — not an error — when the file has no parseable header, when this
    adapter has no extractor, or when the session contained no reasoning steps at all.
    The raw copy is still made in that last case if --archive was asked for; see below.

    Side effects: a raw copy under <archive>/raw/, a Markdown trail under
    <archive>/readable/, and the registry rows persist() writes. `conn` lets the backfill
    reuse one connection across thousands of sessions. `force` is accepted for symmetry
    with other drivers and currently unused (see _is_fresh).
    """
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
    # This ordering is what makes the raw vault complete, and completeness is what
    # restore.py depends on: a session that produced no trail is still a session the
    # user may want back after the CLI's own cleanup deletes it.
    # REASONING_ENABLED is the config kill switch ([reasoning].enabled): with the archive
    # turned off nothing is copied anywhere.
    if do_archive and sbconfig.REASONING_ENABLED:
        reasoning.archive_raw(path, header)
    if not steps:
        return False
    readable = reasoning.write_readable(steps, header)
    reasoning.persist(sid, steps, readable, conn=conn)
    return True


def _is_fresh(path: Path, header_last: str) -> bool:
    """Reserved skip-if-unchanged hook; always False, so every pass re-renders.

    Kept as a named seam rather than deleted: re-rendering is cheap next to parsing the
    transcript, and the Stop hook's whole job is to refresh the session that just ended,
    so a freshness check would have to be right about mtimes to be worth anything.
    """
    return False  # placeholder; the hook always re-renders the just-finished session


def main() -> None:
    """Parse the flags, pick the mode, and print one line per processed session."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="transcript file path")
    ap.add_argument("--session-id", help="session_id to look up")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--source", default="claude", help="adapter for --session/--session-id")
    args = ap.parse_args()

    # Create ~/.session-browser and the archive directories if this is a first run.
    sbconfig.ensure_dirs()
    adapter = _adapter_for(args.source)

    if args.session:
        ok = process_one(adapter, Path(args.session), args.archive)
        print("reasoning:", "ok" if ok else "no-steps", args.session)
        return

    if args.session_id:
        conn = indexer.connect()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (args.session_id,)
        ).fetchone()
        conn.close()
        if not row:
            print("unknown session_id", file=sys.stderr)
            sys.exit(1)
        # restore_path() doubles as "where this row's transcript lives"
        # — the adapter already knows how to turn a row into a path, and that is exactly
        # the question here. The fallback covers adapters without the hook and assumes
        # the Claude convention of <project_path>/<session_id>.jsonl.
        locate = getattr(adapter, "restore_path", None)
        path = (locate(row) if callable(locate) else None) or \
            Path(row["project_path"]) / f"{args.session_id}.jsonl"
        ok = process_one(adapter, path, args.archive)
        print("reasoning:", "ok" if ok else "no-steps", args.session_id)
        return

    if args.backfill:
        conn = indexer.connect()
        # only_available=True: adapters whose transcript directory does not exist on this
        # machine are left out entirely, so a laptop with just one CLI installed does not
        # spend the pass reporting three kinds of absence.
        registry = build_source_registry(only_available=True)
        total = done = 0
        for name, adp in registry.items():
            if name not in _EXTRACTORS:
                continue
            files = list(adp.discover())
            total += len(files)
            print(f"[{name}] {len(files)} files")
            for path in files:
                try:
                    if process_one(adp, path, args.archive, conn=conn):
                        done += 1
                except Exception as e:  # noqa: BLE001
                    # One unreadable or unsafe transcript reports itself and the sweep
                    # carries on — a backfill over thousands of files must not be
                    # aborted by a single bad one. The line goes to stderr so the
                    # nightly job's error log shows it.
                    print(f"  ! {path.name}: {e}", file=sys.stderr)
                # Commit per file: each iteration parses a whole transcript and
                # copies it into the vault, so batching 20 held the write lock
                # for tens of seconds and blocked the Stop hook's own upsert.
                conn.commit()
            conn.commit()
        conn.close()
        print(f"Reasoning backfill complete: {done}/{total} sessions had reasoning.")
        return

    ap.error("one of --session / --session-id / --backfill is required")


if __name__ == "__main__":
    main()
