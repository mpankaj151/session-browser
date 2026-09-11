#!/usr/bin/env python3
"""Archive registry rows that no live transcript maps to any more.

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI and one
row in registry.db; its *transcript* is the file the CLI wrote while it happened; a
*subagent sidechain* is a helper conversation a CLI starts for a sub-task — a child of a
session, never a session in its own right.

Housekeeping pass, run by hand (or after tightening an adapter's rules), NOT part of the
nightly refresh:

    .venv/bin/python scripts/prune-sessions.py            # dry run: prints a plan
    .venv/bin/python scripts/prune-sessions.py --apply    # actually archives

Reads every adapter's transcript tree plus registry.db; writes only the archived flag and
archived_reason on rows it decides are dead.

The predicate is adapter-agnostic and derived, never a hardcoded id pattern: build
the set of session ids every enabled adapter currently claims (discover() +
session_id_for_path()), then archive any non-archived row not in that set.

Two things produce such rows:

  * A path that used to pass an adapter's session_id_for_path() gate and no longer
    does. Multi-agent runs write sidechain transcripts to
    <project>/<session>/subagents/**, and the claude adapter accepted those until
    the gate was tightened — one permanently-empty row per subagent, plus every
    workflow journal.jsonl colliding onto a single row keyed "journal".
  * A transcript deleted while the watcher was down, so its archive-on-delete
    event was never seen.

Rows are archived (archived=1), never deleted — same invariant as indexer.archive():
history is preserved, and a later upsert from a real file resurrects the row.

Archive, never delete, for two reasons. A row carries work the transcript no longer does —
tokens, cost, summary, topics, the rendered reasoning trail — so deleting it would silently
change your usage history. And the decision "this file is gone" is only ever as good as the
adapter that made it: if a CLI home moved, or a disk was not mounted, the next backfill
sees the file again and the row comes straight back to life.

Each archived row must say WHY, because "the transcript aged out" and "this never was a
conversation" look identical from the flag alone. The rule is derived from the row's own
content and is shared with the watcher and with migrate-db.py's one-off backfill — it lives
in indexer.infer_archive_reason(): zero turns AND no first message (and no tokens, cost,
model, summary or trail proving work happened) means sidechain noise, filed as
`not-a-session` and hidden for good; anything else was a real session, filed as
`transcript-missing`, which keeps it in the UI's Archived tab, counted in usage stats and
restorable from the raw vault.

Safe by default: prints a plan and changes nothing unless you pass --apply.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402


def _live_ids(registry) -> set[str]:
    """Every session id currently backed by a real transcript on disk.

    Uses `session_id_for_path()` — the adapter's identity-without-reading method — so this
    costs one directory walk, not a parse of every transcript. That method is also the gate
    that rejects subagent sidechain files, which is exactly what makes a tightened gate show
    up here as "these rows no longer map to anything".
    """
    live: set[str] = set()
    for adapter in registry.values():
        for path in adapter.discover():
            sid = adapter.session_id_for_path(path)
            if sid:
                live.add(sid)
    return live


def archive_stale(rows, conn) -> collections.Counter:
    """Archive each stale row with the reason its own content implies: a row
    that never held a conversation is sidechain noise (hidden for good); one
    with turns is a real session whose transcript went missing (still shown in
    the UI's Archived view and counted in usage stats).

    `rows` must be WHOLE rows (`SELECT *`), not a narrow projection: the shared rule also
    weighs tokens, cost, model, summary and reasoning trail. Returns a Counter of reason ->
    how many, which the caller prints. Does not commit — the caller owns the transaction.

    tests/test_smoke.py calls this directly with one empty row and one 9-turn row and
    asserts the two get different reasons, i.e. that the batch is never blanket-labelled.
    """
    reasons: collections.Counter = collections.Counter()
    for r in rows:
        reason = indexer.infer_archive_reason(r)
        indexer.archive(r["session_id"], reason, conn=conn)
        reasons[reason] += 1
    return reasons


def main() -> None:
    """Print the plan (default) or archive the stale rows (--apply).

    Dry-run is the default on purpose: this pass decides, from the state of a filesystem it
    cannot verify, that sessions are dead. Reading the plan first is how you catch an
    unmounted disk or a mis-set CLI home before it archives a year of history.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry-run)")
    ap.add_argument("--limit", type=int, default=15, help="sample rows to print per source")
    args = ap.parse_args()

    registry = build_source_registry(only_available=True)
    if not registry:
        # Hard stop, not an empty run: with no adapters the "live" set is empty and EVERY
        # row would look dead. Better to do nothing than to archive the whole registry
        # because a CLI home was mis-set or a volume was not mounted.
        print("No available sources — refusing to prune (every row would look stale).")
        return

    live = _live_ids(registry)
    print(f"{len(live)} live transcript(s) across {', '.join(registry)}\n")

    conn = indexer.connect()
    try:
        # Whole rows: infer_archive_reason weighs tokens/model/summary/trail too.
        # `indexer.LIVE` is the named SQL predicate for "this row's transcript still
        # exists" — the archived flag is clear. The literal SQL lives only in indexer.py
        # (and the schema migration); a test greps the tree to keep it that way, so that a
        # change to what "live" means can never be half-applied across the code base.
        rows = conn.execute(f"SELECT * FROM sessions WHERE {indexer.LIVE}").fetchall()

        # Only prune rows belonging to a source we can actually see right now; a
        # disabled/unavailable CLI must never have its history archived wholesale.
        stale = [r for r in rows if r["cli_source"] in registry and r["session_id"] not in live]
        if not stale:
            print(f"✓ All {len(rows)} active row(s) are backed by a live transcript.")
            return

        # Grouped by CLI so the plan reads as "[claude] 38, [codex] 2" before the samples.
        by_source = collections.Counter(r["cli_source"] for r in stale)
        print(f"{len(stale)} of {len(rows)} active row(s) have no live transcript "
              f"({'APPLYING' if args.apply else 'dry-run — pass --apply to act'}):\n")
        for src, n in by_source.most_common():
            print(f"  [{src}] {n}")
            sample = [r for r in stale if r["cli_source"] == src][: args.limit]
            for r in sample:
                print(f"      {r['session_id'][:40]:42} {str(r['folder_name'] or '-')[:28]:30} "
                      f"turns={r['turn_count'] or 0:<4} {r['last_activity'] or '-'}  "
                      f"-> {indexer.infer_archive_reason(r)}")
            if n > len(sample):
                print(f"      ... and {n - len(sample)} more")
            print()

        if args.apply:
            reasons = archive_stale(stale, conn)
            conn.commit()
            detail = ", ".join(f"{n} {k}" for k, n in reasons.most_common())
            print(f"Done. Archived {len(stale)} row(s): {detail} (archived=1 — nothing deleted).")
        else:
            print("Nothing changed. Re-run with --apply to archive these rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
