"""Bring an aged-out session back from the reasoning archive.

Claude Code deletes transcripts older than `cleanupPeriodDays` (default 30).
The watcher sees the unlink and archives the row as transcript-missing — but
refresh-all had already copied every indexable transcript into
<archive>/raw/YYYY/MM/<session_id>.jsonl (reasoning.archive_raw). Restore puts
the newest such copy back where the CLI's own resume looks for it, re-indexes
it, and the row goes live again.

Shared by the UI's Restore button and scripts/restore-session.py. Every way a
restore can't proceed is a distinct RestoreResult.status, never a silent no-op,
and no refusal path writes anything.

The lifecycle in plain words (docs/GLOSSARY.md, "Lifecycle of a row"):

  live      the CLI's own transcript file still exists where the CLI keeps it, so the
            session can be reopened with `claude --resume <id>` / `cr <id>`.
  archived  the transcript is gone but the registry row survives, with archived_reason
            saying why — `transcript-missing` (the CLI deleted an old file; recoverable)
            or `not-a-session` (a subagent side-conversation that never was a session;
            nothing to recover).
  restored  this module copied the newest raw-vault copy back to the path the CLI looks
            at, re-indexed it, and the row is live again.

Rows are never deleted, which is the whole point: the row keeps the title, summary,
topics, token counts and reasoning trail even while the transcript is missing, so a
"lost" session is still searchable and, usually, bringable back.
"""
from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import indexer
import reasoning
from sources.registry import build_source_registry


@dataclass
class RestoreResult:
    """The outcome of one restore attempt — what happened, and enough detail to say so.

    `status` is the machine-readable verdict; every caller (the UI's Restore button, the
    CLI script) branches on it and shows `detail` as prose:
      restored      the raw copy was put back and the row re-indexed (the happy path)
      already-live  the transcript was already where the CLI expects it; nothing copied,
                    the row was re-indexed anyway so it leaves the Archived view
      not-found     no registry row with that session id
      not-a-session a subagent sidechain / journal row; there is nothing to bring back
      no-raw-copy   the row is genuinely aged out, but the raw vault has no copy of it
                    (indexed after the transcript had already been deleted)
      unsupported   this row's CLI has no way to accept a transcript back — decided per
                    ROW, see supported_for()
    `path` is the live transcript path for the two success statuses, None otherwise.
    """
    session_id: str
    # restored | already-live | not-found | not-a-session | no-raw-copy | unsupported
    status: str
    path: Path | None = None
    detail: str = ""
    # Sources whose CLI keeps sessions elsewhere (OpenCode: a database) also
    # re-import after the file copy: True/False = it ran; None = not applicable.
    # The three-way value matters: None means "this CLI needs no re-import step, the file
    # on disk IS the session", while False means a re-import was attempted and failed —
    # the row is live and browsable here, but the CLI itself may not be able to resume it
    # until the user retries. Reporting False as None would hide a real half-restore.
    reimported: bool | None = None


def _dest_for(row, registry) -> Path | None:
    """Where this row's transcript would have to live for its CLI to see it, or None.

    Asks the row's adapter for its optional `restore_path(row)` hook. None means "this
    restore cannot happen", for either of two quite different reasons, and the caller
    treats both as unsupported:
      * the adapter has no such hook at all (a source that was never taught to accept a
        transcript back), or
      * the hook looked at the row and refused — typically because the path the row
        recorded lies outside the tree this adapter owns on THIS machine, which is what a
        registry copied over from another laptop looks like.
    Pure: it computes a path, it never creates anything.
    """
    adapter = registry.get(row["cli_source"])
    hook = getattr(adapter, "restore_path", None)
    return hook(row) if callable(hook) else None


def _reindex(adapter, path: Path, conn: sqlite3.Connection) -> None:
    """Re-read the restored transcript's header and upsert it, flipping the row live.

    The upsert is the same one the hook and watcher use, so it clears the archived flag
    and the reason, refreshes cheap fields (turn count, last activity) and — thanks to
    the COALESCE upsert — preserves every enriched column the row still carries. A header
    that cannot be parsed is skipped silently: the file is back on disk either way, and a
    later indexing pass will pick it up.
    """
    header = adapter.parse_header(path)
    if header is not None:
        indexer.upsert(header, conn=conn)  # clears the archived flag and the reason


def restore_session(session_id: str, conn: sqlite3.Connection | None = None,
                    registry: dict | None = None) -> RestoreResult:
    """Bring one archived session back to life; report exactly what happened.

    Steps: look the row up, refuse the cases that cannot work, find the newest raw copy
    in <archive>/raw/, copy it to the path the row's adapter nominates (creating the
    parent directory), let the adapter re-import it if it needs to, and re-index.

    Side effects on the success paths only: one file written under the CLI's own
    transcript tree, one row updated in the registry, and possibly a subprocess/DB write
    by the adapter's reimport hook. Every refusal path returns without writing anything —
    tests assert that a refused restore does not even create the projects directory.

    `conn` / `registry` are injectable for tests and for callers that already hold them
    (the Flask app). With conn=None a connection is opened, committed and closed here.
    Never raises for the ordinary failure modes; they come back as a status.
    """
    own = conn is None
    conn = conn or indexer.connect()
    registry = registry if registry is not None else build_source_registry()
    try:
        # SELECT * because the adapter's restore_path() hook reads whichever columns it
        # needs off the row (project path, source, id) — this module does not know which.
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return RestoreResult(session_id, "not-found")
        # A row archived as "not a session" is a subagent side-conversation or a workflow
        # journal — it never was a session, so there is no transcript to put back and no
        # vault copy to find. Saying so beats letting it fall through to "no-raw-copy",
        # which would read as "your data is lost".
        if row["archived"] and row["archived_reason"] == indexer.NOT_A_SESSION:
            return RestoreResult(session_id, "not-a-session",
                                 detail="this row never held a conversation; nothing to restore")
        dest = _dest_for(row, registry)
        if dest is None:
            return RestoreResult(session_id, "unsupported",
                                 detail=f"restore is not supported for {row['cli_source']} sessions")
        adapter = registry[row["cli_source"]]
        src = reasoning.find_archived_raw(session_id)
        # A compressed raw copy (Codex cold rollout) stays compressed: the
        # adapter reads .jsonl.zst, and zstd bytes under a .jsonl name would be
        # unreadable — the very corruption archive_raw now avoids. Decide the
        # real destination BEFORE the already-live check so the .zst twin of a
        # plain name counts as live and is never overwritten.
        if src is not None and src.name.endswith(".zst") and not dest.name.endswith(".zst"):
            dest = dest.with_name(dest.name + ".zst")
        # "Is it already there?" checks BOTH spellings: the plain name and its .zst twin.
        # Codex compresses cold rollouts in place, so <id>.jsonl.zst existing means the
        # session is live even though <id>.jsonl does not exist — without this a restore
        # would overwrite a perfectly good compressed transcript with an older copy.
        live = dest if dest.exists() else next(
            (t for t in (dest.with_name(dest.name + ".zst"),) if t.exists()), None)
        if live is not None:
            _reindex(adapter, live, conn)
            if own:
                conn.commit()
            result = RestoreResult(session_id, "already-live", path=live)
            # A previous restore may have put the file back while the CLI-side
            # re-import failed; offer it again so the UI can retry.
            _reimport(adapter, live, result)
            return result
        if src is None:
            return RestoreResult(session_id, "no-raw-copy",
                                 detail=f"no raw copy under {reasoning.ARCHIVE / 'raw'}")
        # The CLI may have removed the whole per-project directory along with the file.
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Sources with a watcher-side "deleted upstream" rule (OpenCode) mark the
        # file as deliberately restored BEFORE it appears, or a sync in between
        # archives it straight back.
        # Concretely: OpenCode's adapter treats "a mirror file with no matching row in
        # OpenCode's database" as proof the session was deleted upstream, and archives it
        # again. protect() drops a small sidecar marker next to the destination saying
        # "this reappearance is deliberate", so a sync racing us in the next second reads
        # the marker instead of undoing the restore. It must be written BEFORE the copy —
        # the race starts the moment the file exists.
        protect = getattr(adapter, "protect", None)
        if callable(protect):
            protect(dest)
        # copyfile, not copy2: the restored file must carry a FRESH mtime, or an
        # age-based cleanup would delete it again on its next pass. The archive
        # copy is left in place — it is the durable vault, not a staging area.
        # shutil.copy2 copies the metadata too, including the modification time. The
        # archive copy carries the ORIGINAL timestamp — that is what made it age out in
        # the first place — so copy2 would restore a file that Claude Code's
        # cleanupPeriodDays sweep considers 40 days old and deletes again within the day.
        # copyfile copies bytes only, so the restored file is stamped "now".
        shutil.copyfile(src, dest)
        _reindex(adapter, dest, conn)
        if own:
            conn.commit()
        result = RestoreResult(session_id, "restored", path=dest,
                               detail=f"copied {src.name} from the reasoning archive")
        # The row is live and browsable regardless of what follows; the
        # re-import only decides whether the CLI itself can resume it.
        _reimport(adapter, dest, result)
        return result
    finally:
        if own:
            conn.close()


def _reimport(adapter, path: Path, result: RestoreResult) -> None:
    """Give the adapter a chance to tell its CLI about the file we just put back.

    For Claude, Copilot and Codex the transcript file IS the session — dropping it in
    place is the whole restore, those adapters have no `reimport` hook, and
    `result.reimported` stays None ("not applicable"). OpenCode is different: it keeps
    sessions in its own SQLite database and the mirror file is only a projection, so its
    adapter re-inserts the rows and reports (ok, detail).

    Failure is recorded, never raised: the row is already live and browsable here, and
    only the CLI-side resume depends on the re-import. A raising hook is caught for the
    same reason — one adapter's bad day must not turn a successful file restore into a
    traceback. `detail` is appended to whatever the caller already wrote so the user sees
    both "copied X from the archive" and why resume may still not work.
    """
    hook = getattr(adapter, "reimport", None)
    if not callable(hook):
        return
    try:
        ok, detail = hook(path)
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"re-import raised: {e}"
    result.reimported = ok
    if detail:
        result.detail = f"{result.detail}; {detail}" if result.detail else detail


def supported_for(row, registry) -> bool:
    """Per ROW, not per source: an adapter refuses rows whose recorded path is
    outside its tree (a registry carried over from another machine), and the
    UI's Restore button must say the same thing restore_session() will.

    "Can Codex sessions be restored?" is the wrong question — the right one is "can THIS
    Codex row be restored on THIS machine?". A registry synced from another laptop holds
    rows whose recorded transcript path points at a directory that does not exist here;
    the adapter checks that and declines, and the UI then shows an honest "unsupported"
    marker instead of a Restore button that would fail with a 409 when clicked.
    """
    return _dest_for(row, registry) is not None


def plan(conn: sqlite3.Connection | None = None, registry: dict | None = None) -> list[dict]:
    """Every aged-out row and whether it can come back. Run this FIRST on a
    machine that lost sessions — it says what's recoverable before anyone
    counts on it. Noise rows (not-a-session) never appear.

    Read-only: nothing is copied, nothing is written. Returns one dict per row with
    `supported` (this machine's adapter would accept the file back) and `restorable`
    (supported AND a raw copy exists) plus the fields a listing needs — id, source,
    folder, title, last activity, when it was archived, and the raw copy's path.

    Backs `restore-session.py --all` (a dry run by default) and the UI's Archived tab.
    """
    own = conn is None
    conn = conn or indexer.connect()
    registry = registry if registry is not None else build_source_registry()
    try:
        # indexer.ARCHIVED_VISIBLE is the shared predicate for "archived because the
        # transcript went missing" — i.e. real conversations the user lost, excluding the
        # subagent-noise rows. Consumers never spell the flag out themselves; a test
        # greps the tree to keep it that way. Newest first, which is the order both the
        # CLI listing and the Archived tab want.
        rows = conn.execute(
            f"SELECT * FROM sessions WHERE {indexer.ARCHIVED_VISIBLE} ORDER BY last_activity DESC"
        ).fetchall()
    finally:
        if own:
            conn.close()
    # One directory walk of the whole raw vault, not one per row: 500 archived rows used
    # to mean 500 walks of the same tree.
    raw = reasoning.archived_raw_index()
    out = []
    for r in rows:
        sid = r["session_id"]
        supported = _dest_for(r, registry) is not None
        out.append({
            "session_id": sid, "cli_source": r["cli_source"], "folder_name": r["folder_name"],
            "title": r["title"] or r["first_message"], "last_activity": r["last_activity"],
            "archived_at": r["archived_at"], "supported": supported,
            "restorable": supported and sid in raw,
            "raw_path": str(raw[sid]) if sid in raw else None,
        })
    return out
