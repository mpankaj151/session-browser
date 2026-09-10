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
    session_id: str
    # restored | already-live | not-found | not-a-session | no-raw-copy | unsupported
    status: str
    path: Path | None = None
    detail: str = ""
    # Sources whose CLI keeps sessions elsewhere (OpenCode: a database) also
    # re-import after the file copy: True/False = it ran; None = not applicable.
    reimported: bool | None = None


def _dest_for(row, registry) -> Path | None:
    adapter = registry.get(row["cli_source"])
    hook = getattr(adapter, "restore_path", None)
    return hook(row) if callable(hook) else None


def _reindex(adapter, path: Path, conn: sqlite3.Connection) -> None:
    header = adapter.parse_header(path)
    if header is not None:
        indexer.upsert(header, conn=conn)  # sets archived=0, clears the reason


def restore_session(session_id: str, conn: sqlite3.Connection | None = None,
                    registry: dict | None = None) -> RestoreResult:
    own = conn is None
    conn = conn or indexer.connect()
    registry = registry if registry is not None else build_source_registry()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return RestoreResult(session_id, "not-found")
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
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Sources with a watcher-side "deleted upstream" rule (OpenCode) mark the
        # file as deliberately restored BEFORE it appears, or a sync in between
        # archives it straight back.
        protect = getattr(adapter, "protect", None)
        if callable(protect):
            protect(dest)
        # copyfile, not copy2: the restored file must carry a FRESH mtime, or an
        # age-based cleanup would delete it again on its next pass. The archive
        # copy is left in place — it is the durable vault, not a staging area.
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
    UI's Restore button must say the same thing restore_session() will."""
    return _dest_for(row, registry) is not None


def plan(conn: sqlite3.Connection | None = None, registry: dict | None = None) -> list[dict]:
    """Every aged-out row and whether it can come back. Run this FIRST on a
    machine that lost sessions — it says what's recoverable before anyone
    counts on it. Noise rows (not-a-session) never appear."""
    own = conn is None
    conn = conn or indexer.connect()
    registry = registry if registry is not None else build_source_registry()
    try:
        rows = conn.execute(
            f"SELECT * FROM sessions WHERE {indexer.ARCHIVED_VISIBLE} ORDER BY last_activity DESC"
        ).fetchall()
    finally:
        if own:
            conn.close()
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
