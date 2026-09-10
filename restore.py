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
        if dest.exists():
            _reindex(adapter, dest, conn)
            if own:
                conn.commit()
            return RestoreResult(session_id, "already-live", path=dest)
        src = reasoning.find_archived_raw(session_id)
        if src is None:
            return RestoreResult(session_id, "no-raw-copy",
                                 detail=f"no raw copy under {reasoning.ARCHIVE / 'raw'}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # copyfile, not copy2: the restored file must carry a FRESH mtime, or an
        # age-based cleanup would delete it again on its next pass. The archive
        # copy is left in place — it is the durable vault, not a staging area.
        shutil.copyfile(src, dest)
        _reindex(adapter, dest, conn)
        if own:
            conn.commit()
        return RestoreResult(session_id, "restored", path=dest,
                             detail=f"copied {src.name} from the reasoning archive")
    finally:
        if own:
            conn.close()


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
