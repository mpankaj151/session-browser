"""OpenCode source adapter — a SQLite database projected to one JSONL per session.

Since v1.2.0 OpenCode keeps every session in ONE SQLite DB (WAL mode):
  ~/.local/share/opencode/opencode.db     ($XDG_DATA_HOME/opencode; $OPENCODE_DB
                                           overrides; channel builds: opencode-<ch>.db)
  session(id, project_id, parent_id, slug, directory, path, title, version, ...
          time_created/time_updated — epoch ms)
  message(id, session_id, time_created, data JSON)      role, providerID/modelID,
                                                        cost, tokens, finish, error
  part(id, message_id, session_id, time_created, data JSON)   text / reasoning /
                                                        tool / step-* / patch / ...
The JSON `data` omits the promoted columns (id, sessionID, messageID).

Every other part of this repo assumes ONE plain-text file per session at the
path discover() yields: the watcher's delete handler, reasoning.archive_raw,
restore.py, build-fts, the cost and reasoning extractors. So this adapter
PROJECTS the DB into <mirror_dir>/<ses_id>.jsonl — one file per ROOT session,
sub-agent children embedded, their tokens/cost rolled up — and everything
downstream works unchanged. The mirror doubles as the backup: OpenCode never
auto-deletes, but `opencode session delete` hard-cascades, spilled tool
outputs are purged after 7 days, and the DB grows fast enough that people prune it.

    L1  {"type":"session","schema":1,"info":{...Session.Info...},
         "children":[...],"stats":{...},"source":{...}}
    L2+ {"type":"message","session":"ses_…","info":{...},"parts":[...]}

Never invoke the `opencode` binary on this path: `opencode export` truncates at
64 KiB on a pipe, merely running `opencode db path` opened the DB read-write
and checkpointed the WAL, and a newer binary would run migrations. The DB is
opened read-only (`mode=ro`, the WAL must be readable) and the path comes from
env/config only. Ids are NOT chronologically sortable (36-bit truncated clock)
— always order by time_created.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# Sidecar left beside a restored mirror file until OpenCode holds the session again.
_RESTORED = ".restored"


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


_XDG = os.environ.get("XDG_DATA_HOME")
DATA_DIR = (Path(_XDG) if _XDG else Path.home() / ".local" / "share") / "opencode"
MIRROR_DIR = Path(os.path.expanduser("~/.session-browser/opencode-mirror"))

SCHEMA = 1
_SID = re.compile(r"^ses_[A-Za-z0-9]{26}$")
_PLACEHOLDER_TITLE = re.compile(r"^(New session|Child session) - \d{4}-\d{2}-\d{2}T")
_SPILL_RE = re.compile(r"\S*tool-output/tool_[A-Za-z0-9]+")
# Part types this adapter understands — only to tell "no user turns" apart
# from "a schema I have never seen" (the codex lesson: never index nothing silently).
_KNOWN_PARTS = frozenset({
    "text", "reasoning", "tool", "step-start", "step-finish", "patch", "file",
    "subtask", "agent", "retry", "compaction", "snapshot",
})
_REQUIRED = {
    "session": {"id", "project_id", "parent_id", "slug", "directory", "title", "version", "time_created"},
    "message": {"id", "session_id", "time_created", "data"},
    "part": {"id", "message_id", "session_id", "time_created", "data"},
}
# Newest OpenCode DB migration (YYYYMMDD prefix) this projection was verified against.
_KNOWN_MIGRATION_DATE = "20260622"

# Our own headless enrichment runs carry this title (sbconfig.OPENCODE_ENRICHMENT_TITLE
# once the enrichment provider lands); they must never be indexed as sessions.
try:  # pragma: no cover - the constant may not exist on older branches
    import sbconfig as _sbconfig
    ENRICHMENT_TITLE = getattr(_sbconfig, "OPENCODE_ENRICHMENT_TITLE", "session-browser-enrichment")
except Exception:  # noqa: BLE001
    ENRICHMENT_TITLE = "session-browser-enrichment"

_WARNED: set[str] = set()


def _warn(msg: str) -> None:
    """Loud, once per distinct message per process."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[opencode] {msg}", file=sys.stderr, flush=True)


def _get(row: sqlite3.Row, col: str, default=None):
    return row[col] if col in row.keys() else default


def _loads(blob) -> dict:
    try:
        v = json.loads(blob) if isinstance(blob, (str, bytes)) else blob
    except (json.JSONDecodeError, TypeError):
        return {}
    return v if isinstance(v, dict) else {}


def _loads_any(blob):
    """A JSON column as-is — dict OR list (session.permission is a
    PermissionRuleset list; coercing it to {} made `opencode import` reject
    every re-import). None when empty or unparseable."""
    if isinstance(blob, (dict, list)):
        return blob or None
    if not isinstance(blob, (str, bytes)) or not blob:
        return None
    try:
        v = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, (dict, list)) and v else None


@dataclass
class SyncReport:
    written: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    db_missing: bool = False


class OpenCodeSource:
    name = "opencode"

    def __init__(self, data_dir: Path | str = DATA_DIR, mirror_dir: Path | str = MIRROR_DIR,
                 db: Path | str | None = None, reimport_on_restore: bool = True):
        # No disk access here: CI has no OpenCode, and the registry builds every adapter.
        self.data_dir = Path(os.path.expanduser(str(data_dir)))
        self.mirror_dir = Path(os.path.expanduser(str(mirror_dir)))
        self._db = Path(os.path.expanduser(str(db))) if db else None
        self.reimport_on_restore = reimport_on_restore
        self.sync_interval = 5.0     # discover() re-syncs at most this often per process
        self._last_sync = 0.0

    # -- database ----------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        """Resolved like OpenCode does, without asking the binary: $OPENCODE_DB
        (absolute, or relative to the data dir) > config `db` > opencode.db >
        the newest channel DB (opencode-<channel>.db)."""
        env = os.environ.get("OPENCODE_DB")
        if env and env != ":memory:":
            p = Path(os.path.expanduser(env))
            return p if p.is_absolute() else self.data_dir / p
        if self._db is not None:
            return self._db
        default = self.data_dir / "opencode.db"
        if default.exists() or not self.data_dir.is_dir():
            return default
        channel = sorted(self.data_dir.glob("opencode*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        return channel[0] if channel else default

    def _open_ro(self) -> sqlite3.Connection | None:
        db = self.db_path
        if not db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            return conn
        except sqlite3.Error as e:
            _warn(f"cannot open {db} read-only: {e}")
            return None

    def _schema_check(self, conn: sqlite3.Connection) -> list[str]:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        warnings: list[str] = []
        for table, cols in _REQUIRED.items():
            if table not in tables:
                warnings.append(f"table '{table}' is missing from {self.db_path} — nothing can be indexed")
                continue
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            missing = sorted(cols - have)
            if missing:
                warnings.append(f"table '{table}' lacks columns {missing} — projecting what is there")
        if "message" in tables and "session_message" in tables:
            n_v1 = conn.execute("SELECT COUNT(*) FROM message").fetchone()[0]
            n_v2 = conn.execute("SELECT COUNT(*) FROM session_message").fetchone()[0]
            if n_v1 == 0 and n_v2 > 0:
                warnings.append("message/part tables are empty but session_message has rows — OpenCode "
                                "has moved to its v2 session store; this adapter reads message/part "
                                "and needs updating (existing mirror files are still served)")
        newest = self._migration_fingerprint(conn)
        if newest[:8].isdigit() and newest[:8] > _KNOWN_MIGRATION_DATE:
            warnings.append(f"DB migration {newest} is newer than this adapter was verified against "
                            f"({_KNOWN_MIGRATION_DATE}); the projection may be incomplete")
        return warnings

    @staticmethod
    def _migration_fingerprint(conn: sqlite3.Connection) -> str:
        """Newest applied migration name, informational only — never raises.
        On 1.18.15 `migration(id, time_completed)` keeps the NAME in `id`;
        `__drizzle_migrations` has a `name` column; either may change shape."""
        for table in ("migration", "__drizzle_migrations"):
            try:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                col = next((c for c in ("name", "id") if c in cols), None)
                if col is None:
                    continue
                v = conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone()[0]
                if isinstance(v, str) and v[:8].isdigit():
                    return v
            except sqlite3.Error:
                continue
        return ""

    # -- tree / fingerprints -----------------------------------------------------
    @staticmethod
    def _tree(conn: sqlite3.Connection) -> tuple[dict, dict[str, list[str]]]:
        """by_id, {root_id: [descendant ids, breadth-first, by time_created]}.
        A child whose parent is gone is treated as a root; our own enrichment
        runs are skipped outright."""
        rows = conn.execute("SELECT * FROM session").fetchall()
        by_id = {r["id"]: r for r in rows}
        kids: dict[str, list[str]] = defaultdict(list)
        roots: list[str] = []
        for r in rows:
            if (_get(r, "title") or "") == ENRICHMENT_TITLE:
                continue
            pid = _get(r, "parent_id")
            if pid and pid in by_id:
                kids[pid].append(r["id"])
            else:
                roots.append(r["id"])
        for k in kids:
            kids[k].sort(key=lambda i: _get(by_id[i], "time_created") or 0)
        tree: dict[str, list[str]] = {}
        for root in roots:
            out: list[str] = []
            queue = [root]
            while queue:
                cur = queue.pop(0)
                out += kids.get(cur, [])
                queue += kids.get(cur, [])
            tree[root] = out
        return by_id, tree

    @staticmethod
    def _fingerprints(conn: sqlite3.Connection, by_id: dict, tree: dict) -> dict[str, list]:
        """[newest activity ms, message count] per root, spanning the whole tree —
        a child can update after its root, and old rows have NULL time_updated."""
        stats = {r[0]: (r[1], r[2] or 0) for r in conn.execute(
            "SELECT session_id, COUNT(*), MAX(time_created) FROM message GROUP BY session_id")}
        fps = {}
        for root, kids in tree.items():
            ids = [root, *kids]
            newest = max((_get(by_id[i], "time_updated") or _get(by_id[i], "time_created") or 0) for i in ids)
            newest = max(newest, *(stats.get(i, (0, 0))[1] for i in ids))
            fps[root] = [int(newest), sum(stats.get(i, (0, 0))[0] for i in ids)]
        return fps

    # -- projection --------------------------------------------------------------
    @staticmethod
    def _info(row: sqlite3.Row) -> dict:
        """Session.Info in the export's camelCase shape (what `opencode import` decodes)."""
        created = _get(row, "time_created")
        info = {
            "id": row["id"], "slug": _get(row, "slug") or "", "projectID": _get(row, "project_id"),
            "directory": _get(row, "directory") or "", "path": _get(row, "path"),
            "title": _get(row, "title") or "", "version": _get(row, "version") or "",
            "time": {"created": created, "updated": _get(row, "time_updated") or created},
        }
        if _get(row, "parent_id"):
            info["parentID"] = row["parent_id"]
        for col, key in (("agent", "agent"), ("workspace_id", "workspaceID")):
            if _get(row, col):
                info[key] = row[col]
        for col in ("model", "revert", "permission", "metadata"):
            v = _loads_any(_get(row, col))
            if v is not None:
                info[col] = v
        if _get(row, "share_url"):
            info["share"] = {"url": row["share_url"]}
        summary = {k: _get(row, f"summary_{k}") for k in ("additions", "deletions", "files")}
        if any(v is not None for v in summary.values()):
            info["summary"] = {k: v for k, v in summary.items() if v is not None}
            diffs = _loads_any(_get(row, "summary_diffs"))
            if diffs is not None:
                info["summary"]["diffs"] = diffs
        for col, key in (("time_archived", "archived"), ("time_compacting", "compacting")):
            if _get(row, col):
                info["time"][key] = row[col]
        if _get(row, "cost"):
            info["cost"] = row["cost"]
        tok = {k: _get(row, f"tokens_{k}") or 0 for k in ("input", "output", "reasoning", "cache_read", "cache_write")}
        if any(tok.values()):
            info["tokens"] = {"input": tok["input"], "output": tok["output"], "reasoning": tok["reasoning"],
                              "cache": {"read": tok["cache_read"], "write": tok["cache_write"]}}
        return {k: v for k, v in info.items() if v is not None}

    def _project_root(self, conn: sqlite3.Connection, root: str, kids: list[str],
                      by_id: dict, fingerprint: list, migration: str) -> tuple[dict, list[dict]]:
        lines: list[dict] = []
        models: dict[str, dict] = {}
        turn_count = 0
        first_message = ""
        message_count = recognised = 0
        for sid in [root, *kids]:
            msgs = conn.execute("SELECT id, time_created, data FROM message WHERE session_id = ? "
                                "ORDER BY time_created, id", (sid,)).fetchall()
            parts_by_msg: dict[str, list] = defaultdict(list)
            for pr in conn.execute("SELECT id, message_id, time_created, data FROM part WHERE session_id = ? "
                                   "ORDER BY time_created, id", (sid,)):
                parts_by_msg[pr["message_id"]].append(pr)
            for m in msgs:
                info = _loads(m["data"])
                info["id"], info["sessionID"] = m["id"], sid
                parts = []
                for pr in parts_by_msg.get(m["id"], []):
                    pd = _loads(pr["data"])
                    pd["id"], pd["messageID"], pd["sessionID"] = pr["id"], m["id"], sid
                    if pd.get("type") == "tool":
                        self._inline_spill(pd)
                    parts.append(pd)
                lines.append({"type": "message", "session": sid, "info": info, "parts": parts})
                message_count += 1
                recognised += sum(1 for pd in parts if pd.get("type") in _KNOWN_PARTS)
                role = info.get("role")
                if role == "user" and sid == root:
                    if any(pd.get("type") == "compaction" for pd in parts):
                        continue                      # OpenCode's own context summary, not a turn
                    text = _user_text(parts)
                    if text:
                        turn_count += 1
                        first_message = first_message or text
                elif role == "assistant":
                    key = _model_key(info)
                    if key:
                        tk = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
                        cache = tk.get("cache") if isinstance(tk.get("cache"), dict) else {}
                        acc = models.setdefault(key, {"input": 0, "output": 0, "reasoning": 0,
                                                      "cache_read": 0, "cache_write": 0, "cost": 0.0})
                        acc["input"] += int(tk.get("input") or 0)
                        acc["output"] += int(tk.get("output") or 0)
                        acc["reasoning"] += int(tk.get("reasoning") or 0)
                        acc["cache_read"] += int(cache.get("read") or 0)
                        acc["cache_write"] += int(cache.get("write") or 0)
                        acc["cost"] += float(info.get("cost") or 0)
        for acc in models.values():
            acc["cost"] = round(acc["cost"], 6)
        row = by_id[root]
        raw_title = _get(row, "title") or ""
        stats = {
            "turn_count": turn_count,
            "first_message": first_message[:500],
            "title": None if (not raw_title or _PLACEHOLDER_TITLE.match(raw_title)) else raw_title,
            "model_used": max(models, key=lambda k: (models[k]["output"], models[k]["input"])) if models else None,
            "models": models,
            "cost_usd": round(sum(a["cost"] for a in models.values()), 6),
            "start_time": _get(row, "time_created"),
            "last_activity": fingerprint[0],
            "message_count": message_count,
            "child_count": len(kids),
            "recognised_parts": recognised,
        }
        head = {"type": "session", "schema": SCHEMA, "info": self._info(row),
                "children": [self._info(by_id[k]) for k in kids], "stats": stats,
                "source": {"db": str(self.db_path), "migration": migration,
                           "mirrored_at": int(time.time() * 1000)}}
        return head, lines

    _SPILL_MAX = 16 * 1024 * 1024

    def _inline_spill(self, pd: dict) -> None:
        """Outputs over 2000 lines / 50 KB are spilled to <data>/tool-output/tool_<id>
        (state.metadata.outputPath; older builds only leave the path in the
        preview text) and purged after SEVEN days. The mirror is the backup, so
        inline the blob while it exists; a missing blob leaves the part as stored."""
        st = pd.get("state")
        if not isinstance(st, dict):
            return
        meta = st.get("metadata") if isinstance(st.get("metadata"), dict) else {}
        ref = meta.get("outputPath")
        if not ref:
            m = _SPILL_RE.search(st.get("output") or "")
            ref = m.group(0) if m else None
        if not ref:
            return
        blob = Path(ref)
        if not blob.is_absolute():
            blob = self.data_dir / "tool-output" / blob.name
        try:
            if not blob.is_file() or blob.stat().st_size > self._SPILL_MAX:
                return
            st["output"] = blob.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        meta["inlined"] = True
        st["metadata"] = meta

    @staticmethod
    def _write_atomic(path: Path, head: dict, lines: list[dict]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(head, separators=(",", ":")) + "\n")
            for line in lines:
                fh.write(json.dumps(line, separators=(",", ":")) + "\n")
        os.replace(tmp, path)

    # -- manifest ----------------------------------------------------------------
    @property
    def _manifest_path(self) -> Path:
        return self.mirror_dir / ".manifest.json"

    def _load_manifest(self) -> dict[str, list]:
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            roots = data.get("roots") if isinstance(data, dict) else None
            return dict(roots) if isinstance(roots, dict) else {}
        except (OSError, ValueError):
            return {}          # lost or corrupt: one full resync, no harm

    def _save_manifest(self, roots: dict[str, list]) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"schema": SCHEMA, "roots": roots}, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self._manifest_path)

    # -- sync --------------------------------------------------------------------
    def sync(self, only: list[str] | None = None, *, force: bool = False,
             on_delete=None) -> SyncReport:
        """Project every changed root session into the mirror. Read-only on the
        DB, never spawns a subprocess. An unreadable DB changes nothing."""
        report = SyncReport()
        conn = self._open_ro()
        if conn is None:
            report.db_missing = True
            return report
        try:
            for w in self._schema_check(conn):
                _warn(w)
                report.warnings.append(w)
            if any("is missing" in w or "v2 session store" in w for w in report.warnings):
                return report
            migration = self._migration_fingerprint(conn)
            try:
                by_id, tree = self._tree(conn)
                fps = self._fingerprints(conn, by_id, tree)
            except sqlite3.Error as e:
                # An odd DB must never take the whole backfill (every source) down.
                msg = f"cannot read sessions from {self.db_path}: {e}"
                _warn(msg)
                report.warnings.append(msg)
                return report
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._load_manifest()
            for root, kids in tree.items():
                if only is not None and root not in only:
                    continue
                path = self.mirror_dir / f"{root}.jsonl"
                if not force and path.exists() and manifest.get(root) == fps[root]:
                    report.skipped += 1
                    continue
                try:
                    head, lines = self._project_root(conn, root, kids, by_id, fps[root], migration)
                    self._write_atomic(path, head, lines)
                    manifest[root] = fps[root]
                    report.written.append(root)
                except Exception as e:  # noqa: BLE001 — one bad session must not stop the rest
                    msg = f"projection of {root} failed: {e}"
                    _warn(msg)
                    report.warnings.append(msg)
            if only is None:
                self._remove_deleted(set(tree), manifest, on_delete or self._archive_before_delete, report)
            self._save_manifest(manifest)
        finally:
            conn.close()
        return report

    # -- deletions ---------------------------------------------------------------
    @staticmethod
    def _archive_before_delete(path: Path, header: dict) -> None:
        """Default archiver: the versioned raw vault (<archive>/raw/YYYY/MM/<sid>.jsonl),
        the same copy refresh-all makes nightly and Restore reads back."""
        import reasoning
        import sbconfig
        if sbconfig.REASONING_ENABLED:
            reasoning.archive_raw(path, header)

    def _remove_deleted(self, live_roots: set[str], manifest: dict, on_delete, report: SyncReport) -> None:
        """A mirror file whose root is gone from the DB is the LAST copy of that
        session: hand it to the archiver first, unlink only if that succeeded.
        The unlink is what the watcher turns into archived=transcript-missing."""
        from dataclasses import asdict
        # A restore that could not (yet) re-import leaves <file>.restored beside
        # the mirror file: OpenCode does not have that session, but the user
        # deliberately put it back — deleting it here would un-restore it on
        # the next WAL write. The marker goes once the session is in the DB.
        for marker in self.mirror_dir.glob("ses_*.jsonl" + _RESTORED):
            if marker.name[:-len(".jsonl" + _RESTORED)] in live_roots:
                marker.unlink(missing_ok=True)
        for path in sorted(self.mirror_dir.glob("ses_*.jsonl")):
            sid = path.stem
            if not _SID.match(sid) or sid in live_roots:
                continue
            if path.with_name(path.name + _RESTORED).exists():
                report.skipped += 1
                continue
            header = self.parse_header(path)
            hdr = asdict(header) if header is not None else {"session_id": sid, "last_activity": ""}
            try:
                on_delete(path, hdr)
            except Exception as e:  # noqa: BLE001 — keep the file, say why
                msg = f"{sid}: deleted in OpenCode but archiving the mirror failed ({e}); keeping it"
                _warn(msg)
                report.warnings.append(msg)
                continue
            try:
                path.unlink()
            except OSError as e:
                report.warnings.append(f"{sid}: could not unlink mirror file: {e}")
                continue
            manifest.pop(sid, None)
            report.removed.append(sid)

    # -- identity ----------------------------------------------------------------
    def session_id_for_path(self, path: Path) -> Optional[str]:
        """The stem, for exactly <mirror>/ses_<26>.jsonl — never the manifest, a
        half-written .tmp, the DB/WAL, or an archive copy's ses_…@vN name. The
        file may already be gone (the watcher's delete handler)."""
        if path.suffix != ".jsonl":
            return None
        return path.stem if _SID.match(path.stem) else None

    # -- cheap header ------------------------------------------------------------
    @staticmethod
    def _read_head(path: Path) -> Optional[dict]:
        """Line 1 only — the whole point of putting the stats there."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                line = fh.readline()
        except OSError:
            return None
        head = _loads(line.strip())
        return head if head.get("type") == "session" and isinstance(head.get("info"), dict) else None

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        head = self._read_head(path)
        if head is None:
            return None
        info, st = head["info"], head.get("stats") or {}
        # From info, never the stem: archive copies are named <sid>@vN.jsonl.
        session_id = info.get("id") or path.stem
        turn_count = int(st.get("turn_count") or 0)
        first_message = st.get("first_message") or ""
        if turn_count == 0 and not first_message:
            if st.get("message_count") and not st.get("recognised_parts"):
                # Messages exist but no part type we know: a schema we have not
                # seen. Index it (0 turns) so the warning is actionable rather
                # than silently returning None — the codex outage lesson.
                _warn(f"{path.name}: messages carry no part type this adapter knows — "
                      f"indexing with 0 turns; OpenCode's part schema may have changed")
            else:
                return None            # no conversation at all (aborted / meta-only)
        cwd = info.get("directory") or ""
        return SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=str(path.parent),
            cwd=cwd,
            folder_name=Path(cwd).name if cwd else "",
            start_time=to_iso_utc(st.get("start_time") or (info.get("time") or {}).get("created")),
            last_activity=to_iso_utc(st.get("last_activity") or (info.get("time") or {}).get("updated")),
            first_message=first_message[:500],
            turn_count=turn_count,
            title=st.get("title") or None,
            model_used=st.get("model_used") or None,
            cli_version=info.get("version") or None,
        )

    # -- full parse ----------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Root turns only, in order. Children stay lossless in the file and
        surface as the `subtask` tool call that spawned them; the compaction
        pair (OpenCode's own context summary), synthetic/ignored text and
        part-less aborted messages are not turns. Works on an archive copy too:
        nothing here touches the DB, the manifest or mirror_dir."""
        header = self.parse_header(path)
        if header is None:
            return None
        root = header.session_id
        turns: list[Turn] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    rec = _loads(line.strip())
                    if rec.get("type") != "message" or rec.get("session") != root:
                        continue
                    info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
                    parts = [pd for pd in (rec.get("parts") or []) if isinstance(pd, dict)]
                    role = info.get("role")
                    if role == "user":
                        if any(pd.get("type") == "compaction" for pd in parts):
                            continue
                        text = _user_text(parts)
                        if text:
                            turns.append(Turn(role="user", content=text))
                    elif role == "assistant":
                        if info.get("summary"):
                            continue
                        text = _user_text(parts)
                        tools = _tool_calls(parts)
                        if text or tools:
                            turns.append(Turn(role="assistant", content=text, tool_calls=tools))
        except OSError:
            return None
        return ParsedSession(header=header, turns=turns)

    # -- resume / availability ---------------------------------------------------
    def resume_command(self, session_id: str) -> str:
        # A global lookup by id that runs in the CURRENT cwd — exactly `cr`'s meaning.
        return f"opencode --session {shlex.quote(session_id)}"

    def has_binary(self) -> bool:
        """Whether `opencode` is runnable from here — a UI hint for resume/re-import only."""
        return shutil.which("opencode") is not None

    def is_available(self) -> bool:
        """Something to read: the DB, or a mirror that keeps serving after
        OpenCode is gone. Never gated on the binary being on PATH."""
        if self.db_path.exists():
            return True
        return self.mirror_dir.is_dir() and any(self.mirror_dir.glob("ses_*.jsonl"))

    # -- watcher hooks -------------------------------------------------------------
    def watch_roots(self) -> list[Path]:
        """The mirror (session files → the ordinary handler) and the data dir
        (DB/WAL writes → sync_trigger). The mirror is our own directory, so it
        is created here: a watcher started before the first sync must be able
        to subscribe to it, or the files that sync writes go unseen."""
        try:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return [self.mirror_dir, self.data_dir]

    def sync_trigger(self, path: Path) -> bool:
        """A write to opencode*.db or its WAL means sessions changed. Never the
        -shm, log/, tool-output/, auth.json — those churn constantly."""
        name = path.name
        return name.startswith("opencode") and (name.endswith(".db") or name.endswith(".db-wal"))

    # -- restore -------------------------------------------------------------------
    def restore_path(self, row) -> Optional[Path]:
        """Where this row's mirror file lives; restoring the raw copy here makes
        the row live again. Contained to the mirror dir so a corrupted row can
        never make restore write elsewhere."""
        dest = self.mirror_dir / f"{row['session_id']}.jsonl"
        try:
            dest.resolve().relative_to(self.mirror_dir.resolve())
        except ValueError:
            return None
        return dest

    def reimport(self, path: Path) -> tuple[bool | None, str]:
        """Second half of a restore: put the session back into OpenCode itself
        with `opencode import`, root first, then each child. Import re-homes a
        session to the directory it runs in, so it runs in the session's own
        directory ($HOME when that is gone). Idempotent upstream, but sessions
        still present are skipped. None = disabled by config."""
        # Mark first: until OpenCode holds the session again, sync() must not
        # treat this mirror file as "deleted in OpenCode" (see _remove_deleted).
        marker = path.with_name(path.name + _RESTORED)
        try:
            marker.write_text(_now_iso(), encoding="utf-8")
        except OSError:
            pass
        ok, detail = self._reimport(path)
        if ok is True:
            marker.unlink(missing_ok=True)
        return ok, detail

    def _reimport(self, path: Path) -> tuple[bool | None, str]:
        if not self.reimport_on_restore:
            return None, "re-import into OpenCode disabled ([sources.opencode] reimport_on_restore = false)"
        docs = to_export_docs(path)
        if not docs:
            return False, "mirror file holds no session document"
        existing = self._existing_ids([d["info"]["id"] for d in docs])
        todo = [d for d in docs if d["info"]["id"] not in existing]
        if not todo:
            return True, "already present in OpenCode — nothing to re-import"
        # Path("") is Path(".") and "." is a directory: an empty recorded
        # directory used to re-home the session into the UI's own cwd, silently.
        raw_dir = docs[0]["info"].get("directory") or ""
        directory = Path(raw_dir) if raw_dir else None
        cwd = directory if directory is not None and directory.is_dir() else Path.home()
        note = ("" if directory is not None and cwd == directory else
                f" (original directory {raw_dir or '(none recorded)'} is gone; imported from {cwd})")
        # Export docs live under the mirror, not a temp dir: a failure message
        # that says "run: opencode import <file>" must name a file that exists.
        export_dir = self.mirror_dir / ".reimport"
        export_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for doc in todo:
            f = export_dir / f"{doc['info']['id']}.json"
            f.write_text(json.dumps(doc), encoding="utf-8")
            ok, detail = self._run_import(f, cwd)
            if not ok:
                return False, (f"opencode import failed for {doc['info']['id']} "
                               f"({done} of {len(todo)} imported): {detail}")
            f.unlink(missing_ok=True)
            done += 1
        return True, f"re-imported {done} session(s) into OpenCode{note}"

    def _run_import(self, file: Path, cwd: Path) -> tuple[bool, str]:
        """The only place this adapter runs the binary — user-initiated Restore."""
        if not self.has_binary():
            return False, f"`opencode` is not on PATH — run manually: opencode import {file}"
        import subprocess
        try:
            sid = json.loads(Path(file).read_text(encoding="utf-8"))["info"]["id"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            return False, f"not an export document: {e}"
        try:
            proc = subprocess.run(["opencode", "import", str(file)], cwd=str(cwd),
                                  capture_output=True, text=True, timeout=120,
                                  env={**os.environ, "OPENCODE_DISABLE_AUTOUPDATE": "1"})
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        err = (proc.stderr or "").strip()
        tail = (err + "\n" + (proc.stdout or "")).strip()[-300:]
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {tail}"
        # A schema-decode failure prints "Error: Unexpected error" and STILL
        # exits 0 (a Bun defect): the session landing in the DB is the proof.
        if "Error:" in err:
            return False, tail
        if sid not in self._existing_ids([sid]):
            return False, f"{sid} did not appear in {self.db_path} after import (exit 0): {tail}"
        return True, tail

    def _existing_ids(self, ids: list[str]) -> set[str]:
        conn = self._open_ro()
        if conn is None:
            return set()
        try:
            marks = ",".join("?" * len(ids))
            return {r[0] for r in conn.execute(f"SELECT id FROM session WHERE id IN ({marks})", ids)}
        except sqlite3.Error:
            return set()
        finally:
            conn.close()

    # -- discovery ---------------------------------------------------------------
    def discover(self) -> Iterator[Path]:
        if time.time() - self._last_sync >= self.sync_interval:
            self.sync()
            self._last_sync = time.time()
        if not self.mirror_dir.is_dir():
            return
        for p in sorted(self.mirror_dir.glob("ses_*.jsonl")):
            if not p.is_symlink() and _SID.match(p.stem):
                yield p


def to_export_docs(path: Path | str) -> list[dict]:
    """The mirror file as `opencode export` documents — {info, messages:[{info,
    parts}]} — root first, then each embedded child, for `opencode import`."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = _loads(fh.readline().strip())
            if head.get("type") != "session" or not isinstance(head.get("info"), dict):
                return []
            order = [head["info"]["id"]] + [c["id"] for c in head.get("children") or [] if isinstance(c, dict)]
            infos = {head["info"]["id"]: head["info"], **{c["id"]: c for c in head.get("children") or []}}
            messages: dict[str, list] = {sid: [] for sid in order}
            for line in fh:
                rec = _loads(line.strip())
                if rec.get("type") != "message":
                    continue
                sid = rec.get("session")
                if sid in messages:
                    messages[sid].append({"info": rec.get("info") or {}, "parts": rec.get("parts") or []})
    except OSError:
        return []
    return [{"info": infos[sid], "messages": messages[sid]} for sid in order]


def _user_text(parts: list[dict]) -> str:
    """The human-typed text of a user message: non-synthetic, non-ignored text parts."""
    out = []
    for pd in parts:
        if pd.get("type") == "text" and not pd.get("synthetic") and not pd.get("ignored"):
            t = pd.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
    return "\n".join(out)


# Tool inputs are camelCase in OpenCode (filePath), snake_case elsewhere.
_INPUT_KEYS = ("command", "cmd", "filePath", "file_path", "path", "pattern", "query",
               "description", "url")


def _summarize_input(inp) -> str:
    if isinstance(inp, str):
        return inp[:140]
    if not isinstance(inp, dict):
        return ""
    for key in _INPUT_KEYS:
        if key in inp:
            return f"{key}={str(inp[key])[:140]}"
    return ", ".join(list(inp.keys())[:5])


def _tool_calls(parts: list[dict]) -> list[dict]:
    """[{name, input}] — build-fts indexes `input`, the decision trail lists both."""
    calls = []
    for pd in parts:
        t = pd.get("type")
        if t == "tool":
            st = pd.get("state") if isinstance(pd.get("state"), dict) else {}
            calls.append({"name": pd.get("tool") or "", "input": _summarize_input(st.get("input"))})
        elif t == "subtask":
            what = pd.get("prompt") or pd.get("description") or ""
            calls.append({"name": "subtask", "input": f"agent={pd.get('agent') or '?'}: {what}"[:160]})
    return calls


def _model_key(info: dict) -> str:
    """'provider/model' — OpenCode's own spelling (-m provider/model)."""
    model = info.get("modelID") or ""
    provider = info.get("providerID") or ""
    if not model:
        m = info.get("model")
        if isinstance(m, dict):
            model, provider = m.get("modelID") or m.get("id") or "", m.get("providerID") or provider
    if not model:
        return ""
    return f"{provider}/{model}" if provider else model
