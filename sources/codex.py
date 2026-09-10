"""OpenAI Codex CLI source adapter.

Sessions: ~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl
Archived: ~/.codex/archived_sessions/YYYY/MM/DD/<same>   (`codex archive` MOVES)
Each line is {"type": ..., "payload": {...}} (or a top-level record).

Two on-disk dialects, both live:

  legacy    - payload.type "user_message"/"agent_message" carry payload.message.
  paginated - Codex >=~0.144 starts every non-ephemeral CLI thread in
              `paginated` history mode (codex-rs/tui/src/app_server_session.rs
              and codex-rs/exec/src/lib.rs both set
              `history_mode: (!ephemeral).then_some(Paginated)`). In that mode
              codex-rs/rollout/src/policy.rs persists the legacy events ONLY
              when the mode is Legacy; the same content arrives as
              "item_completed" wrapping a TurnItem instead.

Rollouts older than 7 days are zstd-compressed in place to `<name>.jsonl.zst`
by the worker spawned from codex-rs/core/src/thread_manager.rs, so every read
path has to handle both representations.

Other records, unchanged across dialects:
  - first line: type "session_meta", payload {id, timestamp, cwd, cli_version,
    model_provider, history_mode}
  - model: type "turn_context", payload.model (e.g. "gpt-5.5")
  - reasoning: payload.type "agent_reasoning"/"agent_reasoning_raw_content"
  - title: payload.type "thread_name_updated" (also ~/.codex/session_index.jsonl)
  - tokens: payload.type "token_count", payload.info.total_token_usage (cumulative)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

SESSIONS_DIR = Path(os.path.expanduser("~/.codex/sessions"))

_JSONL = ".jsonl"
_ZST = ".zst"
_PREFIX = "rollout-"
# "2026-08-01T10-00-00" — the fixed-width stamp codex-rs/rollout/src/
# rollout_file_name.rs writes between the prefix and the ids.
_TS_LEN = 19

# parse_header streams the whole rollout (last_activity lives on the final
# line), so memoize by (size, mtime): the watcher re-parses on a 500ms debounce
# and would otherwise pin a core on an active multi-MB session.
_HDR_CACHE: dict[str, tuple[tuple[int, int], "SessionHeader | None"]] = {}

# Record types this adapter understands. Used only to tell "this session has no
# user turns" apart from "this file is in a dialect I have never seen" — the
# distinction whose absence hid the paginated-rollout outage for weeks.
_KNOWN_TYPES = frozenset({
    "session_meta", "turn_context", "user_message", "agent_message",
    "item_completed", "item_started", "token_count", "task_started",
    "task_complete", "turn_started", "turn_complete", "turn_aborted",
    "thread_name_updated", "agent_reasoning", "agent_reasoning_raw_content",
    "message", "reasoning", "function_call", "function_call_output",
    "custom_tool_call", "custom_tool_call_output", "local_shell_call",
    "web_search_call", "web_search_end", "patch_apply_end", "mcp_tool_call_end",
    "compacted", "context_compacted", "world_state", "security_risk_score",
})

# TurnItem tags are PascalCase (codex-rs/protocol/src/items.rs has no
# rename_all on the enum); matched case-insensitively for safety.
_ITEM_ROLES = {"usermessage": "user", "agentmessage": "assistant"}

_WARNED: set[str] = set()


def _payload(rec: dict) -> tuple[str, dict]:
    """(record type, payload dict). Codex nests real data under 'payload'."""
    p = rec.get("payload")
    if isinstance(p, dict):
        return p.get("type") or rec.get("type") or "", p
    return rec.get("type") or "", rec


def _rollout_errors() -> tuple:
    """What a reader must tolerate: I/O, a truncated frame (ValueError from the
    text wrapper) and zstandard's own ZstdError — which is NOT a ValueError."""
    errs: tuple = (OSError, ValueError)
    try:
        import zstandard
        errs += (zstandard.ZstdError,)
    except ImportError:
        pass
    return errs


ROLLOUT_ERRORS = _rollout_errors()


@contextlib.contextmanager
def open_rollout(path: Path):
    """Line reader for a rollout, plain `.jsonl` or zstd `.jsonl.zst`.

    Mirrors codex_rollout::open_rollout_line_reader: callers should not need to
    know which representation is currently on disk. Public because every
    consumer of a rollout (reasoning, costs, full-text) must go through it —
    a plain open() reads a zstd frame as garbage and silently yields nothing.
    """
    if path.name.endswith(_ZST):
        import zstandard  # imported lazily: only compressed rollouts need it
        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh


def _item_text(item: dict) -> str:
    """Flatten a paginated TurnItem's content[] to plain text.

    Codex tags the chunks inconsistently — UserInput is snake_case
    ({"type": "text"}), AgentMessageContent is PascalCase ({"type": "Text"}) —
    so the type is compared case-insensitively.
    """
    parts = []
    for chunk in item.get("content") or []:
        if isinstance(chunk, dict) and str(chunk.get("type", "")).lower() == "text":
            text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _turn_of(pt: str, p: dict) -> Optional[tuple[str, str]]:
    """(role, content) for any record carrying a message, in either dialect.

    Returns None for every other record. The two dialects never overlap: policy
    .rs persists the legacy events only in Legacy mode and UserMessage/
    AgentMessage items only in Paginated, so a turn is never counted twice.
    """
    if pt in ("user_message", "agent_message"):
        message = p.get("message")
        if isinstance(message, str) and message.strip():
            return ("user" if pt == "user_message" else "assistant", message.strip())
        return None
    if pt == "item_completed":
        item = p.get("item")
        if isinstance(item, dict):
            role = _ITEM_ROLES.get(str(item.get("type", "")).lower())
            if role:
                text = _item_text(item)
                if text:
                    return (role, text)
    return None


def _warn_unknown_dialect(path: Path, session_id: str) -> None:
    """Surface schema drift loudly ONCE per session.

    The paginated switch was invisible because an unparseable rollout and an
    empty one both returned None, and the watcher drops None without logging.
    """
    if session_id in _WARNED:
        return
    _WARNED.add(session_id)
    print(f"[codex] unrecognised rollout dialect in {path.name} — indexing with "
          f"0 turns; the adapter likely needs updating for a new Codex format",
          file=sys.stderr, flush=True)


class CodexSource:
    name = "codex"

    def __init__(self, sessions_dir: Path | str = SESSIONS_DIR,
                 archived_dir: Path | str | None = None):
        self.sessions_dir = Path(os.path.expanduser(str(sessions_dir)))
        self.archived_dir = (
            Path(os.path.expanduser(str(archived_dir))) if archived_dir
            else self.sessions_dir.parent / "archived_sessions"
        )

    # -- discovery -------------------------------------------------------------
    def watch_roots(self) -> list[Path]:
        """Every directory the watcher must subscribe to.

        `codex archive` MOVES a rollout into the sibling archived_sessions tree.
        Watching only sessions/ made that move look like a deletion.
        """
        return [self.sessions_dir, self.archived_dir]

    def discover(self) -> Iterator[Path]:
        for root in self.watch_roots():
            if not root.exists():
                continue
            for pattern in (f"*/*/*/{_PREFIX}*{_JSONL}", f"*/*/*/{_PREFIX}*{_JSONL}{_ZST}"):
                for p in root.glob(pattern):
                    if not p.is_symlink():
                        yield p

    def session_id_for_path(self, path: Path) -> Optional[str]:
        """Thread id from the filename alone — the file may already be gone.

        Mirrors codex-rs/rollout/src/rollout_file_name.rs: `rollout-` + a
        19-char timestamp + `-` + ids, where a reverted thread appends
        `_<rollout_id>` after the stable thread id. Anchoring on the timestamp
        width keeps both that shape and `.jsonl.zst` correct, which the old
        "last five dash-separated groups" heuristic did not.
        """
        name = path.name
        if name.endswith(_ZST):
            name = name[: -len(_ZST)]
        if not (name.startswith(_PREFIX) and name.endswith(_JSONL)):
            return None
        core = name[len(_PREFIX):-len(_JSONL)]
        if len(core) <= _TS_LEN or core[_TS_LEN] != "-":
            return None
        return core[_TS_LEN + 1:].split("_", 1)[0] or None

    # -- cheap header ----------------------------------------------------------
    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        try:
            st = path.stat()
        except OSError:
            return None
        cache_key = (st.st_size, st.st_mtime_ns)
        hit = _HDR_CACHE.get(str(path))
        if hit is not None and hit[0] == cache_key:
            return hit[1]
        session_id = self.session_id_for_path(path) or path.stem
        cwd = ""
        start_time = ""
        version = None
        model = None
        title = None
        first_message = ""
        last_ts = ""
        turn_count = 0
        events = 0        # records that are not the session_meta preamble
        recognised = 0    # ...of those, ones this adapter knows
        try:
            with open_rollout(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp") or ""
                    if ts:
                        last_ts = ts
                    pt, p = _payload(rec)
                    if pt == "session_meta" or rec.get("type") == "session_meta":
                        session_id = p.get("id") or session_id
                        cwd = cwd or p.get("cwd", "")
                        start_time = start_time or p.get("timestamp") or ts
                        version = version or p.get("cli_version")
                        continue
                    events += 1
                    if pt in _KNOWN_TYPES:
                        recognised += 1
                    if pt == "turn_context":
                        model = model or p.get("model")
                        cwd = cwd or p.get("cwd", "")
                    elif pt == "thread_name_updated":
                        title = p.get("name") or p.get("thread_name") or title
                    turn = _turn_of(pt, p)
                    if turn and turn[0] == "user":
                        turn_count += 1
                        first_message = first_message or turn[1]
        except ROLLOUT_ERRORS:
            # a truncated or corrupt zstd frame is "not a session", never a traceback
            return None

        if len(_HDR_CACHE) > 4096:
            _HDR_CACHE.clear()
        if turn_count == 0 and not first_message:
            if events and not recognised:
                # A dialect we do not know. Index it anyway so the session is
                # visible and the warning is actionable — silently returning
                # None here is exactly how the paginated switch went unnoticed.
                _warn_unknown_dialect(path, session_id)
            else:
                # A rollout with no user turns (aborted / meta-only) — not
                # browsable. Cached too: these files are exactly the ones
                # re-scanned pointlessly.
                _HDR_CACHE[str(path)] = (cache_key, None)
                return None

        folder = Path(cwd).name if cwd else path.parent.name
        header = SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=str(path.parent),
            cwd=cwd,
            folder_name=folder,
            start_time=to_iso_utc(start_time),
            last_activity=to_iso_utc(last_ts or start_time),
            first_message=first_message[:500],
            turn_count=turn_count,
            title=title,
            model_used=model,
            cli_version=version,
            metadata={"transcript": path.name},
        )
        _HDR_CACHE[str(path)] = (cache_key, header)
        return header

    # -- full parse ------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        try:
            with open_rollout(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    turn = _turn_of(*_payload(rec))
                    if turn:
                        turns.append(Turn(role=turn[0], content=turn[1]))
        except ROLLOUT_ERRORS:
            return None
        return ParsedSession(header=header, turns=turns)

    # -- resume / availability -------------------------------------------------
    def resume_command(self, session_id: str) -> str:
        return f"codex resume {shlex.quote(session_id)}"

    # -- restore / locate --------------------------------------------------------
    def _contained(self, path: Path) -> bool:
        for root in (self.sessions_dir, self.archived_dir):
            try:
                path.resolve().relative_to(root.resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def restore_path(self, row) -> Optional[Path]:
        """Where this row's rollout lives — or, for restore, must be written.

        Codex names files rollout-<19-char ts>-<id>.jsonl inside a date dir
        and may zstd-compress or `codex archive` them; the row records the dir
        (project_path) but not the name. Look the file up (row dir first, then
        both trees), else synthesise the canonical name from start_time so a
        restored copy is exactly what discover()/session_id_for_path expect.
        Refuses anything outside the codex trees, like ClaudeSource."""
        sid = row["session_id"]
        project_path = row["project_path"] or ""
        day = Path(os.path.expanduser(project_path)) if project_path else None
        if day is not None and not self._contained(day):
            return None
        patterns = (f"{_PREFIX}*{sid}{_JSONL}", f"{_PREFIX}*{sid}{_JSONL}{_ZST}")
        found: list[Path] = []
        if day is not None and day.is_dir():
            for pat in patterns:
                found += sorted(day.glob(pat))
        if not found:
            for root in (self.sessions_dir, self.archived_dir):
                if root.exists():
                    for pat in patterns:
                        found += sorted(root.glob(f"*/*/*/{pat}"))
        for candidate in found:
            if self.session_id_for_path(candidate) == sid:
                return candidate
        try:
            start = row["start_time"]
        except (KeyError, IndexError):
            start = ""
        stamp = (start or "")[:19].replace(":", "-")
        if len(stamp) != _TS_LEN:
            from datetime import datetime, timezone
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        if day is None:
            day = self.sessions_dir / stamp[0:4] / stamp[5:7] / stamp[8:10]
        return day / f"{_PREFIX}{stamp}-{sid}{_JSONL}"

    def has_binary(self) -> bool:
        """Whether `codex` is runnable from here — a UI hint for resume, only."""
        return shutil.which("codex") is not None

    def is_available(self) -> bool:
        """Whether there are transcripts to read.

        Deliberately NOT gated on `which codex`: the unified ChatGPT/Codex app
        moves the binary out of the PATH the watcher runs under, and gating a
        filesystem watcher on that silently unsubscribed ~/.codex/sessions with
        no log line at all.
        """
        return self.sessions_dir.exists() or self.archived_dir.exists()


_open_rollout = open_rollout  # backwards-compatible alias
