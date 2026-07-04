"""Claude Code source adapter.

Transcripts: ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl
Newer-format aware (verified against real transcripts):
  - substantive user turn = type=="user" && promptSource=="typed" && isMeta!=true
  - free title from `ai-title` records (aiTitle field)
  - model at assistant.message.model; usage at assistant.message.usage
  - reasoning lives in assistant.message.content blocks of type "thinking"
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

PROJECTS_DIR = Path(os.path.expanduser("~/.claude/projects"))

_CMD_LINE = re.compile(r"^\s*<(command-name|command-message|command-args|local-command|"
                       r"bash-input|bash-stdout|bash-stderr|system-reminder)")
_USER_MARKER = '"type":"user"'


class ClaudeSource:
    name = "claude"

    def __init__(self, projects_dir: Path | str = PROJECTS_DIR):
        self.projects_dir = Path(os.path.expanduser(str(projects_dir)))

    # -- discovery -------------------------------------------------------------
    def discover(self) -> Iterator[Path]:
        if not self.projects_dir.exists():
            return
        # Skip symlinks: `cr` links a session into other project dirs as resume
        # conduits, but the canonical transcript is the real file at its origin.
        # Indexing only real files keeps one row per session, at its true home.
        for p in self.projects_dir.glob("*/*.jsonl"):
            if not p.is_symlink():
                yield p

    # -- cheap header ----------------------------------------------------------
    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        head = _read_head(path, 60)
        if not head:
            return None
        tail = _read_tail_lines(path, 60)

        session_id = path.stem
        cwd = ""
        version = None
        git_branch = None
        start_time = ""
        first_message = ""
        entrypoint = None

        for rec in head:
            cwd = cwd or rec.get("cwd", "")
            version = version or rec.get("version")
            git_branch = git_branch or rec.get("gitBranch")
            entrypoint = entrypoint or rec.get("entrypoint")
            if not start_time and rec.get("timestamp"):
                start_time = rec["timestamp"]
            if not first_message and _is_substantive_user(rec):
                first_message = _extract_text(rec.get("message", {}).get("content"))

        # Skip headless/SDK sessions (e.g. our own `claude --print` enrichment calls,
        # entrypoint "sdk-cli") so they never pollute the browsable index.
        if entrypoint == "sdk-cli":
            return None
        # if first typed message wasn't in the head window, scan a bit more
        if not first_message:
            first_message = self._first_typed_message(path)

        last_activity = start_time
        title = None
        model = None
        for rec in tail:
            if rec.get("timestamp"):
                last_activity = rec["timestamp"]
            if rec.get("type") == "ai-title" and rec.get("aiTitle"):
                title = rec["aiTitle"]
            if rec.get("type") == "assistant":
                m = rec.get("message", {}).get("model")
                if m and m != "<synthetic>":
                    model = m
        if title is None:
            title = self._last_ai_title(path)

        project_path = str(path.parent)
        folder_name = _folder_from_cwd(cwd) or path.parent.name

        return SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=project_path,
            cwd=cwd,
            folder_name=folder_name,
            start_time=to_iso_utc(start_time),
            last_activity=to_iso_utc(last_activity),
            first_message=first_message[:500],
            turn_count=_count_typed_turns(path),
            title=title,
            model_used=model,
            cli_version=version,
            metadata={"gitBranch": git_branch} if git_branch else {},
        )

    # -- full parse ------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")
                if t == "user" and _is_substantive_user(rec):
                    content = _extract_text(rec.get("message", {}).get("content"))
                    if content:
                        turns.append(Turn(role="user", content=content))
                elif t == "assistant":
                    msg = rec.get("message", {})
                    text, tools = _assistant_text_and_tools(msg.get("content"))
                    if text or tools:
                        turns.append(Turn(role="assistant", content=text, tool_calls=tools))
        return ParsedSession(header=header, turns=turns)

    # -- identity / resume / availability ----------------------------------------
    def session_id_for_path(self, path: Path) -> Optional[str]:
        return path.stem if path.suffix == ".jsonl" else None

    def resume_command(self, session_id: str) -> str:
        return f"claude --resume {shlex.quote(session_id)}"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None and self.projects_dir.exists()

    # -- helpers ---------------------------------------------------------------
    def _first_typed_message(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in islice(fh, 0, 1200):
                if _USER_MARKER not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_substantive_user(rec):
                    text = _extract_text(rec.get("message", {}).get("content"))
                    if text:
                        return text
        return ""

    def _last_ai_title(self, path: Path) -> Optional[str]:
        title = None
        for rec in _read_tail_lines(path, 200):
            if rec.get("type") == "ai-title" and rec.get("aiTitle"):
                title = rec["aiTitle"]
        return title


# --- module-level parsing helpers ---------------------------------------------
def _is_substantive_user(rec: dict) -> bool:
    """A real human-typed turn, format-agnostic.

    Newer transcripts mark these with promptSource=="typed"; older ones lack that
    field. The cross-format signal: a non-meta, non-sidechain user record whose
    message.content is a *string* (tool-results are lists) that is non-empty after
    stripping command/caveat wrapper lines.
    """
    if rec.get("type") != "user" or rec.get("isMeta") or rec.get("isSidechain"):
        return False
    if rec.get("promptSource") and rec.get("promptSource") != "typed":
        return False
    content = rec.get("message", {}).get("content")
    if not isinstance(content, str):
        return False
    return bool(_strip_command_lines(content).strip())


def _extract_text(content) -> str:
    """Normalize message.content (string or block list) to clean text."""
    if isinstance(content, str):
        return _strip_command_lines(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    continue  # skip tool results for "message" text
            elif isinstance(block, str):
                parts.append(block)
        return _strip_command_lines("\n".join(parts))
    return ""


def _strip_command_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _CMD_LINE.match(ln)]
    return "\n".join(lines).strip()


def _assistant_text_and_tools(content) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    tools: list[dict] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text" and block.get("text"):
                text_parts.append(block["text"])
            elif bt == "tool_use":
                tools.append({"name": block.get("name", ""),
                              "input": _summarize_input(block.get("input"))})
    elif isinstance(content, str):
        text_parts.append(content)
    return "\n".join(text_parts).strip(), tools


def _summarize_input(inp) -> str:
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "query", "description"):
        if key in inp:
            return f"{key}={str(inp[key])[:120]}"
    return ", ".join(list(inp.keys())[:4])


def _folder_from_cwd(cwd: str) -> str:
    return Path(cwd).name if cwd else ""


def _read_head(path: Path, n: int) -> list[dict]:
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in islice(fh, 0, n):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _read_tail_lines(path: Path, n: int, blocksize: int = 65536) -> list[dict]:
    """Read approximately the last n JSONL records without loading the whole file."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            data = b""
            read = 0
            while read < size and data.count(b"\n") <= n:
                step = min(blocksize, size - read)
                read += step
                fh.seek(size - read)
                data = fh.read(step) + data
        lines = data.split(b"\n")[-(n + 1):]
    except OSError:
        return []
    out = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw.decode("utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    return out


# (path -> (size, mtime_ns, count)) — parse_header runs on every debounced watcher
# tick; without this an active multi-MB transcript would be fully re-streamed
# every ~0.5s just to recount turns.
_TURN_CACHE: dict[str, tuple[int, int, int]] = {}


def _count_typed_turns(path: Path) -> int:
    """Count substantive user turns. Streams the file and JSON-parses only the
    lines that look like user records (cheap substring prefilter), so it works
    across both transcript formats without materializing the whole file.
    Results are memoized by (size, mtime) so unchanged files cost one stat()."""
    try:
        st = path.stat()
    except OSError:
        return 0
    key = str(path)
    cached = _TURN_CACHE.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
        return cached[2]
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _USER_MARKER not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_substantive_user(rec):
                    n += 1
    except OSError:
        return 0
    if len(_TURN_CACHE) > 4096:  # bound memory in the long-lived watcher
        _TURN_CACHE.clear()
    _TURN_CACHE[key] = (st.st_size, st.st_mtime_ns, n)
    return n
