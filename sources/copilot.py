"""GitHub Copilot CLI source adapter.

Sessions: ~/.copilot/session-state/<uuid>/events.jsonl  (+ workspace.yaml)
The sibling session.db holds only todos/inbox state — NOT the transcript — so we
parse events.jsonl for turns and read workspace.yaml for cheap header metadata.

Event stream (verified): session.start, session.model_change (data.newModel),
user.message (data.content), assistant.message (data.content, data.reasoningText,
data.toolRequests, data.outputTokens). Unlike Claude, Copilot DOES persist
reasoning text in assistant.message.reasoningText.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# (events.jsonl size, events mtime, workspace.yaml mtime) -> parsed header
_HDR_CACHE: dict[str, tuple[tuple[int, int, int], "SessionHeader | None"]] = {}

STATE_DIR = Path(os.path.expanduser("~/.copilot/session-state"))


def _text(v) -> str:
    """Event content defensively coerced to str (a non-string content value must
    degrade to '' rather than crash the whole session parse)."""
    return v if isinstance(v, str) else ""


class CopilotSource:
    name = "copilot"

    def __init__(self, state_dir: Path | str = STATE_DIR):
        self.state_dir = Path(os.path.expanduser(str(state_dir)))

    def discover(self) -> Iterator[Path]:
        if not self.state_dir.exists():
            return
        for d in self.state_dir.iterdir():
            ev = d / "events.jsonl"
            if ev.exists():
                yield ev

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        sess_dir = path.parent
        session_id = sess_dir.name
        # Memoized like codex: the watcher debounce would otherwise re-stream
        # events.jsonl every 500ms. Key covers workspace.yaml too — the title
        # and updated_at live there, not in the events file.
        try:
            st = path.stat()
            ws_path = sess_dir / "workspace.yaml"
            ws_mtime = ws_path.stat().st_mtime_ns if ws_path.exists() else 0
        except OSError:
            return None
        cache_key = (st.st_size, st.st_mtime_ns, ws_mtime)
        hit = _HDR_CACHE.get(str(path))
        if hit is not None and hit[0] == cache_key:
            return hit[1]
        ws = self._workspace(sess_dir)
        cwd = ws.get("cwd", "")
        title = ws.get("name") or None
        # PyYAML auto-parses ISO timestamps into datetime objects; to_iso_utc
        # normalizes both datetimes and strings to the canonical sortable form.
        start = to_iso_utc(ws.get("created_at", ""))
        last = to_iso_utc(ws.get("updated_at", "")) or start

        first_message = ""
        model = None
        turn_count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"type"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = rec.get("type")
                    data = rec.get("data", {}) if isinstance(rec.get("data"), dict) else {}
                    if t == "user.message":
                        turn_count += 1
                        if not first_message:
                            first_message = _text(data.get("content"))[:500]
                    elif t == "session.model_change" and data.get("newModel"):
                        model = data["newModel"]
        except OSError:
            return None

        header = SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=str(sess_dir),
            cwd=cwd,
            folder_name=Path(cwd).name if cwd else sess_dir.name,
            start_time=start,
            last_activity=last,
            first_message=first_message,
            turn_count=turn_count,
            title=title,
            model_used=model,
            metadata={"workspace_name": ws.get("name")},
        )
        if len(_HDR_CACHE) > 4096:
            _HDR_CACHE.clear()
        _HDR_CACHE[str(path)] = (cache_key, header)
        return header

    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")
                data = rec.get("data", {}) if isinstance(rec.get("data"), dict) else {}
                if t == "user.message":
                    content = _text(data.get("content")).strip()
                    if content:
                        turns.append(Turn(role="user", content=content))
                elif t == "assistant.message":
                    content = _text(data.get("content")).strip()
                    tools = [{"name": tr.get("name", ""), "input": ""}
                             for tr in (data.get("toolRequests") or []) if isinstance(tr, dict)]
                    if content or tools:
                        turns.append(Turn(role="assistant", content=content, tool_calls=tools))
        return ParsedSession(header=header, turns=turns)

    def session_id_for_path(self, path: Path) -> Optional[str]:
        # Transcript lives at <state_dir>/<session-id>/events.jsonl; the id is the
        # directory name, NOT the filename stem ("events").
        return path.parent.name if path.name == "events.jsonl" else None

    def resume_command(self, session_id: str) -> str:
        return f"copilot --resume={shlex.quote(session_id)}"

    def is_available(self) -> bool:
        return shutil.which("copilot") is not None and self.state_dir.exists()

    # -- helpers --
    @staticmethod
    def _workspace(sess_dir: Path) -> dict:
        wf = sess_dir / "workspace.yaml"
        if not wf.exists():
            return {}
        try:
            return yaml.safe_load(wf.read_text()) or {}
        except (yaml.YAMLError, OSError):
            return {}
