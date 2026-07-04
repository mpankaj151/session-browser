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
import shutil
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .base import ParsedSession, SessionHeader, Turn

STATE_DIR = Path(os.path.expanduser("~/.copilot/session-state"))


def _iso(v) -> str:
    """Coerce a yaml-parsed timestamp (datetime or str) to an ISO string."""
    if not v:
        return ""
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


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
        ws = self._workspace(sess_dir)
        cwd = ws.get("cwd", "")
        title = ws.get("name") or None
        # PyYAML auto-parses ISO timestamps into datetime objects; coerce to strings.
        start = _iso(ws.get("created_at", ""))
        last = _iso(ws.get("updated_at", "")) or start

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
                            first_message = (data.get("content") or "")[:500]
                    elif t == "session.model_change" and data.get("newModel"):
                        model = data["newModel"]
        except OSError:
            return None

        return SessionHeader(
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
                    content = (data.get("content") or "").strip()
                    if content:
                        turns.append(Turn(role="user", content=content))
                elif t == "assistant.message":
                    content = (data.get("content") or "").strip()
                    tools = [{"name": tr.get("name", ""), "input": ""}
                             for tr in (data.get("toolRequests") or []) if isinstance(tr, dict)]
                    if content or tools:
                        turns.append(Turn(role="assistant", content=content, tool_calls=tools))
        return ParsedSession(header=header, turns=turns)

    def resume_command(self, session_id: str) -> str:
        return f"copilot --resume={session_id}"

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
