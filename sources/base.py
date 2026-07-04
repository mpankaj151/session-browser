"""Data model + adapter Protocol shared by every CLI source.

Adding a new CLI (codex, opencode, ollama, ...) means writing one module that
implements SessionSource and registering it in app.py and watcher.py. Nothing
else in the system needs to change — the indexer, DB, UI, and MCP server are all
source-agnostic and speak only SessionHeader / Turn / ParsedSession.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional, Protocol, runtime_checkable

SourceName = str  # "claude" | "copilot" | "codex" | "opencode" | ...


@dataclass
class SessionHeader:
    session_id: str
    cli_source: SourceName
    project_path: str
    cwd: str
    folder_name: str
    start_time: str
    last_activity: str
    first_message: str
    turn_count: int
    title: Optional[str] = None           # human-friendly title (e.g. Claude ai-title)
    topics: Optional[str] = None          # JSON array string e.g. '["python","testing"]'
    model_used: Optional[str] = None
    cli_version: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ParsedSession:
    header: SessionHeader
    turns: list[Turn]


@runtime_checkable
class SessionSource(Protocol):
    name: SourceName

    def discover(self) -> Iterator[Path]:
        """Yield transcript file paths for this CLI."""
        ...

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """Cheap metadata extraction — must NOT read the whole transcript."""
        ...

    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Full transcript parse into ordered turns."""
        ...

    def resume_command(self, session_id: str) -> str:
        """The CLI's own command to resume this session in that CLI."""
        ...

    def is_available(self) -> bool:
        """True if the CLI binary and its session directory are present."""
        ...
