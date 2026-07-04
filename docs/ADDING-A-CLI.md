# Adding a CLI

The whole system is source-agnostic. Adding a CLI (Gemini, OpenCode, Aider,
Ollama, …) is **one new file + two small registrations**. The indexer, DB, UI,
watcher, MCP server, cost pipeline, and reasoning trails all work automatically.

## 1. Write `sources/<cli>.py`

Implement the `SessionSource` protocol from `sources/base.py`:

```python
from pathlib import Path
from typing import Iterator, Optional
import os, shutil
from .base import ParsedSession, SessionHeader, Turn, to_iso_utc


class MyCliSource:
    name = "mycli"

    def __init__(self, sessions_dir="~/.mycli/sessions"):
        self.sessions_dir = Path(os.path.expanduser(str(sessions_dir)))

    def discover(self) -> Iterator[Path]:
        """Yield each transcript file. Skip symlinks (cr creates them)."""
        if not self.sessions_dir.exists():
            return
        for p in self.sessions_dir.glob("*.jsonl"):
            if not p.is_symlink():
                yield p

    def session_id_for_path(self, path: Path) -> Optional[str]:
        """Map a path to its session id WITHOUT reading the file (used on delete,
        when the file may already be gone). Return None if not a transcript."""
        return path.stem if path.suffix == ".jsonl" else None

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """CHEAP metadata — do not read the whole file for big transcripts.
        Always run timestamps through to_iso_utc() so ordering is correct."""
        # ... parse id, cwd, model, first user message, turn count ...
        return SessionHeader(
            session_id=..., cli_source=self.name, project_path=str(path.parent),
            cwd=..., folder_name=..., start_time=to_iso_utc(...),
            last_activity=to_iso_utc(...), first_message=...[:500],
            turn_count=..., title=..., model_used=..., cli_version=...,
        )

    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Full parse into ordered Turn(role, content) — used by enrichment."""
        header = self.parse_header(path)
        if header is None:
            return None
        turns = [...]  # Turn(role="user"/"assistant", content=...)
        return ParsedSession(header=header, turns=turns)

    def resume_command(self, session_id: str) -> str:
        return f"mycli resume {session_id}"

    def is_available(self) -> bool:
        return shutil.which("mycli") is not None and self.sessions_dir.exists()
```

Tips:
- Guard every field: coerce non-string content to `""`, wrap `json.loads` in
  try/except so one malformed line can't drop the whole session.
- Return `None` from `parse_header` for sessions you want excluded (headless/
  meta-only). Those are skipped everywhere consistently.

## 2. Register it in `sources/registry.py`

```python
def _make_mycli():
    from sources.mycli import MyCliSource
    cfg = sbconfig.source_config("mycli")
    return MyCliSource(cfg.get("sessions_dir", "~/.mycli/sessions"))

_FACTORIES = { ..., "mycli": _make_mycli }
```

## 3. Add a config block to `config.toml`

```toml
[sources.mycli]
enabled      = true
sessions_dir = "~/.mycli/sessions"
binary       = "mycli"
```

## Optional extras

- **Cost:** add a `_usage_mycli(path) -> (totals, per_model)` function to
  `scripts/compute-costs.py` and register it in its `_EXTRACTORS` dict. Add your
  model names to `pricing.json` `aliases`.
- **Reasoning trail:** add an `extract_mycli(path)` to `reasoning.py` (return
  `list[ReasoningStep]`) and register it in `extract-reasoning.py` `_EXTRACTORS`.
- **Bridge target:** add a command template to `_BRIDGE_CMD` in
  `session-ui/app.py` and the name to `BRIDGE_TARGETS` in the SPA.
- **cr resume:** add a branch to `bin/resume-here.sh` if the CLI needs memory
  porting (Claude) or in-place resume (Codex).
- **UI badge color:** add your source to `SOURCE_COLORS` in `static/index.html`.

## 4. Verify

```bash
sb refresh                  # indexes your new source
python tests/test_smoke.py  # add a fixture test for your adapter
sb doctor                   # your source shows under [sources]
```

Look at `sources/codex.py` for a complete, recently-added reference adapter.
