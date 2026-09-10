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

## Checklist — everything that silently degrades if missed

`sources/codex.py` and `sources/opencode.py` are the reference adapters. The
system is source-agnostic, but a handful of per-source registrations exist and
**each one fails quietly when absent**:

| Site | If missed |
|---|---|
| `scripts/compute-costs.py` `_EXTRACTORS[name]` → `(totals, per_model)` or `(totals, per_model, cost_usd)` when the source already knows the true spend | token columns stay NULL, cost 0 forever |
| `pricing.json` aliases (substring match) for models you price yourself | unknown model ⇒ `cost_usd = 0.0` + a stderr line |
| `scripts/extract-reasoning.py` `_EXTRACTORS[name]` + `reasoning.extract_<name>()` | **no decision trail AND no raw-archive copy → never restorable** |
| `watch_roots()` when your config key is not `projects_dir` / `state_dir` / `sessions_dir` | indexed on backfill, **never watched**, no log line |
| `session_id_for_path(p) == parse_header(p).session_id` | `prune-sessions.py` archives every row whose ids disagree |
| `restore_path(row)` (optional) | Restore reports `unsupported`; also used as "where this row's transcript lives" by the context primer and `extract-reasoning --session-id` |
| `session-ui/app.py` `_BRIDGE_CMD`, `index.html` `SOURCE_COLORS` / `BRIDGE_TARGETS` / `SRC_HEX` | bridge 400s; grey badge and chart |
| `bin/resume-here.sh` (auto-detect + case), `bin/check-cli-access.sh`, `install.sh` JOB_PATH loop | `cr <id>` fails; launchd/systemd jobs can't find the binary |
| `scripts/demo.py` row, README Supported CLIs table, `docs/ARCHITECTURE.md`, `docs/SETUP.md`, `skills/work-journal/SKILL.md` | docs drift |
| `tests/test_smoke.py::test_adapters` + a fixture section (mirror the codex/opencode tests) | no regression net |

## Not one-file-per-session? Use the mirror pattern

Every consumer assumes one plain-text file per session at the path `discover()`
yields (`reasoning.archive_raw` copies it, `restore.py` copies it back,
`build-fts` and the extractors open it). If your CLI stores sessions as many
files or as database rows, do what `sources/opencode.py` does: **project** each
session into `<mirror_dir>/<id>.jsonl` from `discover()` (line 1 = a header
with everything `parse_header` needs, then the records), keep a manifest so only
changed sessions are rewritten, archive-then-unlink when a session disappears at
the source, and implement `watch_roots()` plus a `sync_trigger(path)` so the
watcher re-syncs on source changes. `parse_header`/`parse_full` must then work
on the file alone — they are also run on the raw-archive copy.

## 4. Verify

```bash
sb refresh                  # indexes your new source
python tests/test_smoke.py        # add a fixture test for your adapter
python tests/test_portability.py  # add a one-CLI laptop scenario (seed + stub binary)
sb doctor                   # your source shows under [sources]
```

Look at `sources/codex.py` for a complete, recently-added reference adapter.
