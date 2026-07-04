"""Enrichment via the Claude CLI in headless mode (`claude --print`).

Sends the rendered transcript prompt to the user's existing Claude backend (no
extra API key) and parses the returned facet JSON.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .provider import parse_facet_json, render_prompt

# The spawned CLI inherits this; our Stop hook sees it and no-ops, so headless
# enrichment sessions can never re-enter the indexing pipeline via the hook.
_SUPPRESS = {"SESSION_BROWSER_SUPPRESS_HOOK": "1"}

_REPO = Path(__file__).resolve().parent.parent


class ClaudeHeadless:
    name = "claude-headless"

    def __init__(self, config: dict):
        self.binary = config.get("binary", "claude")
        self.exec_args = config.get("exec_args", ["--print"])
        self.timeout = int(config.get("timeout_secs", 180))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "") -> dict:
        prompt = render_prompt(turns, cli_source, model, cwd, self.template)
        proc = subprocess.run(
            [self.binary, *self.exec_args],
            input=prompt, capture_output=True, text=True, timeout=self.timeout,
            env={**os.environ, **_SUPPRESS},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:200]}")
        return parse_facet_json(proc.stdout, self.name, model)
