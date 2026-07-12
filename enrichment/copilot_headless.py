"""Enrichment via the Copilot CLI in headless mode (`copilot -p`).

Note: the Copilot CLI only accepts the prompt as an argv parameter (no stdin
mode), so the rendered transcript is briefly visible in the local process list
while the call runs. The prompt is capped well below ARG_MAX so the call can
never fail with E2BIG. If Copilot gains a stdin prompt mode, switch to it (see
claude_headless.py for the pattern).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .provider import parse_facet_json, render_prompt

_SUPPRESS = {"SESSION_BROWSER_SUPPRESS_HOOK": "1"}

_REPO = Path(__file__).resolve().parent.parent


class CopilotHeadless:
    name = "copilot-headless"

    def __init__(self, config: dict):
        self.binary = config.get("binary", "copilot")
        self.exec_args = config.get("exec_args", ["-p"])
        self.timeout = int(config.get("timeout_secs", 180))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        prompt = render_prompt(turns, cli_source, model, cwd, self.template, prior=prior)
        prompt = prompt[:120_000]  # argv-passed; stay far below ARG_MAX
        proc = subprocess.run(
            [self.binary, *self.exec_args, prompt],
            capture_output=True, text=True, timeout=self.timeout,
            env={**os.environ, **_SUPPRESS},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:200]}")
        return parse_facet_json(proc.stdout, self.name, model)
