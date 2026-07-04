"""Enrichment via the Copilot CLI in headless mode (`copilot -p`)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .provider import parse_facet_json, render_prompt

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

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "") -> dict:
        prompt = render_prompt(turns, cli_source, model, cwd, self.template)
        proc = subprocess.run(
            [self.binary, *self.exec_args, prompt],
            capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:200]}")
        return parse_facet_json(proc.stdout, self.name, model)
