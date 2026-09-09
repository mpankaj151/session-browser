"""Enrichment via OpenCode in headless mode (`opencode run`).

The team harness: one CLI in front of every provider (Anthropic, OpenAI,
open-weight models through gateways). Selected per machine with
[enrichment].provider = "opencode-headless"; machines without a suitable
credential keep claude-headless.

Verified against OpenCode 1.18.15 + source:
  * the prompt goes on STDIN — positional args containing spaces are wrapped in
    literal quotes and inner quotes escaped, which corrupts a transcript prompt;
  * `--format json` streams NDJSON on stdout; the answer is the `text` events'
    part.text, spend is in `step_finish` events; all UI chatter is on stderr;
  * non-interactive runs auto-REJECT every permission unless `--auto` is passed
    — we never pass it, so the summariser cannot use tools;
  * `--title` tags the run for the adapter to skip AND suppresses OpenCode's
    separate title-generation model call (a second billed call otherwise);
  * `--pure` keeps external plugins (including our own) from firing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import sbconfig

from .provider import parse_facet_json, render_prompt

_REPO = Path(__file__).resolve().parent.parent

# provider/model, OpenCode's own spelling (`-m` splits on the first "/").
# Sonnet-tier is the right cost/quality for structured summarisation; the
# same model is also reachable as opencode/claude-sonnet-5 (Zen gateway) or
# github-copilot/claude-sonnet-5. "" = whatever OpenCode's config defaults to.
_DEFAULT_MODEL = "anthropic/claude-sonnet-5"


class OpenCodeHeadless:
    name = "opencode-headless"

    def __init__(self, config: dict):
        self.binary = config.get("binary", "opencode")
        self.model = config.get("model", _DEFAULT_MODEL)
        self.agent = config.get("agent", "")        # optional PRIMARY agent (no-tools summariser)
        self.variant = config.get("variant", "")    # optional reasoning effort, e.g. "minimal"
        self.timeout = int(config.get("timeout_secs", 300))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def command(self) -> list[str]:
        args = [self.binary, "run", "--pure", "--format", "json"]
        if self.model:
            args += ["--model", self.model]
        if self.agent:
            args += ["--agent", self.agent]
        if self.variant:
            args += ["--variant", self.variant]
        # Never --auto / --share / --attach: permissions must be auto-rejected,
        # the transcript must never be uploaded, and --attach breaks exit codes.
        args += ["--title", sbconfig.OPENCODE_ENRICHMENT_TITLE]
        return args

    def env(self) -> dict[str, str]:
        """Environment for the run. OPENCODE_DB=:memory: is the isolation lever:
        no session rows persist anywhere, while auth.json (resolved from the
        data dir, not the DB path) keeps working — XDG_DATA_HOME would move
        the credentials too. OPENCODE_PERMISSION is deep-merged over config,
        so it still denies everything if --agent silently falls back to the
        all-tools default; invalid JSON there is silently ignored by OpenCode,
        hence a constant, never built dynamically."""
        return {
            **os.environ,
            "OPENCODE_DB": ":memory:",
            "OPENCODE_PERMISSION": '{"*":"deny"}',
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            # our Claude Stop hook sees this and no-ops (symmetry with the
            # other providers; an OpenCode run never fires Claude hooks anyway)
            "SESSION_BROWSER_SUPPRESS_HOOK": "1",
        }

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    # Injectable for tests; the real thing is subprocess.run.
    _run = staticmethod(subprocess.run)

    @staticmethod
    def _parse_stream(stdout: str) -> tuple[str, float, dict, str | None]:
        """NDJSON -> (text, cost_usd, tokens, error). Non-JSON lines are ignored
        (stdout is clean by construction, but never let a stray line kill a run)."""
        texts: list[str] = []
        cost = 0.0
        tokens = {"input": 0, "output": 0}
        error: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            part = ev.get("part") or {}
            if kind == "text":
                texts.append(part.get("text") or "")
            elif kind == "step_finish":
                cost += float(part.get("cost") or 0)
                tk = part.get("tokens") or {}
                tokens["input"] += int(tk.get("input") or 0)
                tokens["output"] += int(tk.get("output") or 0)
            elif kind == "error":
                err = ev.get("error") or {}
                data = err.get("data") if isinstance(err.get("data"), dict) else {}
                error = f"{err.get('name', 'error')}: {data.get('message') or err}"
        return "".join(texts), cost, tokens, error

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        prompt = render_prompt(turns, cli_source, model, cwd, self.template, prior=prior)
        try:
            proc = self._run(
                self.command(),
                input=prompt, capture_output=True, text=True, timeout=self.timeout,
                env=self.env(), cwd=str(Path.home()),
            )
        except subprocess.TimeoutExpired:
            # OpenCode retries connection errors with growing backoff rather than
            # failing fast, so a dead/unauthenticated provider only ends here.
            raise RuntimeError(
                f"{self.binary} run timed out after {self.timeout}s — provider unreachable "
                f"or not authenticated? (`opencode auth list`, `opencode models <provider>`)"
            ) from None
        text, cost, tokens, error = self._parse_stream(proc.stdout or "")
        if error:
            raise RuntimeError(f"{self.binary} run failed: {error}")
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {(proc.stderr or '')[-300:]}")
        if not text.strip():
            raise RuntimeError(f"{self.binary} produced no text (stderr: {(proc.stderr or '')[-200:]})")
        # The summariser's model goes in _meta — not the enriched session's.
        facet = parse_facet_json(text, self.name, self.model)
        facet.setdefault("_meta", {})
        facet["_meta"]["enrich_cost_usd"] = round(cost, 6)
        facet["_meta"]["enrich_tokens"] = tokens
        return facet
