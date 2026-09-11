"""Enrichment via the Copilot CLI in headless mode (`copilot -p`).

Note: the Copilot CLI only accepts the prompt as an argv parameter (no stdin
mode), so the rendered transcript is briefly visible in the local process list
while the call runs. The prompt is capped well below ARG_MAX so the call can
never fail with E2BIG. If Copilot gains a stdin prompt mode, switch to it (see
claude_headless.py for the pattern).

Vocabulary, for a reader new to AI tooling (docs/GLOSSARY.md has the rest):
**enrichment** is this tool's only call to an AI **model** — read one session and
return a small JSON summary, the **facet**. A **headless run** means driving a
coding CLI non-interactively rather than through its terminal UI; `copilot -p`
takes the **prompt** (the text sent to the model) and prints the answer. The call
is billed against the user's own GitHub Copilot subscription.

Unpacking the note above: "argv" is the list of command-line arguments a process
is started with, and on Unix any user on the machine can read another process's
argv (`ps auxww`). Passing a transcript that way is the least private of the
three providers, which is exactly why enrichment/provider.py puts Copilot last in
the "auto" preference order. ARG_MAX is the kernel's ceiling on the total size of
that argument list — exceed it and the spawn fails outright with E2BIG — hence
the hard cap in summarize() below.

Where it sits: chosen by enrichment/provider.py when `[enrichment].provider` is
"copilot-headless", or when "auto" finds neither `claude` nor `opencode`. The
prompt is built by provider.render_prompt() and the reply validated by
provider.parse_facet_json(); this module owns only the subprocess.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .provider import parse_facet_json, render_prompt

# Inherited by the spawned CLI; our Claude Stop hook sees it and no-ops. Kept for
# symmetry with the other providers — a Copilot run does not fire Claude's hooks.
_SUPPRESS = {"SESSION_BROWSER_SUPPRESS_HOOK": "1"}

# Repo root — the prompt templates live in <repo>/prompts/.
_REPO = Path(__file__).resolve().parent.parent


class CopilotHeadless:
    """Summariser backed by `copilot -p`. Satisfies provider.EnrichmentProvider.

    Note the absence of a model pin, unlike the other two providers: the Copilot
    CLI does not take a model flag, so the run uses whatever Copilot is set to.
    That is why parse_facet_json is handed "" as the summariser model below.
    """

    name = "copilot-headless"

    def __init__(self, config: dict):
        """`config` is the `[enrichment.copilot_headless]` table from config.toml
        ({} when absent — binary, exec_args, timeout_secs and prompt_template all
        have defaults). `-p` is what makes the run headless."""
        self.binary = config.get("binary", "copilot")
        self.exec_args = config.get("exec_args", ["-p"])
        self.timeout = int(config.get("timeout_secs", 180))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def is_available(self) -> bool:
        """Whether `copilot` is on PATH. Never executes it — a probe run costs
        tokens, and the driver calls this on every startup."""
        return shutil.which(self.binary) is not None

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        """Run one headless call and return the validated facet.

        `turns` are the (already sliced) turns to describe; `cli_source`, `model`
        and `cwd` describe the session BEING summarised and go into the prompt as
        context; `prior` is the previous facet on a re-enrichment.

        Spawns a subprocess with the prompt as an argument and reads the reply from
        stdout — the only network activity in this tool, paid for out of the user's
        own Copilot quota. Raises RuntimeError on timeout or a non-zero exit, and
        FacetValidationError from parse_facet_json if the reply is not a usable
        facet; the driver counts either as one failed session.
        """
        prompt = render_prompt(turns, cli_source, model, cwd, self.template, prior=prior)
        # 120k characters: render_prompt already caps 60 turns at 1500 characters
        # each, so this is a second, absolute belt against a spawn failing with
        # E2BIG. Typical ARG_MAX is ~1 MB on Linux and ~256 KB on macOS.
        prompt = prompt[:120_000]  # argv-passed; stay far below ARG_MAX
        try:
            proc = subprocess.run(
                [self.binary, *self.exec_args, prompt],
                capture_output=True, text=True, timeout=self.timeout,
                env={**os.environ, **_SUPPRESS},
            )
        except subprocess.TimeoutExpired as e:
            # TimeoutExpired.__str__ interpolates argv — here the whole prompt —
            # and would land the transcript in the nightly error log.
            # So the exception is replaced rather than re-raised: the message must
            # stay short and boring. (`from e` keeps the original as the chained
            # cause, which is only ever seen by a human reading a traceback.)
            raise RuntimeError(f"{self.binary} timed out after {self.timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:200]}")
        # Copilot pins no model; the enriched session's model is NOT the summariser's.
        # Hence "" rather than `model`: we genuinely do not know which model wrote
        # this facet, and claiming the summarised session's model would be a lie.
        facet = parse_facet_json(proc.stdout, self.name, "")
        # Explicit None, not 0.0: "this provider cannot report what the call cost"
        # is not the same claim as "the call was free".
        facet["_meta"]["enrich_cost_usd"] = None
        return facet
