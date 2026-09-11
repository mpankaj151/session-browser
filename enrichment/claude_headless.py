"""Enrichment via the Claude CLI in headless mode (`claude --print`).

Sends the rendered transcript prompt to the user's existing Claude backend (no
extra API key) and parses the returned facet JSON.

For a reader new to this vocabulary (see docs/GLOSSARY.md for the rest):
**enrichment** is the one place this tool asks an AI **model** to do something —
read one session and return a short structured summary, the **facet**. A
**headless run** means the coding CLI is driven non-interactively rather than
through its terminal UI: `claude --print` takes the **prompt** (the text sent to
the model) on standard input and writes the answer to standard output. Because
it runs the user's own installed CLI, the call is billed against the user's own
Claude subscription or API quota — this module never holds a key of its own.

Position in the pipeline: scripts/enrich-sessions.py picks this class via
enrichment/provider.py when `[enrichment].provider` is "claude-headless" (or
when "auto" finds `claude` first on PATH). It owns only the subprocess; the
prompt is built by provider.render_prompt() and the reply validated by
provider.parse_facet_json(), so all providers are interchangeable.

Two gotchas worth knowing before editing:
  * Every headless call makes the Claude CLI write a transcript of ITSELF. Left
    unchecked those enrichment runs would be indexed as sessions, which is why
    run_cwd() below exists.
  * The spawned CLI inherits our environment, so the Stop hook has to be told to
    stand down (`SESSION_BROWSER_SUPPRESS_HOOK`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .provider import parse_facet_json, render_prompt

# The spawned CLI inherits this; our Stop hook sees it and no-ops, so headless
# enrichment sessions can never re-enter the indexing pipeline via the hook.
# (A "hook" is a program the CLI runs when a reply finishes — scripts/session-hook.py
# here, which normally indexes the session that just ended. Without this variable an
# enrichment run would index itself and then be enriched, forever.)
_SUPPRESS = {"SESSION_BROWSER_SUPPRESS_HOOK": "1"}

# Repo root — the prompt templates live in <repo>/prompts/.
_REPO = Path(__file__).resolve().parent.parent

# Journal extraction is structured summarization — Sonnet-tier quality at a
# fraction of Opus/Fable cost. Pinning (rather than inheriting whatever model
# the user's CLI defaults to) keeps enrichment spend predictable on both
# flat-rate plans and API billing. Override with [enrichment.claude_headless]
# model = "..." in config.toml; set model = "" to fall back to the CLI default.
_DEFAULT_MODEL = "claude-sonnet-5"


class ClaudeHeadless:
    """Summariser backed by `claude --print`. Satisfies provider.EnrichmentProvider."""

    name = "claude-headless"

    def __init__(self, config: dict):
        """`config` is the `[enrichment.claude_headless]` table from config.toml
        (an empty dict when the block is absent — every key has a default).

          binary           executable to run; override for a wrapper or odd path
          model            summariser model, see _DEFAULT_MODEL above
          exec_args        flags before the model flag; `--print` is what makes
                           the run headless
          timeout_secs     hard ceiling on one call (see summarize)
          prompt_template  filename under <repo>/prompts/
        """
        self.binary = config.get("binary", "claude")
        self.model = config.get("model", _DEFAULT_MODEL)
        self.exec_args = config.get("exec_args", ["--print"])
        self.timeout = int(config.get("timeout_secs", 180))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def command(self) -> list[str]:
        """The argv to spawn, e.g. `['claude', '--print', '--model', 'claude-sonnet-5']`.

        Split out from summarize() so tests can assert the model pin without
        running anything. An empty `model` omits `--model` entirely, which lets
        the CLI's own default win."""
        args = [self.binary, *self.exec_args]
        if self.model:
            args += ["--model", self.model]
        return args

    def is_available(self) -> bool:
        """Whether `claude` is on PATH. Nothing is executed — a probe run would
        cost tokens, and the driver calls this on every startup."""
        return shutil.which(self.binary) is not None

    @staticmethod
    def run_cwd() -> str:
        """Every headless call writes an sdk-cli transcript under its cwd. Run
        from one dedicated directory that the Claude adapter refuses to index
        BY PATH — a filesystem marker no upstream field rename can break.

        Claude Code records a session per working directory: it turns the cwd into
        a folder name under `~/.claude/projects/` (`/Users/me/app` becomes
        `-Users-me-app/`) and drops a `<uuid>.jsonl` transcript there. So every
        enrichment run leaves a transcript of itself behind, and if the run
        happened in a real project directory those files would land among that
        project's genuine sessions and be indexed as work the user never did.

        Fix: always run from `<data dir>/enrichment-cwd` (in practice
        `~/.session-browser/enrichment-cwd`). sources/claude.py skips that one
        project folder by path, which is robust in a way that filtering on a JSON
        field inside the transcript is not — upstream can rename fields, but our
        own directory name is ours. Covered by the "dedicated cwd that is never
        indexed" test in tests/test_work_journal.py.

        Falls back to the parent data directory if the folder cannot be created
        (read-only home, quota): still not a project folder, so still safe.
        Imported inside the function to keep module import cheap for callers that
        only need is_available().
        """
        import sbconfig
        d = sbconfig.FACETS_DIR.parent / "enrichment-cwd"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return str(sbconfig.FACETS_DIR.parent)
        return str(d)

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        """Run one headless call and return the validated facet.

        `turns` are the (already sliced) turns to describe; `cli_source`, `model`
        and `cwd` describe the session BEING summarised, and go into the prompt as
        context. `prior` is the previous facet on a re-enrichment.

        Spawns a subprocess, writes the prompt to its stdin and reads the reply
        from stdout — the only network activity in this tool, paid for out of the
        user's own Claude quota. Raises RuntimeError on timeout or a non-zero exit
        (stderr truncated to 200 characters so a stack trace cannot flood the
        nightly log), and FacetValidationError from parse_facet_json if the reply
        is not a usable facet. The driver counts either as one failed session.

        The timeout matters: a hung CLI would otherwise stall a nightly sweep of
        hundreds of sessions indefinitely.
        """
        prompt = render_prompt(turns, cli_source, model, cwd, self.template, prior=prior)
        try:
            proc = subprocess.run(
                self.command(),
                input=prompt, capture_output=True, text=True, timeout=self.timeout,
                env={**os.environ, **_SUPPRESS}, cwd=self.run_cwd(),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{self.binary} timed out after {self.timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:200]}")
        # _meta.model is the SUMMARISER's model (this provider's pin), not the
        # enriched session's; `claude --print` exposes no per-call cost.
        facet = parse_facet_json(proc.stdout, self.name, self.model or None)
        # Explicit None, not 0.0: "this provider cannot tell us what the call cost"
        # must not be reported as "the call was free". The OpenCode provider does
        # know, and records a real number there.
        facet["_meta"]["enrich_cost_usd"] = None
        return facet
