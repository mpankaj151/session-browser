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

Vocabulary, for a reader new to AI tooling (docs/GLOSSARY.md has the rest):
**enrichment** is this tool's only call to an AI **model** — read one session,
return a small JSON summary called a **facet**. A **headless run** means driving
a coding CLI non-interactively instead of through its terminal UI; here that is
`opencode run --format json`, prompt on stdin, machine-readable events on stdout.
The call is billed to whichever provider credential OpenCode is configured with,
i.e. the user's own quota. **NDJSON** ("newline-delimited JSON") is one complete
JSON object per line, read as the process streams it.

Where it sits: scripts/enrich-sessions.py gets this class from
enrichment/provider.py when `[enrichment].provider` is "opencode-headless" (or
when "auto" finds no `claude` but does find `opencode`). This module owns the
subprocess and the event stream only; provider.render_prompt() builds the prompt
and provider.parse_facet_json() validates the reply.
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
    """Summariser backed by `opencode run`. Satisfies provider.EnrichmentProvider."""

    name = "opencode-headless"

    def __init__(self, config: dict):
        """`config` is the `[enrichment.opencode_headless]` table from config.toml
        ({} when absent — every key below has a default).

          binary, model, timeout_secs, prompt_template — as for the other providers
          agent    an OpenCode "agent" (a named preset). Optional; a no-tools
                   summariser agent is a useful belt to the OPENCODE_PERMISSION
                   braces, but see env() for why we never rely on it alone.
          variant  OpenCode's reasoning-effort knob, e.g. "minimal". Less
                   deliberation means fewer output tokens, so less spend, which is
                   the right trade for filling in a fixed JSON template.

        The timeout defaults higher than the other providers' (300s vs 180s)
        because OpenCode retries connection errors internally before giving up.
        """
        self.binary = config.get("binary", "opencode")
        self.model = config.get("model", _DEFAULT_MODEL)
        self.agent = config.get("agent", "")        # optional PRIMARY agent (no-tools summariser)
        self.variant = config.get("variant", "")    # optional reasoning effort, e.g. "minimal"
        self.timeout = int(config.get("timeout_secs", 300))
        self.template = _REPO / "prompts" / config.get("prompt_template", "summarize-multi-source.md")

    def command(self) -> list[str]:
        """The argv to spawn. Split out from summarize() so tests can assert the
        exact flag set without running anything — the flags below are load-bearing.

        `--pure` stops OpenCode loading external plugins, including the Session
        Browser plugin the user may have installed, so an enrichment run cannot
        re-enter our own indexing pipeline. `--format json` switches stdout to the
        NDJSON event stream parsed by _parse_stream(); the human-facing chatter
        goes to stderr, which keeps stdout clean enough to parse.

        The prompt is NOT in this argv: it goes on stdin (see summarize), because
        OpenCode wraps a positional argument containing spaces in literal quotes
        and escapes the quotes inside it, which mangles a transcript.
        """
        args = [self.binary, "run", "--pure", "--format", "json"]
        if self.model:
            args += ["--model", self.model]
        if self.agent:
            args += ["--agent", self.agent]
        if self.variant:
            args += ["--variant", self.variant]
        # Never --auto / --share / --attach: permissions must be auto-rejected,
        # the transcript must never be uploaded, and --attach breaks exit codes.
        # (--auto would approve every tool request the model makes. A summariser
        # has no business reading files or running commands, and without --auto a
        # non-interactive run rejects them all automatically — so the safe
        # behaviour is simply the default, as long as nobody adds the flag.)
        # --title does double duty: it tags the run so sources/opencode.py can skip
        # it when mirroring, AND it stops OpenCode making a SECOND, separately
        # billed model call just to invent a title for the session.
        args += ["--title", sbconfig.OPENCODE_ENRICHMENT_TITLE]
        return args

    def env(self) -> dict[str, str]:
        """Environment for the run. OPENCODE_DB=:memory: is the isolation lever:
        no session rows persist anywhere, while auth.json (resolved from the
        data dir, not the DB path) keeps working — XDG_DATA_HOME would move
        the credentials too. OPENCODE_PERMISSION is deep-merged over config,
        so it still denies everything if --agent silently falls back to the
        all-tools default; invalid JSON there is silently ignored by OpenCode,
        hence a constant, never built dynamically.

        Longer version, because the details are easy to get wrong:

          * OpenCode normally records every session it runs into its own SQLite
            database, so enrichment would litter the user's session list with
            hundreds of "summarise this transcript" runs — and this tool would
            then index them. `:memory:` gives the process a throwaway in-RAM
            database instead. Crucially it moves ONLY the database. The stored
            credentials (auth.json) are found through the data directory, so
            they keep working; reaching for XDG_DATA_HOME instead would relocate
            the credentials too and every run would fail unauthenticated.
          * OPENCODE_PERMISSION is JSON meaning "deny every tool request". It is
            deep-merged over whatever the config file says, so it still holds if
            `--agent` is misspelled and OpenCode quietly falls back to its
            default all-tools agent. Two properties make it a hard-coded
            constant: OpenCode ignores invalid JSON here in silence (a typo would
            disable the protection with no error), and building it dynamically
            would put that silent failure one f-string away.
          * SESSION_BROWSER_SUPPRESS_HOOK is for symmetry with the other
            providers — an OpenCode run never fires Claude Code's hooks anyway.

        Returns a full environment (os.environ plus the overrides), ready to pass
        as subprocess `env=`.
        """
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
        """Whether `opencode` is on PATH. Never executes it — a probe run costs
        tokens, and the driver calls this on every startup."""
        return shutil.which(self.binary) is not None

    # Injectable for tests; the real thing is subprocess.run.
    # Tests replace this attribute with a fake so the whole provider can be
    # exercised — flags, environment, stream parsing, error paths — without ever
    # spawning OpenCode or spending anything.
    _run = staticmethod(subprocess.run)

    @staticmethod
    def _parse_stream(stdout: str) -> tuple[str, float, dict, str | None]:
        """NDJSON -> (text, cost_usd, tokens, error). Non-JSON lines are ignored
        (stdout is clean by construction, but never let a stray line kill a run).

        `--format json` prints one JSON object per line as the run progresses.
        Only three event types matter; everything else is skipped:

          {"type":"text","part":{"type":"text","text":"..."}}
              a slice of the model's answer. There are many, and the facet JSON is
              split across them arbitrarily (a fence can end one part and the
              object start in the next), so they are concatenated before parsing.
          {"type":"step_finish","part":{"cost":0.013,"tokens":{"input":1250,
                                                               "output":310}}}
              what that step actually cost. Summed across steps, this is the only
              provider that can report real spend per enrichment.
          {"type":"error","error":{"name":"...","data":{"message":"..."}}}
              the run failed. Reported before the exit code is even looked at,
              because the message names the cause and the exit code does not.

        Returns (joined text, total cost in USD, {"input": n, "output": n}, error
        message or None). Pure function — no I/O — which is why tests can drive it
        with a hand-written stream.
        """
        texts: list[str] = []
        cost = 0.0
        tokens = {"input": 0, "output": 0}
        error: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            # Cheap guard before paying for a JSON parse: a real event line always
            # starts with "{". Blank lines and any stray banner are dropped here.
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
        """Run one headless call and return the validated facet.

        `turns` are the (already sliced) turns to describe; `cli_source`, `model`
        and `cwd` describe the session BEING summarised and go into the prompt as
        context; `prior` is the previous facet on a re-enrichment.

        Spawns OpenCode with the prompt on stdin, parses its event stream, and
        records what the call actually cost in the facet's `_meta`. Runs from the
        user's home directory: the cwd is irrelevant to a summariser, and a
        neutral one keeps the run out of any project.

        Raises RuntimeError for every failure mode, each with the cause spelled
        out: timeout, an `error` event, a non-zero exit, or an empty answer. The
        driver counts any of them as one failed session and moves on.
        """
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
        # Order matters: an `error` event names the actual cause ("provider
        # returned 401"), whereas the exit code alone says only "it failed".
        if error:
            raise RuntimeError(f"{self.binary} run failed: {error}")
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {(proc.stderr or '')[-300:]}")
        if not text.strip():
            raise RuntimeError(f"{self.binary} produced no text (stderr: {(proc.stderr or '')[-200:]})")
        # The summariser's model goes in _meta — not the enriched session's.
        # i.e. "anthropic/claude-sonnet-5 wrote this facet", never "the session
        # being described ran on claude-opus-5". Asserted by the journal tests.
        facet = parse_facet_json(text, self.name, self.model)
        facet.setdefault("_meta", {})
        # Real spend, unlike the other two providers, which can only record None.
        # 6 decimal places: an individual enrichment costs fractions of a cent.
        facet["_meta"]["enrich_cost_usd"] = round(cost, 6)
        facet["_meta"]["enrich_tokens"] = tokens
        return facet
