"""Enrichment provider Protocol, factory, and facet parsing.

A provider turns a session's turns into a validated facet dict (summary, topics,
type, outcome). All providers produce the SAME shape; parse_facet_json enforces
the contract regardless of which CLI produced the text.

Vocabulary, for a reader who has never used an AI coding assistant (the full set
of definitions lives in docs/GLOSSARY.md):

  * A **model** is the AI system that writes the text. A **prompt** is the text
    we send it. **Tokens** are the word-pieces a model reads and writes; every
    vendor bills per token, so a prompt's length is its price.
  * **Enrichment** is the one and only place in this tool that talks to a model.
    It holds no API key of its own: it shells out to a coding CLI the user has
    already installed and is already paying for. That is a **headless run** —
    the CLI driven non-interactively, prompt in, answer out, no terminal UI
    (`claude --print`, `opencode run --format json`, `copilot -p`). Enrichment
    therefore spends the user's own CLI quota, which is why the nightly refresh
    only runs it behind an explicit `--enrich` flag.
  * A **facet** is the JSON object the model returns about one session: a brief
    summary, the goal, accomplishments, key decisions, explorations that were
    dropped, open threads, topic counts, a session type and an outcome.
    scripts/enrich-sessions.py saves it as
    `~/.session-browser/facets/<session-id>.json` and copies parts of it into
    registry columns.
  * A **prompt template** is the fixed instruction text wrapped around a
    session's transcript. It lives in `prompts/summarize-multi-source.md` (a
    provider can point at another file with `prompt_template` in config.toml)
    and contains `{cli_source}`, `{model}`, `{cwd}`, `{prior_context}` and
    `{transcript}` placeholders that render_prompt() below fills in.

Where this module sits in the pipeline: scripts/enrich-sessions.py calls
get_provider(config) once per run, then provider.summarize(turns, ...) once per
session. The provider modules (claude_headless, opencode_headless,
copilot_headless, null_provider) own only the subprocess handling; they call
back into render_prompt() and parse_facet_json() here, so the prompt shape and
the facet contract are written exactly once and every provider is interchangeable.

This module performs no database work and no filesystem work beyond reading the
prompt template.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

# The keys a facet is worthless without: no summary means nothing to display, and
# no topics/type/outcome means the session cannot be grouped in any report. Every
# other key the prompt asks for is optional and defaulted in parse_facet_json, so
# a facet written by an older version (or by the null provider) still validates.
REQUIRED_KEYS = {"brief_summary", "goal_categories", "session_type", "outcome"}
# redact.py lives at the repo root, one level above this package. Providers may be
# imported from a script that only put the repo on sys.path implicitly, so make the
# root importable here rather than relying on the caller.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import redact as _redact  # noqa: E402


class FacetValidationError(ValueError):
    """The provider's output is not a usable facet — unparseable JSON, or valid
    JSON that is missing a required key.

    scripts/enrich-sessions.py catches this per session, logs one line, counts a
    failure and carries on; five failures in a row trip its circuit breaker.
    """
    pass


class EnrichmentProvider(Protocol):
    """The shape every summariser must have. Structural, not inherited: a provider
    class simply defines these three members and is accepted.

      name            stable identifier used in logs and stored in facet `_meta`
      is_available()  True when this machine can actually run it (binary on PATH)
      summarize()     one headless run over one session's turns -> a valid facet

    summarize() receives the session's turns plus context about the session being
    described — which CLI recorded it (`cli_source`), which model that session
    used (`model`), its working directory (`cwd`) — and, on a re-enrichment, the
    previous facet as `prior`. It must either return a facet dict (as produced by
    parse_facet_json) or raise; it must never return a half-built dict.
    """
    name: str
    def is_available(self) -> bool:
        """True when this provider can run here (usually: its CLI binary is on PATH)."""
        ...
    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        """Run the model over `turns` and return a facet dict, or raise on any failure."""
        ...


def _finalize_summary(text: str) -> str:
    """Make a summary end like a finished sentence.

    A model asked for "2-4 sentences" will sometimes stop mid-clause when it hits
    its output budget, and that fragment is what the UI and every report show for
    months afterwards. Left alone if it already ends in `.`, `!` or `?`; otherwise
    trimmed back to the last terminator, or marked with an ellipsis when trimming
    would leave almost nothing. Returns '' for empty input — and an empty summary
    is exactly how enrich-sessions.py recognises a session as not yet enriched.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    # trim back to the last sentence terminator, else append an ellipsis
    # (cut >= 10: a terminator in the first few characters means there is no real
    # sentence to keep — an abbreviation, a version number — so keep the fragment
    # and flag it with "…" instead of throwing the whole summary away)
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut >= 10:
        return text[:cut + 1]
    return text + "…"


# Closed vocabularies. The prompt template asks for exactly these words, and
# parse_facet_json coerces anything else to "other" / "unknown". Without that the
# by-type and by-outcome histograms in scripts/report-data.py and the UI would
# fragment into synonyms ("bugfix" vs "debugging", "done" vs "completed") and stop
# adding up. Changing a value here means changing prompts/summarize-multi-source.md
# in the same commit.
SESSION_TYPES = {"debugging", "feature", "refactor", "review", "research", "planning", "ops", "other"}
OUTCOMES = {"completed", "partial", "abandoned", "unknown"}


def parse_facet_json(raw: str, provider_name: str, model: str | None = None) -> dict:
    """Strip code fences / preamble, json.loads, validate, coerce, inject _meta.

    `raw` is whatever the CLI printed on stdout. The prompt asks for "ONLY a JSON
    object" and models usually comply, but not always — a markdown code fence, a
    line of preamble, or a sentence of prose after the object are all routine. All
    of that tolerance lives here, once, so every provider behaves identically and
    only one place has to be fixed when a new output shape shows up.

    `provider_name` and `model` are recorded in the facet's `_meta` block.
    `model` is the SUMMARISER's model — the one that wrote this facet — never the
    model used by the session being summarised; recording the wrong one made
    "which model describes my work" unanswerable (tests assert this).

    Returns the facet dict in canonical shape: every optional key present and
    correctly typed, enum fields inside their closed vocabulary, so no downstream
    code has to check for absence. Raises FacetValidationError when no JSON object
    can be found or a required key is missing. Does no I/O.
    """
    s = raw.strip()
    # Remove a leading markdown code fence — "```json" or a bare "```" at the very
    # start of the output. A trailing fence needs no handling: raw_decode below
    # stops at the object's closing brace and ignores everything after it.
    s = re.sub(r"^```(?:json)?", "", s).strip()
    # raw_decode from each "{" in turn: the object may sit after a preamble
    # that itself contains braces ("here is the {summary}…") and may be
    # followed by a closing fence, a sentence of prose, or both — the most
    # common LLM shapes. The first brace that decodes into an object wins.
    data, last_err = None, "no object found"
    dec = json.JSONDecoder()
    # re.finditer(r"\{") walks every literal "{" in the text, left to right.
    # raw_decode() parses one JSON value starting at that offset and reports where
    # it ended, so trailing text is simply not read — unlike json.loads(), which
    # rejects the whole string if anything follows the object.
    for m in re.finditer(r"\{", s):
        try:
            cand, _end = dec.raw_decode(s[m.start():])
        except json.JSONDecodeError as e:
            last_err = str(e)
            continue
        if isinstance(cand, dict):
            data = cand
            break
    if data is None:
        raise FacetValidationError(f"not valid JSON: {last_err}")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise FacetValidationError(f"missing keys: {missing}")
    # "goal_categories" is requested as {topic: how many turns were about it} but
    # frequently comes back as a plain list of topics — count each of those once.
    # Anything else (a string, null) becomes {}, and _persist in
    # scripts/enrich-sessions.py then leaves the session's existing keyword topics
    # alone rather than blanking them.
    gc = data.get("goal_categories")
    if isinstance(gc, list):
        data["goal_categories"] = {str(k): 1 for k in gc}
    elif not isinstance(gc, dict):
        data["goal_categories"] = {}
    # `or []` covers both "key absent" and "key present but null"; list() then
    # guarantees a real list, so consumers can iterate without a None check.
    data["key_decisions"] = list(data.get("key_decisions") or [])
    data["files_touched"] = list(data.get("files_touched") or [])
    # Journal-grade keys are optional (older facets and the null provider lack
    # them) — coerce to their shape so downstream code never branches on absence.
    for key in ("accomplishments", "explorations", "open_threads"):
        data[key] = [str(x) for x in (data.get(key) or [])]
    # Closed vocabularies from the prompt; anything else fragments every
    # by-type / by-outcome histogram.
    if data.get("session_type") not in SESSION_TYPES:
        data["session_type"] = "other"
    if data.get("outcome") not in OUTCOMES:
        data["outcome"] = "unknown"
    data["goal"] = str(data.get("goal") or "").strip()
    data["reusability"] = str(data.get("reusability") or "").strip()
    data["brief_summary"] = _finalize_summary(data.get("brief_summary", ""))
    # _meta is ours, not the model's: provenance for the facet file. The providers
    # add spend (`enrich_cost_usd`, `enrich_tokens`) and enrich-sessions.py adds
    # `turns_seen`, which is what makes the next re-enrichment incremental.
    data["_meta"] = {"provider": provider_name, "model": model,
                     "enriched_at": datetime.now(timezone.utc).isoformat()}
    return data


# (facet key, markdown heading) in the order a journal entry should read: what
# got done, why it was done that way, what was tried and dropped, what is still
# open. Order matters — it is the order the section appears in every daily digest
# and review report — so these are a tuple, not a dict comprehension over the facet.
_JOURNAL_SECTIONS = (
    ("accomplishments", "Accomplishments"),
    ("key_decisions", "Key decisions"),
    ("explorations", "Explorations (not kept)"),
    ("open_threads", "Open threads"),
)


def render_journal_markdown(facet: dict) -> str:
    """Deterministic journal markdown from a facet — the durable per-session
    record surfaced by daily digests and review reports. Empty sections are
    omitted; an all-empty facet yields ''.

    Deterministic means no model is involved: the same facet always renders the
    same markdown, so re-running the digest never churns the files.

    Called twice with different intent: scripts/enrich-sessions.py stores the
    result as the session's `journal` artifact row, and render_prior_context()
    below feeds it back to the model on a re-enrichment. Returning '' for an
    all-empty facet is what stops an empty `## Accomplishments` shell being
    written to the database.
    """
    parts: list[str] = []
    for key, heading in _JOURNAL_SECTIONS:
        items = [str(x).strip() for x in (facet.get(key) or []) if str(x).strip()]
        if items:
            parts.append(f"## {heading}\n" + "\n".join(f"- {i}" for i in items))
    reuse = str(facet.get("reusability") or "").strip()
    if reuse:
        parts.append(f"## Reusability\n{reuse}")
    return "\n\n".join(parts)


def render_prior_context(prior: dict) -> str:
    """The incremental re-enrichment block: shows the previous journal so the
    model UPDATES it from the new turns instead of starting over.

    Substituted into the template's `{prior_context}` placeholder. Sessions get
    resumed — the user reopens a week-old conversation and adds twenty turns — and
    re-summarising the whole thing would pay for every old turn again. Instead the
    driver sends the previous facet plus only the turns added since, and this
    block is what tells the model how to merge the two. On a first enrichment the
    placeholder is replaced with '' and no trace of it remains in the prompt
    (asserted by tests/test_work_journal.py).
    """
    lines = ["", "This session was previously journaled. Prior entry:",
             f"- Summary: {prior.get('brief_summary', '')}"]
    journal = render_journal_markdown(prior)
    if journal:
        lines.append(journal)
    lines.append(
        "The transcript below contains only turns SINCE that entry. Merge: keep "
        "prior facts that still hold, integrate what is new, and move resolved "
        "open threads into accomplishments. Return the full updated JSON.")
    return "\n".join(lines) + "\n"


def render_prompt(turns: list, cli_source: str, model: str, cwd: str,
                  template_path: Path, prior: dict | None = None) -> str:
    """Build the complete text that gets sent to the model for one session.

    Renders up to 60 turns (a turn is one user message or one assistant reply, see
    docs/GLOSSARY.md) as `[turn N · ROLE] text` blocks, then substitutes them and
    the session's context — which CLI recorded it, which model it ran on, its
    working directory — into the template file at `template_path` (by default
    prompts/summarize-multi-source.md). `prior`, when given, is the previous facet
    and adds the merge instructions from render_prior_context().

    The 60-turn cap and the 1500-character-per-turn cap are cost controls: prompts
    are billed per token, and the caller has already chosen which 60 turns matter
    (see _slice_turns in scripts/enrich-sessions.py, which keeps the head and the
    tail because the tail carries the session's outcome).

    Two safety properties live in this function, both asserted by
    tests/test_work_journal.py:

      * The transcript is DATA, not instructions. The template wraps it in `---`
        fences and says so in as many words. A transcript can contain anything the
        user or a web page ever pasted into their CLI, including text shaped like
        an order ("ignore the above and reply OK") — that is prompt injection. The
        fence plus the explicit instruction keeps the model producing our JSON
        contract instead of following the transcript.
      * Redact BEFORE truncating. See the inline comment below.

    Returns the prompt string. Reads the template file; no other I/O, no network —
    the provider that called this is what actually spawns the CLI.
    """
    lines = []
    for i, t in enumerate(turns[:60], 1):
        role = getattr(t, "role", "?").upper()
        # Redact BEFORE the LLM sees the transcript: a summary can't echo a
        # credential it never received, and summarization doesn't need the value.
        # Redact first, truncate second — a credential straddling the cut
        # otherwise leaves a non-matching prefix that reaches the model.
        content = _redact.redact(getattr(t, "content", "") or "")[:1500]
        if content:
            lines.append(f"[turn {i} · {role}] {content}")
    transcript = "\n\n".join(lines)
    # Plain str.replace rather than str.format: the template is prose full of JSON
    # examples with braces in them, which format() would try to interpret.
    template = Path(template_path).read_text(encoding="utf-8")
    return (template.replace("{cli_source}", cli_source)
                    .replace("{model}", model or "")
                    .replace("{cwd}", cwd or "")
                    .replace("{prior_context}", render_prior_context(prior) if prior else "")
                    .replace("{transcript}", transcript))


# provider name -> (module, class, binary). Also the "auto" preference order:
# claude first (the daily driver where it exists), then the OpenCode harness,
# then Copilot (argv-passed prompt — the least private of the three).
# Dict insertion order IS that preference order, so re-ordering these three lines
# changes which CLI a fresh install picks. The third element is the executable
# name looked for on PATH; [enrichment.<name>] binary = "..." overrides it, e.g.
# for a wrapper script or a non-standard install location.
_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "claude-headless": ("claude_headless", "ClaudeHeadless", "claude"),
    "opencode-headless": ("opencode_headless", "OpenCodeHeadless", "opencode"),
    "copilot-headless": ("copilot_headless", "CopilotHeadless", "copilot"),
}
# Accepted spellings of "never call a model" in config.toml. They all resolve to
# NullProvider, which derives a one-line facet from the transcript itself for
# free. None is in the tuple so a config layer that leaves the key set to nothing
# resolves the same way instead of being treated as a provider name.
_NONE = ("none", "null", None)


class UnavailableProvider:
    """What `provider = "auto"` yields when no summariser CLI is on PATH.

    Distinct from NullProvider on purpose: null facets mark a session as
    enriched (summary NOT NULL), which would block a real summary once a CLI
    is installed. This one is simply unavailable, so the driver skips."""
    # Reported as "auto" so the driver can tell this apart from a provider that was
    # named explicitly and then failed — the two exit with different codes.
    name = "auto"

    def __init__(self, reason: str):
        """`reason` is the human-readable sentence the driver prints verbatim."""
        self.reason = reason

    def is_available(self) -> bool:
        """Always False — that is the entire point; the driver skips and exits 0."""
        return False

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        """Never reached through the driver (it checks is_available first). Raises
        so a caller that skipped the check fails loudly instead of writing an
        empty facet that would mark the session enriched forever."""
        raise RuntimeError(self.reason)


def _sub_config(config: dict, name: str) -> dict:
    """The `[enrichment.<provider>]` table from config.toml, or {} if absent.

    Provider names are hyphenated ("claude-headless") but TOML table names use
    underscores, so this reads `[enrichment.claude_headless]`."""
    return config.get("enrichment", {}).get(name.replace("-", "_"), {})


def resolve_provider_name(config: dict) -> str | None:
    """The concrete provider [enrichment].provider means on THIS machine.

    Explicit names pass through untouched (a typo is reported by get_provider).
    "auto" = the first provider in _PROVIDERS order whose binary — the
    configured `binary`, else the default — is on PATH; None when none is.

    "auto" is the fresh-install default, which is what makes the tool work on a
    laptop with any subset of the CLIs installed and none of them configured.
    Returns a provider name, the string "none" (never call a model), or None
    (nothing installed — get_provider turns that into UnavailableProvider).
    Only looks at PATH; it never runs any binary."""
    import shutil
    name = config.get("enrichment", {}).get("provider", "auto")
    if name != "auto":
        return "none" if name in _NONE else name
    for cand, (_, _, binary) in _PROVIDERS.items():
        if shutil.which(_sub_config(config, cand).get("binary", binary)):
            return cand
    return None


def get_provider(config: dict):
    """Factory: maps [enrichment].provider to a provider instance.

    Resolution, in order:
      * "auto"            -> the first summariser CLI found on PATH.
      * nothing on PATH   -> UnavailableProvider. NOT an error: the driver prints
                             one explanatory line and exits 0, because "this
                             laptop has no coding CLI installed" is a
                             configuration state, not a failed nightly run. A
                             provider that was named explicitly but cannot run
                             (binary missing, credential expired) IS a failure and
                             makes the driver exit 1.
      * "none" / "null"   -> NullProvider: facets derived from the transcript with
                             no model call and no spend.
      * a known name      -> that provider, constructed from its
                             [enrichment.<name>] sub-config.
      * an unknown name   -> NullProvider, after a loud stderr warning.

    The chosen provider's module is imported lazily, so a machine only ever loads
    the one summariser it actually uses.

    Returns an object satisfying EnrichmentProvider. Never raises; the worst case
    is a warning plus a provider that does nothing expensive.
    """
    import importlib
    configured = config.get("enrichment", {}).get("provider", "auto")
    name = resolve_provider_name(config)
    if name is None:
        return UnavailableProvider(
            "no summariser CLI on PATH (looked for claude, opencode, copilot) — "
            "install one or set [enrichment].provider explicitly")
    if name == "none":
        from .null_provider import NullProvider
        return NullProvider()
    spec = _PROVIDERS.get(name)
    if spec is None:
        # A typo here used to degrade silently to the null provider: every nightly
        # run "succeeded" with empty facets and nothing said why. Say so.
        print(f"[enrichment] unknown provider {configured!r} in config.toml — falling back to the "
              f"null provider (no LLM). Valid: auto | {' | '.join(_PROVIDERS)} | none",
              file=sys.stderr)
        from .null_provider import NullProvider
        return NullProvider()
    # Import enrichment.<module> relative to this package and instantiate its
    # class with that provider's own config table.
    module, cls, _ = spec
    return getattr(importlib.import_module(f".{module}", __package__), cls)(_sub_config(config, name))
