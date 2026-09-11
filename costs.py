"""Token → USD cost helpers, shared by compute-costs.py and the Flask app.

A **token** is the unit an AI model reads and writes text in — roughly four characters of
English, so ~750 words is ~1000 tokens. Every request a coding CLI makes is billed per
token, which is why token counts are the raw measure of "how much did this session use".
There are four kinds, and they cost wildly different amounts:

  * input       — tokens sent to the model (your prompt plus the conversation so far)
  * output      — tokens the model generated (its reply); several times dearer than input
  * cache_write — the first time a long, unchanging prefix of a conversation is stored on
                  the vendor's side so later requests need not re-send it; a bit dearer
                  than a plain input token
  * cache_read  — re-using that stored prefix on a later request; roughly a tenth the
                  price of an input token, which is why long sessions cost far less than
                  their raw size suggests

See docs/GLOSSARY.md ("Token", "Cache read / cache write tokens", "Pricing tier").

Pricing comes from pricing.json (rates per million tokens). Model strings are
matched to a tier by substring alias (longest alias first so 'gpt-5-mini' wins
over 'gpt-5'). Token-key mapping (verified against real transcripts):
    input_tokens               -> input
    output_tokens              -> output
    cache_read_input_tokens    -> cache_read
    cache_creation_input_tokens-> cache_write   (may be a dict; sum sub-fields)

Who uses this module: scripts/compute-costs.py walks every transcript, totals its tokens
and writes the USD figure into the registry; session-ui/app.py re-prices on the fly for
its usage panels. Nothing here touches the network or the database — it is pure
arithmetic over pricing.json, so it is cheap to call in a loop.

Caveat worth repeating to a newcomer: under a flat monthly subscription no money is
actually billed per session, so the figure is the public list-price *equivalent* and the
UI shows it with `≈` as an intensity signal (see docs/ARCHITECTURE.md, "Cost is notional
under a subscription").
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sbconfig

# Process-wide memo of the parsed pricing.json, keyed on the file's modification time.
# compute-costs.py calls load_pricing() once per session and the Flask app once per
# request, so re-reading and re-parsing the JSON every time would be pure waste — but
# editing pricing.json must take effect without a restart, hence the mtime check rather
# than a plain "load once".
_cache: dict = {"mtime": 0, "data": None}


def load_pricing() -> dict:
    """The parsed pricing.json, cached until the file's mtime changes.

    Shape (see pricing.json itself, which documents where the numbers came from):
        {"tiers":    {"opus-4.5": {"input": 5.0, "output": 25.0,
                                   "cache_read": 0.5, "cache_write": 6.25}, ...},
         "aliases":  {"opus-4-5": "opus-4.5", "gpt-5-mini": "gpt-5-mini", ...},
         "gateway_discount": 1.0}
    Every tier number is USD per MILLION tokens.

    A missing or unreadable pricing.json is not an error: it returns empty tables, which
    makes every model unpriced, which callers turn into "leave the stored cost alone"
    rather than a crash or a wrong $0. Raises only if the file exists but holds invalid
    JSON (a typo in the table should be loud, not silently priced at zero).
    """
    p = sbconfig.PRICING_PATH
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {"tiers": {}, "aliases": {}, "gateway_discount": 1.0}
    if _cache["data"] is None or mtime != _cache["mtime"]:
        with open(p) as fh:
            _cache["data"] = json.load(fh)
        _cache["mtime"] = mtime
    return _cache["data"]


def tier_for_model(model: str, pricing: dict | None = None) -> str | None:
    """Map a model name to its pricing tier, or None when nothing matches.

    Transcripts record whatever model string the CLI used, and those strings drift
    constantly: `claude-opus-4-5-20260514`, `anthropic/claude-sonnet-5`, `gpt-5-mini`.
    Rather than enumerate every exact id, pricing.json lists short ALIASES and we take
    the longest alias that appears anywhere in the (lower-cased) model string. Longest
    first is the whole trick: `gpt-5` is a substring of `gpt-5-mini`, so scanning
    shortest-first would price the cheap mini model at the flagship rate.

    Returning None (not a guessed tier, not a default) is deliberate: callers print a
    loud "unknown model" line and refuse to overwrite an already-stored cost, so a new
    model release shows up as a visible gap instead of a quietly wrong dollar figure.
    tests/test_smoke.py::test_cost_mapping pins exactly this behaviour.
    """
    pricing = pricing or load_pricing()
    model = (model or "").lower()
    # longest alias first to avoid 'gpt-5' shadowing 'gpt-5-mini'
    for alias in sorted(pricing.get("aliases", {}), key=len, reverse=True):
        if alias in model:
            return pricing["aliases"][alias]
    return None


def cost_usd(model: str, tokens: dict, pricing: dict | None = None) -> float:
    """tokens = {input, output, cache_read, cache_write}.

    Returns the public list price in USD for those token counts at this model's tier.
    An unknown model (no alias) or an empty pricing table yields 0.0 — the caller, not
    this function, decides what an unpriced session means (compute-costs.py treats it as
    "don't touch the stored cost"; it never writes the 0.0 through).

    `gateway_discount` is a single multiplier for people who route through a reseller or
    enterprise gateway that bills a flat percentage off list; it is 1.0 by default.
    Missing token keys count as zero, so a source that reports only input/output works.
    """
    pricing = pricing or load_pricing()
    tier = tier_for_model(model, pricing)
    rates = pricing.get("tiers", {}).get(tier or "", {})
    if not rates:
        return 0.0
    disc = pricing.get("gateway_discount", 1.0)
    total = 0.0
    # Rates are per million tokens, hence the /1e6 on each of the four token kinds.
    for key in ("input", "output", "cache_read", "cache_write"):
        total += (tokens.get(key, 0) or 0) / 1e6 * rates.get(key, 0.0)
    return total * disc


def coerce_cache_write(val) -> int:
    """cache_creation_input_tokens may be an int or a dict of ephemeral sub-tiers.

    Anthropic's caches have lifetimes, and newer transcripts break the cache-write count
    out per lifetime instead of reporting one number, e.g.
        {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 5}  -> 15
    Both lifetimes are billed at the same cache-write rate, so summing is correct.
    Non-numeric values are ignored; None or a missing field becomes 0.
    """
    if isinstance(val, dict):
        return sum(int(v) for v in val.values() if isinstance(v, (int, float)))
    return int(val or 0)
