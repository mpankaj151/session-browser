"""Token → USD cost helpers, shared by compute-costs.py and the Flask app.

Pricing comes from pricing.json (rates per million tokens). Model strings are
matched to a tier by substring alias (longest alias first so 'gpt-5-mini' wins
over 'gpt-5'). Token-key mapping (verified against real transcripts):
    input_tokens               -> input
    output_tokens              -> output
    cache_read_input_tokens    -> cache_read
    cache_creation_input_tokens-> cache_write   (may be a dict; sum sub-fields)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sbconfig

_cache: dict = {"mtime": 0, "data": None}


def load_pricing() -> dict:
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
    pricing = pricing or load_pricing()
    model = (model or "").lower()
    # longest alias first to avoid 'gpt-5' shadowing 'gpt-5-mini'
    for alias in sorted(pricing.get("aliases", {}), key=len, reverse=True):
        if alias in model:
            return pricing["aliases"][alias]
    return None


def cost_usd(model: str, tokens: dict, pricing: dict | None = None) -> float:
    """tokens = {input, output, cache_read, cache_write}."""
    pricing = pricing or load_pricing()
    tier = tier_for_model(model, pricing)
    rates = pricing.get("tiers", {}).get(tier or "", {})
    if not rates:
        return 0.0
    disc = pricing.get("gateway_discount", 1.0)
    total = 0.0
    for key in ("input", "output", "cache_read", "cache_write"):
        total += (tokens.get(key, 0) or 0) / 1e6 * rates.get(key, 0.0)
    return total * disc


def coerce_cache_write(val) -> int:
    """cache_creation_input_tokens may be an int or a dict of ephemeral sub-tiers."""
    if isinstance(val, dict):
        return sum(int(v) for v in val.values() if isinstance(v, (int, float)))
    return int(val or 0)
