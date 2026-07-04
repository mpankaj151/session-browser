#!/usr/bin/env python3
"""Compute per-session token usage and USD cost — for every source.

Claude: sums each assistant record's TOP-LEVEL usage (never iterations[], which
would double-count). Copilot: reads the per-model `modelMetrics` totals from the
session.shutdown event. Both accumulate tokens per model, pick the dominant model,
and write token columns + cost_usd to registry.db.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import costs  # noqa: E402
import indexer  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402


def _usage_claude(path: Path) -> tuple[dict, dict]:
    """Return (totals, per_model). totals has input/output/cache_read/cache_write."""
    totals = defaultdict(int)
    per_model = defaultdict(lambda: defaultdict(int))
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return totals, per_model
    with fh:
        for line in fh:
            if '"usage"' not in line or '"assistant"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            msg = rec.get("message", {})
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            model = msg.get("model", "") or ""
            mapped = {
                "input": int(usage.get("input_tokens", 0) or 0),
                "output": int(usage.get("output_tokens", 0) or 0),
                "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
                "cache_write": costs.coerce_cache_write(usage.get("cache_creation_input_tokens")),
            }
            for k, v in mapped.items():
                totals[k] += v
                per_model[model][k] += v
    return totals, per_model


def _usage_copilot(path: Path) -> tuple[dict, dict]:
    """Copilot persists complete per-model usage in the session.shutdown event's
    data.modelMetrics.<model>.usage. reasoningTokens are billed as output."""
    totals = defaultdict(int)
    per_model = defaultdict(lambda: defaultdict(int))
    mm = None
    try:
        for line in open(path, "r", encoding="utf-8", errors="replace"):
            if "modelMetrics" not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data", {})
            if isinstance(data, dict) and isinstance(data.get("modelMetrics"), dict):
                mm = data["modelMetrics"]  # keep the last one seen
    except OSError:
        return totals, per_model
    if not mm:
        return totals, per_model
    for model, info in mm.items():
        u = info.get("usage", {}) if isinstance(info, dict) else {}
        mapped = {
            "input": int(u.get("inputTokens", 0) or 0),
            "output": int(u.get("outputTokens", 0) or 0) + int(u.get("reasoningTokens", 0) or 0),
            "cache_read": int(u.get("cacheReadTokens", 0) or 0),
            "cache_write": int(u.get("cacheWriteTokens", 0) or 0),
        }
        for k, v in mapped.items():
            totals[k] += v
            per_model[model][k] += v
    return totals, per_model


_EXTRACTORS = {"claude": _usage_claude, "copilot": _usage_copilot}


def process(path: Path, adapter, conn) -> dict | None:
    header = adapter.parse_header(path)
    if header is None:
        return None
    extractor = _EXTRACTORS.get(adapter.name)
    if extractor is None:
        return None
    totals, per_model = extractor(path)
    if not per_model:
        return None
    pricing = costs.load_pricing()
    total_cost = 0.0
    for model, toks in per_model.items():
        total_cost += costs.cost_usd(model, toks, pricing)
    # dominant model = most output tokens
    dominant = max(per_model, key=lambda m: per_model[m]["output"], default=header.model_used)
    models_used = json.dumps(sorted(per_model.keys()))
    conn.execute(
        "UPDATE sessions SET input_tokens=?, output_tokens=?, cache_read_tokens=?, "
        "cache_write_tokens=?, model_used=COALESCE(model_used, ?), models_used=?, cost_usd=? "
        "WHERE session_id=?",
        (totals["input"], totals["output"], totals["cache_read"], totals["cache_write"],
         dominant, models_used, round(total_cost, 6), header.session_id),
    )
    return {"session": header.session_id, "cost": round(total_cost, 4), **totals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="limit to one source (claude|copilot)")
    args = ap.parse_args()
    registry = build_source_registry(only_available=True)
    if args.source:
        registry = {k: v for k, v in registry.items() if k == args.source}
    conn = indexer.connect()
    n = 0
    for name, adapter in registry.items():
        if name not in _EXTRACTORS:
            continue
        files = list(adapter.discover())
        print(f"[{name}] {len(files)} files")
        for i, path in enumerate(files, 1):
            try:
                r = process(path, adapter, conn)
                if r:
                    n += 1
                    print(f"  ${r['cost']:.4f}  in={r['input']} out={r['output']} "
                          f"cr={r['cache_read']} cw={r['cache_write']}  {r['session'][:8]}")
            except Exception as e:  # noqa: BLE001
                print(f"  ! {path.name}: {e}", file=sys.stderr)
            if i % 20 == 0:
                conn.commit()
        conn.commit()
    conn.close()
    print(f"Cost computed for {n} sessions.")


if __name__ == "__main__":
    main()
