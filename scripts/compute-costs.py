#!/usr/bin/env python3
"""Compute per-session token usage and USD cost — for every source.

A **token** is the unit a model reads and writes text in — about four characters of
English — and every request a coding CLI makes is billed per token. Transcripts record
those counts; this script totals them per session and turns them into a dollar figure.
There are four kinds and they cost very differently: input (what was sent), output (what
the model wrote, several times dearer), cache_write (storing a long unchanging prefix on
the vendor's side) and cache_read (re-using it, roughly a tenth of an input token). See
costs.py and docs/GLOSSARY.md.

Under a flat monthly subscription no money is billed per session at all, so the figure is
the public list-price equivalent and the UI shows it with `≈` — an intensity signal, not
an invoice (docs/ARCHITECTURE.md, "Cost is notional under a subscription").

Claude: sums each assistant record's TOP-LEVEL usage (never iterations[], which
would double-count). Copilot: reads the per-model `modelMetrics` totals from the
session.shutdown event. Both accumulate tokens per model, pick the dominant model,
and write token columns + cost_usd to registry.db.

One extractor per CLI, and they come in two flavours — see the _EXTRACTORS comment:
a 2-tuple `(totals, per_model)` means "here are the tokens, you price them", while
OpenCode returns a 3-tuple whose third element is the real spend it recorded itself.

Two rules that keep the numbers honest:
  * A model with no pricing alias is LOUD (a stderr line naming it) and priced at $0 —
    never quietly mapped to a similar-looking tier.
  * A session that could not be priced at all never overwrites a cost already in the
    database with 0. The UPDATE uses COALESCE(?, cost_usd) with a NULL parameter, which
    is SQL for "keep whatever is already there". Otherwise one pricing.json typo, or one
    run on a machine that cannot read it, would silently zero every historical figure.

Pipeline position: run by the nightly `refresh-all`, after indexing. Reads every
discovered transcript through its adapter; writes only the token columns, model_used,
models_used and cost_usd on existing registry rows. Never creates rows, never deletes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Scripts run directly, not as a package: put the repo root on the import path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import costs  # noqa: E402
import indexer  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402


def _usage_claude(path: Path) -> tuple[dict, dict]:
    """Return (totals, per_model). totals has input/output/cache_read/cache_write.

    Claude Code reports usage per assistant message, so this sums across the file. One
    such line looks like:

      {"type":"assistant","message":{"model":"claude-opus-4-5-20260514","usage":{
         "input_tokens":12,"output_tokens":340,"cache_read_input_tokens":18400,
         "cache_creation_input_tokens":{"ephemeral_5m_input_tokens":210}}}}

    Only the TOP-LEVEL `message.usage` is counted. Some records also carry an
    `iterations[]` array whose entries repeat the same usage; adding those would
    double-count a long turn.

    `per_model` matters because one session can span several models (a `/model` switch
    mid-session, a fast model for small steps). Pricing differs per model, so tokens are
    attributed as they are earned rather than lumped under whichever model was last seen.
    An unreadable file returns empty tallies, which process() turns into "skip this one".
    """
    totals = defaultdict(int)
    per_model = defaultdict(lambda: defaultdict(int))
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return totals, per_model
    with fh:
        for line in fh:
            # Substring prefilter before the JSON parse: transcripts run to megabytes and
            # only a minority of lines are assistant records carrying usage.
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
            # Vendor key names -> this tool's four canonical names. cache_creation may be
            # an int or a dict of per-lifetime sub-counts, hence the helper.
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
    data.modelMetrics.<model>.usage. reasoningTokens are billed as output.

    So unlike Claude there is nothing to accumulate: one event holds the session's final
    totals per model. Shape:

      {"type":"session.shutdown","data":{"modelMetrics":{
         "gpt-5.4":{"usage":{"inputTokens":1000,"outputTokens":100,
                             "cacheReadTokens":500,"cacheWriteTokens":0,
                             "reasoningTokens":40}}}}}

    "Reasoning tokens" are the ones the model spent thinking before replying. They are
    not shown to the user but they are generated, and vendors bill them at the output
    rate — so they are folded into `output` rather than tracked separately.

    A session that never shut down cleanly has no such event; that yields empty tallies
    and the session is skipped rather than reported as free.
    """
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
                # Keep the LAST one: a session that was resumed writes more than one
                # shutdown event, and the final one carries the complete totals.
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


def _usage_codex(path: Path) -> tuple[dict, dict]:
    """Codex logs cumulative usage in token_count.info.total_token_usage. input_tokens
    INCLUDES cached; split it so cache_read isn't double-counted. reasoning billed as
    output. Keep the LAST token_count seen (it's the running total). No per-model
    breakdown in the event, so attribute to the session's model.

    "Cumulative" is the trap here: each token_count event restates the session total, so
    summing them would multiply the real usage. Only the last one is kept.

    "input_tokens INCLUDES cached" is the second trap. Codex's `input_tokens` already
    contains the `cached_input_tokens`, so adding both would count the cached portion
    twice AND at the wrong (10x) rate. Subtracting gives the genuinely-fresh input.
    max(0, …) guards against an event where the cached figure exceeds the total.

    Codex bills no cache writes, hence the constant 0 for cache_write.
    """
    from sources.codex import ROLLOUT_ERRORS, open_rollout   # plain or zstd — never a bare open()
    totals = defaultdict(int)
    per_model = defaultdict(lambda: defaultdict(int))
    last = None
    model = ""
    try:
        with open_rollout(path) as fh:
            for line in fh:
                if '"model"' in line and not model:
                    try:
                        rec = json.loads(line)
                        p = rec.get("payload", {}) if isinstance(rec, dict) else {}
                        # Current rollouts write turn_context as the RECORD type
                        # (payload = {model, cwd, ...}); older ones nest it as
                        # payload.type. The adapter accepts both — so must this.
                        if isinstance(p, dict) and "turn_context" in (rec.get("type"), p.get("type")):
                            model = p.get("model", "") or ""
                    except (json.JSONDecodeError, AttributeError):
                        pass
                if "token_count" not in line:
                    continue
                try:
                    p = json.loads(line).get("payload", {})
                except json.JSONDecodeError:
                    continue
                info = p.get("info") if isinstance(p, dict) else None
                if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                    last = info["total_token_usage"]
    except ROLLOUT_ERRORS:            # I/O, or a corrupt/truncated zstd frame
        return totals, per_model
    if not last:
        return totals, per_model
    cached = int(last.get("cached_input_tokens", 0) or 0)
    # See the docstring: input_tokens is inclusive of `cached`, so split them apart.
    mapped = {
        "input": max(0, int(last.get("input_tokens", 0) or 0) - cached),
        "output": int(last.get("output_tokens", 0) or 0) + int(last.get("reasoning_output_tokens", 0) or 0),
        "cache_read": cached,
        "cache_write": 0,
    }
    for k, v in mapped.items():
        totals[k] += v
        # No turn_context at all: leave the key empty so process() prices it via
        # the adapter's model_used or reports it unknown — never a silent guess.
        per_model[model][k] += v
    return totals, per_model


def _usage_opencode(path: Path) -> tuple[dict, dict, float]:
    """OpenCode stores per-message USD and tokens for EVERY provider (priced from
    its models.dev catalogue), and the adapter rolls them up per provider/model
    on line 1 of the mirror file. That cost is authoritative — GLM, Qwen, Kimi,
    MiniMax are not in pricing.json and never will be — so this returns a
    3-tuple; process() skips its own pricing when the third element is present.
    Reasoning tokens fold into output (the copilot precedent).

    Everything is read from line 1 of the mirror file — the header the adapter writes,
    whose `stats.models` maps each model to its rolled-up tokens and cost:

      {"type":"session","info":{…},"stats":{"models":{
         "anthropic/claude-sonnet-5":{"input":120,"output":900,"reasoning":40,
                                      "cache_read":5000,"cache_write":300,"cost":0.0185}}}}

    Costs are summed across models; the tokens are still tallied so the usage columns and
    the UI's token charts work the same as for every other source. An unreadable or
    header-less mirror returns empty tallies and a 0.0 cost, and process() then skips.
    """
    totals = defaultdict(int)
    per_model = defaultdict(lambda: defaultdict(int))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = json.loads(fh.readline())
    except (OSError, json.JSONDecodeError):
        return totals, per_model, 0.0
    models = (head.get("stats") or {}).get("models") if isinstance(head, dict) else None
    cost = 0.0
    for model, tk in (models or {}).items():
        mapped = {
            "input": int(tk.get("input") or 0),
            "output": int(tk.get("output") or 0) + int(tk.get("reasoning") or 0),
            "cache_read": int(tk.get("cache_read") or 0),
            "cache_write": int(tk.get("cache_write") or 0),
        }
        for k, v in mapped.items():
            totals[k] += v
            per_model[model][k] += v
        cost += float(tk.get("cost") or 0)
    return totals, per_model, cost


# An extractor returns (totals, per_model) — priced here via pricing.json — or
# (totals, per_model, cost_usd) when the source already knows the true spend.
# The 3-tuple exists for OpenCode specifically: it can be pointed at any provider, and
# models like GLM, Qwen, Kimi or MiniMax will never appear in pricing.json — but OpenCode
# already priced every message from its own catalogue, so its number is better than any
# we could compute. process() reads len(result) to tell the two contracts apart, which
# keeps every other extractor unchanged.
_EXTRACTORS = {"claude": _usage_claude, "copilot": _usage_copilot, "codex": _usage_codex,
               "opencode": _usage_opencode}


def process(path: Path, adapter, conn) -> dict | None:
    """Price one transcript and write its usage columns. Returns a summary dict or None.

    None means "nothing to record": the header would not parse, this source has no
    extractor, or the extractor found no usage at all.

    The write updates token counts unconditionally (they are facts read from the file)
    but treats cost carefully — see the module docstring on COALESCE. model_used is also
    COALESCEd so an already-known model is never replaced, while models_used is rewritten
    as a JSON array of every model the session touched.

    Does not commit; main() batches commits. Tests call it with a temp connection.
    """
    header = adapter.parse_header(path)
    if header is None:
        return None
    extractor = _EXTRACTORS.get(adapter.name)
    if extractor is None:
        return None
    res = extractor(path)
    # The 2-tuple / 3-tuple contract: a third element is the source's own authoritative
    # cost, and its presence (not its value — 0.0 is a legitimate free session) is what
    # tells us to skip local pricing entirely.
    totals, per_model = res[0], res[1]
    authoritative = res[2] if len(res) > 2 else None
    # An extractor that found tokens but no model name leaves the key empty;
    # the adapter's model_used is the authority (never a silent tier guess).
    per_model = {(m or header.model_used or ""): t for m, t in per_model.items()}
    if not per_model:
        return None
    if authoritative is not None:
        total_cost = float(authoritative)
    else:
        pricing = costs.load_pricing()
        total_cost = 0.0
        priced_any = False
        for model, toks in per_model.items():
            if costs.tier_for_model(model, pricing) is None:
                if any(toks.values()):
                    print(f"  ? unknown model '{model}' ({header.session_id[:8]}) — cost counted as $0; "
                          f"add an alias for it in pricing.json", file=sys.stderr)
                continue
            priced_any = True
            total_cost += costs.cost_usd(model, toks, pricing)
        if not priced_any:
            # Nothing in this session is priced (unknown model, or an unreadable
            # pricing.json): leave the stored cost alone rather than zero it.
            total_cost = None
    # dominant model = most output tokens
    # A session can touch several models; the list shows all of them, but the row needs
    # ONE model to display and group by. Output tokens are the right yardstick — they are
    # the expensive half and the best proxy for "which model actually did the work". A
    # cheap model used for a hundred one-line steps should not outrank the model that
    # wrote the code just because it appeared more often.
    dominant = max(per_model, key=lambda m: per_model[m]["output"], default=header.model_used)
    # Sorted so the stored JSON is stable run to run — an unsorted set would make the
    # column churn and every diff of the database look like a change.
    models_used = json.dumps(sorted(per_model.keys()))
    # cost_usd=COALESCE(?, cost_usd): with a NULL parameter (an unpriced session) the
    # existing value is kept. Never overwrite a real figure with 0 — see the module
    # docstring. model_used=COALESCE(model_used, ?) fills the column only if it is empty.
    conn.execute(
        "UPDATE sessions SET input_tokens=?, output_tokens=?, cache_read_tokens=?, "
        "cache_write_tokens=?, model_used=COALESCE(model_used, ?), models_used=?, "
        "cost_usd=COALESCE(?, cost_usd) WHERE session_id=?",
        (totals["input"], totals["output"], totals["cache_read"], totals["cache_write"],
         dominant, models_used, None if total_cost is None else round(total_cost, 6),
         header.session_id),
    )
    return {"session": header.session_id, "cost": None if total_cost is None else round(total_cost, 4), **totals}


def main() -> None:
    """Walk every available source's transcripts, price each one, print a line per hit.

    Always exits 0: an individual transcript that fails is reported on stderr and the
    sweep continues, because a nightly run must not be marked failed by one bad file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="limit to one source (claude|copilot)")
    args = ap.parse_args()
    # only_available=True skips CLIs with no transcripts on this machine.
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
                    shown = "unpriced" if r["cost"] is None else f"${r['cost']:.4f}"
                    print(f"  {shown}  in={r['input']} out={r['output']} "
                          f"cr={r['cache_read']} cw={r['cache_write']}  {r['session'][:8]}")
            except Exception as e:  # noqa: BLE001
                # Report and carry on: one malformed transcript must not abort a sweep
                # over thousands.
                print(f"  ! {path.name}: {e}", file=sys.stderr)
            # Commit every 20 files: often enough that a kill loses almost nothing,
            # rarely enough that the write lock is not held constantly against the hook.
            if i % 20 == 0:
                conn.commit()
        conn.commit()
    conn.close()
    print(f"Cost computed for {n} sessions.")


if __name__ == "__main__":
    main()
