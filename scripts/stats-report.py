#!/usr/bin/env python3
"""Terminal usage report — a ccusage-style breakdown, no UI needed.

    sb stats            # today / 7d / 30d / all + per-model + per-project
    sb stats --days 90

Reads the same registry columns the dashboard uses. Costs are public-API
list-price equivalents; under a flat subscription they are notional (≈).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402

_TOK = "input_tokens+output_tokens+cache_read_tokens+cache_write_tokens"
PFX = "≈" if sbconfig.COST_IS_NOTIONAL else ""


def _h(n: int) -> str:
    n = n or 0
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.1f}{unit}"
    return str(n)


def _bold(s):  # noqa: ANN001
    return f"\033[1m{s}\033[0m"


def _window(conn, label: str, sql_filter: str):
    row = conn.execute(
        f"SELECT COUNT(*) c, COALESCE(SUM({_TOK}),0) t, COALESCE(SUM(cost_usd),0) cost "
        f"FROM sessions WHERE archived=0 {sql_filter}"
    ).fetchone()
    return f"  {label:<8} {row['c']:>4} sessions   {_h(row['t']):>7} tok   {PFX}${row['cost']:>10,.2f}"


def _table(conn, title: str, col: str, expr: str, limit: int):
    rows = conn.execute(
        f"SELECT {expr} AS k, COUNT(*) c, COALESCE(SUM({_TOK}),0) t, COALESCE(SUM(cost_usd),0) cost "
        f"FROM sessions WHERE archived=0 GROUP BY k ORDER BY cost DESC LIMIT {limit}"
    ).fetchall()
    print(f"\n{_bold(title)}")
    for r in rows:
        print(f"  {str(r['k'])[:28]:<28} {r['c']:>4}   {_h(r['t']):>7} tok   {PFX}${r['cost']:>10,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="also show a custom window")
    args = ap.parse_args()

    conn = indexer.connect()
    plan = sbconfig.BILLING.get("plan", "")
    print(_bold("Session Browser — usage report"))
    if sbconfig.COST_IS_NOTIONAL:
        print(f"  plan: {plan} (flat-rate) · $ = public API list-price equivalent, not billed")

    print(f"\n{_bold('By window')}")
    print(_window(conn, "today", "AND last_activity >= date('now')"))
    print(_window(conn, "7 days", "AND last_activity >= datetime('now','-7 days')"))
    print(_window(conn, "30 days", "AND last_activity >= datetime('now','-30 days')"))
    if args.days:
        print(_window(conn, f"{args.days}d", f"AND last_activity >= datetime('now','-{args.days} days')"))
    print(_window(conn, "all", ""))

    _table(conn, "By model", "model", "COALESCE(model_used,'unknown')", 10)
    _table(conn, "By CLI", "source", "cli_source", 5)
    _table(conn, "By project", "project", "COALESCE(NULLIF(folder_name,''),'—')", 10)
    conn.close()


if __name__ == "__main__":
    main()
