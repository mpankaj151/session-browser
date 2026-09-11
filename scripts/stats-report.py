#!/usr/bin/env python3
"""Terminal usage report — a ccusage-style breakdown, no UI needed.

    sb stats            # today / 7d / 30d / all + per-model + per-project
    sb stats --days 90

Reads the same registry columns the dashboard uses. Costs are public-API
list-price equivalents; under a flat subscription they are notional (≈).

What is being counted, for a reader new to AI tooling (docs/GLOSSARY.md has the
rest): **tokens** are the word-pieces a model reads and writes — roughly four
characters of English each — and every vendor bills per token, so token totals are
the raw measure of how much a session used. They come in four flavours, stored in
four columns and summed by _TOK below: input (what was sent), output (what the
model wrote, several times more expensive), and cache write / cache read (the
unchanged start of a long conversation, kept on the vendor's side; writing costs a
little more than a normal input token, reading about a tenth). Long sessions are
mostly cache reads, which is why they cost far less than their size suggests.

No model is called here and nothing is spent — this is a read-only report over
columns scripts/compute-costs.py already filled in. Output is a plain terminal
table; `sb stats` is the shell wrapper.

The dollar figures are the public list price for those tokens. Under a flat
subscription nothing is billed per session, so they are printed with a "≈" and
mean "how heavy was this", not "what did this cost me".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import sbconfig  # noqa: E402

# SQL expression for "all tokens this session used", spliced into the queries
# below. The COALESCE is PER COLUMN, and that is the whole point: in SQL, NULL
# poisons arithmetic, so `a + b` is NULL if either side is NULL. Without a COALESCE
# on each of the four columns, one missing value (a CLI that reports no cache
# figures, a row indexed before a column existed) zeroed the row's ENTIRE token
# count. Regression-tested in tests/test_smoke.py.
_TOK = ("COALESCE(input_tokens,0)+COALESCE(output_tokens,0)"
        "+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)")
# Printed before every dollar amount. "≈" under a flat subscription, where the
# figure is a list-price equivalent rather than money billed; empty when
# [billing].mode = "api" and the numbers are real.
PFX = "≈" if sbconfig.COST_IS_NOTIONAL else ""


def _h(n: int) -> str:
    """Human-readable token count: 1234567 -> '1.2M'. Largest unit that fits wins."""
    n = n or 0
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.1f}{unit}"
    return str(n)


def _bold(s):  # noqa: ANN001
    """Wrap text in the ANSI bold escape sequence for terminal output.

    \\033[1m turns bold on, \\033[0m resets everything. Harmless in a pipe or a
    file — the sequences simply show as text — so there is no isatty() check."""
    return f"\033[1m{s}\033[0m"


def since_sql(days: int) -> str:
    """`last_activity` cutoff for a rolling window, spelled like the column
    ('YYYY-MM-DDTHH:MM:SS.mmmZ'). datetime('now', '-N days') would yield
    'YYYY-MM-DD HH:MM:SS' — and 'T' sorts above ' ', so every row of the cutoff
    day compared >= the cutoff and the window leaked up to 24 h.

    Longer version, because this is a subtle class of bug. Timestamps are stored as
    TEXT, so `>=` is a character-by-character string comparison; it gives the right
    answer only when both sides are spelled identically. The stored spelling puts a
    literal 'T' between the date and the time. A space (chr 32) sorts BELOW 'T'
    (chr 84), so a cutoff spelled '2026-06-01 14:00:00' compares less than
    '2026-06-01T00:00:00.000Z' — and every session from earlier that day was pulled
    into a window that should have started at 2pm.

    strftime with the exact format string produces the matching spelling, and
    'now' with '-N days' does the arithmetic inside SQLite. int(days) is the guard
    against injection, since this value is interpolated rather than bound.

    Returns a SQL fragment, e.g. `last_activity >= strftime(...)`, to be ANDed into
    a WHERE clause. Regression-tested in tests/test_smoke.py.
    """
    return (f"last_activity >= strftime('%Y-%m-%dT%H:%M:%S.000Z', 'now', "
            f"'-{int(days)} days')")


def _window(conn, label: str, sql_filter: str):
    """One summary line for a time window: session count, total tokens, total cost.

    The query counts rows and sums tokens and dollars over indexer.VISIBLE — the
    sessions the user should see and that usage stats should count: those whose
    transcript is still on disk, plus real sessions whose transcript has since aged
    out. Their token totals are real history and must keep counting, so the window
    must not be narrowed to rows whose archived flag is clear. Subagent sidechain
    noise is excluded.

    `sql_filter` is an already-built fragment ('' for all time, else
    'AND <predicate>') appended to the WHERE clause. COALESCE(SUM(...),0) turns the
    NULL that SUM returns over zero rows into 0, so an empty window prints "0"
    rather than "None".

    Returns the formatted line; the caller prints it. Read-only.
    """
    row = conn.execute(
        f"SELECT COUNT(*) c, COALESCE(SUM({_TOK}),0) t, COALESCE(SUM(cost_usd),0) cost "
        f"FROM sessions WHERE {indexer.VISIBLE} {sql_filter}"
    ).fetchone()
    return f"  {label:<8} {row['c']:>4} sessions   {_h(row['t']):>7} tok   {PFX}${row['cost']:>10,.2f}"


def _table(conn, title: str, col: str, expr: str, limit: int):
    """Print an all-time breakdown grouped by one expression (model, CLI, project).

    `expr` is the SQL that produces the grouping key — e.g.
    COALESCE(model_used,'unknown'), which folds rows with no recorded model into a
    single visible bucket instead of a row labelled None. `limit` caps the table;
    ORDER BY cost DESC puts the heaviest first, so a `LIMIT 10` keeps what matters.
    Same VISIBLE scope and same COALESCEd sums as _window.

    `col` is the column heading and is currently unused in the output — the title
    above the table already names the grouping.

    `limit` is interpolated, not bound, so it must stay an int from this module's
    own call sites; never pass user input. Prints directly; returns None.
    """
    rows = conn.execute(
        f"SELECT {expr} AS k, COUNT(*) c, COALESCE(SUM({_TOK}),0) t, COALESCE(SUM(cost_usd),0) cost "
        f"FROM sessions WHERE {indexer.VISIBLE} GROUP BY k ORDER BY cost DESC LIMIT {limit}"
    ).fetchall()
    print(f"\n{_bold(title)}")
    for r in rows:
        # Keys are truncated to 28 characters so a long project path cannot break
        # the column alignment.
        print(f"  {str(r['k'])[:28]:<28} {r['c']:>4}   {_h(r['t']):>7} tok   {PFX}${r['cost']:>10,.2f}")


def main() -> None:
    """Print the whole report: four windows, then three breakdowns. Read-only.

    Always exits 0 — there is no failure mode beyond an unreadable database, which
    raises. No model is called and nothing is spent.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="also show a custom window")
    args = ap.parse_args()

    conn = indexer.connect()
    plan = sbconfig.BILLING.get("plan", "")
    print(_bold("Session Browser — usage report"))
    if sbconfig.COST_IS_NOTIONAL:
        print(f"  plan: {plan} (flat-rate) · $ = public API list-price equivalent, not billed")

    print(f"\n{_bold('By window')}")
    # "today" can use date('now') directly: it yields a bare 'YYYY-MM-DD', which is
    # a prefix of the stored spelling, so the string comparison lands exactly on
    # midnight. The since_sql() spelling problem only arises once a TIME is
    # involved. Note both are UTC days, unlike the local-date grouping the reports
    # and the daily digest use.
    print(_window(conn, "today", "AND last_activity >= date('now')"))
    print(_window(conn, "7 days", "AND " + since_sql(7)))
    print(_window(conn, "30 days", "AND " + since_sql(30)))
    if args.days:
        print(_window(conn, f"{args.days}d", "AND " + since_sql(args.days)))
    # Empty filter = no time bound at all.
    print(_window(conn, "all", ""))

    # Three breakdowns. COALESCE(model_used,'unknown') buckets rows with no
    # recorded model; NULLIF(folder_name,'') turns an EMPTY project name into NULL
    # first so that COALESCE catches it too, and both the missing and the blank
    # case land in one '—' row rather than two confusing ones. Limits differ
    # because there are only a handful of CLIs but many models and projects.
    _table(conn, "By model", "model", "COALESCE(model_used,'unknown')", 10)
    _table(conn, "By CLI", "source", "cli_source", 5)
    _table(conn, "By project", "project", "COALESCE(NULLIF(folder_name,''),'—')", 10)
    conn.close()


if __name__ == "__main__":
    main()
