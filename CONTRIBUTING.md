# Contributing

Thanks for your interest! This is a local-first tool with a small, dependency-light
codebase (SQLite + Flask + vanilla JS + numpy).

## Setup

```bash
git clone https://github.com/mpankaj151/session-browser.git
cd session-browser
./install.sh --no-hook --no-launchd     # dev install; skip the system hooks
```

## Before you open a PR

```bash
python tests/test_smoke.py       # must pass (isolated temp DB; no real data touched)
python -m compileall -q sources scripts session-ui mcp enrichment *.py
for f in install.sh uninstall.sh bin/*.sh; do bash -n "$f"; done
```

CI runs these on Ubuntu + macOS across Python 3.11 and 3.12.

## Guidelines

- **Add a test** for any parsing, cost, redaction, or DB-behavior change — see
  `tests/test_smoke.py` for the temp-DB pattern.
- **Never leak secrets.** Anything that leaves the tool must pass through
  `redact.py`. If you add a new egress, redact it and add a test.
- **Keep the UI offline.** No CDN scripts, remote fonts, or external requests.
  Regenerate `static/tailwind.css` if you add classes (command in
  `tailwind.config.js`).
- **Adapters:** adding a CLI should stay a one-file change. See
  [docs/ADDING-A-CLI.md](docs/ADDING-A-CLI.md).
- Match the surrounding style; keep comments about *why*, not *what*.

## Good first issues

- New source adapters (Gemini CLI, OpenCode, Aider, Ollama).
- More redaction patterns (open a test with a synthetic example — never a real key).
- Pricing updates in `pricing.json` as model prices change.

## Reporting bugs

Include your OS, Python version, which CLIs you use, and `sb doctor` output
(it never prints secrets or session content).
