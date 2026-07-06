# Setup guide

Step-by-step install for a fresh machine, plus a troubleshooting matrix.

## 1. Prerequisites

| Need | macOS | Linux |
|------|-------|-------|
| Python 3.11+ | `brew install python@3.12` | `sudo apt install python3 python3-venv` |
| git | preinstalled / `brew install git` | `sudo apt install git` |
| A supported CLI | [Claude Code](https://claude.com/claude-code) and/or [Copilot CLI](https://github.com/github/copilot-cli) / [Codex](https://github.com/openai/codex) | same |

You need at least one supported CLI **with existing session history** for the
browser to have anything to show. No history yet? Run `sb demo` after install to
see the UI with synthetic data.

## 2. One-command install

```bash
curl -fsSL https://raw.githubusercontent.com/mpankaj151/session-browser/main/bootstrap.sh | bash
```

This clones to `~/session-browser` (override with `SB_HOME`), runs `install.sh`,
and adds the `cr`/`sb` shell shortcuts. Pass installer flags with
`SB_INSTALL_ARGS`, e.g.:

```bash
SB_INSTALL_ARGS="--lite" curl -fsSL .../bootstrap.sh | bash
```

### Or clone and install manually

```bash
git clone https://github.com/mpankaj151/session-browser.git
cd session-browser
./install.sh                 # add --lite to skip the ~2 GB semantic-search stack
./bin/install-cr.sh          # adds cr / sb to your shell rc
source ~/.zshrc
```

## 3. What install.sh does

1. Creates a venv and installs `requirements.txt` (plus sentence-transformers
   unless `--lite`).
2. Copies `config.toml.example` → `config.toml` (edit anytime).
3. Builds the SQLite schema and backfills every existing session.
4. Computes costs, extracts reasoning trails, builds the full-text and vector
   indexes.
5. Registers the Claude Stop hook in `~/.claude/settings.json` (backup kept).
6. **macOS:** installs launchd jobs (live watcher + nightly 01:00 refresh).
   **Linux:** prints the two commands to schedule yourself.

Install flags: `--lite`, `--enrich`, `--no-hook`, `--no-launchd`, `--no-backfill`.

## 4. Post-install checklist

```bash
sb doctor
```

Everything should be green (or amber for optional pieces). Then:

```bash
sb ui        # http://127.0.0.1:7655
sb stats     # terminal usage report
```

## 5. Linux scheduling (systemd --user)

macOS gets launchd automatically. On Linux, schedule the two background jobs:

`~/.config/systemd/user/session-browser-watcher.service`:

```ini
[Unit]
Description=Session Browser live watcher
[Service]
ExecStart=%h/session-browser/.venv/bin/python %h/session-browser/watcher.py
Restart=on-failure
[Install]
WantedBy=default.target
```

`~/.config/systemd/user/session-browser-refresh.service` + `.timer`:

```ini
# .service
[Service]
Type=oneshot
ExecStart=%h/session-browser/.venv/bin/python %h/session-browser/scripts/refresh-all.py --enrich
# .timer
[Timer]
OnCalendar=*-*-* 01:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
systemctl --user enable --now session-browser-watcher.service
systemctl --user enable --now session-browser-refresh.timer
```

(Or a plain `cron` line for the refresh + `nohup watcher.py &` in your shell rc.)

## 6. Register the MCP server (optional)

Let Claude recall past sessions. Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "session-memory": {
      "command": "/absolute/path/to/session-browser/.venv/bin/python",
      "args": ["/absolute/path/to/session-browser/mcp/session-memory/server.py"]
    }
  }
}
```

## 7. Uninstall

```bash
./uninstall.sh            # removes hook, launchd jobs, cr/sb shell blocks
./uninstall.sh --purge    # also deletes ~/.session-browser (the database)
```

The reasoning archive at `~/claude-reasoning-archive` is always left intact.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `venv` creation fails | Missing `python3-venv` (Linux): `sudo apt install python3-venv`. |
| Install aborts on `settings.json` | Your `~/.claude/settings.json` is malformed. The installer now warns and skips the hook — fix the JSON and re-run with `--no-backfill --no-launchd`. |
| No sessions shown | No history for enabled sources yet, or backfill was skipped. Run `sb refresh`, or `sb demo` to preview with synthetic data. |
| Port 7655 busy | `sb stop`, or change `[ui].port` in `config.toml`. |
| Stop hook not firing | Open Claude Code's `/hooks` once to reload settings, or restart it. `sb doctor` shows whether it's registered. |
| Semantic search empty / errors | `--lite` install (no model) → it falls back to keyword/full-text automatically. To enable: `pip install sentence-transformers` then `sb refresh`. |
| Enrichment summaries never appear | The nightly job needs `claude`/`copilot` on PATH. On Intel Macs check `sb doctor` → sources; re-run `./install.sh` so the launchd PATH picks up your binary. |
| Semantic search crashes after changing `[embeddings].model` | Run `sb refresh` (or `scripts/embed-sessions.py --force`) to re-embed at the new dimension. |
| Something else — where are the logs? | `~/.session-browser/logs/`: `watcher.log` (live indexing), `refresh.log` + `refresh.err` (nightly pipeline), `ui.log` (`sb ui`). |
| Everything broke after moving the repo | The launchd jobs, Stop hook, and `cr`/`sb` functions bake in absolute paths. Re-run `./install.sh && ./bin/install-cr.sh` from the new location — both repoint stale entries automatically — then restart the UI (`sb stop; sb ui`) and confirm with `sb doctor`. |
