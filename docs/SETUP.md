# Setup guide

Step-by-step install for a fresh machine, plus a troubleshooting matrix.

## 1. Prerequisites

| Need | macOS | Linux |
|------|-------|-------|
| Python 3.11+ | `brew install python@3.12` | `sudo apt install python3 python3-venv` |
| git | preinstalled / `brew install git` | `sudo apt install git` |
| A supported CLI | [Claude Code](https://claude.com/claude-code) and/or [Copilot CLI](https://github.com/github/copilot-cli) / [Codex](https://github.com/openai/codex) / [OpenCode](https://opencode.ai) | same |

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
5. Registers the Claude Stop + SessionEnd hooks in `~/.claude/settings.json`
   (backup kept) — Stop indexes instantly; SessionEnd journals the ended
   session — and links the shipped skills (work-journal, snapshot, checkpoint)
   into `~/.claude/skills`.
6. Installs the background jobs: launchd agents on macOS, systemd --user units on
   Linux (live watcher + nightly 01:00 refresh). Without either, it prints the
   two commands to schedule yourself.

Install flags:

| Flag | Effect |
|---|---|
| `--lite` | Skip the ~2 GB semantic-search ML stack; search falls back to keyword + full-text |
| `--enrich` | Also LLM-journal your **existing** history during install (spends your plan's quota; the hooks + nightly job cover *new* sessions regardless). Later: `sb refresh --enrich`. Enrichment runs on `claude-sonnet-5` by default — change or clear it via `[enrichment.claude_headless] model` in `config.toml`, or switch the backend to OpenCode with `[enrichment] provider = "opencode-headless"` (model `anthropic/claude-sonnet-5`; needs `opencode auth login`) |
| `--no-hook` | Don't register the Claude hooks or link the skills |
| `--opencode-plugin` | Install the OpenCode plugin (`~/.config/opencode/plugins/session-browser.js`): indexes a session the moment a turn settles, re-syncs on deletion. Optional — the watcher already picks OpenCode changes up within seconds |
| `--no-scheduler` (alias `--no-launchd`) | Don't install the background jobs (launchd on macOS, systemd --user on Linux) |
| `--no-backfill` | Don't index existing sessions now |

## 4. Post-install checklist

```bash
sb doctor
```

Everything should be green (or amber for optional pieces). Then:

```bash
sb ui        # http://127.0.0.1:7655
sb stats     # terminal usage report
```

## 5. Background jobs on Linux (systemd --user)

`install.sh` installs the two background jobs itself: launchd agents on macOS,
and on Linux — when a `systemctl --user` session is reachable — these units in
`~/.config/systemd/user/` (rendered from `systemd/*.template` with your repo,
venv and log paths baked in):

| unit | what |
|---|---|
| `session-browser-watcher.service` | live indexing (`watcher.py`), restarts on failure |
| `session-browser-refresh.timer` → `.service` | nightly 01:00 `refresh-all.py --enrich`, catches up after sleep (`Persistent=true`) |

Check them with `sb doctor` or `systemctl --user status session-browser-watcher`.
User units run while you are logged in; to keep the watcher alive after logout:
`loginctl enable-linger $USER`. `./uninstall.sh` disables and removes them.

Without systemd (or over SSH with no user session bus) the installer prints the
two commands to schedule yourself — a `cron` line for the refresh plus
`nohup .venv/bin/python watcher.py &` in your shell rc is enough.

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
| `cr` says `<cli> is not on PATH` | The session exists but that CLI is not installed in this shell (or a daemon PATH). Indexing, search and the Archived/Restore flow never need the binary; only resume and bridge do. |
| Sessions of a relocated CLI are missing | `CLAUDE_CONFIG_DIR`, `CODEX_HOME` and `XDG_DATA_HOME` are followed when the source's path in `config.toml` is left at the default — export them in the shell that runs `sb`/the watcher, or set the path explicitly. |
| `venv` creation fails | Missing `python3-venv` (Linux): `sudo apt install python3-venv`. |
| Install aborts on `settings.json` | Your `~/.claude/settings.json` is malformed. The installer now warns and skips the hook — fix the JSON and re-run with `--no-backfill --no-launchd`. |
| No sessions shown | No history for enabled sources yet, or backfill was skipped. Run `sb refresh`, or `sb demo` to preview with synthetic data. |
| Port 7655 busy | `sb stop`, or change `[ui].port` in `config.toml`. |
| Stop hook not firing | Open Claude Code's `/hooks` once to reload settings, or restart it. `sb doctor` shows whether it's registered. |
| Semantic search empty / errors | `--lite` install (no model) → it falls back to keyword/full-text automatically. To enable: `pip install sentence-transformers` then `sb refresh`. |
| Enrichment summaries never appear | `[enrichment].provider` defaults to `auto` (first of `claude`/`opencode`/`copilot` on PATH; `sb doctor` → `[enrichment]` shows which one it picked, or that none was found). The nightly job needs that binary on PATH. On Intel Macs check `sb doctor` → sources; re-run `./install.sh` so the launchd PATH picks up your binary. `sb doctor` → `[enrichment]` shows the configured provider + model; a typo'd provider name is reported in `refresh.err.log`. |
| OpenCode enrichment fails with a provider/auth error | `opencode-headless` needs a credential for the configured provider: `opencode auth login` (or the provider's API-key env var), then confirm with `opencode models anthropic \| grep claude-sonnet-5`. A machine without one should stay on `claude-headless`. |
| Semantic search crashes after changing `[embeddings].model` | Run `sb refresh` (or `scripts/embed-sessions.py --force`) to re-embed at the new dimension. |
| Something else — where are the logs? | `~/.session-browser/logs/`: `watcher.log` (live indexing), `refresh.log` + `refresh.err` (nightly pipeline), `ui.log` (`sb ui`). |
| Everything broke after moving the repo | The launchd jobs, Stop hook, and `cr`/`sb` functions bake in absolute paths. Re-run `./install.sh && ./bin/install-cr.sh` from the new location — both repoint stale entries automatically — then restart the UI (`sb stop; sb ui`) and confirm with `sb doctor`. |
