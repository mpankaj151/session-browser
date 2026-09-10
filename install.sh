#!/usr/bin/env bash
# Session Browser installer.
#   ./install.sh [--no-hook] [--no-scheduler] [--no-backfill] [--enrich] [--lite] [--opencode-plugin]
# Idempotent. Creates the venv, builds the DB, backfills, optionally registers the
# Claude Stop hook and the background jobs (launchd on macOS, systemd --user on
# Linux). --no-launchd is kept as an alias of --no-scheduler.
#   --lite  skip sentence-transformers/torch (~2 GB download); semantic search
#           falls back to keyword + full-text.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
HOME_DIR="$HOME"
LOG_DIR="$HOME/.session-browser/logs"
# Claude Code relocates its state with CLAUDE_CONFIG_DIR; hooks/skills go there.
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
# Claude-specific steps (hook, skills) only make sense where Claude Code is —
# on an OpenCode/Codex/Copilot-only laptop they would just create ~/.claude.
HAVE_CLAUDE=0
if command -v claude >/dev/null 2>&1 || [ -d "$CLAUDE_DIR" ]; then HAVE_CLAUDE=1; fi

NO_HOOK=0; NO_LAUNCHD=0; NO_BACKFILL=0; NO_ENRICH=1; LITE=0; OC_PLUGIN=0   # enrich off by default (uses LLM quota)
for a in "$@"; do case "$a" in
  --no-hook) NO_HOOK=1;; --no-launchd|--no-scheduler) NO_LAUNCHD=1;;
  --no-backfill) NO_BACKFILL=1;; --enrich) NO_ENRICH=0;; --lite) LITE=1;;
  --opencode-plugin) OC_PLUGIN=1;;
  *) echo "unknown flag: $a"; exit 1;; esac; done

echo "==> Session Browser install ($REPO)"

# Python 3.11+ required (tomllib, sqlite FTS5). Don't hard-require that plain
# `python3` be new: stock macOS resolves it to the Xcode CLT build (often 3.9)
# even when a modern interpreter is installed as python3.12 — hunt for one.
PYBOOT=""
# An existing venv that is new enough needs no bootstrap interpreter at all —
# re-running the installer must not fail just because python3.12 left PATH.
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  PYBOOT="$PY"
fi
for c in python3.13 python3.12 python3.11 python3; do
  [ -n "$PYBOOT" ] && break
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PYBOOT="$(command -v "$c")"; break
  fi
done
if [ -z "$PYBOOT" ]; then
  echo "!! Python 3.11+ is required (found: $(python3 -V 2>/dev/null || echo none))."
  echo "   macOS:  brew install python@3.12"
  echo "   Linux:  sudo apt install python3.12 python3.12-venv  (or your distro's equivalent)"
  echo "   Then re-run this script."
  exit 1
fi

# 1. venv + dependencies (idempotent — pip skips what's already satisfied).
#    --system-site-packages reuses an existing torch/sentence-transformers
#    install when one is present; harmless otherwise.
if [ ! -x "$PY" ]; then
  echo "==> creating venv ($PYBOOT)"
  "$PYBOOT" -m venv --system-site-packages "$REPO/.venv"
  "$PY" -m pip install --quiet --upgrade pip
fi
echo "==> installing dependencies"
"$PY" -m pip install --quiet -r "$REPO/requirements.txt"
if [ "$LITE" -eq 0 ]; then
  "$PY" -m pip install --quiet "sentence-transformers>=2.7" \
    || echo "   ! sentence-transformers install failed — semantic search will fall back to keyword"
  # Pre-download the embedding model NOW (the one moment a big fetch is
  # expected). Runtime queries never go online — semsearch is offline-only
  # unless SB_ALLOW_MODEL_DOWNLOAD=1.
  echo "==> caching the embedding model (one-time download)"
  SB_REPO="$REPO" SB_ALLOW_MODEL_DOWNLOAD=1 "$PY" - <<'PYEOF' \
    || echo "   ! model download failed — the nightly embed job will retry"
import os, sys
sys.path.insert(0, os.environ["SB_REPO"])
import semsearch
semsearch.get_model()
PYEOF
fi

# 2. config — a minimal OVERRIDE file, not a copy of the example. config.toml is
#    layered over config.toml.example at load time, so a full copy would freeze
#    today's defaults and silently miss every source/provider a later git pull
#    adds (this is exactly how a pre-OpenCode config hid OpenCode).
if [ ! -f "$REPO/config.toml" ]; then
  cat > "$REPO/config.toml" <<'EOF'
# Per-machine overrides for Session Browser (this file is git-ignored).
# Layered over config.toml.example: anything NOT set here keeps the example's
# documented default, so sources and providers added by `git pull` just work.
# Only write the keys you want to change, e.g.
#
# [enrichment]
# provider = "opencode-headless"     # summarise through OpenCode on this machine
#
# [sources.codex]
# enabled = false                    # never index Codex here
EOF
  echo "==> wrote config.toml (overrides only; defaults come from config.toml.example)"
fi

# 3. runtime dirs + schema
mkdir -p "$LOG_DIR"
"$PY" "$REPO/scripts/migrate-db.py"

# 4. backfill + full processing pipeline (idempotent)
if [ "$NO_BACKFILL" -eq 0 ]; then
  echo "==> running full pipeline (backfill, cost, reasoning, full-text, embeddings)"
  if [ "$NO_ENRICH" -eq 0 ]; then
    "$PY" "$REPO/scripts/refresh-all.py" --enrich
  else
    "$PY" "$REPO/scripts/refresh-all.py"
  fi
fi

# 6. Stop + SessionEnd hooks (Stop = instant indexing; SessionEnd = indexing +
#    journal-grade enrichment of the just-ended session)
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 0 ]; then
  echo "==> Claude Code not detected — skipping its Stop/SessionEnd hook and skills (the watcher indexes everything)"
fi
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 1 ]; then
  echo "==> registering Claude Stop + SessionEnd hooks"
  "$PY" - "$REPO" <<'PYEOF' || echo "   ! hook registration failed — everything else still works (watcher covers indexing)"
import json, os, sys, shutil
from pathlib import Path
repo = Path(sys.argv[1])
# A brand-new install has no ~/.claude yet; CLAUDE_CONFIG_DIR relocates it.
settings = Path(os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))/"settings.json"
settings.parent.mkdir(parents=True, exist_ok=True)
# Back up BEFORE parsing: if the user's file is malformed we must not have
# touched anything, and they keep a copy either way.
if settings.exists():
    shutil.copy2(settings, settings.with_suffix(".json.sb-backup"))
try:
    cfg = json.loads(settings.read_text()) if settings.exists() else {}
except (json.JSONDecodeError, OSError) as e:
    print(f"   ! {settings} is not valid JSON ({e}) — skipping hook registration.")
    print("     Fix the file, then re-run: ./install.sh --no-backfill --no-scheduler")
    sys.exit(0)
if not isinstance(cfg, dict):
    print("   ! settings.json is not a JSON object — skipping hook registration"); sys.exit(0)
# Paths quoted: the hook command runs through a shell, and the repo path may
# contain spaces (e.g. ~/My Projects/session-browser).
cmd = f'"{repo}/.venv/bin/python" "{repo}/scripts/session-hook.py"'
hooks = cfg.setdefault("hooks", {})
if not isinstance(hooks, dict):
    print("   ! settings.json 'hooks' is not an object — skipping hook registration"); sys.exit(0)
changed = []
for event in ("Stop", "SessionEnd"):
    entries = hooks.get(event)
    if not isinstance(entries, list):
        entries = hooks[event] = []
    # Presence isn't enough: after the repo is moved, a session-hook.py entry
    # still exists but points at the dead path. Rebuild the desired state (all
    # non-ours entries + exactly one entry with the CURRENT path).
    kept = [h for h in entries if "session-hook.py" not in json.dumps(h)]
    desired = kept + [{"hooks": [{"type": "command", "command": cmd}]}]
    if json.dumps(desired, sort_keys=True) != json.dumps(entries, sort_keys=True):
        hooks[event] = desired
        changed.append(event)
if not changed:
    print("   hooks already present")
else:
    settings.write_text(json.dumps(cfg, indent=2))
    print(f"   hook(s) registered for {', '.join(changed)} (backup: settings.json.sb-backup)")
PYEOF
fi

# 6a'. OpenCode plugin (opt-in) — the Stop-hook equivalent: index a session the
#      moment a turn settles, re-sync on deletion. Auto-loaded from
#      ~/.config/opencode/plugins/; the watcher already covers OpenCode within
#      seconds, so this is for the instant-index feel.
if [ "$OC_PLUGIN" -eq 1 ]; then
  echo "==> installing the OpenCode plugin"
  "$PY" "$REPO/scripts/install-opencode-plugin.py"
elif command -v opencode >/dev/null 2>&1; then
  echo "==> OpenCode detected: add --opencode-plugin for instant indexing (the watcher already covers it within seconds)"
fi

# 6b. Claude skills — symlink each shipped skill into ~/.claude/skills so the
#     work-journal / snapshot / checkpoint skills are discoverable. Symlinks (not
#     copies) so a repo update updates the skills; skill updates ride git pull.
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 1 ]; then
  echo "==> linking Claude skills"
  mkdir -p "$CLAUDE_DIR/skills"
  for d in "$REPO"/skills/*/; do
    name="$(basename "$d")"
    target="$CLAUDE_DIR/skills/$name"
    if [ -L "$target" ]; then
      # repoint after a repo move; no-op when already correct
      [ "$(readlink "$target")" = "${d%/}" ] || { ln -sfn "${d%/}" "$target"; echo "   repointed $name"; }
    elif [ -e "$target" ]; then
      echo "   ! $target exists and is not ours — leaving it alone"
    else
      ln -s "${d%/}" "$target"
      echo "   linked $name"
    fi
  done
fi

# 7. background jobs — launchd (macOS) or systemd --user (Linux). The watcher is
#    live indexing; refresh is the nightly full pipeline (cost/reasoning/fts/
#    embed/enrich). Both are what keep the browser current without a hand run.
if [ "$NO_LAUNCHD" -eq 0 ]; then
  # Job PATH: system dirs + Homebrew (both arches) + wherever the user's CLI
  # binaries actually live (npm globals, volta, nvm…) — enrichment shells out
  # to the CLI, and a PATH miss silently disables it.
  JOB_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$HOME_DIR/.local/bin"
  for c in claude copilot codex opencode; do
    B="$(command -v "$c" 2>/dev/null || true)"
    if [ -n "$B" ]; then D="$(dirname "$B")"; case ":$JOB_PATH:" in *":$D:"*) ;; *) JOB_PATH="$JOB_PATH:$D";; esac; fi
  done
  export SB_VENV_PY="$PY" SB_REPO="$REPO" SB_LOG_DIR="$LOG_DIR" SB_HOME_DIR="$HOME_DIR" SB_JOB_PATH="$JOB_PATH"
  if [ "$(uname)" = "Darwin" ]; then
    echo "==> installing launchd jobs"
    AGENTS="$HOME/Library/LaunchAgents"; mkdir -p "$AGENTS"
    for job in watcher refresh; do
      "$PY" "$REPO/scripts/render-job.py" "$REPO/launchd/$job.plist.template" \
        "$AGENTS/com.sessionbrowser.$job.plist" --format plist
      launchctl unload "$AGENTS/com.sessionbrowser.$job.plist" 2>/dev/null || true
      launchctl load "$AGENTS/com.sessionbrowser.$job.plist"
    done
  elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    echo "==> installing systemd --user units"
    UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"; mkdir -p "$UNITS"
    for u in session-browser-watcher.service session-browser-refresh.service session-browser-refresh.timer; do
      "$PY" "$REPO/scripts/render-job.py" "$REPO/systemd/$u.template" "$UNITS/$u" --format systemd
    done
    systemctl --user daemon-reload
    systemctl --user enable --now session-browser-watcher.service session-browser-refresh.timer \
      || echo "   ! could not enable the units — check: systemctl --user status session-browser-watcher"
    echo "   (units run while you are logged in; to keep them after logout: loginctl enable-linger $USER)"
  else
    echo "==> no launchd / systemd --user session here. Schedule these yourself (docs/SETUP.md §5):"
    echo "      watcher (live indexing):  $PY $REPO/watcher.py"
    echo "      nightly refresh:          $PY $REPO/scripts/refresh-all.py --enrich"
  fi
fi

printf '\n\033[1;32m✓ Session Browser installed.\033[0m\n\n'
echo "Next:"
echo "  ./bin/install-cr.sh      # once — adds the cr / sb shell commands"
echo "  source ~/.zshrc          # or open a new terminal (installer prints which rc)"
echo "  sb ui                    # start the web UI  ->  http://127.0.0.1:7655"
echo "  sb demo                  # or try it on synthetic data first"
echo "  sb doctor                # health check"
