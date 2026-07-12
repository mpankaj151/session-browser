#!/usr/bin/env bash
# Session Browser installer.
#   ./install.sh [--no-hook] [--no-launchd] [--no-backfill] [--enrich] [--lite]
# Idempotent. Creates the venv, builds the DB, backfills, optionally registers the
# Claude Stop hook and (macOS) launchd jobs.
#   --lite  skip sentence-transformers/torch (~2 GB download); semantic search
#           falls back to keyword + full-text.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
HOME_DIR="$HOME"
LOG_DIR="$HOME/.session-browser/logs"

NO_HOOK=0; NO_LAUNCHD=0; NO_BACKFILL=0; NO_ENRICH=1; LITE=0   # enrich off by default (uses LLM quota)
for a in "$@"; do case "$a" in
  --no-hook) NO_HOOK=1;; --no-launchd) NO_LAUNCHD=1;;
  --no-backfill) NO_BACKFILL=1;; --enrich) NO_ENRICH=0;; --lite) LITE=1;;
  *) echo "unknown flag: $a"; exit 1;; esac; done

echo "==> Session Browser install ($REPO)"

# Python 3.11+ required (tomllib, sqlite FTS5). Don't hard-require that plain
# `python3` be new: stock macOS resolves it to the Xcode CLT build (often 3.9)
# even when a modern interpreter is installed as python3.12 — hunt for one.
PYBOOT=""
for c in python3.13 python3.12 python3.11 python3; do
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

# 2. config
[ -f "$REPO/config.toml" ] || cp "$REPO/config.toml.example" "$REPO/config.toml"

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
if [ "$NO_HOOK" -eq 0 ]; then
  echo "==> registering Claude Stop + SessionEnd hooks"
  "$PY" - "$REPO" <<'PYEOF' || echo "   ! hook registration failed — everything else still works (watcher covers indexing)"
import json, sys, shutil
from pathlib import Path
repo = Path(sys.argv[1])
settings = Path.home()/".claude"/"settings.json"
# Back up BEFORE parsing: if the user's file is malformed we must not have
# touched anything, and they keep a copy either way.
if settings.exists():
    shutil.copy2(settings, settings.with_suffix(".json.sb-backup"))
try:
    cfg = json.loads(settings.read_text()) if settings.exists() else {}
except (json.JSONDecodeError, OSError) as e:
    print(f"   ! ~/.claude/settings.json is not valid JSON ({e}) — skipping hook registration.")
    print("     Fix the file, then re-run: ./install.sh --no-backfill --no-launchd")
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

# 7. launchd (macOS only; on Linux schedule watcher.py + refresh-all.py via systemd/cron)
if [ "$(uname)" != "Darwin" ] && [ "$NO_LAUNCHD" -eq 0 ]; then
  NO_LAUNCHD=1
  echo "==> skipping launchd (not macOS). Schedule these yourself:"
  echo "      watcher (live indexing):  $PY $REPO/watcher.py"
  echo "      nightly refresh:          $PY $REPO/scripts/refresh-all.py --enrich"
fi
if [ "$NO_LAUNCHD" -eq 0 ]; then
  echo "==> installing launchd jobs"
  AGENTS="$HOME/Library/LaunchAgents"; mkdir -p "$AGENTS"
  # launchd PATH: system dirs + Homebrew (both arches) + wherever the user's CLI
  # binaries actually live (npm globals, volta, etc.) — enrichment shells out to
  # `claude`, and a PATH miss silently disables it.
  JOB_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$HOME_DIR/.local/bin"
  for c in claude copilot codex; do
    B="$(command -v "$c" 2>/dev/null || true)"
    if [ -n "$B" ]; then D="$(dirname "$B")"; case ":$JOB_PATH:" in *":$D:"*) ;; *) JOB_PATH="$JOB_PATH:$D";; esac; fi
  done
  # watcher = live indexing; refresh = nightly full pipeline (cost/reasoning/fts/embed/enrich)
  # Rendered in Python, not sed: paths containing &, <, > or sed metacharacters
  # would otherwise produce malformed plist XML and abort the install half-done.
  for job in watcher refresh; do
    SB_TEMPLATE="$REPO/launchd/$job.plist.template" \
    SB_DEST="$AGENTS/com.sessionbrowser.$job.plist" \
    SB_VENV_PY="$PY" SB_REPO="$REPO" SB_LOG_DIR="$LOG_DIR" \
    SB_HOME_DIR="$HOME_DIR" SB_JOB_PATH="$JOB_PATH" \
    "$PY" - <<'PYEOF'
import os
from xml.sax.saxutils import escape
tpl = open(os.environ["SB_TEMPLATE"], encoding="utf-8").read()
for marker, env in (("__VENV_PY__", "SB_VENV_PY"), ("__REPO__", "SB_REPO"),
                    ("__LOG_DIR__", "SB_LOG_DIR"), ("__HOME__", "SB_HOME_DIR"),
                    ("__PATH__", "SB_JOB_PATH")):
    tpl = tpl.replace(marker, escape(os.environ[env]))
open(os.environ["SB_DEST"], "w", encoding="utf-8").write(tpl)
PYEOF
    launchctl unload "$AGENTS/com.sessionbrowser.$job.plist" 2>/dev/null || true
    launchctl load "$AGENTS/com.sessionbrowser.$job.plist"
  done
fi

printf '\n\033[1;32m✓ Session Browser installed.\033[0m\n\n'
echo "Next:"
echo "  ./bin/install-cr.sh      # once — adds the cr / sb shell commands"
echo "  source ~/.zshrc          # or open a new terminal (installer prints which rc)"
echo "  sb ui                    # start the web UI  ->  http://127.0.0.1:7655"
echo "  sb demo                  # or try it on synthetic data first"
echo "  sb doctor                # health check"
