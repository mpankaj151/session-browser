#!/usr/bin/env bash
# Reverse install.sh. Leaves registry.db and the reasoning archive intact unless
# you pass --purge.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURGE=0; [ "${1:-}" = "--purge" ] && PURGE=1

if [ "$(uname)" = "Darwin" ]; then
  echo "==> removing launchd jobs"
  AGENTS="$HOME/Library/LaunchAgents"
  for job in watcher refresh enrich; do
    P="$AGENTS/com.sessionbrowser.$job.plist"
    [ -f "$P" ] && launchctl unload "$P" 2>/dev/null; rm -f "$P"
  done
fi

echo "==> removing Stop hook"
"$REPO/.venv/bin/python" - <<'PYEOF' 2>/dev/null || true
import json
from pathlib import Path
s = Path.home()/".claude"/"settings.json"
if s.exists():
    cfg = json.loads(s.read_text())
    stop = cfg.get("hooks",{}).get("Stop",[])
    stop = [h for h in stop if "session-hook.py" not in json.dumps(h)]
    if stop: cfg["hooks"]["Stop"]=stop
    elif "hooks" in cfg and "Stop" in cfg["hooks"]: del cfg["hooks"]["Stop"]
    s.write_text(json.dumps(cfg, indent=2))
    print("   hook removed")
PYEOF

if [ "$PURGE" -eq 1 ]; then
  echo "==> purging data (~/.session-browser)"
  rm -rf "$HOME/.session-browser"
  echo "   (reasoning archive at ~/claude-reasoning-archive left intact)"
fi
echo "==> uninstalled."
