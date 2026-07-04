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
# The repo venv may already be gone — any python3 can strip the hook.
UNPY="$REPO/.venv/bin/python"
[ -x "$UNPY" ] || UNPY="$(command -v python3 || true)"
if [ -n "$UNPY" ]; then
  "$UNPY" - <<'PYEOF' || echo "   ! could not edit ~/.claude/settings.json — remove the session-hook.py Stop hook manually"
import json
from pathlib import Path
s = Path.home()/".claude"/"settings.json"
if s.exists():
    cfg = json.loads(s.read_text())
    stop = cfg.get("hooks",{}).get("Stop",[])
    if isinstance(stop, list):
        stop = [h for h in stop if "session-hook.py" not in json.dumps(h)]
        if stop: cfg["hooks"]["Stop"]=stop
        elif "hooks" in cfg and "Stop" in cfg["hooks"]: del cfg["hooks"]["Stop"]
        s.write_text(json.dumps(cfg, indent=2))
        print("   hook removed")
PYEOF
else
  echo "   ! no python3 found — remove the session-hook.py Stop hook from ~/.claude/settings.json manually"
fi

echo "==> removing cr/sb shell functions"
for RC in "$HOME/.zshrc" "$HOME/.bashrc"; do
  [ -f "$RC" ] || continue
  if grep -q "# >>> session-browser" "$RC"; then
    cp "$RC" "$RC.sb-uninstall-backup"
    # delete both marker-delimited blocks (cr and sb)
    sed -i.tmp '/# >>> session-browser cr >>>/,/# <<< session-browser cr <<</d;/# >>> session-browser sb >>>/,/# <<< session-browser sb <<</d' "$RC"
    rm -f "$RC.tmp"
    echo "   removed from $RC (backup: $RC.sb-uninstall-backup)"
  fi
done

if [ "$PURGE" -eq 1 ]; then
  echo "==> purging data (~/.session-browser)"
  rm -rf "$HOME/.session-browser"
  echo "   (reasoning archive at ~/claude-reasoning-archive left intact)"
fi
echo "==> uninstalled. Also present if you want them gone:"
echo "    ~/.claude/settings.json.sb-backup   (pre-install settings backup)"
[ "$PURGE" -eq 0 ] && echo "    ~/.session-browser                  (data — rerun with --purge)"
