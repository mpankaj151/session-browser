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

echo "==> removing Stop + SessionEnd hooks"
# The repo venv may already be gone — any python3 can strip the hooks.
UNPY="$REPO/.venv/bin/python"
[ -x "$UNPY" ] || UNPY="$(command -v python3 || true)"
if [ -n "$UNPY" ]; then
  "$UNPY" - <<'PYEOF' || echo "   ! could not edit ~/.claude/settings.json — remove the session-hook.py hooks manually"
import json
from pathlib import Path
s = Path.home()/".claude"/"settings.json"
if s.exists():
    cfg = json.loads(s.read_text())
    hooks = cfg.get("hooks", {})
    for event in ("Stop", "SessionEnd"):
        entries = hooks.get(event, [])
        if isinstance(entries, list):
            entries = [h for h in entries if "session-hook.py" not in json.dumps(h)]
            if entries: hooks[event] = entries
            elif event in hooks: del hooks[event]
    s.write_text(json.dumps(cfg, indent=2))
    print("   hooks removed")
PYEOF
else
  echo "   ! no python3 found — remove the session-hook.py hooks from ~/.claude/settings.json manually"
fi

echo "==> removing Claude skill links"
for d in "$REPO"/skills/*/; do
  name="$(basename "$d")"
  target="$HOME/.claude/skills/$name"
  # only remove links that point INTO this repo — never a user's own skill
  if [ -L "$target" ] && [ "$(readlink "$target")" = "${d%/}" ]; then
    rm -f "$target" && echo "   unlinked $name"
  fi
done

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
