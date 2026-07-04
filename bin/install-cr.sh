#!/usr/bin/env bash
# Add the `cr` and `sb` shell functions to your shell rc so you can:
#   cr <session_id>   resume any Claude/Copilot session in the CURRENT directory
#   sb ui|open|stop|doctor|refresh   control the Session Browser
# Idempotent — safe to run repeatedly. Targets ~/.zshrc (or ~/.bashrc if zsh absent).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RC="$HOME/.zshrc"
[ -f "$RC" ] || RC="$HOME/.bashrc"

if ! grep -qF "# >>> session-browser cr >>>" "$RC" 2>/dev/null; then
  cat >> "$RC" <<EOF

# >>> session-browser cr >>>
# Resume any Claude/Copilot session in the CURRENT directory: cr <session_id>
cr() { "$REPO/bin/resume-here.sh" "\$@"; }
# <<< session-browser cr <<<
EOF
  echo "Added cr() to $RC"
else
  echo "cr already installed in $RC"
fi

if ! grep -qF "# >>> session-browser sb >>>" "$RC" 2>/dev/null; then
  cat >> "$RC" <<EOF

# >>> session-browser sb >>>
# Control the Session Browser: sb {ui|stop|open|stats|demo|doctor|refresh [--enrich]}
sb() {
  local REPO="$REPO"
  case "\${1:-}" in
    ui)      nohup "\$REPO/.venv/bin/python" "\$REPO/session-ui/app.py" >/tmp/sb-app.log 2>&1 & echo "Session Browser UI -> http://127.0.0.1:7655" ;;
    stop)    lsof -ti tcp:7655 2>/dev/null | xargs kill 2>/dev/null && echo "UI stopped" || echo "UI not running" ;;
    open)    open http://127.0.0.1:7655 2>/dev/null || xdg-open http://127.0.0.1:7655 ;;
    stats)   "\$REPO/.venv/bin/python" "\$REPO/scripts/stats-report.py" "\${@:2}" ;;
    demo)    "\$REPO/.venv/bin/python" "\$REPO/scripts/demo.py" ;;
    doctor)  "\$REPO/bin/doctor.sh" ;;
    refresh) "\$REPO/.venv/bin/python" "\$REPO/scripts/refresh-all.py" "\${@:2}" ;;
    *)       echo "usage: sb {ui|stop|open|stats|demo|doctor|refresh [--enrich]}" ;;
  esac
}
# <<< session-browser sb <<<
EOF
  echo "Added sb() to $RC"
else
  echo "sb already installed in $RC"
fi

echo "Run:  source $RC   (or open a new terminal), then:  cr <session_id>  /  sb ui"
