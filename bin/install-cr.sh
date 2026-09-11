#!/usr/bin/env bash
# Add the `cr` and `sb` shell functions to your shell rc so you can:
#   cr <session_id>   resume any Claude/Copilot session in the CURRENT directory
#   sb ui|open|stop|doctor|refresh   control the Session Browser
# Idempotent — safe to run repeatedly. Re-running REPLACES the managed blocks,
# so it also repairs stale paths after the repo directory is moved.
# Targets ~/.zshrc (or ~/.bashrc if zsh absent).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Pick the rc by the LOGIN SHELL, not file existence: a fresh macOS account has
# zsh but no ~/.zshrc yet — writing to ~/.bashrc there installs functions that
# the user's shell never sources.
case "${SHELL:-}" in
  */zsh)  RC="$HOME/.zshrc" ;;
  */bash) RC="$HOME/.bashrc" ;;
  *)      RC="$HOME/.zshrc"; [ -f "$RC" ] || RC="$HOME/.bashrc" ;;
esac

# Remove an existing managed block (between its >>> / <<< markers) so the
# append below always installs the current repo path. Presence-checking alone
# left stale paths behind after a repo move.
strip_block() {
  local tag="$1"
  [ -f "$RC" ] || return 0
  grep -qF "# >>> session-browser $tag >>>" "$RC" || return 0
  # A failed rewrite must ABORT (not fall through to append) — appending on top
  # of an unstripped block would leave two competing function definitions.
  awk -v start="# >>> session-browser $tag >>>" -v end="# <<< session-browser $tag <<<" '
    index($0, start) == 1 {skip=1; next}
    index($0, end)   == 1 {skip=0; next}
    !skip {print}
  ' "$RC" > "$RC.sb-tmp" || { rm -f "$RC.sb-tmp"; echo "! failed to rewrite $RC" >&2; exit 1; }
  mv "$RC.sb-tmp" "$RC"
  return 42   # signal "replaced" to the caller
}

CR_STATE="Added"
strip_block "cr" || CR_STATE="Updated"
if [ "$CR_STATE" = "Added" ]; then printf '\n' >> "$RC"; fi
cat >> "$RC" <<EOF
# >>> session-browser cr >>>
# Resume any Claude/Copilot session in the CURRENT directory: cr <session_id>
cr() { "$REPO/bin/resume-here.sh" "\$@"; }
# <<< session-browser cr <<<
EOF
echo "$CR_STATE cr() in $RC"

SB_STATE="Added"
strip_block "sb" || SB_STATE="Updated"
if [ "$SB_STATE" = "Added" ]; then printf '\n' >> "$RC"; fi
cat >> "$RC" <<EOF
# >>> session-browser sb >>>
# Control the Session Browser: sb {ui|stop|open|stats|demo|doctor|refresh [--enrich]}
sb() {
  local REPO="$REPO"
  # [ui].port from config (docs/SETUP.md tells users to change it): a fixed
  # number here printed the wrong URL and made \`sb stop\` kill whatever
  # unrelated process held it.
  local P; P="\$("\$REPO/.venv/bin/python" -c 'import sys; sys.path.insert(0, sys.argv[1]); import sbconfig; print(int(sbconfig.CONFIG.get("ui", {}).get("port", 7655)))' "\$REPO" 2>/dev/null || echo 7655)"
  local PIDS
  case "\${1:-}" in
    ui)      if [ -n "\$(lsof -ti tcp:"\$P" 2>/dev/null || fuser "\$P"/tcp 2>/dev/null)" ]; then echo "Port \$P is already in use — the UI may already be running: http://127.0.0.1:\$P  (sb stop to restart)"; else mkdir -p "\$HOME/.session-browser/logs"; nohup "\$REPO/.venv/bin/python" "\$REPO/session-ui/app.py" >"\$HOME/.session-browser/logs/ui.log" 2>&1 & echo "Session Browser UI -> http://127.0.0.1:\$P"; fi ;;
    stop)    PIDS="\$(lsof -ti tcp:"\$P" 2>/dev/null || fuser "\$P"/tcp 2>/dev/null)"; if [ -n "\$PIDS" ]; then kill \$PIDS 2>/dev/null && echo "UI stopped"; else echo "UI not running"; fi ;;
    open)    open "http://127.0.0.1:\$P" 2>/dev/null || xdg-open "http://127.0.0.1:\$P" ;;
    stats)   "\$REPO/.venv/bin/python" "\$REPO/scripts/stats-report.py" "\${@:2}" ;;
    demo)    "\$REPO/.venv/bin/python" "\$REPO/scripts/demo.py" "\${@:2}" ;;
    doctor)  "\$REPO/bin/doctor.sh" ;;
    refresh) "\$REPO/.venv/bin/python" "\$REPO/scripts/refresh-all.py" "\${@:2}" ;;
    *)       echo "usage: sb {ui|stop|open|stats|demo|doctor|refresh [--enrich]}" ;;
  esac
}
# <<< session-browser sb <<<
EOF
echo "$SB_STATE sb() in $RC"

echo "Run:  source $RC   (or open a new terminal), then:  cr <session_id>  /  sb ui"
