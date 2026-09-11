#!/usr/bin/env bash
# Add the `cr` and `sb` shell functions to your shell rc so you can:
#   cr <session_id>   resume any Claude/Copilot session in the CURRENT directory
#   sb ui|open|stop|doctor|refresh   control the Session Browser
# Idempotent — safe to run repeatedly. Re-running REPLACES the managed blocks,
# so it also repairs stale paths after the repo directory is moved.
# Targets ~/.zshrc (or ~/.bashrc if zsh absent).
#
# WHEN IT RUNS
#   By hand, once, after ./install.sh (which prints it as the next step but deliberately
#   does NOT run it — editing someone's shell startup file is a separate consent).
#   bootstrap.sh does run it, because a `curl | bash` user expects `sb ui` to work after.
#   Re-run it after moving the repo: the blocks are rewritten with the new absolute path.
#
# WHAT IT WRITES
#   Exactly two marker-delimited blocks appended to ONE rc file:
#     # >>> session-browser cr >>>  ...  # <<< session-browser cr <<<
#     # >>> session-browser sb >>>  ...  # <<< session-browser sb <<<
#   Everything else in the rc is preserved byte for byte. Nothing outside that file is
#   touched — no hooks, no services, no database. ./uninstall.sh deletes the same two
#   blocks by the same markers, keeping a .sb-uninstall-backup copy.
#
# WHY SHELL FUNCTIONS AND NOT SYMLINKS ON PATH
#   `cr` needs to run in (and see) the caller's CURRENT directory, and `sb ui` starts a
#   background process the user's shell owns. A function does both without asking for a
#   writable directory on PATH.
#
# Takes no arguments. Exits non-zero only if the rc file cannot be rewritten.
set -euo pipefail
# The repo root is one level up from bin/. Every path written into the rc is absolute and
# derived from it, which is why re-running this repairs a moved repo.
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
# strip_block — delete one managed block from the rc file, in place.
#   $1  the block tag, "cr" or "sb" (it selects the >>> / <<< marker pair).
# Returns 0 when there was nothing to remove (no rc file yet, or no such block) and 42
# when a block WAS removed. That odd return value is the signal the callers below read to
# decide whether to report "Added" or "Updated"; it is not an error. Aborts the whole
# script (exit 1) if the rewrite fails — see the comment on the awk pipeline.
strip_block() {
  local tag="$1"
  [ -f "$RC" ] || return 0
  # -qF: quiet, and treat the marker as a FIXED string — the markers contain > and <,
  # and we want a literal match rather than a regex.
  grep -qF "# >>> session-browser $tag >>>" "$RC" || return 0
  # A failed rewrite must ABORT (not fall through to append) — appending on top
  # of an unstripped block would leave two competing function definitions.
  # awk, not sed: the markers contain characters sed would read as regex metacharacters.
  # `index($0, start) == 1` means "the line STARTS with this exact text" (awk's index is
  # 1-based and returns 0 when not found). Between the two markers `skip` is 1 and lines
  # are dropped; the marker lines themselves are dropped by `next`; everything else is
  # printed unchanged. Output goes to a temp file, which replaces the rc only on success,
  # so an interrupted run can never leave a truncated ~/.zshrc behind.
  awk -v start="# >>> session-browser $tag >>>" -v end="# <<< session-browser $tag <<<" '
    index($0, start) == 1 {skip=1; next}
    index($0, end)   == 1 {skip=0; next}
    !skip {print}
  ' "$RC" > "$RC.sb-tmp" || { rm -f "$RC.sb-tmp"; echo "! failed to rewrite $RC" >&2; exit 1; }
  mv "$RC.sb-tmp" "$RC"
  return 42   # signal "replaced" to the caller
}

# --- the cr block -----------------------------------------------------------------
# `strip_block || CR_STATE="Updated"` reads the 42 above: a non-zero return means a block
# was replaced, so report "Updated"; a clean 0 means this is a first install ("Added"),
# and only then is a blank separator line added before the block. `set -e` does not abort
# on strip_block's 42 because it is the left side of an `||`.
# The heredoc below is DATA — the literal text appended to the user's rc file. It is
# unquoted (<<EOF) so "$REPO" is substituted now, while \$@ is escaped to survive into
# the function body. Do not add anything to it that is not meant to land in the rc.
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

# --- the sb block -----------------------------------------------------------------
# Same Added/Updated dance, same DATA heredoc rules. What the generated function does:
#   sb ui       start session-ui/app.py detached, logging to ~/.session-browser/logs
#   sb stop     kill whatever holds the UI port
#   sb open     open the UI in a browser (macOS `open`, else `xdg-open`)
#   sb stats    terminal usage report        sb demo     synthetic-data preview
#   sb doctor   bin/doctor.sh               sb refresh  the full pipeline, `--enrich` opt
# Note the escaping inside the heredoc: "$REPO" is expanded HERE (so the rc holds this
# machine's absolute path), while \$, \` and \${...} are escaped so they reach the rc
# untouched and are evaluated later, each time the user runs `sb`.
# The UI port is resolved at call time by asking sbconfig for [ui].port — the same value
# session-ui/app.py binds — because the port is documented as user-changeable. A literal
# here meant `sb ui` printed a URL nobody was serving and `sb stop` killed an unrelated
# process that happened to hold the old number; tests/test_smoke.py now greps this block
# to make sure no literal port survives except as the last-resort fallback.
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
