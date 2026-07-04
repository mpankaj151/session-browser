#!/usr/bin/env bash
# One-command install for Session Browser. Clone (or update) + install.
#
#   curl -fsSL https://raw.githubusercontent.com/mpankaj151/session-browser/main/bootstrap.sh | bash
#
# Env knobs:
#   SB_HOME=~/somewhere    where to clone (default: ~/session-browser)
#   SB_INSTALL_ARGS="--lite --no-launchd"   passed through to install.sh
#     (--lite skips the ~2 GB semantic-search ML stack)
set -euo pipefail

# Everything lives inside main() and the LAST line calls it: a truncated
# `curl | bash` download parses nothing executable, so it can never run half
# a script.
main() {
  local REPO_URL="https://github.com/mpankaj151/session-browser.git"
  local SB_HOME="${SB_HOME:-$HOME/session-browser}"
  local INSTALL_ARGS="${SB_INSTALL_ARGS:-}"

  command -v git >/dev/null 2>&1 || die "git is required. Install it and re-run."
  command -v python3 >/dev/null 2>&1 || die "python3 is required (3.11+). Install it and re-run."
  python3 - <<'PYEOF' || die "Python 3.11+ is required (found older)."
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
PYEOF

  case "$(uname)" in
    Darwin|Linux) : ;;
    *) die "Unsupported OS '$(uname)'. macOS and Linux only." ;;
  esac

  if [ -d "$SB_HOME/.git" ]; then
    say "Updating existing clone at $SB_HOME"
    git -C "$SB_HOME" pull --ff-only || die \
      "Update blocked: local changes in $SB_HOME. Commit/stash them there, or set SB_HOME to a fresh path and re-run."
  else
    say "Cloning into $SB_HOME"
    git clone --depth 1 "$REPO_URL" "$SB_HOME"
  fi

  say "Running installer"
  # shellcheck disable=SC2086
  "$SB_HOME/install.sh" $INSTALL_ARGS

  say "Installing shell shortcuts (cr / sb)"
  "$SB_HOME/bin/install-cr.sh" || true

  # Mirror install-cr.sh's rc choice so "Next steps" names the right file.
  local RC
  case "${SHELL:-}" in
    */zsh)  RC="$HOME/.zshrc" ;;
    */bash) RC="$HOME/.bashrc" ;;
    *)      RC="$HOME/.zshrc"; [ -f "$RC" ] || RC="$HOME/.bashrc" ;;
  esac

  printf '\n\033[1;32mSession Browser installed.\033[0m\n\n'
  printf 'Next:\n'
  printf '  source %s          # load the cr / sb shortcuts (or open a new terminal)\n' "$RC"
  printf '  sb ui                    # start the web UI  -> http://127.0.0.1:7655\n'
  printf '  sb doctor                # health check\n'
  printf '  sb stats                 # usage report in the terminal\n'
  printf '  sb demo                  # a synthetic-data demo (no real sessions needed)\n'
  printf '\nRepo: %s\n' "$SB_HOME"
}

say() { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
die() { printf "\033[1;31m!!\033[0m %s\n" "$1" >&2; exit 1; }

main "$@"
