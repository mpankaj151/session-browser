#!/usr/bin/env bash
# One-command install for Session Browser. Clone (or update) + install.
#
#   curl -fsSL https://raw.githubusercontent.com/mpankaj151/session-browser/main/bootstrap.sh | bash
#
# Env knobs:
#   SB_HOME=~/somewhere    where to clone (default: ~/session-browser)
#   SB_INSTALL_ARGS="--lite --no-launchd"   passed through to install.sh
set -euo pipefail

REPO_URL="https://github.com/mpankaj151/session-browser.git"
SB_HOME="${SB_HOME:-$HOME/session-browser}"
INSTALL_ARGS="${SB_INSTALL_ARGS:-}"

say() { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
die() { printf "\033[1;31m!!\033[0m %s\n" "$1" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required. Install it and re-run."
command -v python3 >/dev/null 2>&1 || die "python3 is required (3.11+). Install it and re-run."
python3 - <<'PY' || die "Python 3.11+ is required (found older)."
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY

case "$(uname)" in
  Darwin|Linux) : ;;
  *) die "Unsupported OS '$(uname)'. macOS and Linux only." ;;
esac

if [ -d "$SB_HOME/.git" ]; then
  say "Updating existing clone at $SB_HOME"
  git -C "$SB_HOME" pull --ff-only
else
  say "Cloning into $SB_HOME"
  git clone --depth 1 "$REPO_URL" "$SB_HOME"
fi

say "Running installer"
# shellcheck disable=SC2086
"$SB_HOME/install.sh" $INSTALL_ARGS

say "Installing shell shortcuts (cr / sb)"
"$SB_HOME/bin/install-cr.sh" || true

cat <<EOF

\033[1;32mSession Browser installed.\033[0m

Next:
  source ~/.zshrc          # load the cr / sb shortcuts (or open a new terminal)
  sb ui                    # start the web UI  -> http://127.0.0.1:7655
  sb doctor                # health check
  sb stats                 # usage report in the terminal
  sb demo                  # a synthetic-data demo (no real sessions needed)

Repo: $SB_HOME
EOF
