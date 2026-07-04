#!/usr/bin/env bash
# Start the Session Browser web UI.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/.venv/bin/python" "$REPO/session-ui/app.py"
