#!/usr/bin/env bash
# Start the Session Browser web UI.
#
#   ./bin/start-session-ui.sh        no arguments
#
# Runs session-ui/app.py — the Flask server plus the single-page front end — in the
# FOREGROUND, attached to this terminal, logging to the screen. Stop it with Ctrl-C.
# It binds 127.0.0.1 only (never reachable from another machine) on the port from
# [ui].port in config.toml. It writes no files of its own; the app reads the registry
# database and serves it.
#
# WHEN TO USE THIS RATHER THAN `sb ui`
#   `sb ui` (the shell function bin/install-cr.sh writes) is the everyday way: it starts
#   the same app detached, with output redirected to ~/.session-browser/logs/ui.log, and
#   prints the URL. Use this script when you want the server in the foreground with its
#   log on screen — debugging a request, or a machine without the shell functions.
#
# `exec` replaces this shell with the Python process, so there is no extra wrapper process
# between your terminal and the server: Ctrl-C and signals reach the app directly, and it
# is the app's own exit status you get back.
set -euo pipefail
# The venv interpreter by absolute path, so the UI runs with this tool's dependencies no
# matter which directory or Python environment the caller is in.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/.venv/bin/python" "$REPO/session-ui/app.py"
