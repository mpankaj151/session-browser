#!/usr/bin/env python3
"""Install / remove / report the Session Browser OpenCode plugin.

    install-opencode-plugin.py            render + install (idempotent)
    install-opencode-plugin.py --remove
    install-opencode-plugin.py --status   one line for `sb doctor`

The plugin file is rendered from plugins/opencode/session-browser.js.template
with this repo's venv python and hook script baked in, and written to
<config>/opencode/plugins/session-browser.js — OpenCode auto-loads every
plugins/*.js there for every project, so no opencode.json edit is needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "plugins" / "opencode" / "session-browser.js.template"
NAME = "session-browser.js"


def _config_dir(config_dir: Path | str | None = None) -> Path:
    if config_dir:
        return Path(os.path.expanduser(str(config_dir)))
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "opencode"


def plugin_path(config_dir: Path | str | None = None) -> Path:
    return _config_dir(config_dir) / "plugins" / NAME


def render(repo: Path = REPO, python: Path | None = None) -> str:
    python = python or repo / ".venv" / "bin" / "python"
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__VENV_PY__", str(python))
            .replace("__HOOK__", str(repo / "scripts" / "opencode-hook.py")))


def install(config_dir: Path | str | None = None, repo: Path = REPO, python: Path | None = None) -> Path:
    dest = plugin_path(config_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(repo, python), encoding="utf-8")
    return dest


def uninstall(config_dir: Path | str | None = None) -> bool:
    dest = plugin_path(config_dir)
    if not dest.exists():
        return False
    dest.unlink()
    return True


def status(config_dir: Path | str | None = None, repo: Path = REPO, python: Path | None = None) -> str:
    dest = plugin_path(config_dir)
    if not dest.exists():
        return "not installed"
    if dest.read_text(encoding="utf-8") == render(repo, python):
        return "installed"
    return f"stale (differs from this repo's template — re-run ./install.sh --opencode-plugin): {dest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(status())
        return
    if args.remove:
        print("removed" if uninstall() else "not installed")
        return
    dest = install()
    print(f"OpenCode plugin installed -> {dest}")
    print("   fires on session.idle (index now) and session.deleted (re-sync); restart open OpenCode sessions to load it")


if __name__ == "__main__":
    main()
