#!/usr/bin/env python3
"""Install / remove / report the Session Browser OpenCode plugin.

    install-opencode-plugin.py            render + install (idempotent)
    install-opencode-plugin.py --remove
    install-opencode-plugin.py --status   one line for `sb doctor`

The plugin file is rendered from plugins/opencode/session-browser.js.template
with this repo's venv python and hook script baked in, and written to
<config>/opencode/plugins/session-browser.js — OpenCode auto-loads every
plugins/*.js there for every project, so no opencode.json edit is needed.

Background for anyone who has not met these words (docs/GLOSSARY.md has more): a *session*
is one conversation with a coding CLI; a *hook* is a small program a CLI runs at a fixed
moment; OpenCode's name for that idea is a *plugin* — a JavaScript file it loads at
start-up and calls back for each internal event. Our plugin does no work itself: on
`session.idle` and `session.deleted` it spawns scripts/opencode-hook.py, which indexes the
session. It is the OpenCode counterpart of the Claude Code Stop hook.

RENDERING. The template is copied verbatim except for three placeholder strings, replaced
with absolute paths belonging to THIS checkout: the virtualenv interpreter, the hook script,
and the log directory. The reason they are baked in rather than resolved at run time is that
the plugin runs inside OpenCode's process, with OpenCode's environment — no PATH of ours, no
config of ours, no way to find this repo. A rendered plugin is therefore tied to one
checkout at one path.

STALE DETECTION. Because of that tie, a plugin goes stale whenever the repo moves, the venv
is rebuilt somewhere else, or the template itself changes in a `git pull`. There is no
version stamp to compare; instead status() re-renders the template from the current repo and
compares the result with the installed file byte for byte. Equal means current; different
means stale, and the fix is always the same — re-run the installer, which overwrites it.
`sb doctor` prints that one line.
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
    """OpenCode's config directory: $XDG_CONFIG_HOME/opencode, else ~/.config/opencode.

    Resolved the same way OpenCode itself does, so the plugin lands where it will actually
    be loaded. The explicit `config_dir` argument exists for the tests, which point this at
    a temporary directory instead of the user's real one.
    """
    if config_dir:
        return Path(os.path.expanduser(str(config_dir)))
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "opencode"


def plugin_path(config_dir: Path | str | None = None) -> Path:
    """Where the rendered plugin lives: <config dir>/plugins/session-browser.js."""
    return _config_dir(config_dir) / "plugins" / NAME


def render(repo: Path = REPO, python: Path | None = None) -> str:
    """Return the plugin source with the three install-time placeholders filled in.

    Pure: reads the template and returns text, writes nothing — which is what lets status()
    use it as the comparison baseline for stale detection.

    `python` defaults to this repo's own .venv interpreter. It is an argument because the
    tests render with sys.executable (CI has no .venv), and because a `--lite` install may
    keep its interpreter elsewhere. The log directory comes from sbconfig, imported lazily
    so that merely importing this module does not read config.
    """
    python = python or repo / ".venv" / "bin" / "python"
    import sbconfig
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__VENV_PY__", str(python))
            .replace("__HOOK__", str(repo / "scripts" / "opencode-hook.py"))
            .replace("__LOG_DIR__", str(sbconfig.LOG_DIR)))


def install(config_dir: Path | str | None = None, repo: Path = REPO, python: Path | None = None) -> Path:
    """Write the rendered plugin, creating the plugins directory. Returns the path written.

    Idempotent and unconditional: an existing file is overwritten, which is also how a stale
    plugin is repaired. OpenCode loads plugins at start-up, so an already-running session
    keeps the old copy until it is restarted.
    """
    dest = plugin_path(config_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(repo, python), encoding="utf-8")
    return dest


def uninstall(config_dir: Path | str | None = None) -> bool:
    """Delete the installed plugin. True if one was there, False if there was nothing to do.

    Only ever removes the single file this script writes — never the plugins directory,
    which may hold other people's plugins.
    """
    dest = plugin_path(config_dir)
    if not dest.exists():
        return False
    dest.unlink()
    return True


def status(config_dir: Path | str | None = None, repo: Path = REPO, python: Path | None = None) -> str:
    """One human-readable line for `sb doctor`: not installed / installed / stale.

    "Stale" is decided by re-rendering the template from this repo and comparing the text
    with the installed file — see the module docstring. A byte difference means the baked-in
    paths or the template changed, so the installed copy may point at a venv or a hook
    script that no longer exists; the message names the file and the command that fixes it.
    """
    dest = plugin_path(config_dir)
    if not dest.exists():
        return "not installed"
    if dest.read_text(encoding="utf-8") == render(repo, python):
        return "installed"
    return f"stale (differs from this repo's template — re-run ./install.sh --opencode-plugin): {dest}"


def main() -> None:
    """CLI entry point: --status reports, --remove uninstalls, no flag installs."""
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
