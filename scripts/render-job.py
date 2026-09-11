#!/usr/bin/env python3
"""Render a background-job template — a launchd plist (macOS) or a systemd
--user unit (Linux) — with this install's paths baked in.

    render-job.py TEMPLATE DEST --format plist|systemd

Two things have to run in the background for the browser to stay current: the *watcher*
(a long-lived process that reacts to transcript files changing) and the *nightly refresh*
(scripts/refresh-all.py). macOS and Linux each have their own way to keep such a job alive:

  * **launchd** (macOS) reads an XML property list ("plist") from ~/Library/LaunchAgents/.
    Templates live in launchd/ — watcher.plist.template, refresh.plist.template.
  * **systemd --user** (Linux) reads INI-style unit files from ~/.config/systemd/user/.
    Templates live in systemd/ — one .service per job plus a .timer for the schedule.

The two formats differ in syntax but need exactly the same five facts, so both template
sets use the same `__MARKER__` placeholders and this one renderer fills either. install.sh
exports the values as environment variables and calls this script once per template; the
`--format` flag only chooses how values are ESCAPED and how the extra-environment block is
shaped. Adding a marker means adding one line to MARKERS, not a second renderer.

Markers and the environment variables install.sh sets for them:
    __VENV_PY__  SB_VENV_PY     __REPO__  SB_REPO      __LOG_DIR__  SB_LOG_DIR
    __HOME__     SB_HOME_DIR    __PATH__  SB_JOB_PATH
    __JOB_ENV__  SB_JOB_ENV     optional "NAME=value" lines (CLAUDE_CONFIG_DIR,
                                CODEX_HOME, XDG_DATA_HOME, OPENCODE_DB) rendered
                                as extra EnvironmentVariables / Environment= lines

Why `__JOB_ENV__` exists at all. A background job does not inherit your shell's
environment — launchd and systemd start it from a nearly empty one. This tool lets you move
each CLI's home directory (CLAUDE_CONFIG_DIR, CODEX_HOME, XDG_DATA_HOME, OPENCODE_DB), and
those variables are normally set in .zshrc. A job that does not see them resolves the
DEFAULT locations instead — so the watcher and the nightly refresh would quietly index
empty trees, or (worse, for OPENCODE_DB) a different database than the one you use. Baking
the values into the job file at install time is what keeps the background jobs and your
shell looking at the same data. SB_JOB_ENV carries them as plain "NAME=value" lines, one
per line, and _job_env_lines() turns each into the right syntax for the target format.

Rendered in Python, not sed: a repo path containing &, <, > or sed
metacharacters produced a malformed plist and aborted the install half-done.
plist values are XML-escaped; systemd values are escaped for the double-quoted
strings the templates put them in (ExecStart="…" "…", Environment="PATH=…").

tests/test_smoke.py renders every real template into a temp repo deliberately named
"My Repo & Co" — spaces and an ampersand — and, where `systemd-analyze` exists, asks
systemd itself to validate the result.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

# (placeholder in the template, environment variable install.sh sets). Required: a template
# that mentions a marker whose variable is unset aborts the render rather than writing a job
# file with a literal placeholder in it.
#   __VENV_PY__  absolute path to .venv/bin/python — jobs get a bare PATH with no pyenv
#                shims, so the interpreter must be named in full.
#   __REPO__     the checkout, used to build "<repo>/watcher.py" etc.
#   __LOG_DIR__  where the job's stdout/stderr files go.
#   __HOME__     the user's home; some templates set HOME explicitly for the job.
#   __PATH__     the PATH the job should run with (the summariser CLIs must be findable).
MARKERS = (("__VENV_PY__", "SB_VENV_PY"), ("__REPO__", "SB_REPO"), ("__LOG_DIR__", "SB_LOG_DIR"),
           ("__HOME__", "SB_HOME_DIR"), ("__PATH__", "SB_JOB_PATH"))


def _systemd_escape(value: str) -> str:
    """Escape a value for the double-quoted strings systemd units put it in.

    The templates write ExecStart="__VENV_PY__" "__REPO__/watcher.py" and
    Environment="PATH=__PATH__", so only two characters can break out of the quotes: a
    backslash and a double quote. Backslashes are doubled FIRST — doing it the other way
    round would then escape the backslashes this function just inserted.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _job_env_lines(raw: str, fmt: str) -> str:
    """The jobs used to propagate only PATH: CLAUDE_CONFIG_DIR / CODEX_HOME /
    XDG_DATA_HOME / OPENCODE_DB set in the shell were unknown to the watcher and
    the nightly refresh, which then indexed the default (usually empty) trees.

    `raw` is the SB_JOB_ENV value: zero or more "NAME=value" lines, e.g.

        CLAUDE_CONFIG_DIR=/Volumes/work/.claude
        OPENCODE_DB=/Users/me/alt/opencode.db

    Returns the block to substitute for __JOB_ENV__ — plist <key>/<string> pairs (indented
    two spaces to sit inside the template's <dict>) or systemd Environment="NAME=value"
    lines. An empty input returns an empty string, and render() then removes the marker line
    entirely rather than leaving a dangling <key></key> that would make the plist invalid.
    """
    out = []
    for line in raw.splitlines():
        # Skip anything that is not NAME=value. split("=", 1) keeps the rest of the line
        # intact, so a value containing "=" (a path with a query-ish suffix) survives.
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        # An empty name or value would render a legal-looking but meaningless entry —
        # a job with CLAUDE_CONFIG_DIR set to "" resolves nothing at all.
        if not name or not value:
            continue
        if fmt == "plist":
            out.append(f"<key>{_xml_escape(name)}</key><string>{_xml_escape(value)}</string>")
        else:
            out.append(f'Environment="{_systemd_escape(name + "=" + value)}"')
    return "\n".join(("  " + o if fmt == "plist" else o) for o in out)


def render(template: Path, env: dict, fmt: str) -> str:
    """Substitute every marker in one template and return the finished job file as text.

    `env` is a mapping of variable name to value — os.environ in production, a plain dict in
    the tests. `fmt` selects the escaping ("plist" or "systemd"). Pure: reads the template
    and returns a string, writes nothing.

    Raises SystemExit with a readable message when a marker appears in the template but its
    variable is not set. Failing loudly is the point: the alternative is a job file
    containing a literal __VENV_PY__, which launchd/systemd accept and then fail to run.
    """
    esc = _xml_escape if fmt == "plist" else _systemd_escape
    text = Path(template).read_text(encoding="utf-8")
    for marker, var in MARKERS:
        if marker in text:
            if var not in env:
                raise SystemExit(f"render-job: {var} is not set (needed for {marker} in {template})")
            text = text.replace(marker, esc(env[var]))
    if "__JOB_ENV__" in text:
        block = _job_env_lines(env.get("SB_JOB_ENV", ""), fmt)
        # Two replacements on purpose. The first consumes the marker's whole LINE, so an
        # empty block leaves no blank line behind (and no empty element in the plist). The
        # second is the fallback for a template whose marker is not at the end of a line.
        text = text.replace("__JOB_ENV__\n", block + "\n" if block else "").replace("__JOB_ENV__", block)
    return text


def main() -> None:
    """CLI entry point: render TEMPLATE to DEST, creating parent directories as needed.

    Values come from this process's environment, which install.sh has exported into.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("dest")
    ap.add_argument("--format", choices=("plist", "systemd"), required=True)
    args = ap.parse_args()
    out = render(Path(args.template), os.environ, args.format)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8")
    print(f"   rendered {dest}")


if __name__ == "__main__":
    # `raise SystemExit("message")` would print the message to stderr anyway, but with a
    # traceback-ish framing; this turns a missing-variable error into one clean line and a
    # conventional exit 1, which is what install.sh's `set -e` reacts to. An integer code
    # (argparse's exit 2, or a plain exit 0) is re-raised untouched.
    try:
        main()
    except SystemExit as e:
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
            sys.exit(1)
        raise
