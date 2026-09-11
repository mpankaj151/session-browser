"""One adapter per AI coding CLI, all speaking the SessionSource protocol.

`base.py` defines the protocol and the SessionHeader/Turn records; `claude.py`,
`copilot.py`, `codex.py` and `opencode.py` each know one CLI's on-disk transcript
layout; `registry.py` builds the enabled set from config.toml. Nothing outside this
package knows which CLI a session came from. See docs/ADDING-A-CLI.md and
docs/GLOSSARY.md.
"""
