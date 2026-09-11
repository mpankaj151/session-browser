"""Pluggable summarisers ("providers") for enrichment.

Enrichment is the tool's only call to an AI model: a headless, non-interactive run of
one of the coding CLIs (`claude --print`, `opencode run`, `copilot -p`) that reads a
session transcript and returns a JSON "facet" (title, summary, topics, outcome ...).
`provider.py` holds the shared protocol, prompt rendering, JSON parsing and facet
storage; the *_headless.py modules wrap one CLI each; `null_provider.py` is the no-op
used when no CLI is available. See docs/GLOSSARY.md for "enrichment", "facet",
"headless run" and "token".
"""
