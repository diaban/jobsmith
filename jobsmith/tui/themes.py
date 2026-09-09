"""The palettes the terminal UI ships with, and how one is chosen.

Textual already carries 21 themes and puts `ThemeProvider` in the default
command palette, so **Ctrl+P switches theme with nothing written here**. What
this module adds is two of our own — Ember and Tide, in a dark and a light
cut each — registered so they appear in that same picker next to `nord`.

Ember is blue-grey surfaces with one warm accent, so the running job is the
only warm thing on screen. Tide is a deeper blue with a cyan accent, where
structure reads before state. Both are built so that the **standard roles
carry the intent**: `primary` dresses chrome, `accent` says "something is
happening", `success`/`error` are done and failed. That is what lets
`tui/render.py` name nothing but standard roles — and naming nothing but
standard roles is what lets a foreign theme work at all, because a
stylesheet mentioning a variable the current theme does not define **fails to
parse** and the app does not start.

Choosing one follows the house pattern (`app/persistence.py::pick_db`,
`app/agent.py::pick_report_formats`): argument > `--theme=` > `$JOBSMITH_THEME`
> `ember-dark`. A theme picked at runtime through Ctrl+P is deliberately not
persisted: remembering it means this project's first configuration file, which
is a concept rather than a setting.
"""
from __future__ import annotations

import os
import sys

from textual.theme import Theme

DEFAULT_THEME = "ember-dark"

EMBER_DARK = Theme(
    name="ember-dark", dark=True,
    background="#0e1116", surface="#161b22", panel="#1c232e", boost="#222b38",
    foreground="#ccd3da", primary="#6d9fc4", secondary="#7d8da3", accent="#e8873c",
    success="#4bb861", warning="#d9a227", error="#f0605d",
)
EMBER_LIGHT = Theme(
    name="ember-light", dark=False,
    background="#faf8f5", surface="#ffffff", panel="#f4f0ea", boost="#eae4db",
    foreground="#2b2620", primary="#33648a", secondary="#6b7280", accent="#b8560f",
    success="#2f8f45", warning="#a86a08", error="#c73b38",
)
TIDE_DARK = Theme(
    name="tide-dark", dark=True,
    background="#0a0f18", surface="#101827", panel="#16202f", boost="#1d2a3c",
    foreground="#d5dde7", primary="#5b7fa6", secondary="#64748b", accent="#38bdf8",
    success="#34d399", warning="#fbbf24", error="#fb7185",
)
TIDE_LIGHT = Theme(
    name="tide-light", dark=False,
    background="#f7fafc", surface="#ffffff", panel="#eef3f8", boost="#e3ebf3",
    foreground="#1f2b3a", primary="#4a6b8a", secondary="#546b82", accent="#0369a1",
    success="#15803d", warning="#a16207", error="#be123c",
)

THEMES: dict[str, Theme] = {
    theme.name: theme for theme in (EMBER_DARK, EMBER_LIGHT, TIDE_DARK, TIDE_LIGHT)
}


def pick_theme(explicit: str | None = None) -> str:
    """Resolve the theme name: argument > --theme= flag > $JOBSMITH_THEME > default."""
    if explicit:
        return explicit
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--theme=")), None)
    return flag or os.environ.get("JOBSMITH_THEME") or DEFAULT_THEME
