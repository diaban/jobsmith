"""`jobsmith ui` — the terminal UI, and the one honest thing to say without it.

Textual is an extra (`.[tui]`), exactly like the PDF renderer and the deck
writer: what nothing can serve stays out of the way. So the import lives
inside `run_tui`, and its absence is reported as the one-line install
instruction rather than as a traceback about a module nobody asked for.
"""
from __future__ import annotations

from typing import Any

MISSING = (
    "the terminal UI needs the .[tui] extra — install it with:\n"
    "    pip install 'jobsmith[tui]'    (in this checkout: "
    "uv pip install -e '.[tui]')\n"
    "`jobsmith chat` is the same conversation without it."
)


class TuiUnavailable(RuntimeError):
    """Textual is not installed. Carries what to install, not a stack trace."""


async def run_tui(service: Any, session_id: str, *, theme: str | None = None) -> None:
    """Run the UI against a composed service until the user leaves it."""
    try:
        from .app import JobsmithApp
    except ImportError as missing:      # textual absent
        raise TuiUnavailable(MISSING) from missing
    await JobsmithApp(service, session_id, theme=theme).run_async()
