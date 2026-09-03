"""Composition — wiring only, no content.

This package assembles a running product out of pieces defined elsewhere:
it picks the providers (`providers.py`), opens the persistence backend
(`persistence.py`), and composes an agent definition into a job engine plus
a chat-session factory (`agent.py`).

What each agent can *do* and how it speaks lives in `jobsmith/agents/`, not
here: adding an agent never means touching this package.
"""
from .agent import AgentApp, build_app

__all__ = ["AgentApp", "build_app"]
