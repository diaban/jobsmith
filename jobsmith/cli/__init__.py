"""Command line for the global agent: a daemon plus thin clients."""
from .client import AgentClient, DaemonClient, EmbeddedClient, open_client
from .main import main

__all__ = ["AgentClient", "DaemonClient", "EmbeddedClient", "main", "open_client"]
