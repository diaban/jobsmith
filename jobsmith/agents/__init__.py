"""The agents this build can compose.

Registering one here is all it takes for `build_app(agent=...)`, the CLI's
`--agent` flag and the API to serve it.
"""
from __future__ import annotations

from .banking import BANKING_AGENT
from .base import AgentDefinition
from .default import DEFAULT_AGENT

AGENTS: dict[str, AgentDefinition] = {
    DEFAULT_AGENT.name: DEFAULT_AGENT,
    BANKING_AGENT.name: BANKING_AGENT,
}

DEFAULT_AGENT_NAME = DEFAULT_AGENT.name


def get_agent(name: str | None) -> AgentDefinition:
    if not name:
        return AGENTS[DEFAULT_AGENT_NAME]
    try:
        return AGENTS[name]
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; known: {', '.join(sorted(AGENTS))}") from None


def agent_names() -> list[str]:
    return sorted(AGENTS)


__all__ = ["AGENTS", "DEFAULT_AGENT_NAME", "AgentDefinition", "agent_names", "get_agent"]
