"""Core dependency injection: the framework only needs an LLM.

Domain-specific clients (search engines, object stores, ...) belong to the
capabilities that use them — capability constructors take exactly the clients
they need. `Deps` can be subclassed by a domain profile to aggregate more
clients for its own composition root (see examples/banking/deps.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface used by the planner / generator / refiner."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class Deps:
    """Framework-level dependencies. Subclass to add domain clients."""
    llm: LLMClient
