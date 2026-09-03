"""Core dependency injection: the framework only needs an LLM.

Domain-specific clients (search engines, object stores, ...) belong to the
capabilities that use them — capability constructors take exactly the clients
they need. `Deps` can be subclassed by a domain profile to aggregate more
clients for its own composition root (see jobsmith/agents/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface used by the planner / generator / refiner.

    `chat` returns the text and nothing else, on purpose. Token usage is
    reported through the ambient ledger in `core/usage.py`
    (`record_usage(...)` after each response) rather than by widening this
    return type: every capability, the planner and the generator are written
    against `-> str`, and threading a `(text, usage)` tuple through all of
    them — including capabilities living outside this repo — would be a large
    edit to carry a number most call sites never look at. Implementing `chat`
    stays a one-method job; an adapter that reports no usage is simply absent
    from the accounting.
    """

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
