"""Dependency injection: protocols + container.

Pattern: factory functions close over a `Deps` instance.
No framework, no magic; trivial to mock in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ---------- Protocols (match your existing client interfaces) ----------

@runtime_checkable
class SearchEngine(Protocol):
    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]: ...
    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]: ...


@runtime_checkable
class OpenAIClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime_type: str = "image/png",
    ) -> str: ...


@runtime_checkable
class S3Client(Protocol):
    async def get_object(self, key: str) -> bytes: ...


# Postgres checkpointer + store types are imported directly from langgraph
# in graph.py — no protocol needed, the lib types are stable.


# ---------- Container ----------

@dataclass(frozen=True, slots=True)
class Deps:
    """Aggregate of injected dependencies passed to every node factory."""
    search: SearchEngine
    llm: OpenAIClient
    s3: S3Client
