"""Banking-domain dependency protocols and aggregate container.

The framework core only knows LLMClient (core/deps.py). The domain clients
below are consumed directly by the banking capabilities' constructors —
`BankingDeps` is just a convenience aggregate for the composition root.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...core.deps import Deps


@runtime_checkable
class SearchEngine(Protocol):
    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]: ...
    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]: ...


@runtime_checkable
class VisionClient(Protocol):
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


@dataclass(frozen=True, slots=True)
class BankingDeps(Deps):
    """Aggregate for the banking composition root (llm inherited from Deps)."""
    search: SearchEngine
    s3: S3Client
    vision: VisionClient  # may be the same object as llm
