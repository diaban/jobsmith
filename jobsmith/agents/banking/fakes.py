"""In-memory adapters for the banking agent's ports.

They exist so the example runs with no backend at all. Swapping in a real
search engine or object store means writing another class with the same
methods and wiring it in `__init__.py` — the capabilities never change,
because they depend on the Protocols in `deps.py`, not on these.
"""
from __future__ import annotations

from typing import Any


class KeywordSearch:
    """`SearchEngine` — canned hits, deterministic."""

    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        if "past_slides" in query:
            return [{"id": "ref_0", "summary": f"deck related to '{query[:40]}'"}]
        return [
            {"id": "doc_0", "text": f"KB article matching '{query[:40]}'."},
            {"id": "doc_1", "text": "Second supporting document."},
        ]

    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        return await self.search(query, top_k=top_k)


class FakeS3:
    """`S3Client` — returns bytes that look like an image payload."""

    async def get_object(self, key: str) -> bytes:
        return f"fake-image-bytes:{key}".encode()
