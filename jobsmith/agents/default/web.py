"""Web search as a `DocumentSource`.

The second adapter behind the port `sources.py` defines — and the reason that
port was shaped by the capability's need rather than by any backend: nothing
in `DocumentsCapability` changes to gain the web, only what is injected into
it.

Tavily returns passages already extracted and ranked for LLM use, which is
what makes the mapping to `Document` honest: `content` is a real excerpt, not
a title plus ellipsis. The HTTP client is injected so tests never reach the
network, and `raw_content` is deliberately left off — fetching whole pages is
a separate, slower, costlier decision.
"""
from __future__ import annotations

from typing import Any

from .sources import Document

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 20          # the API's own ceiling


class TavilySource:
    """`DocumentSource` over Tavily's search endpoint.

    Errors are raised, not swallowed: `DocumentsCapability` already isolates
    one failing query from the others, and a silent empty result would look
    exactly like "the web knows nothing about this".
    """

    def __init__(
        self,
        api_key: str,
        client: Any,
        *,
        search_depth: str = "basic",
        topic: str = "general",
        timeout: float = 20.0,
    ):
        self.api_key = api_key
        self.client = client            # an httpx.AsyncClient owned by the caller
        self.search_depth = search_depth
        self.topic = topic
        self.timeout = timeout

    async def search(self, query: str, *, limit: int = 8) -> list[Document]:
        if not query.strip():
            return []
        response = await self.client.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "max_results": max(1, min(limit, MAX_RESULTS)),
                "search_depth": self.search_depth,
                "topic": self.topic,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        documents = []
        for hit in payload.get("results") or []:
            url = (hit.get("url") or "").strip()
            text = (hit.get("content") or "").strip()
            if not url or not text:
                continue            # a result with no excerpt grounds nothing
            documents.append(Document(
                # the URL IS the id: a citation a human can open and check
                id=url,
                text=text,
                title=(hit.get("title") or url).strip(),
                source=url,
                score=float(hit.get("score") or 0.0),
            ))
        return documents
