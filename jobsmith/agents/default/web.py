"""Web search as a `DocumentSource`.

The second adapter behind the port `sources.py` defines — and the reason that
port was shaped by the capability's need rather than by any backend: nothing
in `DocumentsCapability` changes to gain the web, only what is injected into
it.

**What a result carries is an adapter decision, and it was the wrong one**
(#75). Tavily returns two texts per hit: `content`, a relevance-cropped
snippet of a few hundred characters, and `raw_content`, the page's extracted
text — returned only when `include_raw_content` is asked for. This adapter
asked for neither and mapped the snippet, so `web_search`, the step whose
whole purpose is to ground a run in something outside the model, grounded it
in ten excerpts. A request for spec sheets, warranties and prices was
unreachable by construction, and the run reported it as "the material is
insufficient" — a statement about the world, made about our retrieval.

So the page wins when there is one and the snippet is the fallback (Tavily
returns `raw_content: null` for a page it could not fetch, and a snippet
grounds more than nothing), and three consequences are handled *here*, where
the cost is known: the depth the search runs at, the size of what comes back,
and the fact that a page which was cut says so.
"""
from __future__ import annotations

from typing import Any

from .sources import Document

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 20          # the API's own ceiling
SEARCH_DEPTHS = ("basic", "advanced")

#: Written into the text at the cut, never left implicit — see `_bounded`.
TRUNCATION_NOTE = "\n\n…[truncated: only the first {kept} characters of this page were read]"


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
        search_depth: str = "advanced",
        topic: str = "general",
        timeout: float = 20.0,
        max_chars: int = 8_000,
    ):
        # A depth the API does not know is a misconfiguration, and sending it
        # on would either 400 in the middle of a job or — worse — be ignored
        # server-side and quietly retrieve less. Refused where it is read.
        if search_depth not in SEARCH_DEPTHS:
            raise ValueError(
                f"search_depth must be one of {', '.join(SEARCH_DEPTHS)}, got {search_depth!r}")
        self.api_key = api_key
        self.client = client            # an httpx.AsyncClient owned by the caller
        self.search_depth = search_depth
        self.topic = topic
        self.timeout = timeout
        # The budget, per document, and the arithmetic behind it. The consumer
        # (`DocumentsCapability`) keeps `max_documents=10`, and `render_context`
        # concatenates every one of them with no bound of its own, so the worst
        # case this adapter can hand it is 10 × 8 000 = 80 000 characters, i.e.
        # roughly 20 000 tokens at ~4 characters a token. That block is then
        # re-sent by every downstream step that reads the merged context
        # (research, analysis, critique, generation), so the *billed* figure is
        # a small multiple of it — which is why the bound is this side of
        # generous. It is not a claim that 8 000 characters is a whole page:
        # an extracted page runs 50–100k, and ten of those would be ~250k
        # tokens, more than the window before a single step has thought. What
        # it buys over the snippet it replaces (~500 characters) is sixteen
        # times the material, and the part of a product or spec page that
        # answers a question — the table, the price, the warranty line —
        # is almost always above the navigation-and-footer padding that fills
        # the rest.
        self.max_chars = max_chars

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
                # the point of #75: the page, not the search engine's crop of it
                "include_raw_content": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        documents = []
        for hit in payload.get("results") or []:
            url = (hit.get("url") or "").strip()
            # The page when there is one; the snippet when there is not.
            # `raw_content` is null for a hit Tavily could not fetch, and an
            # excerpt grounds more than an empty document does.
            text = (hit.get("raw_content") or "").strip() or (hit.get("content") or "").strip()
            if not url or not text:
                continue            # a result with no text grounds nothing
            documents.append(Document(
                # the URL IS the id: a citation a human can open and check
                id=url,
                text=self._bounded(text),
                title=(hit.get("title") or url).strip(),
                source=url,
                score=float(hit.get("score") or 0.0),
            ))
        return documents

    def _bounded(self, text: str) -> str:
        """The first `max_chars` characters, cut at a boundary and marked.

        Both halves matter. The cut lands on whitespace rather than inside a
        word, so the last thing the model reads is a word and not a fragment
        it has to guess at. And the cut is *written into the text*, for the
        same reason `LocalFileReader` writes its own: a page silently
        shortened reads as a complete page, so a model told nothing about the
        missing half answers confidently from the half it got — a new way to
        be wrong about the world, which is the defect this issue is about.
        """
        if len(text) <= self.max_chars:
            return text                      # untouched, to the byte
        head = text[: self.max_chars]
        boundary = max(head.rfind("\n"), head.rfind(" "))
        # Only honour a boundary that is actually near the end: a page with no
        # whitespace in its last 20% would otherwise lose real material to
        # tidiness, and a mid-word cut costs one word.
        if boundary > int(self.max_chars * 0.8):
            head = head[:boundary]
        head = head.rstrip()
        return head + TRUNCATION_NOTE.format(kept=len(head))
