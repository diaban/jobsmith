"""Web search: the second adapter behind the DocumentSource port.

The point being protected is that gaining the web required no change to the
capability that consumes it — so these tests exercise the adapter's mapping
and the wiring, not retrieval logic that was already covered.
"""
from __future__ import annotations

from contextlib import AsyncExitStack

import pytest

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import DefaultResources, default_capabilities, open_default_resources
from jobsmith.agents.default.documents import DocumentsCapability, WebSearchCapability
from jobsmith.agents.default.web import TavilySource


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class FakeHTTP:
    """Records the request instead of making it."""

    def __init__(self, payload=None, status=200):
        self.payload = payload if payload is not None else {"results": []}
        self.status = status
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(self.payload, self.status)

    async def aclose(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()
        return False


HIT = {
    "title": "Hexagonal architecture",
    "url": "https://example.org/hex",
    "content": "Ports and adapters keep the domain independent of I/O.",
    "score": 0.91,
}


# ---------------------------------------------------------------- the adapter


async def test_maps_a_result_onto_the_shared_document_shape():
    http = FakeHTTP({"results": [HIT]})
    (doc,) = await TavilySource("tvly-k", http).search("hexagonal architecture")

    assert doc.id == "https://example.org/hex"      # the URL IS the citation
    assert doc.source == doc.id                     # openable by a human
    assert doc.title == "Hexagonal architecture"
    assert doc.text.startswith("Ports and adapters")
    assert doc.score == pytest.approx(0.91)


async def test_sends_bearer_auth_and_a_bounded_request():
    http = FakeHTTP()
    await TavilySource("tvly-k", http, topic="news").search("q", limit=99)

    (call,) = http.calls
    assert call["url"] == "https://api.tavily.com/search"
    assert call["headers"]["Authorization"] == "Bearer tvly-k"
    assert call["json"]["query"] == "q"
    assert call["json"]["max_results"] == 20        # clamped to the API ceiling
    assert call["json"]["topic"] == "news"
    assert call["timeout"] == 20.0


async def test_results_without_an_excerpt_or_a_url_are_dropped():
    """A title with no text grounds nothing — keeping it would be noise the
    generator is invited to cite."""
    http = FakeHTTP({"results": [
        HIT,
        {"title": "no text", "url": "https://example.org/empty", "content": "  "},
        {"title": "no url", "url": "", "content": "orphan"},
    ]})
    found = await TavilySource("k", http).search("q")
    assert [d.id for d in found] == ["https://example.org/hex"]


async def test_an_empty_query_never_reaches_the_network():
    http = FakeHTTP()
    assert await TavilySource("k", http).search("   ") == []
    assert http.calls == []


async def test_http_errors_are_raised_not_swallowed():
    """The capability isolates one failing query from the others; an empty
    list here would instead read as 'the web knows nothing about this'."""
    http = FakeHTTP(status=500)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await TavilySource("k", http).search("q")


# ---------------------------------------------------------------- the capability


def test_web_search_is_a_distinct_step_the_planner_can_choose():
    assert WebSearchCapability.spec.name == "web_search"
    assert WebSearchCapability.spec.name != DocumentsCapability.spec.name
    # the description must let the planner tell the two apart
    description = WebSearchCapability.spec.description
    assert "web" in description and "documents" in description


def test_web_results_are_reported_as_links():
    capability = WebSearchCapability(object(), object())
    report = capability.render_report({"ok": True, "data": {"documents": [
        {"id": "https://example.org/hex", "title": "Hex", "source": "https://example.org/hex",
         "text": "body"}
    ]}})
    assert report == "- [Hex](https://example.org/hex)"
    assert "body" not in report          # provenance, not a copy of the corpus


# ---------------------------------------------------------------- the wiring


async def test_no_key_means_no_web_capability_at_all(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("JOBSMITH_DOCS", raising=False)
    monkeypatch.setattr("sys.argv", ["pytest"])
    async with AsyncExitStack() as stack:
        resources = await open_default_resources(stack)
    assert resources.web is None
    names = {c.spec.name for c in default_capabilities(AgentContext(object(), resources))}
    assert "web_search" not in names


async def test_the_http_client_is_closed_with_the_app(monkeypatch):
    """The first adapter that genuinely holds a handle: it must be entered on
    the app's stack, not created and forgotten."""
    http = FakeHTTP()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-k")
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: http)

    async with AsyncExitStack() as stack:
        resources = await open_default_resources(stack)
        assert isinstance(resources.web, TavilySource)
        assert http.closed is False
    assert http.closed is True           # released when the app closed


def test_both_sources_feed_the_same_capability_contract():
    """Local files and the web are one port: the capability cannot tell them
    apart, which is what made this feature an adapter rather than a rewrite."""
    resources = DefaultResources(documents=object(), web=object())
    capabilities = default_capabilities(AgentContext(object(), resources))
    by_name = {c.spec.name: c for c in capabilities}
    assert isinstance(by_name["documents"], DocumentsCapability)
    assert isinstance(by_name["web_search"], DocumentsCapability)   # same machinery
    assert by_name["documents"].__class__ is not by_name["web_search"].__class__
