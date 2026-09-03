"""Grounding: the `documents` capability and the LocalFiles adapter.

The capability is tested against a fake `DocumentSource`, because it must not
know what is behind the port — if a test here needs the filesystem, the
capability has learned something it should not know.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, plan_json

from jobsmith.agents.default.documents import DocumentsCapability
from jobsmith.agents.default.sources import Document, LocalFiles, _chunks
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry


class FakeSource:
    """A DocumentSource that records what it was asked."""

    def __init__(self, hits: dict[str, list[Document]] | None = None, fail_on: str = ""):
        self.hits = hits or {}
        self.fail_on = fail_on
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 8) -> list[Document]:
        self.queries.append(query)
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("source is down")
        return self.hits.get(query, [])[:limit]


def QueryLLM(queries: list[str], script: dict | None = None, **kw) -> FakeLLM:
    """FakeLLM scripted to answer the query-planning prompt (keyed on its
    system prompt, like every other scripted response)."""
    return FakeLLM({"search queries": json.dumps({"queries": queries}), **(script or {})}, **kw)


def doc(doc_id: str, text: str, score: float = 1.0) -> Document:
    return Document(id=doc_id, text=text, title=doc_id, source=f"/docs/{doc_id}", score=score)


async def run(capability, query="what is the exposure?"):
    graph = capability.build()
    return await graph.ainvoke({"query": query, "inputs": {}})


# ---------------------------------------------------------------- capability


async def test_retrieves_deduplicates_and_ranks_across_queries():
    source = FakeSource({
        "exposure": [doc("a#0", "Exposure is 12M", score=3.0), doc("b#0", "Other", 1.0)],
        "credit":   [doc("a#0", "Exposure is 12M", score=9.0), doc("c#0", "Third", 2.0)],
    })
    capability = DocumentsCapability(QueryLLM(["exposure", "credit"]), source)
    out = await run(capability)

    result = out["results"]["documents"]
    assert result["ok"] is True
    assert source.queries == ["exposure", "credit"]
    found = result["data"]["documents"]
    # a#0 appears in both queries: kept once, with its BEST score, ranked first
    assert [d["id"] for d in found] == ["a#0", "c#0", "b#0"]
    assert result["meta"]["document_count"] == 3


async def test_one_failing_query_does_not_lose_the_others():
    source = FakeSource({"good": [doc("a#0", "kept")]}, fail_on="bad")
    capability = DocumentsCapability(QueryLLM(["bad", "good"]), source)
    out = await run(capability)

    result = out["results"]["documents"]
    assert result["ok"] is True
    assert [d["id"] for d in result["data"]["documents"]] == ["a#0"]


async def test_unusable_llm_reply_degrades_to_the_raw_request():
    """Query planning is a convenience, not a dependency."""
    class BrokenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            return "not json at all"

    source = FakeSource({"what is the exposure?": [doc("a#0", "found anyway")]})
    capability = DocumentsCapability(BrokenLLM(), source)
    out = await run(capability)

    assert source.queries == ["what is the exposure?"]
    assert out["results"]["documents"]["ok"] is True


async def test_finding_nothing_is_a_recoverable_failure():
    capability = DocumentsCapability(QueryLLM(["nope"]), FakeSource())
    out = await run(capability)

    result = out["results"]["documents"]
    assert result["ok"] is False
    assert "no document matched" in result["error"]


def test_rendering_targets_two_different_readers():
    capability = DocumentsCapability(QueryLLM([]), FakeSource())
    result = {"ok": True, "data": {"documents": [
        {"id": "a#0", "title": "notes.md", "source": "/docs/notes.md", "text": "Body text."}
    ]}}
    # the model gets quotable material...
    context = capability.render_context(result)
    assert "[a#0]" in context and "Body text." in context
    # ...the human gets provenance, not a copy of the corpus
    report = capability.render_report(result)
    assert "/docs/notes.md" in report and "Body text." not in report


async def test_plans_through_the_full_graph(checkpointer):
    """The planner can put `documents` in a DAG like any other capability."""
    source = FakeSource({"q": [doc("a#0", "Grounded material.")]})
    llm = QueryLLM(["q"], {"planner": plan_json("documents")},
                   default="A sufficiently long grounded answer for the test.")
    graph = build_agent(Deps(llm=llm),
                        CapabilityRegistry([DocumentsCapability(llm, source)]),
                        checkpointer=checkpointer)
    out = await graph.ainvoke({"query": "ground me", "inputs": {}},
                              config={"configurable": {"thread_id": "t1"}})
    assert out["terminal_kind"] == "answer"
    assert out["results"]["documents"]["ok"] is True


# ---------------------------------------------------------------- adapter


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_local_files_ranks_by_term_overlap(tmp_path):
    write(tmp_path, "match.md", "Credit exposure to Acme is twelve million euros.")
    write(tmp_path, "partial.md", "Acme is a company.")
    write(tmp_path, "unrelated.md", "Cooking recipes for winter soup.")

    hits = await LocalFiles(tmp_path).search("credit exposure acme")
    ids = [h.id.split("#")[0] for h in hits]
    assert ids[0] == "match.md"          # covers all three terms
    assert "partial.md" in ids           # covers one
    assert "unrelated.md" not in ids     # covers none: not returned at all


async def test_local_files_skips_noise_directories_and_binaries(tmp_path):
    write(tmp_path, "keep.md", "alpha beta gamma")
    write(tmp_path, ".venv/lib/vendored.md", "alpha beta gamma")
    write(tmp_path, "node_modules/pkg/readme.md", "alpha beta gamma")
    write(tmp_path, "image.png", "alpha beta gamma")

    hits = await LocalFiles(tmp_path).search("alpha beta gamma")
    assert [h.title for h in hits] == ["keep.md"]


async def test_local_files_returns_nothing_for_an_empty_query(tmp_path):
    write(tmp_path, "a.md", "content")
    assert await LocalFiles(tmp_path).search("  ") == []


async def test_local_files_ids_are_stable_and_point_at_a_real_file(tmp_path):
    write(tmp_path, "sub/notes.md", "alpha alpha alpha")
    (hit,) = await LocalFiles(tmp_path).search("alpha")
    assert hit.id == "sub/notes.md#0"        # relative, quotable, chunk-indexed
    assert hit.source.endswith("sub/notes.md")
    assert hit.title == "sub/notes.md"


@pytest.mark.parametrize("text,expected", [
    ("one paragraph", 1),
    ("first\n\nsecond", 1),                      # glued: both fit the target
    ("x" * 900 + "\n\n" + "y" * 900, 2),         # split: together they exceed it
    ("   \n\n   ", 0),                           # nothing but blanks
])
def test_chunking_splits_on_paragraphs_up_to_a_target(text, expected):
    assert len(_chunks(text)) == expected
