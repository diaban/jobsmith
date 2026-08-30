"""Behavior parity: the banking example via the new framework reproduces the
pre-refactor behavior (see git history: tests/test_baseline_smoke.py).
"""
from __future__ import annotations

from conftest import FakeLLM, FakeS3, FakeSearch, plan_json

from agent_oo.core.builder import AgentBuilder
from agent_oo.core.deps import Deps
from agent_oo.core.registry import CapabilityRegistry
from agent_oo.examples.banking.capabilities.refs import RefsCapability
from agent_oo.examples.banking.capabilities.search import SearchCapability
from agent_oo.examples.banking.capabilities.vision import VisionCapability
from agent_oo.examples.banking.profile import BANKING_PROFILE


def banking_builder(llm: FakeLLM, checkpointer, store, *, search=None) -> AgentBuilder:
    search = search if search is not None else FakeSearch()
    registry = CapabilityRegistry([
        SearchCapability(llm, search),
        VisionCapability(llm, FakeS3({"img1": b"bytes"})),
        RefsCapability(search),
    ])
    return AgentBuilder(
        Deps(llm=llm), registry,
        profile=BANKING_PROFILE, checkpointer=checkpointer, store=store,
    )


async def test_happy_path_search_only(checkpointer, store):
    llm = FakeLLM({
        "planner": plan_json("search"),
        "Rewrite the banker": '{"query": "rewritten query"}',
        "banking assistant": "Answer citing [doc_1] with enough length to pass validation.",
    })
    graph = banking_builder(llm, checkpointer, store).build()
    out = await graph.ainvoke(
        {"query": "what is the exposure to X?", "job_id": "t1"},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert out["terminal_kind"] == "answer"
    assert "[doc_1]" in out["final_answer"]
    assert out["completed_capabilities"] == ["search"]
    assert out["results"]["search"]["ok"] is True


async def test_empty_query_rejected(checkpointer, store):
    graph = banking_builder(FakeLLM(), checkpointer, store).build()
    out = await graph.ainvoke(
        {"query": "   ", "job_id": "t2"},
        config={"configurable": {"thread_id": "t2"}},
    )
    assert out["terminal_kind"] == "user_error"
    assert out["user_error_message"] == "Votre requête est vide."


async def test_planner_garbage_routes_to_user_error(checkpointer, store):
    llm = FakeLLM({"planner": "not json at all"})
    graph = banking_builder(llm, checkpointer, store).build()
    out = await graph.ainvoke(
        {"query": "hello", "job_id": "t3"},
        config={"configurable": {"thread_id": "t3"}},
    )
    assert out["terminal_kind"] == "user_error"


async def test_vision_dropped_without_image_and_dangling_dep_pruned(checkpointer, store):
    """Regression for the fixed latent bug: refs depends_on [vision], no image
    provided → vision dropped AND the dangling dep pruned (old code raised)."""
    llm = FakeLLM({
        "planner": plan_json("vision", "refs", deps={"refs": ["vision"]}),
        "banking assistant": "Answer with refs (ref_1) long enough to pass validation.",
    })
    graph = banking_builder(llm, checkpointer, store).build()
    out = await graph.ainvoke(
        {"query": "show me past decks", "job_id": "t4"},
        config={"configurable": {"thread_id": "t4"}},
    )
    assert out["terminal_kind"] == "answer"
    assert out["completed_capabilities"] == ["refs"]
    assert "vision" not in out.get("results", {})


async def test_vision_runs_with_image_input(checkpointer, store):
    llm = FakeLLM({
        "planner": plan_json("vision"),
        "banking assistant": "The image shows quarterly revenue (per the analysis).",
    })
    graph = banking_builder(llm, checkpointer, store).build()
    out = await graph.ainvoke(
        {"query": "what's in this chart?", "job_id": "t5",
         "inputs": {"image_s3_keys": ["img1"]}},
        config={"configurable": {"thread_id": "t5"}},
    )
    assert out["terminal_kind"] == "answer"
    assert out["results"]["vision"]["ok"] is True
    assert out["results"]["vision"]["data"]["description"] == "a chart showing quarterly revenue"


async def test_search_failure_escalates_with_partial_refs(checkpointer, store):
    """Search hard-fails (recoverable) but refs succeeds; generation then fails
    unrecoverably → escalate (partial results exist)."""
    class FlakySearch(FakeSearch):
        async def search(self, query, *, top_k=10):
            self.calls.append(query)
            if "past_slides" in query:
                return [{"id": "ref_1", "summary": "old deck"}]
            raise RuntimeError("search down")

        async def search_cached(self, query, *, top_k=10):
            raise RuntimeError("cache down")

    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            system = self._system_of(messages)
            if "banking assistant" in system:
                raise RuntimeError("llm down")
            return await super().chat(messages, **kwargs)

    llm = ExplodingGenLLM({
        "planner": plan_json("search", "refs"),
        "Rewrite the banker": '{"query": "q"}',
    })
    search = FlakySearch()
    builder = banking_builder(llm, checkpointer, store, search=search)
    builder.search_cap = builder.registry.get("search")
    builder.search_cap.max_retries = 0  # keep the test fast
    graph = builder.build()
    out = await graph.ainvoke(
        {"query": "anything", "job_id": "t6"},
        config={"configurable": {"thread_id": "t6"}},
    )
    assert out["terminal_kind"] == "escalated"
    # escalation payload persisted for a human analyst
    stored = await store.aget(("escalations", "t6"), "t6")
    assert stored is not None
    assert stored.value["partial_results"]["refs"]["ok"] is True
