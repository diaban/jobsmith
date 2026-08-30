"""Behavioral baseline: run the CURRENT banking graph end-to-end with fakes.

This locks in the pre-refactor behavior before any logic changes.
It will be rewritten against the new API in the cutover phase.
"""
from __future__ import annotations

from conftest import FakeLLM, FakeS3, FakeSearch, plan_json

from agent_oo.deps import Deps
from agent_oo.graph import build_agent


def make_deps(llm: FakeLLM) -> Deps:
    return Deps(search=FakeSearch(), llm=llm, s3=FakeS3())


async def test_happy_path_search_only(checkpointer, store):
    llm = FakeLLM({
        "planner": plan_json("search"),
        "Rewrite the banker": '{"query": "rewritten query"}',
        "banking assistant": "Answer citing [doc_1] with enough length to pass validation.",
    })
    graph = build_agent(make_deps(llm), checkpointer, store)
    out = await graph.ainvoke(
        {"query": "what is the exposure to X?", "thread_id": "t1"},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert out["terminal_kind"] == "answer"
    assert "[doc_1]" in out["final_answer"]
    assert out["completed_subgraphs"] == ["search"]


async def test_empty_query_rejected(checkpointer, store):
    graph = build_agent(make_deps(FakeLLM()), checkpointer, store)
    out = await graph.ainvoke(
        {"query": "   ", "thread_id": "t2"},
        config={"configurable": {"thread_id": "t2"}},
    )
    assert out["terminal_kind"] == "user_error"
    assert out["user_error_message"] == "Votre requête est vide."


async def test_planner_garbage_routes_to_user_error(checkpointer, store):
    llm = FakeLLM({"planner": "not json at all"})
    graph = build_agent(make_deps(llm), checkpointer, store)
    out = await graph.ainvoke(
        {"query": "hello", "thread_id": "t3"},
        config={"configurable": {"thread_id": "t3"}},
    )
    # No partial results -> user_error terminal
    assert out["terminal_kind"] == "user_error"
