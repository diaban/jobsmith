"""The job runner drives any LangGraph graph and knows nothing about it: the
root's nodes as they finish, the facts published at any depth, and what the
run returned — exactly what `ainvoke` returns (→ docs/design/core-v1.md).
"""
from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from jobsmith.engine.facts import publish
from jobsmith.engine.runner import Fact, GraphRunner, NodeFinished, Output


class In(TypedDict):
    a: int
    b: int


class Out(TypedDict):
    total: int


class State(In, Out, total=False):
    scratch: str


def _inner(state: State) -> dict:
    publish("deep", state["a"])          # from inside a sub-graph
    return {"total": state["a"] + state["b"]}


def graph(*, fail: bool = False):
    sub = StateGraph(State)
    sub.add_node("inner", _inner)
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)

    def first(state: State) -> dict:
        publish("plan", ["sum"])
        if fail:
            raise RuntimeError("the graph blew up")
        return {"scratch": "kept inside"}

    g = StateGraph(State, input_schema=In, output_schema=Out)
    g.add_node("first", first)
    g.add_node("sum", sub.compile())
    g.add_edge(START, "first")
    g.add_edge("first", "sum")
    g.add_edge("sum", END)
    return g.compile(checkpointer=MemorySaver())


async def test_the_root_s_steps_the_facts_at_any_depth_and_what_ainvoke_returns():
    compiled = graph()
    updates = [u async for u in GraphRunner(compiled).stream("j1", {"a": 1, "b": 2})]

    assert updates == [Fact("plan", ["sum"]), NodeFinished("first"),
                       Fact("deep", 1), NodeFinished("sum"), Output({"total": 3})]
    assert await compiled.ainvoke({"a": 1, "b": 2}, {"configurable": {"thread_id": "j2"}}) \
        == {"total": 3}


async def test_a_run_that_raises_returns_nothing():
    runner = GraphRunner(graph(fail=True))
    seen = []
    with pytest.raises(RuntimeError, match="blew up"):
        async for update in runner.stream("j1", {"a": 1, "b": 2}):
            seen.append(update)
    assert seen == [Fact("plan", ["sum"])]
    assert await runner.pending("j1") == ("first",)      # left to resume
