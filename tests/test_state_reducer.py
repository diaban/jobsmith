"""merge_results reducer: union semantics + real parallel Send fan-in proof."""
from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from jobsmith.core.state import AgentState, merge_results


def test_union_semantics():
    assert merge_results(None, None) == {}
    assert merge_results({"a": {"ok": True}}, None) == {"a": {"ok": True}}
    assert merge_results(None, {"b": {"ok": False}}) == {"b": {"ok": False}}
    merged = merge_results({"a": {"ok": True}}, {"b": {"ok": False}})
    assert set(merged) == {"a", "b"}
    # right wins per key (documented; disjointness is enforced upstream)
    assert merge_results({"a": {"ok": True}}, {"a": {"ok": False}}) == {"a": {"ok": False}}


async def test_parallel_send_fan_in():
    """Two branches fanned out via Send both land their key in `results`."""

    async def writer_a(state: AgentState) -> dict:
        return {"results": {"cap_a": {"ok": True, "data": {"n": 1}}},
                "completed_capabilities": ["cap_a"]}

    async def writer_b(state: AgentState) -> dict:
        return {"results": {"cap_b": {"ok": True, "data": {"n": 2}}},
                "completed_capabilities": ["cap_b"]}

    def fan_out(state: AgentState):
        return [Send("writer_a", state), Send("writer_b", state)]

    g = StateGraph(AgentState)
    g.add_node("writer_a", writer_a)
    g.add_node("writer_b", writer_b)
    g.add_conditional_edges(START, fan_out, {"writer_a": "writer_a", "writer_b": "writer_b"})
    g.add_edge("writer_a", END)
    g.add_edge("writer_b", END)
    out = await g.compile().ainvoke({"query": "q"})

    assert set(out["results"]) == {"cap_a", "cap_b"}
    assert sorted(out["completed_capabilities"]) == ["cap_a", "cap_b"]
