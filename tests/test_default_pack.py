"""Default capability pack: research → analysis → critique (LLM-only)."""
from __future__ import annotations

from conftest import FakeLLM, plan_json

from jobsmith.agents.default import default_capabilities
from jobsmith.agents.default.research import ResearchCapability
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry

PACK_SCRIPT = {
    "key aspects": '{"aspects": ["history", "impact"]}',
    "research notes": "Notes: the history is long; the impact is broad.",
    "You are an analyst": "Findings: impact outweighs history.",
    "critical reviewer": "Gap: no numbers back the impact claim.",
}


async def test_research_decomposes_and_emits_notes():
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    result = out["results"]["research"]
    assert result["ok"] is True
    assert result["data"]["aspects"] == ["history", "impact"]
    assert "history is long" in result["data"]["notes"]
    assert result["meta"]["aspect_count"] == 2


async def test_research_lenient_on_bad_decompose_json():
    llm = FakeLLM({**PACK_SCRIPT, "key aspects": "not json"})
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    assert out["results"]["research"]["data"]["aspects"] == ["study X"]  # degraded, not failed


async def test_pack_chain_end_to_end(checkpointer):
    llm = FakeLLM({
        "planner": plan_json(
            "research", "analysis", "critique",
            deps={"analysis": ["research"], "critique": ["analysis"]},
        ),
        **PACK_SCRIPT,
        "ONLY the provided": "A sufficiently long final answer built from the pack context.",
    })
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry(default_capabilities(llm)), checkpointer=checkpointer
    )
    out = await graph.ainvoke(
        {"query": "study X in depth", "job_id": "p1"},
        config={"configurable": {"thread_id": "p1"}},
    )
    assert out["terminal_kind"] == "answer"
    assert all(out["results"][name]["ok"] for name in ("research", "analysis", "critique"))

    # analysis actually consumed the research notes
    analyst_call = next(
        c for c in llm.calls if "You are an analyst" in c["messages"][0]["content"]
    )
    assert "history is long" in analyst_call["messages"][1]["content"]

    # merged context follows plan order with each capability's heading
    ctx = out["merged_context"]
    assert ctx.index("# Research notes") < ctx.index("# Analysis") < ctx.index("# Critique")


async def test_pack_degrades_when_one_step_fails(checkpointer):
    class FailingAnalystLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "You are an analyst" in self._system_of(messages):
                raise RuntimeError("llm down")
            return await super().chat(messages, **kwargs)

    llm = FailingAnalystLLM({
        "planner": plan_json("research", "analysis", deps={"analysis": ["research"]}),
        **PACK_SCRIPT,
        "ONLY the provided": "A sufficiently long final answer from research alone.",
    })
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry(default_capabilities(llm)), checkpointer=checkpointer
    )
    out = await graph.ainvoke(
        {"query": "study X", "job_id": "p2"},
        config={"configurable": {"thread_id": "p2"}},
    )
    assert out["results"]["analysis"]["ok"] is False       # recoverable failure
    assert out["terminal_kind"] == "answer"                # run still completes
    assert "# Analysis" not in out["merged_context"]       # failed step renders nothing
