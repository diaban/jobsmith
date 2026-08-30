"""End-to-end framework test with pure stub capabilities (no banking)."""
from __future__ import annotations

from langgraph.constants import END
from conftest import FakeLLM, plan_json

from agent_oo.core.builder import AgentBuilder, build_agent
from agent_oo.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from agent_oo.core.deps import Deps
from agent_oo.core.registry import CapabilityRegistry
from agent_oo.core.state import CapabilityResult


class EchoCapability(Capability):
    """Single-node capability that echoes a configured payload."""

    def __init__(self, name: str, payload: str, *, fail: bool = False):
        self.spec = CapabilitySpec(name=name, description=f"echoes {payload}")
        self.payload = payload
        self.fail = fail

    async def work(self, state: CapabilityBaseState) -> dict:
        if self.fail:
            return self._emit_failure(f"{self.spec.name} broke")
        return self._emit_success({"echo": self.payload})

    def render_context(self, result: CapabilityResult) -> str | None:
        return f"# {self.spec.name}\n{result['data']['echo']}"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


async def test_two_capabilities_to_final_answer(checkpointer, store):
    llm = FakeLLM(
        {"planner": plan_json("first", "second", deps={"second": ["first"]})},
        default="A sufficiently long final answer built from context.",
    )
    registry = CapabilityRegistry([
        EchoCapability("first", "hello"),
        EchoCapability("second", "world"),
    ])
    builder = AgentBuilder(Deps(llm=llm), registry, checkpointer=checkpointer)
    graph = builder.build()
    out = await graph.ainvoke(
        {"query": "do the thing", "job_id": "e1"},
        config={"configurable": {"thread_id": "e1"}},
    )
    assert out["terminal_kind"] == "answer"
    assert set(out["results"]) == {"first", "second"}
    # merged context is in plan order and uses render_context
    gen_call = next(c for c in llm.calls if "Context:" in c["messages"][1]["content"])
    ctx = gen_call["messages"][1]["content"]
    assert ctx.index("# first") < ctx.index("# second")


async def test_all_capabilities_fail_routes_to_user_error(checkpointer, store):
    """Recoverable failures everywhere → merge yields no context, generation
    still answers; but if generation ALSO fails, no ok results → user_error."""

    class ExplodingLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            system = self._system_of(messages)
            if "planner" in system:
                return plan_json("only")
            raise RuntimeError("llm down")

    registry = CapabilityRegistry([EchoCapability("only", "x", fail=True)])
    graph = build_agent(Deps(llm=ExplodingLLM()), registry, checkpointer=checkpointer)
    out = await graph.ainvoke(
        {"query": "q", "job_id": "e2"},
        config={"configurable": {"thread_id": "e2"}},
    )
    assert out["terminal_kind"] == "user_error"


async def test_partial_success_escalates(checkpointer, store):
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            system = self._system_of(messages)
            if "planner" in system:
                return plan_json("good", "bad")
            raise RuntimeError("llm down")

    registry = CapabilityRegistry([
        EchoCapability("good", "ok"),
        EchoCapability("bad", "x", fail=True),
    ])
    graph = build_agent(Deps(llm=ExplodingGenLLM()), registry, checkpointer=checkpointer)
    out = await graph.ainvoke(
        {"query": "q", "job_id": "e3"},
        config={"configurable": {"thread_id": "e3"}},
    )
    assert out["terminal_kind"] == "escalated"


async def test_refine_loop_recovers(checkpointer, store):
    llm = FakeLLM({
        "planner": plan_json("first"),
        "ONLY the provided": ["too short", "Now a sufficiently long refined answer indeed."],
        "failed validation": "Now a sufficiently long refined answer indeed.",
    })
    registry = CapabilityRegistry([EchoCapability("first", "hello")])
    graph = build_agent(Deps(llm=llm), registry, checkpointer=checkpointer)
    out = await graph.ainvoke(
        {"query": "q", "job_id": "e4"},
        config={"configurable": {"thread_id": "e4"}},
    )
    assert out["terminal_kind"] == "answer"
    assert out["refine_count"] == 1
    assert "refined answer" in out["final_answer"]


async def test_registry_frozen_after_build(checkpointer, store):
    import pytest

    registry = CapabilityRegistry([EchoCapability("first", "x")])
    build_agent(Deps(llm=FakeLLM()), registry, checkpointer=checkpointer)
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(EchoCapability("second", "y"))
