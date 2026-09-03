"""An agent is a capability pack + a profile — nothing else.

These tests protect the property that makes `agents/` worth having: a new
agent is composed by the SAME build_app, chat layer, CLI and API as every
other one. If composing an agent ever needs new shared code, the boundary
has been broken.
"""
from __future__ import annotations

import pytest
from conftest import FakeLLM, plan_json

from jobsmith.agents import AGENTS, agent_names, get_agent
from jobsmith.agents.base import AgentDefinition
from jobsmith.app.agent import build_app
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.profile import AgentProfile


def test_registry_lists_both_shipped_agents():
    assert agent_names() == ["banking", "default"]
    assert get_agent(None) is AGENTS["default"]          # None = the default agent
    with pytest.raises(KeyError, match="unknown agent 'nope'"):
        get_agent("nope")


@pytest.mark.parametrize("name", ["default", "banking"])
async def test_every_shipped_agent_composes_through_the_same_build_app(name):
    app = await build_app(agent=name, llm=object(), chat_model=object())
    try:
        assert app.agent_name == name
        capability_nodes = {n for n in app.manager.graph.nodes if n.startswith("cap_")}
        assert capability_nodes, "an agent must contribute capabilities to the graph"
        assert app.new_session().build() is not None     # chat works for any agent
    finally:
        await app.aclose()


def test_the_two_agents_contribute_different_capabilities():
    llm = object()
    names = {n: {c.spec.name for c in get_agent(n).capabilities(llm)}
             for n in ("default", "banking")}
    assert names["default"] == {"research", "analysis", "critique"}
    assert names["banking"] == {"search", "vision", "refs"}
    assert not names["default"] & names["banking"]


async def test_a_third_party_agent_needs_no_shared_code(store, checkpointer, tmp_path):
    """The whole contract: define capabilities + a profile, and it runs."""

    class Echo(Capability):
        spec = CapabilitySpec(name="echo", description="echoes the request")

        def __init__(self, llm):
            self.llm = llm

        async def work(self, state: CapabilityBaseState) -> dict:
            return self._emit_success({"echo": state["query"]})

        def render_context(self, result):
            return result["data"]["echo"]

        def build(self):
            from langgraph.constants import END
            g = self.state_graph(CapabilityBaseState)
            g.add_node("work", self.work)
            g.set_entry_point("work")
            g.add_edge("work", END)
            return g.compile()

    mine = AgentDefinition(
        name="mine",
        description="a third-party agent",
        capabilities=lambda llm: [Echo(llm)],
        profile=AgentProfile(),
        chat_prompt="You are terse.",
    )
    AGENTS[mine.name] = mine
    try:
        llm = FakeLLM({"planner": plan_json("echo")},
                      default="A sufficiently long final answer for this run.")
        app = await build_app(agent="mine", llm=llm, chat_model=object(),
                              reports_dir=str(tmp_path))
        try:
            job = await app.manager.create_job("hello there")
            done = await app.manager.run_job(job.job_id)
            assert done.status.value == "done"
            assert done.results["echo"]["data"]["echo"] == "hello there"
            assert app.new_session().system_prompt == "You are terse."
        finally:
            await app.aclose()
    finally:
        del AGENTS[mine.name]
