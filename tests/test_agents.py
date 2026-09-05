"""An agent is a capability pack + a profile — nothing else.

These tests protect the property that makes `agents/` worth having: a new
agent is composed by the SAME build_app, chat layer, CLI and API as every
other one. If composing an agent ever needs new shared code, the boundary
has been broken.
"""
from __future__ import annotations

from contextlib import AsyncExitStack

import pytest
from conftest import FakeLLM, plan_json

from jobsmith.agents import AGENTS, agent_names, get_agent
from jobsmith.agents.base import AgentContext, AgentDefinition, open_agent_resources
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


async def test_the_two_agents_contribute_different_capabilities():
    async with AsyncExitStack() as stack:
        names = {}
        for n in ("default", "banking"):
            definition = get_agent(n)
            resources = await open_agent_resources(definition, stack)
            # FakeLLM chats *and* sees, which is what the banking pack asks for.
            ctx = AgentContext(llm=FakeLLM(), resources=resources)
            names[n] = {c.spec.name for c in definition.capabilities(ctx)}
    assert names["default"] == {"research", "analysis", "critique"}
    assert names["banking"] == {"search", "vision", "refs"}
    assert not names["default"] & names["banking"]


async def test_banking_drops_vision_when_the_llm_cannot_see():
    """A capability nothing can serve stays out of the registry.

    `LLMClient` promises `chat` and nothing more; `vision` is the banking
    agent's own port. Handed a chat-only adapter, the pack must leave the
    vision capability unregistered rather than let the planner plan a step
    that can only raise AttributeError halfway through a job.
    """

    class ChatOnly:
        async def chat(self, messages, **kwargs) -> str:
            return "..."

    async with AsyncExitStack() as stack:
        definition = get_agent("banking")
        resources = await open_agent_resources(definition, stack)
        ctx = AgentContext(llm=ChatOnly(), resources=resources)
        names = {c.spec.name for c in definition.capabilities(ctx)}
    assert names == {"search", "refs"}


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
        capabilities=lambda ctx: [Echo(ctx.llm)],
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
