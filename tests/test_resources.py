"""An agent's external dependencies: who opens them, who closes them.

The agent knows WHAT to open (which backend, which collection); the
composition root owns the LIFETIME and the event loop. These tests pin both
halves, and the shape recommended when several capabilities need the same
backend differently.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.agents import AGENTS
from jobsmith.agents.base import AgentContext, AgentDefinition
from jobsmith.app.agent import build_app
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.profile import AgentProfile


class FakePool:
    """Stands in for a connection pool: one per app, shared by adapters."""

    def __init__(self, log: list[str]):
        self.log = log
        self.queries: list[tuple[str, str]] = []

    async def query(self, collection: str, term: str) -> list[str]:
        self.queries.append((collection, term))
        return [f"{collection}:{term}"]


@asynccontextmanager
async def open_pool(log: list[str]):
    log.append("open")
    pool = FakePool(log)
    try:
        yield pool
    finally:
        log.append("close")


# -- two ports, two adapters, ONE pool --------------------------------------

class TextIndex:                       # adapter for the text capability's port
    def __init__(self, pool: FakePool):
        self.pool = pool

    async def search_text(self, query: str) -> list[str]:
        return await self.pool.query("passages", query)


class VisualIndex:                     # adapter for the visual capability's port
    def __init__(self, pool: FakePool):
        self.pool = pool

    async def search_similar(self, query: str) -> list[str]:
        return await self.pool.query("frames", query)


@dataclass(frozen=True)
class Resources:
    pool: FakePool
    text: TextIndex
    visual: VisualIndex


class IndexCapability(Capability):
    """Depends on ONE narrow port, never on the pool."""

    def __init__(self, name: str, index, method: str):
        self.spec = CapabilitySpec(name=name, description=f"{name} lookup")
        self.index = index
        self.method = method

    async def work(self, state: CapabilityBaseState) -> dict:
        hits = await getattr(self.index, self.method)(state["query"])
        return self._emit_success({"hits": hits})

    def render_context(self, result):
        return ", ".join(result["data"]["hits"])

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


def shared_backend_agent(log: list[str]) -> AgentDefinition:
    async def open_resources(stack):
        pool = await stack.enter_async_context(open_pool(log))
        return Resources(pool=pool, text=TextIndex(pool), visual=VisualIndex(pool))

    def capabilities(ctx: AgentContext):
        res: Resources = ctx.resources
        return [
            IndexCapability("text_lookup", res.text, "search_text"),
            IndexCapability("visual_lookup", res.visual, "search_similar"),
        ]

    return AgentDefinition(
        name="shared_backend",
        description="two capabilities over one pool",
        capabilities=capabilities,
        profile=AgentProfile(),
        open_resources=open_resources,
    )


@asynccontextmanager
async def registered(definition: AgentDefinition):
    AGENTS[definition.name] = definition
    try:
        yield definition
    finally:
        del AGENTS[definition.name]


async def test_two_capabilities_share_one_connection_with_an_adapter_each(tmp_path):
    log: list[str] = []
    async with registered(shared_backend_agent(log)):
        llm = FakeLLM({"planner": plan_json("text_lookup", "visual_lookup")},
                      default="A sufficiently long answer for this run.")
        app = await build_app(agent="shared_backend", llm=llm, chat_model=object(),
                              reports_dir=str(tmp_path))
        try:
            job = await app.manager.create_job("acme")
            done = await app.manager.run_job(job.job_id)
            assert done.status.value == "done"
            # one pool opened once, both capabilities went through it...
            assert log == ["open"]
            assert app.resources.pool.queries == [("passages", "acme"), ("frames", "acme")]
            # ...each through its own adapter and its own collection
            assert done.results["text_lookup"]["data"]["hits"] == ["passages:acme"]
            assert done.results["visual_lookup"]["data"]["hits"] == ["frames:acme"]
        finally:
            await app.aclose()
    assert log == ["open", "close"]        # closed exactly once, by the app


async def test_resources_are_released_when_startup_fails(tmp_path):
    """A crash after opening must not leak the pool."""
    log: list[str] = []
    definition = shared_backend_agent(log)

    def exploding(ctx):
        raise RuntimeError("bad capability pack")

    broken = AgentDefinition(
        name="broken", description="x", capabilities=exploding,
        profile=AgentProfile(), open_resources=definition.open_resources,
    )
    async with registered(broken):
        with pytest.raises(RuntimeError, match="bad capability pack"):
            await build_app(agent="broken", llm=object(), chat_model=object())
    assert log == ["open", "close"]


async def test_an_agent_that_declares_no_resources_gets_none(tmp_path):
    bare = AgentDefinition(
        name="bare", description="declares no open_resources",
        capabilities=lambda ctx: [], profile=AgentProfile(),
    )
    async with registered(bare):
        app = await build_app(agent="bare", llm=object(), chat_model=object())
        try:
            assert app.resources is None
        finally:
            await app.aclose()


async def test_the_default_agent_has_no_document_source_unless_configured(monkeypatch):
    """`documents` must stay out of the registry when nothing backs it —
    otherwise the planner plans a step that can only fail."""
    monkeypatch.delenv("JOBSMITH_DOCS", raising=False)
    monkeypatch.setattr("sys.argv", ["pytest"])
    app = await build_app(agent="default", llm=object(), chat_model=object())
    try:
        assert app.resources.documents is None
        assert not [n for n in app.manager.graph.nodes if n == "cap_documents"]
    finally:
        await app.aclose()
