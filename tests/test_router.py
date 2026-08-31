"""Triage router: route selection + fail-open fallback to "plan"."""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

from agent_oo.core.capability import Capability, CapabilitySpec
from agent_oo.core.deps import Deps
from agent_oo.core.registry import CapabilityRegistry
from agent_oo.core.router import Router


class StubCap(Capability):
    def __init__(self, name: str):
        self.spec = CapabilitySpec(name=name, description=f"the {name} capability")

    def build(self):
        raise NotImplementedError


@pytest.fixture
def registry():
    return CapabilityRegistry([StubCap("alpha"), StubCap("beta")])


def make_router(registry, response: str) -> Router:
    return Router(Deps(llm=FakeLLM({"triage": response})), registry)


def test_prompt_lists_routes_and_capabilities(registry):
    prompt = make_router(registry, "").system_prompt()
    assert '"plan"' in prompt
    assert '"direct"' in prompt
    assert "- alpha: the alpha capability" in prompt


async def test_direct_route_selected(registry):
    router = make_router(registry, json.dumps({"route": "direct", "rationale": "meta"}))
    assert await router.run({"query": "what can you do?"}) == {"route": "direct"}


async def test_plan_route_selected(registry):
    router = make_router(registry, json.dumps({"route": "plan"}))
    assert await router.run({"query": "find the docs"}) == {"route": "plan"}


async def test_unknown_route_falls_back_to_plan(registry):
    router = make_router(registry, json.dumps({"route": "teleport"}))
    assert await router.run({"query": "q"}) == {"route": "plan"}


async def test_malformed_json_falls_back_to_plan(registry):
    router = make_router(registry, "not json at all")
    assert await router.run({"query": "q"}) == {"route": "plan"}


async def test_llm_exception_falls_back_to_plan(registry):
    class ExplodingLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            raise RuntimeError("llm down")

    router = Router(Deps(llm=ExplodingLLM()), registry)
    assert await router.run({"query": "q"}) == {"route": "plan"}


def test_conditional_edge_reads_state(registry):
    router = make_router(registry, "")
    assert router.route({"route": "direct"}) == "direct"
    assert router.route({}) == "plan"  # missing decision → safe default


async def test_custom_route_accepted(registry):
    routes = {"plan": "p", "direct": "d", "handoff": "give it to a human"}
    router = Router(
        Deps(llm=FakeLLM({"triage": json.dumps({"route": "handoff"})})),
        registry,
        routes=routes,
    )
    assert "handoff" in router.system_prompt()
    assert await router.run({"query": "q"}) == {"route": "handoff"}
