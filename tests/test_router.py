"""Triage router: route selection + fail-open fallback to "plan"."""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

from jobsmith.core.capability import Capability, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.router import Router


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


# ---------------- Empty registry: nothing to plan with (#38) ----------------

@pytest.fixture
def empty_registry():
    return CapabilityRegistry([])


async def test_empty_registry_routes_direct_without_an_llm_call(empty_registry):
    """The decision is structural: no capability can be orchestrated, so there
    is nothing for the planner to do whatever the message says. It must cost
    no LLM call at all — that is the whole point of deciding it here."""
    llm = FakeLLM({"triage": json.dumps({"route": "plan"})})
    router = Router(Deps(llm=llm), empty_registry)
    assert await router.run({"query": "compare two architectures in depth"}) == {
        "route": "direct"
    }
    assert llm.calls == []


async def test_empty_registry_beats_the_plan_fallback(empty_registry):
    """FALLBACK_ROUTE is "plan"; with nothing to plan with that is not failing
    open, it is failing into the wall the planner raises against."""
    class ExplodingLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            raise RuntimeError("llm down")

    router = Router(Deps(llm=ExplodingLLM()), empty_registry)
    assert await router.run({"query": "q"}) == {"route": "direct"}


def test_empty_registry_conditional_edge_defaults_to_direct(empty_registry):
    router = Router(Deps(llm=FakeLLM()), empty_registry)
    assert router.route({}) == "direct"          # no decision recorded
    assert router.route({"route": "direct"}) == "direct"


async def test_empty_registry_without_a_direct_route_keeps_the_old_fallback(
    empty_registry,
):
    """Degenerate composition: a route map with no "direct" target. There is
    nowhere better to send the message, so the plain fallback stands."""
    router = Router(
        Deps(llm=FakeLLM()), empty_registry, routes={"plan": "the only route"}
    )
    assert await router.run({"query": "q"}) == {"route": "plan"}
    assert router.route({}) == "plan"


def test_non_empty_registry_fallback_is_unchanged(registry):
    router = make_router(registry, "")
    assert router.route({}) == "plan"
