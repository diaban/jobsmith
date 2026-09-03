"""Registry-driven planner: prompt rendering + plan validation."""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, plan_json

from jobsmith.core.capability import Capability, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.planner import Planner
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.state import CONVERSATION_INPUT_KEY


class StubCap(Capability):
    def __init__(self, name: str, requires_inputs: tuple[str, ...] = ()):
        self.spec = CapabilitySpec(
            name=name,
            description=f"the {name} capability",
            requires_inputs=requires_inputs,
            output_schema={"type": "object"},
        )

    def build(self):
        raise NotImplementedError


@pytest.fixture
def registry():
    return CapabilityRegistry([
        StubCap("alpha"),
        StubCap("beta"),
        StubCap("gamma", requires_inputs=("attachment",)),
    ])


def make_planner(registry, response: str = "") -> Planner:
    return Planner(Deps(llm=FakeLLM({"planner": response})), registry)


def test_prompt_rendered_from_registry(registry):
    prompt = make_planner(registry).system_prompt()
    assert '"alpha": the alpha capability' in prompt
    assert '"beta": the beta capability' in prompt
    assert "only if these inputs are provided: attachment" in prompt


async def test_valid_plan_accepted(registry):
    planner = make_planner(registry, plan_json("alpha", "beta", deps={"beta": ["alpha"]}))
    out = await planner.run({"query": "q"})
    assert out["plan"]["steps"] == [
        {"capability": "alpha", "depends_on": []},
        {"capability": "beta", "depends_on": ["alpha"]},
    ]


async def test_unknown_capability_rejected(registry):
    planner = make_planner(registry, plan_json("alpha", "nope"))
    out = await planner.run({"query": "q"})
    assert out["errors"][0]["kind"] == "planner_fail"
    assert not out["errors"][0]["recoverable"]
    assert "unknown capability" in out["errors"][0]["detail"]


async def test_cycle_rejected(registry):
    planner = make_planner(
        registry, plan_json("alpha", "beta", deps={"alpha": ["beta"], "beta": ["alpha"]})
    )
    out = await planner.run({"query": "q"})
    assert "cycle" in out["errors"][0]["detail"]


async def test_duplicate_rejected(registry):
    raw = json.dumps({"steps": [
        {"capability": "alpha", "depends_on": []},
        {"capability": "alpha", "depends_on": []},
    ]})
    out = await make_planner(registry, raw).run({"query": "q"})
    assert "duplicate" in out["errors"][0]["detail"]


async def test_inapplicable_dropped_and_dangling_deps_pruned(registry):
    """gamma requires 'attachment' input; absent → gamma dropped and beta's
    dep on gamma pruned instead of raising (the fixed latent bug)."""
    planner = make_planner(registry, plan_json("gamma", "beta", deps={"beta": ["gamma"]}))
    out = await planner.run({"query": "q"})  # no inputs
    assert out["plan"]["steps"] == [{"capability": "beta", "depends_on": []}]


async def test_applicable_kept_when_input_present(registry):
    planner = make_planner(registry, plan_json("gamma"))
    out = await planner.run({"query": "q", "inputs": {"attachment": "x"}})
    assert out["plan"]["steps"] == [{"capability": "gamma", "depends_on": []}]


async def test_all_steps_inapplicable_is_planner_fail(registry):
    planner = make_planner(registry, plan_json("gamma"))
    out = await planner.run({"query": "q"})
    assert "empty after validation" in out["errors"][0]["detail"]


async def test_malformed_json_is_unrecoverable(registry):
    out = await make_planner(registry, "not json").run({"query": "q"})
    assert out["errors"][0]["kind"] == "planner_fail"
    assert not out["errors"][0]["recoverable"]


async def test_dep_on_step_absent_from_plan_rejected(registry):
    # beta depends on alpha, but alpha is not a step (and not dropped) → error
    planner = make_planner(registry, plan_json("beta", deps={"beta": ["alpha"]}))
    out = await planner.run({"query": "q"})
    assert "unknown step" in out["errors"][0]["detail"]


# ---------------- Conversational context in `inputs` -------------------------

def test_user_message_is_the_bare_query_without_conversation(registry):
    planner = make_planner(registry)
    assert planner.user_message({"query": "q", "inputs": {}}) == "q"


async def test_conversation_context_reaches_the_prompt(registry):
    """A chat-launched job carries the turns its request refers to; the planner
    must see them, and must still be told which part is the request."""
    llm = FakeLLM({"planner": plan_json("alpha")})
    planner = Planner(Deps(llm=llm), registry)

    out = await planner.run({
        "query": "analyse that",
        "inputs": {CONVERSATION_INPUT_KEY: "user: the Q3 churn spike\nassistant: noted"},
    })

    assert out["plan"]["steps"] == [{"capability": "alpha", "depends_on": []}]
    user_msg = next(m["content"] for m in llm.calls[0]["messages"] if m["role"] == "user")
    assert "the Q3 churn spike" in user_msg
    assert user_msg.endswith("Request to plan for:\nanalyse that")


def test_planner_prompt_tells_the_model_what_the_excerpt_is_for(registry):
    prompt = make_planner(registry).system_prompt()
    assert "excerpt of the conversation" in prompt
