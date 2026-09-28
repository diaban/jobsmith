"""Executor wave dispatch over a diamond dependency DAG."""
from __future__ import annotations

import pytest
from langgraph.types import Send

from jobsmith.dag.capability import Capability, CapabilitySpec
from jobsmith.dag.executor import Executor
from jobsmith.dag.registry import CapabilityRegistry


class StubCap(Capability):
    def __init__(self, name: str):
        self.spec = CapabilitySpec(name=name, description=name)

    def build(self):
        raise NotImplementedError


DIAMOND = {
    "steps": [
        {"capability": "a", "depends_on": []},
        {"capability": "b", "depends_on": ["a"]},
        {"capability": "c", "depends_on": ["a"]},
        {"capability": "d", "depends_on": ["b", "c"]},
    ],
    "rationale": "diamond",
}


def make_executor() -> Executor:
    return Executor(CapabilityRegistry([StubCap(n) for n in "abcd"]))


def sends(routed) -> list[str]:
    assert isinstance(routed, list) and all(isinstance(s, Send) for s in routed)
    return sorted(s.node for s in routed)


def test_wave_progression():
    ex = make_executor()
    state = {"plan": DIAMOND, "completed_capabilities": []}
    assert sends(ex.route(state)) == ["cap_a"]

    state["completed_capabilities"] = ["a"]
    assert sends(ex.route(state)) == ["cap_b", "cap_c"]

    state["completed_capabilities"] = ["a", "b"]          # c still running
    assert sends(ex.route(state)) == ["cap_c"]

    state["completed_capabilities"] = ["a", "b", "c"]
    assert sends(ex.route(state)) == ["cap_d"]

    state["completed_capabilities"] = ["a", "b", "c", "d"]
    assert ex.route(state) == "merge_results"


def test_unrecoverable_error_short_circuits():
    ex = make_executor()
    state = {
        "plan": DIAMOND,
        "completed_capabilities": ["a"],
        "errors": [{"source": "b", "kind": "x", "detail": "boom", "recoverable": False}],
    }
    assert ex.route(state) == "execution_error"


def test_recoverable_errors_do_not_block():
    ex = make_executor()
    state = {
        "plan": DIAMOND,
        "completed_capabilities": ["a"],
        "errors": [{"source": "a", "kind": "x", "detail": "meh", "recoverable": True}],
    }
    assert sends(ex.route(state)) == ["cap_b", "cap_c"]


def test_missing_plan_is_execution_error():
    ex = make_executor()
    assert ex.route({"completed_capabilities": []}) == "execution_error"


FAILED_ONCE = {"ok": False, "error": "the model call failed", "retryable": True}


@pytest.mark.parametrize(("runs", "result", "routed"), [
    (["a"], FAILED_ONCE, ["cap_a"]),                             # again, before b and c
    (["a", "a"], FAILED_ONCE, ["cap_b", "cap_c"]),               # its retry spent
    (["a"], {"ok": False, "error": "nothing found"}, ["cap_b", "cap_c"]),   # not transient
], ids=["retried", "bounded", "not-retryable"])
def test_a_step_that_failed_transiently_runs_again_before_its_dependents(runs, result, routed):
    """→ 0191"""
    state = {"plan": DIAMOND, "completed_capabilities": runs, "results": {"a": result}}
    assert sends(make_executor().route(state)) == routed


async def test_a_retried_step_feeds_its_dependents_its_second_result():
    from conftest import FakeLLM, plan_json
    from langgraph.checkpoint.memory import MemorySaver
    from support import OneStep

    from jobsmith.dag.builder import build_agent
    from jobsmith.dag.deps import Deps

    class Flaky(OneStep):
        calls = 0

        async def work(self, state):
            Flaky.calls += 1
            if Flaky.calls == 1:
                return self._emit_failure("the model call failed", retryable=True)
            return self._emit_success({"echo": "second try"})

    class Reader(OneStep):
        async def work(self, state):
            return self._emit_success({"saw": state["results"]["flaky"]["ok"]})

    llm = FakeLLM({"planner": plan_json("flaky", "reader", deps={"reader": ["flaky"]})},
                  default="an answer long enough to pass the length floor")
    graph = build_agent(Deps(llm=llm), CapabilityRegistry([Flaky("flaky"), Reader("reader")]),
                        checkpointer=MemorySaver())
    out = await graph.ainvoke({"query": "do it", "job_id": "r1"},
                              config={"configurable": {"thread_id": "r1"}})
    assert Flaky.calls == 2
    assert out["results"]["flaky"]["ok"] and out["results"]["reader"]["data"] == {"saw": True}


@pytest.mark.parametrize(("reply", "retryable"), [(None, True), ("", False)],
                         ids=["the-call-raised", "the-answer-was-empty"])
async def test_the_default_steps_call_a_failed_model_call_transient_and_nothing_else(
        reply, retryable):
    from jobsmith.agents.default.analysis import AnalysisCapability

    class Model:
        async def chat(self, **_):
            if reply is None:
                raise TimeoutError("the provider did not answer")
            return reply

    out = await AnalysisCapability(Model()).build().ainvoke({"query": "q", "results": {}})
    result = out["results"]["analysis"]
    assert result["ok"] is False and bool(result.get("retryable")) is retryable


async def test_each_run_of_a_step_is_counted_once():
    """`Send(node, state)` hands a sub-graph the whole parent state; the
    append-only channels must not come back echoed, or a step's runs — which
    bound its retries — and its errors are counted several times. → 0194"""
    from conftest import FakeLLM, plan_json
    from langgraph.checkpoint.memory import MemorySaver
    from support import OneStep

    from jobsmith.dag.builder import build_agent
    from jobsmith.dag.deps import Deps
    from jobsmith.dag.profile import AgentProfile as Profile

    class Step(OneStep):
        runs = 0

        async def work(self, state):
            if self.spec.name == "b":
                Step.runs += 1
                return self._emit_failure("the model call failed", retryable=True)
            return self._emit_success({"echo": self.spec.name})

    llm = FakeLLM({"planner": plan_json("a", "b", "c", deps={"b": ["a"], "c": ["b"]})},
                  default="an answer long enough to pass the length floor")
    graph = build_agent(Deps(llm=llm), CapabilityRegistry([Step(n) for n in "abc"]),
                        profile=Profile(max_step_retries=2), checkpointer=MemorySaver())
    out = await graph.ainvoke({"query": "q", "job_id": "n1"},
                              config={"configurable": {"thread_id": "n1"}})
    assert out["completed_capabilities"] == ["a", "b", "b", "b", "c"]
    assert Step.runs == 3 and len(out["errors"]) == 3
