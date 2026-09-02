"""Executor wave dispatch over a diamond dependency DAG."""
from __future__ import annotations

from langgraph.types import Send

from jobsmith.core.capability import Capability, CapabilitySpec
from jobsmith.core.executor import Executor
from jobsmith.core.registry import CapabilityRegistry


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
