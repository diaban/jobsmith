"""A program is an IR (`dag/ir.py`): references are parsed, never
interpolated; dependencies are derived from them and `after`; today's plans
read as programs with no args; a reference resolves to what the run has, or
says why not; and through the engine a step's argument comes from a
reference (→ docs/design/compiler-v1.md, "The IR", step 1b; 0230)."""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM
from support import ANSWER, OneStep, make_manager

from jobsmith.dag.capability import CapabilityBaseState, CapabilitySpec
from jobsmith.dag.generation import ContextMerger
from jobsmith.dag.ir import (
    Program,
    Step,
    UnresolvedReference,
    dependencies,
    from_plan,
    parse_ref,
    resolve,
)
from jobsmith.dag.profile import AgentProfile
from jobsmith.dag.registry import CapabilityRegistry
from jobsmith.dag.state import step_args


@pytest.mark.parametrize("value, parsed", [
    ("$compare", ("compare", [])),
    ("$compare.items", ("compare", ["items"])),
    ("$input.source_files", ("input", ["source_files"])),
    ("$item.source", ("item", ["source"])),
    ("see $compare", None),          # a whole value or nothing: no interpolation
    ("$Compare", None),
    ("$", None),
    (42, None),
])
def test_a_reference_is_a_whole_value_of_one_shape(value, parsed):
    assert parse_ref(value) == parsed


def test_dependencies_are_derived_from_references_then_after_each_once():
    step = Step(id="c", op="x", after=["b", "a"],
                args={"material": ["$a", "$b.notes"], "files": "$input.paths", "n": 3})
    assert dependencies(step) == ["a", "b"]


def test_a_plan_of_today_reads_as_a_program_with_no_args():
    program = from_plan({"steps": [{"capability": "a", "depends_on": []},
                                   {"id": "b2", "capability": "b", "depends_on": ["a"]}],
                         "rationale": "r"})
    assert [(s.id, s.op, s.args, s.after) for s in program.steps] == [
        ("a", "a", {}, []), ("b2", "b", {}, ["a"])]
    assert program.result.answer.model_dump() == {"mode": "reply", "instruction": "",
                                                  "material": None}
    stored = program.stored()
    assert [(s["capability"], s["depends_on"]) for s in stored["steps"]] == [("a", []), ("b", ["a"])]
    assert stored["version"] == 1 and stored["rationale"] == "r"


RESULTS = {"a": {"ok": True, "data": {"text": "alpha", "nested": {"n": 1}}},
           "broken": {"ok": False, "error": "boom"}}


@pytest.mark.parametrize("value, resolved", [
    ("$a", {"text": "alpha", "nested": {"n": 1}}),
    ("$a.nested.n", 1),
    ("$input.k", "v"),
    (["$a.text", "literal", {"deep": "$a.text"}], ["alpha", "literal", {"deep": "alpha"}]),
    ("see $a", "see $a"),
])
def test_a_reference_resolves_to_what_the_run_has(value, resolved):
    assert resolve(value, results=RESULTS, inputs={"k": "v"}) == resolved


@pytest.mark.parametrize("value", ["$broken", "$missing", "$a.nope", "$input.absent", "$item"])
def test_a_reference_to_nothing_says_so(value):
    with pytest.raises(UnresolvedReference):
        resolve(value, results=RESULTS, inputs={})


class Marked(OneStep):
    """Renders its result under its own name, so the merged context shows
    which results the answer read."""

    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": self.spec.name})

    def render_context(self, result):
        return result["data"]["text"]


async def test_the_answer_reads_the_material_its_slot_names_in_that_order():
    merger = ContextMerger(CapabilityRegistry([Marked("a"), Marked("b"), Marked("c")]),
                           AgentProfile())
    program = Program(steps=[Step(id=n, op=n) for n in "abc"])
    results = {n: {"ok": True, "data": {"text": n}} for n in "abc"}
    every = await merger.run({"query": "q", "plan": program.stored(), "results": results})
    assert every["merged_context"] == "a\n\nb\n\nc"            # null: every result, in order

    program.result.answer.material = ["$c", "$a"]
    named = await merger.run({"query": "q", "plan": program.stored(), "results": results})
    assert named["merged_context"] == "c\n\na"


class Shout(OneStep):
    """Reads its argument, never the request."""

    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": str(step_args(state).get("text", "")).upper()})


class Source(OneStep):
    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": "from the source"})


async def test_through_the_engine_a_steps_argument_comes_from_a_reference(
        store, checkpointer, tmp_path):
    """Gate C1, its first half: the planner writes a reference, the
    interpreter resolves it, the step reads it as its argument."""
    program = json.dumps({"steps": [
        {"id": "src", "op": "source"},
        {"id": "loud", "op": "shout", "args": {"text": "$src.text"}}]})
    manager = make_manager(store, checkpointer, tmp_path,
                           caps=[Source("source"), Shout("shout")],
                           llm=FakeLLM({"planner": program}, default=ANSWER))
    job = await manager.run_job((await manager.create_job("shout it", {})).job_id)

    assert job.results["loud"]["data"] == {"text": "FROM THE SOURCE"}
    assert [s["depends_on"] for s in job.plan["steps"]] == [[], ["src"]]


async def test_through_the_engine_a_step_whose_reference_has_nothing_fails_alone(
        store, checkpointer, tmp_path):
    """A reference to a step that failed fails the referencing step,
    recoverably: the job still answers from what it has."""
    class Broken(OneStep):
        async def work(self, state):
            return self._emit_failure("no source today")

    program = json.dumps({"steps": [
        {"id": "src", "op": "broken"}, {"id": "ok", "op": "source"},
        {"id": "loud", "op": "shout", "args": {"text": "$src.text"}}]})
    manager = make_manager(store, checkpointer, tmp_path,
                           caps=[Broken("broken"), Source("source"), Shout("shout")],
                           llm=FakeLLM({"planner": program}, default=ANSWER))
    job = await manager.run_job((await manager.create_job("shout it", {})).job_id)

    assert job.status.value == "done"
    assert job.results["loud"]["ok"] is False
    assert "step 'src' has no successful result" in job.results["loud"]["error"]
    assert job.results["ok"]["ok"] is True


def test_an_op_declares_effects_and_a_spec_is_an_op():
    """`CapabilitySpec` is `OpSpec` while the code migrates; effects are
    declared, read-only and idempotent unless said otherwise."""
    spec = CapabilitySpec(name="x", description="x")
    assert type(spec).__name__ == "OpSpec"
    assert (spec.effects.read_only, spec.effects.idempotent, spec.output_from) == (True, True, None)
