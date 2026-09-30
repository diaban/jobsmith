"""A plan step is known by its id: its result, its fact and its run count are
keyed by it, a capability learns it without passing anything along, and
whoever reads a plan (`drop_steps`, the chat, the REPL, the TUI) speaks ids
(→ docs/design/compiler-v1.md, "Step identity"; 0211, 0213)."""
from __future__ import annotations

import pytest
from langgraph.graph import END
from support import OneStep, dag_job

from jobsmith.chat.tools import progress_line, running_steps
from jobsmith.cli.repl import plan_line
from jobsmith.dag.capability import CapabilityBaseState
from jobsmith.dag.executor import Executor
from jobsmith.dag.jobs import DagJobs
from jobsmith.dag.planner import without_steps
from jobsmith.dag.registry import CapabilityRegistry
from jobsmith.engine.facts import FACT_KEY
from jobsmith.engine.models import JobStatus


class Echo(OneStep):
    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": state["query"]})


class SyncEcho(OneStep):
    """The same, from a synchronous node: the id must reach it too."""

    def work_sync(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": state["query"]})

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work_sync)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


PLAN = {"steps": [
    {"id": "early", "capability": "echo", "depends_on": []},
    {"id": "late", "capability": "sync_echo", "depends_on": ["early"]},
], "rationale": "ids that are not names"}


async def run(capability, payload) -> tuple[dict, dict]:
    """Its output and the facts it published, run on `payload` as sent."""
    facts, output = {}, {}
    async for mode, chunk in capability.build().astream(payload, stream_mode=["custom", "values"]):
        if mode == "custom" and FACT_KEY in chunk:
            facts[chunk[FACT_KEY]] = chunk["value"]
        elif mode == "values":
            output = chunk
    return output, facts


async def test_a_step_runs_and_reports_under_its_id_not_its_capability_name():
    executor = Executor(CapabilityRegistry([Echo("echo"), SyncEcho("sync_echo")]))
    state = {"query": "q", "plan": PLAN, "completed_capabilities": [], "results": {}}

    [first] = executor.route(state)
    assert (first.node, first.arg["step"]) == ("cap_echo", {"id": "early", "capability": "echo"})
    output, facts = await run(Echo("echo"), first.arg)
    assert output["results"].keys() == {"early"} and output["completed_capabilities"] == ["early"]
    assert "step:early" in facts and "step:echo" not in facts

    state |= {"completed_capabilities": ["early"], "results": output["results"]}
    [second] = executor.route(state)
    assert second.arg["step"]["id"] == "late"
    output, _ = await run(SyncEcho("sync_echo"), second.arg)
    assert output["completed_capabilities"] == ["late"]

    state["completed_capabilities"].append("late")
    assert executor.route(state) == "merge_results"


async def test_a_capability_outside_a_plan_and_a_plan_without_ids_key_by_name():
    """A capability run on its own, and a plan checkpointed before ids, read as
    ids = names: nothing written before 0a changes meaning."""
    output, facts = await run(Echo("echo"), {"query": "q"})
    assert output["completed_capabilities"] == ["echo"] and "step:echo" in facts

    executor = Executor(CapabilityRegistry([Echo("echo")]))
    [sent] = executor.route({"plan": {"steps": [{"capability": "echo", "depends_on": []}],
                                      "rationale": ""}, "completed_capabilities": []})
    assert sent.arg["step"] == {"id": "echo", "capability": "echo"}


def test_the_view_orders_and_drops_steps_by_id():
    job = dag_job(job_id="j1", status=JobStatus.DONE, plan=PLAN,
                  results={"late": {"ok": True}, "early": {"ok": True}},
                  step_finished_at={"early": "t1"})
    assert [name for name, _ in job.ordered_results()] == ["early", "late"]
    assert job.step_finished_at == {"early": "t1"}
    assert [s["id"] for s in without_steps(PLAN, ["late"])["steps"]] == ["early"]


TWICE = {"steps": [
    {"id": "notes_a", "capability": "research", "depends_on": []},
    {"id": "notes_b", "capability": "research", "depends_on": []},
    {"id": "review", "capability": "critique", "depends_on": ["notes_a", "notes_b"]},
], "rationale": "one capability, two steps"}


@pytest.mark.parametrize("name, meant", [
    ("notes_b", "notes_b"),          # an id is itself
    ("critique", "review"),          # a name stands for the one step that runs it
    ("slides", None),                # neither: not in the plan
])
def test_a_step_to_drop_is_named_by_its_id_or_by_its_only_capability(name, meant):
    assert DagJobs._one_step("j1", TWICE, name) == meant


def test_a_capability_that_runs_as_several_steps_is_refused_as_ambiguous():
    with pytest.raises(ValueError, match="notes_a, notes_b"):
        DagJobs._one_step("j1", TWICE, "research")


def test_the_repl_and_the_chat_show_and_key_steps_by_id():
    """While ids are names nothing shows; once they differ, the REPL and the
    chat say which step, not which capability (the TUI: `test_tui.py`, → 0213)."""
    assert plan_line(TWICE["steps"]) == "notes_a + notes_b → review"

    job = dag_job(job_id="j1", status=JobStatus.RUNNING, plan=TWICE,
                  results={"notes_a": {"ok": True}}, step_finished_at={"notes_a": "t1"})
    assert running_steps(job) == ["notes_b"]
    assert "1/3 steps done (notes_a)" in progress_line(job)
