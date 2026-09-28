"""A running job's course can be changed without cancelling it for good: the
engine stops it, writes the change into its checkpoint and runs it on; the DAG
uses that to take a step out of a running plan ("skip the critique"). → 0177
"""
from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from support import OneStep, make_manager, planning, until

from jobsmith.engine.graph import GraphSpec
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import JobStatus


class State(TypedDict):
    n: int


class Recorder:
    """A return address that keeps every ending it is handed."""
    kind = "rec"

    def __init__(self):
        self.endings: list[str] = []

    def key(self, reply_to):
        return "rec"

    async def deliver(self, job):
        self.endings.append(job.status.value)
        return True


async def settled(jobs, job_id):
    job = await jobs.get_job(job_id)
    return job if job.status is JobStatus.DONE else None


async def test_an_amended_job_runs_on_from_the_change_and_its_stop_is_told_to_nobody():
    gate, entered = asyncio.Event(), asyncio.Event()

    async def slow(state: State) -> State:
        entered.set()
        await gate.wait()
        return {"n": state["n"] + 10}

    graph = StateGraph(State)
    graph.add_node("first", lambda state: {"n": state["n"] + 1})
    graph.add_node("second", slow)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    recorder = Recorder()
    jobs = JobManager(GraphSpec("g", graph.compile(checkpointer=MemorySaver())), InMemoryStore(),
                      deliverers=[recorder])
    job = await jobs.create_job({"n": 1}, reply_to={"kind": "rec"})
    jobs.start_job(job.job_id)
    await entered.wait()                                    # `second` is running

    amended = await jobs.amend_job(job.job_id, {"n": 100})
    assert (amended.status, amended.attempt) == (JobStatus.RUNNING, 2)
    gate.set()
    done = await until(lambda: settled(jobs, job.job_id), what="the amended run")
    assert done.result == {"n": 110}                        # `second` ran again, on the change
    assert recorder.endings == ["done"]                     # the stop was not an ending


async def test_a_step_taken_out_of_a_running_plan_never_runs(store, checkpointer, tmp_path):
    gate, entered = asyncio.Event(), asyncio.Event()

    class Step(OneStep):
        ran: list[str] = []

        async def work(self, state):
            Step.ran.append(self.spec.name)
            if self.spec.name == "analysis":
                entered.set()
                await gate.wait()
            return self._emit_success({"echo": self.spec.name})

    caps = [Step(name) for name in ("research", "analysis", "critique")]
    jobs = make_manager(store, checkpointer, tmp_path, caps=caps, llm=planning(
        "research", "analysis", "critique",
        deps={"analysis": ["research"], "critique": ["analysis"]}))
    job = await jobs.create_job("compare A and B")
    jobs.start_job(job.job_id)
    await entered.wait()

    with pytest.raises(ValueError, match="already done"):
        await jobs.drop_steps(job.job_id, ["research"])
    with pytest.raises(ValueError, match="not in the plan"):
        await jobs.drop_steps(job.job_id, ["slide_deck"])

    await jobs.drop_steps(job.job_id, ["critique"])
    gate.set()
    done = await until(lambda: settled(jobs, job.job_id), what="the run without critique")
    assert "critique" not in Step.ran and Step.ran.count("analysis") == 2   # stopped, re-run
    assert [s["capability"] for s in done.plan["steps"]] == ["research", "analysis"]
    assert sorted(done.results) == ["analysis", "research"] and done.final_answer


async def test_a_plan_cannot_be_emptied(store, checkpointer, tmp_path):
    gate, entered = asyncio.Event(), asyncio.Event()

    class First(OneStep):
        async def work(self, state):
            entered.set()
            await gate.wait()
            return self._emit_success({"echo": "first"})

    jobs = make_manager(store, checkpointer, tmp_path, caps=[First("research")])
    job = await jobs.create_job("compare A and B")
    jobs.start_job(job.job_id)
    await entered.wait()
    with pytest.raises(ValueError, match="nothing to do"):
        await jobs.drop_steps(job.job_id, ["research"])
    gate.set()
    assert (await until(lambda: settled(jobs, job.job_id), what="the untouched run")).final_answer
