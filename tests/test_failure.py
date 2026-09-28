"""A FAILED job says why as data: its kind, what a resume would run, and
whether a resume takes it — which is exactly what `resume_job` answers. → 0187
"""
from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from jobsmith.engine.graph import GraphSpec, JobFailed
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import JobStatus


class State(TypedDict):
    n: int


def _boom(state: State) -> State:
    raise ValueError("boom")


def two_steps(second=lambda state: {"n": state["n"] + 1}):
    graph = StateGraph(State)
    graph.add_node("first", lambda state: {"n": state["n"] + 1})
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile(checkpointer=MemorySaver())


def declared(output):
    raise JobFailed("nothing to say", result=output)


@pytest.mark.parametrize(("spec", "failure"), [
    (GraphSpec("g", two_steps(_boom)),
     {"kind": "raised", "pending": ["second"], "retryable": True, "exception": "ValueError"}),
    (GraphSpec("g", two_steps(), result=declared),
     {"kind": "declared", "pending": [], "retryable": False}),
    (GraphSpec("g", two_steps(), result=lambda output: object()),
     {"kind": "unreadable", "pending": [], "retryable": False}),
], ids=["raised", "declared", "unreadable"])
async def test_each_way_a_run_fails_is_said_as_data_and_matches_what_resume_answers(
        spec, failure):
    jobs = JobManager(spec, InMemoryStore())
    failed = await jobs.run_job((await jobs.create_job({"n": 1})).job_id)
    assert failed.status is JobStatus.FAILED and failed.error
    assert failed.failure == failure
    assert (await jobs.get_job(failed.job_id)).failure == failure         # on the record
    await _resume_answers(jobs, failed.job_id, failure["retryable"])


async def test_a_job_its_process_left_running_fails_as_interrupted_and_resumable():
    """A dead process leaves RUNNING behind, and a checkpoint with work left."""
    store, graph = InMemoryStore(), two_steps()
    jobs = JobManager(GraphSpec("g", graph), store)
    job = await jobs.create_job({"n": 1})
    config = {"configurable": {"thread_id": job.job_id}}
    await graph.ainvoke({"n": 1}, config, interrupt_before=["second"])   # stopped midway
    job.status = JobStatus.RUNNING
    await jobs.repo.save_summary(job)

    restarted = JobManager(GraphSpec("g", graph), store)
    [settled] = await restarted.recover_interrupted()
    assert settled.failure == {"kind": "interrupted", "pending": ["second"], "retryable": True}
    resumed = await restarted.resume_job(job.job_id)
    assert (resumed.status, resumed.failure, resumed.result) == (JobStatus.DONE, None, {"n": 3})


async def _resume_answers(jobs: JobManager, job_id: str, retryable: bool) -> None:
    if retryable:
        assert (await jobs.start_resume(job_id)).failure is None           # taken, and cleared
    else:
        with pytest.raises(ValueError, match="no checkpoint to resume from"):
            await jobs.start_resume(job_id)
