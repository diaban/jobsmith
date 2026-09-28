"""A job a dead process left is resumed with nobody asking only where its graph
says a node of it may run again, within a bound, by one process. → 0189
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from support import until

from jobsmith.app.persistence import open_persistence
from jobsmith.engine.graph import GraphSpec
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import JobStatus
from jobsmith.engine.ownership import LeasePolicy

FAST = LeasePolicy(heartbeat=0.05, ttl=30.0, poll=0.02)


class State(TypedDict):
    n: int


def two_steps(checkpointer):
    graph = StateGraph(State)
    graph.add_node("first", lambda state: {"n": state["n"] + 1})
    graph.add_node("second", lambda state: {"n": state["n"] + 1})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile(checkpointer=checkpointer)


async def left_running(jobs: JobManager, *, attempt: int = 1):
    """What a process that died between `first` and `second` leaves behind."""
    job = await jobs.create_job({"n": 1})
    await jobs.graph.ainvoke({"n": 1}, {"configurable": {"thread_id": job.job_id}},
                             interrupt_before=["second"])
    job.status, job.attempt = JobStatus.RUNNING, attempt
    await jobs.repo.save_summary(job)
    return job


async def done(jobs: JobManager, job_id: str):
    job = await jobs.get_job(job_id)
    return job if job.status is JobStatus.DONE else None


async def test_an_interrupted_job_is_relaunched_where_its_graph_allows_it():
    store, saver = InMemoryStore(), MemorySaver()
    job = await left_running(JobManager(GraphSpec("g", two_steps(saver), relaunch=1), store))

    restarted = JobManager(GraphSpec("g", two_steps(saver), relaunch=1), store)
    await restarted.recover_interrupted()
    [relaunched] = await restarted.relaunch_interrupted()
    assert relaunched.job_id == job.job_id
    finished = await until(lambda: done(restarted, job.job_id), what="the relaunched job")
    assert (finished.result, finished.attempt, finished.failure) == ({"n": 3}, 2, None)


@pytest.mark.parametrize(("relaunch", "attempt"), [(0, 1), (1, 2)],
                         ids=["the-graph-says-never", "past-its-bound"])
async def test_an_interrupted_job_is_left_failed_otherwise(relaunch, attempt):
    store, saver = InMemoryStore(), MemorySaver()
    job = await left_running(JobManager(GraphSpec("g", two_steps(saver), relaunch=relaunch),
                                        store), attempt=attempt)

    restarted = JobManager(GraphSpec("g", two_steps(saver), relaunch=relaunch), store)
    await restarted.recover_interrupted()
    assert await restarted.relaunch_interrupted() == []
    assert (await restarted.get_job(job.job_id)).failure["kind"] == "interrupted"


async def test_two_processes_starting_together_relaunch_a_job_once(tmp_path, monkeypatch):
    """Both read the job FAILED before either writes it RUNNING: a resume
    checks, then writes, with no compare-and-set between — widened here."""
    from jobsmith.engine.runner import GraphRunner

    checked = GraphRunner.pending

    async def slow_pending(self, job_id):
        frontier = await checked(self, job_id)
        await asyncio.sleep(0.1)
        return frontier

    monkeypatch.setattr(GraphRunner, "pending", slow_pending)
    db = str(tmp_path / "jobs.db")
    async with AsyncExitStack() as stack:
        managers = []
        for _ in range(2):
            saver, store = await open_persistence(db, stack)
            managers.append(JobManager(GraphSpec("g", two_steps(saver), relaunch=1), store,
                                       lease=FAST))
        job = await left_running(managers[0])
        await managers[1].recover_interrupted()

        first, second = await asyncio.gather(*(m.relaunch_interrupted() for m in managers))
        assert len(first) + len(second) == 1
        owner = managers[0] if first else managers[1]
        finished = await until(lambda: done(owner, job.job_id), what="the one relaunch")
        assert finished.attempt == 2


async def test_jobsmith_serve_relaunches_before_it_serves(tmp_path, monkeypatch):
    import uvicorn

    from jobsmith.cli.main import build_parser, serve

    called = []

    async def relaunch(self):
        called.append(self)
        return []

    async def no_server(server):
        assert called, "serving before relaunching"

    monkeypatch.setattr(JobManager, "relaunch_interrupted", relaunch)
    monkeypatch.setattr(uvicorn.Server, "serve", no_server)
    monkeypatch.setenv("JOBSMITH_LLM", "fake")
    monkeypatch.setenv("JOBSMITH_REPORTS_DIR", str(tmp_path))
    assert await serve(build_parser().parse_args(["--db", "memory", "serve"])) == 0
    assert len(called) == 1
