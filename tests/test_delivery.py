"""A job's ending goes to its return address, on every backend
(→ docs/design/core-v1.md, "Delivery").

The address is JSON on the record; the store finds an address's jobs by its
flat key only, because a nested filter fails on SQLite and a dotted one finds
nothing in memory — the same family of bug as 0141, hence every backend.
"""
from __future__ import annotations

import uuid
from contextlib import AsyncExitStack
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from jobsmith.app.persistence import open_persistence
from jobsmith.engine.delivery import Pulled
from jobsmith.engine.graph import GraphSpec
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import JobStatus


class State(TypedDict):
    n: int


def _graph(checkpointer):
    graph = StateGraph(State)
    graph.add_node("step", lambda state: {"n": state["n"] + 1})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.fixture
async def jobs(db):
    async with AsyncExitStack() as stack:
        checkpointer, store = await open_persistence(db, stack)
        yield JobManager(GraphSpec("plus_one", _graph(checkpointer)), store,
                         deliverers=[Pulled("inbox")])


def inbox() -> dict:
    return {"kind": "inbox", "id": uuid.uuid4().hex}      # a reused database


async def test_a_job_nobody_waits_on_is_delivered_as_it_settles(jobs):
    done = await jobs.run_job((await jobs.create_job({"n": 1})).job_id)
    stopped = await jobs.cancel_job((await jobs.create_job({"n": 1})).job_id)

    assert done.reply_to == {"kind": "none"} and done.delivered_at
    assert stopped.status is JobStatus.CANCELLED and stopped.delivered_at


async def test_a_pulled_job_is_pending_for_its_address_until_marked(jobs):
    mine, theirs = inbox(), inbox()
    job = await jobs.create_job({"n": 1}, reply_to=mine)
    assert await jobs.pending_deliveries(mine) == []              # not settled yet
    await jobs.run_job(job.job_id)
    await jobs.create_job({"n": 2}, reply_to=mine)                # queued: not pending

    assert [j.job_id for j in await jobs.pending_deliveries(mine)] == [job.job_id]
    assert await jobs.pending_deliveries(theirs) == []
    assert [j.job_id for j in await jobs.list_jobs(reply_to=mine, status=JobStatus.DONE)] \
        == [job.job_id]

    await jobs.mark_delivered(job.job_id)
    stamped = (await jobs.get_job(job.job_id)).delivered_at
    await jobs.mark_delivered(job.job_id)
    assert await jobs.pending_deliveries(mine) == []
    assert stamped and (await jobs.get_job(job.job_id)).delivered_at == stamped   # once


@pytest.mark.parametrize("reply_to", [{"kind": "pigeon", "id": "x"}, {"kind": "inbox"}],
                         ids=["no-deliverer", "no-id"])
async def test_an_address_nobody_here_can_deliver_to_is_refused(reply_to):
    jobs = JobManager(GraphSpec("plus_one", _graph(MemorySaver())), InMemoryStore(),
                      deliverers=[Pulled("inbox")])
    with pytest.raises(ValueError):
        await jobs.create_job({"n": 1}, reply_to=reply_to)
