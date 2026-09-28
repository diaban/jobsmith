"""A job's ending goes to its return address, on every backend
(→ docs/design/core-v1.md, "Delivery").

The address is JSON on the record; the store finds an address's jobs by its
flat key only, because a nested filter fails on SQLite and a dotted one finds
nothing in memory — the same family of bug as 0141, hence every backend.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack
from typing import TypedDict

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from support import until

from jobsmith.app.persistence import open_persistence
from jobsmith.engine.delivery import Pulled, Pushed, Webhook
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
async def backend(db):
    async with AsyncExitStack() as stack:
        yield await open_persistence(db, stack)


@pytest.fixture
def jobs(backend):
    checkpointer, store = backend
    return JobManager(GraphSpec("plus_one", _graph(checkpointer)), store,
                      deliverers=[Pulled("inbox")])


def inbox() -> dict:
    return {"kind": "inbox", "id": uuid.uuid4().hex}      # a reused database


async def test_a_job_nobody_waits_on_is_delivered_as_it_settles(jobs):
    queued = await jobs.create_job({"n": 1})
    assert queued.delivered_at is None                            # not before
    done = await jobs.run_job(queued.job_id)
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


@pytest.mark.parametrize("reply_to", [{"kind": "pigeon", "id": "x"}, {"kind": "inbox"},
                                      {"kind": "inbox", "id": ""}],
                         ids=["no-deliverer", "no-id", "empty-id"])
async def test_an_address_nobody_here_can_deliver_to_is_refused(reply_to):
    jobs = JobManager(GraphSpec("plus_one", _graph(MemorySaver())), InMemoryStore(),
                      deliverers=[Pulled("inbox")])
    with pytest.raises(ValueError):
        await jobs.create_job({"n": 1}, reply_to=reply_to)


class Receiver(Pushed):
    """A pushed kind whose receiver refuses its first `failures` pushes, and
    takes none before `opened` is set."""

    kind = "hook"
    first_retry, max_retry = 0.01, 0.02

    def __init__(self, failures: int = 0, *, opened: bool = True):
        self.failures, self.calls, self.received = failures, 0, []
        self.opened = asyncio.Event()
        if opened:
            self.opened.set()

    def key(self, reply_to: dict) -> str:
        return f"hook:{reply_to['id']}"

    async def push(self, job) -> None:
        await self.opened.wait()
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("receiver down")
        self.received.append((job.job_id, job.attempt))


def pushing(backend, receiver: Receiver) -> JobManager:
    checkpointer, store = backend
    return JobManager(GraphSpec("plus_one", _graph(checkpointer)), store,
                      deliverers=[receiver])


async def delivered(jobs: JobManager, job_id: str):
    return (await jobs.get_job(job_id)).delivered_at


async def test_a_push_lands_off_the_persist_path_after_its_retries(backend):
    """The ending is saved before anything is pushed, a failing receiver never
    breaks it, and the stamp waits for the push that landed. → 0166"""
    receiver = Receiver(failures=2, opened=False)
    jobs = pushing(backend, receiver)
    job = await jobs.create_job({"n": 1}, reply_to={"kind": "hook", "id": uuid.uuid4().hex})
    done = await jobs.run_job(job.job_id)
    assert (done.status, done.result, done.delivered_at) == (JobStatus.DONE, {"n": 2}, None)
    assert (await jobs.get_job(job.job_id)).status is JobStatus.DONE     # saved, not pushed

    receiver.opened.set()
    await until(lambda: delivered(jobs, job.job_id), what="the push landing")
    assert receiver.calls == 3 and receiver.received == [(job.job_id, 1)]


async def test_a_push_a_stopped_process_left_pending_is_made_at_the_next_start(backend):
    down = Receiver(opened=False)
    first = pushing(backend, down)
    job = await first.create_job({"n": 1}, reply_to={"kind": "hook", "id": uuid.uuid4().hex})
    await first.run_job(job.job_id)
    for task in list(first._pushes.values()):        # the process stops
        task.cancel()

    up = Receiver()
    restarted = pushing(backend, up)
    await restarted.recover_interrupted()
    await until(lambda: delivered(restarted, job.job_id), what="the pending push, pushed again")
    assert up.received == [(job.job_id, 1)] and down.received == []


async def test_a_webhook_posts_the_ending_where_it_may_and_is_refused_elsewhere():
    posted = []

    def receive(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(500 if len(posted) == 1 else 204)

    hook = Webhook(["https://hooks.example/"],
                   client=httpx.AsyncClient(transport=httpx.MockTransport(receive)))
    hook.first_retry = hook.max_retry = 0.01
    jobs = JobManager(GraphSpec("plus_one", _graph(MemorySaver())), InMemoryStore(),
                      deliverers=[hook])
    with pytest.raises(ValueError, match="hooks.example"):
        await jobs.create_job({"n": 1}, reply_to={"kind": "webhook", "url": "http://10.0.0.1/"})

    job = await jobs.create_job({"n": 1}, reply_to={"kind": "webhook",
                                                    "url": "https://hooks.example/in"})
    await jobs.run_job(job.job_id)
    await until(lambda: delivered(jobs, job.job_id), what="the webhook answering 2xx")
    assert len(posted) == 2                                     # the 500 was retried
    body = httpx.Response(200, content=posted[-1].content).json()
    assert (body["job_id"], body["result"], body["attempt"]) == (job.job_id, {"n": 2}, 1)
    assert posted[-1].headers["Idempotency-Key"] == f"{job.job_id}:1"
