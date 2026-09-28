"""A graph that pauses at a LangGraph `interrupt()` waits for an answer: its
own status, what it asks on the record, and a way to answer it. → 0167

It used to end FAILED with "its result is not JSON": the last root `values`
of a paused run carries the interrupts, and was read as its result.
"""
from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt
from support import until

from jobsmith.engine.graph import GraphSpec
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import JobStatus


class State(TypedDict, total=False):
    n: int
    answer: str


def asking_graph(checkpointer):
    """`first`, then `ask`, which asks which one and keeps the answer."""
    graph = StateGraph(State)
    graph.add_node("first", lambda state: {"n": state["n"] + 1})
    graph.add_node("ask", lambda state: {"answer": interrupt({"which": ["A", "B"]})})
    graph.add_edge(START, "first")
    graph.add_edge("first", "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.fixture
def jobs():
    return JobManager(GraphSpec("asking", asking_graph(MemorySaver())), InMemoryStore())


async def test_a_paused_run_waits_with_its_question_and_an_answer_runs_it_on(jobs):
    job = await jobs.create_job({"n": 1})
    paused = await jobs.run_job(job.job_id)
    assert (paused.status, paused.asked, paused.error) == (
        JobStatus.NEEDS_INPUT, [{"which": ["A", "B"]}], None)
    assert list(paused.steps) == ["first"] and paused.delivered_at is None   # not an ending
    stored = await jobs.get_job(job.job_id)
    assert (stored.status, stored.asked) == (JobStatus.NEEDS_INPUT, [{"which": ["A", "B"]}])

    done = await jobs.answer_job(job.job_id, "B")
    assert (done.status, done.result, done.asked, done.attempt) == (
        JobStatus.DONE, {"n": 2, "answer": "B"}, None, 2)
    assert done.delivered_at


async def test_only_a_paused_job_takes_an_answer_and_a_resume_does_not_take_it(jobs):
    paused = await jobs.run_job((await jobs.create_job({"n": 1})).job_id)
    with pytest.raises(ValueError, match="expected cancelled or failed"):
        await jobs.start_resume(paused.job_id)
    done = await jobs.answer_job(paused.job_id, "A")
    with pytest.raises(ValueError, match="expected needs_input"):
        await jobs.answer_job(done.job_id, "A")


async def test_a_job_nobody_answers_can_be_cancelled_and_that_ending_is_told(jobs):
    paused = await jobs.run_job((await jobs.create_job({"n": 1})).job_id)
    stopped = await jobs.cancel_job(paused.job_id)
    assert stopped.status is JobStatus.CANCELLED and stopped.delivered_at


async def test_a_paused_job_is_answered_through_the_engine_door(tmp_path):
    from support import client_for

    from jobsmith.agents.base import AgentDefinition
    from jobsmith.api import create_api
    from jobsmith.app.agent import build_app
    from jobsmith.app.providers import make_llm

    asking = AgentDefinition(name="asking", description="asks which one",
                             graph=lambda ctx: GraphSpec("asking", asking_graph(ctx.checkpointer)))
    app = await build_app(agent=asking, llm=make_llm("fake"), chat_model=object(),
                          db="memory", reports_dir=str(tmp_path))
    async with client_for(create_api(app.service())) as client:
        job = (await client.post("/engine/jobs?wait=10", json={"input": {"n": 1}})).json()
        assert (job["status"], job["asked"]) == ("needs_input", [{"which": ["A", "B"]}])

        answered = await client.post(f"/engine/jobs/{job['job_id']}/answer", json={"answer": "A"})
        assert answered.status_code == 200

        async def settled():
            current = (await client.get(f"/engine/jobs/{job['job_id']}")).json()
            return current if current["status"] == "done" else None

        done = await until(settled, what="the answered job ending")
        assert done["result"] == {"n": 2, "answer": "A"}
        again = await client.post(f"/engine/jobs/{job['job_id']}/answer", json={"answer": "A"})
        assert again.status_code == 409
    await app.aclose()
