"""HTTP entrypoint: chat sessions with HITL over HTTP, jobs endpoints, report.

The SSE /events endpoint streams forever, which httpx's ASGITransport cannot
consume — its pub/sub mechanism is covered by test_jobs.py's subscribe test.
"""
from __future__ import annotations

import asyncio

from conftest import ScriptedChatModel
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_jobs import make_manager

from jobsmith.api import create_api
from jobsmith.chat import ChatSession
from jobsmith.service import LocalAgentService


def make_app(store, checkpointer, tmp_path, responses):
    manager = make_manager(store, checkpointer, tmp_path)
    checkpointer_for_sessions = MemorySaver()

    def session_factory(session_id: str | None = None) -> ChatSession:
        return ChatSession(
            manager,
            ScriptedChatModel(responses=list(responses)),
            session_id=session_id,
            checkpointer=checkpointer_for_sessions,
        )

    return create_api(LocalAgentService(manager, session_factory)), manager


def client_for(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def wait_done(client: AsyncClient, job_id: str) -> dict:
    for _ in range(300):
        job = (await client.get(f"/jobs/{job_id}")).json()
        if job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


async def test_chat_flow_proposal_approval_report(store, checkpointer, tmp_path):
    app, _ = make_app(store, checkpointer, tmp_path, [
        launch_call("analyse the data", "several steps needed"),
        AIMessage(content="Job launched — report coming."),
    ])
    async with client_for(app) as client:
        sid = (await client.post("/sessions")).json()["session_id"]

        r = (await client.post(f"/sessions/{sid}/messages",
                               json={"text": "please analyse the data"})).json()
        assert r == {"type": "proposal", "query": "analyse the data",
                     "rationale": "several steps needed"}

        r = (await client.post(f"/sessions/{sid}/approval", json={"approved": True})).json()
        assert r["type"] == "message" and "report coming" in r["content"]

        jobs = (await client.get("/jobs", params={"session_id": sid})).json()
        assert len(jobs) == 1
        job = await wait_done(client, jobs[0]["job_id"])
        assert job["status"] == "done"
        assert job["plan"] is not None and job["results"]  # DAG + artifacts for the UI

        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.status_code == 200
        assert report.text.startswith("# analyse the data")   # deliverable first

        # the same file is listed as the job's main output, and downloadable
        (output,) = (await client.get(f"/jobs/{job['job_id']}/outputs")).json()
        assert (output["role"], output["format"]) == ("main", "markdown")
        download = await client.get(f"/jobs/{job['job_id']}/outputs/{output['name']}")
        assert download.status_code == 200


async def test_chat_flow_decline_creates_no_job(store, checkpointer, tmp_path):
    app, _ = make_app(store, checkpointer, tmp_path, [
        launch_call("big task", "complex"),
        AIMessage(content="Ok, not launching it."),
    ])
    async with client_for(app) as client:
        sid = (await client.post("/sessions")).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages", json={"text": "do the big task"})
        r = (await client.post(f"/sessions/{sid}/approval", json={"approved": False})).json()
        assert r["type"] == "message"
        assert (await client.get("/jobs", params={"session_id": sid})).json() == []


async def test_direct_job_launch_and_cancel_and_404s(store, checkpointer, tmp_path):
    app, _ = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])
    async with client_for(app) as client:
        r = await client.post("/jobs", json={"query": "direct run"})
        assert r.status_code == 201
        job = await wait_done(client, r.json()["job_id"])
        assert job["report_path"] is not None

        # cancel on a finished job is a no-op status echo
        r = await client.post(f"/jobs/{job['job_id']}/cancel")
        assert r.json()["status"] == "done"

        assert (await client.get("/jobs/nope")).status_code == 404
        assert (await client.get("/jobs/nope/report")).status_code == 404


async def test_session_is_resumable_by_id(store, checkpointer, tmp_path):
    """The registry is a cache: chatting on a known id rebuilds the session,
    so a client keeps its conversation across a daemon restart."""
    app, _ = make_app(store, checkpointer, tmp_path, [AIMessage(content="hello again")])
    async with client_for(app) as client:
        sid = (await client.post("/sessions")).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages", json={"text": "first"})

        # an id this process never registered is accepted, not rejected
        r = await client.post("/sessions/unknown-but-valid/messages", json={"text": "hi"})
        assert r.status_code == 200

        # explicit resume returns the same id
        r = await client.post("/sessions", json={"session_id": sid})
        assert r.json()["session_id"] == sid


async def test_health(store, checkpointer, tmp_path):
    app, _ = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])
    async with client_for(app) as client:
        assert (await client.get("/health")).json()["service"] == "jobsmith"
