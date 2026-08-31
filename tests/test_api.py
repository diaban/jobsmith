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

from agent_oo.api import create_api
from agent_oo.chat import ChatSession


def make_app(store, checkpointer, tmp_path, responses):
    manager = make_manager(store, checkpointer, tmp_path)

    def session_factory() -> ChatSession:
        return ChatSession(
            manager,
            ScriptedChatModel(responses=list(responses)),
            checkpointer=MemorySaver(),
        )

    return create_api(manager, session_factory), manager


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
        assert report.text.startswith("# Job report")


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
        assert (await client.post("/sessions/nope/messages",
                                  json={"text": "x"})).status_code == 404
