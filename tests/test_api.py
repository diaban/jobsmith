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
from jobsmith.jobs.report import compose_reporters, make_reporter
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
        assert report.headers["content-type"].startswith("text/markdown")

        # the same file is listed as the job's main output, and downloadable
        (output,) = (await client.get(f"/jobs/{job['job_id']}/outputs")).json()
        assert (output["role"], output["format"]) == ("main", "markdown")
        download = await client.get(f"/jobs/{job['job_id']}/outputs/{output['name']}")
        assert download.status_code == 200


async def test_report_content_type_follows_the_deliverable_format(
    store, checkpointer, tmp_path
):
    """/report announces what it actually serves. Every other test asserts on
    the body, which is how an HTML report kept being labelled text/markdown —
    correct bytes, wrong header, unreadable in a browser."""
    app, manager = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])

    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "a markdown run"})).json()["job_id"])
        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.headers["content-type"].startswith("text/markdown")

        # the Reporter is the manager's documented swap seam: same run, other format
        manager.reporter = make_reporter("html")
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "an html run"})).json()["job_id"])
        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.headers["content-type"].startswith("text/html")
        assert report.text.startswith("<!doctype html>")


async def test_every_deliverable_is_listed_and_downloadable(store, checkpointer, tmp_path):
    """A run asked for two formats: both are outputs of the job, both can be
    downloaded, and /report still serves the main one — with its own type."""
    app, manager = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])
    manager.reporter = compose_reporters("markdown,html")

    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "two formats"})).json()["job_id"])
        assert len(job["outputs"]) == 2

        outputs = (await client.get(f"/jobs/{job['job_id']}/outputs")).json()
        assert [(o["role"], o["format"]) for o in outputs] == [
            ("main", "markdown"), ("alternate", "html")]
        for output in outputs:
            download = await client.get(f"/jobs/{job['job_id']}/outputs/{output['name']}")
            assert download.status_code == 200
        assert outputs[1]["name"].endswith(".html")

        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.headers["content-type"].startswith("text/markdown")
        assert report.text.startswith("# two formats")
        assert job["report_path"] == outputs[0]["path"]


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


async def test_resume_endpoint_restarts_a_stopped_job(store, checkpointer, tmp_path):
    """A cancelled job is restarted from its checkpoint over HTTP; a job with
    nothing left to run is refused with 409 rather than silently accepted."""
    from test_jobs import cancelled_midway

    manager, job, alpha, slow = await cancelled_midway(store, checkpointer, tmp_path)
    app = create_api(LocalAgentService(manager, lambda session_id=None: None))
    async with client_for(app) as client:
        slow.delay = 0.0
        r = await client.post(f"/jobs/{job.job_id}/resume")
        assert r.status_code == 200 and r.json()["status"] == "running"

        finished = await wait_done(client, job.job_id)
        assert finished["status"] == "done"
        assert finished["results"]["alpha"]["data"]["echo"] == "alpha#1"   # not re-run

        # done: resuming it again is a refusal, not a no-op
        again = await client.post(f"/jobs/{job.job_id}/resume")
        assert again.status_code == 409 and "expected cancelled" in again.json()["detail"]
        assert (await client.post("/jobs/nope/resume")).status_code == 404


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
