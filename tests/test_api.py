"""The HTTP adapter: what only raw HTTP shows — status codes, media types,
downloads. What the port answers is `test_service.py`, on both backings;
`/events` is tested there too, under uvicorn (ASGITransport hangs on it)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from support import StubPdf, cancelled_midway, client_for, make_app, wait_done

from jobsmith.api import create_api
from jobsmith.service import LocalAgentService


@pytest.mark.parametrize(("formats", "media_type", "starts"), [
    (["markdown", "html"], "text/markdown", "# two formats"),
    (["html", "markdown"], "text/html", "<!doctype html>"),
])
async def test_report_serves_the_main_deliverable_with_its_own_type_and_all_are_downloadable(
    store, checkpointer, tmp_path, formats, media_type, starts
):
    app, _ = make_app(store, checkpointer, tmp_path, [])
    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "two formats", "formats": formats})).json()["job_id"])

        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.headers["content-type"].startswith(media_type)
        assert report.text.startswith(starts)

        outputs = (await client.get(f"/jobs/{job['job_id']}/outputs")).json()
        assert [(o["role"], o["format"]) for o in outputs] == [
            ("main", formats[0]), ("alternate", formats[1])]
        for output in outputs:
            download = await client.get(f"/jobs/{job['job_id']}/outputs/{output['name']}")
            assert download.status_code == 200


async def test_a_binary_deliverable_is_refused_by_report_and_offered_by_outputs(
    store, checkpointer, tmp_path
):
    """/report serves text inline. A PDF main deliverable is neither servable
    that way nor absent, so it is a 415 that names the download — a 404 would
    say the job has no report, which is the one thing that is false. The bytes
    are on /outputs/{name}, and this asserts they really are."""
    app, manager = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])
    manager.reporter = StubPdf()

    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "a printed run", "formats": ["default"]})).json()["job_id"])

        report = await client.get(f"/jobs/{job['job_id']}/report")
        assert report.status_code == 415
        name = f"{job['job_id']}.pdf"
        assert report.json()["detail"].endswith(f"/jobs/{job['job_id']}/outputs/{name}")

        download = await client.get(f"/jobs/{job['job_id']}/outputs/{name}")
        assert download.status_code == 200 and download.content.startswith(b"%PDF-")


async def test_direct_job_launch_and_cancel_and_404s(store, checkpointer, tmp_path):
    app, _ = make_app(store, checkpointer, tmp_path, [AIMessage(content="hi")])
    async with client_for(app) as client:
        r = await client.post("/jobs", json={"query": "direct run"})
        assert r.status_code == 201
        job = await wait_done(client, r.json()["job_id"])
        assert job["plan"] is not None and job["results"]   # the UI's DAG data

        # cancel on a finished job is a no-op status echo
        r = await client.post(f"/jobs/{job['job_id']}/cancel")
        assert r.json()["status"] == "done"

        assert (await client.get("/jobs/nope")).status_code == 404
        assert (await client.get("/jobs/nope/report")).status_code == 404


async def test_resume_endpoint_restarts_a_stopped_job(store, checkpointer, tmp_path):
    """A cancelled job is restarted from its checkpoint over HTTP; a job with
    nothing left to run is refused with 409 rather than silently accepted."""

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
