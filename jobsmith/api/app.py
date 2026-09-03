"""FastAPI app factory — the HTTP shape of the future chat + jobs UI.

    create_api(service) -> FastAPI

This is an **adapter**, not a second implementation of the use cases: every
route below is serialization plus one call into `AgentService`. The same port
backs the CLI, so a command behaves identically whether it runs embedded or
against this API.

- Chat tab:   POST /sessions, then POST /sessions/{id}/messages. A reply is
  either {"type": "message"} or {"type": "proposal"} (the agent wants to
  launch a background job — human-in-the-loop); the client answers with
  POST /sessions/{id}/approval {"approved": bool}.
- Jobs tab:   GET /jobs (+?session_id/?status), GET /jobs/{id} (plan/DAG,
  step timestamps, artifacts), POST /jobs (direct launch, bypassing chat),
  POST /jobs/{id}/cancel.
- Outputs:    GET /jobs/{id}/outputs — the files the job produced for the
  human; /outputs/{name} downloads one; /report is a shortcut to the main one.
- Live:       GET /events — SSE stream of job-progress events
  (in-process pub/sub, same v1 scope as cancellation).

Domain-agnostic: the domain arrives entirely through the injected service,
which was composed from an agent definition by `build_app`. Sessions are
rebuildable by id, so a conversation resumes after a restart.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..jobs.models import JobStatus
from ..service import LocalAgentService


class SessionIn(BaseModel):
    session_id: str | None = None   # resume an existing conversation


class MessageIn(BaseModel):
    text: str


class ApprovalIn(BaseModel):
    approved: bool


class JobIn(BaseModel):
    query: str
    inputs: dict[str, Any] | None = None
    session_id: str | None = None


def create_api(service: LocalAgentService) -> FastAPI:
    app = FastAPI(title="jobsmith", version="0.1.0")

    async def _job_or_404(job_id: str) -> dict:
        job = await service.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job: {job_id}")
        return job

    # ---------------- chat ----------------

    @app.get("/health")
    async def health() -> dict:
        """Probe used by the CLI to decide between daemon and embedded mode."""
        return {"status": "ok", "service": "jobsmith", "version": app.version}

    @app.post("/sessions", status_code=201)
    async def create_session(body: SessionIn | None = None) -> dict:
        return {"session_id": await service.new_session(body.session_id if body else None)}

    @app.post("/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageIn) -> dict:
        return await service.send(session_id, body.text)

    @app.post("/sessions/{session_id}/approval")
    async def post_approval(session_id: str, body: ApprovalIn) -> dict:
        return await service.approve(session_id, body.approved)

    # ---------------- jobs ----------------

    @app.get("/jobs")
    async def list_jobs(session_id: str | None = None, status: JobStatus | None = None):
        return await service.list_jobs(status=status.value if status else None,
                                       session_id=session_id)

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return await _job_or_404(job_id)

    @app.post("/jobs", status_code=201)
    async def launch_job(body: JobIn) -> dict:
        return await service.launch_job(body.query, session_id=body.session_id,
                                        inputs=body.inputs)

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        await _job_or_404(job_id)
        return await service.cancel_job(job_id)

    @app.get("/jobs/{job_id}/outputs")
    async def list_outputs(job_id: str):
        """Everything the job produced for the human (deliverable + annexes)."""
        await _job_or_404(job_id)
        return await service.list_outputs(job_id)

    @app.get("/jobs/{job_id}/outputs/{name}")
    async def download_output(job_id: str, name: str):
        await _job_or_404(job_id)
        path = await service.find_output(job_id, name)
        if path is None:
            raise HTTPException(404, f"no output {name!r} for job {job_id}")
        return FileResponse(path, filename=name)

    @app.get("/jobs/{job_id}/report")
    async def get_report(job_id: str) -> PlainTextResponse:
        """The main deliverable, inline (shortcut over /outputs)."""
        await _job_or_404(job_id)
        report = await service.get_report(job_id)
        if report is None:
            raise HTTPException(404, "no report for this job (not DONE yet?)")
        return PlainTextResponse(report, media_type="text/markdown")

    # ---------------- live events ----------------

    @app.get("/events")
    async def events() -> StreamingResponse:
        async def stream():
            queue = service.subscribe()
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                service.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
