"""FastAPI app factory — the HTTP shape of the future chat + jobs UI.

    create_api(manager, session_factory) -> FastAPI

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
  (JobManager.subscribe; in-process pub/sub, same v1 scope as cancellation).

Domain-agnostic: the domain arrives entirely through the injected manager and
session factory (a runnable composition ships with the bundled example). The
factory must accept an optional session_id, so a conversation can be resumed
by id after a restart.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel

from ..chat.session import ChatSession
from ..jobs.manager import JobManager
from ..jobs.models import JobStatus


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


class _SessionEntry:
    def __init__(self, session: ChatSession):
        self.session = session
        self.agent = session.build()
        self.config: RunnableConfig = {"configurable": {"thread_id": session.session_id}}


def _shape_reply(result: dict) -> dict:
    """Chat result → API reply: a plain message, or a job proposal to approve."""
    if "__interrupt__" in result:
        proposal = result["__interrupt__"][0].value
        return {
            "type": "proposal",
            "query": proposal.get("query"),
            "rationale": proposal.get("rationale"),
        }
    return {"type": "message", "content": result["messages"][-1].content}


def create_api(manager: JobManager, session_factory: Callable[..., ChatSession]) -> FastAPI:
    app = FastAPI(title="agent_oo", version="0.1.0")
    sessions: dict[str, _SessionEntry] = {}

    def _entry(session_id: str) -> _SessionEntry:
        """Sessions are rebuildable: the registry is just a cache, the actual
        conversation lives in the checkpointer under thread_id=session_id. So a
        client can keep chatting on its session id across a daemon restart."""
        entry = sessions.get(session_id)
        if entry is None:
            entry = _SessionEntry(session_factory(session_id))
            sessions[session_id] = entry
        return entry

    async def _job_or_404(job_id: str):
        job = await manager.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job: {job_id}")
        return job

    # ---------------- chat ----------------

    @app.get("/health")
    async def health() -> dict:
        """Probe used by the CLI to decide between daemon and embedded mode."""
        return {"status": "ok", "service": "agent_oo", "version": app.version}

    @app.post("/sessions", status_code=201)
    async def create_session(body: SessionIn | None = None) -> dict:
        entry = _SessionEntry(session_factory(body.session_id) if body and body.session_id
                              else session_factory())
        sessions[entry.session.session_id] = entry
        return {"session_id": entry.session.session_id}

    @app.post("/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageIn) -> dict:
        entry = _entry(session_id)
        result = await entry.agent.ainvoke(
            {"messages": [HumanMessage(body.text)]}, entry.config
        )
        return _shape_reply(result)

    @app.post("/sessions/{session_id}/approval")
    async def post_approval(session_id: str, body: ApprovalIn) -> dict:
        entry = _entry(session_id)
        result = await entry.agent.ainvoke(
            Command(resume={"approved": body.approved}), entry.config
        )
        return _shape_reply(result)

    # ---------------- jobs ----------------

    @app.get("/jobs")
    async def list_jobs(session_id: str | None = None, status: JobStatus | None = None):
        jobs = await manager.list_jobs(session_id=session_id, status=status, limit=100)
        return [j.summary() | {"job_id": j.job_id} for j in jobs]

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return (await _job_or_404(job_id)).to_dict()

    @app.post("/jobs", status_code=201)
    async def launch_job(body: JobIn) -> dict:
        job = await manager.create_job(body.query, body.inputs, session_id=body.session_id)
        manager.start_job(job.job_id)
        return {"job_id": job.job_id, "status": job.status.value}

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = await _job_or_404(job_id)
        cancelled = await manager.cancel_job(job.job_id) or job
        return {"job_id": job.job_id, "status": cancelled.status.value}

    @app.get("/jobs/{job_id}/outputs")
    async def list_outputs(job_id: str):
        """Everything the job produced for the human (deliverable + annexes)."""
        job = await _job_or_404(job_id)
        return [dataclasses.asdict(o) | {"name": o.name} for o in job.outputs]

    @app.get("/jobs/{job_id}/outputs/{name}")
    async def download_output(job_id: str, name: str):
        job = await _job_or_404(job_id)
        output = next((o for o in job.outputs if o.name == name), None)
        if output is None or not Path(output.path).is_file():
            raise HTTPException(404, f"no output {name!r} for job {job_id}")
        return FileResponse(output.path, filename=name)

    @app.get("/jobs/{job_id}/report")
    async def get_report(job_id: str) -> PlainTextResponse:
        """The main deliverable, inline (shortcut over /outputs)."""
        job = await _job_or_404(job_id)
        if not job.report_path or not Path(job.report_path).is_file():
            raise HTTPException(404, "no report for this job (not DONE yet?)")
        return PlainTextResponse(
            Path(job.report_path).read_text(encoding="utf-8"), media_type="text/markdown"
        )

    # ---------------- live events ----------------

    @app.get("/events")
    async def events() -> StreamingResponse:
        async def stream():
            queue = manager.subscribe()
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                manager.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
