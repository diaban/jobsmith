"""LangChain tools wrapping the JobManager use-cases for the chat agent.

`launch_job` is human-in-the-loop: it `interrupt()`s with the agent's
rationale before anything runs; the session owner resumes the graph with
`Command(resume={"approved": bool})`. The other tools are read/cancel
operations scoped to the chat session's own jobs.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt

from ..jobs.manager import JobManager
from ..jobs.models import Job


def _line(job: Job) -> str:
    return f"{job.job_id[:8]} [{job.status.value}] {job.query[:60]!r}"


async def _find(manager: JobManager, session_id: str, prefix: str) -> Job | None:
    """Resolve a job-id prefix among THIS session's jobs only."""
    jobs = await manager.list_jobs(session_id=session_id, limit=100)
    matches = [j for j in jobs if j.job_id.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def make_job_tools(manager: JobManager, session_id: str) -> list[Any]:
    """Build the job tools bound to one manager + one chat session."""

    @tool
    async def launch_job(query: str, rationale: str, inputs: dict[str, Any] | None = None) -> str:
        """Launch a complex task as a BACKGROUND job (research, analysis,
        anything needing several capability steps). `query` is the task for the
        job engine; `rationale` explains to the user why a job is needed and
        what it will do — the user must approve before it starts. Returns
        immediately; the conversation continues while the job runs.
        """
        decision = interrupt({
            "action": "launch_job",
            "query": query,
            "rationale": rationale,
        })
        if not (isinstance(decision, dict) and decision.get("approved")):
            return "The user DECLINED the launch. Do not launch this job; continue the conversation."
        job = await manager.create_job(query, inputs, session_id=session_id)
        manager.start_job(job.job_id)
        return (
            f"Job {job.job_id} launched in the background (short id {job.job_id[:8]}). "
            "Tell the user; you will be notified here when it finishes."
        )

    @tool
    async def job_status(job_id_prefix: str) -> str:
        """Get the status and progress (finished steps) of one of this
        session's jobs, by id prefix."""
        job = await _find(manager, session_id, job_id_prefix)
        if job is None:
            return f"No unique job of this session matches prefix {job_id_prefix!r}."
        parts = [_line(job)]
        if job.plan:
            for step in job.plan["steps"]:
                name = step["capability"]
                mark = "done" if name in job.step_finished_at else "pending"
                parts.append(f"  - {name}: {mark}")
        if job.report_path:
            parts.append(f"report: {job.report_path}")
        if job.error:
            parts.append(f"error: {job.error}")
        return "\n".join(parts)

    @tool
    async def list_my_jobs() -> str:
        """List this session's jobs (id, status, query)."""
        jobs = await manager.list_jobs(session_id=session_id, limit=50)
        return "\n".join(_line(j) for j in jobs) or "No jobs in this session yet."

    @tool
    async def cancel_job(job_id_prefix: str) -> str:
        """Cancel one of this session's running or queued jobs, by id prefix."""
        job = await _find(manager, session_id, job_id_prefix)
        if job is None:
            return f"No unique job of this session matches prefix {job_id_prefix!r}."
        cancelled = await manager.cancel_job(job.job_id)
        return f"Job {job.job_id[:8]} is now {cancelled.status.value}."

    return [launch_job, job_status, list_my_jobs, cancel_job]
