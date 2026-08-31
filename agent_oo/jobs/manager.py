"""JobManager: persistent, trackable, cancellable orchestration runs.

Store schema (LangGraph BaseStore namespaces):

| namespace                     | key        | value                                  |
|-------------------------------|------------|----------------------------------------|
| ("jobs", "index")             | job_id     | summary record (status, query, ...)    |
| ("jobs", job_id, "meta")      | "plan"     | validated plan + rationale             |
| ("jobs", job_id, "meta")      | "errors"   | accumulated NodeError list             |
| ("jobs", job_id, "artifacts") | cap name   | that capability's CapabilityResult     |

Fine-grained execution state additionally lives in the *checkpointer* under
thread_id == job_id, so a paused/cancelled job can later be resumed on the
same thread.

Status updates happen via streaming, not node instrumentation: `run_job`
drives `graph.astream(stream_mode="updates")` and persists as node updates
arrive — the graph nodes stay job-agnostic.

Cancellation semantics: `cancel_job` cancels the in-process asyncio.Task;
cancellation propagates into the running LangGraph invocation, the
checkpointer retains the last completed superstep, and the job is marked
CANCELLED. If no task is registered in this process (other process, or
already finished), a CANCELLED tombstone is written best-effort — true
cross-process preemption is out of scope for v1.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.state import NodeError
from .models import Job, JobStatus

_TERMINAL_NODES = ("post_process", "escalate", "user_error")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobManager:
    def __init__(self, graph: Any, store: Any, *, reports_dir: str | Path = "artifacts"):
        self.graph = graph
        self.store = store
        self.reports_dir = Path(reports_dir)  # markdown reports of finished jobs
        self._tasks: dict[str, asyncio.Task] = {}  # in-process cancellation handles

    # ---------------- Persistence helpers ----------------

    async def _persist_summary(self, job: Job) -> None:
        job.updated_at = _now()
        await self.store.aput(("jobs", "index"), job.job_id, job.summary())

    async def _persist_meta(self, job_id: str, key: str, value: Any) -> None:
        await self.store.aput(("jobs", job_id, "meta"), key, value)

    async def _persist_artifact(self, job_id: str, cap_name: str, result: dict) -> None:
        await self.store.aput(("jobs", job_id, "artifacts"), cap_name, result)

    # ---------------- API ----------------

    async def create_job(
        self,
        query: str,
        inputs: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            query=query,
            inputs=inputs or {},
            session_id=session_id,
            created_at=_now(),
        )
        await self._persist_summary(job)
        return job

    async def run_job(self, job_id: str) -> Job:
        """Run a QUEUED job to completion, persisting progress as it streams."""
        job = await self.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        if job.status is not JobStatus.QUEUED:
            raise ValueError(f"job {job_id} is {job.status.value}, expected queued")

        job.status = JobStatus.RUNNING
        await self._persist_summary(job)

        errors: list[NodeError] = []
        try:
            async for update in self.graph.astream(
                {"query": job.query, "inputs": job.inputs, "job_id": job.job_id},
                config={"configurable": {"thread_id": job.job_id}},
                stream_mode="updates",
            ):
                await self._apply_update(job, update, errors)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            await self._persist_summary(job)
            raise
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            await self._persist_summary(job)
            return job

        if errors:
            await self._persist_meta(job.job_id, "errors", list(errors))
        job.status = JobStatus.DONE if job.terminal_kind == "answer" else JobStatus.FAILED
        if job.status is JobStatus.DONE:
            job.report_path = str(self._write_report(job))
        await self._persist_summary(job)
        return job

    def start_job(self, job_id: str) -> asyncio.Task:
        """Fire-and-forget: run the job in a background task (cancellable)."""
        task = asyncio.create_task(self.run_job(job_id), name=f"job:{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return task

    @staticmethod
    def _job_from_summary(job_id: str, s: dict[str, Any]) -> Job:
        return Job(
            job_id=job_id,
            status=JobStatus(s["status"]),
            query=s["query"],
            inputs=s.get("inputs") or {},
            session_id=s.get("session_id"),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            step_finished_at=s.get("step_finished_at") or {},
            terminal_kind=s.get("terminal_kind"),
            final_answer=s.get("final_answer"),
            error=s.get("error"),
            report_path=s.get("report_path"),
            announced=bool(s.get("announced")),
        )

    async def get_job(self, job_id: str) -> Job | None:
        item = await self.store.aget(("jobs", "index"), job_id)
        if item is None:
            return None
        job = self._job_from_summary(job_id, item.value)
        plan_item = await self.store.aget(("jobs", job_id, "meta"), "plan")
        if plan_item is not None:
            job.plan = plan_item.value
        for artifact in await self.store.asearch(("jobs", job_id, "artifacts"), limit=100):
            job.results[artifact.key] = artifact.value
        return job

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        items = await self.store.asearch(("jobs", "index"), limit=limit)
        jobs = [self._job_from_summary(item.key, item.value) for item in items]
        if status is not None:
            jobs = [j for j in jobs if j.status is status]
        if session_id is not None:
            jobs = [j for j in jobs if j.session_id == session_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    # ---------------- Chat-session support ----------------

    async def list_finished_unannounced(self, session_id: str) -> list[Job]:
        """Finished jobs of a session whose completion was not yet surfaced
        in its conversation (the chat layer announces, then marks them)."""
        jobs = await self.list_jobs(session_id=session_id, limit=100)
        return [
            j for j in jobs
            if j.status in (JobStatus.DONE, JobStatus.FAILED) and not j.announced
        ]

    async def mark_announced(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if job is not None and not job.announced:
            job.announced = True
            await self._persist_summary(job)

    async def cancel_job(self, job_id: str) -> Job | None:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return await self.get_job(job_id)
        # No in-process task: best-effort tombstone (see module docstring).
        job = await self.get_job(job_id)
        if job is not None and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            await self._persist_summary(job)
        return job

    # ---------------- Markdown report ----------------

    def _write_report(self, job: Job) -> Path:
        """Write the job's markdown report to reports_dir/<job_id>.md."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{job.job_id}.md"
        path.write_text(self._render_report(job), encoding="utf-8")
        return path

    def _render_report(self, job: Job) -> str:
        lines = [
            f"# Job report — {job.query[:80]}",
            "",
            f"- **Job**: `{job.job_id}`",
            f"- **Created**: {job.created_at}",
            f"- **Finished**: {_now()}",
        ]
        if job.session_id:
            lines.append(f"- **Session**: `{job.session_id}`")
        lines += ["", "## Answer", "", job.final_answer or "_(no answer)_"]

        if job.plan and job.plan.get("steps"):
            steps = job.plan["steps"]
            lines += ["", "## Plan", ""]
            if job.plan.get("rationale"):
                lines += [f"_{job.plan['rationale']}_", ""]
            lines += ["| step | depends on | status | finished at |", "|---|---|---|---|"]
            for s in steps:
                name = s["capability"]
                result = job.results.get(name)
                status = "ok" if result and result.get("ok") else (
                    f"failed ({result.get('error')})" if result else "not run"
                )
                lines.append(
                    f"| {name} | {', '.join(s['depends_on']) or '—'} | {status} "
                    f"| {job.step_finished_at.get(name, '—')} |"
                )
            lines += ["", "```mermaid", "flowchart LR"]
            for s in steps:
                if s["depends_on"]:
                    lines += [f"  {dep} --> {s['capability']}" for dep in s["depends_on"]]
                else:
                    lines.append(f"  {s['capability']}")
            lines += ["```"]

        if job.results:
            lines += ["", "## Capability artifacts"]
            order = [s["capability"] for s in (job.plan or {}).get("steps", [])]
            for name in sorted(job.results, key=lambda n: order.index(n) if n in order else 99):
                result = job.results[name]
                lines += [
                    "",
                    f"### {name} — {'ok' if result.get('ok') else 'failed'}",
                    "",
                    "```json",
                    json.dumps(result.get("data") or {"error": result.get("error")},
                               indent=2, ensure_ascii=False),
                    "```",
                ]
        return "\n".join(lines) + "\n"

    # ---------------- Streaming updates ----------------

    async def _apply_update(
        self, job: Job, update: dict[str, Any], errors: list[NodeError]
    ) -> None:
        """React to one astream(stream_mode='updates') event: {node: update}."""
        for node, val in update.items():
            if not isinstance(val, dict):
                continue
            errors.extend(val.get("errors") or [])
            if node == "planner" and val.get("plan"):
                job.plan = val["plan"]
                await self._persist_meta(job.job_id, "plan", job.plan)
            elif node.startswith("cap_"):
                for cap_name, result in (val.get("results") or {}).items():
                    job.results[cap_name] = result
                    job.step_finished_at[cap_name] = _now()
                    await self._persist_artifact(job.job_id, cap_name, result)
                await self._persist_summary(job)  # touch updated_at for progress
            elif node in _TERMINAL_NODES:
                job.terminal_kind = val.get("terminal_kind")
                job.final_answer = val.get("final_answer")
                if job.terminal_kind != "answer":
                    job.error = val.get("user_error_message")
