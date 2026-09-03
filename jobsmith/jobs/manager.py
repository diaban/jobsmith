"""JobManager: the job use cases.

Everything the product can *do* with a job lives here — create, run, track,
cancel, recover, announce. Everything else is delegated to a collaborator, so
each of them can change (or be swapped) for its own reasons:

    JobRepository   where records live and what the schema is  (repository.py)
    GraphRunner     how a run is driven and read back          (runner.py)
    JobEvents       how progress is broadcast                  (events.py)
    Reporter        how the deliverable is produced            (report.py)

The defaults wire the v1 stack (LangGraph store, LangGraph graph, in-process
events, markdown report), so `JobManager(graph, store)` still works.

Cancellation semantics: `cancel_job` cancels the in-process asyncio.Task;
cancellation propagates into the running invocation, the checkpointer retains
the last completed superstep, and the job is marked CANCELLED. With no task
registered in this process (another process, or already finished), a CANCELLED
tombstone is written best-effort — true cross-process preemption is out of
scope for v1.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

from ..core.state import NodeError
from .events import InProcessEvents, JobEvents, job_event
from .models import Job, JobStatus, now_iso
from .report import MarkdownReport
from .repository import JobRepository, StoreJobRepository
from .runner import GraphRunner, JobUpdate, NodeErrors, PlanReady, StepFinished, Terminal


class JobManager:
    def __init__(
        self,
        graph: Any = None,
        store: Any = None,
        *,
        reporter: Any = None,
        reports_dir: str | Path = "artifacts",
        repository: JobRepository | None = None,
        runner: GraphRunner | None = None,
        events: JobEvents | None = None,
    ):
        if repository is None and store is None:
            raise ValueError("JobManager needs a store or an explicit repository")
        if runner is None and graph is None:
            raise ValueError("JobManager needs a graph or an explicit runner")
        self.graph = graph
        self.repo: JobRepository = repository or StoreJobRepository(store)
        self.runner: GraphRunner = runner or GraphRunner(graph)
        self.events: JobEvents = events or InProcessEvents()
        # Producing the deliverable is a rendering concern, not the manager's:
        # swap the Reporter for another format without touching this class.
        self.reporter = reporter if reporter is not None else MarkdownReport()
        self.reports_dir = Path(reports_dir)  # where deliverables are written
        self._tasks: dict[str, asyncio.Task] = {}  # in-process cancellation handles

    async def _persist_summary(self, job: Job) -> None:
        job.updated_at = now_iso()
        await self.repo.save_summary(job)
        self.events.publish(job_event(job))

    # ---------------- Lifecycle ----------------

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
            created_at=now_iso(),
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
            async for update in self.runner.stream(job.job_id, job.query, job.inputs):
                await self._apply(job, update, errors)
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
            await self.repo.save_errors(job.job_id, errors)
        job.status = JobStatus.DONE if job.terminal_kind == "answer" else JobStatus.FAILED
        if job.status is JobStatus.DONE:
            job.outputs = [self.reporter.write(job, self.reports_dir)]
        await self._persist_summary(job)
        return job

    async def _apply(self, job: Job, update: JobUpdate, errors: list[NodeError]) -> None:
        """Fold one domain update from the runner into the job."""
        match update:
            case NodeErrors(node_errors):
                errors.extend(node_errors)
            case PlanReady(plan):
                job.plan = plan
                await self.repo.save_plan(job.job_id, plan)
            case StepFinished(capability, result):
                job.results[capability] = result
                job.step_finished_at[capability] = now_iso()
                await self.repo.save_result(job.job_id, capability, result)
                await self._persist_summary(job)   # touch updated_at for progress
            case Terminal(terminal_kind, final_answer, user_error_message):
                job.terminal_kind = terminal_kind
                job.final_answer = final_answer
                if terminal_kind != "answer":
                    job.error = user_error_message

    def start_job(self, job_id: str) -> asyncio.Task:
        """Fire-and-forget: run the job in a background task (cancellable)."""
        task = asyncio.create_task(self.run_job(job_id), name=f"job:{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return task

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

    # ---------------- Queries ----------------

    async def get_job(self, job_id: str) -> Job | None:
        return await self.repo.load(job_id)

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        jobs = await self.repo.load_all(limit=limit)
        if status is not None:
            jobs = [j for j in jobs if j.status is status]
        if session_id is not None:
            jobs = [j for j in jobs if j.session_id == session_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def recover_interrupted(self) -> list[Job]:
        """Settle jobs left RUNNING by a process that died (persistent stores).

        Call once at startup, before any job runs: with no in-process task
        alive, a RUNNING record can only be a leftover. They are marked FAILED
        while their checkpoint is retained, so a resume stays possible later.
        QUEUED jobs are left alone — they never started and can still be run.
        """
        stale = [j for j in await self.list_jobs(status=JobStatus.RUNNING, limit=1000)
                 if j.job_id not in self._tasks]
        for job in stale:
            job.status = JobStatus.FAILED
            job.error = "interrupted: the process running this job stopped"
            await self._persist_summary(job)
        if stale:
            print(f"[jobs: {len(stale)} interrupted job(s) marked failed on startup]",
                  file=sys.stderr)
        return stale

    # ---------------- Live events ----------------

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        """Get a queue of job-progress events (every summary persist emits one)."""
        return self.events.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.events.unsubscribe(queue)

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
