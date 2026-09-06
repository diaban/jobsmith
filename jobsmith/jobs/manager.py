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

Resume semantics: a stopped job kept its checkpoint, so `resume_job` re-enters
the thread instead of paying for the whole plan again. Only the steps that had
not finished run; the ones already in the store are kept as they are, and the
run settles through the same persistence, events and reporting path as a first
attempt. What is *not* here: re-running part of the DAG of a job that already
finished — that needs a way to say which results are stale, and is its own
feature.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

from ..core.artifacts import artifact_refs
from ..core.state import NodeError
from ..core.usage import Usage, UsageLedger, current_ledger, usage_ledger
from .events import InProcessEvents, JobEvents, job_event
from .models import Job, JobOutput, JobStatus, now_iso
from .report import MarkdownReport, ReportWriteError
from .repository import JobRepository, StoreJobRepository
from .runner import GraphRunner, JobUpdate, NodeErrors, PlanReady, StepFinished, Terminal

# Statuses a job can be resumed from — see `JobManager._begin_resume`.
RESUMABLE = (JobStatus.CANCELLED, JobStatus.FAILED)

# Ledger scope carrying what previous attempts of a resumed job already spent.
EARLIER_ATTEMPTS = "earlier attempts"


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
        # Inside `run_job` a usage ledger is installed for this run, so every
        # persist (each finished step, and the terminal one) carries the spend
        # so far — a job that is still running already shows what it has cost.
        ledger = current_ledger()
        if ledger is not None:
            job.usage = ledger.total().to_dict()
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
        job = await self._require(job_id)
        if job.status is not JobStatus.QUEUED:
            raise ValueError(f"job {job_id} is {job.status.value}, expected queued")
        await self._begin(job)
        return await self._drive(job, self.runner.stream(job.job_id, job.query, job.inputs))

    async def resume_job(self, job_id: str) -> Job:
        """Re-enter a stopped job's checkpoint and run it to completion.

        See `_begin_resume` for what may be resumed and why.
        """
        job = await self._begin_resume(job_id)
        return await self._drive(job, self.runner.resume(job.job_id), resumed=True)

    async def _require(self, job_id: str) -> Job:
        job = await self.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        return job

    async def _begin(self, job: Job) -> None:
        """Mark the job RUNNING before anything is driven, so a caller that
        starts it in the background already sees the new status."""
        job.status = JobStatus.RUNNING
        await self._persist_summary(job)

    async def _begin_resume(self, job_id: str) -> Job:
        """Check that this job can be resumed, and open the attempt.

        Resumable = **stopped with work left to do**, which is exactly two
        cases, both of which kept their checkpoint:

        - CANCELLED — `cancel_job` interrupted a run mid-capability;
        - FAILED after `recover_interrupted()` — the process died mid-run.

        The status alone is not enough: a job that FAILED *because a node
        raised* reached a terminal node (`escalate`/`user_error`), so its
        thread has nothing left to run. Re-entering it would replay the last
        superstep and yield nothing — a silent no-op that would look like a
        successful resume. The runner is asked instead, and an empty
        `pending()` is refused out loud. Same for a job cancelled before it
        ever started: it has no checkpoint, and `run_job` is what it needs.

        DONE is deliberately not resumable — pushing a finished job further is
        a different feature (re-running part of the DAG), not this one.
        """
        job = await self._require(job_id)
        if job.status not in RESUMABLE:
            raise ValueError(
                f"job {job_id} is {job.status.value}, expected "
                f"{' or '.join(s.value for s in RESUMABLE)}"
            )
        if not await self.runner.pending(job_id):
            raise ValueError(
                f"job {job_id} has no checkpoint to resume from: it either never "
                f"started or already reached its last step"
            )
        job.error = None          # the stopped attempt's message is stale now
        await self._begin(job)
        return job

    async def _drive(self, job: Job, updates: Any, *, resumed: bool = False) -> Job:
        """Fold a run's updates into the job, and settle it.

        The single place a run is driven, whichever way it was entered: a
        resumed attempt therefore persists, reports and emits events exactly
        like a first one.
        """
        errors: list[NodeError] = []
        # One ledger per run — a fresh one, so a job launched from inside
        # another run can never bill its parent. Every LLM call underneath
        # books into it, attributed to the graph step that made it.
        ledger = UsageLedger()
        if resumed and job.usage:
            # A resume is a second attempt at ONE job, and the tokens the
            # first attempt burned are just as spent. Seeding the ledger keeps
            # `job.usage` the job's total cost rather than the last attempt's;
            # the per-step breakdown stays in each result's own `meta`.
            ledger.add(EARLIER_ATTEMPTS, Usage.from_dict(job.usage))
        with usage_ledger(ledger):
            try:
                async for update in updates:
                    await self._apply(job, update, errors)
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                self._collect_artifacts(job)       # the steps that did finish left files
                await self._persist_summary(job)   # cancelled work was still paid for
                raise
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                self._collect_artifacts(job)
                await self._persist_summary(job)
                return job

            if errors:
                await self.repo.save_errors(job.job_id, errors)
            job.status = JobStatus.DONE if job.terminal_kind == "answer" else JobStatus.FAILED
            if job.status is JobStatus.DONE:
                # The reporter reads job.usage, so settle it before writing.
                job.usage = ledger.total().to_dict()
                self._write_outputs(job)
            else:
                self._collect_artifacts(job)
            await self._persist_summary(job)
        return job

    def _write_outputs(self, job: Job) -> None:
        """Produce the deliverables of a job that answered — and survive failing to.

        Whatever the reporter hands back IS the job's deliverables: one
        Reporter writes one file, a composed one writes several. The files a
        capability produced for itself are collected next to them, so
        `Job.outputs` is the whole of what this job leaves behind.

        A write that raises must NOT escape: this runs after the stream's
        own `try`, so an exception here would skip the final persist and
        leave the store holding the RUNNING row the last finished step
        wrote — a job with an answer that no caller can ever reach, until
        some later process start settles it as interrupted.

        The job therefore stays DONE: the graph answered, the answer is
        persisted, only the file failed — and `job.error` says which format
        and why. FAILED would misreport the work *and* be a dead end, since
        it is resumable in name only (the graph ran to completion, so the
        checkpoint has nothing pending and `resume_job` would refuse it).
        A separate `try` on purpose: the run's own `except` means "the graph
        blew up", which a full disk is not.

        **Only a job that answered gets here.** Running the Reporter for a
        run that stopped would write a report of nothing; `_collect_artifacts`
        is the half that still applies, and it is called on its own there.
        """
        try:
            outputs = list(self.reporter.write(job, self.reports_dir))
        except Exception as e:
            failure = e if isinstance(e, ReportWriteError) else ReportWriteError(
                getattr(self.reporter, "format", "unknown"), e)
            outputs = failure.outputs       # keep what did make it to disk
            job.error = str(failure)
        # Annexes are collected regardless of how the report went: they are on
        # disk either way, and a file with no JobOutput is a file nobody can
        # find — the same reason `ReportWriteError` carries its outputs.
        self._collect_artifacts(job, deliverables=outputs, answered=True)

    def _collect_artifacts(
        self,
        job: Job,
        *,
        deliverables: list[JobOutput] | None = None,
        answered: bool = False,
    ) -> None:
        """Record the files the steps left behind, as this job's annexes.

        Called for **every** terminal a run reaches, not only for an answer.
        A chart a step produced exists whether or not the run that followed
        it reached a conclusion, and this project has twice already chosen to
        keep the evidence over discarding it: usage is booked for a step that
        failed, and `ReportWriteError` carries the outputs already on disk. A
        file recorded nowhere is "a deliverable nobody can find", which is the
        defect #28 fixed — and for a FAILED job it is usually permanent, since
        a run that reached `escalate`/`user_error` has an empty frontier and
        `resume_job` refuses it.

        Annexes come AFTER the deliverables and are never "main": #28's
        invariant (exactly one main, the first format asked for) is what
        `report_path`, `jobsmith report` and `/report` read, and a step's
        chart must not become the thing the report points at. A job that did
        not answer passes no deliverables at all, so it has no `main`,
        `report_path` stays None and `/report` still 404s — the annexes are
        offered as what they are, not dressed up as a partial success.

        `job.outputs` is **assigned**, never appended to: a cancelled job that
        collects and is then resumed keeps the earlier attempt's results (the
        repository loads them), so the second collection re-derives the same
        list rather than doubling it.
        """
        annexes, missing = self._capability_outputs(job)
        job.outputs = list(deliverables or []) + annexes
        # A promised file that is absent is only a *defect* when the run
        # answered — see `_capability_outputs`. On a run that stopped it is an
        # expected consequence of stopping, and `job.error` has to keep saying
        # why the run stopped, which is what the human actually needs.
        if missing and answered:
            job.error = "; ".join(filter(None, [job.error, missing]))

    def _capability_outputs(self, job: Job) -> tuple[list[JobOutput], str]:
        """The files the steps produced, as annexes — plus what went missing.

        A capability declares what it wrote in its result's `meta`
        (`core/artifacts.py`); this reads those declarations back, in plan
        order so two runs of one plan list their files the same way.

        **A ref whose file is not there is dropped, and said out loud.**
        Recording it would repeat exactly the defect #28 fixed — a JobOutput
        for a file nobody can open, offered by `jobsmith outputs` and by
        `GET /jobs/{id}/outputs/{name}` and failing there instead of here.
        Staying silent is no better *for a job that answered*: a promised file
        that is absent is then a capability defect, and the run that hid it
        would look perfect. So the drop is reported back and `_collect_artifacts`
        decides what to do with it — into `job.error` when the run answered
        (the same channel, and the same reasoning, as a failed report write:
        the job stays DONE because the work is done, and the error says which
        part of the delivery is not), silently when the run stopped, where a
        half-written file is a consequence of stopping rather than a defect.

        Two capabilities that wrote the same path are recorded once: the
        second would list one file twice in `Job.outputs`, which is the same
        ambiguity `compose_reporters` refuses at composition time.
        """
        outputs: list[JobOutput] = []
        seen: set[str] = set()
        gone: list[str] = []
        for name, result in job.ordered_results():
            for ref in artifact_refs(result.get("meta")):
                if ref.path in seen:
                    continue
                seen.add(ref.path)
                if not Path(ref.path).is_file():
                    gone.append(f"{name} → {ref.path}")
                    continue
                outputs.append(JobOutput(
                    path=ref.path, format=ref.file_format, title=ref.title,
                    role="annex", produced_by=name,
                ))
        missing = (
            f"{len(gone)} file(s) a step reported producing are missing: "
            f"{', '.join(gone)}" if gone else ""
        )
        return outputs, missing

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
        return self._background(job_id, self.run_job(job_id))

    async def start_resume(self, job_id: str) -> Job:
        """Resume in a background task, and return the job now RUNNING.

        Async where `start_job` is sync, on purpose: a resume can be *refused*
        (wrong status, no checkpoint), and that refusal has to reach the
        caller rather than die inside a background task nobody awaits. So the
        checks run here, and only the driving is backgrounded.
        """
        job = await self._begin_resume(job_id)
        self._background(job.job_id, self._drive(job, self.runner.resume(job.job_id),
                                                 resumed=True))
        return job

    def _background(self, job_id: str, coro: Any) -> asyncio.Task:
        """Run a job's coroutine as a task this manager can cancel."""
        task = asyncio.create_task(coro, name=f"job:{job_id}")
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

        The fourth terminal, and the one that does NOT collect artifacts
        (`_collect_artifacts`), for two reasons that point the same way. It
        works from index summaries, which carry neither the plan nor the
        results — collecting would mean re-loading every stale record in full
        at startup to read declarations this process never saw. And it is the
        one FAILED that keeps a live frontier: the checkpoint is retained on
        purpose, so a resume settles through `_drive`, which collects. Nothing
        is lost here that the next attempt cannot record.
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
