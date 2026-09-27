"""JobManager: the job use cases.

Everything the product can *do* with a job lives here — create, run, track,
cancel, recover, announce. Everything else is delegated to a collaborator, so
each of them can change (or be swapped) for its own reasons:

    JobRepository   where records live and what the schema is  (repository.py)
    GraphRunner     how a run is driven and read back          (runner.py)
    JobEvents       how progress is broadcast                  (events.py)

The defaults wire the v1 stack (LangGraph store, LangGraph graph, in-process
events), so `JobManager(graph, store)` still works. What a run produced —
its steps' results, its files — reaches the job as facts the graph published
(`engine/facts.py`), which the manager records without reading.

Cancellation semantics: `cancel_job` cancels the in-process asyncio.Task;
cancellation propagates into the running invocation, the checkpointer retains
the last completed superstep, and the job is marked CANCELLED. On a store
other processes share (#10) a job may be running in ANOTHER process, and then
the cancel is a request written to the store, which the owner reads on its
heartbeat and turns into that same task cancellation — so the job ends
CANCELLED through the same path, written by the process that actually stopped
it. If the owner is provably gone, the canceller settles the record itself.
On a process-local store there is no other process, and a job with no task
here gets a CANCELLED tombstone as it always did. See `ownership.py`.

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
from typing import Any

from ..dag.state import TERMINAL_UNANSWERED, NodeError
from .events import InProcessEvents, JobEvents, job_event
from .models import Job, JobStatus, now_iso
from .ownership import (
    Heartbeat,
    JobControl,
    LeasePolicy,
    ProcessIdentity,
    death_is_certain,
    owner_is_gone,
)
from .repository import JobRepository, StoreJobRepository
from .runner import Fact, GraphRunner, JobUpdate, NodeFinished, Output
from .usage import Usage, UsageLedger, current_ledger, usage_ledger

# Statuses a job can be resumed from — see `JobManager._begin_resume`.
RESUMABLE = (JobStatus.CANCELLED, JobStatus.FAILED)

# Statuses worth surfacing in the conversation that launched the job: every
# terminal one. CANCELLED belongs here because `cancel_job` is one of the
# tools the chat model holds — the same actor can stop a job and would
# otherwise say nothing about what it produced.
ANNOUNCEABLE = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

# The terminals of a run that did its work and has something to hand back: it
# answered, or it declared — as data, from the generator — that the material
# does not answer the request (#59). Both are DONE and both get a deliverable
# written, because nothing failed in either: the graph ran to the end, every
# step reported, the tokens were spent and the files are on disk. Which of the
# two it was is `terminal_kind`'s to say; making the *status* carry it would
# either call a refusal a crash (FAILED misreports the work, and is a dead end
# — the checkpoint has nothing pending, so `resume_job` refuses it) or add a
# sixth status every consumer would have to learn.
DELIVERED = ("answer", TERMINAL_UNANSWERED)

# Ledger scope carrying what previous attempts of a resumed job already spent.
EARLIER_ATTEMPTS = "earlier attempts"


class JobManager:
    def __init__(
        self,
        graph: Any = None,
        store: Any = None,
        *,
        repository: JobRepository | None = None,
        runner: GraphRunner | None = None,
        events: JobEvents | None = None,
        lease: LeasePolicy | None = None,
    ):
        if repository is None and store is None:
            raise ValueError("JobManager needs a store or an explicit repository")
        if runner is None and graph is None:
            raise ValueError("JobManager needs a graph or an explicit runner")
        self.graph = graph
        self.repo: JobRepository = repository or StoreJobRepository(store)
        self.runner: GraphRunner = runner or GraphRunner(graph)
        self.events: JobEvents = events or InProcessEvents()
        self._tasks: dict[str, asyncio.Task] = {}  # in-process cancellation handles
        # Who this manager is on the leases it writes, and the timings of
        # ownership (#10). Only consulted when the repository is `shared`:
        # a process-local store pays nothing for any of it.
        self.identity = ProcessIdentity.current()
        self.lease = lease or LeasePolicy()

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
        document_name: str = "",
        document_title: str = "",
        formats: list[str] | None = None,
    ) -> Job:
        """Record a job, QUEUED, as the caller already checked it (`dag/jobs.py`).

        `formats == []` — no file at all — is recorded as
        `deliverable_expected=False` here and now, because it is known here
        and now (#84); `None` is not yet a decision — "write me a report" may
        still be read out of the query — so it stays True until the graph's
        document step has read it (`_apply`, the `formats` fact).
        """
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            query=query,
            inputs=inputs or {},
            document_name=document_name,
            document_title=document_title,
            formats=formats,
            deliverable_expected=formats != [],
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
        if not await self._begin(job):
            return job
        return await self._drive(job, self.runner.stream(job.job_id, self._graph_input(job)))

    @staticmethod
    def _graph_input(job: Job) -> dict[str, Any]:
        """What the planner DAG starts from, built from the record it was asked
        with. `formats` is what the CALLER already asked the document to be:
        `None` — the request said nothing — is what lets the graph's document
        step decide; anything else silences it before a single model call
        (#90). The record's DAG fields leave at step 6b (docs/design/core-v1.md)."""
        return {"query": job.query, "inputs": job.inputs, "job_id": job.job_id,
                "document_formats": job.formats, "document_name": job.document_name,
                "document_title": job.document_title}

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

    async def _begin(self, job: Job) -> bool:
        """Mark the job RUNNING before anything is driven, so a caller that
        starts it in the background already sees the new status.

        On a shared store the lease is written FIRST: a record that says
        RUNNING with no owner is exactly what another process's startup
        settles as interrupted, and the gap between two writes is enough.
        A cancel already requested — by a process that saw this job QUEUED
        while this one was picking it up — is honoured before anything runs,
        and `False` says so.
        """
        if self.repo.shared:
            await self.repo.save_lease(job.job_id, self.identity.lease(self.lease.ttl))
            if (await self.repo.load_control(job.job_id)).cancel_requested_at:
                job.status = JobStatus.CANCELLED
                await self._persist_summary(job)
                await self.repo.release_lease(job.job_id)
                return False
        job.status = JobStatus.RUNNING
        await self._persist_summary(job)
        return True

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
        # ...and so is the fact that the stop was announced: a job picked back
        # up is news again. Without this, a cancelled job announced in its
        # session and then resumed to DONE is filtered out of
        # `list_finished_unannounced`, and its answer never reaches the
        # conversation that asked for it.
        job.announced = False
        # A cancel request is a message to the attempt it stopped; left in the
        # store, it would stop the resumed one on its first heartbeat.
        if self.repo.shared:
            await self.repo.clear_cancel(job_id)
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
        # On a shared store the run is watched for a stop requested from
        # another process, and its lease renewed (#10). None otherwise: a
        # process-local store has nobody to hear from.
        watch = self._watch(job)
        with usage_ledger(ledger):
            try:
                async for update in updates:
                    await self._apply(job, update, errors)
            except asyncio.CancelledError:
                if watch is not None:
                    watch.stop()        # before any await: nothing may cancel the settling
                    if watch.lost:
                        return await self._abandon(job, watch)
                job.status = JobStatus.CANCELLED
                await self._persist_summary(job)   # cancelled work was still paid for
                if watch is not None:
                    await self._release(job, watch)
                    # A stop asked for from another process is not this
                    # caller's cancellation: whoever awaits the run here (the
                    # chat's `launch_job`, `jobsmith run --wait`) gets the
                    # CANCELLED job back, as it would any other ending. A
                    # local cancel landing at the same time still propagates.
                    current = asyncio.current_task()
                    if watch.requested and current is not None and current.uncancel() == 0:
                        return job
                raise
            except Exception as e:
                if watch is not None:
                    watch.stop()
                job.status = JobStatus.FAILED
                job.error = str(e)
                await self._persist_summary(job)
                if watch is not None:
                    await self._release(job, watch)
                return job
            finally:
                # Synchronous, and reached with no `await` after the run's
                # last update: from here on nothing can cancel this run, so a
                # stop that arrives now is too late and the ending stands.
                if watch is not None:
                    watch.stop()

            if errors:
                await self.repo.save_errors(job.job_id, errors)
            job.status = JobStatus.DONE if job.terminal_kind in DELIVERED else JobStatus.FAILED
            if job.status is JobStatus.DONE and not self._deliverable_wanted(job):
                # It answered and nobody wanted a document of it: a decision,
                # recorded — the flag goes True → False and never back, so a
                # job that asked for no file does not silently re-promise one
                # by failing. Usually already recorded by now (`create_job`
                # for `[]`, the document step for silence); this is the
                # backstop for a graph with no document step at all.
                job.deliverable_expected = False
            await self._persist_summary(job)
            if watch is not None:
                await self._release(job, watch)
        return job

    def _watch(self, job: Job) -> Heartbeat | None:
        """Start watching this run on a shared store: renew its lease, and
        hear a cancel requested from another process. Cancels the task that
        drives the run — the same mechanism a local `cancel_job` uses."""
        task = asyncio.current_task()
        if not self.repo.shared or task is None:
            return None
        return Heartbeat(self.repo, job.job_id, self.identity, self.lease, task)

    async def _release(self, job: Job, watch: Heartbeat) -> None:
        """Give up the lease of a run that settled its own record."""
        await watch.wait_closed()
        await self.repo.release_lease(job.job_id)

    async def _abandon(self, job: Job, watch: Heartbeat) -> Job:
        """This run lost its lease: another process judged it dead and settled
        the job. Stop WITHOUT writing — the record is that process's now, and
        a write here would overwrite its settlement (and let a resume there
        race this run on one checkpoint). The caller gets the record as the
        store has it; this is not the caller's cancellation, so it does not
        propagate as one."""
        await watch.wait_closed()
        current = asyncio.current_task()
        if current is not None and current.uncancel() > 0:
            raise asyncio.CancelledError
        print(f"[jobs: {job.job_id[:8]} was settled by another process while "
              f"this one ran it; stopped without writing]", file=sys.stderr)
        return await self.get_job(job.job_id) or job

    @staticmethod
    def _deliverable_wanted(job: Job) -> bool:
        """Is this run meant to leave a document behind (#84, #96)?

        **Only if the request asked for one.** `formats` non-empty — named by
        the caller, or read out of the sentence by the graph's document step
        (#90), or asked for without a format and resolved to the deployment's
        default — is a file, whatever the run turned out to be: a reader who
        asked for a PDF gets one even if the router answered on the spot,
        since handing them nothing over how a triage step read their sentence
        would be a second silent decision. `[]` and `None` are no file.

        `None` used to be answered by the plan's shape — a run that planned
        and executed capabilities wrote the deployment's formats, one that
        answered direct did not (#84). That was deliberate while a run
        **promoted** to the background (#83) had no other place its answer
        survived word for word: the completion notice handed it to the chat
        model, which synthesised it. #85 gave the promoted run the same
        verbatim channel a synchronous one uses, and guaranteed that a run
        with no file is delivered whatever its length — which removed the
        only reason silence meant a file. So the rule #84 stated for the
        requests that spoke now holds for the silent ones too: **the request
        decides, not the plan, not the duration and not the door**. A plan's
        shape says how much work the answer took, which is not what anybody
        asked about a file.

        The answer is no harder to reach for it: it is on the record
        (`final_answer`, `GET /jobs/{id}`, `jobsmith job <id>`), `jobsmith run
        --wait` prints it, and the conversation carries it in full (#83, #85).
        """
        return bool(job.formats)

    async def _apply(self, job: Job, update: JobUpdate, errors: list[NodeError]) -> None:
        """Fold one update from the runner into the job.

        A fact is recorded as it arrives, whatever it says (`engine/facts.py`):
        persisted under its key, stamped with when it came, and announced — a
        fact is progress. A root node that finished is noted, and travels with
        the next write. What remains below of the planner DAG's words
        (`formats`, the keys of its output) leaves at step 6b.3 of the core
        split, with the record fields it fills (docs/design/core-v1.md).
        """
        match update:
            case Fact(key, value):
                job.facts[key] = value
                job.facts_at[key] = now_iso()
                await self.repo.save_fact(job.job_id, key, value)
                if key == "formats":
                    self._fold_formats(job, value)
                await self._persist_summary(job)
            case NodeFinished(node):
                job.steps[node] = now_iso()
            case Output(value) if isinstance(value, dict):
                errors.extend(value.get("errors") or [])
                job.terminal_kind = value.get("terminal_kind")
                job.final_answer = value.get("final_answer")
                if job.terminal_kind not in DELIVERED:
                    # `job.error` is why the run could not serve the request.
                    # A declared refusal has no such message: nothing went
                    # wrong, and what was missing is in the answer itself.
                    job.error = value.get("user_error_message")
                # A document write that failed leaves the run DONE (#28): the
                # answer is the work, and the error says which format and why.
                if value.get("document_error"):
                    job.error = "; ".join(filter(None, [job.error, value["document_error"]]))
            case _:
                pass

    @staticmethod
    def _fold_formats(job: Job, formats: list[str] | None) -> None:
        if formats is None:
                # The document step finished and wrote nothing. For a caller
                # who had already spoken that is the node standing aside —
                # nothing to record. For a request nobody named a format for,
                # it is the decision #96 made: silence is no file, and this is
                # the moment it is known — so recorded now, as `create_job`
                # records `[]`, rather than left for the terminal to discover.
                # `formats` stays `None`: "said nothing" and "said no file"
                # remain two facts on the record (#84), with one outcome.
            if job.formats is None:
                job.deliverable_expected = False
            return
                # The engine read the request for a document the caller said
                # nothing about (#90). It reaches the record the way every
                # other graph fact does — the node wrote to state, the runner
                # translated it, and the fold happens here: a node that called
                # the repository itself would be the first one that does.
        job.formats = formats
        if not formats:
            # Known now, so recorded now, exactly as `create_job` records it
            # for a caller who asked for no file. The flag only ever goes
            # True → False, and this is the True → False.
            job.deliverable_expected = False

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
        """Stop a job, wherever it runs, and answer with the status it has.

        Never answers CANCELLED for a run that carries on. In order:

        - the run is a task of THIS process: cancel it, as always;
        - the store is process-local: there is no other process, so a job with
          no task here is not running anywhere and gets the tombstone;
        - QUEUED on a shared store: the request is recorded first and the
          tombstone after, so a process picking the job up at that instant
          still hears it (`_begin`);
        - RUNNING elsewhere: the request is recorded and the owner — the only
          process that can stop the run — acts on it at its next heartbeat;
          this waits for that, up to `LeasePolicy.wait`, and answers with
          what the store then says. If the owner is provably gone, nobody
          will act on it, so the record is settled here. If it is alive and
          slow, the answer is still RUNNING and the request stands: the
          truth, rather than a CANCELLED the run would later contradict.
        """
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return await self.get_job(job_id)
        job = await self.get_job(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return job
        if self.repo.shared:
            await self.repo.request_cancel(job_id, now_iso())
            if job.status is JobStatus.RUNNING:
                return await self._await_remote_stop(job_id)
        job.status = JobStatus.CANCELLED
        await self._persist_summary(job)
        return job

    async def _await_remote_stop(self, job_id: str) -> Job | None:
        """Wait for the owner of a running job to act on a cancel request.

        **A summary that is already terminal is never fenced nor rewritten,
        whatever the lease says** — the owner settled it, and its ending is
        the job's. Two reads decide that, and their ORDER is what makes the
        second one sufficient: the control record first, the summary after.

        It rests on one ordering the owner keeps on every path that gives a
        lease up: **its final summary is written before the lease is
        released** — `_drive` on a normal end, on a cancel and on a failure
        (`_persist_summary`, then `_release`), and `_begin` when it hears a
        cancel before running (the same two, in that order). A fenced owner
        (`_abandon`) writes nothing and releases nothing: the lease it lost
        is the fencer's. So a control record read with no lease means the
        summary read AFTER it is already final, and the check below returns
        it untouched.

        Read the other way round, the two reads straddle the owner's
        settlement: the summary says RUNNING, the owner then writes CANCELLED
        and releases, the control record says "no lease", and the canceller
        took over a job that had just ended — a fence lease left behind and a
        second CANCELLED, with a wrong reason, over the owner's (#63, measured:
        a few runs in forty once `BEGIN IMMEDIATE` made reads queue behind
        writes). `_settle` re-reads before it writes as well, so the rule
        does not depend on this loop alone.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.lease.wait
        while True:
            control = await self.repo.load_control(job_id)
            job = await self.get_job(job_id)
            if job is None or job.status is not JobStatus.RUNNING:
                return job
            if (owner_is_gone(control.lease, here=self.identity)
                    and await self._take_over([(job, control)])):
                return await self._settle(
                    job, JobStatus.CANCELLED,
                    "cancelled after the process running it had already stopped")
            if loop.time() >= deadline:
                return job
            await asyncio.sleep(self.lease.poll)

    async def _take_over(self, orphans: list[tuple[Job, JobControl]]) -> list[Job]:
        """Fence the owners of these jobs, and keep the ones none answered for.

        Each lease is overwritten FIRST with one of ours that has already
        expired: it claims nothing (another process may settle the job too,
        to the same effect), but it names somebody else, which is how an
        owner that was only stalled learns at its next heartbeat that the
        job is no longer its to write (`Heartbeat.lost`).

        A store with no compare-and-set leaves one race: an owner that read
        its own lease just before the overwrite renews it just after, and
        never sees the fence. So when the proof of death is only an expired
        lease — which a stalled owner can come back from — this waits one
        heartbeat and reads again: a lease renewed since is an owner alive,
        and its job is left to it. A missing pid on this machine is certain
        and needs no wait, which keeps the common case (a crash, then a
        restart) immediate. What is left is an owner whose read-then-write
        of one heartbeat straddles that whole wait; the TTL makes reaching
        it at all require a stall of thirty seconds.
        """
        uncertain: set[str] = set()
        for job, control in orphans:
            await self.repo.save_lease(job.job_id, self.identity.lease(0))
            if not death_is_certain(control.lease, here=self.identity):
                uncertain.add(job.job_id)
        if uncertain:
            await asyncio.sleep(self.lease.heartbeat)
        taken: list[Job] = []
        for job, _ in orphans:
            if job.job_id in uncertain:
                lease = (await self.repo.load_control(job.job_id)).lease
                if lease is None or lease.owner != self.identity.token:
                    continue                    # renewed under our fence: alive
            taken.append(job)
        return taken

    async def _settle(self, job: Job, status: JobStatus, error: str) -> Job:
        """Write the ending of a job this process took over (`_take_over`).

        Never over an ending already written: an owner that settled the job
        while the takeover was deciding had the last word, and it stands.
        """
        current = await self.get_job(job.job_id)
        if current is not None and current.status not in (JobStatus.QUEUED,
                                                          JobStatus.RUNNING):
            return current
        job.status = status
        job.error = error
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
        limit: int | None = 50,
    ) -> list[Job]:
        """The matching jobs, newest first; `limit=None` for every one of them.

        Selected by the store, then ordered, and only then cut: a listing for
        humans shows the most recent, and a caller that must see everything
        (an announcement, an id to resolve, an orphan to settle) passes
        `limit=None` — never a big number, which is the same bug later (#141).
        """
        jobs = await self.repo.load_all(status=status, session_id=session_id)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs if limit is None else jobs[:limit]

    async def recover_interrupted(self) -> list[Job]:
        """Settle jobs left RUNNING by a process that died (persistent stores).

        Call once at startup, before any job runs. A RUNNING record with no
        task here is a leftover only if nobody else can be running it: always
        true of a process-local store, and on a shared one only once its
        owner is provably gone (`ownership.owner_is_gone`) — a second
        `jobsmith chat` opened on the same database must not fail the jobs
        the first is still running (#10). Leftovers are marked FAILED while
        their checkpoint is retained, so a resume stays possible later; a job
        whose owner is alive is left to it, and both are said on stderr.
        QUEUED jobs are left alone — they never started and can still be run.

        It is the one FAILED that keeps a live frontier: the checkpoint is
        retained on purpose, so a resume settles through `_drive`.
        """
        running = [j for j in await self.list_jobs(status=JobStatus.RUNNING, limit=None)
                   if j.job_id not in self._tasks]
        if self.repo.shared:
            # On a shared store "not in this process" is not "dead": another
            # terminal may be running it right now (#10). Only an owner that
            # is provably gone — lease expired, or its pid gone from this
            # machine — leaves a job to settle, and only once it is fenced.
            # A process-local store has no other process, so there every
            # RUNNING record is a leftover.
            orphans = []
            for job in running:
                control = await self.repo.load_control(job.job_id)
                if owner_is_gone(control.lease, here=self.identity):
                    orphans.append((job, control))
            leftovers = await self._take_over(orphans)
        else:
            leftovers = running
        stale = [await self._settle(job, JobStatus.FAILED,
                                    "interrupted: the process running this job stopped")
                 for job in leftovers]
        alive = len(running) - len(stale)
        if stale:
            print(f"[jobs: {len(stale)} interrupted job(s) marked failed on startup]",
                  file=sys.stderr)
        if alive:
            print(f"[jobs: {alive} job(s) still running in another process, left to it]",
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
        """Stopped jobs of a session whose end was not yet surfaced in its
        conversation (the chat layer announces, then marks them).

        Every terminal status, not only the ones that produced an answer: a
        job the model itself cancelled has still left results and files
        behind, and a run that ends in silence is the defect `_notice_for`'s
        DONE branch was fixed for. A resumed job is unmarked again by
        `_begin_resume`, so picking one back up does not cost the
        conversation its ending.
        """
        jobs = await self.repo.load_all(session_id=session_id, announced=False)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j for j in jobs if j.status in ANNOUNCEABLE]

    async def mark_announced(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if job is not None and not job.announced:
            job.announced = True
            await self._persist_summary(job)
