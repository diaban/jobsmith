"""JobManager: the job use cases.

Everything the product can *do* with a job lives here — create, run, track,
cancel, recover, deliver. Everything else is delegated to a collaborator, so
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
the thread instead of paying for the whole run again. Only the steps that had
not finished run; the ones already in the store are kept as they are, and the
run settles through the same persistence, events and delivery path as a first
attempt. What is *not* here: re-running part of the DAG of a job that already
finished — that needs a way to say which results are stale, and is its own
feature.
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import sys
import uuid
from collections.abc import Sequence
from typing import Any

from .delivery import Deliverer, Nobody, Pushed, nobody
from .events import InProcessEvents, JobEvents, job_event
from .graph import GraphSpec, JobFailed
from .models import FailureKind, Job, JobStatus, now_iso
from .ownership import (
    Heartbeat,
    JobControl,
    LeasePolicy,
    ProcessIdentity,
    death_is_certain,
    owner_is_gone,
)
from .repository import JobRepository, StoreJobRepository
from .runner import Fact, GraphRunner, Interrupted, JobUpdate, NodeFinished, Output
from .usage import Usage, UsageLedger, current_ledger, usage_ledger

# Statuses a job can be resumed from — see `JobManager._begin_resume`.
RESUMABLE = (JobStatus.CANCELLED, JobStatus.FAILED)

# The endings: a job in one of them is delivered to its return address.
# CANCELLED is one — whoever waits on a job is owed its stop as much as its
# answer, and what a stopped run produced is still there.
SETTLED = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

# The name a graph handed over bare (not as a GraphSpec) is known by.
DEFAULT_GRAPH = "default"

# Ledger scope carrying what previous attempts of a resumed job already spent.
EARLIER_ATTEMPTS = "earlier attempts"


class JobManager:
    def __init__(
        self,
        graph: GraphSpec | Any = None,
        store: Any = None,
        *,
        graphs: Sequence[GraphSpec] = (),
        repository: JobRepository | None = None,
        runner: GraphRunner | None = None,
        events: JobEvents | None = None,
        lease: LeasePolicy | None = None,
        deliverers: Sequence[Deliverer] = (),
    ):
        """`graph` is the default graph — a `GraphSpec`, or a compiled graph
        whose output is its result; `graphs` are more, each run by its name.
        A `runner` replaces the default graph's (tests drive the lifecycle
        with no graph at all). `deliverers` are the return-address kinds
        served besides `none` (`delivery.py`)."""
        if repository is None and store is None:
            raise ValueError("JobManager needs a store or an explicit repository")
        if runner is None and graph is None and not graphs:
            raise ValueError("JobManager needs a graph or an explicit runner")
        specs = list(graphs)
        if graph is not None:
            specs.insert(0, graph if isinstance(graph, GraphSpec) else GraphSpec(DEFAULT_GRAPH, graph))
        self.default_graph = specs[0].name if specs else DEFAULT_GRAPH
        self.specs: dict[str, GraphSpec] = {spec.name: spec for spec in specs}
        self.specs.setdefault(self.default_graph, GraphSpec(self.default_graph, None))
        self._runners: dict[str, GraphRunner] = {
            spec.name: GraphRunner(spec.graph) for spec in specs}
        if runner is not None:
            self._runners[self.default_graph] = runner
        self.graph = self.specs[self.default_graph].graph      # the default one's
        self.runner: GraphRunner = self._runners[self.default_graph]
        self.repo: JobRepository = repository or StoreJobRepository(store)
        self.events: JobEvents = events or InProcessEvents()
        self._tasks: dict[str, asyncio.Task] = {}  # in-process cancellation handles
        self._pushes: dict[str, asyncio.Task] = {}  # "job_id#attempt" → its push
        # Who this manager is on the leases it writes, and the timings of
        # ownership (#10). Only consulted when the repository is `shared`:
        # a process-local store pays nothing for any of it.
        self.identity = ProcessIdentity.current()
        self.lease = lease or LeasePolicy()
        self.deliverers: dict[str, Deliverer] = {}
        for deliverer in (Nobody(), *deliverers):
            self.accept(deliverer)

    def accept(self, deliverer: Deliverer) -> None:
        """Serve one more kind of return address (`delivery.py`)."""
        self.deliverers[deliverer.kind] = deliverer

    async def _persist_summary(self, job: Job) -> None:
        # Delivered as it settles, if its kind delivers then (`delivery.py`).
        # Every ending is written here, whichever path wrote it — the one
        # place to stand. A kind this process does not serve is left to one
        # that does, undelivered. A pushed kind is never called here: the
        # ending is saved first, then pushed by a task of its own (#166).
        deliverer = None
        if job.status in SETTLED and job.delivered_at is None:
            deliverer = self.deliverers.get(job.reply_to.get("kind", ""))
            if deliverer is not None and not isinstance(deliverer, Pushed) \
                    and await deliverer.deliver(job):
                job.delivered_at = now_iso()
        job.updated_at = now_iso()
        # Inside `run_job` a usage ledger is installed for this run, so every
        # persist (each finished step, and the terminal one) carries the spend
        # so far — a job that is still running already shows what it has cost.
        ledger = current_ledger()
        if ledger is not None:
            job.usage = ledger.total().to_dict()
        await self.repo.save_summary(job)
        self.events.publish(job_event(job))
        if isinstance(deliverer, Pushed):
            self._push_later(deliverer, job)

    # ---------------- Lifecycle ----------------

    async def create_job(
        self,
        input: Any = None,
        *,
        graph: str | None = None,
        label: str = "",
        reply_to: dict[str, Any] | None = None,
    ) -> Job:
        """Record a job, QUEUED: `input` reaches the graph as it is (JSON),
        `label` is the one line a listing shows, and `reply_to` is where its
        ending goes (`delivery.py`; nobody by default). An unknown graph or
        address kind is refused here, in front of whoever asked."""
        name = graph or self.default_graph
        if name not in self._runners:
            raise ValueError(f"no graph named {name!r} here: {sorted(self._runners)}")
        reply_to = reply_to or nobody()
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            graph=name,
            label=label,
            input=input,
            reply_to=reply_to,
            reply_key=self._key(reply_to),
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
        return await self._drive(job, self._runner(job).stream(job.job_id, job.input))

    def _key(self, reply_to: dict[str, Any]) -> str:
        """The flat key an address is indexed by; refuses a kind not served here."""
        deliverer = self.deliverers.get(reply_to.get("kind", ""))
        if deliverer is None:
            raise ValueError(f"no deliverer for reply_to {reply_to!r} here: "
                             f"{sorted(self.deliverers)}")
        return deliverer.key(reply_to)

    def _runner(self, job: Job) -> GraphRunner:
        if job.graph == self.default_graph:
            return self.runner                 # swappable, as tests swap it
        runner = self._runners.get(job.graph)
        if runner is None:
            raise ValueError(f"job {job.job_id} runs {job.graph!r}, which this process does not have")
        return runner

    async def resume_job(self, job_id: str) -> Job:
        """Re-enter a stopped job's checkpoint and run it to completion.

        See `_begin_resume` for what may be resumed and why.
        """
        job = await self._begin_resume(job_id)
        return await self._drive(job, self._runner(job).resume(job.job_id), resumed=True)

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

    async def answer_job(self, job_id: str, answer: Any) -> Job:
        """Answer a job paused at an `interrupt()` (NEEDS_INPUT) and run it on:
        the interrupt returns `answer`. A new attempt, like a resume (#167)."""
        job = await self._begin_resume(job_id, expect=(JobStatus.NEEDS_INPUT,))
        return await self._drive(job, self._runner(job).answer(job.job_id, answer), resumed=True)

    async def start_answer(self, job_id: str, answer: Any) -> Job:
        """`answer_job` in a background task; a refusal still reaches the caller."""
        job = await self._begin_resume(job_id, expect=(JobStatus.NEEDS_INPUT,))
        self._background(job.job_id, self._drive(
            job, self._runner(job).answer(job.job_id, answer), resumed=True))
        return job

    async def _begin_resume(self, job_id: str, *,
                            expect: tuple[JobStatus, ...] = RESUMABLE) -> Job:
        """Check that this job can be resumed, and open the attempt.

        Resumable = **stopped with work left to do**, which is exactly two
        cases, both of which kept their checkpoint:

        - CANCELLED — `cancel_job` interrupted a run mid-step;
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
        if job.status not in expect:
            raise ValueError(
                f"job {job_id} is {job.status.value}, expected "
                f"{' or '.join(s.value for s in expect)}"
            )
        if not await self._runner(job).pending(job_id):
            raise ValueError(
                f"job {job_id} has no checkpoint to resume from: it either never "
                f"started or already reached its last step"
            )
        job.error = None          # the stopped attempt's message is stale now
        job.result = None         # ...and so is anything it returned
        job.asked = None          # ...and the question it was paused on
        job.failure = None        # ...and why it stopped
        # ...and so is the delivery of the stop: a job picked back up is news
        # again. Without this, a cancelled job delivered to its address and
        # then resumed to DONE is never pending again, and its answer never
        # reaches whoever asked for it. The attempt says WHICH ending a
        # receiver was told of, with no clock to compare (#170).
        job.delivered_at = None
        job.attempt += 1
        # A cancel request is a message to the attempt it stopped; left in the
        # store, it would stop the resumed one on its first heartbeat.
        if self.repo.shared:
            await self.repo.clear_cancel(job_id)
        await self._begin(job)
        return job

    async def _drive(self, job: Job, updates: Any, *, resumed: bool = False) -> Job:
        """Fold a run's updates into the job, and settle it.

        The single place a run is driven, whichever way it was entered: a
        resumed attempt therefore persists, delivers and emits events exactly
        like a first one.
        """
        returned: list[Any] = []            # what the run returned, if it did
        asked: list[Any] = []               # ...or what it paused to ask
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
                    if isinstance(update, Output):
                        returned[:] = [update.value]
                    elif isinstance(update, Interrupted):
                        asked[:] = [update.asked]
                    else:
                        await self._apply(job, update)
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
                    # caller's cancellation: whoever awaits the run here
                    # (`run_for`, `run_job`) gets the
                    # CANCELLED job back, as it would any other ending. A
                    # local cancel landing at the same time still propagates.
                    current = asyncio.current_task()
                    if watch.requested and current is not None and current.uncancel() == 0:
                        return job
                raise
            except Exception as e:
                if watch is not None:
                    watch.stop()
                _fail(job, FailureKind.RAISED, str(e) or type(e).__name__,
                      await self._pending(job), exception=type(e).__name__)
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

            self._conclude(job, returned, asked)
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

    def _conclude(self, job: Job, returned: list[Any], asked: list[Any]) -> None:
        """What a run that went to the end made of itself: DONE with a result,
        or FAILED with a reason — never FAILED in silence — or, paused at an
        `interrupt()`, NEEDS_INPUT with what it asks (#167).

        The result is its graph's `GraphSpec.result` of what the run returned;
        that may raise `JobFailed` for a run that declared it failed, and must
        be JSON, since the record is. So must the question: one that is not is
        kept as its `repr`, so the job stays readable and answerable. A pause
        is not an ending: nothing is delivered until the run ends.
        """
        if asked:
            questions = asked[0]
            job.status = JobStatus.NEEDS_INPUT
            job.asked = questions if _is_json(questions) else [repr(q) for q in questions]
            return
        # A run that got here reached its end: nothing is left for a resume.
        if not returned:
            _fail(job, FailureKind.NO_RESULT, "the run ended without returning")
            return
        try:
            result = self.specs[job.graph].result(returned[0])
        except JobFailed as failed:
            _fail(job, FailureKind.DECLARED, failed.reason or "the run declared it failed")
            job.result = failed.result if _is_json(failed.result) else None
            return
        except Exception as e:
            _fail(job, FailureKind.UNREADABLE, f"its result could not be read: {e!r}")
            return
        if not _is_json(result):
            _fail(job, FailureKind.UNREADABLE, "its result is not JSON")
            return
        job.status, job.result = JobStatus.DONE, result

    async def _apply(self, job: Job, update: JobUpdate) -> None:
        """Fold one update from the runner into the job.

        A fact is recorded as it arrives, whatever it says (`engine/facts.py`):
        persisted under its key, stamped with when it came, and broadcast — a
        fact is progress. So is a root node that finished: noted and persisted,
        one event per step, so a graph that publishes no facts still shows how
        far it has got while it runs (#171).
        """
        match update:
            case Fact(key, value):
                job.facts[key] = value
                job.facts_at[key] = now_iso()
                await self.repo.save_fact(job.job_id, key, value)
                await self._persist_summary(job)
            case NodeFinished(node):
                job.steps[node] = now_iso()
                await self._persist_summary(job)
            case _:
                pass

    def start_job(self, job_id: str) -> asyncio.Task:
        """Fire-and-forget: run the job in a background task (cancellable)."""
        return self._background(job_id, self.run_job(job_id))

    async def run_for(self, job_id: str, timeout: float) -> Job:
        """Run a QUEUED job, waiting up to `timeout` seconds for it to settle.

        Promotion on the clock, for any synchronous caller (0083): the run is
        a background task from the first instant, so it outlives the wait, and
        promoting it is only stopping waiting — nothing is cancelled, nothing
        restarted. The answer is the record as it then stands: settled, or
        still RUNNING and promoted. `asyncio.wait`, never `wait_for`, which
        would cancel the run it gives up on. A run that crashed outside what
        `_drive` settles raises here, to the one caller still listening.
        """
        task = self.start_job(job_id)
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done and not task.cancelled() and (crashed := task.exception()) is not None:
            raise crashed
        return await self._require(job_id)

    async def start_resume(self, job_id: str) -> Job:
        """Resume in a background task, and return the job now RUNNING.

        Async where `start_job` is sync, on purpose: a resume can be *refused*
        (wrong status, no checkpoint), and that refusal has to reach the
        caller rather than die inside a background task nobody awaits. So the
        checks run here, and only the driving is backgrounded.
        """
        job = await self._begin_resume(job_id)
        self._background(job.job_id, self._drive(job, self._runner(job).resume(job.job_id),
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
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING,
                                             JobStatus.NEEDS_INPUT):
            return job
        # A job waiting for an answer runs nowhere: nobody to ask, so it stops here.
        if self.repo.shared and job.status is not JobStatus.NEEDS_INPUT:
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

    async def _pending(self, job: Job) -> tuple[str, ...]:
        """What a resume would run — `()` when the graph cannot say."""
        try:
            return await self._runner(job).pending(job.job_id)
        except Exception:               # no checkpointer, or no checkpoint at all
            return ()

    async def _settle(self, job: Job, status: JobStatus, error: str) -> Job:
        """Write the ending of a job this process took over (`_take_over`).

        Never over an ending already written: an owner that settled the job
        while the takeover was deciding had the last word, and it stands.
        """
        current = await self.get_job(job.job_id)
        if current is not None and current.status not in (JobStatus.QUEUED,
                                                          JobStatus.RUNNING):
            return current
        if status is JobStatus.FAILED:
            _fail(job, FailureKind.INTERRUPTED, error, await self._pending(job))
        else:
            job.status, job.error = status, error
        await self._persist_summary(job)
        return job

    # ---------------- Queries ----------------

    async def get_job(self, job_id: str) -> Job | None:
        return await self.repo.load(job_id)

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        reply_to: dict[str, Any] | None = None,
        limit: int | None = 50,
    ) -> list[Job]:
        """The matching jobs, newest first; `limit=None` for every one of them.

        Selected by the store, then ordered, and only then cut: a listing for
        humans shows the most recent, and a caller that must see everything
        (a delivery, an id to resolve, an orphan to settle) passes
        `limit=None` — never a big number, which is the same bug later (#141).
        """
        jobs = await self.repo.load_all(
            status=status, reply_key=None if reply_to is None else self._key(reply_to))
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs if limit is None else jobs[:limit]

    async def recover_interrupted(self) -> list[Job]:
        """Settle jobs left RUNNING by a process that died (persistent stores).

        Call once at startup, before any job runs. A RUNNING record with no
        task here is a leftover only if nobody else can be running it: always
        true of a process-local store, and on a shared one only once its
        owner is provably gone (`ownership.owner_is_gone`) — a second
        process opened on the same database must not fail the jobs
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
        await self._push_pending()
        return stale

    # ---------------- Live events ----------------

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        """Get a queue of job-progress events (every summary persist emits one)."""
        return self.events.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.events.unsubscribe(queue)

    # ---------------- Delivery ----------------

    async def pending_deliveries(self, reply_to: dict[str, Any]) -> list[Job]:
        """Settled jobs for this address not yet delivered, newest first — what
        a receiver that pulls (`delivery.Pulled`) has still to take.

        The store selects by the address's flat key only (`delivery.py`); the
        ending and the stamp are read here, from that one receiver's jobs.
        """
        jobs = [job for job in await self.list_jobs(reply_to=reply_to, limit=None)
                if job.status in SETTLED and job.delivered_at is None]
        return jobs

    def _push_later(self, deliverer: Pushed, job: Job) -> None:
        """Push one ending in a task of its own, once per attempt in this process.

        A fresh context: the task must not carry the run's usage ledger or
        anything else of the run that happened to schedule it.
        """
        key = f"{job.job_id}#{job.attempt}"
        if key in self._pushes:
            return
        task = asyncio.create_task(self._push(deliverer, dataclasses.replace(job)),
                                   name=f"push:{key}", context=contextvars.Context())
        self._pushes[key] = task
        task.add_done_callback(lambda _: self._pushes.pop(key, None))

    async def _push(self, deliverer: Pushed, job: Job) -> None:
        """Retry until the receiver has it (`delivery.Pushed`), then stamp.

        Stamped only while the record is still that ending: a job resumed
        meanwhile ends again, and that later ending has its own push.
        """
        delays = deliverer.delays()
        while True:
            try:
                await deliverer.push(job)
                break
            except Exception as failed:       # a receiver's failure is retried, never raised
                delay = next(delays)
                print(f"[jobs: push of {job.job_id} to {job.reply_key} failed ({failed}); "
                      f"again in {delay:g}s]", file=sys.stderr)
                await asyncio.sleep(delay)
        current = await self.get_job(job.job_id)
        if current is not None and current.attempt == job.attempt \
                and current.status in SETTLED and current.delivered_at is None:
            current.delivered_at = now_iso()
            await self._persist_summary(current)

    async def _push_pending(self) -> None:
        """Push again what a stopped process left pending — at startup."""
        if not any(isinstance(d, Pushed) for d in self.deliverers.values()):
            return
        for job in await self.repo.load_all():
            deliverer = self.deliverers.get(job.reply_to.get("kind", ""))
            if isinstance(deliverer, Pushed) and job.status in SETTLED \
                    and job.delivered_at is None:
                self._push_later(deliverer, job)

    async def mark_delivered(self, job_id: str) -> None:
        """The receiver has it: stamp `delivered_at`, once."""
        job = await self.get_job(job_id)
        if job is not None and job.delivered_at is None:
            job.delivered_at = now_iso()
            await self._persist_summary(job)


def _fail(job: Job, kind: FailureKind, error: str, pending: Sequence[str] = (),
          **detail: str) -> None:
    """FAILED, said twice: `error` for a reader, `failure` as data (#187).
    Retryable is what `_begin_resume` would answer: a live frontier."""
    job.status, job.error = JobStatus.FAILED, error
    job.failure = {"kind": kind.value, "pending": list(pending),
                   "retryable": bool(pending)} | detail


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
