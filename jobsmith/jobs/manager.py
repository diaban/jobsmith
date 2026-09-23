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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.artifacts import artifact_refs
from ..core.state import TERMINAL_UNANSWERED, NodeError
from ..core.usage import Usage, UsageLedger, current_ledger, usage_ledger
from .events import InProcessEvents, JobEvents, job_event
from .models import Job, JobOutput, JobStatus, now_iso
from .ownership import Heartbeat, LeasePolicy, ProcessIdentity, owner_is_gone
from .report import (
    ReportWriteError,
    compose_reporters,
    document_stem,
    ensure_formats_available,
)
from .repository import JobRepository, StoreJobRepository
from .runner import (
    FormatsChosen,
    GraphRunner,
    JobUpdate,
    NodeErrors,
    PlanReady,
    StepFinished,
    Terminal,
)

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
        reporter: Any = None,
        reporter_factory: Callable[[Sequence[str]], Any] | None = None,
        default_formats: Sequence[str] = ("markdown",),
        reports_dir: str | Path = "artifacts",
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
        # Producing the deliverable is a rendering concern, not the manager's:
        # each job's formats are composed into a Reporter by this factory
        # (#55). A factory rather than a Reporter: what the composition root
        # knows — the registry, whether annexes are inlined — has to reach a
        # reporter built for each job, and a manager that rebuilt one itself
        # would be a manager that decides how a deliverable is rendered.
        # Defaults to the plain composer, so a manager wired without one still
        # honours a requested format, just without what its root would add.
        self.reporter_factory: Callable[[Sequence[str]], Any] = (
            reporter_factory or (lambda formats: compose_reporters(formats)))
        # A fixed Reporter, when one is set, writes EVERY document this
        # manager writes, whatever formats the job named — the swap seam
        # tests and single-format embedders use. It is no longer "the
        # deployment's default": that meant "the file a silent request gets",
        # and since #96 a silent request gets none (see `_deliverable_wanted`).
        self.reporter: Any = reporter
        # What "a document" means when a request wants one and names no
        # format (#96): the names `DEFAULT_FORMATS_ALIAS` resolves to in
        # `create_job`. The composition root passes `$JOBSMITH_REPORT_FORMAT`
        # here and hands the same list to the graph's document step, so the
        # two ways of asking — the argument and the sentence — agree.
        self.default_formats: list[str] = list(default_formats)
        self.reports_dir = Path(reports_dir)  # where deliverables are written
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
        formats: Sequence[str] | str | None = None,
    ) -> Job:
        """Record a job, including what the requester asked the document to be.

        Both document decisions are checked HERE, before a job exists, because
        this is the last point at which whoever asked is still listening: the
        chat tool calls it behind the notice it just wrote, the API answers a
        request, the CLI a command. A name with a separator in it and a
        format nothing can render are the two ways to ask for a file this
        deployment cannot produce, and both refuse in the caller's terms —
        never three minutes later, at the write, in a run that already spent
        its tokens.

        `formats` also carries the decision #55 stopped one field short of
        (#84): `None` says the caller named nothing and leaves the reading of
        the sentence to the graph's document step, while **`[]` says there is
        to be no file**. The second is recorded as `deliverable_expected=False`
        here and now, because it is known here and now — and a QUEUED job that
        reads as expecting a document it will never get is exactly the
        confusion this field exists to remove. `None` is not yet a decision —
        "write me a report" may still be read out of the query — so it stays
        True until that step has read it (`_apply`, `FormatsChosen`).

        `DEFAULT_FORMATS_ALIAS` ("default") is resolved here into this
        deployment's `default_formats` (#96): it is how a caller asks for a
        document without naming its format, and the record carries the names
        it became, never the alias.
        """
        wanted = ensure_formats_available(formats, default=self.default_formats)
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            query=query,
            inputs=inputs or {},
            document_name=document_stem(document_name) if document_name.strip() else "",
            document_title=document_title.strip(),
            formats=wanted,
            deliverable_expected=wanted != [],
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
        return await self._drive(job, self.runner.stream(
            job.job_id, job.query, job.inputs, job.formats))

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
                self._collect_artifacts(job)       # the steps that did finish left files
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
                self._collect_artifacts(job)
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
            if job.status is JobStatus.DONE and self._deliverable_wanted(job):
                # The reporter reads job.usage, so settle it before writing.
                job.usage = ledger.total().to_dict()
                self._write_outputs(job)
            else:
                # Either the run stopped, or it answered and nobody wanted a
                # document of it. Only the second is a decision, and only it
                # is recorded — the flag goes True → False and never back, so
                # a job that asked for no file does not silently re-promise
                # one by failing. Usually already recorded by now (`create_job`
                # for `[]`, the document step for silence); this is the
                # backstop for a graph with no document step at all.
                if job.status is JobStatus.DONE:
                    job.deliverable_expected = False
                # The files its steps left behind are still this job's,
                # whichever of the two it was.
                self._collect_artifacts(job)
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

        **Only a job that reached a DELIVERED terminal AND was meant to leave
        a document gets here** — one that answered, or one that declared it
        could not (#59), which is a run with something to hand back either
        way, and one whose request or whose shape asked for a file
        (`_deliverable_wanted`, #84). Running the Reporter for a run that
        *stopped* would write a report of nothing, and running it for a run
        nobody asked a document of writes a chat turn with a provenance
        section; `_collect_artifacts` is the half that still applies to both,
        and it is called on its own there.
        """
        reporter: Any = None
        try:
            # Composed INSIDE the `try`: a Reporter that cannot be built is a
            # deliverable that could not be written, exactly like one whose
            # write raised. Outside it, the exception skipped the final
            # persist and left the job RUNNING forever — found by falsifying
            # #96, where composing nothing raises by design.
            reporter = self._reporter_for(job)
            outputs = list(reporter.write(job, self.reports_dir))
        except Exception as e:
            failure = e if isinstance(e, ReportWriteError) else ReportWriteError(
                getattr(reporter, "format", None) or ",".join(job.formats or [])
                or "unknown", e)
            outputs = failure.outputs       # keep what did make it to disk
            job.error = str(failure)
        # Annexes are collected regardless of how the report went: they are on
        # disk either way, and a file with no JobOutput is a file nobody can
        # find — the same reason `ReportWriteError` carries its outputs.
        self._collect_artifacts(job, deliverables=outputs)

    def _reporter_for(self, job: Job) -> Any:
        """The Reporter this job's deliverable goes through.

        Composed from the job's own formats — from the caller, which
        `create_job` already accepted, or from the engine's own document step,
        which can only name what this deployment renders (#90). Either way
        this cannot be where a format is found wanting. A fixed `reporter`,
        when one was set, overrides the composition.
        """
        if self.reporter is not None:
            return self.reporter
        # Only ever reached with a non-empty `formats` (`_deliverable_wanted`).
        # There is no fallback for an empty one on purpose: "no formats" is
        # "no document" (#84, #96), and `compose_reporters([])` raises rather
        # than guess markdown.
        return self.reporter_factory(job.formats or [])

    def _collect_artifacts(
        self,
        job: Job,
        *,
        deliverables: list[JobOutput] | None = None,
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
        not answer — and one nobody asked a document of (#84) — passes no
        deliverables at all, so it has no `main`, `report_path` stays None
        and `/report` still 404s. The annexes are offered as what they are,
        not dressed up as a partial success; and a run that wanted no report
        can still have produced files, which is why this half never depends
        on the other.

        `job.outputs` is **assigned**, never appended to: a cancelled job that
        collects and is then resumed keeps the earlier attempt's results (the
        repository loads them), so the second collection re-derives the same
        list rather than doubling it.
        """
        annexes, missing = self._capability_outputs(job)
        job.outputs = list(deliverables or []) + annexes
        # Said out loud whatever the terminal. A declaration can only exist on
        # a step that FINISHED (`_apply` writes `results` on `StepFinished`
        # alone), so a ref with no file is a capability that promised what it
        # did not leave — a defect the run stopping afterwards cannot explain
        # away. It is appended after the failure reason, so "why the run
        # stopped" still reads first, and a run someone is already inspecting
        # is the last place to hide a second bug.
        if missing:
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
        Staying silent is no better, **whatever terminal the run reached**: a
        declaration only exists on a step that FINISHED (`_apply` writes
        `results` on `StepFinished` alone), so a ref with no file behind it is
        a capability that promised what it did not leave, and a run stopping
        later cannot retroactively explain a promise made by a completed step.
        It lands in `job.error` — the same channel, and the same reasoning, as
        a failed report write: the job keeps the status the work earned, and
        the error says which part of the delivery did not happen. Appended
        after any failure reason already there, so "why the run stopped" is
        still the first thing read.

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
            case FormatsChosen(None):
                # The document step finished and wrote nothing. For a caller
                # who had already spoken that is the node standing aside —
                # nothing to record. For a request nobody named a format for,
                # it is the decision #96 made: silence is no file, and this is
                # the moment it is known — so recorded now, as `create_job`
                # records `[]`, rather than left for the terminal to discover.
                # `formats` stays `None`: "said nothing" and "said no file"
                # remain two facts on the record (#84), with one outcome.
                if job.formats is None and job.deliverable_expected:
                    job.deliverable_expected = False
                    await self._persist_summary(job)
            case FormatsChosen(formats):
                # The engine read the request for a document the caller said
                # nothing about (#90). It reaches the record the way every
                # other graph fact does — the node wrote to state, the runner
                # translated it, and the fold happens here: a node that called
                # the repository itself would be the first one that does.
                job.formats = formats
                if not formats:
                    # Known now, so recorded now, exactly as `create_job`
                    # records it for a caller who asked for no file. The flag
                    # only ever goes True → False, and this is the True → False.
                    job.deliverable_expected = False
                await self._persist_summary(job)
            case PlanReady(plan):
                job.plan = plan
                await self.repo.save_plan(job.job_id, plan)
                # The plan is the first thing a watcher can see of a run, and
                # it exists long before the first step lands. Without a
                # summary here the event stream says nothing until then, and
                # a UI drawing the DAG shows "no plan yet" for the whole of
                # the first step — a picture that was stale when it was drawn.
                await self._persist_summary(job)
            case StepFinished(capability, result):
                job.results[capability] = result
                job.step_finished_at[capability] = now_iso()
                await self.repo.save_result(job.job_id, capability, result)
                await self._persist_summary(job)   # touch updated_at for progress
            case Terminal(terminal_kind, final_answer, user_error_message):
                job.terminal_kind = terminal_kind
                job.final_answer = final_answer
                if terminal_kind not in DELIVERED:
                    # `job.error` is why the run could not serve the request.
                    # A declared refusal has no such message: nothing went
                    # wrong, and what was missing is in the answer itself.
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
        """Wait for the owner of a running job to act on a cancel request."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.lease.wait
        while True:
            job = await self.get_job(job_id)
            if job is None or job.status is not JobStatus.RUNNING:
                return job
            control = await self.repo.load_control(job_id)
            if owner_is_gone(control.lease, here=self.identity):
                return await self._settle_orphan(
                    job, JobStatus.CANCELLED,
                    "cancelled after the process running it had already stopped")
            if loop.time() >= deadline:
                return job
            await asyncio.sleep(self.lease.poll)

    async def _settle_orphan(self, job: Job, status: JobStatus, error: str) -> Job:
        """Settle a job whose owner is provably gone, from this process.

        The lease is overwritten FIRST with one of ours that has already
        expired: it claims nothing (another process may settle it too, to the
        same effect), but it names somebody else, which is how an owner that
        was only stalled — alive past its TTL — learns at its next heartbeat
        that the job is no longer its to write (`Heartbeat.lost`).
        """
        await self.repo.save_lease(job.job_id, self.identity.lease(0))
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

        Call once at startup, before any job runs. A RUNNING record with no
        task here is a leftover only if nobody else can be running it: always
        true of a process-local store, and on a shared one only once its
        owner is provably gone (`ownership.owner_is_gone`) — a second
        `jobsmith chat` opened on the same database must not fail the jobs
        the first is still running (#10). Leftovers are marked FAILED while
        their checkpoint is retained, so a resume stays possible later; a job
        whose owner is alive is left to it, and both are said on stderr.
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
        running = [j for j in await self.list_jobs(status=JobStatus.RUNNING, limit=1000)
                   if j.job_id not in self._tasks]
        stale: list[Job] = []
        alive = 0
        for job in running:
            # On a shared store "not in this process" is not "dead": another
            # terminal may be running it right now (#10). Only an owner that
            # is provably gone — lease expired, or its pid gone from this
            # machine — leaves a job to settle. A process-local store has no
            # other process, so there every RUNNING record is a leftover.
            if self.repo.shared:
                control = await self.repo.load_control(job.job_id)
                if not owner_is_gone(control.lease, here=self.identity):
                    alive += 1
                    continue
                await self.repo.save_lease(job.job_id, self.identity.lease(0))
            job.status = JobStatus.FAILED
            job.error = "interrupted: the process running this job stopped"
            await self._persist_summary(job)
            stale.append(job)
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
        jobs = await self.list_jobs(session_id=session_id, limit=100)
        return [j for j in jobs if j.status in ANNOUNCEABLE and not j.announced]

    async def mark_announced(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if job is not None and not job.announced:
            job.announced = True
            await self._persist_summary(job)
