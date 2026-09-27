"""The planner DAG's jobs: the use cases the bench calls, over the job engine.

What a request to the DAG is — a query, its inputs, and what its document is
to be — is decided and checked here; the engine is handed a job to run and
knows none of it (docs/design/core-v1.md, step 6b). Every entrypoint that
runs the DAG — the chat's tools, the service and so the API and the CLI, the
evals — goes through `DagJobs`; `engine` is the `JobManager` underneath.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ..artifacts.store import ARTIFACT_FACT, JobOutput
from ..engine.manager import JobManager
from ..engine.models import Job, JobStatus
from .report import document_stem, ensure_formats_available
from .state import CapabilityResult, Plan

#: The facts the DAG publishes (`dag/planner.py`, `dag/capability.py`).
PLAN_FACT = "plan"
STEP_FACT = "step:"
_ROLE_ORDER = {"main": 0, "alternate": 1}


class DagJob:
    """A job of the planner DAG, as the bench reads it.

    The engine's record knows facts, not plans: this view derives what a DAG
    run IS from what it published — its plan, each step's result and when it
    landed, the files it declared — and renders it in the shape every
    front-end already reads. What the record still carries of the DAG itself
    (the request, the answer, the document decisions) leaves it at step 6b.3
    of the core split (docs/design/core-v1.md), and this view reads it from
    the input and the result then.

    A view of a SUMMARY (a listing) has no fact values: no plan, no results,
    no files — `get_job` has them. Step times come with every summary.
    """

    def __init__(self, record: Job):
        self.record = record

    # ---- what the record carries as is ----

    @property
    def job_id(self) -> str:
        return self.record.job_id

    @property
    def status(self) -> JobStatus:
        return self.record.status

    @property
    def query(self) -> str:
        return self.record.query

    @property
    def inputs(self) -> dict[str, Any]:
        return self.record.inputs

    @property
    def document_name(self) -> str:
        return self.record.document_name

    @property
    def document_title(self) -> str:
        return self.record.document_title

    @property
    def formats(self) -> list[str] | None:
        return self.record.formats

    @property
    def session_id(self) -> str | None:
        return self.record.session_id

    @property
    def created_at(self) -> str:
        return self.record.created_at

    @property
    def updated_at(self) -> str:
        return self.record.updated_at

    @property
    def final_answer(self) -> str | None:
        return self.record.final_answer

    @property
    def terminal_kind(self) -> str | None:
        return self.record.terminal_kind

    @property
    def deliverable_expected(self) -> bool:
        return self.record.deliverable_expected

    @property
    def announced(self) -> bool:
        return self.record.announced

    @property
    def usage(self) -> dict[str, Any]:
        return self.record.usage

    # ---- what the run published ----

    @property
    def plan(self) -> Plan | None:
        return self.record.facts.get(PLAN_FACT)

    @property
    def results(self) -> dict[str, CapabilityResult]:
        """Each step's result, in ARRIVAL order (when its fact came): a store
        returns facts in an order of its own. Plan order is `ordered_results`."""
        keys = sorted((key for key in self.record.facts if key.startswith(STEP_FACT)),
                      key=lambda key: self.record.facts_at.get(key, ""))
        return {key[len(STEP_FACT):]: self.record.facts[key] for key in keys}

    @property
    def step_finished_at(self) -> dict[str, str]:
        return {key[len(STEP_FACT):]: at for key, at in self.record.facts_at.items()
                if key.startswith(STEP_FACT)}

    def _plan_order(self) -> list[str]:
        return [step["capability"] for step in (self.plan or {}).get("steps", [])]

    def ordered_results(self) -> list[tuple[str, CapabilityResult]]:
        """Results in PLAN order — the only deterministic order there is.

        `results` is filled by parallel waves, so arrival order is not an
        order. Anything a human reads must be stable across two runs of the
        same plan, so it is ordered here once. A result with no plan step (a
        plan that never made it to the store) keeps its place, at the end.
        """
        order, results = self._plan_order(), self.results
        names = sorted(results, key=lambda n: order.index(n) if n in order else len(order))
        return [(name, results[name]) for name in names]

    def step_usage(self, capability: str) -> dict[str, Any]:
        """What one step spent — empty when it made no LLM call."""
        return ((self.results.get(capability) or {}).get("meta") or {}).get("usage") or {}

    # ---- the files it declared ----

    def _declared(self) -> list[tuple[str, JobOutput, bool]]:
        """Every file the run declared: its fact key, the file, and whether it
        was missing when declared (`artifacts.store.declare`)."""
        return [(key, JobOutput(**{k: v for k, v in value.items() if k != "missing"}),
                 bool(value.get("missing")))
                for key, value in self.record.facts.items() if key.startswith(ARTIFACT_FACT)]

    @property
    def outputs(self) -> list[JobOutput]:
        """The files the run declared, in the order a reader wants them: the
        deliverable (`main`), its other renderings, then the steps' annexes
        in PLAN order (0028, 0041). One declared but not there when it was
        declared is not listed, and `error` says so."""
        order = self._plan_order()

        def rank(item: tuple[str, JobOutput, bool]) -> tuple[int, int, str]:
            key, output, _ = item
            producer = output.produced_by or ""
            step = order.index(producer) if producer in order else len(order)
            return (_ROLE_ORDER.get(output.role, 2), step, self.record.facts_at.get(key, ""))

        listed: dict[str, JobOutput] = {}
        for _, output, missing in sorted(self._declared(), key=rank):
            if not missing:
                listed.setdefault(output.path, output)   # declared twice: once, the first
        return list(listed.values())

    @property
    def error(self) -> str | None:
        """Why the run stopped, first; then any declared file that is missing
        — a promise a completed step did not keep, said out loud (0041)."""
        gone = [f"{output.produced_by or output.role} → {output.path}"
                for _, output, missing in self._declared() if missing]
        missing = (f"{len(gone)} file(s) a step reported producing are missing: "
                   f"{', '.join(gone)}" if gone else "")
        return "; ".join(filter(None, [self.record.error, missing])) or None

    @property
    def report_path(self) -> str | None:
        """Path of the main deliverable — `None` when the run never got there,
        when the write failed (`error` says which), or when no document was
        ever going to be written (`deliverable_expected` is False)."""
        main = next((o for o in self.outputs if o.role == "main"), None)
        return main.path if main else None

    # ---- today's shapes ----

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "query": self.query, "inputs": self.inputs,
            "document_name": self.document_name, "document_title": self.document_title,
            "formats": self.formats, "session_id": self.session_id,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "step_finished_at": self.step_finished_at, "terminal_kind": self.terminal_kind,
            "final_answer": self.final_answer, "error": self.error,
            "outputs": [asdict(o) for o in self.outputs], "report_path": self.report_path,
            "deliverable_expected": self.deliverable_expected, "announced": self.announced,
            "usage": self.usage,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **self.summary(), "plan": self.plan,
                "results": self.results}


class DagJobs:
    def __init__(self, engine: JobManager, *, default_formats: Sequence[str] = ("markdown",)):
        self.engine = engine
        # What "a document" means when a request wants one and names no
        # format (#96): the names `DEFAULT_FORMATS_ALIAS` resolves to in
        # `create_job`. The composition root passes `$JOBSMITH_REPORT_FORMAT`
        # here and hands the same list to the graph's document step, so the
        # two ways of asking — the argument and the sentence — agree.
        self.default_formats: list[str] = list(default_formats)

    async def create_job(
        self,
        query: str,
        inputs: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        document_name: str = "",
        document_title: str = "",
        formats: Sequence[str] | str | None = None,
    ) -> DagJob:
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
        to be no file** (the engine records that as `deliverable_expected`).

        `DEFAULT_FORMATS_ALIAS` ("default") is resolved here into this
        deployment's `default_formats` (#96): it is how a caller asks for a
        document without naming its format, and the record carries the names
        it became, never the alias.
        """
        # In a thread: a request for PDF is where its engine is first loaded
        # (#108), seconds of import that must not stall every other session.
        wanted = await asyncio.to_thread(
            ensure_formats_available, formats, default=self.default_formats)
        return DagJob(await self.engine.create_job(
            query, inputs, session_id=session_id,
            document_name=document_stem(document_name) if document_name.strip() else "",
            document_title=document_title.strip(), formats=wanted))

    # ---- the rest is the engine's, as the bench has always called it ----

    async def run_job(self, job_id: str) -> DagJob:
        return DagJob(await self.engine.run_job(job_id))

    def start_job(self, job_id: str) -> asyncio.Task:
        return self.engine.start_job(job_id)

    async def resume_job(self, job_id: str) -> DagJob:
        return DagJob(await self.engine.resume_job(job_id))

    async def start_resume(self, job_id: str) -> DagJob:
        return DagJob(await self.engine.start_resume(job_id))

    async def get_job(self, job_id: str) -> DagJob | None:
        return _view(await self.engine.get_job(job_id))

    async def list_jobs(self, *, status: JobStatus | None = None,
                        session_id: str | None = None, limit: int | None = 50) -> list[DagJob]:
        """Summaries: no plan, no results, no files — `get_job` has them."""
        return [DagJob(job) for job in await self.engine.list_jobs(
            status=status, session_id=session_id, limit=limit)]

    async def cancel_job(self, job_id: str) -> DagJob | None:
        return _view(await self.engine.cancel_job(job_id))

    async def recover_interrupted(self) -> list[DagJob]:
        return [DagJob(job) for job in await self.engine.recover_interrupted()]

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        return self.engine.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.engine.unsubscribe(queue)

    async def list_finished_unannounced(self, session_id: str) -> list[DagJob]:
        """In full: an announcement says what the job produced."""
        return [DagJob(await self.engine.get_job(job.job_id) or job)
                for job in await self.engine.list_finished_unannounced(session_id)]

    async def mark_announced(self, job_id: str) -> None:
        await self.engine.mark_announced(job_id)


def _view(job: Job | None) -> DagJob | None:
    return DagJob(job) if job is not None else None
