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
from ..engine.delivery import Pulled
from ..engine.graph import GraphSpec, JobFailed
from ..engine.manager import JobManager
from ..engine.models import Job, JobStatus
from .report import document_stem, ensure_formats_available
from .state import TERMINAL_UNANSWERED, CapabilityResult, Plan

#: The facts the DAG publishes (`dag/planner.py`, `dag/capability.py`).
PLAN_FACT = "plan"
STEP_FACT = "step:"
_ROLE_ORDER = {"main": 0, "alternate": 1}

#: The name the planner DAG runs under in the engine.
DAG_GRAPH = "dag"

#: The return address of a job a conversation launched: the conversation
#: pulls it (`pending_deliveries`), then marks it delivered once its model has
#: seen it — the engine only knows a kind that pulls (`engine/delivery.py`).
SESSION = "session"


def session_address(session_id: str) -> dict[str, Any]:
    return {"kind": SESSION, "id": session_id}

# The terminals of a run that did its work and has something to hand back: it
# answered, or it declared — as data, from the generator — that the material
# does not answer the request (#59). Both are DONE, because nothing failed in
# either: the graph ran to the end, every step reported, the tokens were spent
# and the files are on disk. Which of the two it was is `terminal_kind`'s to
# say; making the *status* carry it would either call a refusal a crash
# (FAILED misreports the work, and is a dead end — the checkpoint has nothing
# pending, so a resume refuses it) or add a sixth status every consumer would
# have to learn.
DELIVERED = ("answer", TERMINAL_UNANSWERED)


def dag_result(output: dict[str, Any]) -> dict[str, Any]:
    """What a DAG run's final state makes of the job: its ending, its answer,
    the errors its nodes met, and why its document could not be written.

    A run that ended anywhere but a DELIVERED terminal (`user_error`,
    `escalate`) declared it could not serve the request: `JobFailed`, with
    the message it gave and this same result kept.
    """
    result = {
        "terminal_kind": output.get("terminal_kind"),
        "final_answer": output.get("final_answer"),
        "errors": list(output.get("errors") or []),
        "document_error": output.get("document_error"),
    }
    if result["terminal_kind"] not in DELIVERED:
        raise JobFailed(output.get("user_error_message")
                        or f"the run ended as {result['terminal_kind']!r}", result=result)
    return result


def dag_spec(graph: Any) -> GraphSpec:
    """The planner DAG, as the job engine runs it."""
    return GraphSpec(DAG_GRAPH, graph, result=dag_result)


class DagJob:
    """A job of the planner DAG, as the bench reads it.

    The engine's record knows facts, not plans: this view derives what a DAG
    run IS from what it published — its plan, each step's result and when it
    landed, the files it declared — and renders it in the shape every
    front-end already reads. The request and its document decisions are the
    job's `input` (built by `DagJobs.create_job`), the ending and the answer
    its `result` (`dag_result`).

    A view of a SUMMARY (a listing) has no fact values: no plan, no results,
    no files — `get_job` has them. Step times come with every summary.
    """

    def __init__(self, record: Job):
        self.record = record
        self._input: dict[str, Any] = record.input if isinstance(record.input, dict) else {}
        self._result: dict[str, Any] = record.result if isinstance(record.result, dict) else {}

    # ---- what the record carries as is ----

    @property
    def job_id(self) -> str:
        return self.record.job_id

    @property
    def status(self) -> JobStatus:
        return self.record.status

    @property
    def query(self) -> str:
        return self._input.get("query") or self.record.label

    @property
    def inputs(self) -> dict[str, Any]:
        return self._input.get("inputs") or {}

    @property
    def document_name(self) -> str:
        return self._input.get("document_name") or ""

    @property
    def document_title(self) -> str:
        return self._input.get("document_title") or ""

    @property
    def formats(self) -> list[str] | None:
        """What the document is to be: what the caller named, else what the
        graph's document step read out of the request (#90). Three states:
        a list, `[]` for no document at all, `None` for nobody said (#84)."""
        asked = self._input.get("document_formats")
        return asked if asked is not None else self.record.facts.get("formats")

    @property
    def session_id(self) -> str | None:
        """The conversation that launched it, when its address is one."""
        reply_to = self.record.reply_to
        return reply_to.get("id") if reply_to.get("kind") == SESSION else None

    @property
    def created_at(self) -> str:
        return self.record.created_at

    @property
    def updated_at(self) -> str:
        return self.record.updated_at

    @property
    def final_answer(self) -> str | None:
        return self._result.get("final_answer")

    @property
    def terminal_kind(self) -> str | None:
        return self._result.get("terminal_kind")

    @property
    def deliverable_expected(self) -> bool:
        """Was a document meant to be written at all (#84)? False says its
        absence is the *decision* and not a failure: the request asked for no
        document, or said nothing about one — neither the caller nor the
        graph's document step (#96). Until that step has read the request,
        silence is not yet a decision, so it reads True; it only ever goes
        True → False. It exists because `report_path is None` already means
        two other things: the run did not answer, and the write failed."""
        if self.formats:
            return True
        if self._input.get("document_formats") == []:
            return False
        decided = "formats" in self.record.facts or self.status is JobStatus.DONE
        return not decided

    @property
    def announced(self) -> bool:
        """Delivered to its address (`delivered_at`): for a conversation, its
        model has seen the ending; for nobody's job, it settled."""
        return self.record.delivered_at is not None

    @property
    def usage(self) -> dict[str, Any]:
        return self.record.usage

    @property
    def attempt(self) -> int:
        return self.record.attempt

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
        return "; ".join(filter(None, [self.record.error, self._result.get("document_error"),
                                       missing])) or None

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
        engine.accept(Pulled(SESSION))
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
            {
                "query": query,
                "inputs": inputs or {},
                # What the requester asked the DOCUMENT to be (#55), decided
                # once, never re-derived at write time; each defaults on its
                # own. `document_formats` seeded is what silences the graph's
                # document step (#90): None lets it read the request.
                "document_name": document_stem(document_name) if document_name.strip() else "",
                "document_title": document_title.strip(),
                "document_formats": wanted,
            },
            graph=DAG_GRAPH, label=query,
            reply_to=session_address(session_id) if session_id else None))

    # ---- the rest is the engine's, as the bench has always called it ----

    async def run_job(self, job_id: str) -> DagJob:
        return DagJob(await self.engine.run_job(job_id))

    def start_job(self, job_id: str) -> asyncio.Task:
        return self.engine.start_job(job_id)

    async def run_for(self, job_id: str, timeout: float) -> DagJob:
        return DagJob(await self.engine.run_for(job_id, timeout))

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
            status=status, limit=limit,
            reply_to=session_address(session_id) if session_id else None)]

    async def cancel_job(self, job_id: str) -> DagJob | None:
        return _view(await self.engine.cancel_job(job_id))

    async def recover_interrupted(self) -> list[DagJob]:
        return [DagJob(job) for job in await self.engine.recover_interrupted()]

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        return self.engine.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.engine.unsubscribe(queue)

    async def pending_deliveries(self, session_id: str) -> list[DagJob]:
        """A conversation's settled jobs it has not been told of — in full: the
        notice says what each produced."""
        return [DagJob(await self.engine.get_job(job.job_id) or job)
                for job in await self.engine.pending_deliveries(session_address(session_id))]

    async def mark_delivered(self, job_id: str) -> None:
        await self.engine.mark_delivered(job_id)


def _view(job: Job | None) -> DagJob | None:
    return DagJob(job) if job is not None else None
