"""Where job records live.

`JobRepository` is the port: the vocabulary the JobManager uses to persist and
reload jobs, expressed in domain terms (a Job, a plan, one capability's
result). `StoreJobRepository` is the implementation over a LangGraph
BaseStore, and it is the **only** place that knows the namespace schema:

| namespace                     | key        | value                                  |
|-------------------------------|------------|----------------------------------------|
| ("jobs", "index")             | job_id     | summary record (status, query, ...)    |
| ("jobs", job_id, "meta")      | "plan"     | validated plan + rationale             |
| ("jobs", job_id, "meta")      | "errors"   | accumulated NodeError list             |
| ("jobs", job_id, "results")   | cap name   | that capability's CapabilityResult     |

Fine-grained execution state lives in the *checkpointer* under
thread_id == job_id, not here.

Moving job records to a real SQL database is another implementation of this
port — nothing above it needs to change.
"""
from __future__ import annotations

from typing import Any, Protocol

from ..core.state import NodeError, Plan
from .models import Job, JobOutput, JobStatus


class JobRepository(Protocol):
    """Persistence of job records, in the domain's own vocabulary."""

    async def save_summary(self, job: Job) -> None: ...
    async def load(self, job_id: str) -> Job | None: ...
    async def load_all(self, *, limit: int = 50) -> list[Job]: ...
    async def save_plan(self, job_id: str, plan: Plan) -> None: ...
    async def save_errors(self, job_id: str, errors: list[NodeError]) -> None: ...
    async def save_result(self, job_id: str, capability: str, result: dict) -> None: ...


class StoreJobRepository:
    """`JobRepository` over a LangGraph BaseStore (memory, SQLite, Postgres)."""

    def __init__(self, store: Any):
        self.store = store

    # -------- writes --------

    async def save_summary(self, job: Job) -> None:
        await self.store.aput(("jobs", "index"), job.job_id, job.summary())

    async def save_plan(self, job_id: str, plan: Plan) -> None:
        await self.store.aput(("jobs", job_id, "meta"), "plan", plan)

    async def save_errors(self, job_id: str, errors: list[NodeError]) -> None:
        await self.store.aput(("jobs", job_id, "meta"), "errors", list(errors))

    async def save_result(self, job_id: str, capability: str, result: dict) -> None:
        """A capability's own output — intermediate material, not a deliverable."""
        await self.store.aput(("jobs", job_id, "results"), capability, result)

    # -------- reads --------

    async def load(self, job_id: str) -> Job | None:
        item = await self.store.aget(("jobs", "index"), job_id)
        if item is None:
            return None
        job = self._from_summary(job_id, item.value)
        plan_item = await self.store.aget(("jobs", job_id, "meta"), "plan")
        if plan_item is not None:
            job.plan = plan_item.value
        for result in await self.store.asearch(("jobs", job_id, "results"), limit=100):
            job.results[result.key] = result.value
        return job

    async def load_all(self, *, limit: int = 50) -> list[Job]:
        """Summaries only — plan and results are loaded by `load`."""
        items = await self.store.asearch(("jobs", "index"), limit=limit)
        return [self._from_summary(item.key, item.value) for item in items]

    @staticmethod
    def _from_summary(job_id: str, s: dict[str, Any]) -> Job:
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
            outputs=[JobOutput(**o) for o in (s.get("outputs") or [])],
            announced=bool(s.get("announced")),
            usage=s.get("usage") or {},      # absent on records written before #2
        )


__all__ = ["JobRepository", "StoreJobRepository"]
