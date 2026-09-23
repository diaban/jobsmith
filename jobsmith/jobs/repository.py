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
| ("jobs", job_id, "control")   | "lease"    | the owner's claim on a running job     |
| ("jobs", job_id, "control")   | "cancel"   | a stop requested from any process      |

The two `control` keys exist only on a store other processes can see
(`shared`, #10), and each has ONE writer class, which is why they are not
fields of the summary: the owner rewrites the summary on every step, so a
cancel written *there* by another process is overwritten by the next step —
the tombstone the running process never saw. A key nobody else writes is a
message that survives until it is read. See `ownership.py`.

Fine-grained execution state lives in the *checkpointer* under
thread_id == job_id, not here.

Moving job records to a real SQL database is another implementation of this
port — nothing above it needs to change.
"""
from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Protocol

from langgraph.store.memory import InMemoryStore

from ..core.state import CapabilityResult, NodeError, Plan
from .models import Job, JobOutput, JobStatus
from .ownership import JobControl, Lease

CONTROL = "control"

# How many times a store call that met a busy SQLite file is tried again, and
# the first pause (doubled each time: 20 ms … ~5 s in total).
BUSY_RETRIES = 8
BUSY_PAUSE = 0.02


def _is_busy(error: BaseException) -> bool:
    return isinstance(error, sqlite3.OperationalError) and "locked" in str(error)


class JobRepository(Protocol):
    """Persistence of job records, in the domain's own vocabulary.

    `shared` says whether another process can see these records. When it is
    False the manager never touches the control half of this port (leases,
    cancel requests): a single process has nothing to coordinate.
    """

    shared: bool

    async def save_summary(self, job: Job) -> None: ...
    async def load(self, job_id: str) -> Job | None: ...
    async def load_all(self, *, limit: int = 50) -> list[Job]: ...
    async def save_plan(self, job_id: str, plan: Plan) -> None: ...
    async def save_errors(self, job_id: str, errors: list[NodeError]) -> None: ...
    async def save_result(self, job_id: str, capability: str, result: CapabilityResult) -> None: ...
    # -- control: only used when `shared` (see ownership.py) --
    async def save_lease(self, job_id: str, lease: Lease) -> None: ...
    async def release_lease(self, job_id: str) -> None: ...
    async def request_cancel(self, job_id: str, at: str) -> None: ...
    async def clear_cancel(self, job_id: str) -> None: ...
    async def load_control(self, job_id: str) -> JobControl: ...


class StoreJobRepository:
    """`JobRepository` over a LangGraph BaseStore (memory, SQLite, Postgres).

    `shared` defaults to "anything but the in-memory store": a SQLite file or
    a Postgres database is visible to every process that opens it, which is
    the whole reason to open one. Only this class knows what store it holds,
    so only it can say — the manager reads the answer, never the type.
    """

    def __init__(self, store: Any, *, shared: bool | None = None):
        self.store = store
        self.shared: bool = (not isinstance(store, InMemoryStore)
                             if shared is None else shared)

    async def _io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """One store call, tried again while a SQLite file reports itself busy.

        Measured with two processes on one file (#10): LangGraph's SQLite
        store opens each batch with a *deferred* `BEGIN`, reads, then writes
        — and a read transaction that must upgrade to a write after another
        connection committed fails at once with "database is locked", the
        busy timeout never consulted (SQLite's WAL snapshot rule). The failed
        statement is the batch's first write, so nothing of it was written
        and every op here is an upsert, a delete or a read: trying again is
        safe, and the conflict is gone as soon as the other writer is. Any
        other error, and a file still busy after ~5 s, is raised as it was.
        """
        pause = BUSY_PAUSE
        for attempt in range(BUSY_RETRIES + 1):
            try:
                return await getattr(self.store, method)(*args, **kwargs)
            except Exception as e:
                if attempt == BUSY_RETRIES or not _is_busy(e):
                    raise
            await asyncio.sleep(pause)
            pause *= 2

    # -------- writes --------

    async def save_summary(self, job: Job) -> None:
        await self._io("aput", ("jobs", "index"), job.job_id, job.summary())

    async def save_plan(self, job_id: str, plan: Plan) -> None:
        await self._io("aput", ("jobs", job_id, "meta"), "plan", plan)

    async def save_errors(self, job_id: str, errors: list[NodeError]) -> None:
        await self._io("aput", ("jobs", job_id, "meta"), "errors", list(errors))

    async def save_result(self, job_id: str, capability: str, result: CapabilityResult) -> None:
        """A capability's own output — intermediate material, not a deliverable."""
        await self._io("aput", ("jobs", job_id, "results"), capability, result)

    # -------- control (#10) --------

    async def save_lease(self, job_id: str, lease: Lease) -> None:
        await self._io("aput", ("jobs", job_id, CONTROL), "lease", lease.to_dict())

    async def release_lease(self, job_id: str) -> None:
        await self._io("adelete", ("jobs", job_id, CONTROL), "lease")

    async def request_cancel(self, job_id: str, at: str) -> None:
        await self._io("aput", ("jobs", job_id, CONTROL), "cancel", {"requested_at": at})

    async def clear_cancel(self, job_id: str) -> None:
        await self._io("adelete", ("jobs", job_id, CONTROL), "cancel")

    async def load_control(self, job_id: str) -> JobControl:
        """Lease and cancel request in ONE read — this is the heartbeat's
        round trip, paid every couple of seconds per running job."""
        items = {i.key: i.value for i in
                 await self._io("asearch", ("jobs", job_id, CONTROL), limit=10)}
        lease = items.get("lease")
        cancel = items.get("cancel") or {}
        return JobControl(
            lease=Lease.from_dict(lease) if lease else None,
            cancel_requested_at=cancel.get("requested_at"),
        )

    # -------- reads --------

    async def load(self, job_id: str) -> Job | None:
        item = await self._io("aget", ("jobs", "index"), job_id)
        if item is None:
            return None
        job = self._from_summary(job_id, item.value)
        plan_item = await self._io("aget", ("jobs", job_id, "meta"), "plan")
        if plan_item is not None:
            job.plan = plan_item.value
        for result in await self._io("asearch", ("jobs", job_id, "results"), limit=100):
            job.results[result.key] = result.value
        return job

    async def load_all(self, *, limit: int = 50) -> list[Job]:
        """Summaries only — plan and results are loaded by `load`."""
        items = await self._io("asearch", ("jobs", "index"), limit=limit)
        return [self._from_summary(item.key, item.value) for item in items]

    @staticmethod
    def _from_summary(job_id: str, s: dict[str, Any]) -> Job:
        return Job(
            job_id=job_id,
            status=JobStatus(s["status"]),
            query=s["query"],
            inputs=s.get("inputs") or {},
            document_name=s.get("document_name") or "",
            document_title=s.get("document_title") or "",
            formats=_stored_formats(s),
            session_id=s.get("session_id"),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            step_finished_at=s.get("step_finished_at") or {},
            terminal_kind=s.get("terminal_kind"),
            final_answer=s.get("final_answer"),
            error=s.get("error"),
            outputs=[JobOutput(**o) for o in (s.get("outputs") or [])],
            deliverable_expected=bool(s.get("deliverable_expected", True)),
            announced=bool(s.get("announced")),
            usage=s.get("usage") or {},      # absent on records written before #2
        )


def _stored_formats(summary: dict[str, Any]) -> list[str] | None:
    """`Job.formats` as it was recorded — keeping `None` apart from `[]` (#84).

    The two are different asks now ("you decide" and "no document"), so the
    obvious `list(s.get("formats") or [])` would read every silent job as one
    that refused a file. Two vintages of record have to survive it:

    - **before #55** the key is absent — `None`, which is what it meant;
    - **between #55 and #84** it is `[]` for a job that named no format, which
      also meant `None`. Such a record is recognised by carrying no
      `deliverable_expected` key at all, so the empty list is read as silence
      rather than as a refusal it could not yet express.

    Anything a record written since says, it says on purpose.
    """
    formats = summary.get("formats")
    if formats is None:
        return None
    if not formats and "deliverable_expected" not in summary:
        return None                     # a pre-#84 record: empty meant unstated
    return list(formats)


__all__ = ["JobRepository", "StoreJobRepository"]
