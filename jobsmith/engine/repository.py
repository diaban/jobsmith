"""Where job records live.

`JobRepository` is the port: the vocabulary the JobManager uses to persist and
reload jobs, expressed in domain terms (a Job, the facts its run published). `StoreJobRepository` is the implementation over a LangGraph
BaseStore, and it is the **only** place that knows the namespace schema:

| namespace                     | key        | value                                  |
|-------------------------------|------------|----------------------------------------|
| ("jobs_v1", "index")           | job_id     | summary record (status, graph, input…) |
| ("jobs_v1", job_id, "facts")   | fact key   | {"value": what the run published}      |
| ("jobs_v1", job_id, "control") | "lease"    | the owner's claim on a running job     |
| ("jobs_v1", job_id, "control") | "cancel"   | a stop requested from any process      |

`jobs_v1` because the record changed shape with the core split (2026-09-27):
records written before it, under `jobs`, are left where they are and read by
nothing — starting clean was decided over a reading shim
(docs/design/core-v1.md).

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

from typing import Any, Protocol

from langgraph.store.memory import InMemoryStore

from .models import Job, JobStatus
from .ownership import JobControl, Lease

CONTROL = "control"
ROOT = "jobs_v1"                    # see the schema above

class JobRepository(Protocol):
    """Persistence of job records, in the domain's own vocabulary.

    `shared` says whether another process can see these records. When it is
    False the manager never touches the control half of this port (leases,
    cancel requests): a single process has nothing to coordinate.
    """

    shared: bool

    async def save_summary(self, job: Job) -> None: ...
    async def load(self, job_id: str) -> Job | None: ...
    async def load_all(
        self, *, reply_key: str | None = None, status: JobStatus | None = None,
        updated_since: str | None = None,
    ) -> list[Job]: ...
    async def save_fact(self, job_id: str, key: str, value: Any) -> None: ...
    # -- control: only used when `shared` (see ownership.py) --
    async def save_lease(self, job_id: str, lease: Lease) -> None: ...
    async def release_lease(self, job_id: str) -> None: ...
    async def request_cancel(self, job_id: str, at: str) -> None: ...
    async def clear_cancel(self, job_id: str) -> None: ...
    async def load_control(self, job_id: str) -> JobControl: ...


#: "No limit" for a store search, which requires one: the largest a Postgres
#: INTEGER takes, far beyond any job history.
_EVERY_ROW = 2**31 - 1


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
        """One store call — the single door every record goes through.

        It used to retry "database is locked" (#10): LangGraph's SQLite store
        opened each batch with a *deferred* `BEGIN`, and a batch whose read
        snapshot another process had committed past was refused at once,
        the busy timeout never consulted. That is fixed where the connection
        is opened, for every writer rather than for job records only
        (`app/persistence.py::_ImmediateBegin`, #63): each batch takes the
        write lock up front and waits. What is left to see here is a lock
        held past the busy timeout, which is raised as the error it is.
        """
        return await getattr(self.store, method)(*args, **kwargs)

    # -------- writes --------

    async def save_summary(self, job: Job) -> None:
        await self._io("aput", (ROOT, "index"), job.job_id, job.summary())

    async def save_fact(self, job_id: str, key: str, value: Any) -> None:
        """A fact the run published: one key, last write wins."""
        await self._io("aput", (ROOT, job_id, "facts"), key, {"value": value})

    # -------- control (#10) --------

    async def save_lease(self, job_id: str, lease: Lease) -> None:
        await self._io("aput", (ROOT, job_id, CONTROL), "lease", lease.to_dict())

    async def release_lease(self, job_id: str) -> None:
        await self._io("adelete", (ROOT, job_id, CONTROL), "lease")

    async def request_cancel(self, job_id: str, at: str) -> None:
        await self._io("aput", (ROOT, job_id, CONTROL), "cancel", {"requested_at": at})

    async def clear_cancel(self, job_id: str) -> None:
        await self._io("adelete", (ROOT, job_id, CONTROL), "cancel")

    async def load_control(self, job_id: str) -> JobControl:
        """Lease and cancel request in ONE read — this is the heartbeat's
        round trip, paid every couple of seconds per running job."""
        items = {i.key: i.value for i in
                 await self._io("asearch", (ROOT, job_id, CONTROL), limit=10)}
        lease = items.get("lease")
        cancel = items.get("cancel") or {}
        return JobControl(
            lease=Lease.from_dict(lease) if lease else None,
            cancel_requested_at=cancel.get("requested_at"),
        )

    # -------- reads --------

    async def load(self, job_id: str) -> Job | None:
        item = await self._io("aget", (ROOT, "index"), job_id)
        if item is None:
            return None
        job = self._from_summary(job_id, item.value)
        for fact in await self._io("asearch", (ROOT, job_id, "facts"), limit=_EVERY_ROW):
            job.facts[fact.key] = fact.value["value"]
        return job

    async def load_all(
        self, *, reply_key: str | None = None, status: JobStatus | None = None,
        updated_since: str | None = None,
    ) -> list[Job]:
        """Every summary matching the filters — complete, however long the history.

        Summaries only; fact values are loaded by `load`. The filters are
        applied BY the store and nothing is cut after them. A `limit` here once
        kept the OLDEST rows of the whole base (a store returns them in
        insertion order), so past a hundred jobs the newest vanished from
        deliveries, cancellation, events and listings (#141). A listing
        for humans orders and cuts in `JobManager.list_jobs`, after this.

        One search rather than pages: Postgres orders a search by prefix only,
        so paging by offset could skip or repeat rows.
        """
        where: dict[str, Any] = {}
        if reply_key is not None:           # flat: see `engine/delivery.py`
            where["reply_key"] = reply_key
        if status is not None:
            where["status"] = status.value
        if updated_since is not None:       # ISO timestamps order as text
            where["updated_at"] = {"$gte": updated_since}
        items = await self._io("asearch", (ROOT, "index"), filter=where or None,
                               limit=_EVERY_ROW)
        return [self._from_summary(item.key, item.value) for item in items]

    @staticmethod
    def _from_summary(job_id: str, s: dict[str, Any]) -> Job:
        return Job(
            job_id=job_id,
            status=JobStatus(s["status"]),
            graph=s.get("graph") or "",
            label=s.get("label") or "",
            input=s.get("input"),
            result=s.get("result"),
            error=s.get("error"),
            asked=s.get("asked"),
            reply_to=s.get("reply_to") or {"kind": "none"},
            reply_key=s.get("reply_key") or "none",
            delivered_at=s.get("delivered_at"),
            attempt=s.get("attempt") or 1,
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            facts_at=s.get("facts_at") or {},
            steps=s.get("steps") or {},
            usage=s.get("usage") or {},
        )


__all__ = ["JobRepository", "StoreJobRepository"]
