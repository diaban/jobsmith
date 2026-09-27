"""Job progress events.

`JobEvents` is the port; `InProcessEvents` is the v1 implementation — queues
in this process, feeding the API's SSE stream. Delivery is best-effort: a
subscriber that stops draining is dropped rather than allowed to block a
running job.

Progress across processes is another implementation of this port, never a
change to the JobManager: `SqliteWatchEvents` for a SQLite file (#100),
`PostgresNotifyEvents` for Postgres (#138). #10 did not take it because
cancellation crosses processes through per-job keys the owner already knows,
while a feed asks "what changed anywhere" — which a BaseStore answers only by
rereading the index, so it is asked only when the file says something moved.
"""
from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from .models import Job, now_iso


def job_event(job: Job) -> dict[str, Any]:
    """The public shape of a progress event (also what SSE clients receive)."""
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "session_id": job.session_id,
        "query": job.query[:80],
        "updated_at": job.updated_at,
        "usage": job.usage,          # spend so far: live, not only at the end
    }


class JobEvents(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...
    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue: ...
    def unsubscribe(self, queue: asyncio.Queue) -> None: ...


class InProcessEvents:
    """Fan-out to in-process queues: what THIS process persists."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # slow consumer: drop rather than block a run
                pass


class WatchedEvents(InProcessEvents):
    """In-process fan-out, plus the jobs OTHER processes persist to the same database.

    The shared half of #100/#138. A subclass supplies `_watch()`, a task that
    runs only while at least one queue is subscribed and calls `_announce`
    for a job another process moved. A job is announced when its `updated_at`
    differs from the last one seen, so this process's own persists (already
    published) and what existed before anyone subscribed are not news.

    "Before anyone subscribed" is the instant `subscribe()` starts the watch,
    stamped there, synchronously — not the moment the watch first gets to
    look, which comes later: a job another process moves in between is news,
    and was once lost there (#145).
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: dict[str, str] = {}
        self._task: asyncio.Task | None = None
        self._since: str = ""           # when the running watch was asked for

    def publish(self, event: dict[str, Any]) -> None:
        self._seen[event["job_id"]] = event.get("updated_at") or ""
        super().publish(event)

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        queue = super().subscribe(max_queue=max_queue)
        if self._task is None or self._task.done():
            self._since = now_iso()
            try:
                self._task = asyncio.get_running_loop().create_task(self._watch())
            except RuntimeError:        # no running loop: this process's events only
                pass
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        super().unsubscribe(queue)
        if not self._subscribers and self._task is not None:
            self._task.cancel()
            self._task = None

    def _announce(self, job: Job, *, quietly: bool = False) -> None:
        """Record `job` as seen and, unless `quietly`, fan it out — never re-broadcast."""
        if self._seen.get(job.job_id) == job.updated_at:
            return
        self._seen[job.job_id] = job.updated_at
        if not quietly:
            InProcessEvents.publish(self, job_event(job))

    async def _watch(self) -> None:
        raise NotImplementedError

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


class SqliteWatchEvents(WatchedEvents):
    """What other processes commit to the same SQLite file, by `PRAGMA data_version`.

    SQLite has no push (#100). The watch asks a dedicated read-only
    connection for `PRAGMA data_version` every `interval` seconds: a single
    call, and its value moves only when another connection committed. Only
    then are the summaries updated since the last look re-read.

    The connection is this class's own, never the saver's or the store's: a
    pragma on a live connection is the SQLite gotcha in CLAUDE.md.
    """

    def __init__(
        self,
        path: str | Path,
        load_since: Callable[[str | None], Awaitable[list[Job]]],
        *,
        interval: float = 1.0,
    ) -> None:
        super().__init__()
        self._path = str(path)
        self._load_since = load_since
        self._watermark: str | None = None
        self._interval = interval

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self._path}?mode=ro", uri=True,
                               isolation_level=None, check_same_thread=False)

    @staticmethod
    def _data_version(conn: sqlite3.Connection) -> int:
        return conn.execute("PRAGMA data_version").fetchall()[0][0]

    async def _refresh(self, *, quiet_before: str = "") -> None:
        # Only what moved since the last look (`$gte`, so a job persisted in
        # the same instant is not lost; `_announce` drops what was seen): the
        # whole index is read once, when the watch starts (#141), and that
        # first look is quiet only about what moved before the subscription.
        for job in await self._load_since(self._watermark):
            self._announce(job, quietly=job.updated_at < quiet_before)
            if self._watermark is None or job.updated_at > self._watermark:
                self._watermark = job.updated_at

    async def _watch(self) -> None:
        conn = await asyncio.to_thread(self._connect)
        try:
            version = await asyncio.to_thread(self._data_version, conn)
            await self._refresh(quiet_before=self._since)   # what existed is not news
            while True:
                await asyncio.sleep(self._interval)
                try:
                    now = await asyncio.to_thread(self._data_version, conn)
                    if now != version:
                        version = now
                        await self._refresh()
                except sqlite3.Error:    # a busy file this tick: ask again next one
                    continue
        finally:
            conn.close()


class PostgresNotifyEvents(WatchedEvents):
    """What other processes persist to the same Postgres database, by LISTEN/NOTIFY (#138).

    Push, no polling: every local publish also sends `NOTIFY jobsmith_jobs,
    '<job_id>'`, fire-and-forget on one connection of its own; while someone
    is subscribed, the watch holds one `LISTEN` connection and loads each
    notified job. Both connections are this class's, off the store's pool, so
    a long LISTEN never takes a pooled connection from a running job.
    """

    CHANNEL = "jobsmith_jobs"

    def __init__(
        self,
        dsn: str,
        load: Callable[[str], Awaitable[Job | None]],
        load_since: Callable[[str], Awaitable[list[Job]]],
    ) -> None:
        super().__init__()
        self._dsn = dsn
        self._load = load
        self._load_since = load_since
        self._notifier: Any = None
        self._lock = asyncio.Lock()
        self._sending: set[asyncio.Task] = set()

    def publish(self, event: dict[str, Any]) -> None:
        super().publish(event)
        try:
            task = asyncio.get_running_loop().create_task(self._notify(event["job_id"]))
        except RuntimeError:            # no running loop: nothing to send it with
            return
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    async def _notify(self, job_id: str) -> None:
        from psycopg import AsyncConnection

        try:
            async with self._lock:
                if self._notifier is None or self._notifier.closed:
                    self._notifier = await AsyncConnection.connect(self._dsn, autocommit=True)
                await self._notifier.execute("SELECT pg_notify(%s, %s)", (self.CHANNEL, job_id))
        except Exception:               # best effort, like every event: a later one repaints
            pass

    async def _watch(self) -> None:
        from psycopg import AsyncConnection

        conn = await AsyncConnection.connect(self._dsn, autocommit=True)
        try:
            await conn.execute(f"LISTEN {self.CHANNEL}")
            # A NOTIFY sent before the LISTEN reached nobody: what moved since
            # the subscription is read once, now that nothing more can be missed.
            for job in await self._load_since(self._since):
                self._announce(job)
            async for note in conn.notifies():
                job = await self._load(note.payload)
                if job is not None:
                    self._announce(job)
        finally:
            await conn.close()

    async def aclose(self) -> None:
        await super().aclose()
        if self._sending:
            await asyncio.gather(*self._sending, return_exceptions=True)
        if self._notifier is not None:
            await self._notifier.close()
