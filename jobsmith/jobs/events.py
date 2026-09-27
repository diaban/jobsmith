"""Job progress events.

`JobEvents` is the port; `InProcessEvents` is the v1 implementation — queues
in this process, feeding the API's SSE stream. Delivery is best-effort: a
subscriber that stops draining is dropped rather than allowed to block a
running job.

Progress across processes is another implementation of this port, never a
change to the JobManager: `SqliteWatchEvents` below for a SQLite file (#100),
LISTEN/NOTIFY for Postgres still to come (#138). #10 did not take it because
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

from .models import Job


def job_event(job: Job) -> dict[str, Any]:
    """The public shape of a progress event (also what SSE clients receive)."""
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "session_id": job.session_id,
        "query": job.query[:80],
        "steps_done": sorted(job.step_finished_at),
        "report_path": job.report_path,
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


class SqliteWatchEvents(InProcessEvents):
    """In-process fan-out, plus what OTHER processes commit to the same SQLite file.

    SQLite has no push (#100). While at least one queue is subscribed, one
    task asks a dedicated read-only connection for `PRAGMA data_version`
    every `interval` seconds: a single call, and its value moves only when
    another connection committed. Only then are the index summaries re-read,
    and each job whose `updated_at` moved since it was last seen is published
    like a local persist. Nobody subscribed, nothing polls.

    The connection is this class's own, never the saver's or the store's: a
    pragma on a live connection is the SQLite gotcha in CLAUDE.md. A commit
    by this process moves `data_version` too; the jobs it published itself
    are already seen, so the re-read finds nothing new for them.
    """

    def __init__(
        self,
        path: str | Path,
        load_all: Callable[[], Awaitable[list[Job]]],
        *,
        interval: float = 1.0,
    ) -> None:
        super().__init__()
        self._path = str(path)
        self._load_all = load_all
        self._interval = interval
        self._seen: dict[str, str] = {}
        self._task: asyncio.Task | None = None

    def publish(self, event: dict[str, Any]) -> None:
        self._seen[event["job_id"]] = event.get("updated_at") or ""
        super().publish(event)

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        queue = super().subscribe(max_queue=max_queue)
        if self._task is None or self._task.done():
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

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self._path}?mode=ro", uri=True,
                               isolation_level=None, check_same_thread=False)

    @staticmethod
    def _data_version(conn: sqlite3.Connection) -> int:
        return conn.execute("PRAGMA data_version").fetchall()[0][0]

    async def _refresh(self, *, announce: bool) -> None:
        for job in await self._load_all():
            if self._seen.get(job.job_id) != job.updated_at:
                self._seen[job.job_id] = job.updated_at
                if announce:
                    super().publish(job_event(job))

    async def _watch(self) -> None:
        conn = await asyncio.to_thread(self._connect)
        try:
            version = await asyncio.to_thread(self._data_version, conn)
            await self._refresh(announce=False)      # what exists is not news
            while True:
                await asyncio.sleep(self._interval)
                try:
                    now = await asyncio.to_thread(self._data_version, conn)
                    if now != version:
                        version = now
                        await self._refresh(announce=True)
                except sqlite3.Error:    # a busy file this tick: ask again next one
                    continue
        finally:
            conn.close()
