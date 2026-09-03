"""Job progress events.

`JobEvents` is the port; `InProcessEvents` is the v1 implementation — queues
in this process, feeding the API's SSE stream. Delivery is best-effort: a
subscriber that stops draining is dropped rather than allowed to block a
running job.

Making progress visible across processes (Postgres LISTEN/NOTIFY, Redis) is
another implementation of this port, not a change to the JobManager.
"""
from __future__ import annotations

import asyncio
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
    }


class JobEvents(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...
    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue: ...
    def unsubscribe(self, queue: asyncio.Queue) -> None: ...


class InProcessEvents:
    """Fan-out to in-process queues. Same v1 scope as task cancellation."""

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
