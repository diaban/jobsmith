"""A LangChain agent told how its running jobs are doing — only when that moved.

The engine records every root node of any graph as it finishes (`Job.steps`)
and when each fact arrived (`Job.facts_at`), in every summary. This turns
that into a one-line digest per job, pushed on the model REQUEST (never in
the thread) on the turns where it CHANGED, and remembered in memory only:
repeating yesterday's news costs tokens, and a restart costs at most one
repeated line. The endings are `JobDeliveryMiddleware`'s, not this one's.

Override `line`, `signature` and `notice` to say it in an application's own
words, and `load` to read more than the summary.
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from ...engine.manager import JobManager
from ...engine.models import Job, JobStatus
from .delivery import inject

IN_FLIGHT = (JobStatus.QUEUED, JobStatus.RUNNING)


class JobProgressMiddleware(AgentMiddleware):
    """Tells the model how the running jobs addressed to `reply_to` are doing."""

    def __init__(self, jobs: JobManager, reply_to: dict[str, Any], *, max_jobs: int = 5):
        super().__init__()
        self.jobs = jobs
        self.reply_to = reply_to
        self.max_jobs = max_jobs     # detailed in one notice; the rest are counted
        # job_id → the signature the model was last shown.
        self._reported: dict[str, str] = {}

    async def load(self, job: Job) -> Any:
        """What `line` and `signature` read for one job: its summary, here."""
        return job

    def signature(self, job: Any) -> str:
        """What must change before a job is worth reporting again. Elapsed
        time is not part of it, or every turn would look like news."""
        return f"{job.status.value}:{len(job.steps)}:{len(job.facts_at)}"

    def line(self, job: Any) -> str:
        """One line: how far the job has got."""
        done = sorted(job.steps, key=lambda node: job.steps[node])
        return (f"Job {job.job_id[:8]} ({job.label[:50]!r}): {job.status.value}"
                + (f", done so far: {', '.join(done)}" if done else ""))

    async def notice(self, lines: list[str]) -> SystemMessage:
        return SystemMessage("[job progress] These jobs are still running — no results "
                             "yet, do not present them as finished. Mention their state "
                             "only if the user asks or it is genuinely useful.\n"
                             + "\n".join(lines))

    async def _in_flight(self) -> tuple[list[Any], int]:
        """The address's running and queued jobs, newest first; the few newest
        loaded (`load`), the others only counted. A job that settled between
        the listing and the load is left to the completion notice."""
        listed = [job for status in IN_FLIGHT
                  for job in await self.jobs.list_jobs(reply_to=self.reply_to, status=status,
                                                       limit=None)]
        listed.sort(key=lambda job: job.created_at, reverse=True)
        loaded = [await self.load(job) for job in listed[:self.max_jobs]]
        shown = [job for job in loaded if job is not None and job.status in IN_FLIGHT]
        return shown, max(len(listed) - len(shown), 0)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        shown, others = await self._in_flight()
        moved = [job for job in shown
                 if self.signature(job) != self._reported.get(job.job_id)]
        if not moved:
            return await handler(request)
        lines = [self.line(job) for job in moved]
        if others:
            lines.append(f"(+{others} more still running)")
        response = await handler(
            request.override(messages=inject(list(request.messages), [await self.notice(lines)])))
        # Only now that the model has seen it: re-baseline, rebuilt from the
        # in-flight set, so a job that settles drops out of the map.
        self._reported = {job.job_id: self.signature(job) for job in shown}
        return response
