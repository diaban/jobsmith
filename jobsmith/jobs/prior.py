"""`PriorJobSource` over the job records — the adapter for #74.

The port is `core/prior_jobs.py`; the capability that consumes it
(`agents/default/prior_jobs.py`) imports neither this module nor anything
else in this package. This is where the two meet, and it lives here because
everything it has to know is this layer's: that a job's records are reached
through `JobRepository`, that `ordered_results()` is what puts them in plan
order, that the answer is `final_answer` and the status is a `JobStatus`.

That is also the whole argument for the port. Put the same knowledge in a
capability and the agent layer starts importing the job engine; put it here
and a second backing — a daemon answering over HTTP, a store of archived runs
— is another adapter rather than a rewrite.

**Rendering is this side's job, and it is deliberately the dumb one.**
`default_result_markdown` is the framework's answer for "a payload nobody
described", and nobody can describe these here: the capability that produced
a step's result may not be registered in the app doing the reading (a run
from a deployment with `web_search`, read back without a Tavily key), so
calling its `render_context` is not available and calling *this* app's
namesake would be presenting one capability's payload through another's idea
of it. A prior run's material is read as what it is — a payload — and the
one thing this adapter refuses to do is invent a shape for it.
"""
from __future__ import annotations

from ..core.capability import default_result_markdown
from ..core.prior_jobs import PriorJob, PriorJobUnavailable, PriorStep
from .repository import JobRepository


class RepositoryPriorJobs:
    """`PriorJobSource` over a `JobRepository` — no filesystem, no report.

    It reads the records the run already wrote: the answer it reached and
    each step's own material, which is precisely what re-reading the
    deliverable could not give back (the report is a deliverable, not a
    trace — per-step material is not inlined into it).

    Scope is NOT enforced here, and that is a decision rather than an
    oversight: this adapter is composed once per app and a job reference
    reaches it as an id with no asker attached. Who may reference what is
    settled where sessions exist — `chat/tools.py` resolves every reference
    against the session's own jobs before it becomes an input — for the same
    reason the job engine never sees the thread.
    """

    def __init__(self, repository: JobRepository):
        self.repository = repository

    async def load(self, job_id: str) -> PriorJob:
        job = await self.repository.load(job_id)
        if job is None:
            raise PriorJobUnavailable(f"{job_id!r}: no such job")
        steps = tuple(
            PriorStep(
                capability=name,
                # A failed step has no data and an `error` that says what went
                # wrong; carrying it as its text is what keeps "that run's web
                # search failed" from reading as "that run never searched".
                text=(default_result_markdown(result.get("data") or {})
                      if result.get("ok")
                      else str(result.get("error") or "the step failed")),
                ok=bool(result.get("ok")),
            )
            for name, result in job.ordered_results()
        )
        return PriorJob(
            job_id=job.job_id,
            query=job.query,
            answer=job.final_answer or "",
            status=job.status.value,
            steps=steps,
        )


__all__ = ["RepositoryPriorJobs"]
