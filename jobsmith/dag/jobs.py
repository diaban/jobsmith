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
from typing import Any

from ..engine.manager import JobManager
from ..engine.models import Job, JobStatus
from .report import document_stem, ensure_formats_available


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
    ) -> Job:
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
        return await self.engine.create_job(
            query, inputs, session_id=session_id,
            document_name=document_stem(document_name) if document_name.strip() else "",
            document_title=document_title.strip(), formats=wanted)

    # ---- the rest is the engine's, as the bench has always called it ----

    async def run_job(self, job_id: str) -> Job:
        return await self.engine.run_job(job_id)

    def start_job(self, job_id: str) -> asyncio.Task:
        return self.engine.start_job(job_id)

    async def resume_job(self, job_id: str) -> Job:
        return await self.engine.resume_job(job_id)

    async def start_resume(self, job_id: str) -> Job:
        return await self.engine.start_resume(job_id)

    async def get_job(self, job_id: str) -> Job | None:
        return await self.engine.get_job(job_id)

    async def list_jobs(self, *, status: JobStatus | None = None,
                        session_id: str | None = None, limit: int | None = 50) -> list[Job]:
        return await self.engine.list_jobs(status=status, session_id=session_id, limit=limit)

    async def cancel_job(self, job_id: str) -> Job | None:
        return await self.engine.cancel_job(job_id)

    async def recover_interrupted(self) -> list[Job]:
        return await self.engine.recover_interrupted()

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        return self.engine.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.engine.unsubscribe(queue)

    async def list_finished_unannounced(self, session_id: str) -> list[Job]:
        return await self.engine.list_finished_unannounced(session_id)

    async def mark_announced(self, job_id: str) -> None:
        await self.engine.mark_announced(job_id)
