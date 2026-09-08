"""The inbound port: everything a front-end can ask of a running jobsmith.

`AgentService` is the interface; `LocalAgentService` is the in-process
implementation over a composed `AgentApp`. Every entrypoint is an *adapter*
over this port, never a second implementation of the use cases:

    cli/repl.py + cli/main.py   terminal      -> AgentService
    api/app.py                  HTTP + SSE    -> LocalAgentService
    cli/client.py DaemonClient  HTTP client   -> AgentService (remote backing)

That last line is the point: the CLI does not care whether the work happens
in this process or in a daemon, because both answer the same port. Adding a
UI or a bot is one more adapter, with no new use-case code.

Replies are plain dicts on purpose — they are what crosses the HTTP boundary,
so the local and remote backings are indistinguishable to a caller.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .jobs.report import is_binary_format

# ------------------------------------------------------------------ the port


class BinaryDeliverable(RuntimeError):
    """`get_report` was asked for a deliverable that is not text.

    It is a refusal, not a failure. `get_report` promises a string — the
    report `jobsmith report` prints and `GET /jobs/{id}/report` serves
    inline — and a PDF has no such reading: `read_text` on it raises or, with
    another encoding, returns mojibake. The two silent answers are both
    false, `None` most of all: it means "this job has no report", and this
    job has one, on disk, right where it says.

    So the port says the true thing and says where the bytes are —
    `GET /jobs/{id}/outputs/{name}`, whose `FileResponse` infers the type
    from the extension and has always been the way to fetch a file whole.
    Carried as an exception rather than as a value because the alternative is
    a sentinel string, which every caller would have to know not to print.
    The message is built once (`refusing`) and travels verbatim over HTTP, so
    a caller cannot tell which backing refused.
    """

    @classmethod
    def refusing(cls, job_id: str, name: str, report_format: str) -> BinaryDeliverable:
        return cls(
            f"the main deliverable of job {job_id} is {report_format}, which is not "
            f"text — download it from /jobs/{job_id}/outputs/{name}"
        )


class AgentService(ABC):
    """What any front-end needs. Dict shapes match the HTTP API."""

    mode: str = "local"
    persistent: bool = False   # do jobs outlive this process?

    # -- conversation --

    @abstractmethod
    async def new_session(self, session_id: str | None = None) -> str: ...

    @abstractmethod
    async def send(self, session_id: str, text: str) -> dict: ...

    @abstractmethod
    async def approve(self, session_id: str, approved: bool) -> dict: ...

    # -- jobs --

    @abstractmethod
    async def launch_job(
        self, query: str, *, session_id: str | None = None, inputs: dict | None = None
    ) -> dict: ...

    @abstractmethod
    async def list_jobs(
        self, *, status: str | None = None, session_id: str | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def get_job(self, job_id: str) -> dict | None: ...

    @abstractmethod
    async def cancel_job(self, job_id: str) -> dict: ...

    @abstractmethod
    async def resume_job(self, job_id: str) -> dict: ...

    @abstractmethod
    async def get_report(self, job_id: str) -> str | None:
        """The main deliverable as text, or None when the job has none yet.

        Raises `BinaryDeliverable` when it has one and it is bytes — both
        backings, same message.
        """
        ...

    # -- what the job produced, and what it is doing right now --

    @abstractmethod
    async def list_outputs(self, job_id: str) -> list[dict] | None:
        """Every file the job produced for the human, or None if it is unknown.

        Deliverables first, annexes after — the order `Job.outputs` records.
        """
        ...

    @abstractmethod
    async def find_output(self, job_id: str, name: str) -> str | None:
        """Where one named output is, or None when there is no such file.

        The path is on the machine that RAN the job, which is this one only
        when the service is embedded. So it is a locator — what a UI prints
        and what `GET /jobs/{id}/outputs/{name}` serves — and never a promise
        that the caller can open it: a daemon writes to its own disk. A caller
        that wants the bytes downloads them from that route.

        What both backings do promise is the *answer*: None means no such
        file, right now, on the machine that holds it. The remote backing
        therefore asks the daemon rather than trusting the record it already
        has, so an output deleted since the job finished reads as None on both
        sides instead of as a path to nothing.
        """
        ...

    @abstractmethod
    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        """A queue of job-progress events (the `jobs/events.job_event` shape).

        Sync because subscribing is not the I/O — draining the queue is. The
        remote backing keeps an HTTP stream open behind it, so a subscription
        is released with `unsubscribe`, not by dropping the reference.

        Delivery is best-effort on both sides, by the same rule: a consumer
        that stops draining has events dropped (`put_nowait`), never a run
        (embedded) or a reader (remote) blocked behind it.
        """
        ...

    @abstractmethod
    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Stop delivering to a queue `subscribe` returned, and release it."""
        ...

    async def aclose(self) -> None:
        return None

    async def resolve_job(self, prefix: str) -> dict | None:
        """Accept a short job-id prefix, as printed by `jobsmith jobs`."""
        job = await self.get_job(prefix)
        if job is not None:
            return job
        matches = [j for j in await self.list_jobs() if j["job_id"].startswith(prefix)]
        if len(matches) != 1:
            return None
        return await self.get_job(matches[0]["job_id"])


# ------------------------------------------------------- in-process backing


class LocalAgentService(AgentService):
    """Runs the agent in this process.

    Live events and output files were once its private extras; they are part
    of the port now (#48). The HTTP adapter had always republished them
    (`/events`, `/jobs/{id}/outputs/...`), so what was missing was the other
    half — a remote backing that consumes them — and a front-end that had to
    ask which backing it held before it could show progress was a front-end
    written against two ports.

    `mode`/`persistent` describe it as a CLI backing: used directly, jobs stop
    when this process exits. Behind a daemon the same object is long-lived —
    that is exactly what `DaemonClient.persistent = True` reports.
    """

    mode = "embedded"
    persistent = False

    def __init__(self, manager: Any, session_factory: Any, *, on_close: Any = None):
        self.manager = manager
        self.session_factory = session_factory
        self._on_close = on_close
        self._sessions: dict[str, Any] = {}

    async def aclose(self) -> None:
        if self._on_close is not None:
            await self._on_close()

    # -- conversation --

    def _agent(self, session_id: str) -> tuple[Any, dict]:
        """Sessions are rebuildable: this registry is only a cache, the actual
        conversation lives in the checkpointer under thread_id=session_id. So a
        client can keep chatting on its session id across a daemon restart."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self.session_factory(session_id).build()
        return self._sessions[session_id], {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _reply(result: dict) -> dict:
        """Chat result → reply: a plain message, or a job proposal to approve."""
        if "__interrupt__" in result:
            proposal = result["__interrupt__"][0].value
            return {
                "type": "proposal",
                "query": proposal.get("query"),
                "rationale": proposal.get("rationale"),
            }
        return {"type": "message", "content": result["messages"][-1].content}

    async def new_session(self, session_id: str | None = None) -> str:
        session = self.session_factory(session_id) if session_id else self.session_factory()
        self._sessions[session.session_id] = session.build()
        return session.session_id

    async def send(self, session_id: str, text: str) -> dict:
        from langchain_core.messages import HumanMessage

        agent, config = self._agent(session_id)
        return self._reply(await agent.ainvoke({"messages": [HumanMessage(text)]}, config))

    async def approve(self, session_id: str, approved: bool) -> dict:
        from langgraph.types import Command

        agent, config = self._agent(session_id)
        return self._reply(await agent.ainvoke(Command(resume={"approved": approved}), config))

    # -- jobs --

    async def launch_job(self, query, *, session_id=None, inputs=None) -> dict:
        job = await self.manager.create_job(query, inputs, session_id=session_id)
        self.manager.start_job(job.job_id)
        return {"job_id": job.job_id, "status": job.status.value}

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
        from .jobs.models import JobStatus

        jobs = await self.manager.list_jobs(
            status=JobStatus(status) if status else None, session_id=session_id, limit=100
        )
        return [j.summary() | {"job_id": j.job_id} for j in jobs]

    async def get_job(self, job_id: str) -> dict | None:
        job = await self.manager.get_job(job_id)
        return job.to_dict() if job else None

    async def cancel_job(self, job_id: str) -> dict:
        job = await self.manager.cancel_job(job_id)
        return {"job_id": job_id, "status": job.status.value if job else "unknown"}

    async def resume_job(self, job_id: str) -> dict:
        """Restart a stopped job from its checkpoint, in the background.

        A refusal comes back as `{"status": <unchanged>, "error": ...}` rather
        than an exception: the HTTP backing can only answer with a body, and
        the two backings must stay indistinguishable to a front-end.
        """
        try:
            job = await self.manager.start_resume(job_id)
        except KeyError:
            return {"job_id": job_id, "status": "unknown", "error": f"unknown job: {job_id}"}
        except ValueError as e:
            current = await self.manager.get_job(job_id)
            return {"job_id": job_id,
                    "status": current.status.value if current else "unknown",
                    "error": str(e)}
        return {"job_id": job_id, "status": job.status.value}

    async def get_report(self, job_id: str) -> str | None:
        """The main deliverable as text — see the port for what None means.

        Two ways to learn the file is not text, and both answer the same
        refusal: the format it declares, which is the cheap one, and the
        decode itself, which is the true one. The second is not redundant —
        a Reporter added later that forgets to say it is binary still gets a
        truthful answer here instead of a traceback.
        """
        job = await self.manager.get_job(job_id)
        if job is None or not job.report_path:
            return None
        main = next((o for o in job.outputs if o.role == "main"), None)
        if main is not None and is_binary_format(main.format):
            raise BinaryDeliverable.refusing(job_id, main.name, main.format)
        try:
            return Path(job.report_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as not_text:
            fmt = main.format if main else "an unknown format"
            name = main.name if main else Path(job.report_path).name
            raise BinaryDeliverable.refusing(job_id, name, fmt) from not_text
        except OSError:                 # gone from disk since the job finished
            return None

    # -- outputs and live progress (the HTTP adapter re-exposes these) --

    async def list_outputs(self, job_id: str) -> list[dict] | None:
        import dataclasses

        job = await self.manager.get_job(job_id)
        if job is None:
            return None
        return [dataclasses.asdict(o) | {"name": o.name} for o in job.outputs]

    async def find_output(self, job_id: str, name: str) -> str | None:
        job = await self.manager.get_job(job_id)
        if job is None:
            return None
        output = next((o for o in job.outputs if o.name == name), None)
        if output is None or not Path(output.path).is_file():
            return None
        return output.path

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        return self.manager.subscribe(max_queue=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.manager.unsubscribe(queue)
