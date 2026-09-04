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

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ the port


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
    async def get_report(self, job_id: str) -> str | None: ...

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

    It also exposes what only an in-process service can do — live event
    subscription and access to output files. The HTTP adapter turns those into
    endpoints (`/events`, `/jobs/{id}/outputs/...`) so remote callers get them
    too; a remote backing consumes the port above and nothing more.

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
        job = await self.manager.get_job(job_id)
        if job is None or not job.report_path:
            return None
        with contextlib.suppress(OSError):
            return Path(job.report_path).read_text(encoding="utf-8")
        return None

    # -- in-process only (the HTTP adapter re-exposes these) --

    async def list_outputs(self, job_id: str) -> list[dict] | None:
        """Everything the job produced for the human (deliverable + annexes)."""
        import dataclasses

        job = await self.manager.get_job(job_id)
        if job is None:
            return None
        return [dataclasses.asdict(o) | {"name": o.name} for o in job.outputs]

    async def find_output(self, job_id: str, name: str) -> str | None:
        """Path of one named output, if it exists on disk."""
        job = await self.manager.get_job(job_id)
        if job is None:
            return None
        output = next((o for o in job.outputs if o.name == name), None)
        if output is None or not Path(output.path).is_file():
            return None
        return output.path

    def subscribe(self, **kwargs: Any):
        return self.manager.subscribe(**kwargs)

    def unsubscribe(self, queue: Any) -> None:
        self.manager.unsubscribe(queue)
