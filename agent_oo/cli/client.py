"""One client interface, two backings — this is what decouples the UX from
where jobs actually run.

    DaemonClient    talks HTTP to a running `agent-oo serve`
    EmbeddedClient  builds the agent in this process

Both speak the same dict shapes as the HTTP API, so the REPL and every CLI
command are written once and work either way. The difference that matters is
lifetime: with a daemon, a job outlives the command that launched it and any
other command can list or cancel it; embedded, everything dies with the
process — which is why embedded mode says so out loud.
"""
from __future__ import annotations

import contextlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"


class AgentClient(ABC):
    """The operations the CLI and the REPL need. Dict shapes match the API."""

    mode: str
    persistent: bool   # do jobs outlive this process?

    @abstractmethod
    async def new_session(self, session_id: str | None = None) -> str: ...

    @abstractmethod
    async def send(self, session_id: str, text: str) -> dict: ...

    @abstractmethod
    async def approve(self, session_id: str, approved: bool) -> dict: ...

    @abstractmethod
    async def list_jobs(
        self, *, status: str | None = None, session_id: str | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def get_job(self, job_id: str) -> dict | None: ...

    @abstractmethod
    async def cancel_job(self, job_id: str) -> dict: ...

    @abstractmethod
    async def launch_job(
        self, query: str, *, session_id: str | None = None, inputs: dict | None = None
    ) -> dict: ...

    @abstractmethod
    async def get_report(self, job_id: str) -> str | None: ...

    async def aclose(self) -> None:
        return None

    async def resolve_job(self, prefix: str) -> dict | None:
        """Accept a short job-id prefix, as printed by `agent-oo jobs`."""
        job = await self.get_job(prefix)
        if job is not None:
            return job
        matches = [j for j in await self.list_jobs() if j["job_id"].startswith(prefix)]
        if len(matches) != 1:
            return None
        return await self.get_job(matches[0]["job_id"])


class DaemonClient(AgentClient):
    mode = "daemon"
    persistent = True

    def __init__(self, url: str, http: Any):
        self.url = url.rstrip("/")
        self._http = http

    @classmethod
    async def connect(cls, url: str, *, timeout: float = 2.0) -> DaemonClient | None:
        """Return a client if a daemon answers /health, else None."""
        try:
            import httpx
        except ImportError:
            return None
        http = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=None)
        try:
            response = await http.get("/health", timeout=timeout)
            response.raise_for_status()
        except Exception:
            await http.aclose()
            return None
        return cls(url, http)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def new_session(self, session_id: str | None = None) -> str:
        r = await self._http.post("/sessions", json={"session_id": session_id})
        r.raise_for_status()
        return r.json()["session_id"]

    async def send(self, session_id: str, text: str) -> dict:
        r = await self._http.post(f"/sessions/{session_id}/messages", json={"text": text})
        r.raise_for_status()
        return r.json()

    async def approve(self, session_id: str, approved: bool) -> dict:
        r = await self._http.post(
            f"/sessions/{session_id}/approval", json={"approved": approved}
        )
        r.raise_for_status()
        return r.json()

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
        params = {k: v for k, v in (("status", status), ("session_id", session_id)) if v}
        r = await self._http.get("/jobs", params=params)
        r.raise_for_status()
        return r.json()

    async def get_job(self, job_id: str) -> dict | None:
        r = await self._http.get(f"/jobs/{job_id}")
        return r.json() if r.status_code == 200 else None

    async def cancel_job(self, job_id: str) -> dict:
        r = await self._http.post(f"/jobs/{job_id}/cancel")
        r.raise_for_status()
        return r.json()

    async def launch_job(self, query, *, session_id=None, inputs=None) -> dict:
        r = await self._http.post(
            "/jobs", json={"query": query, "session_id": session_id, "inputs": inputs}
        )
        r.raise_for_status()
        return r.json()

    async def get_report(self, job_id: str) -> str | None:
        r = await self._http.get(f"/jobs/{job_id}/report")
        return r.text if r.status_code == 200 else None


class EmbeddedClient(AgentClient):
    """Runs the agent in this process — jobs stop when the command exits."""

    mode = "embedded"
    persistent = False

    def __init__(self, app: Any):
        self.app = app
        self._sessions: dict[str, Any] = {}

    @classmethod
    async def create(cls, **build_kwargs: Any) -> EmbeddedClient:
        from ..app.agent import build_app

        return cls(await build_app(**build_kwargs))

    async def aclose(self) -> None:
        await self.app.aclose()

    # -- chat --

    def _entry(self, session_id: str) -> tuple[Any, dict]:
        """Mirrors the API: the registry is a cache, the thread is the truth."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self.app.new_session(session_id).build()
        return self._sessions[session_id], {"configurable": {"thread_id": session_id}}

    async def new_session(self, session_id: str | None = None) -> str:
        session = self.app.new_session(session_id)
        self._sessions[session.session_id] = session.build()
        return session.session_id

    @staticmethod
    def _shape(result: dict) -> dict:
        from ..api.app import _shape_reply

        return _shape_reply(result)

    async def send(self, session_id: str, text: str) -> dict:
        from langchain_core.messages import HumanMessage

        agent, config = self._entry(session_id)
        return self._shape(await agent.ainvoke({"messages": [HumanMessage(text)]}, config))

    async def approve(self, session_id: str, approved: bool) -> dict:
        from langgraph.types import Command

        agent, config = self._entry(session_id)
        return self._shape(await agent.ainvoke(Command(resume={"approved": approved}), config))

    # -- jobs --

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
        from ..jobs.models import JobStatus

        jobs = await self.app.manager.list_jobs(
            status=JobStatus(status) if status else None, session_id=session_id, limit=100
        )
        return [j.summary() | {"job_id": j.job_id} for j in jobs]

    async def get_job(self, job_id: str) -> dict | None:
        job = await self.app.manager.get_job(job_id)
        return job.to_dict() if job else None

    async def cancel_job(self, job_id: str) -> dict:
        job = await self.app.manager.cancel_job(job_id)
        return {"job_id": job_id, "status": job.status.value if job else "unknown"}

    async def launch_job(self, query, *, session_id=None, inputs=None) -> dict:
        job = await self.app.manager.create_job(query, inputs, session_id=session_id)
        self.app.manager.start_job(job.job_id)
        return {"job_id": job.job_id, "status": job.status.value}

    async def get_report(self, job_id: str) -> str | None:
        job = await self.app.manager.get_job(job_id)
        if job is None or not job.report_path:
            return None
        with contextlib.suppress(OSError):
            return Path(job.report_path).read_text(encoding="utf-8")
        return None


async def open_client(
    *, url: str = DEFAULT_URL, force_local: bool = False, **build_kwargs: Any
) -> AgentClient:
    """Use the daemon when one answers, else run embedded (and say so)."""
    if not force_local:
        client = await DaemonClient.connect(url)
        if client is not None:
            print(f"[daemon: {url} — jobs keep running after you exit]", file=sys.stderr)
            return client
        print(f"[no daemon at {url} — running embedded: jobs stop when you exit]", file=sys.stderr)
    return await EmbeddedClient.create(**build_kwargs)
