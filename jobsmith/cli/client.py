"""Two backings for one port — this is what decouples the UX from where jobs
actually run.

    DaemonClient          talks HTTP to a running `jobsmith serve`
    EmbeddedClient        builds the agent in this process

Both are `AgentService` (see jobsmith/service.py), so the REPL and every CLI
command are written once and work either way. The embedded one adds nothing of
its own: it *is* the local service, which the HTTP API serves too — the use
cases exist in exactly one place.

The difference that matters is lifetime: with a daemon a job outlives the
command that launched it and any other command can list or cancel it;
embedded, everything dies with the process — which is why embedded mode says
so out loud.
"""
from __future__ import annotations

import sys
from typing import Any

from ..service import AgentService, LocalAgentService

DEFAULT_URL = "http://127.0.0.1:8000"

# Kept as the CLI's name for the port (commands are typed against it).
AgentClient = AgentService


class DaemonClient(AgentService):
    """HTTP transport: the use cases run in the daemon, not here."""

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


class EmbeddedClient(LocalAgentService):
    """The local service, owning the app it composed."""

    @classmethod
    async def create(cls, **build_kwargs: Any) -> EmbeddedClient:
        from ..app.agent import build_app

        app = await build_app(**build_kwargs)
        client = cls(app.manager, app.session_factory, on_close=app.aclose)
        client.app = app
        return client


async def open_client(
    *, url: str = DEFAULT_URL, force_local: bool = False, **build_kwargs: Any
) -> AgentService:
    """Use the daemon when one answers, else run embedded (and say so)."""
    if not force_local:
        client = await DaemonClient.connect(url)
        if client is not None:
            print(f"[daemon: {url} — jobs keep running after you exit]", file=sys.stderr)
            return client
        print(f"[no daemon at {url} — running embedded: jobs stop when you exit]", file=sys.stderr)
    return await EmbeddedClient.create(**build_kwargs)
