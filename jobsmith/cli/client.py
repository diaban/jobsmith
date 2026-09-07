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

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

from ..service import AgentService, BinaryDeliverable, LocalAgentService

if TYPE_CHECKING:                      # only to name the app the embedded client owns
    from ..app.agent import AgentApp

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
        self._readers: dict[asyncio.Queue, asyncio.Task] = {}
        self._retiring: set[asyncio.Task] = set()   # cancelled, not yet unwound

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
        """Close the transport — and first the streams that are still on it.

        `unsubscribe` can only *cancel* a reader (it is sync, as the port
        says), and a cancelled task still holds its stream until it has
        unwound. Here there is somewhere to await that, so every reader is
        awaited — the ones still subscribed and the ones retiring — before
        the client they read through goes away.
        """
        readers = list(self._readers.values()) + list(self._retiring)
        self._readers.clear()
        for reader in readers:
            reader.cancel()
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)
        self._retiring.clear()
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

    async def resume_job(self, job_id: str) -> dict:
        r = await self._http.post(f"/jobs/{job_id}/resume")
        if r.status_code in (404, 409):
            # The API says "refused" with a status code; the port says it with
            # an `error` key, so both backings answer a caller the same way.
            job = await self.get_job(job_id)
            return {"job_id": job_id,
                    "status": job["status"] if job else "unknown",
                    "error": r.json().get("detail")}
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
        if r.status_code == 415:
            # A deliverable that is not text: the API says so with a status
            # code, the port with an exception, and the message is the one
            # the service built — so both backings refuse identically.
            raise BinaryDeliverable(r.json().get("detail", ""))
        return r.text if r.status_code == 200 else None

    # -- outputs --

    async def list_outputs(self, job_id: str) -> list[dict] | None:
        r = await self._http.get(f"/jobs/{job_id}/outputs")
        return r.json() if r.status_code == 200 else None

    async def find_output(self, job_id: str, name: str) -> str | None:
        """Ask the daemon whether the file is there, then say where it is.

        Two calls, because the port promises an answer about the file and not
        about the record: the download route is what knows the bytes exist, so
        an output deleted since the job finished is None here exactly as it is
        embedded. It is a *streamed* GET whose body is never read — the status
        line is the whole question, and a deliverable can be megabytes.

        The path that comes back is the daemon's, which the port is explicit
        about: it locates the file, `GET /jobs/{id}/outputs/{name}` fetches it.
        """
        async with self._http.stream("GET", f"/jobs/{job_id}/outputs/{name}") as probe:
            if probe.status_code != 200:
                return None
        outputs = await self.list_outputs(job_id) or []
        return next((o["path"] for o in outputs if o["name"] == name), None)

    # -- live events --

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        """Consume the daemon's SSE stream into a queue of event dicts.

        The daemon fans out to in-process queues (`InProcessEvents`) and
        `/events` serializes one of them; this reads it back, so a caller
        holding either backing drains the same queue of the same dicts and
        never has to ask which one it holds.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._readers[queue] = asyncio.create_task(self._read_events(queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        reader = self._readers.pop(queue, None)
        if reader is None:
            return
        reader.cancel()              # unwinds the `async with`, releasing the stream
        # Held until it has actually unwound: a cancelled task is not a closed
        # stream yet, and a task nothing references can be collected mid-flight.
        self._retiring.add(reader)
        reader.add_done_callback(self._retiring.discard)

    async def _read_events(self, queue: asyncio.Queue) -> None:
        """Decode `/events` into `queue` until cancelled or the daemon stops.

        The drop-on-full policy is `InProcessEvents`' own, and for the same
        reason one step further out: a consumer that stopped draining must not
        stall the reader — which would stall the socket, which would then
        stall the daemon's own publish queue.
        """
        try:
            async with self._http.stream("GET", "/events", timeout=None) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue        # blank separators and SSE comments
                    try:
                        event = json.loads(line[len("data:"):])
                    except json.JSONDecodeError:
                        continue        # not ours to interpret; keep listening
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass            # slow consumer: drop, never block
            # Reached when the daemon closes the stream — a shutdown looks like
            # a stream that simply ends, which is indistinguishable from a quiet
            # one to whoever is awaiting the queue.
            print(f"[event stream from {self.url} closed by the daemon]", file=sys.stderr)
        except asyncio.CancelledError:
            raise
        except Exception as ended:      # daemon gone, connection dropped
            # Diagnostics go to stderr here as everywhere in this layer: the
            # queue carries events, so a failure reported into it would have to
            # be an event that is not one, and silence would leave a caller
            # waiting on a stream that no longer exists.
            print(f"[event stream from {self.url} ended: {ended}]", file=sys.stderr)


class EmbeddedClient(LocalAgentService):
    """The local service, owning the app it composed."""

    app: AgentApp

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
