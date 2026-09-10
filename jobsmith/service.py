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
so the local and remote backings are indistinguishable to a caller. A *turn*
is a flow of those dicts (`stream`), and a reply is simply the one it ends on:
`send` and `approve` are defined here over `stream`, so no backing gets to
drive a turn its own way.
"""
from __future__ import annotations

import asyncio
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .chat.runner import ChatEvent, ChatRunner, Message, Proposal, Token, ToolFinished, ToolStarted
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


class ServiceUnavailable(RuntimeError):
    """The backing that answers this port could not be reached.

    The third thing the port narrows, for the same reason as the other two:
    a front-end must be able to write ONE handler for it, whichever backing
    it holds. Without it a UI would have to know that a remote backing
    answers with `httpx.ConnectError` — which is a fact about the transport,
    not about the use case — and a broad `except` around every call would be
    the only alternative, which is a stance rather than a fix: it swallows
    our own bugs along with the daemon's absence.

    So exactly one thing is translated, and it is narrow: **the request did
    not reach the backing, or the connection died before it answered**
    (`httpx.TransportError`). A daemon that answered — with a 500, or with a
    body nobody can parse — is *there*, and that is a defect worth a
    traceback: it surfaces unchanged. Nothing on the local side is
    translated at all, because a process cannot lose contact with itself: a
    `KeyError` from a store stays a `KeyError`, and must, or a reconnect
    message would be where a stack trace belongs.

    That asymmetry is the same one `subscribe`'s `None` marker already has —
    embedded it never comes — and it is a fact about a backing, not a licence
    for a caller holding the port to skip the case. What the port promises is
    that this is the ONLY way "the agent is not there" can arrive.
    """

    @classmethod
    def reaching(cls, where: str, cause: BaseException) -> ServiceUnavailable:
        return cls(f"cannot reach the agent at {where}: {cause}")


class ChatStreamError(RuntimeError):
    """A streamed turn did not arrive whole.

    The opposite of the rule `/events` follows, and deliberately so. A missed
    progress tick costs nothing, so `subscribe` drops when a consumer is slow.
    A missed **token** is a lie: the sentence arrives shorter than the model
    wrote it and nothing in the text says so — the reader believes an answer
    that was never given. So the chat stream never drops. It back-pressures
    (there is no queue between the model and the reader to overflow), and when
    it cannot deliver — a line it cannot decode, a stream that ends before a
    terminal — it says so here rather than handing back a truncated turn.
    """


# The five domain events of `chat/runner.py`, as the dicts the port carries.
# Dicts because they cross HTTP: the two backings must be indistinguishable,
# and a front-end deserializing a dataclass would be a third implementation.
# The two terminal shapes are byte-for-byte what `send`/`approve` have always
# returned, which is what lets `send` be *defined* as draining this stream.
_EVENT_TYPES: dict[type, str] = {
    Token: "token",
    ToolStarted: "tool_started",
    ToolFinished: "tool_finished",
    Message: "message",
    Proposal: "proposal",
}

TERMINAL_EVENTS = ("message", "proposal")


def as_event(event: ChatEvent) -> dict:
    """One flow event as the dict the port carries."""
    return {"type": _EVENT_TYPES[type(event)]} | dataclasses.asdict(event)


async def terminal_of(events: AsyncIterator[dict]) -> dict:
    """Drain a turn's events and return the reply it ended on.

    The whole of `send`/`approve`: there is one implementation of a turn, and
    it is the stream. A stream that ends without a terminal is refused rather
    than answered with the last thing seen — see `ChatStreamError`.
    """
    terminal = None
    async for event in events:
        if event.get("type") in TERMINAL_EVENTS:
            terminal = event
    if terminal is None:
        raise ChatStreamError(
            "the turn ended without a reply — the stream was cut short")
    return terminal


class AgentService(ABC):
    """What any front-end needs. Dict shapes match the HTTP API.

    Three exceptions are part of this interface, and they are the whole of
    what a caller may plan for: `BinaryDeliverable` (`get_report` was asked
    for a file that is not text), `ChatStreamError` (a turn did not arrive
    whole) and `ServiceUnavailable` (the backing could not be reached).
    Every one of them reads identically on both backings, which is what lets
    a front-end handle them by name instead of guarding every call. Anything
    else that escapes a call is a bug in this process — it is deliberately
    not translated, so it surfaces as itself.

    `subscribe` is the exception to the exception: it says the same fact as a
    value, because a queue is what its consumer is awaiting (see below).
    """

    mode: str = "local"
    persistent: bool = False   # do jobs outlive this process?

    # -- conversation --

    @abstractmethod
    async def new_session(self, session_id: str | None = None) -> str: ...

    @abstractmethod
    def stream(self, session_id: str, text: str) -> AsyncIterator[dict]:
        """One turn, as it happens: the events of `chat/runner.py`, as dicts.

        The primitive, not a variant of `send`. A turn is a flow — tokens,
        tool activity, then exactly one terminal — and `send` is that flow
        drained, so the two can never disagree about what a turn produced.

        Never drops. Where `subscribe` sheds events under back-pressure, this
        one blocks the producer or raises `ChatStreamError`: a progress tick
        nobody saw is invisible, a token nobody saw is a shorter sentence that
        reads as complete.
        """
        ...

    @abstractmethod
    def stream_approval(self, session_id: str, approved: bool) -> AsyncIterator[dict]:
        """Answer a pending proposal and stream the turn that follows."""
        ...

    async def send(self, session_id: str, text: str) -> dict:
        """The turn's reply: a message, or a proposal to approve.

        Concrete, on the port itself, so both backings answer from the same
        code — the stream is the only place a turn is driven.
        """
        return await terminal_of(self.stream(session_id, text))

    async def approve(self, session_id: str, approved: bool) -> dict:
        return await terminal_of(self.stream_approval(session_id, approved))

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
        (embedded) or a reader (remote) blocked behind it. So an event says
        *something changed*, never *what* changed by itself: a consumer
        re-reads (`list_jobs`, `get_job`) rather than accumulating deltas,
        and a dropped tick then costs nothing but a slightly later repaint.

        `None` is the one value that is not an event: it means **no more will
        arrive on this queue**, and it is the last thing the queue carries.
        A stream that ends looks exactly like a stream with nothing to say,
        and a front-end that cannot tell them apart freezes without saying so
        — a diagnostic on stderr does not reach a screen a UI has taken over.
        It is never dropped, because there is no next event to supersede it;
        embedded it never comes (the in-process fan-out lives as long as the
        service), which is a fact about that backing and not a licence for a
        caller holding the port to skip the case.
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

    def _runner(self, session_id: str) -> ChatRunner:
        """Sessions are rebuildable: this registry is only a cache, the actual
        conversation lives in the checkpointer under thread_id=session_id. So a
        client can keep chatting on its session id across a daemon restart."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self.session_factory(session_id).build()
        return ChatRunner(self._sessions[session_id])

    async def new_session(self, session_id: str | None = None) -> str:
        session = self.session_factory(session_id) if session_id else self.session_factory()
        self._sessions[session.session_id] = session.build()
        return session.session_id

    async def stream(self, session_id: str, text: str) -> AsyncIterator[dict]:
        """Nothing between the runner and the caller — no queue, no buffer.

        Which IS the no-drop rule: an async generator delivers at the pace it
        is consumed, so a slow reader slows the turn instead of losing part of
        it. `subscribe` needs a queue because a job runs whether anyone
        watches or not; a turn has exactly one reader, and it is waiting.
        """
        async for event in self._runner(session_id).stream(session_id, text):
            yield as_event(event)

    async def stream_approval(self, session_id: str, approved: bool) -> AsyncIterator[dict]:
        async for event in self._runner(session_id).resume(session_id, approved):
            yield as_event(event)

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
