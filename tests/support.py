"""Builders shared across test files: stub capabilities, a manager, a chat
session, an API, a finished job. A test file imports from here, never from
another test file."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeLLM, ScriptedChatModel, plan_json
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END

from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.usage import Usage
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import Job, JobStatus
from jobsmith.jobs.report import FileReporter, JobDocument, PlanRow

ANSWER = "A sufficiently long final answer for the job test."
CFG = {"configurable": {"thread_id": "chat-1"}}


# ------------------------------------------------------------ capabilities

class SlowEcho(Capability):
    def __init__(self, name: str, *, delay: float = 0.0, fail: bool = False):
        self.spec = CapabilitySpec(name=name, description=f"{name} capability")
        self.delay = delay
        self.fail = fail

    async def work(self, state: CapabilityBaseState) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return self._emit_failure(f"{self.spec.name} broke")
        return self._emit_success({"echo": self.spec.name})

    def render_context(self, result):
        return f"# {self.spec.name}\n{result['data']['echo']}"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


class CountingEcho(SlowEcho):
    """Echoes how many times it has run, so a re-run shows up in the result."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.runs = 0

    async def work(self, state: CapabilityBaseState) -> dict:
        self.runs += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._emit_success({"echo": f"{self.spec.name}#{self.runs}"})


class Gate(SlowEcho):
    """A step that holds the run until the test opens it."""

    def __init__(self, name: str):
        super().__init__(name)
        self.open = asyncio.Event()

    async def work(self, state):
        await self.open.wait()
        return await super().work(state)


# ------------------------------------------------------------ waiting

async def until(check, *, what: str, seconds: float = 5.0):
    """Poll `check` (sync or async) until it returns something truthy, and
    return that; fail naming `what` never happened. Wait on the state a test
    is about, never on a fixed sleep."""
    for _ in range(int(seconds / 0.01)):
        value = check()
        if inspect.isawaitable(value):
            value = await value
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError(f"{what} never happened")


# ------------------------------------------------------------ the engine

def make_manager(
    store, checkpointer, tmp_path, *, caps=None, llm=None,
    document_formats: tuple[str, ...] = (), default_formats: tuple[str, ...] = ("markdown",),
) -> JobManager:
    """A manager over stub capabilities whose planner plans them all.

    `document_formats` wires the document step (off by default: no format to
    choose from, so it asks nothing)."""
    caps = caps if caps is not None else [SlowEcho("alpha")]
    llm = llm or FakeLLM({"planner": plan_json(*[c.spec.name for c in caps])}, default=ANSWER)
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer,
                        document_formats=document_formats,
                        default_document_formats=default_formats if document_formats else ())
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts",
                      default_formats=default_formats)


def direct_llm(**script: str) -> FakeLLM:
    """A model whose triage answers `direct` — the route of a greeting."""
    return FakeLLM({"triage step": '{"route": "direct", "rationale": "trivial"}', **script},
                   default=ANSWER)


def files_under(tmp_path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "artifacts").rglob("*") if p.is_file())


async def cancelled_midway(store, checkpointer, tmp_path, *, session_id=None, formats=None):
    """A job stopped inside its second step: one result stored, one pending.

    Returns the manager, the job and both capabilities (set `slow.delay = 0`
    to let the interrupted step finish instantly on resume)."""
    alpha, slow = CountingEcho("alpha"), CountingEcho("slow", delay=30.0)
    llm = FakeLLM({"planner": plan_json("alpha", "slow", deps={"slow": ["alpha"]})},
                  default=ANSWER)
    mgr = make_manager(store, checkpointer, tmp_path, caps=[alpha, slow], llm=llm)
    job = await mgr.create_job("a job worth resuming", session_id=session_id, formats=formats)
    mgr.start_job(job.job_id)
    await until(lambda: slow.runs, what="the second step starting")
    stopped = await mgr.cancel_job(job.job_id)
    assert stopped.status is JobStatus.CANCELLED
    assert set(stopped.results) == {"alpha"}   # the finished step was persisted
    return mgr, job, alpha, slow


PLANNED_DEPS = {"research": ["web_search", "documents"], "analysis": ["research"]}
PLANNED_STEPS = [
    {"capability": "web_search", "depends_on": []},
    {"capability": "documents", "depends_on": []},
    {"capability": "research", "depends_on": ["web_search", "documents"]},
    {"capability": "analysis", "depends_on": ["research"]},
]


def planned_manager(store, checkpointer, tmp_path, *, gate: Gate | None = None):
    """A manager whose planner answers `PLANNED_STEPS`; `gate`, if given,
    stands in for the first step, so the run cannot finish before it opens."""
    caps = [gate or SlowEcho("web_search"), SlowEcho("documents"),
            SlowEcho("research"), SlowEcho("analysis")]
    llm = FakeLLM({"planner": plan_json(*[c.spec.name for c in caps], deps=PLANNED_DEPS)},
                  default="A sufficiently long final answer for the plan test.")
    return make_manager(store, checkpointer, tmp_path, caps=caps, llm=llm)


# ------------------------------------------------------------ the chat

def launch_call(query: str, rationale: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "launch_job",
        "args": {"query": query, "rationale": rationale, **args},
        "id": "call_1",
    }])


def make_session(
    store, checkpointer, tmp_path, responses, *, llm=None,
    approval=False, sync_timeout=None, inline_answer_max=None,
):
    from jobsmith.chat import ChatSession

    manager = make_manager(store, checkpointer, tmp_path, llm=llm)
    model = ScriptedChatModel(responses=responses)
    session = ChatSession(manager, model, checkpointer=MemorySaver(),
                          approval_required=approval, sync_timeout=sync_timeout,
                          inline_answer_max=inline_answer_max)
    return session, model


def service_over(manager, responses, *, approval=False, sync_timeout=None):
    """A `LocalAgentService` whose every session replays `responses`."""
    from jobsmith.chat import ChatSession
    from jobsmith.service import LocalAgentService

    saver = MemorySaver()

    def session_factory(session_id: str | None = None) -> ChatSession:
        return ChatSession(manager, ScriptedChatModel(responses=list(responses)),
                           session_id=session_id, checkpointer=saver,
                           approval_required=approval, sync_timeout=sync_timeout)

    return LocalAgentService(manager, session_factory)


def planned_service(manager, *, sync_timeout=None, responses=None):
    return service_over(manager, responses or [launch_call("analyse it", "several steps"),
                                               AIMessage(content="done.")],
                        sync_timeout=sync_timeout)


# ------------------------------------------------------------ over HTTP

def make_app(store, checkpointer, tmp_path, responses, *, approval=False):
    from jobsmith.api import create_api

    manager = make_manager(store, checkpointer, tmp_path)
    return create_api(service_over(manager, responses, approval=approval)), manager


def client_for(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def daemon_client_over(app):
    from jobsmith.cli.client import DaemonClient

    http = AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=None)
    return DaemonClient("http://test", http)


async def wait_done(client: Any, job_id: str) -> dict:
    """The job's dict once DONE or FAILED — over raw HTTP or through the port."""
    async def settled():
        if isinstance(client, AsyncClient):
            job = (await client.get(f"/jobs/{job_id}")).json()
        else:
            job = await client.get_job(job_id)
        return job if job and job["status"] in ("done", "failed") else None

    return await until(settled, what=f"job {job_id} settling")


def mock_client(handler):
    """A DaemonClient over `httpx.MockTransport(handler)` — no app behind it."""
    import httpx

    from jobsmith.cli.client import DaemonClient

    return DaemonClient("http://test", httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test", timeout=None))


def sse_client(*lines: str):
    """A DaemonClient whose every streamed route answers exactly these lines."""
    import httpx

    body = "".join(f"{line}\n" for line in lines)
    return mock_client(lambda request: httpx.Response(200, text=body))


async def wait_settled(manager, job_id: str):
    """The `Job` (not a dict) once DONE or FAILED."""
    async def settled():
        job = await manager.get_job(job_id)
        return job if job and job.status in (JobStatus.DONE, JobStatus.FAILED) else None

    return await until(settled, what=f"job {job_id} settling")


# ------------------------------------------------------------ reports

class StubPdf(FileReporter):
    """A binary Reporter with no engine behind it: bytes on disk, declared
    binary — for properties of the port or manager, not of WeasyPrint."""

    format = "pdf"
    extension = "pdf"
    binary = True

    def serialize(self, document: JobDocument, path: Path) -> None:
        path.write_bytes(b"%PDF-1.7\n\xe2\xe3\xcf\xd3 not text\n%%EOF\n")


def make_document(**over) -> JobDocument:
    """The archive-shaped document (`provenance=True`) unless a test says otherwise."""
    doc = JobDocument(
        provenance=True,
        title="compare A and B",
        request="compare A and B",
        job_id="j1",
        created_at="2026-09-01T00:00:00Z",
        finished_at="2026-09-01T00:03:00Z",
        answer="## Verdict\n\nA beats B.",
        plan_rationale="chain of three",
        plan=[
            PlanRow("research", [], "ok", "t1",
                    Usage(input_tokens=12_000, output_tokens=3_000, calls=2,
                          cost_usd=0.135, models=("claude-opus-5",))),
            PlanRow("analysis", ["research"], "ok", "t2"),
            PlanRow("critique", ["analysis"], "failed (boom)", "t3"),
            PlanRow("aside", [], "not run", "—"),
        ],
        usage=Usage(input_tokens=20_000, output_tokens=5_000, calls=6,
                    cost_usd=0.225, models=("claude-opus-5",)),
    )
    for key, value in over.items():
        setattr(doc, key, value)
    return doc


def done_job(job_id: str = "j10") -> Job:
    return Job(job_id=job_id, status=JobStatus.DONE, query="compare A and B",
               created_at="2026-09-01T00:00:00Z", final_answer="A beats B.")


# ------------------------------------------------------------ markers

PDF_MARKS = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("weasyprint") is None,
                       reason="the optional .[pdf] extra is not installed"),
]


def requires_pdf(func):
    """Needs the real engine: skipped without `.[pdf]`, and `slow` — the first
    import in a process costs ~4 s (→ 0109). `PDF_MARKS` for a `pytest.param`."""
    for mark in PDF_MARKS:
        func = mark(func)
    return func


# ------------------------------------------------------------ the default pack

PACK_SCRIPT = {
    "key aspects": '{"aspects": ["history", "impact"]}',
    "research notes": "Notes: the history is long; the impact is broad.",
    "You are an analyst": "Findings: impact outweighs history.",
    "checking the findings": "- The impact claim: the notes give no numbers for it.",
}


def notes_call(llm: FakeLLM) -> dict:
    """The call that wrote the research notes, found by its system prompt."""
    from jobsmith.agents.default.research import ResearchCapability

    return next(c for c in llm.calls if c["messages"][0]["content"].startswith(
        (ResearchCapability.NOTES_SYSTEM, ResearchCapability.GROUNDED_NOTES_SYSTEM)))


# ------------------------------------------------------------ a broken PDF engine

def no_distribution(monkeypatch) -> None:
    """`import weasyprint` halts with ImportError: the extra is not installed."""
    import sys

    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setitem(sys.modules, "weasyprint", None)


def no_libraries(monkeypatch) -> None:
    """The distribution is found and its import raises OSError, as cffi does
    when it cannot dlopen pango: PDF is offered and cannot be rendered."""
    import importlib.util
    import sys

    from jobsmith.jobs import report_pdf

    real_find_spec = importlib.util.find_spec

    class NoLibraries:
        def find_spec(self, name, path=None, target=None):
            if name == "weasyprint":
                raise OSError("cannot load library 'libpango-1.0.so.0'")
            return None

    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoLibraries(), *sys.meta_path])
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec",
                        lambda name, *a: object() if name == "weasyprint"
                        else real_find_spec(name, *a))
