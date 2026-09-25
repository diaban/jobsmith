"""The inbound port: one interface, two backings, no second implementation.

A front-end cannot tell where the work happens: the same calls through the
local service and through HTTP give the same answers (the `through` fixture
runs each test on both). And a caller that waits (`send`) is told what a
caller that watches (`stream`) was shown. → 0048, 0050, 0064, 0083
"""
from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage
from support import (
    PLANNED_STEPS,
    Gate,
    StubPdf,
    daemon_client_over,
    launch_call,
    make_manager,
    mock_client,
    planned_manager,
    planned_service,
    service_over,
    sse_client,
    until,
    wait_done,
)

from jobsmith.api import create_api
from jobsmith.cli.client import DaemonClient, EmbeddedClient
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.service import (
    AgentService,
    BinaryDeliverable,
    LocalAgentService,
    ServiceUnavailable,
)


def test_both_backings_fully_implement_the_port():
    for backing in (DaemonClient, EmbeddedClient):
        assert issubclass(backing, AgentService)
        assert not inspect.isabstract(backing), f"{backing.__name__} leaves the port unimplemented"


def test_the_api_adds_no_use_case_of_its_own():
    """Every route is serialization + one service call."""
    import jobsmith.api.app as api_module

    source = inspect.getsource(api_module)
    for leaked in ("create_job(", "start_job(", "ainvoke(", "__interrupt__", ".summary()"):
        assert leaked not in source, f"{leaked} belongs in the service, not the API adapter"


@pytest.fixture(params=["local", "http"])
async def through(request):
    """`through(service)` is that service, or a DaemonClient over its API."""
    opened: list[AgentService] = []

    def over(service: LocalAgentService) -> AgentService:
        client = daemon_client_over(create_api(service)) if request.param == "http" else service
        opened.append(client)
        return client

    yield over
    for client in opened:
        await client.aclose()


# The model launches one job, asking for a named document in the deployment's
# format — resolved to names before anything is shown (→ 0096).
PROPOSED = {"type": "proposal", "query": "analyse it", "rationale": "multi-step",
            "sources": [], "document_name": "chair_notes", "document_title": "Comparatif",
            "formats": ["markdown"], "from_jobs": []}


def chair_service(store, checkpointer, tmp_path, *, approval=True, sync_timeout=None):
    """`approval=True` by default: the proposal is the richest terminal the port carries."""
    return service_over(make_manager(store, checkpointer, tmp_path), [
        launch_call("analyse it", "multi-step", document_name="chair_notes",
                    document_title="Comparatif", formats=["default"]),
        AIMessage(content="launched!"),
    ], approval=approval, sync_timeout=sync_timeout)


# ------------------------------------------------------------ jobs and outputs

async def test_identical_answers_through_either_backing(store, checkpointer, tmp_path, through):
    client = through(chair_service(store, checkpointer, tmp_path))
    session_id = await client.new_session()
    assert await client.send(session_id, "please analyse it") == PROPOSED
    assert (await client.approve(session_id, True))["type"] == "message"

    (job,) = await client.list_jobs(session_id=session_id)
    finished = await wait_done(client, job["job_id"])
    assert (finished["status"], finished["session_id"]) == ("done", session_id)
    assert set(finished["results"]) == {"alpha"}
    assert (await client.resolve_job(job["job_id"][:8]))["job_id"] == job["job_id"]

    assert (await client.get_report(job["job_id"])).startswith("# ")
    assert await client.get_report("nope") is None
    assert await client.get_job("nope") is None

    # a refusal is the port's dict on both sides, never an HTTP exception
    refused = await client.resume_job(job["job_id"])
    assert refused["status"] == "done" and "expected cancelled or failed" in refused["error"]
    assert await client.resume_job("nope") == {
        "job_id": "nope", "status": "unknown", "error": "unknown job: nope"}

    outputs = await client.list_outputs(job["job_id"])
    assert [(o["role"], o["format"]) for o in outputs] == [("main", "markdown")]
    assert outputs[0]["path"] == finished["report_path"]
    assert await client.find_output(job["job_id"], outputs[0]["name"]) == outputs[0]["path"]
    assert await client.find_output(job["job_id"], "nothing.md") is None
    assert await client.list_outputs("nope") is None
    # the promise is about the FILE: deleted since, it is None on both sides
    Path(outputs[0]["path"]).unlink()
    assert await client.find_output(job["job_id"], outputs[0]["name"]) is None


async def test_a_document_that_cannot_be_produced_is_refused_by_both_backings(
    store, checkpointer, tmp_path, through
):
    """`ValueError` locally, a 400 turned back into it over HTTP. → 0055"""
    client = through(chair_service(store, checkpointer, tmp_path))
    with pytest.raises(ValueError, match="unknown report format"):
        await client.launch_job("compare them", formats=["docx"])
    with pytest.raises(ValueError, match="document name"):
        await client.launch_job("compare them", document_name="../escape")
    assert await client.list_jobs() == []

    named = await client.launch_job("compare them", document_name="chair_notes",
                                    document_title="Comparatif", formats=["markdown"])
    finished = await wait_done(client, named["job_id"])
    assert finished["document_name"] == "chair_notes"
    assert finished["report_path"].endswith("chair_notes.md")


@pytest.mark.parametrize("formats", [[], None], ids=["no-file", "silent"])
async def test_no_document_reads_the_same_on_both_backings(
    store, checkpointer, tmp_path, through, formats
):
    """`[]` and `null` each cross HTTP as themselves, and both end with no file. → 0096"""
    client = through(chair_service(store, checkpointer, tmp_path))
    job = await wait_done(client, (await client.launch_job("just answer me",
                                                           formats=formats))["job_id"])

    assert job["formats"] == formats and job["deliverable_expected"] is False
    assert job["report_path"] is None and job["error"] is None and job["final_answer"]
    assert await client.get_report(job["job_id"]) is None
    assert await client.list_outputs(job["job_id"]) == []


async def test_a_binary_deliverable_is_refused_the_same_way_by_both_backings(
    store, checkpointer, tmp_path, through
):
    """`get_report` promises a string: a PDF is refused, naming its download. → 0034"""
    service = chair_service(store, checkpointer, tmp_path)
    service.manager.reporter = StubPdf()
    client = through(service)
    job = await wait_done(client, (await client.launch_job("print it",
                                                           formats=["default"]))["job_id"])
    job_id = job["job_id"]
    assert [(o["role"], o["format"]) for o in job["outputs"]] == [("main", "pdf")]

    with pytest.raises(BinaryDeliverable) as refused:
        await client.get_report(job_id)
    assert str(refused.value) == (
        f"the main deliverable of job {job_id} is pdf, which is not text "
        f"— download it from /jobs/{job_id}/outputs/{job_id}.pdf")


async def test_a_deliverable_declared_binary_is_refused_without_reading_it(tmp_path):
    """The declared format is believed; bytes that happen to decode change nothing."""
    path = tmp_path / "j1.pdf"
    path.write_text("this decodes perfectly well", encoding="utf-8")
    job = Job(job_id="j1", status=JobStatus.DONE, query="q",
              outputs=[JobOutput(path=str(path), format="pdf", role="main")])

    class OneJob:
        async def get_job(self, job_id):
            return job if job_id == "j1" else None

    with pytest.raises(BinaryDeliverable, match=r"outputs/j1\.pdf"):
        await LocalAgentService(OneJob(), None).get_report("j1")


# ------------------------------------------------------------ live progress

@asynccontextmanager
async def _no_server():
    yield ""


@asynccontextmanager
async def _serving(app):
    """A real socket: `ASGITransport` buffers the whole body, so `/events` hangs on it."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           lifespan="off", timeout_graceful_shutdown=2))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving


async def _await_subscription(service) -> None:
    """The daemon-backed subscription connects in the background: wait for it."""
    await until(lambda: service.manager.events._subscribers,
                what="the event stream reaching the service")


@pytest.mark.parametrize("over_http", [
    pytest.param(False, id="local"),
    pytest.param(True, id="http", marks=pytest.mark.slow),   # binds a real socket
])
async def test_progress_events_reach_either_backing(store, checkpointer, tmp_path, over_http):
    """Same dicts, same order, same terminal event — the plan before any step lands."""
    service = chair_service(store, checkpointer, tmp_path)
    async with (_serving(create_api(service)) if over_http else _no_server()) as url:
        client = (DaemonClient(url, httpx.AsyncClient(base_url=url, timeout=None))
                  if over_http else service)
        try:
            queue = client.subscribe()
            await _await_subscription(service)
            job_id = (await client.launch_job("watch it", formats=["default"]))["job_id"]
            seen = []
            while not seen or seen[-1]["status"] not in ("done", "failed"):
                seen.append(await asyncio.wait_for(queue.get(), timeout=10))

            assert [e["status"] for e in seen] == [
                "queued", "running", "running", "running", "done"]
            assert all(e["job_id"] == job_id for e in seen)
            assert [e["steps_done"] for e in seen] == [[], [], [], ["alpha"], ["alpha"]]
            assert seen[-1]["report_path"].endswith(f"{job_id}.md")
            client.unsubscribe(queue)
        finally:
            await client.aclose()


# ------------------------------------------------------------ a turn

async def test_a_proposed_turn_is_the_same_flow_through_either_backing(
    store, checkpointer, tmp_path, through
):
    """The gate path: one terminal, then an approval that is a turn like any other."""
    client = through(chair_service(store, checkpointer, tmp_path))
    session_id = await client.new_session()

    proposing = [e async for e in client.stream(session_id, "please analyse it")]
    assert {"type": "tool_started", "name": "launch_job"} in proposing
    assert proposing[-1] == PROPOSED                     # a list on both sides, never a tuple
    assert [e["type"] for e in proposing].count("proposal") == 1

    answering = [e async for e in client.stream_approval(session_id, True)]
    assert {"type": "tool_finished", "name": "launch_job"} in answering
    tokens = [e["text"] for e in answering if e["type"] == "token"]
    assert len(tokens) > 1, "the answer arrived in one piece on this backing"
    streamed = "".join(tokens)
    # the run's answer, then the model's sentence — never what the tool told the model
    assert streamed.endswith("launched!") and len(streamed) > len("launched!")
    assert "ALREADY been shown" not in streamed
    assert answering[-1] == {"type": "message", "content": streamed}


async def test_a_task_runs_in_the_turn_and_send_is_told_what_stream_showed(
    store, checkpointer, tmp_path, through
):
    """The nominal path: the notice, then the job's answer as tokens; and the
    terminal of `send` is that same text, not the model's last message. → 0083"""
    client = through(chair_service(store, checkpointer, tmp_path, approval=False))
    watched = await client.new_session()
    events = [e async for e in client.stream(watched, "please analyse it")]

    (started,) = [e for e in events if e["type"] == "job_started"]
    (job,) = await client.list_jobs(session_id=watched)
    expected = {k: v for k, v in PROPOSED.items() if k != "type"}
    assert started == {"type": "job_started", "job_id": job["job_id"], **expected}

    answer = (await client.get_job(job["job_id"]))["final_answer"]
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer and answer in streamed and streamed.endswith("launched!")
    assert events[-1] == {"type": "message", "content": streamed}

    # a second session, drained by `send`: the same text, the job's answer included
    waited = await client.new_session()
    assert await client.send(waited, "please analyse it") == {
        "type": "message", "content": streamed}


LONG_QUERY = "compare the two ergonomic chairs on price, lumbar support, warranty and delivery time"


@pytest.mark.parametrize("approval", [False, True], ids=["notice", "proposal"])
async def test_the_jobs_a_run_builds_on_cross_either_backing(
    store, checkpointer, tmp_path, through, approval
):
    """Plain `{job_id, query}` dicts, the query cut on a word. → 0104"""
    manager = make_manager(store, checkpointer, tmp_path)
    first = await manager.create_job(LONG_QUERY, session_id="s-builds")
    second = await manager.create_job("price the standing desk", session_id="s-builds")
    client = through(service_over(manager, [
        launch_call("a one-pager out of both", "builds on them",
                    from_jobs=[first.job_id[:8], second.job_id[:8]]),
        AIMessage(content="launched!"),
    ], approval=approval))

    session_id = await client.new_session("s-builds")
    events = [e async for e in client.stream(session_id, "one-pager out of both")]
    (shown,) = [e for e in events if e["type"] in ("job_started", "proposal")]
    assert shown["type"] == ("proposal" if approval else "job_started")
    assert shown["from_jobs"] == [
        {"job_id": first.job_id,
         "query": "compare the two ergonomic chairs on price, lumbar support…"},
        {"job_id": second.job_id, "query": "price the standing desk"},
    ]


# ------------------------------------------------------------ the plan, in the turn → 0086

async def test_the_plan_crosses_either_backing_between_the_notice_and_the_answer(
    store, checkpointer, tmp_path, through
):
    client = through(planned_service(planned_manager(store, checkpointer, tmp_path)))
    session_id = await client.new_session()
    events = [e async for e in client.stream(session_id, "please analyse it")]
    (job,) = await client.list_jobs(session_id=session_id)

    (planned,) = [e for e in events if e["type"] == "job_planned"]
    assert planned == {"type": "job_planned", "job_id": job["job_id"], "steps": PLANNED_STEPS}
    kinds = [e["type"] for e in events]
    assert kinds.index("job_started") < kinds.index("job_planned") < kinds.index("token")
    assert "analysis" not in events[-1]["content"], "the plan leaked into the reply"


async def test_the_plan_is_shown_while_the_run_is_still_going(store, checkpointer, tmp_path):
    """The first step is held until the plan is shown: a late plan hangs, not passes."""
    gate = Gate("web_search")
    manager = planned_manager(store, checkpointer, tmp_path, gate=gate)
    service = planned_service(manager)
    session_id = await service.new_session()
    seen: list[dict] = []

    async def turn():
        async for event in service.stream(session_id, "please analyse it"):
            seen.append(event)
            if event["type"] == "job_planned":
                (job,) = await service.list_jobs(session_id=session_id)
                assert job["status"] == "running"
                assert not any(e["type"] == "token" for e in seen)
                gate.open.set()

    await asyncio.wait_for(turn(), timeout=10)
    assert gate.open.is_set(), "the turn ended without ever showing the plan"
    assert not manager.events._subscribers, "the turn left its subscription behind"


async def test_a_run_faster_than_its_watch_still_has_its_plan_said_first(
    store, checkpointer, tmp_path, monkeypatch
):
    manager = planned_manager(store, checkpointer, tmp_path)

    class Lagging(asyncio.Queue):
        async def get(self):
            await asyncio.sleep(0.05)
            return await super().get()

    def lagging_subscribe(*, max_queue=256):
        queue = Lagging(maxsize=max_queue)
        manager.events._subscribers.add(queue)
        return queue

    monkeypatch.setattr(manager, "subscribe", lagging_subscribe)
    service = planned_service(manager)
    session_id = await service.new_session()
    kinds = [e["type"] async for e in service.stream(session_id, "please analyse it")]
    assert "job_planned" in kinds, "the plan was lost to a watch that fell behind"
    assert kinds.index("job_planned") < kinds.index("token")


async def test_a_one_step_plan_and_a_direct_answer_announce_nothing(
    store, checkpointer, tmp_path
):
    one_step = chair_service(store, checkpointer, tmp_path, approval=False)
    session_id = await one_step.new_session()
    kinds = [e["type"] async for e in one_step.stream(session_id, "please analyse it")]
    assert "job_started" in kinds and "job_planned" not in kinds

    direct = planned_service(planned_manager(store, checkpointer, tmp_path),
                             responses=[AIMessage(content="Paris.")])
    session_id = await direct.new_session()
    kinds = [e["type"] async for e in direct.stream(session_id, "capital of France?")]
    assert kinds == ["token"] * (len(kinds) - 1) + ["message"]


async def test_a_promoted_run_says_nothing_of_its_plan_once_the_turn_is_over(
    store, checkpointer, tmp_path
):
    """The plan travels only while the turn waits; it stays on the record."""
    gate = Gate("web_search")
    manager = planned_manager(store, checkpointer, tmp_path, gate=gate)
    service = planned_service(manager, sync_timeout=0)
    session_id = await service.new_session()
    kinds = [e["type"] async for e in service.stream(session_id, "please analyse it")]
    assert "job_started" in kinds and "job_planned" not in kinds
    assert not manager.events._subscribers, "the watch outlived the turn"

    (job,) = await service.list_jobs(session_id=session_id)
    async def planned():                       # the run goes on, unwatched
        return (await service.get_job(job["job_id"]))["plan"]

    await until(planned, what="the plan reaching the record")
    gate.open.set()
    finished = await wait_done(service, job["job_id"])
    assert [s["capability"] for s in finished["plan"]["steps"]] == [
        s["capability"] for s in PLANNED_STEPS]


# ------------------------------------------------------------ the event reader → 0048

async def test_the_event_reader_survives_a_line_it_cannot_read():
    """A comment or a non-JSON `data:` is skipped, not the end of the stream."""
    client = sse_client(": keep-alive", "", "data: {not json at all}", "",
                        'data: {"job_id": "a", "status": "running"}', "",
                        'data: {"job_id": "a", "status": "done"}', "")
    try:
        queue = client.subscribe()
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert [queue.get_nowait() for _ in range(queue.qsize())] == [
            {"job_id": "a", "status": "running"}, {"job_id": "a", "status": "done"}, None]
    finally:
        await client.aclose()


async def test_a_slow_consumer_loses_events_but_never_the_end_of_the_stream():
    """Events are dropped rather than stalling the reader; `None` takes the
    oldest tick's place, since nothing supersedes "there is no next one"."""
    client = sse_client(*[line for i in range(5) for line in (f'data: {{"job_id": "{i}"}}', "")])
    try:
        queue = client.subscribe(max_queue=2)
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert [queue.get_nowait() for _ in range(2)] == [{"job_id": "1"}, None]
    finally:
        await client.aclose()


async def test_unsubscribing_releases_the_stream_the_subscription_held():
    client = sse_client('data: {"job_id": "a"}', "")
    queue = client.subscribe()
    reader = client._readers[queue]
    client.unsubscribe(queue)
    assert queue not in client._readers
    await client.aclose()
    assert reader.cancelled() or reader.done()


async def test_a_stream_that_ends_is_announced_rather_than_going_quiet(capsys):
    """On stderr for a human, and `None` on the queue for a front-end that owns the terminal."""
    client = sse_client('data: {"job_id": "a"}', "")
    try:
        queue = client.subscribe()
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert queue.get_nowait() == {"job_id": "a"}
        assert "closed by the daemon" in capsys.readouterr().err
        assert queue.get_nowait() is None and queue.empty()
    finally:
        await client.aclose()


# ------------------------------------------------------------ a backing that is not there → 0064

def _port_calls(client: AgentService) -> dict:
    """One per use case: a method added without a translation is a missing key."""
    return {
        "new_session": lambda: client.new_session(),
        "launch_job": lambda: client.launch_job("q"),
        "list_jobs": lambda: client.list_jobs(),
        "get_job": lambda: client.get_job("a"),
        "cancel_job": lambda: client.cancel_job("a"),
        "resume_job": lambda: client.resume_job("a"),
        "get_report": lambda: client.get_report("a"),
        "list_outputs": lambda: client.list_outputs("a"),
        "find_output": lambda: client.find_output("a", "f.md"),
        "send": lambda: client.send("s", "hello"),
        "approve": lambda: client.approve("s", True),
    }


async def test_a_backing_that_cannot_be_reached_is_refused_by_the_port():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = mock_client(refused)
    try:
        for name, call in _port_calls(client).items():
            with pytest.raises(ServiceUnavailable) as gone:
                await call()
            assert "http://test" in str(gone.value), f"{name} does not say what it cannot reach"
    finally:
        await client.aclose()


async def test_a_daemon_that_answers_badly_is_not_called_unreachable():
    """A 500 is a defect, on the plain and on the streamed path."""
    client = mock_client(lambda r: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.list_jobs()
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("s", "hello")
    finally:
        await client.aclose()


async def test_the_local_backing_never_dresses_a_bug_as_an_absent_backing():
    class Broken:
        async def list_jobs(self, **kwargs):
            raise KeyError("a defect in this process")

    with pytest.raises(KeyError):
        await LocalAgentService(Broken(), None).list_jobs()
