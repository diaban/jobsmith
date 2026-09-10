"""The inbound port: one interface, two backings, no second implementation.

The property worth protecting is that a front-end cannot tell where the work
happens. So the same sequence is driven through the local service and through
HTTP, and the answers must match — not merely "both work".
"""
from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_cli import daemon_client_over, wait_done
from test_jobs import make_manager
from test_report_pdf import StubPdf

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
    """Every route is serialization + one service call; the API module must
    not grow its own chat or job logic again."""
    import jobsmith.api.app as api_module

    source = inspect.getsource(api_module)
    for leaked in ("create_job(", "start_job(", "ainvoke(", "__interrupt__", ".summary()"):
        assert leaked not in source, f"{leaked} belongs in the service, not the API adapter"


def _service_over(store, checkpointer, tmp_path):
    manager = make_manager(store, checkpointer, tmp_path)
    saver = MemorySaver()
    responses = [launch_call("analyse it", "multi-step"), AIMessage(content="launched!")]

    def session_factory(session_id=None):
        from jobsmith.chat import ChatSession
        return ChatSession(manager, ScriptedChatModel(responses=list(responses)),
                           session_id=session_id, checkpointer=saver)

    return LocalAgentService(manager, session_factory)


@asynccontextmanager
async def _no_server():
    """The local backing needs no server — it is the service itself."""
    yield ""


@asynccontextmanager
async def _serving(app):
    """The same API on a real socket, for the one thing ASGI cannot carry.

    httpx's `ASGITransport` buffers the whole response body before handing it
    back, so a request to `/events` — a body that never ends — returns
    nothing and the test hangs. Every other HTTP test in this repo uses that
    transport and should; this one cannot, and the next person to "simplify"
    it back will get a hang, not a failure.
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning",
        lifespan="off", timeout_graceful_shutdown=2,
    ))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:                      # bound by the test timeout
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving


async def _await_subscription(service) -> None:
    """Both backings subscribe, but only one of them does it locally.

    `LocalAgentService.subscribe` registers the queue before it returns; the
    daemon-backed one hands back a queue and connects in the background, so
    the events published before that connection lands are events nobody
    asked for yet. The test drives the service directly, so it can simply
    wait for the subscriber to show up rather than race it.
    """
    for _ in range(500):
        if service.manager.events._subscribers:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the event stream never reached the service")


def _scripted_client(*lines: str) -> DaemonClient:
    """A DaemonClient whose /events answers with exactly these SSE lines."""
    body = "".join(f"{line}\n" for line in lines)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    return DaemonClient("http://test", httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=None))


@pytest.mark.parametrize("over_http", [False, True], ids=["local", "http"])
async def test_identical_answers_through_either_backing(
    store, checkpointer, tmp_path, over_http
):
    service = _service_over(store, checkpointer, tmp_path)
    client = daemon_client_over(create_api(service)) if over_http else service
    try:
        session_id = await client.new_session()
        assert isinstance(session_id, str) and session_id

        reply = await client.send(session_id, "please analyse it")
        assert reply == {"type": "proposal", "query": "analyse it",
                         "rationale": "multi-step", "sources": []}

        approved = await client.approve(session_id, True)
        assert approved["type"] == "message"

        (job,) = await client.list_jobs(session_id=session_id)
        finished = await wait_done(client, job["job_id"])
        assert finished["status"] == "done"
        assert finished["session_id"] == session_id
        assert set(finished["results"]) == {"alpha"}

        # a short prefix resolves the same way on both sides
        by_prefix = await client.resolve_job(job["job_id"][:8])
        assert by_prefix["job_id"] == job["job_id"]

        report = await client.get_report(job["job_id"])
        assert report.startswith("# ")
        assert await client.get_report("nope") is None
        assert await client.get_job("nope") is None

        # a refusal must read the same on both sides: the HTTP status code is
        # translated back into the port's dict, never leaked as an exception
        refused = await client.resume_job(job["job_id"])
        assert refused["status"] == "done"
        assert "expected cancelled or failed" in refused["error"]
        assert (await client.resume_job("nope")) == {
            "job_id": "nope", "status": "unknown", "error": "unknown job: nope"}

        # what the job produced: the same list, and the same locator for one
        outputs = await client.list_outputs(job["job_id"])
        assert [(o["role"], o["format"]) for o in outputs] == [("main", "markdown")]
        assert outputs[0]["path"] == finished["report_path"]
        assert await client.find_output(job["job_id"], outputs[0]["name"]) == outputs[0]["path"]
        assert await client.find_output(job["job_id"], "nothing.md") is None
        assert await client.list_outputs("nope") is None

        # ...and the promise is about the FILE, not the record: a deliverable
        # deleted since the job finished is None on both sides, which is why
        # the remote backing asks the daemon instead of reading its own copy
        # of `outputs`.
        Path(outputs[0]["path"]).unlink()
        assert await client.find_output(job["job_id"], outputs[0]["name"]) is None
    finally:
        await client.aclose()


@pytest.mark.parametrize("over_http", [False, True], ids=["local", "http"])
async def test_a_binary_deliverable_is_refused_the_same_way_by_both_backings(
    store, checkpointer, tmp_path, over_http
):
    """`get_report` promises a string. A PDF has no reading as one, and both
    of the silent answers would be false — `None` says the job has no report,
    and decoding it says nothing intelligible. So the port refuses and names
    the download, and it must refuse in the same words whether the job ran in
    this process or behind a daemon: over HTTP that is a 415 the client turns
    back into the same exception.
    """
    service = _service_over(store, checkpointer, tmp_path)
    service.manager.reporter = StubPdf()
    client = daemon_client_over(create_api(service)) if over_http else service
    try:
        launched = await client.launch_job("print it")
        job = await wait_done(client, launched["job_id"])
        job_id = job["job_id"]
        assert [(o["role"], o["format"]) for o in job["outputs"]] == [("main", "pdf")]

        with pytest.raises(BinaryDeliverable) as refused:
            await client.get_report(job_id)
        assert str(refused.value) == (
            f"the main deliverable of job {job_id} is pdf, which is not text "
            f"\u2014 download it from /jobs/{job_id}/outputs/{job_id}.pdf"
        )
    finally:
        await client.aclose()


async def test_a_deliverable_declared_binary_is_refused_without_reading_it(tmp_path):
    """The declared format is believed on its own — the refusal must not hang
    on the bytes happening to fail a decode. A PDF whose first kilobyte is
    valid UTF-8 would otherwise be printed to a terminal, and the job's own
    word for what it wrote is the cheaper and the earlier answer."""
    path = tmp_path / "j1.pdf"
    path.write_text("this decodes perfectly well", encoding="utf-8")
    job = Job(job_id="j1", status=JobStatus.DONE, query="q",
              outputs=[JobOutput(path=str(path), format="pdf", role="main")])

    class OneJob:
        async def get_job(self, job_id):
            return job if job_id == "j1" else None

    service = LocalAgentService(OneJob(), None)
    with pytest.raises(BinaryDeliverable, match=r"outputs/j1\.pdf"):
        await service.get_report("j1")


@pytest.mark.parametrize("over_http", [False, True], ids=["local", "http"])
async def test_progress_events_reach_either_backing(store, checkpointer, tmp_path, over_http):
    """A front-end must not have to ask which backing it holds to see a job move.

    `subscribe` was in-process only until #48: the daemon published to
    `/events` and nothing consumed it, so live progress worked embedded and
    not at all against a daemon. Both now hand back a queue of the same
    dicts, in the same order, ending on the same terminal event.
    """
    service = _service_over(store, checkpointer, tmp_path)
    async with (_serving(create_api(service)) if over_http else _no_server()) as url:
        client = (DaemonClient(url, httpx.AsyncClient(base_url=url, timeout=None))
                  if over_http else service)
        try:
            queue = client.subscribe()
            await _await_subscription(service)

            launched = await client.launch_job("watch it")
            job_id = launched["job_id"]
            seen = []
            while not seen or seen[-1]["status"] not in ("done", "failed"):
                seen.append(await asyncio.wait_for(queue.get(), timeout=10))

            assert [e["status"] for e in seen] == [
                "queued", "running", "running", "running", "done"]
            assert all(e["job_id"] == job_id for e in seen)
            # Four events before a step lands, and the third is the plan: a
            # watcher drawing the DAG sees it as soon as it exists, rather
            # than waiting out the first step for a picture it already has.
            assert [e["steps_done"] for e in seen] == [[], [], [], ["alpha"], ["alpha"]]
            assert seen[-1]["report_path"].endswith(f"{job_id}.md")

            client.unsubscribe(queue)
        finally:
            await client.aclose()


@pytest.mark.parametrize("over_http", [False, True], ids=["local", "http"])
async def test_a_turn_is_the_same_flow_through_either_backing(
    store, checkpointer, tmp_path, over_http
):
    """A turn is a flow, and both backings emit the same one.

    Not "both work": the same events, in the same order, ending on the same
    terminal — because a front-end that rendered a daemon-backed turn
    differently from an embedded one would be written against two ports. The
    approval round trip is streamed too: a post-approval reply is a turn like
    any other.
    """
    service = _service_over(store, checkpointer, tmp_path)
    client = daemon_client_over(create_api(service)) if over_http else service
    try:
        session_id = await client.new_session()

        proposing = [e async for e in client.stream(session_id, "please analyse it")]
        assert {"type": "tool_started", "name": "launch_job"} in proposing
        # `sources` rides on the terminal and must survive the HTTP round
        # trip as the same JSON — a list on both sides, never a tuple.
        assert proposing[-1] == {"type": "proposal", "query": "analyse it",
                                 "rationale": "multi-step", "sources": []}

        answering = [e async for e in client.stream_approval(session_id, True)]
        assert {"type": "tool_finished", "name": "launch_job"} in answering
        tokens = [e["text"] for e in answering if e["type"] == "token"]
        assert len(tokens) > 1, "the answer arrived in one piece on this backing"
        assert "".join(tokens) == "launched!"
        assert answering[-1] == {"type": "message", "content": "launched!"}

        # ...and `send` is that same flow drained, on either backing
        assert await client.send(session_id, "anything else?") == {
            "type": "message", "content": "launched!"}
    finally:
        await client.aclose()


async def test_the_event_reader_survives_a_line_it_cannot_read():
    """One unreadable line must not end the stream.

    Scripted through `httpx.MockTransport` rather than a server because the
    point is a body a well-behaved daemon does not send: an SSE comment, and
    a `data:` that is not JSON. Dropping the connection there would cost the
    caller every event after it, which is a worse answer than skipping the
    line nobody can interpret.
    """
    client = _scripted_client(
        ": keep-alive", "",
        "data: {not json at all}", "",
        'data: {"job_id": "a", "status": "running"}', "",
        'data: {"job_id": "a", "status": "done"}', "",
    )
    try:
        queue = client.subscribe()
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert [queue.get_nowait() for _ in range(queue.qsize())] == [
            {"job_id": "a", "status": "running"},
            {"job_id": "a", "status": "done"},
            None,                       # ... and then the stream ended
        ]
    finally:
        await client.aclose()


async def test_a_slow_consumer_loses_events_rather_than_stalling_the_reader():
    """`InProcessEvents`' policy, one step further out.

    A consumer that stopped draining must not block the reader: that would
    stop reading the socket, which back-pressures the daemon's own stream —
    so a UI that froze would slow the jobs it is watching. Events are dropped
    instead, exactly as they are for an in-process subscriber.

    The end-of-stream marker is the one thing that is not dropped, and the
    drop rule is why: an event is droppable because the next one supersedes
    it, and nothing supersedes "there will be no next one". So it takes the
    place of the oldest tick still waiting.
    """
    client = _scripted_client(*[
        line for i in range(5) for line in (f'data: {{"job_id": "{i}"}}', "")
    ])
    try:
        queue = client.subscribe(max_queue=2)          # a consumer that never drains
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert queue.qsize() == 2                      # three events were dropped
        assert [queue.get_nowait() for _ in range(2)] == [{"job_id": "1"}, None]
    finally:
        await client.aclose()


async def test_unsubscribing_releases_the_stream_the_subscription_held():
    """A subscription is a live HTTP stream, so dropping the reference is not
    enough — `unsubscribe` cancels the reader, and `aclose` awaits the unwind
    (which is the difference between a sync port method and a closing one)."""
    client = _scripted_client('data: {"job_id": "a"}', "")
    queue = client.subscribe()
    reader = client._readers[queue]
    client.unsubscribe(queue)
    assert queue not in client._readers
    await client.aclose()
    assert reader.cancelled() or reader.done()


async def test_a_stream_that_ends_is_announced_rather_than_going_quiet(capsys):
    """A stream that ends looks exactly like a stream with nothing to say.

    A daemon shutting down closes `/events` without an error, and whoever is
    awaiting the queue would wait on a connection that no longer exists. The
    note goes to stderr, where every diagnostic in this layer goes: the queue
    carries events, so saying it there would mean inventing one.
    """
    client = _scripted_client('data: {"job_id": "a"}', "")
    try:
        queue = client.subscribe()
        await asyncio.wait_for(asyncio.shield(client._readers[queue]), timeout=5)
        assert queue.get_nowait() == {"job_id": "a"}
        assert "closed by the daemon" in capsys.readouterr().err
        # And on the queue as well, because stderr is the wrong place to say
        # it to a front-end: `jobsmith ui` owns the terminal, so the note
        # above goes nowhere it can show. `None` is the port's marker for
        # "nothing more will arrive here", and it is the last thing carried.
        assert queue.get_nowait() is None
        assert queue.empty()
    finally:
        await client.aclose()


# ------------------------------------- a backing that is not there at all


def _unreachable_client() -> DaemonClient:
    """A DaemonClient whose every request fails to reach anything."""
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return DaemonClient("http://test", httpx.AsyncClient(
        transport=httpx.MockTransport(refused), base_url="http://test", timeout=None))


def _port_calls(client: AgentService) -> dict:
    """Every use case a front-end can drive, as callables. One per method, so
    a method added to the port without a translation is a missing key here."""
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
    """Every call, one exception, and it is the port's own.

    Without this the remote backing answers with `httpx.ConnectError` — a fact
    about the transport, not about the use case — so a front-end would have to
    know which backing it holds to catch it, or wrap every call in a broad
    `except` that swallows its own bugs along with the daemon's absence. It is
    asked of *every* method rather than of one, because a translation that
    covers nine calls and misses the tenth is exactly the crash the tenth
    caller gets.
    """
    client = _unreachable_client()
    try:
        for name, call in _port_calls(client).items():
            with pytest.raises(ServiceUnavailable) as gone:
                await call()
            assert "http://test" in str(gone.value), f"{name} does not say what it cannot reach"
    finally:
        await client.aclose()


async def test_a_daemon_that_answers_badly_is_not_called_unreachable():
    """The narrowing must stay narrow: only *not reaching* the daemon is
    translated. A 500 means the daemon is there and something in it broke —
    a defect somebody should see as one, not a reconnect message. Anything
    else caught here would be the broad `except` this exists to avoid."""
    client = DaemonClient("http://test", httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
        base_url="http://test", timeout=None))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.list_jobs()
        # ...and on the streamed path too, which is where a tuple one class
        # too wide would actually swallow it: `raise_for_status` is called
        # inside the `async with`, so the translation sees it.
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("s", "hello")
    finally:
        await client.aclose()


async def test_the_local_backing_never_dresses_a_bug_as_an_absent_backing():
    """The other half of the same rule, on the other backing.

    A process cannot lose contact with itself, so `LocalAgentService` raises
    `ServiceUnavailable` never — and a `KeyError` from our own code stays a
    `KeyError`. Translating it would put a reconnect message where a stack
    trace belongs, which is the failure mode the issue that asked for this
    named first.
    """
    class Broken:
        async def list_jobs(self, **kwargs):
            raise KeyError("a defect in this process")

    service = LocalAgentService(Broken(), None)
    with pytest.raises(KeyError):
        await service.list_jobs()
