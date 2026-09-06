"""The inbound port: one interface, two backings, no second implementation.

The property worth protecting is that a front-end cannot tell where the work
happens. So the same sequence is driven through the local service and through
HTTP, and the answers must match — not merely "both work".
"""
from __future__ import annotations

import inspect

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
from jobsmith.service import AgentService, BinaryDeliverable, LocalAgentService


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
                         "rationale": "multi-step"}

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
