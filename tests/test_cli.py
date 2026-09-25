"""CLI clients: the daemon and embedded backings must be interchangeable.

The daemon client is exercised against the real FastAPI app through httpx's
ASGI transport — no socket, but the same HTTP contract the daemon serves.
"""
from __future__ import annotations

from conftest import registered_capabilities
from support import (
    StubPdf,
    cancelled_midway,
    wait_done,
)

from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.cli.client import DaemonClient, EmbeddedClient, open_client
from jobsmith.cli.main import build_parser
from jobsmith.service import LocalAgentService


async def embedded(tmp_path) -> EmbeddedClient:
    return await EmbeddedClient.create(
        llm=KeywordLLM(), chat_model=KeywordChatModel(),
        db="memory", reports_dir=str(tmp_path / "artifacts"),
    )


async def test_embedded_client_same_shapes(tmp_path):
    assert DaemonClient.persistent is True        # jobs outlive the command
    client = await embedded(tmp_path)
    try:
        assert client.persistent is False         # jobs die with the process
        launched = await client.launch_job("research something", formats=["default"])
        job = await wait_done(client, launched["job_id"])
        assert job["status"] == "done"
        # the registry depends on what is installed (`slide_deck` needs
        # `.[pptx]`), so the fake's chain is read back from the app it composed
        assert set(job["results"]) == set(registered_capabilities(client))
        assert (await client.get_report(job["job_id"])).startswith("# research something")
        assert [o["role"] for o in job["outputs"] if o["role"] != "annex"] == ["main"]

        # a summary carries the keys the CLI prints
        (summary,) = await client.list_jobs()
        assert {"job_id", "status", "query", "step_finished_at"} <= set(summary)
    finally:
        await client.aclose()


async def test_prefix_resolution_and_ambiguity(tmp_path):
    client = await embedded(tmp_path)
    try:
        launched = await client.launch_job("some task")
        job = await client.resolve_job(launched["job_id"][:8])
        assert job["job_id"] == launched["job_id"]
        assert await client.resolve_job("zzzz") is None
    finally:
        await client.aclose()


async def test_open_client_falls_back_to_embedded(tmp_path, capsys):
    client = await open_client(
        url="http://127.0.0.1:9",            # nothing listens there
        llm=KeywordLLM(), chat_model=KeywordChatModel(),
        db="memory", reports_dir=str(tmp_path / "artifacts"),
    )
    try:
        assert isinstance(client, EmbeddedClient)
        # the trade-off is stated on stderr, so stdout stays pipeable
        assert "running embedded" in capsys.readouterr().err
    finally:
        await client.aclose()


def test_parser_shape():
    parser = build_parser()
    args = parser.parse_args(["--llm", "fake", "--db", "x.db", "jobs", "--status", "done"])
    assert (args.command, args.llm, args.db, args.status) == ("jobs", "fake", "x.db", "done")
    assert parser.parse_args(["serve", "--port", "9100"]).port == 9100
    assert parser.parse_args(["chat", "--session", "abc"]).session == "abc"
    assert parser.parse_args(["job", "1a2b"]).job_id == "1a2b"
    assert parser.parse_args(["resume", "1a2b"]).job_id == "1a2b"
    assert parser.parse_args([]).command is None      # bare call -> main() maps to chat


async def test_embedded_run_actually_runs_the_job(tmp_path, capsys):
    """Without a daemon the job runs in this process: `run` must not return
    before it finishes, or the job would die with the command."""
    from types import SimpleNamespace

    from jobsmith.cli.main import cmd_run

    client = await embedded(tmp_path)
    try:
        rc = await cmd_run(client, SimpleNamespace(task="do the thing", wait=False))
        assert rc == 0
        (job,) = await client.list_jobs()
        assert job["status"] == "done"                     # it really ran
        assert "no daemon" in capsys.readouterr().err      # and said why it waited
    finally:
        await client.aclose()


async def test_resume_command_restarts_a_stopped_job(store, checkpointer, tmp_path, capsys):
    """`jobsmith resume <prefix>` finishes a cancelled job, and refuses one
    that has nothing left to run with a non-zero exit code."""
    from types import SimpleNamespace

    from jobsmith.cli.main import cmd_resume

    manager, job, _alpha, slow = await cancelled_midway(store, checkpointer, tmp_path)
    client = LocalAgentService(manager, lambda session_id=None: None)
    slow.delay = 0.0

    assert await cmd_resume(client, SimpleNamespace(job_id=job.job_id[:8])) == 0
    assert (await client.get_job(job.job_id))["status"] == "done"
    # embedded: the command waited here instead of orphaning the resumed run
    assert "no daemon" in capsys.readouterr().err

    assert await cmd_resume(client, SimpleNamespace(job_id=job.job_id[:8])) == 1
    assert "cannot resume" in capsys.readouterr().out
    assert await cmd_resume(client, SimpleNamespace(job_id="zzzz")) == 1


async def test_report_on_a_binary_deliverable_says_where_the_file_is(tmp_path, capsys):
    """`jobsmith report` prints text and a PDF is not any, but the job did
    produce a deliverable — so the command names it instead of claiming there
    is none, which is what a bare `None` from the port would have printed."""
    from types import SimpleNamespace

    from jobsmith.cli.main import cmd_report

    client = await embedded(tmp_path)
    try:
        client.manager.reporter = StubPdf()
        launched = await client.launch_job("print it", formats=["default"])
        job_id = launched["job_id"]
        await wait_done(client, job_id)

        rc = await cmd_report(client, SimpleNamespace(job_id=job_id))
        printed = capsys.readouterr().out
        assert rc == 1
        assert "is pdf, which is not text" in printed
        assert f"/jobs/{job_id}/outputs/{job_id}.pdf" in printed
        assert "no report available" not in printed
    finally:
        await client.aclose()
