"""CLI clients: the daemon and embedded backings must be interchangeable.

The daemon client is exercised against the real FastAPI app through httpx's
ASGI transport — no socket, but the same HTTP contract the daemon serves.
"""
from __future__ import annotations

from conftest import ScriptedChatModel
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_jobs import make_manager

from agent_oo.api import create_api
from agent_oo.app.providers import KeywordChatModel, KeywordLLM
from agent_oo.chat import ChatSession
from agent_oo.cli.client import DaemonClient, EmbeddedClient, open_client
from agent_oo.cli.main import build_parser

CLIENT_OPS = ("new_session", "send", "approve", "list_jobs", "get_job",
              "cancel_job", "launch_job", "get_report", "resolve_job")


def daemon_client_over(app) -> DaemonClient:
    http = AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=None)
    return DaemonClient("http://test", http)


async def embedded(tmp_path) -> EmbeddedClient:
    return await EmbeddedClient.create(
        llm=KeywordLLM(), chat_model=KeywordChatModel(),
        db="memory", reports_dir=str(tmp_path / "artifacts"),
    )


async def wait_done(client, job_id):
    import asyncio
    for _ in range(300):
        job = await client.get_job(job_id)
        if job and job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("job never finished")


async def test_daemon_client_full_chat_flow(store, checkpointer, tmp_path):
    manager = make_manager(store, checkpointer, tmp_path)
    saver = MemorySaver()
    responses = [launch_call("analyse it", "multi-step"), AIMessage(content="launched!")]

    def session_factory(session_id=None):
        return ChatSession(manager, ScriptedChatModel(responses=list(responses)),
                           session_id=session_id, checkpointer=saver)

    client = daemon_client_over(create_api(manager, session_factory))
    try:
        assert client.persistent is True          # jobs outlive the command
        sid = await client.new_session()
        reply = await client.send(sid, "please analyse it")
        assert reply["type"] == "proposal" and reply["query"] == "analyse it"
        assert (await client.approve(sid, True))["type"] == "message"

        (job,) = await client.list_jobs(session_id=sid)
        finished = await wait_done(client, job["job_id"])
        assert finished["status"] == "done"
        assert (await client.get_report(job["job_id"])).startswith("# analyse it")
        assert [o["role"] for o in finished["outputs"]] == ["main"]
        assert await client.get_job("nope") is None
    finally:
        await client.aclose()


async def test_embedded_client_same_shapes(tmp_path):
    client = await embedded(tmp_path)
    try:
        assert client.persistent is False         # jobs die with the process
        launched = await client.launch_job("research something")
        job = await wait_done(client, launched["job_id"])
        assert job["status"] == "done"
        assert set(job["results"]) == {"research", "analysis", "critique"}
        assert (await client.get_report(job["job_id"])).startswith("# research something")
        assert [o["role"] for o in job["outputs"]] == ["main"]

        # a summary carries the keys the CLI prints
        (summary,) = await client.list_jobs()
        assert {"job_id", "status", "query", "step_finished_at"} <= set(summary)
    finally:
        await client.aclose()


async def test_both_clients_expose_the_same_operations(tmp_path):
    client = await embedded(tmp_path)
    try:
        for op in CLIENT_OPS:
            assert callable(getattr(client, op)), op
            assert callable(getattr(DaemonClient, op)), op
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
    assert parser.parse_args([]).command is None      # bare call -> main() maps to chat


async def test_embedded_run_actually_runs_the_job(tmp_path, capsys):
    """Without a daemon the job runs in this process: `run` must not return
    before it finishes, or the job would die with the command."""
    from types import SimpleNamespace

    from agent_oo.cli.main import cmd_run

    client = await embedded(tmp_path)
    try:
        rc = await cmd_run(client, SimpleNamespace(task="do the thing", wait=False))
        assert rc == 0
        (job,) = await client.list_jobs()
        assert job["status"] == "done"                     # it really ran
        assert "no daemon" in capsys.readouterr().err      # and said why it waited
    finally:
        await client.aclose()
