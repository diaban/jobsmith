"""Persistence backends: state survives a process restart; orphans are settled.

SQLite stands in for the real backends here (Postgres shares the same
checkpointer/store contract but needs a server — it is exercised manually).
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent_oo.app import build_app
from agent_oo.app.persistence import MEMORY, pick_db
from agent_oo.app.providers import KeywordChatModel, KeywordLLM
from agent_oo.jobs.models import JobStatus


async def open_app(tmp_path, db: str):
    return await build_app(
        llm=KeywordLLM(),
        chat_model=KeywordChatModel(),
        db=db,
        reports_dir=str(tmp_path / "artifacts"),
    )


def test_pick_db_precedence(monkeypatch):
    monkeypatch.delenv("AGENT_OO_DB", raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    assert pick_db() == MEMORY
    monkeypatch.setenv("AGENT_OO_DB", "from-env.db")
    assert pick_db() == "from-env.db"
    monkeypatch.setattr("sys.argv", ["prog", "--db=from-flag.db"])
    assert pick_db() == "from-flag.db"          # flag beats env
    assert pick_db("explicit.db") == "explicit.db"  # argument beats both


async def test_job_and_conversation_survive_restart(tmp_path):
    db = str(tmp_path / "agent.db")

    app = await open_app(tmp_path, db)
    session = app.new_session()
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}
    await agent.ainvoke({"messages": [HumanMessage("hello there")]}, cfg)

    job = await app.manager.create_job("research topic X", session_id=session.session_id)
    done = await app.manager.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    await app.aclose()  # process "exits"

    # --- new process, same database file ---
    app2 = await open_app(tmp_path, db)
    fetched = await app2.manager.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status is JobStatus.DONE
    assert fetched.session_id == session.session_id
    assert set(fetched.results) == {"research", "analysis", "critique"}   # artifacts kept
    assert fetched.plan is not None                                      # meta kept
    assert [j.job_id for j in await app2.manager.list_jobs()] == [job.job_id]

    # the conversation thread is still in the checkpointer
    session2 = app2.new_session(session.session_id)
    state = await session2.build().aget_state(cfg)
    assert any(isinstance(m, HumanMessage) for m in state.values["messages"])
    await app2.aclose()


async def test_interrupted_job_settled_on_startup(tmp_path):
    db = str(tmp_path / "agent.db")

    app = await open_app(tmp_path, db)
    job = await app.manager.create_job("long thing")
    job.status = JobStatus.RUNNING              # simulate a process killed mid-run
    await app.manager._persist_summary(job)
    await app.aclose()

    app2 = await open_app(tmp_path, db)         # build_app recovers on startup
    recovered = await app2.manager.get_job(job.job_id)
    assert recovered.status is JobStatus.FAILED
    assert "interrupted" in recovered.error
    await app2.aclose()


async def test_queued_jobs_are_left_runnable(tmp_path):
    db = str(tmp_path / "agent.db")
    app = await open_app(tmp_path, db)
    job = await app.manager.create_job("not started yet")
    await app.aclose()

    app2 = await open_app(tmp_path, db)
    assert (await app2.manager.get_job(job.job_id)).status is JobStatus.QUEUED
    done = await app2.manager.run_job(job.job_id)  # still runnable after restart
    assert done.status is JobStatus.DONE
    await app2.aclose()


async def test_memory_backend_isolated_per_app(tmp_path):
    app = await open_app(tmp_path, MEMORY)
    await app.manager.create_job("ephemeral")
    await app.aclose()

    app2 = await open_app(tmp_path, MEMORY)
    assert await app2.manager.list_jobs() == []  # nothing survives, by design
    await app2.aclose()
