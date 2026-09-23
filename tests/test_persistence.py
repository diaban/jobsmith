"""Persistence backends: state survives a process restart; orphans are settled.

SQLite stands in for the real backends here (Postgres shares the same
checkpointer/store contract but needs a server — it is exercised manually).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import registered_capabilities
from langchain_core.messages import HumanMessage

from jobsmith.app import build_app
from jobsmith.app.persistence import (
    MEMORY,
    data_dir,
    default_db_path,
    default_reports_dir,
    pick_db,
    pick_reports_dir,
)
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.core.state import SOURCE_FILES_INPUT_KEY
from jobsmith.jobs.models import JobStatus


async def open_app(tmp_path, db: str):
    return await build_app(
        llm=KeywordLLM(),
        chat_model=KeywordChatModel(),
        db=db,
        reports_dir=str(tmp_path / "artifacts"),
    )


def test_pick_db_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("JOBSMITH_DB", raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    # an unconfigured jobsmith keeps its jobs (#63): a FILE, in the data dir
    assert pick_db() == str(tmp_path / "jobsmith" / "jobs.db") == str(default_db_path())
    assert pick_db(MEMORY) == MEMORY                 # still there, when asked for
    monkeypatch.setenv("JOBSMITH_DB", "from-env.db")
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
    # every step the fake chained is still there (the pack's registry depends
    # on what is installed, so it is read back from the app)
    assert set(fetched.results) == set(registered_capabilities(app2))      # results kept
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


# ------------------------------------------ the default keeps its jobs (#63)


@pytest.fixture
def unconfigured(monkeypatch, tmp_path):
    """A machine where nobody configured anything, with its data dir HERE.

    The working directory is moved elsewhere on purpose: the default must not
    depend on where the command was typed.
    """
    home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(home))
    for name in ("JOBSMITH_DB", "JOBSMITH_REPORTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.argv", ["jobsmith"])
    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return home / "jobsmith"


async def test_an_unconfigured_app_keeps_its_jobs_across_a_restart(unconfigured, tmp_path, capsys):
    """The issue, end to end: run, "exit", come back from another directory."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel())
    job = await app.manager.create_job("study the topic in depth and write a report")
    done = await app.manager.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    await app.aclose()

    assert (unconfigured / "jobs.db").is_file(), "the data dir was created on first use"
    banner = capsys.readouterr().err
    assert f"jobs kept in {unconfigured / 'jobs.db'}" in banner
    assert "--db=memory" in banner, "the default says how to get the old behaviour"

    other = tmp_path / "elsewhere"
    other.mkdir()
    os.chdir(other)                              # monkeypatch restores the cwd
    app2 = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel())
    try:
        fetched = await app2.manager.get_job(job.job_id)
        assert fetched is not None and fetched.status is JobStatus.DONE
        # ...and the file it recorded is where it says, from here too
        assert fetched.report_path and Path(fetched.report_path).is_absolute()
        assert Path(fetched.report_path).is_file()
        assert Path(fetched.report_path).is_relative_to(unconfigured / "reports")
    finally:
        await app2.aclose()
    assert not (other / "artifacts").exists() and not (tmp_path / "somewhere" / "artifacts").exists()


async def test_the_default_reports_dir_is_what_read_files_may_read(unconfigured):
    """#60's loop still closes when the reports dir is the data dir's."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(), db=MEMORY)
    try:
        first = await app.manager.run_job((await app.manager.create_job(
            "study the topic in depth and write a report")).job_id)
        assert first.report_path and first.report_path.startswith(str(default_reports_dir()))
        second = await app.manager.run_job((await app.manager.create_job(
            "make a one-pager out of the report",
            {SOURCE_FILES_INPUT_KEY: [first.report_path]})).job_id)
        result = second.results["read_files"]
        assert result["ok"] is True, result.get("error")
    finally:
        await app.aclose()


def test_a_relative_reports_dir_is_recorded_as_where_it_pointed(monkeypatch, tmp_path):
    """A record outlives the process, so it may not carry a cwd-relative path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JOBSMITH_REPORTS_DIR", raising=False)
    assert pick_reports_dir("artifacts") == tmp_path.resolve() / "artifacts"
    monkeypatch.setenv("JOBSMITH_REPORTS_DIR", "out")
    assert pick_reports_dir() == tmp_path.resolve() / "out"          # env beats default
    assert pick_reports_dir("explicit") == tmp_path.resolve() / "explicit"  # arg beats env
    monkeypatch.delenv("JOBSMITH_REPORTS_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert pick_reports_dir() == (tmp_path / "xdg" / "jobsmith" / "reports").resolve()


@pytest.mark.parametrize(("platform", "env", "expected"), [
    ("linux", {}, ".local/share/jobsmith"),
    ("darwin", {}, "Library/Application Support/jobsmith"),
    ("win32", {}, "AppData/Local/jobsmith"),
])
def test_the_data_dir_follows_the_platform(monkeypatch, tmp_path, platform, env, expected):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert data_dir() == tmp_path / expected


def test_xdg_data_home_wins_everywhere_and_a_relative_one_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert data_dir() == tmp_path / "xdg" / "jobsmith"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/dir")   # invalid per the XDG spec
    assert data_dir() == tmp_path / "Library" / "Application Support" / "jobsmith"
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert data_dir() == tmp_path / "Local" / "jobsmith"
