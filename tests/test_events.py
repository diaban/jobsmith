"""Job events reach a subscriber whichever process ran the job, on a SQLite file.

`subscribe()` used to fan out only what this process persisted: a `jobsmith ui`
did not repaint for a job a `jobsmith chat` ran on the same database until F5.
On a SQLite file it now also hears other connections' commits (→ 0100), on
Postgres their NOTIFY (→ 0138: set `$JOBSMITH_TEST_PG` to a DSN to run those
cases, e.g. a throwaway `docker run postgres`); on memory it polls nothing.
"""
from __future__ import annotations

import os

import pytest
from support import until

from jobsmith.app.agent import build_app
from jobsmith.app.providers import KeywordChatModel, make_llm
from jobsmith.engine.events import InProcessEvents, PostgresNotifyEvents, SqliteWatchEvents

PG = os.environ.get("JOBSMITH_TEST_PG")


@pytest.fixture(params=["sqlite", "postgres"])
def shared_db(request, tmp_path):
    """A database two apps share: a SQLite file, or the Postgres named by $JOBSMITH_TEST_PG."""
    if request.param == "postgres":
        if not PG:
            pytest.skip("set $JOBSMITH_TEST_PG to a Postgres DSN")
        return PG
    return str(tmp_path / "jobs.db")


async def _app(tmp_path, db):
    return await build_app(llm=make_llm("fake"), chat_model=KeywordChatModel(),
                           db=db, reports_dir=str(tmp_path / "reports"))


async def test_a_job_another_process_runs_reaches_this_subscriber(tmp_path, shared_db):
    db = shared_db
    watcher, runner = await _app(tmp_path, db), await _app(tmp_path, db)
    try:
        queue = watcher.manager.subscribe()
        job = await runner.manager.create_job("compare A and B")
        await runner.manager.run_job(job.job_id)

        async def heard_it_done():
            while not queue.empty():
                event = queue.get_nowait()
                if event["job_id"] == job.job_id and event["status"] == "done":
                    return event
            return None

        event = await until(heard_it_done, what="the other process's job reaching the subscriber")
        assert event["query"] == "compare A and B"
        watcher.manager.unsubscribe(queue)
        assert watcher.manager.events._task is None      # nobody listening: nothing polls
    finally:
        await runner.aclose()
        await watcher.aclose()


async def _drain(queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def test_what_existed_before_subscribing_is_not_news_and_one_leaver_stops_nothing(
        tmp_path, shared_db):
    db = shared_db
    watcher, runner = await _app(tmp_path, db), await _app(tmp_path, db)
    try:
        old = await runner.manager.create_job("an earlier job")
        await runner.manager.run_job(old.job_id)
        staying, leaving = watcher.manager.subscribe(), watcher.manager.subscribe()
        watcher.manager.unsubscribe(leaving)        # the other subscriber still listens
        new = await runner.manager.create_job("compare A and B")
        await runner.manager.run_job(new.job_id)
        seen: list[dict] = []

        async def heard_the_new_one():
            seen.extend(await _drain(staying))
            return any(e["job_id"] == new.job_id and e["status"] == "done" for e in seen)

        await until(heard_the_new_one, what="the remaining subscriber hearing the new job")
        assert all(e["job_id"] != old.job_id for e in seen)
    finally:
        await runner.aclose()
        await watcher.aclose()


async def test_nothing_is_watched_before_someone_subscribes(tmp_path, shared_db):
    app = await _app(tmp_path, shared_db)
    try:
        kind = PostgresNotifyEvents if shared_db == PG else SqliteWatchEvents
        assert isinstance(app.manager.events, kind)
        assert app.manager.events._task is None
    finally:
        await app.aclose()


async def test_memory_keeps_the_in_process_fan_out_and_polls_nothing(tmp_path):
    app = await _app(tmp_path, "memory")
    try:
        assert type(app.manager.events) is InProcessEvents
    finally:
        await app.aclose()
