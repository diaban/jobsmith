"""A long history hides nothing: the newest job stays announced, findable, listed.

Every query here once went through a `limit` applied before any filter, to a
store that returns the oldest rows first — so past a hundred jobs, the newest
one vanished from announcements, cancellation, listings and events (→ 0141).
Seeded well past every old threshold, on each backend.
"""
from __future__ import annotations

import os
import uuid

import pytest
from support import until

from jobsmith.app.agent import build_app
from jobsmith.app.providers import KeywordChatModel, make_llm
from jobsmith.chat.tools import _find
from jobsmith.engine.models import JobStatus

HISTORY = 250           # past every former cut: 50, 100 and 200
PG = os.environ.get("JOBSMITH_TEST_PG")


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "postgres":
        if not PG:
            pytest.skip("set $JOBSMITH_TEST_PG to a Postgres DSN")
        return PG
    return "memory" if request.param == "memory" else str(tmp_path / "jobs.db")


async def _app(tmp_path, db):
    return await build_app(llm=make_llm("fake"), chat_model=KeywordChatModel(),
                           db=db, reports_dir=str(tmp_path / "reports"))


async def test_the_newest_job_survives_a_long_history(tmp_path, db):
    app = await _app(tmp_path, db)
    # On a shared database the subscriber is ANOTHER process's view — the one
    # a cut once blinded; on memory there is no other, so it is this one.
    watcher = app if db == "memory" else await _app(tmp_path, db)
    manager = app.manager
    me, others = f"me-{uuid.uuid4().hex[:8]}", f"others-{uuid.uuid4().hex[:8]}"   # a reused DB
    try:
        for i in range(HISTORY):
            await manager.create_job(f"an old job {i}", session_id=others)
        queue = watcher.manager.subscribe()
        mine = await manager.create_job("compare A and B", session_id=me)
        await manager.run_job(mine.job_id)

        assert [j.job_id for j in await manager.list_jobs(session_id=me)] == [mine.job_id]
        assert (await manager.list_jobs(limit=1))[0].job_id == mine.job_id   # newest first
        assert [j.job_id for j in await manager.list_finished_unannounced(me)] == [mine.job_id]
        assert (await _find(manager, me, mine.job_id[:8])).job_id == mine.job_id
        queued = await manager.list_jobs(status=JobStatus.QUEUED, session_id=others, limit=None)
        assert len(queued) == HISTORY                                         # complete

        async def heard_it():
            while not queue.empty():
                if (event := queue.get_nowait())["job_id"] == mine.job_id:
                    return event
            return None

        await until(heard_it, what="the newest job reaching a subscriber")
    finally:
        if watcher is not app:
            await watcher.aclose()
        await app.aclose()
