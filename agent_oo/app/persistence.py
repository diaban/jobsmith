"""Persistence backends: where checkpoints and job records actually live.

One spec string selects the backend (CLI `--db=`, env `AGENT_OO_DB`, or the
`db=` argument of `build_app`):

    memory                     in-process only (default) — nothing survives exit
    jobs.db / sqlite:jobs.db   a local SQLite file          [.[sqlite]]
    postgresql://user@host/db  Postgres                     [.[postgres]]

Both real backends give the two pieces the product needs: a *checkpointer*
(fine-grained graph state, keyed by thread_id == job_id) and a *store* (the
job index, plans, artifacts). For Postgres they share ONE AsyncConnectionPool,
so the whole app holds a single, properly sized pool.

Everything here is async and registers its teardown on the caller's
AsyncExitStack: resources are created inside the event loop that will use
them, which is why `build_app` is a coroutine (see app/main.py — uvicorn is
served from that same loop rather than through `uvicorn.run`).
"""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from typing import Any

MEMORY = "memory"
_POSTGRES_SCHEMES = ("postgresql://", "postgres://", "postgresql+psycopg://")


def pick_db(explicit: str | None = None) -> str:
    """Resolve the backend spec: argument > --db= flag > AGENT_OO_DB > memory."""
    if explicit:
        return explicit
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--db=")), None)
    return flag or os.environ.get("AGENT_OO_DB") or MEMORY


async def open_persistence(spec: str, stack: AsyncExitStack) -> tuple[Any, Any]:
    """Open (checkpointer, store) for `spec`, teardown registered on `stack`."""
    if spec in (MEMORY, "", ":memory:"):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        print("[persistence: in-memory — nothing survives this process]")
        return MemorySaver(), InMemoryStore()

    if spec.startswith(_POSTGRES_SCHEMES):
        return await _open_postgres(spec, stack)

    path = spec.split(":", 1)[1] if spec.startswith("sqlite:") else spec
    return await _open_sqlite(path, stack)


async def _open_sqlite(path: str, stack: AsyncExitStack) -> tuple[Any, Any]:
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.store.sqlite.aio import AsyncSqliteStore
    except ImportError:  # pragma: no cover - depends on install extras
        sys.exit('SQLite persistence needs:  uv pip install -e ".[sqlite]"')

    # WAL is a persistent property of the file, and it is what lets a running
    # job write while the chat reads. Set it on a throwaway connection: a
    # pragma left un-consumed on a live connection holds a lock and the next
    # connection's DDL then blocks on "database is locked".
    async with aiosqlite.connect(path) as setup_conn:
        async with setup_conn.execute("PRAGMA journal_mode=WAL") as cur:
            await cur.fetchone()
        await setup_conn.commit()

    # One connection each, with the transaction mode each backend expects (as
    # in their own from_conn_string): the store drives BEGIN/COMMIT itself, so
    # it needs autocommit — under sqlite3's implicit transactions its first
    # write leaves one open and the next BEGIN raises. The saver commits
    # itself and keeps the default. `timeout` is sqlite's busy timeout: wait
    # for the other connection's write lock instead of failing. Nothing else
    # is executed here — a stray pragma would open a transaction too.
    async def connect(**kwargs: Any) -> Any:
        conn = await aiosqlite.connect(path, timeout=5.0, **kwargs)
        stack.push_async_callback(conn.close)
        return conn

    checkpointer = AsyncSqliteSaver(await connect())  # sets its schema up lazily
    store = AsyncSqliteStore(await connect(isolation_level=None))
    await store.setup()
    print(f"[persistence: sqlite — {path}]")
    return checkpointer, store


async def _open_postgres(dsn: str, stack: AsyncExitStack, *, max_size: int = 10) -> tuple[Any, Any]:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError:  # pragma: no cover - depends on install extras
        sys.exit('Postgres persistence needs:  uv pip install -e ".[postgres]"')

    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=max_size,
        open=False,
        # Required by both langgraph Postgres backends.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    stack.push_async_callback(pool.close)

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()          # creates tables / runs migrations, idempotent
    store = AsyncPostgresStore(pool)
    await store.setup()
    print(f"[persistence: postgres — pool of {max_size} on {_safe_dsn(dsn)}]")
    return checkpointer, store


def _safe_dsn(dsn: str) -> str:
    """DSN without credentials — safe to print."""
    if "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[1]}"
