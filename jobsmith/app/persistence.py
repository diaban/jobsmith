"""Persistence backends: where checkpoints and job records actually live.

One spec string selects the backend (CLI `--db=`, env `JOBSMITH_DB`, or the
`db=` argument of `build_app`):

    (nothing)                  <data dir>/jobs.db — SQLite, the default (#63)
    jobs.db / sqlite:jobs.db   a SQLite file you name
    memory                     in-process only — nothing survives exit
    postgresql://user@host/db  Postgres                     [.[postgres]]

**An unconfigured jobsmith keeps its jobs** (#63). The default used to be
`memory`, so `jobsmith jobs` from a new process listed nothing and every
surface built to find a job again — `job <id>`, `resume`, the TUI's job list,
the chat's finished-job announcement — was dead on arrival and looked broken
rather than unconfigured. SQLite is therefore a CORE dependency, not an
extra: "a file when the extra happens to be installed, memory otherwise"
would make one command behave differently on two machines, the silent
difference this project refuses everywhere else (a capability nothing can
serve stays out of the registry; a format nothing renders fails at startup).
`memory` stays a supported value, and it is what the tests and `evals/` ask
for by name.

**Where**: a per-user data directory (`data_dir`), never the working
directory — a job list that depends on where you happened to type the command
is the same surprise one level up. The deliverables live under it too
(`default_reports_dir`), for the reason the job list does: a persistent record
pointing at `artifacts/…` relative to wherever it was launched is a path to
nothing from anywhere else.

Both real backends give the two pieces the product needs: a *checkpointer*
(fine-grained graph state, keyed by thread_id == job_id) and a *store* (the
job index, plans, artifacts). For Postgres they share ONE AsyncConnectionPool,
so the whole app holds a single, properly sized pool.

Everything here is async and registers its teardown on the caller's
AsyncExitStack: resources are created inside the event loop that will use
them, which is why `build_app` is a coroutine (see cli/main.py — uvicorn is
served from that same loop rather than through `uvicorn.run`).
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, cast

MEMORY = "memory"
APP_DIR = "jobsmith"
DB_FILENAME = "jobs.db"
REPORTS_DIRNAME = "reports"
_POSTGRES_SCHEMES = ("postgresql://", "postgres://", "postgresql+psycopg://")


def data_dir() -> Path:
    """This user's jobsmith directory: where jobs and their files are kept.

    `$XDG_DATA_HOME/jobsmith` when that variable is set to an absolute path —
    on every platform, because setting it is an explicit choice and it is the
    one lever that relocates everything at once (the test suite uses it to
    prove it never touches the real directory). Otherwise the platform's own
    convention, stdlib only — three branches are not worth a dependency:

        Linux & co   ~/.local/share/jobsmith            (the XDG default)
        macOS        ~/Library/Application Support/jobsmith
        Windows      %LOCALAPPDATA%\\jobsmith  (~\\AppData\\Local if unset)

    A relative `XDG_DATA_HOME` is ignored, as the XDG spec says it must be.
    Nothing is created here; whoever writes creates what it needs.
    """
    xdg = os.environ.get("XDG_DATA_HOME", "")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / APP_DIR
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_DIR
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        return (Path(local) if local else home / "AppData" / "Local") / APP_DIR
    return home / ".local" / "share" / APP_DIR


def default_db_path() -> Path:
    """The SQLite file an unconfigured jobsmith keeps its jobs in."""
    return data_dir() / DB_FILENAME


def default_reports_dir() -> Path:
    """Where deliverables and annexes go when nobody said: next to the jobs."""
    return data_dir() / REPORTS_DIRNAME


def pick_db(explicit: str | None = None) -> str:
    """Resolve the backend spec: argument > --db= flag > JOBSMITH_DB > the data dir.

    The fallback is a FILE (#63): an unconfigured product keeps its jobs.
    `memory` is still a value anyone may ask for, and must ask for.
    """
    if explicit:
        return explicit
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--db=")), None)
    return flag or os.environ.get("JOBSMITH_DB") or str(default_db_path())


def pick_reports_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Where deliverables are written: argument > JOBSMITH_REPORTS_DIR > data dir.

    Always ABSOLUTE, resolved once at startup: every path a job records is
    built from this, and a record that outlives the process (#63) must not
    carry a path that only means something from the directory it was
    launched in. A relative value is therefore read against the working
    directory NOW, and stored as what it pointed at.
    """
    raw = explicit or os.environ.get("JOBSMITH_REPORTS_DIR") or default_reports_dir()
    return Path(raw).expanduser().resolve()


async def open_persistence(spec: str, stack: AsyncExitStack) -> tuple[Any, Any]:
    """Open (checkpointer, store) for `spec`, teardown registered on `stack`."""
    if spec in (MEMORY, "", ":memory:"):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        print("[persistence: in-memory — nothing survives this process]", file=sys.stderr)
        return MemorySaver(), InMemoryStore()

    if spec.startswith(_POSTGRES_SCHEMES):
        return await _open_postgres(spec, stack)

    path = spec.split(":", 1)[1] if spec.startswith("sqlite:") else spec
    return await _open_sqlite(path, stack)


async def _open_sqlite(spec_path: str, stack: AsyncExitStack) -> tuple[Any, Any]:
    # A core dependency since #63, so no ImportError guard: its absence is a
    # broken install, and a traceback is the honest answer to one.
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    # Absolute for the banner (a relative name says nothing about where it
    # is), and the directory is created on first use — the default lives in
    # a data dir a fresh machine does not have yet.
    file = Path(spec_path).expanduser().resolve()
    file.parent.mkdir(parents=True, exist_ok=True)
    path = str(file)

    # WAL is a persistent property of the file, and it is what lets a running
    # job write while the chat reads. Set it on a throwaway connection: a
    # pragma left un-consumed on a live connection holds a lock and the next
    # connection's DDL then blocks on "database is locked".
    # Switching a FRESH file to WAL needs the file to itself for an instant,
    # and SQLite answers "database is locked" without waiting when another
    # process is doing the same — measured, rarely, with several processes
    # opening a new file at once. Already-WAL files never ask.
    async def set_wal() -> None:
        async with aiosqlite.connect(path) as setup_conn:
            async with setup_conn.execute("PRAGMA journal_mode=WAL") as cur:
                await cur.fetchone()
            await setup_conn.commit()

    await _while_another_process_sets_up(set_wal)

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

    # The saver needs nothing more, MEASURED (tests/test_sqlite_concurrency.py):
    # its connection keeps sqlite3's implicit transactions, which open on the
    # first INSERT — a transaction that starts by writing asks for the write
    # lock before it has read anything, so the busy timeout applies.
    checkpointer = AsyncSqliteSaver(await connect())  # sets its schema up lazily
    # The store does not start by writing — see `_ImmediateBegin`.
    store = AsyncSqliteStore(cast(Any, _ImmediateBegin(await connect(isolation_level=None))))
    await _while_another_process_sets_up(store.setup)
    # Say where the jobs are kept, and — for the default nobody chose — how
    # to get the old behaviour back, since this is the line a person who
    # never configured anything reads.
    note = "  (default; --db=memory keeps nothing)" if file == default_db_path().resolve() else ""
    print(f"[persistence: sqlite — jobs kept in {path}{note}]", file=sys.stderr)
    return checkpointer, store


class _ImmediateBegin:
    """The store's connection, with its `BEGIN` taking the write lock up front.

    Every `AsyncSqliteStore` batch runs inside `BEGIN … COMMIT` it issues
    itself (the connection is in autocommit so that it can), and that `BEGIN`
    is DEFERRED: the batch reads, takes a snapshot, then writes. If another
    connection committed in between, SQLite refuses to upgrade that stale
    read to a write and says "database is locked" AT ONCE — in WAL mode that
    is SQLITE_BUSY_SNAPSHOT, which never consults the busy timeout, because
    waiting cannot make an old snapshot current. With two processes on one
    file that is not an edge case (#10 measured it first); since #63 made a
    per-user file the default, every `jobsmith chat` and `jobsmith run` on a
    machine is such a process.

    `BEGIN IMMEDIATE` asks for the write lock before the first read, so a
    contended batch WAITS (the busy timeout, `connect(timeout=...)`) instead
    of failing, and the snapshot it then reads is current by construction.

    Why a wrapper over the connection and not a patch of the store: the
    store takes its connection as a constructor argument, which is the one
    seam it offers, and "which BEGIN" is a property of the connection it is
    handed. Nothing of LangGraph is subclassed, overridden or replaced — a
    private `_cursor` override would silently diverge from the next release,
    while this depends only on the store saying `BEGIN`, which the
    two-process test in tests/test_sqlite_concurrency.py pins: if the store
    ever starts a transaction some other way, that test fails, not a user.

    Its cost, stated: the store's READ batches go through the same `BEGIN`,
    so they queue behind another process's write (milliseconds; every store
    write here is a small upsert) where WAL would have let them through.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, parameters: Any = None) -> Any:
        # Not async on purpose: aiosqlite's `execute` returns an object that
        # is both awaitable and an async context manager, and the store uses
        # it both ways — so it is handed back untouched, not awaited here.
        if sql.strip().upper() == "BEGIN":
            sql = "BEGIN IMMEDIATE"
        return self._conn.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


SETUP_RETRIES = 8
SETUP_PAUSE = 0.02


async def _while_another_process_sets_up(step: Callable[[], Awaitable[None]]) -> None:
    """Run one setup step, tolerating another process running it at the same time.

    Measured, and only reachable on a FRESH file — which is exactly what the
    default is on a machine's first two concurrent commands. Two steps race:

    - the WAL switch (see `_open_sqlite`), refused at once while another
      process holds the file;
    - the store's migrations: it reads its migration version, then applies
      what is missing one autocommitted statement at a time, with nothing
      held between the read and the writes, so two processes collide
      ("UNIQUE constraint failed: store_migrations.v", "duplicate column
      name"). Each such failure is the OTHER process having just done the
      same step — every statement is a single DDL or a single insert, so
      nothing is half-applied — and the next attempt re-reads the version
      and resumes after it.

    Serialising setup instead would need a lock `executescript` does not
    break (it commits any open transaction first), which SQLite does not
    offer on the store's own connection. Anything else, and a file still
    contended after ~5 s, is raised as it was.
    """
    pause = SETUP_PAUSE
    for attempt in range(SETUP_RETRIES + 1):
        try:
            await step()
            return
        except sqlite3.DatabaseError as e:
            if attempt == SETUP_RETRIES or not _is_concurrent_setup(e):
                raise
        await asyncio.sleep(pause)
        pause *= 2


def _is_concurrent_setup(error: Exception) -> bool:
    text = str(error)
    return any(sign in text for sign in (
        "UNIQUE constraint failed", "duplicate column", "already exists", "locked"))


async def _open_postgres(dsn: str, stack: AsyncExitStack, *, max_size: int = 10) -> tuple[Any, Any]:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg import AsyncConnection
        from psycopg.rows import DictRow, dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError:  # pragma: no cover - depends on install extras
        sys.exit('Postgres persistence needs:  uv pip install -e ".[postgres]"')

    # The annotation says out loud what `row_factory: dict_row` below means:
    # this pool hands out dict-row connections, which is exactly the `Conn`
    # both langgraph Postgres backends accept. Buried in a kwargs dict, that
    # is invisible to a reader and to a checker.
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
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
    print(f"[persistence: postgres — pool of {max_size} on {_safe_dsn(dsn)}]", file=sys.stderr)
    return checkpointer, store


def _safe_dsn(dsn: str) -> str:
    """DSN without credentials — safe to print."""
    if "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[1]}"
