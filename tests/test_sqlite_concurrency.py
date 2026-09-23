"""Several OS processes on ONE SQLite file — the default since #63.

Measured, not argued: each test starts real processes (`tests/sqlite_hammer.py`)
that write to the same file at the same moment, straight through LangGraph's
backends as `open_persistence` configures them — never through
`StoreJobRepository`, so nothing between the backend and the file can hide it.

Before the fixes in `app/persistence.py`, measured with 4–6 processes:

- store  — most processes died at once with "database is locked": a deferred
  `BEGIN` reads a snapshot, another process commits, and the upgrade to a write
  is refused without the busy timeout ever being consulted. `_ImmediateBegin`.
- setup  — on a fresh file, processes opening it together raced the store's
  migrations ("UNIQUE constraint failed: store_migrations.v", "duplicate column
  name: ttl_minutes"). `_setup_store`.
- saver  — never failed, before or after: its implicit transaction opens on
  the first INSERT, so it asks for the write lock before reading anything and
  the busy timeout applies. Kept as a test so that stays measured.

Before a fix, these fail often but not always — a race is a race. After, they
cannot fail short of a writer holding the lock past the 5 s busy timeout.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from jobsmith.app.persistence import open_persistence

TESTS = Path(__file__).parent
PROCESSES = 5
ROUNDS = 150


async def hammer(db: str, mode: str, *, rounds: int = ROUNDS) -> list[str]:
    """Run PROCESSES hammers at once; answer with what each one printed."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(
        [str(TESTS), os.environ.get("PYTHONPATH", "")])}
    start_at = time.time() + 4.0            # past every interpreter's startup
    procs = [await asyncio.create_subprocess_exec(
        sys.executable, str(TESTS / "sqlite_hammer.py"),
        db, mode, str(rounds), f"p{i}", str(start_at),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env)
        for i in range(PROCESSES)]
    outs = await asyncio.wait_for(
        asyncio.gather(*(p.communicate() for p in procs)), timeout=120)
    return [out.decode().strip() for out, _ in outs]


async def initialised(tmp_path) -> str:
    """A file whose schema exists, so a write test measures writes only."""
    db = str(tmp_path / "shared.db")
    async with AsyncExitStack() as stack:
        checkpointer, _ = await open_persistence(db, stack)
        await checkpointer.setup()
    return db


@pytest.mark.parametrize("mode", ["store", "saver"])
async def test_concurrent_processes_write_one_file_without_locking_out(tmp_path, mode):
    results = await hammer(await initialised(tmp_path), mode)
    assert results == [f"ok {ROUNDS}"] * PROCESSES, results


async def test_processes_opening_a_fresh_file_together_all_get_it(tmp_path):
    results = await hammer(str(tmp_path / "fresh.db"), "setup", rounds=0)
    assert results == ["ok 0"] * PROCESSES, results


async def test_a_batch_cancelled_while_waiting_for_the_lock_leaves_no_transaction(tmp_path):
    """The price of `BEGIN IMMEDIATE` is that BEGIN waits, and a caller is
    often cancelled while it waits (a heartbeat stopped, a run cancelled). A
    statement already on aiosqlite's thread lands anyway, so if the caller's
    cancellation could interrupt the batch between its BEGIN and its COMMIT,
    the store's connection would sit in a write transaction nobody commits —
    holding the file's lock against every other process and refusing its own
    next BEGIN. It cannot today, because the store runs batches in a task of
    its own and a cancelled call only stops waiting for it; this pins that,
    deterministically, in one process: another connection holds the lock
    while the call is cancelled, then lets go."""
    import aiosqlite

    db = await initialised(tmp_path)
    async with AsyncExitStack() as stack:
        _, store = await open_persistence(db, stack)
        async with aiosqlite.connect(db, isolation_level=None) as holder:
            await holder.execute("BEGIN IMMEDIATE")          # the lock is taken
            put = asyncio.create_task(store.aput(("t",), "k", {"v": 1}))
            await asyncio.sleep(0.2)                         # ...waiting on it
            put.cancel()
            with pytest.raises(asyncio.CancelledError):
                await put
            await holder.execute("COMMIT")                   # the lock is free

            await asyncio.sleep(0.2)                         # the BEGIN has landed
            # the store's own connection is usable, and the file is not locked
            await asyncio.wait_for(store.aput(("t",), "k2", {"v": 2}), 2)
            await holder.execute("BEGIN IMMEDIATE")
            await holder.execute("COMMIT")
        assert (await store.aget(("t",), "k2")) is not None
