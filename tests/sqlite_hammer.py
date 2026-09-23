"""One writer, in a process of its own, hammering a shared SQLite file.

    python tests/sqlite_hammer.py <db> <store|saver|setup> <rounds> <tag> <start_at>

Driven by tests/test_sqlite_concurrency.py, several at once on ONE file, which
is the normal case since #63 made a per-user SQLite file the default: every
`jobsmith chat` / `jobsmith run` on the machine writes to it.

It opens the file through `open_persistence` — the production path, so the
connections are configured exactly as the product's are — then waits until
`start_at` (a wall-clock time shared by every hammer) so the processes really
overlap, and writes `rounds` times through ONE of the two LangGraph backends,
straight on the backend and never through `StoreJobRepository`, whose retry
(#10) would hide exactly what this measures:

    store   a batch that READS then WRITES (a GetOp and a PutOp in one
            `abatch`) — the shape of every store batch that fails with a
            deferred `BEGIN`: the read takes a snapshot, another process
            commits, and the upgrade to a write is refused at once.
    saver   `aput` + `aput_writes` on a thread of its own — what every job
            superstep and every chat turn does.
    setup   nothing but OPENING the file, all processes at `start_at`: on a
            fresh file that is every process running the store's migrations
            at once — the default's first-ever `jobsmith chat` next to a
            first-ever `jobsmith run`.

Prints `ok <rounds>` and exits 0, or prints the first error and exits 1.
Kept outside `test_*` so pytest does not collect it.
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import AsyncExitStack

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.store.base import GetOp, PutOp

from jobsmith.app.persistence import open_persistence


async def hammer_store(store, rounds: int, tag: str) -> None:
    for i in range(rounds):
        await store.abatch([
            GetOp(("hammer", tag), f"k{i - 1}"),
            PutOp(("hammer", tag), f"k{i}", {"i": i, "pad": "x" * 512}),
        ])


async def hammer_saver(saver, rounds: int, tag: str) -> None:
    config = {"configurable": {"thread_id": f"hammer-{tag}", "checkpoint_ns": ""}}
    for i in range(rounds):
        checkpoint = empty_checkpoint()
        saved = await saver.aput(config, checkpoint, {"step": i}, {})
        await saver.aput_writes(saved, [("channel", {"i": i, "pad": "x" * 512})], f"t{i}")
        config = saved


async def main(db: str, mode: str, rounds: int, tag: str, start_at: float) -> int:
    if mode == "setup":
        await asyncio.sleep(max(0.0, start_at - time.time()))
    async with AsyncExitStack() as stack:
        try:
            checkpointer, store = await open_persistence(db, stack)
        except Exception as e:
            print(f"error {type(e).__name__}: {e}", flush=True)
            return 1
        await asyncio.sleep(max(0.0, start_at - time.time()))
        try:
            if mode == "setup":
                pass
            elif mode == "store":
                await hammer_store(store, rounds, tag)
            else:
                await hammer_saver(checkpointer, rounds, tag)
        except Exception as e:
            print(f"error {type(e).__name__}: {e}", flush=True)
            return 1
    print(f"ok {rounds}", flush=True)
    return 0


if __name__ == "__main__":
    db, mode, rounds, tag, start_at = sys.argv[1:6]
    sys.exit(asyncio.run(main(db, mode, int(rounds), tag, float(start_at))))
