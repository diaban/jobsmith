"""Which process owns a running job, and how it hears that it must stop (#10).

A job record lives in a store every process on the machine may share (a SQLite
file, a Postgres database). `JobManager._tasks` only knows the jobs *this*
process runs, so two questions have no answer there, and both have to be
answered from the store — the one channel every backing already shares:

- **is the owner of this RUNNING job still alive?** — asked at startup by
  `recover_interrupted`, and by a `cancel_job` that has no task to cancel.
  Getting it wrong in one direction declares dead a job another terminal is
  still running (the defect that made this module); in the other it leaves a
  job RUNNING forever.
- **has somebody asked this run to stop?** — asked by the owner, which is the
  only process that can actually stop it.

The answer to the first is a **lease**: the owner writes who it is and an
expiry *before* it marks the job RUNNING, and renews it on a heartbeat while
the run lasts. The owner is **provably gone** when either

1. the lease has expired — it stopped renewing, wherever it ran; or
2. it ran on *this* machine, in *this* pid namespace, and its pid no longer
   exists — a crash is recognised at once rather than a TTL later.

Neither condition alone would do. A pid is meaningless across machines sharing
Postgres and is reused on one machine, so (2) is only ever consulted as a
proof of death, never of life: a reused pid makes a dead owner look alive,
which (1) corrects a TTL later. A lease alone would make every restart after
a crash wait out the TTL before the orphan could be settled — and startup is
the one moment `recover_interrupted` looks.

The answer to the second is a **cancel request** written next to the lease.
The owner reads both in one round trip on every heartbeat, so the latency of a
cross-process cancel is one heartbeat plus however long the run takes to reach
its next `await` (see `LeasePolicy`).

Nothing here runs for a process-local store: `JobRepository.shared` is False
for the in-memory backend, and the manager then takes none of these paths — a
single process has nothing to coordinate and tests stay as fast as they were.
"""
from __future__ import annotations

import asyncio
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .repository import JobRepository


@dataclass(frozen=True)
class LeasePolicy:
    """The timings of ownership. Defaults are the product's; tests shrink them.

    - `heartbeat` (2 s) is how often the owner renews its lease **and** reads
      the cancel request, so it is the worst-case delay before a cancel issued
      from another process reaches the run. Cancellation is the user's only
      undo (#83): two seconds is inside the time a person waits for "did that
      work?" without pressing it again, and a tiny write + read every two
      seconds per running job is nothing to a SQLite WAL or a Postgres pool.
    - `ttl` (30 s) is how long a lease is believed without renewal — fifteen
      missed heartbeats. Deliberately generous: the owner's event loop can
      stall on synchronous work (a PDF rendered in-loop, a slow disk), and a
      lease that expires under a live owner is the one mistake that would let
      another process settle — and later resume — a job that is still
      running. A crash on this machine does not wait for it (the pid check).
      Across machines it also absorbs ordinary clock skew, since expiry is a
      wall-clock timestamp compared by another host.
    - `cancel_wait` is how long `cancel_job` waits for the owner to confirm
      the stop before answering with the status as it stands (still RUNNING,
      the request recorded). Defaults to three heartbeats plus a second, so
      a live owner always makes it; `poll` is how often it looks.
    """

    heartbeat: float = 2.0
    ttl: float = 30.0
    cancel_wait: float | None = None
    poll: float = 0.25

    @property
    def wait(self) -> float:
        return self.cancel_wait if self.cancel_wait is not None else 3 * self.heartbeat + 1.0


def _pid_namespace() -> str:
    """The pid namespace this process lives in, where the OS says so.

    Two containers on one kernel can share a hostname and a SQLite volume, and
    pid 42 in one is not pid 42 in the other — so a pid is only compared
    within a namespace known to be the same one.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    """Does `pid` exist on this machine? `True` whenever it cannot tell."""
    if os.name == "nt":            # os.kill(pid, 0) TERMINATES a process on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:                # EPERM: it exists, it is just not ours
        return True
    return True


@dataclass(frozen=True)
class ProcessIdentity:
    """Who a JobManager is, as written on the leases it holds.

    `token` is unique per manager instance, not per pid: two managers in one
    process (tests, an app composed twice) are two owners, and a pid reused
    by a later process never inherits an earlier one's jobs.
    """

    token: str
    host: str
    pid: int
    pidns: str = ""

    @classmethod
    def current(cls) -> ProcessIdentity:
        return cls(token=uuid.uuid4().hex, host=socket.gethostname(),
                   pid=os.getpid(), pidns=_pid_namespace())

    @property
    def label(self) -> str:
        return f"{self.host}:{self.pid}"

    def lease(self, ttl: float, *, now: datetime | None = None) -> Lease:
        now = now or datetime.now(UTC)
        return Lease(owner=self.token, host=self.host, pid=self.pid, pidns=self.pidns,
                     expires_at=(now + timedelta(seconds=ttl)).isoformat())


@dataclass(frozen=True)
class Lease:
    """An owner's claim on a running job, valid until `expires_at`."""

    owner: str
    host: str
    pid: int
    expires_at: str
    pidns: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "host": self.host, "pid": self.pid,
                "pidns": self.pidns, "expires_at": self.expires_at}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lease:
        return cls(owner=str(d.get("owner", "")), host=str(d.get("host", "")),
                   pid=int(d.get("pid", 0)), pidns=str(d.get("pidns", "")),
                   expires_at=str(d.get("expires_at", "")))

    def expired(self, now: datetime | None = None) -> bool:
        try:
            expiry = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True            # a claim nobody can read claims nothing
        return (now or datetime.now(UTC)) >= expiry


@dataclass(frozen=True)
class JobControl:
    """What the store says about a job's owner and whether it must stop."""

    lease: Lease | None = None
    cancel_requested_at: str | None = None


def owner_is_gone(lease: Lease | None, *, here: ProcessIdentity,
                  now: datetime | None = None) -> bool:
    """Is the owner of this lease provably no longer running the job?

    No lease at all is gone: a RUNNING record written before leases existed
    says nothing about an owner, so it has none (a live owner cannot leave
    one — it writes its lease before RUNNING).
    """
    if lease is None or lease.expired(now):
        return True
    return death_is_certain(lease, here=here)


def death_is_certain(lease: Lease | None, *, here: ProcessIdentity) -> bool:
    """Gone beyond doubt — not merely silent past its TTL.

    An expired lease proves the owner stopped *renewing*, not that it stopped
    running: an event loop stalled past the TTL wakes up and renews. A pid
    missing from this machine cannot. The difference decides whether a
    takeover must wait to see if the owner answers (`JobManager._take_over`).
    """
    if lease is None:
        return True
    same_machine = lease.host == here.host and lease.pidns == here.pidns
    return same_machine and not pid_alive(lease.pid)


class Heartbeat:
    """The owner's side of a running job on a shared store.

    Renews the lease every `policy.heartbeat` seconds and, on the same round
    trip, reads the cancel request. Two outcomes stop the run it watches,
    both by cancelling `run_task` — the one mechanism an in-process cancel
    already uses, so a remote stop settles through exactly the same path:

    - `requested` — somebody asked this job to stop;
    - `lost` — the lease now names another owner: a process judged this one
      dead (an event loop stalled past the TTL) and settled the job. This run
      is no longer the job's, so it must stop *without persisting*, or its
      next write would overwrite the settlement and a resume elsewhere would
      race it on the same checkpoint.

    `stop()` is synchronous on purpose: the manager calls it immediately after
    the run's last update, with no `await` in between, so a cancel can never
    land on the settlement of a run that already finished.
    """

    def __init__(self, repo: JobRepository, job_id: str, me: ProcessIdentity,
                 policy: LeasePolicy, run_task: asyncio.Task):
        self.repo = repo
        self.job_id = job_id
        self.me = me
        self.policy = policy
        self.run_task = run_task
        self.requested = False
        self.lost = False
        self._closed = False
        self._task = asyncio.create_task(self._loop(), name=f"heartbeat:{job_id}")

    def stop(self) -> None:
        self._closed = True
        self._task.cancel()

    async def wait_closed(self) -> None:
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.policy.heartbeat)
            try:
                control = await self.repo.load_control(self.job_id)
            except Exception:
                # A busy store (another writer holding the SQLite lock past
                # its timeout) must not end the heartbeat: a dead heartbeat
                # under a live run is a lease left to expire, and a job
                # another process will settle. The next tick tries again,
                # and the TTL is fifteen of them.
                continue
            if self._closed:
                return
            if control.lease is not None and control.lease.owner != self.me.token:
                self.lost = True
                self.run_task.cancel()
                return
            if control.cancel_requested_at:
                self.requested = True
                self.run_task.cancel()
                return
            try:
                await self.repo.save_lease(self.job_id, self.me.lease(self.policy.ttl))
            except Exception:
                continue


__all__ = ["Heartbeat", "JobControl", "Lease", "LeasePolicy", "ProcessIdentity",
           "death_is_certain", "owner_is_gone", "pid_alive"]
