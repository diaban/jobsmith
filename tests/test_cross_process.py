"""Cross-process ownership and cancellation (#10).

Two `JobManager`s over the same SQLite file, each on its own connections, are
two processes as far as anything here can tell: neither sees the other's
`_tasks`, and the store is the only thing they share. Most properties are
driven that way; the last two cross a REAL process boundary, because
in-process doubles are exactly how this class of defect stayed invisible —
a pid that is ours is always alive, and a task we hold is always cancellable.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest
from conftest import FakeLLM, plan_json
from test_jobs import CountingEcho

from jobsmith.app.persistence import open_persistence
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import Job, JobStatus, now_iso
from jobsmith.jobs.ownership import LeasePolicy, ProcessIdentity

FAST = LeasePolicy(heartbeat=0.05, ttl=30.0, poll=0.02)
TESTS = Path(__file__).parent


def plan_llm() -> FakeLLM:
    return FakeLLM({"planner": plan_json("alpha", "slow", deps={"slow": ["alpha"]})},
                   default="A sufficiently long final answer for the job test.")


@asynccontextmanager
async def process(db: str, tmp_path, *, slow_delay: float = 30.0,
                  policy: LeasePolicy = FAST):
    """One 'process': its own connections, graph, capabilities and manager."""
    async with AsyncExitStack() as stack:
        checkpointer, store = await open_persistence(db, stack)
        alpha, slow = CountingEcho("alpha"), CountingEcho("slow", delay=slow_delay)
        graph = build_agent(Deps(llm=plan_llm()), CapabilityRegistry([alpha, slow]),
                            checkpointer=checkpointer)
        mgr = JobManager(graph, store, reports_dir=tmp_path / "artifacts", lease=policy)
        mgr.caps = (alpha, slow)          # type: ignore[attr-defined]  (test handle)
        yield mgr


async def until(predicate, *, timeout: float = 10.0, every: float = 0.02):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = await predicate()
        if value:
            return value
        await asyncio.sleep(every)
    raise AssertionError("condition never held")


async def running_in_slow(mgr: JobManager, job_id: str) -> Job | None:
    """The job is RUNNING and inside its second step (alpha is stored)."""
    job = await mgr.get_job(job_id)
    if job and job.status is JobStatus.RUNNING and "alpha" in job.results:
        return job
    return None


async def start_slow_job(owner: JobManager) -> tuple[Job, asyncio.Task]:
    job = await owner.create_job("a job worth stopping")
    task = owner.start_job(job.job_id)
    await until(lambda: running_in_slow(owner, job.job_id))
    return job, task


# ---------------- ownership ----------------


async def test_a_second_process_does_not_fail_the_first_one_s_live_job(tmp_path):
    """The defect that made this urgent: `recover_interrupted` in a second
    `jobsmith chat` declared dead every RUNNING job it did not hold itself."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as first:
        job, task = await start_slow_job(first)
        async with process(db, tmp_path) as second:
            assert await second.recover_interrupted() == []
            still = await second.get_job(job.job_id)
            assert still.status is JobStatus.RUNNING and still.error is None
        assert not task.done()
        await first.cancel_job(job.job_id)


async def test_the_lease_is_written_before_the_job_says_running(tmp_path):
    """A RUNNING record with no lease is what a startup settles — so the
    lease must already be there the instant RUNNING is readable."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as owner:
        seen: list[bool] = []
        save = owner.repo.save_summary

        async def watching(job):
            if job.status is JobStatus.RUNNING:
                seen.append((await owner.repo.load_control(job.job_id)).lease is not None)
            await save(job)

        owner.repo.save_summary = watching      # type: ignore[method-assign]
        job, _ = await start_slow_job(owner)
        assert seen and all(seen)
        await owner.cancel_job(job.job_id)


async def test_a_job_whose_owner_died_is_recovered(tmp_path):
    """Both proofs of death: an expired lease (any machine), and a pid that no
    longer exists on this one (at once, without waiting out the TTL)."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as survivor:
        me = survivor.identity
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        expired = ProcessIdentity(token="elsewhere", host="another-host", pid=1)
        crashed = ProcessIdentity(token="crashed", host=me.host, pid=dead.pid,
                                  pidns=me.pidns)
        alive = ProcessIdentity(token="alive", host="another-host", pid=1)
        jobs = {}
        for name, who, ttl in (("expired", expired, -1), ("crashed", crashed, 30),
                               ("alive", alive, 30)):
            job = await survivor.create_job(name)
            await survivor.repo.save_lease(job.job_id, who.lease(ttl))
            job.status = JobStatus.RUNNING
            await survivor.repo.save_summary(job)
            jobs[name] = job.job_id

        recovered = {j.job_id for j in await survivor.recover_interrupted()}
        assert recovered == {jobs["expired"], jobs["crashed"]}
        assert (await survivor.get_job(jobs["alive"])).status is JobStatus.RUNNING
        for name in ("expired", "crashed"):
            job = await survivor.get_job(jobs[name])
            assert job.status is JobStatus.FAILED and "interrupted" in job.error


# ---------------- cancellation ----------------


async def test_a_cancel_from_another_process_stops_the_run(tmp_path):
    """Not a tombstone the runner overwrites: the owner stops, the record ends
    CANCELLED, and stays so — no DONE after the cancel was answered."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as owner, process(db, tmp_path) as other:
        job, task = await start_slow_job(owner)

        stopped = await other.cancel_job(job.job_id)

        assert stopped.status is JobStatus.CANCELLED
        # the owner's run really stopped — and its awaiter got the job back
        # rather than a cancellation it never asked for
        settled = await asyncio.wait_for(task, 5)
        assert settled.status is JobStatus.CANCELLED
        _alpha, slow = owner.caps                  # type: ignore[attr-defined]
        assert slow.runs == 1
        await asyncio.sleep(0.3)                   # several heartbeats later...
        final = await other.get_job(job.job_id)
        assert final.status is JobStatus.CANCELLED  # ...still what was answered
        assert set(final.results) == {"alpha"}
        assert (await owner.repo.load_control(job.job_id)).lease is None  # released


async def test_resume_works_after_a_cross_process_cancel(tmp_path):
    """The checkpoint is retained, and the stale request does not stop the
    resumed attempt on its first heartbeat."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as owner, process(db, tmp_path,
                                                       slow_delay=0.2) as other:
        job, task = await start_slow_job(owner)
        await other.cancel_job(job.job_id)
        await asyncio.wait_for(task, 5)

        done = await other.resume_job(job.job_id)     # heartbeats at 0.05 s during 0.2 s
        assert done.status is JobStatus.DONE
        alpha, _ = owner.caps                          # type: ignore[attr-defined]
        assert alpha.runs == 1                         # the finished step was kept
        assert done.results["alpha"]["data"]["echo"] == "alpha#1"
        assert done.results["slow"]["data"]["echo"] == "slow#1"   # run by `other`


async def test_a_slow_owner_is_never_reported_cancelled_early(tmp_path):
    """If the owner has not acted by the time the canceller stops waiting,
    the answer is what is TRUE — still running, request recorded — and the
    owner still acts on it later."""
    db = str(tmp_path / "agent.db")
    slow_heart = LeasePolicy(heartbeat=1.0, ttl=30.0)
    impatient = LeasePolicy(heartbeat=0.05, ttl=30.0, cancel_wait=0.1, poll=0.02)
    async with process(db, tmp_path, policy=slow_heart) as owner, \
            process(db, tmp_path, policy=impatient) as other:
        job, task = await start_slow_job(owner)

        answer = await other.cancel_job(job.job_id)
        assert answer.status is JobStatus.RUNNING
        assert (await other.repo.load_control(job.job_id)).cancel_requested_at

        settled = await asyncio.wait_for(task, 5)
        assert settled.status is JobStatus.CANCELLED


async def test_a_cancel_of_a_dead_owner_s_job_is_settled_by_the_canceller(tmp_path):
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as other:
        job = await other.create_job("orphaned")
        gone = ProcessIdentity(token="gone", host="another-host", pid=1)
        await other.repo.save_lease(job.job_id, gone.lease(-1))
        job.status = JobStatus.RUNNING
        await other.repo.save_summary(job)

        stopped = await other.cancel_job(job.job_id)
        assert stopped.status is JobStatus.CANCELLED
        assert "already stopped" in stopped.error


async def test_a_queued_job_cancelled_elsewhere_never_runs(tmp_path):
    """The race a tombstone alone loses: a process picks the job up at the
    instant another cancels it. The request is heard before anything runs."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as owner:
        job = await owner.create_job("cancelled before it started")
        await owner.repo.request_cancel(job.job_id, now_iso())   # arrived in between
        settled = await owner.run_job(job.job_id)
        assert settled.status is JobStatus.CANCELLED
        alpha, slow = owner.caps                                  # type: ignore[attr-defined]
        assert alpha.runs == slow.runs == 0


async def test_an_owner_that_lost_its_lease_stops_without_writing(tmp_path):
    """A live owner stalled past its TTL may be judged dead and its job taken
    over by another process. When it wakes it must not overwrite that
    settlement — nor run on beside a resume of the same checkpoint."""
    db = str(tmp_path / "agent.db")
    owner_policy = LeasePolicy(heartbeat=0.5, ttl=30.0)
    async with process(db, tmp_path, policy=owner_policy) as owner, \
            process(db, tmp_path) as other:
        job, task = await start_slow_job(owner)
        # what `recover_interrupted` does to a job whose lease it saw expire
        judged = await other.get_job(job.job_id)
        control = await other.repo.load_control(job.job_id)
        (taken,) = await other._take_over([(judged, control)])
        await other._settle(taken, JobStatus.FAILED, "interrupted: judged dead")

        returned = await asyncio.wait_for(task, 5)
        assert returned.status is JobStatus.FAILED
        await asyncio.sleep(0.6)                     # past another owner heartbeat
        final = await other.get_job(job.job_id)
        assert final.status is JobStatus.FAILED and final.error == "interrupted: judged dead"


async def test_a_takeover_backs_off_when_the_owner_answers(tmp_path):
    """An expired lease proves silence, not death. If the owner renews under
    the fence, it is alive, and its job is left to it."""
    db = str(tmp_path / "agent.db")
    async with process(db, tmp_path) as owner, process(db, tmp_path) as other:
        job, task = await start_slow_job(owner)
        judged = await other.get_job(job.job_id)
        control = await other.repo.load_control(job.job_id)
        renew = other.repo.save_lease

        async def owner_renews_right_after(job_id, lease):
            await renew(job_id, lease)                  # the fence...
            await renew(job_id, owner.identity.lease(30))  # ...lost to a renewal

        other.repo.save_lease = owner_renews_right_after   # type: ignore[method-assign]
        assert await other._take_over([(judged, control)]) == []
        assert not task.done()
        assert (await other.get_job(job.job_id)).status is JobStatus.RUNNING
        await owner.cancel_job(job.job_id)


async def test_a_process_local_store_coordinates_nothing(store, checkpointer, tmp_path,
                                                         monkeypatch):
    """Memory is what tests and evals force: no lease, no heartbeat, no poll."""
    def no_heartbeat(*a, **kw):
        raise AssertionError("a heartbeat was started on a process-local store")

    monkeypatch.setattr("jobsmith.jobs.manager.Heartbeat", no_heartbeat)
    alpha, slow = CountingEcho("alpha"), CountingEcho("slow")
    graph = build_agent(Deps(llm=plan_llm()), CapabilityRegistry([alpha, slow]),
                        checkpointer=checkpointer)
    mgr = JobManager(graph, store, reports_dir=tmp_path / "artifacts")
    assert mgr.repo.shared is False
    job = await mgr.create_job("q")
    assert (await mgr.run_job(job.job_id)).status is JobStatus.DONE
    assert await store.asearch(("jobs", job.job_id, "control")) == []


# ---------------- a real process boundary ----------------


async def spawn_owner(db: str, tmp_path) -> tuple[asyncio.subprocess.Process, str]:
    env = os.environ | {"PYTHONPATH": os.pathsep.join(
        [str(TESTS), os.environ.get("PYTHONPATH", "")])}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(TESTS / "owner_process.py"), db, str(tmp_path / "artifacts"),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env)
    assert proc.stdout is not None
    line = await asyncio.wait_for(proc.stdout.readline(), 60)
    return proc, line.decode().strip()


async def settled_checkpoint(mgr: JobManager, job_id: str) -> None:
    """Wait until the owner's checkpoint says `slow` is what runs next.

    LangGraph saves a superstep's checkpoint and its pending writes in
    separate commits, so a process killed between the two leaves a thread
    with nothing to resume — its crash-consistency, not this issue's. The
    subprocess reaches `slow` a few milliseconds after `alpha` is visible, so
    this waits for the frontier to be written and to stay written.
    """
    for _ in range(2):
        await until(lambda: mgr.runner.pending(job_id), timeout=10)
        await asyncio.sleep(0.5)


@pytest.mark.skipif(os.name == "nt", reason="pid probing is POSIX-only")
async def test_a_cancel_crosses_a_real_process_boundary_and_resumes(tmp_path):
    db = str(tmp_path / "agent.db")
    proc, job_id = await spawn_owner(db, tmp_path)
    try:
        async with process(db, tmp_path, slow_delay=0.0) as here:
            await until(lambda: running_in_slow(here, job_id), timeout=30)
            await settled_checkpoint(here, job_id)

            stopped = await here.cancel_job(job_id)
            assert stopped.status is JobStatus.CANCELLED

            out, _ = await asyncio.wait_for(proc.communicate(), 30)
            assert proc.returncode == 0                   # its awaiter got a job back
            assert out.decode().strip() == "cancelled"    # ...a CANCELLED one

            done = await here.resume_job(job_id)
            assert done.status is JobStatus.DONE
            assert done.results["alpha"]["data"]["echo"] == "alpha@0.0"  # the owner's
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.skipif(os.name == "nt", reason="pid probing is POSIX-only")
async def test_a_killed_owner_is_recovered_at_once_and_resumes(tmp_path):
    """SIGKILL leaves a lease valid for another 30 s: the pid is what proves
    the owner gone, so the next startup settles it without waiting."""
    db = str(tmp_path / "agent.db")
    proc, job_id = await spawn_owner(db, tmp_path)
    try:
        async with process(db, tmp_path, slow_delay=0.0,
                           policy=LeasePolicy()) as here:     # production TTL
            await until(lambda: running_in_slow(here, job_id), timeout=30)
            await settled_checkpoint(here, job_id)
            proc.kill()
            await proc.wait()

            (stale,) = await here.recover_interrupted()
            assert stale.job_id == job_id and stale.status is JobStatus.FAILED

            done = await here.resume_job(job_id)
            assert done.status is JobStatus.DONE
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
