"""The JobManager's collaborators are ports: each one can be replaced.

These tests drive the job use cases with NO LangGraph graph and NO store —
which is the point of the split. If one of them starts needing a real graph
or a real store again, a responsibility has leaked back into the manager.
"""
from __future__ import annotations

import asyncio

import pytest

from jobsmith.core.usage import record_usage
from jobsmith.jobs.events import InProcessEvents, job_event
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import Job, JobStatus
from jobsmith.jobs.repository import StoreJobRepository
from jobsmith.jobs.runner import NodeErrors, PlanReady, StepFinished, Terminal


class FakeRunner:
    """Replays a scripted list of domain updates, as a real run would.

    `pending` is what the graph's checkpoint would still have to run — the
    only thing the manager asks before resuming; `on_resume` lets a test do
    something (spend, say) inside the resumed run.
    """

    def __init__(self, *updates, pending=(), on_resume=None):
        self.updates = updates
        self.pending_nodes = tuple(pending)
        self.on_resume = on_resume
        self.calls: list[tuple[str, str, dict]] = []
        self.resumed: list[str] = []

    async def stream(self, job_id, query, inputs):
        self.calls.append((job_id, query, inputs))
        for update in self.updates:
            yield update

    async def pending(self, job_id):
        return self.pending_nodes

    async def resume(self, job_id):
        self.resumed.append(job_id)
        if self.on_resume is not None:
            self.on_resume()
        for update in self.updates:
            yield update


class DictRepository:
    """A JobRepository backed by plain dicts — no store, no namespaces."""

    def __init__(self):
        self.summaries: dict[str, dict] = {}
        self.plans: dict[str, object] = {}
        self.results: dict[tuple[str, str], dict] = {}
        self.errors: dict[str, list] = {}

    async def save_summary(self, job: Job) -> None:
        self.summaries[job.job_id] = job.summary()

    async def save_plan(self, job_id, plan) -> None:
        self.plans[job_id] = plan

    async def save_errors(self, job_id, errors) -> None:
        self.errors[job_id] = list(errors)

    async def save_result(self, job_id, capability, result) -> None:
        self.results[(job_id, capability)] = result

    async def load(self, job_id):
        summary = self.summaries.get(job_id)
        if summary is None:
            return None
        job = StoreJobRepository._from_summary(job_id, summary)
        job.plan = self.plans.get(job_id)
        for (jid, capability), result in self.results.items():
            if jid == job_id:
                job.results[capability] = result
        return job

    async def load_all(self, *, limit=50):
        return [StoreJobRepository._from_summary(jid, s) for jid, s in self.summaries.items()]


PLAN = {"rationale": "because", "steps": [{"capability": "alpha", "depends_on": []}]}


def make_manager(tmp_path, *updates, pending=(), on_resume=None, **kwargs):
    return JobManager(
        repository=DictRepository(),
        runner=FakeRunner(*updates, pending=pending, on_resume=on_resume),
        reports_dir=tmp_path / "artifacts",
        **kwargs,
    )


async def test_use_cases_run_without_a_graph_or_a_store(tmp_path):
    mgr = make_manager(
        tmp_path,
        PlanReady(PLAN),
        StepFinished("alpha", {"ok": True, "data": {"echo": "hi"}}),
        Terminal("answer", "The final answer.", None),
    )
    job = await mgr.create_job("do it", {"k": "v"}, session_id="s1")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE
    assert done.final_answer == "The final answer."
    assert done.results["alpha"]["data"]["echo"] == "hi"
    assert done.step_finished_at["alpha"]
    assert done.report_path is not None
    # the runner received the job's own parameters
    assert mgr.runner.calls == [(job.job_id, "do it", {"k": "v"})]
    # the repository saw the plan and the step result, through the port only
    assert mgr.repo.plans[job.job_id] == PLAN
    assert (job.job_id, "alpha") in mgr.repo.results


async def test_terminal_without_answer_fails_and_writes_no_report(tmp_path):
    mgr = make_manager(
        tmp_path,
        NodeErrors([{"source": "planner", "kind": "plan_fail", "message": "nope"}]),
        Terminal("user_error", None, "cannot help with that"),
    )
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.FAILED
    assert done.error == "cannot help with that"
    assert done.outputs == [] and done.report_path is None
    assert mgr.repo.errors[job.job_id][0]["kind"] == "plan_fail"


async def test_a_broken_runner_fails_the_job_rather_than_the_caller(tmp_path):
    class Exploding:
        async def stream(self, *a, **kw):
            raise RuntimeError("engine down")
            yield  # pragma: no cover — makes this an async generator

    mgr = JobManager(repository=DictRepository(), runner=Exploding(),
                     reports_dir=tmp_path / "artifacts")
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.FAILED
    assert "engine down" in done.error


async def test_resuming_goes_through_the_runner_port_too(tmp_path):
    """A resume asks the runner what is still pending and re-enters it — no
    graph, no checkpoint API, nothing the manager knows about LangGraph."""
    mgr = make_manager(tmp_path, Terminal("answer", "Finished on the second try.", None),
                       pending=("cap_alpha",))
    job = await mgr.create_job("resume me")
    await mgr.cancel_job(job.job_id)

    done = await mgr.resume_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.final_answer == "Finished on the second try."
    assert mgr.runner.resumed == [job.job_id]   # re-entered...
    assert mgr.runner.calls == []               # ...never re-run from the query
    assert done.report_path is not None         # and it still produces its deliverable


async def test_resume_is_refused_when_the_runner_has_nothing_pending(tmp_path):
    mgr = make_manager(tmp_path, Terminal("answer", "unreachable", None))  # pending: ()
    job = await mgr.create_job("q")
    await mgr.cancel_job(job.job_id)
    with pytest.raises(ValueError, match="no checkpoint to resume from"):
        await mgr.resume_job(job.job_id)
    assert mgr.runner.resumed == []


async def test_a_resumed_attempt_bills_on_top_of_the_stopped_one(tmp_path):
    """`job.usage` is what the JOB cost, not what its last attempt cost — the
    tokens the interrupted attempt burned were spent all the same."""
    mgr = make_manager(
        tmp_path, Terminal("answer", "Done at last.", None), pending=("cap_alpha",),
        on_resume=lambda: record_usage("claude-opus-5", input_tokens=10, output_tokens=5),
    )
    job = await mgr.create_job("expensive")
    await mgr.cancel_job(job.job_id)
    mgr.repo.summaries[job.job_id]["usage"] = {
        "input_tokens": 100, "output_tokens": 50, "calls": 1,
        "cost_usd": 0.5, "models": ["claude-opus-5"],
    }

    done = await mgr.resume_job(job.job_id)
    assert done.usage["calls"] == 2
    assert (done.usage["input_tokens"], done.usage["output_tokens"]) == (110, 55)
    assert done.usage["cost_usd"] > 0.5


async def test_manager_refuses_to_be_built_without_a_backing(tmp_path):
    with pytest.raises(ValueError, match="store or an explicit repository"):
        JobManager(runner=FakeRunner())
    with pytest.raises(ValueError, match="graph or an explicit runner"):
        JobManager(repository=DictRepository())


async def test_events_are_published_through_the_port(tmp_path):
    events = InProcessEvents()
    mgr = make_manager(tmp_path, Terminal("answer", "done.", None), events=events)
    queue = mgr.subscribe()
    job = await mgr.create_job("watched", session_id="s1")
    await mgr.run_job(job.job_id)

    seen = []
    while not queue.empty():
        seen.append(queue.get_nowait())
    assert [e["status"] for e in seen][:2] == ["queued", "running"]
    assert seen[-1]["status"] == "done"
    assert all(e["session_id"] == "s1" for e in seen)

    mgr.unsubscribe(queue)
    await mgr.mark_announced(job.job_id)
    assert queue.empty()


async def test_a_full_subscriber_is_dropped_not_awaited():
    """A stalled consumer must never block a running job."""
    events = InProcessEvents()
    queue = events.subscribe(max_queue=1)
    job = Job(job_id="j", status=JobStatus.RUNNING, query="q")

    for _ in range(5):
        await asyncio.wait_for(asyncio.to_thread(events.publish, job_event(job)), timeout=1)
    assert queue.qsize() == 1  # the rest were dropped, publish never blocked
