"""The JobManager's collaborators are ports: each one can be replaced.

These tests drive the job use cases with NO LangGraph graph and NO store —
which is the point of the split. If one of them starts needing a real graph
or a real store again, a responsibility has leaked back into the manager.
"""
from __future__ import annotations

import asyncio

import pytest

from jobsmith.engine.events import InProcessEvents, job_event
from jobsmith.engine.manager import JobManager
from jobsmith.engine.models import Job, JobStatus
from jobsmith.engine.repository import StoreJobRepository
from jobsmith.engine.runner import Fact, Output
from jobsmith.engine.usage import record_usage


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
        self.calls: list[tuple[str, dict]] = []
        self.resumed: list[str] = []

    async def stream(self, job_id, input):
        self.calls.append((job_id, input))
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
    """A JobRepository backed by plain dicts — no store, no namespaces.

    Process-local, so the manager never asks it for the control half of the
    port (leases, cancel requests): that half exists for other processes.
    """

    shared = False

    def __init__(self):
        self.summaries: dict[str, dict] = {}
        self.facts: dict[tuple[str, str], object] = {}
        self.errors: dict[str, list] = {}

    async def save_summary(self, job: Job) -> None:
        self.summaries[job.job_id] = job.summary()

    async def save_errors(self, job_id, errors) -> None:
        self.errors[job_id] = list(errors)

    async def save_fact(self, job_id, key, value) -> None:
        self.facts[(job_id, key)] = value

    async def load(self, job_id):
        summary = self.summaries.get(job_id)
        if summary is None:
            return None
        job = StoreJobRepository._from_summary(job_id, summary)
        for (jid, key), value in self.facts.items():
            if jid == job_id:
                job.facts[key] = value
        return job

    async def load_all(self, *, session_id=None, status=None, announced=None,
                       updated_since=None):
        jobs = [StoreJobRepository._from_summary(jid, s) for jid, s in self.summaries.items()]
        return [j for j in jobs
                if (session_id is None or j.session_id == session_id)
                and (status is None or j.status is status)
                and (announced is None or j.announced == announced)
                and (updated_since is None or j.updated_at >= updated_since)]


PLAN = {"rationale": "because", "steps": [{"capability": "alpha", "depends_on": []}]}


def ended(kind, answer=None, *, message=None, errors=(), write_error=None):
    """What a DAG run returns, reduced to the keys the job reads of it."""
    return Output({"terminal_kind": kind, "final_answer": answer, "user_error_message": message,
                   "errors": list(errors), "document_error": write_error})


def make_manager(tmp_path, *updates, pending=(), on_resume=None, **kwargs):
    return JobManager(
        repository=DictRepository(),
        runner=FakeRunner(*updates, pending=pending, on_resume=on_resume),
        **kwargs,
    )


async def test_use_cases_run_without_a_graph_or_a_store(tmp_path):
    mgr = make_manager(
        tmp_path,
        Fact("plan", PLAN),
        Fact("step:alpha", {"ok": True, "data": {"echo": "hi"}}),
        ended("answer", "The final answer."),
    )
    job = await mgr.create_job("do it", {"k": "v"}, session_id="s1", formats=["markdown"])
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE
    assert done.final_answer == "The final answer."
    assert done.facts["step:alpha"]["data"]["echo"] == "hi"
    assert done.facts_at["step:alpha"]
    # the runner received the job's own parameters
    [(job_id, graph_input)] = mgr.runner.calls
    assert job_id == job.job_id and graph_input["query"] == "do it"
    assert graph_input["inputs"] == {"k": "v"}
    # the repository saw each fact as it came, through the port only
    assert mgr.repo.facts[(job.job_id, "plan")] == PLAN
    assert (job.job_id, "step:alpha") in mgr.repo.facts


async def test_terminal_without_answer_fails_and_writes_no_report(tmp_path):
    mgr = make_manager(
        tmp_path,
        ended("user_error", message="cannot help with that",
              errors=[{"source": "planner", "kind": "plan_fail", "message": "nope"}]),
    )
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.FAILED
    assert done.error == "cannot help with that"
    assert done.facts == {}
    assert mgr.repo.errors[job.job_id][0]["kind"] == "plan_fail"


async def test_a_broken_runner_fails_the_job_rather_than_the_caller(tmp_path):
    class Exploding:
        async def stream(self, *a, **kw):
            raise RuntimeError("engine down")
            yield  # pragma: no cover — makes this an async generator

    mgr = JobManager(repository=DictRepository(), runner=Exploding())
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.FAILED
    assert "engine down" in done.error


async def test_resuming_goes_through_the_runner_port_too(tmp_path):
    """A resume asks the runner what is still pending and re-enters it — no
    graph, no checkpoint API, nothing the manager knows about LangGraph."""
    mgr = make_manager(tmp_path, Fact("plan", PLAN),
                       ended("answer", "Finished on the second try."), pending=("cap_alpha",))
    job = await mgr.create_job("resume me", formats=["markdown"])
    await mgr.cancel_job(job.job_id)

    done = await mgr.resume_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.final_answer == "Finished on the second try."
    assert mgr.runner.resumed == [job.job_id]   # re-entered...
    assert mgr.runner.calls == []               # ...never re-run from the query
    assert done.facts["plan"] == PLAN           # and what it published is the job's


async def test_resume_is_refused_when_the_runner_has_nothing_pending(tmp_path):
    mgr = make_manager(tmp_path, ended("answer", "unreachable"))  # pending: ()
    job = await mgr.create_job("q")
    await mgr.cancel_job(job.job_id)
    with pytest.raises(ValueError, match="no checkpoint to resume from"):
        await mgr.resume_job(job.job_id)
    assert mgr.runner.resumed == []


async def test_a_resumed_attempt_bills_on_top_of_the_stopped_one(tmp_path):
    """`job.usage` is what the JOB cost, not what its last attempt cost — the
    tokens the interrupted attempt burned were spent all the same."""
    mgr = make_manager(
        tmp_path, ended("answer", "Done at last."), pending=("cap_alpha",),
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
    mgr = make_manager(tmp_path, ended("answer", "done."), events=events)
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


async def test_a_fact_is_kept_as_it_arrives_whatever_it_says(tmp_path):
    """The manager records a fact under its key, stamped, without reading it:
    a file the run declared is one more fact (→ docs/design/core-v1.md)."""
    declared = {"path": "r.md", "format": "markdown", "role": "main"}
    mgr = make_manager(tmp_path, Fact("artifact:r.md", declared), Fact("anything", [1, 2]),
                       ended("answer", "Done."))
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)

    assert done.facts == {"artifact:r.md": declared, "anything": [1, 2]}
    assert set(done.facts_at) == {"artifact:r.md", "anything"}
    assert mgr.repo.facts[(job.job_id, "artifact:r.md")] == declared
    assert set(mgr.repo.summaries[job.job_id]["facts_at"]) == {"artifact:r.md", "anything"}


async def test_a_failed_report_write_leaves_the_job_done_and_persisted(tmp_path):
    """The run answered; only the file failed. That is DONE with the error the
    run reported — and, above all, *persisted*: a job with an answer nobody
    could reach is what an escaping write once left behind. → 0028"""
    mgr = make_manager(tmp_path, Fact("plan", PLAN), ended(
        "answer", "The answer.", write_error="markdown: No space left on device"))
    job = await mgr.create_job("q", formats=["markdown"])
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE          # the work is not the file
    assert done.final_answer == "The answer."
    assert done.error == "markdown: No space left on device"

    stored = mgr.repo.summaries[job.job_id]       # what a later reader sees
    assert stored["status"] == "done" and stored["error"] == done.error
    assert stored["final_answer"] == "The answer."


