"""JobManager: lifecycle, store schema, listing, cancellation."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import JobStatus


class SlowEcho(Capability):
    def __init__(self, name: str, *, delay: float = 0.0, fail: bool = False):
        self.spec = CapabilitySpec(name=name, description=f"{name} capability")
        self.delay = delay
        self.fail = fail

    async def work(self, state: CapabilityBaseState) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return self._emit_failure(f"{self.spec.name} broke")
        return self._emit_success({"echo": self.spec.name})

    def render_context(self, result):
        return f"# {self.spec.name}\n{result['data']['echo']}"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


class CountingEcho(SlowEcho):
    """Echoes how many times it has run, so a re-run shows up in the result."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.runs = 0

    async def work(self, state: CapabilityBaseState) -> dict:
        self.runs += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._emit_success({"echo": f"{self.spec.name}#{self.runs}"})


async def cancelled_midway(store, checkpointer, tmp_path):
    """A job stopped *inside* its second step: one result stored, one pending.

    The shape every resume test needs — and the one a job really stops in,
    since a cancellation lands wherever the run happened to be. Returns the
    manager, the job, and the two capabilities (mutate `slow.delay` to let the
    interrupted step finish instantly when the run is resumed).
    """
    alpha, slow = CountingEcho("alpha"), CountingEcho("slow", delay=30.0)
    llm = FakeLLM(
        {"planner": plan_json("alpha", "slow", deps={"slow": ["alpha"]})},
        default="A sufficiently long final answer for the job test.",
    )
    mgr = make_manager(store, checkpointer, tmp_path, caps=[alpha, slow], llm=llm)
    job = await mgr.create_job("a job worth resuming")
    mgr.start_job(job.job_id)
    for _ in range(500):                       # wait until `slow` is actually running
        await asyncio.sleep(0.01)
        if slow.runs:
            break
    assert slow.runs == 1, "the second step never started"
    stopped = await mgr.cancel_job(job.job_id)
    assert stopped.status is JobStatus.CANCELLED
    assert set(stopped.results) == {"alpha"}   # the finished step was persisted
    return mgr, job, alpha, slow


def make_manager(store, checkpointer, tmp_path, *, caps=None, llm=None) -> JobManager:
    caps = caps if caps is not None else [SlowEcho("alpha")]
    llm = llm or FakeLLM(
        {"planner": plan_json(*[c.spec.name for c in caps])},
        default="A sufficiently long final answer for the job test.",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts")


async def test_create_run_done_with_store_contents(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("alpha"), SlowEcho("beta")])
    job = await mgr.create_job("do the thing", {"key": "val"})
    assert job.status is JobStatus.QUEUED

    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.terminal_kind == "answer"
    assert "final answer" in done.final_answer
    assert set(done.results) == {"alpha", "beta"}
    assert done.plan is not None

    # store schema: index + meta/plan + artifacts
    index = await store.aget(("jobs", "index"), job.job_id)
    assert index.value["status"] == "done"
    plan = await store.aget(("jobs", job.job_id, "meta"), "plan")
    assert [s["capability"] for s in plan.value["steps"]] == ["alpha", "beta"]
    step = await store.aget(("jobs", job.job_id, "results"), "alpha")
    assert step.value["ok"] is True

    # get_job reconstructs the same view
    fetched = await mgr.get_job(job.job_id)
    assert fetched.status is JobStatus.DONE
    assert fetched.results["beta"]["data"]["echo"] == "beta"


async def test_failed_job_records_error_and_errors_meta(store, checkpointer, tmp_path):
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("alpha")
            raise RuntimeError("llm down")

    mgr = make_manager(store, checkpointer, tmp_path, llm=ExplodingGenLLM())
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.FAILED
    assert done.terminal_kind == "escalated"
    assert done.error is not None
    errors = await store.aget(("jobs", job.job_id, "meta"), "errors")
    assert any(e["kind"] == "generation_fail" for e in errors.value)


async def test_run_requires_queued(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("q")
    await mgr.run_job(job.job_id)
    with pytest.raises(ValueError, match="expected queued"):
        await mgr.run_job(job.job_id)
    with pytest.raises(KeyError):
        await mgr.run_job("nonexistent")


async def test_list_jobs_filtering(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    j1 = await mgr.create_job("first")
    j2 = await mgr.create_job("second")
    await mgr.run_job(j1.job_id)

    all_jobs = await mgr.list_jobs()
    assert {j.job_id for j in all_jobs} == {j1.job_id, j2.job_id}
    queued = await mgr.list_jobs(status=JobStatus.QUEUED)
    assert [j.job_id for j in queued] == [j2.job_id]
    done = await mgr.list_jobs(status=JobStatus.DONE)
    assert [j.job_id for j in done] == [j1.job_id]


async def test_start_and_cancel_running_job(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("slow", delay=30.0)])
    job = await mgr.create_job("long thing")
    task = mgr.start_job(job.job_id)

    # wait until the job is actually RUNNING and inside the slow capability
    for _ in range(100):
        await asyncio.sleep(0.01)
        current = await mgr.get_job(job.job_id)
        if current.status is JobStatus.RUNNING and current.plan is not None:
            break

    cancelled = await mgr.cancel_job(job.job_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert task.cancelled()
    # checkpoint retained for the thread → future resume is possible
    snapshot = await mgr.graph.aget_state({"configurable": {"thread_id": job.job_id}})
    assert snapshot is not None and snapshot.values.get("plan") is not None


async def test_cancel_tombstone_without_task(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("q")
    cancelled = await mgr.cancel_job(job.job_id)  # never started
    assert cancelled.status is JobStatus.CANCELLED
    # cancelling a finished job is a no-op
    job2 = await mgr.create_job("q2")
    await mgr.run_job(job2.job_id)
    after = await mgr.cancel_job(job2.job_id)
    assert after.status is JobStatus.DONE


async def test_resume_finishes_a_cancelled_job_without_redoing_finished_steps(
    store, checkpointer, tmp_path
):
    """The point of resuming: the steps already paid for are kept, only the
    interrupted one runs again — and the job ends exactly like a normal run."""
    mgr, job, alpha, slow = await cancelled_midway(store, checkpointer, tmp_path)

    slow.delay = 0.0                              # the pending step can finish now
    done = await mgr.resume_job(job.job_id)

    assert done.status is JobStatus.DONE
    assert alpha.runs == 1                        # the finished step was NOT re-run
    assert slow.runs == 2                         # the interrupted one started over
    assert done.results["alpha"]["data"]["echo"] == "alpha#1"   # the original result
    assert done.results["slow"]["data"]["echo"] == "slow#2"
    # the stored result of the finished step was left untouched
    kept = await store.aget(("jobs", job.job_id, "results"), "alpha")
    assert kept.value["data"]["echo"] == "alpha#1"
    # the plan came back from the repository — a resumed stream never replans
    assert [s["capability"] for s in done.plan["steps"]] == ["alpha", "slow"]
    # and the deliverable is produced the same way a first attempt produces it
    assert done.report_path is not None
    assert "final answer" in Path(done.report_path).read_text()


async def test_resume_settles_a_job_its_process_died_on(store, checkpointer, tmp_path):
    """`recover_interrupted` marks a killed run FAILED but keeps its
    checkpoint. A *new process* over the same store must be able to finish it."""
    mgr, job, alpha, slow = await cancelled_midway(store, checkpointer, tmp_path)
    # what a killed process actually leaves behind: a RUNNING record + checkpoint
    record = await store.aget(("jobs", "index"), job.job_id)
    await store.aput(("jobs", "index"), job.job_id, record.value | {"status": "running"})

    # a fresh manager and a fresh graph over the same store and checkpointer
    restarted = make_manager(store, checkpointer, tmp_path, caps=[alpha, slow],
                             llm=FakeLLM(default="A sufficiently long final answer."))
    (stale,) = await restarted.recover_interrupted()
    assert stale.status is JobStatus.FAILED and "interrupted" in stale.error

    slow.delay = 0.0
    done = await restarted.resume_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.error is None                     # the interrupted message is stale
    assert alpha.runs == 1                        # still not re-run, across processes


async def test_resume_refuses_a_job_with_nothing_left_to_run(store, checkpointer, tmp_path):
    """FAILED is not a licence to resume: a run that reached `escalate` has an
    empty checkpoint frontier, and re-entering it would silently do nothing."""
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("alpha")
            raise RuntimeError("llm down")

    mgr = make_manager(store, checkpointer, tmp_path, llm=ExplodingGenLLM())
    job = await mgr.create_job("q")
    failed = await mgr.run_job(job.job_id)
    assert failed.status is JobStatus.FAILED and failed.terminal_kind == "escalated"
    with pytest.raises(ValueError, match="no checkpoint to resume from"):
        await mgr.resume_job(job.job_id)

    # a job cancelled before it ever started has no checkpoint at all
    never_ran = await mgr.create_job("q2")
    await mgr.cancel_job(never_ran.job_id)
    with pytest.raises(ValueError, match="no checkpoint to resume from"):
        await mgr.resume_job(never_ran.job_id)


async def test_resume_refuses_wrong_statuses(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("q")
    with pytest.raises(ValueError, match="expected cancelled or failed"):
        await mgr.resume_job(job.job_id)          # QUEUED: `run_job` is what it needs
    await mgr.run_job(job.job_id)
    with pytest.raises(ValueError, match="expected cancelled or failed"):
        await mgr.resume_job(job.job_id)          # DONE: iterating is another feature
    with pytest.raises(KeyError):
        await mgr.resume_job("nonexistent")


async def test_report_written_on_done(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("alpha"), SlowEcho("beta")])
    job = await mgr.create_job("write the report")
    done = await mgr.run_job(job.job_id)

    assert done.report_path is not None
    (output,) = done.outputs
    assert (output.role, output.format) == ("main", "markdown")
    report = (tmp_path / "artifacts" / f"{job.job_id}.md").read_text()
    assert report.startswith("# write the report")   # title, then the answer
    assert "final answer" in report                  # the deliverable itself
    assert "| alpha |" in report                     # provenance: plan table
    assert "flowchart LR" in report                  # mermaid DAG
    assert "Step output" not in report               # step material is not inlined
    # per-step timestamps recorded as capabilities finished
    assert set(done.step_finished_at) == {"alpha", "beta"}
    # the path survives a round-trip through the store
    fetched = await mgr.get_job(job.job_id)
    assert fetched.report_path == done.report_path


async def test_no_report_on_failed_job(store, checkpointer, tmp_path):
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("alpha")
            raise RuntimeError("llm down")

    mgr = make_manager(store, checkpointer, tmp_path, llm=ExplodingGenLLM())
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.FAILED
    assert done.report_path is None


async def test_session_filter_and_announcement_flow(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    in_session = await mgr.create_job("mine", session_id="s1")
    _other = await mgr.create_job("other", session_id="s2")
    no_session = await mgr.create_job("bare")

    listed = await mgr.list_jobs(session_id="s1")
    assert [j.job_id for j in listed] == [in_session.job_id]

    # nothing finished yet → nothing to announce
    assert await mgr.list_finished_unannounced("s1") == []

    await mgr.run_job(in_session.job_id)
    await mgr.run_job(no_session.job_id)  # finished but session-less: never announced
    pending = await mgr.list_finished_unannounced("s1")
    assert [j.job_id for j in pending] == [in_session.job_id]
    assert pending[0].final_answer is not None  # enough to build the synthesis

    await mgr.mark_announced(in_session.job_id)
    assert await mgr.list_finished_unannounced("s1") == []


async def test_subscribe_streams_job_events(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    queue = mgr.subscribe()
    job = await mgr.create_job("watched", session_id="s1")
    await mgr.run_job(job.job_id)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert events[0]["status"] == "queued"
    assert events[-1]["status"] == "done"
    assert events[-1]["report_path"] is not None
    assert all(e["job_id"] == job.job_id and e["session_id"] == "s1" for e in events)
    assert any(e["steps_done"] == ["alpha"] for e in events)  # mid-run progress

    mgr.unsubscribe(queue)
    await mgr.mark_announced(job.job_id)  # persists a summary → would emit
    assert queue.empty()


async def test_status_transitions_observed_mid_stream(store, checkpointer, tmp_path):
    """Artifacts are persisted as capabilities finish, before the job ends."""
    seen: list[str] = []

    class ObservantStore:
        def __init__(self, inner):
            self.inner = inner

        async def aput(self, namespace, key, value):
            if namespace[-1] == "results":
                seen.append(f"result:{key}")
            if namespace == ("jobs", "index"):
                seen.append(f"status:{value['status']}")
            return await self.inner.aput(namespace, key, value)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    mgr = make_manager(ObservantStore(store), checkpointer, tmp_path)
    job = await mgr.create_job("q")
    await mgr.run_job(job.job_id)
    assert seen[0] == "status:queued"
    assert "status:running" in seen
    assert "result:alpha" in seen
    assert seen.index("status:running") < seen.index("result:alpha")
    assert seen[-1] == "status:done"


async def test_annexes_are_opt_in_and_rendered_by_the_capability(store, checkpointer, tmp_path):
    """With annexes on, each capability presents its own result (prose stays
    prose); structured payloads still get a JSON block."""

    class ProseCap(SlowEcho):
        async def work(self, state):
            return self._emit_success({
                "aspects": ["first angle", "second angle"],
                "notes": "## Heading\n\nA paragraph of real markdown.",
            })

        def render_context(self, result):
            return result["data"]["notes"]

    class StructuredCap(SlowEcho):
        async def work(self, state):
            return self._emit_success({"docs": [{"id": "d1", "score": 0.9}]})

        def render_context(self, result):
            return str(result["data"]["docs"])

    from jobsmith.core.registry import CapabilityRegistry
    from jobsmith.jobs.report import MarkdownReport

    caps = [ProseCap("prose"), StructuredCap("structured")]
    registry = CapabilityRegistry(caps)
    mgr = make_manager(store, checkpointer, tmp_path, caps=caps)
    mgr.reporter = MarkdownReport(registry, with_annexes=True)
    job = await mgr.create_job("render me")
    done = await mgr.run_job(job.job_id)

    report = (tmp_path / "artifacts" / f"{done.job_id}.md").read_text()
    assert "Step output — prose" in report                # annex, collapsible
    assert "A paragraph of real markdown." in report      # prose, not escaped
    assert "\\n" not in report                            # no JSON-escaped newlines
    assert "- first angle" in report                      # string list -> bullets
    assert '"score": 0.9' in report                       # structured -> json block


def test_mermaid_draws_isolated_steps_once():
    """A root that feeds another step is drawn by its edge; a step wired to
    nothing at all still needs its own line or it vanishes from the DAG."""
    from jobsmith.jobs.report import JobDocument, MarkdownReport, PlanRow

    doc = JobDocument(
        title="t", request="t", job_id="j", created_at="", finished_at="", answer="a",
        plan=[
            PlanRow("research", [], "ok", ""),
            PlanRow("analysis", ["research"], "ok", ""),
            PlanRow("aside", [], "ok", ""),
        ],
    )
    mermaid = MarkdownReport().render(doc).split("```mermaid")[1].split("```")[0]
    assert "research --> analysis" in mermaid
    assert mermaid.count("research") == 1      # not redrawn as a bare node
    assert "\n  aside\n" in mermaid            # isolated step still shown
