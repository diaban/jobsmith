"""JobManager: lifecycle, store schema, listing, cancellation."""
from __future__ import annotations

import asyncio

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END

from agent_oo.core.builder import build_agent
from agent_oo.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from agent_oo.core.deps import Deps
from agent_oo.core.registry import CapabilityRegistry
from agent_oo.jobs.manager import JobManager
from agent_oo.jobs.models import JobStatus


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
    artifact = await store.aget(("jobs", job.job_id, "artifacts"), "alpha")
    assert artifact.value["ok"] is True

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


async def test_report_written_on_done(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("alpha"), SlowEcho("beta")])
    job = await mgr.create_job("write the report")
    done = await mgr.run_job(job.job_id)

    assert done.report_path is not None
    report = (tmp_path / "artifacts" / f"{job.job_id}.md").read_text()
    assert "# Job report — write the report" in report
    assert "final answer" in report          # the answer section
    assert "| alpha |" in report             # plan table
    assert "flowchart LR" in report          # mermaid DAG
    assert '"echo": "beta"' in report        # artifact payload
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


async def test_status_transitions_observed_mid_stream(store, checkpointer, tmp_path):
    """Artifacts are persisted as capabilities finish, before the job ends."""
    seen: list[str] = []

    class ObservantStore:
        def __init__(self, inner):
            self.inner = inner

        async def aput(self, namespace, key, value):
            if namespace[-1] == "artifacts":
                seen.append(f"artifact:{key}")
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
    assert "artifact:alpha" in seen
    assert seen.index("status:running") < seen.index("artifact:alpha")
    assert seen[-1] == "status:done"
