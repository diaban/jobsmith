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


def make_manager(store, checkpointer, *, caps=None, llm=None) -> JobManager:
    caps = caps if caps is not None else [SlowEcho("alpha")]
    llm = llm or FakeLLM(
        {"planner": plan_json(*[c.spec.name for c in caps])},
        default="A sufficiently long final answer for the job test.",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    return JobManager(graph, store)


async def test_create_run_done_with_store_contents(store, checkpointer):
    mgr = make_manager(store, checkpointer, caps=[SlowEcho("alpha"), SlowEcho("beta")])
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


async def test_failed_job_records_error_and_errors_meta(store, checkpointer):
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("alpha")
            raise RuntimeError("llm down")

    mgr = make_manager(store, checkpointer, llm=ExplodingGenLLM())
    job = await mgr.create_job("q")
    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.FAILED
    assert done.terminal_kind == "escalated"
    assert done.error is not None
    errors = await store.aget(("jobs", job.job_id, "meta"), "errors")
    assert any(e["kind"] == "generation_fail" for e in errors.value)


async def test_run_requires_queued(store, checkpointer):
    mgr = make_manager(store, checkpointer)
    job = await mgr.create_job("q")
    await mgr.run_job(job.job_id)
    with pytest.raises(ValueError, match="expected queued"):
        await mgr.run_job(job.job_id)
    with pytest.raises(KeyError):
        await mgr.run_job("nonexistent")


async def test_list_jobs_filtering(store, checkpointer):
    mgr = make_manager(store, checkpointer)
    j1 = await mgr.create_job("first")
    j2 = await mgr.create_job("second")
    await mgr.run_job(j1.job_id)

    all_jobs = await mgr.list_jobs()
    assert {j.job_id for j in all_jobs} == {j1.job_id, j2.job_id}
    queued = await mgr.list_jobs(status=JobStatus.QUEUED)
    assert [j.job_id for j in queued] == [j2.job_id]
    done = await mgr.list_jobs(status=JobStatus.DONE)
    assert [j.job_id for j in done] == [j1.job_id]


async def test_start_and_cancel_running_job(store, checkpointer):
    mgr = make_manager(store, checkpointer, caps=[SlowEcho("slow", delay=30.0)])
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


async def test_cancel_tombstone_without_task(store, checkpointer):
    mgr = make_manager(store, checkpointer)
    job = await mgr.create_job("q")
    cancelled = await mgr.cancel_job(job.job_id)  # never started
    assert cancelled.status is JobStatus.CANCELLED
    # cancelling a finished job is a no-op
    job2 = await mgr.create_job("q2")
    await mgr.run_job(job2.job_id)
    after = await mgr.cancel_job(job2.job_id)
    assert after.status is JobStatus.DONE


async def test_status_transitions_observed_mid_stream(store, checkpointer):
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

    mgr = make_manager(ObservantStore(store), checkpointer)
    job = await mgr.create_job("q")
    await mgr.run_job(job.job_id)
    assert seen[0] == "status:queued"
    assert "status:running" in seen
    assert "artifact:alpha" in seen
    assert seen.index("status:running") < seen.index("artifact:alpha")
    assert seen[-1] == "status:done"
