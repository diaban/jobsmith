"""A step that writes a FILE: the `ArtifactStore` port, the declaration in its
result's `meta`, and the job's annexes — collected at every terminal, in
plan order, once, never disturbing which output is the report. → 0035, 0041
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeLLM, plan_json
from support import (
    SVG,
    ChartCapability,
    CountingEcho,
    make_manager,
    planning,
    until,
)

from jobsmith.core.artifacts import (
    ArtifactRef,
    ArtifactStore,
    LocalArtifactStore,
    artifact_meta,
    artifact_refs,
)
from jobsmith.core.capability import CapabilityBaseState
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.report import compose_reporters


class PhantomCapability(ChartCapability):
    """Declares a file it never wrote — a capability defect, on purpose."""

    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"chart": self.filename},
                                  meta=artifact_meta(ArtifactRef(self.filename)))


class HalfChartCapability(ChartCapability):
    """Writes its file, then fails: `_emit_failure` takes a `meta` too."""

    async def work(self, state: CapabilityBaseState) -> dict:
        path = await self.artifacts.write(state.get("job_id", ""), self.filename, SVG)
        return self._emit_failure("the export died after the file was written",
                                  meta=artifact_meta(ArtifactRef(path, title="Half a chart")))


def charts(tmp_path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


def drawing(store, checkpointer, tmp_path, *caps, deps=None, llm=None) -> JobManager:
    caps = caps or (ChartCapability(charts(tmp_path)),)
    return make_manager(store, checkpointer, tmp_path, caps=list(caps),
                        llm=llm or planning(*[c.spec.name for c in caps], deps=deps))


async def run(mgr: JobManager, query="draw me something", **create):
    return await mgr.run_job((await mgr.create_job(query, **create)).job_id)


# ------------------------------------------------------------ the port

async def test_the_store_writes_per_job_and_returns_the_path(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = await store.write("job1", "chart.svg", SVG)
    assert path == str(tmp_path / "job1" / "chart.svg") and Path(path).read_text() == SVG
    other = await store.write("job2", "chart.svg", b"\x89PNG")        # bytes, own folder
    assert Path(other).read_bytes() == b"\x89PNG" and Path(path).read_text() == SVG


async def test_a_capability_cannot_escape_its_job_directory(tmp_path):
    """`name` is a filename, not a path."""
    store = LocalArtifactStore(tmp_path / "root")
    assert await store.write("job1", "../../etc/passwd", "x") == str(
        tmp_path / "root" / "job1" / "passwd")
    for bad in ("", "..", "/"):
        with pytest.raises(ValueError):
            await store.write(bad, "chart.svg", "x")
        with pytest.raises(ValueError):
            await store.write("job1", bad, "x")


def test_declared_refs_are_read_back_leniently():
    """`meta` is written by code the framework does not control: junk is dropped."""
    assert artifact_refs(None) == []
    assert artifact_refs({"artifacts": "of/a/path.svg"}) == [ArtifactRef("of/a/path.svg")]
    assert artifact_refs({"artifacts": [{"path": "a.svg"}, {"title": "no path"}, 7, None]}) \
        == [ArtifactRef("a.svg")]
    assert artifact_refs({"artifacts": "  "}) == []
    assert ArtifactRef("x/y.png").file_format == "png"
    assert ArtifactRef("x/y.png", format="image").file_format == "image"
    with pytest.raises(ValueError):
        ArtifactRef("  ")


def test_a_failed_step_can_declare_the_file_it_wrote():
    emitted = ChartCapability(LocalArtifactStore("/nowhere"))._emit_failure(
        "boom", meta=artifact_meta(ArtifactRef("of/a/chart.svg")))
    result = emitted["results"]["chart"]
    assert result["ok"] is False and result["error"] == "boom"
    assert artifact_refs(result["meta"]) == [ArtifactRef("of/a/chart.svg", format="svg")]


# ------------------------------------------------------------ a finished run

async def test_a_file_a_step_produced_becomes_an_annex(store, checkpointer, tmp_path):
    chart = ChartCapability(charts(tmp_path))
    mgr = drawing(store, checkpointer, tmp_path, chart)
    done = await run(mgr, formats=["markdown"])

    assert chart.seen_job_id == done.job_id
    main, annex = done.outputs
    assert (main.role, main.format) == ("main", "markdown")
    assert (annex.role, annex.format, annex.produced_by, annex.title) == (
        "annex", "svg", "chart", "Revenue chart")
    assert annex.path == str(tmp_path / "artifacts" / done.job_id / "chart.svg")
    assert Path(annex.path).read_text() == SVG
    assert done.report_path == main.path

    fetched = await mgr.get_job(done.job_id)          # persisted with role + attribution
    assert [(o.role, o.produced_by) for o in fetched.outputs] == [
        ("main", None), ("annex", "chart")]


async def test_annexes_come_after_the_deliverables_and_there_is_one_main(
    store, checkpointer, tmp_path
):
    mgr = drawing(store, checkpointer, tmp_path)
    mgr.reporter = compose_reporters("markdown,html")
    done = await run(mgr, formats=["markdown"])
    assert [(o.format, o.role) for o in done.outputs] == [
        ("markdown", "main"), ("html", "alternate"), ("svg", "annex")]


async def test_a_declared_file_that_is_missing_is_dropped_and_said(store, checkpointer, tmp_path):
    phantom = PhantomCapability(charts(tmp_path), filename=str(tmp_path / "never-written.svg"),
                                name="phantom")
    mgr = drawing(store, checkpointer, tmp_path, phantom)
    done = await run(mgr, formats=["markdown"])

    assert done.status is JobStatus.DONE and [o.role for o in done.outputs] == ["main"]
    assert "never-written.svg" in done.error and "phantom" in done.error
    assert (await mgr.get_job(done.job_id)).error == done.error


async def test_annexes_survive_a_report_that_could_not_be_written(
    store, checkpointer, tmp_path
):
    class Boom:
        format, extension = "markdown", "md"

        def write(self, job, directory):
            raise OSError("No space left on device")

    mgr = drawing(store, checkpointer, tmp_path)
    mgr.reporter = Boom()
    done = await run(mgr, formats=["markdown"])

    [annex] = done.outputs
    assert annex.role == "annex" and Path(annex.path).is_file()
    assert done.report_path is None and "No space left on device" in done.error


async def test_files_are_listed_in_plan_order_and_never_twice(store, checkpointer, tmp_path):
    """Steps finish in arrival order; annexes follow the PLAN. A path declared
    twice is one output, and not a missing file."""
    first = ChartCapability(charts(tmp_path), filename="first.svg")
    second = ChartCapability(charts(tmp_path), filename="second.svg", name="chart_two")

    class Copycat(ChartCapability):
        async def work(self, state):
            path = str(tmp_path / "artifacts" / state.get("job_id", "") / "first.svg")
            return self._emit_success({"chart": path}, meta=artifact_meta(ArtifactRef(path)))

    mgr = drawing(store, checkpointer, tmp_path, second, first,
                  Copycat(charts(tmp_path), name="copycat"),
                  llm=planning("chart_two", "chart", "copycat",
                               deps={"chart_two": ["chart"], "copycat": ["chart_two"]}))
    done = await run(mgr, "two charts")

    assert list(done.results) == ["chart", "chart_two", "copycat"]      # arrival
    annexes = [o for o in done.outputs if o.role == "annex"]
    assert [(o.produced_by, Path(o.path).name) for o in annexes] == [
        ("chart_two", "second.svg"), ("chart", "first.svg")]           # plan, once
    assert done.error is None


async def test_the_composition_root_hands_a_capability_a_store(tmp_path):
    """A third-party agent gets the port from `AgentContext`, rooted where the
    manager keeps deliverables."""
    from jobsmith.agents import AGENTS
    from jobsmith.agents.base import AgentContext, AgentDefinition
    from jobsmith.app.agent import build_app
    from jobsmith.core.profile import AgentProfile

    seen: dict[str, ArtifactStore | None] = {}

    def capabilities(ctx: AgentContext):
        seen["store"] = ctx.artifacts
        assert ctx.artifacts is not None, "build_app must supply an artifact store"
        return [ChartCapability(ctx.artifacts)]

    AGENTS["drawer"] = AgentDefinition(name="drawer", description="draws things",
                                       capabilities=capabilities, profile=AgentProfile())
    try:
        app = await build_app(agent="drawer", llm=planning("chart"), chat_model=object(),
                              db="memory", reports_dir=str(tmp_path))
        try:
            done = await run(app.manager, formats=["markdown"])
        finally:
            await app.aclose()
    finally:
        del AGENTS["drawer"]

    assert isinstance(seen["store"], LocalArtifactStore)
    [_, annex] = done.outputs
    assert annex.path == str(tmp_path / done.job_id / "chart.svg") and Path(annex.path).is_file()


# ------------------------------------------------------------ a run that did not finish

async def test_a_job_that_failed_still_lists_the_files_its_steps_produced(
    store, checkpointer, tmp_path
):
    class ExplodingGenLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("half_chart")
            raise RuntimeError("llm down")

    half = HalfChartCapability(charts(tmp_path), name="half_chart")
    mgr = drawing(store, checkpointer, tmp_path, half, llm=ExplodingGenLLM())
    done = await run(mgr)

    assert done.status is JobStatus.FAILED
    [annex] = done.outputs
    assert (annex.role, annex.produced_by, annex.title) == ("annex", "half_chart", "Half a chart")
    assert done.report_path is None and not list((tmp_path / "artifacts").glob("*.md"))
    assert done.error and "chart" not in done.error          # why the JOB failed, only
    assert [o.path for o in (await mgr.get_job(done.job_id)).outputs] == [annex.path]


async def cancelled_after_the_chart(store, checkpointer, tmp_path, **create):
    """chart → slow, cancelled while `slow` runs: one file on disk, one step pending."""
    slow = CountingEcho("slow", delay=30.0)
    mgr = drawing(store, checkpointer, tmp_path, ChartCapability(charts(tmp_path)), slow,
                  deps={"slow": ["chart"]})
    job = await mgr.create_job("draw, then take forever", **create)
    mgr.start_job(job.job_id)
    await until(lambda: slow.runs, what="the second step starting")
    return mgr, job, slow, await mgr.cancel_job(job.job_id)


async def test_a_cancelled_job_lists_what_its_finished_steps_produced(
    store, checkpointer, tmp_path
):
    mgr, job, _, stopped = await cancelled_after_the_chart(store, checkpointer, tmp_path)
    assert stopped.status is JobStatus.CANCELLED and stopped.report_path is None
    [annex] = stopped.outputs
    assert (annex.role, annex.produced_by) == ("annex", "chart")
    assert [o.path for o in (await mgr.get_job(job.job_id)).outputs] == [annex.path]


async def test_a_resumed_job_lists_each_file_exactly_once(store, checkpointer, tmp_path):
    """`job.outputs` is assigned, never appended to."""
    mgr, job, slow, _ = await cancelled_after_the_chart(
        store, checkpointer, tmp_path, formats=["markdown"])
    slow.delay = 0.0
    resumed = await mgr.resume_job(job.job_id)

    assert resumed.status is JobStatus.DONE
    assert [o.role for o in resumed.outputs] == ["main", "annex"]
    assert resumed.report_path.endswith(".md")


async def test_a_run_that_blew_up_mid_stream_still_lists_what_landed(store, tmp_path):
    """The runner itself raised — a terminal no real graph produces on demand."""
    from jobsmith.jobs.runner import PlanReady, StepFinished

    chart = tmp_path / "artifacts" / "landed.svg"
    chart.parent.mkdir(parents=True)
    chart.write_text(SVG)

    class ExplodingRunner:
        async def stream(self, job_id, query, inputs, formats=None):
            yield PlanReady({"rationale": "r",
                             "steps": [{"capability": "chart", "depends_on": []}]})
            yield StepFinished("chart", {"ok": True, "data": {}, "meta": artifact_meta(
                ArtifactRef(str(chart)), ArtifactRef(str(chart.parent / "half.svg")))})
            raise RuntimeError("the graph blew up")

        async def pending(self, job_id):
            return ()

    mgr = JobManager(store=store, runner=ExplodingRunner(), reports_dir=tmp_path / "artifacts")
    done = await run(mgr)

    assert done.status is JobStatus.FAILED and done.report_path is None
    assert [(o.role, o.path) for o in done.outputs] == [("annex", str(chart))]
    # why the run stopped reads first; the promised file that is missing is still named
    assert done.error.startswith("the graph blew up; ")
    assert "chart → " in done.error and "half.svg" in done.error
