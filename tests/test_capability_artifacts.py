"""A capability that produces a FILE, and the job that records it.

`JobOutput` has carried `role="annex"` and `produced_by` since it was
written, and nothing could reach them: the manager's only assignment to
`Job.outputs` was what the Reporter handed back. These tests pin the seam
that closes that gap — a capability writes through the `ArtifactStore` port,
declares what it wrote in its result's `meta`, and the manager turns the
declarations into annexes without disturbing which output is *the* report.

The capability lives here, on purpose: exercising the mechanism must not mean
reshaping the default agent's pack.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.core.artifacts import (
    ArtifactRef,
    ArtifactStore,
    LocalArtifactStore,
    artifact_meta,
    artifact_refs,
)
from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import JobStatus

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'


class ChartCapability(Capability):
    """A step that draws something and hands the file back.

    It knows a store and a filename — never a directory, never a job's
    layout. `job_id` comes from the state the executor handed it.
    """

    spec = CapabilitySpec(name="chart", description="draws a chart of the request")

    def __init__(self, artifacts: ArtifactStore, *, filename: str = "chart.svg"):
        self.artifacts = artifacts
        self.filename = filename
        self.seen_job_id: str | None = None

    async def draw(self, state: CapabilityBaseState) -> dict:
        # `.get()`: a graph driven outside a job has no job_id (see the note
        # on CapabilityBaseState) — writing then is the capability's call.
        self.seen_job_id = state.get("job_id", "")
        path = await self.artifacts.write(self.seen_job_id, self.filename, SVG)
        return self._emit_success(
            {"chart": path},
            meta=artifact_meta(ArtifactRef(path, title="Revenue chart")),
        )

    def render_context(self, result):
        return "a chart was drawn"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("draw", self.draw)
        g.set_entry_point("draw")
        g.add_edge("draw", END)
        return g.compile()


class PhantomCapability(ChartCapability):
    """Declares a file it did not write — a capability defect, on purpose."""

    spec = CapabilitySpec(name="phantom", description="claims a file it never wrote")

    async def draw(self, state: CapabilityBaseState) -> dict:
        missing = str(Path(self.filename))
        return self._emit_success({"chart": missing},
                                  meta=artifact_meta(ArtifactRef(missing)))


class PlainCapability(Capability):
    """A step that produces no file at all — the ordinary case."""

    spec = CapabilitySpec(name="plain", description="produces prose only")

    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"text": "some prose"})

    def render_context(self, result):
        return result["data"]["text"]

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


def make_manager(store, checkpointer, tmp_path, caps, **kwargs) -> JobManager:
    llm = FakeLLM(
        {"planner": plan_json(*[c.spec.name for c in caps])},
        default="A sufficiently long final answer for the artifact test.",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts", **kwargs)


# ---------------------------------------------------------------- the store

async def test_the_store_writes_per_job_and_returns_the_path(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = await store.write("job1", "chart.svg", SVG)
    assert path == str(tmp_path / "job1" / "chart.svg")
    assert Path(path).read_text() == SVG
    # bytes too, and a second job never lands in the first one's directory
    other = await store.write("job2", "chart.svg", b"\x89PNG")
    assert Path(other).read_bytes() == b"\x89PNG"
    assert Path(path).read_text() == SVG


async def test_a_capability_cannot_escape_its_job_directory(tmp_path):
    """`name` is a filename, not a path: the capability does not know the
    layout and must not be able to reach outside it."""
    store = LocalArtifactStore(tmp_path / "root")
    path = await store.write("job1", "../../etc/passwd", "x")
    assert path == str(tmp_path / "root" / "job1" / "passwd")
    for bad in ("", "..", "/"):
        with pytest.raises(ValueError):
            await store.write(bad, "chart.svg", "x")
        with pytest.raises(ValueError):
            await store.write("job1", bad, "x")


def test_declared_refs_are_read_back_leniently():
    """`meta` is written by code the framework does not control, so junk is
    dropped — a malformed entry must not turn an answered job into a failed
    report."""
    assert artifact_refs(None) == []
    assert artifact_refs({"artifacts": "of/a/path.svg"}) == [ArtifactRef("of/a/path.svg")]
    assert artifact_refs({"artifacts": [{"path": "a.svg"}, {"title": "no path"}, 7, None]}) \
        == [ArtifactRef("a.svg")]
    assert artifact_refs({"artifacts": "  "}) == []
    assert ArtifactRef("x/y.png").file_format == "png"      # extension when unsaid
    assert ArtifactRef("x/y.png", format="image").file_format == "image"
    with pytest.raises(ValueError):
        ArtifactRef("  ")


# ---------------------------------------------------------------- the seam

async def test_a_file_a_step_produced_becomes_an_annex(store, checkpointer, tmp_path):
    """The whole point: the capability wrote a file, and the job records it —
    as an annex, attributed to the step, next to the deliverable."""
    chart = ChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    mgr = make_manager(store, checkpointer, tmp_path, [chart])
    job = await mgr.create_job("draw me something")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE
    assert chart.seen_job_id == done.job_id          # the job reached the capability
    main, annex = done.outputs
    assert (main.role, main.format) == ("main", "markdown")
    assert (annex.role, annex.format, annex.produced_by) == ("annex", "svg", "chart")
    assert annex.title == "Revenue chart"
    assert Path(annex.path).read_text() == SVG
    assert annex.path == str(tmp_path / "artifacts" / done.job_id / "chart.svg")
    # the report still points at the report
    assert done.report_path == main.path and done.report_path.endswith(".md")

    # persisted, and reloaded with role + attribution intact
    summary = (await store.aget(("jobs", "index"), job.job_id)).value
    assert [o["role"] for o in summary["outputs"]] == ["main", "annex"]
    fetched = await mgr.get_job(job.job_id)
    assert fetched.outputs[1].produced_by == "chart"
    assert fetched.report_path == done.report_path


async def test_a_job_without_artifacts_hands_back_exactly_what_it_did_before(
    store, checkpointer, tmp_path
):
    """The single-Reporter path is untouched: no declaration, no annex."""
    mgr = make_manager(store, checkpointer, tmp_path, [PlainCapability()])
    done = await mgr.run_job((await mgr.create_job("no files here")).job_id)

    assert [(o.role, o.format) for o in done.outputs] == [("main", "markdown")]
    assert done.error is None
    assert sorted(p.name for p in (tmp_path / "artifacts").iterdir()) \
        == [f"{done.job_id}.md"]


async def test_a_missing_file_is_dropped_and_said_out_loud(store, checkpointer, tmp_path):
    """Recording a JobOutput for a file that is not there would offer a
    deliverable nobody can open (#28). Staying silent would hide a capability
    defect behind a perfect-looking run — so it is dropped AND reported."""
    phantom = PhantomCapability(LocalArtifactStore(tmp_path / "artifacts"),
                                filename=str(tmp_path / "never-written.svg"))
    mgr = make_manager(store, checkpointer, tmp_path, [phantom])
    done = await mgr.run_job((await mgr.create_job("promise me a file")).job_id)

    assert done.status is JobStatus.DONE             # the answer is still the work
    assert [o.role for o in done.outputs] == ["main"]
    assert "never-written.svg" in done.error and "phantom" in done.error
    assert (await mgr.get_job(done.job_id)).error == done.error   # persisted


async def test_annexes_survive_a_report_that_could_not_be_written(
    store, checkpointer, tmp_path
):
    """The chart is on disk whatever the reporter did. Dropping it because
    the report failed would leave a file no caller can find."""

    class Boom:
        format, extension = "markdown", "md"

        def write(self, job, directory):
            raise OSError("No space left on device")

    chart = ChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    mgr = make_manager(store, checkpointer, tmp_path, [chart], reporter=Boom())
    done = await mgr.run_job((await mgr.create_job("draw me something")).job_id)

    assert done.status is JobStatus.DONE
    [annex] = done.outputs
    assert annex.role == "annex" and Path(annex.path).is_file()
    assert done.report_path is None                  # there is no main one
    assert "No space left on device" in done.error


async def test_annexes_do_not_disturb_which_output_is_the_report(
    store, checkpointer, tmp_path
):
    """#28's invariant with a step's file in the mix: exactly one `main`, the
    first format asked for, and the annexes after the deliverables."""
    from jobsmith.jobs.report import compose_reporters

    chart = ChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    mgr = make_manager(store, checkpointer, tmp_path, [chart],
                       reporter=compose_reporters("markdown,html"))
    done = await mgr.run_job((await mgr.create_job("draw me something")).job_id)

    assert [(o.format, o.role) for o in done.outputs] == [
        ("markdown", "main"), ("html", "alternate"), ("svg", "annex")]
    assert sum(o.role == "main" for o in done.outputs) == 1
    assert done.report_path.endswith(".md")


async def test_files_are_listed_in_plan_order_and_never_twice(
    store, checkpointer, tmp_path
):
    """Waves finish in arrival order, reports must not: the annexes follow the
    PLAN, which is why the plan below lists its steps in the reverse of the
    order they can run in. And one path recorded twice would list one file as
    two outputs."""
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    first = ChartCapability(artifacts, filename="first.svg")
    second = ChartCapability(artifacts, filename="second.svg")
    second.spec = CapabilitySpec(name="chart_two", description="draws another chart")

    class Copycat(ChartCapability):
        """Declares the file another step already declared."""
        spec = CapabilitySpec(name="copycat", description="declares a known file")

        async def draw(self, state):
            path = str(tmp_path / "artifacts" / state.get("job_id", "") / "first.svg")
            return self._emit_success({"chart": path},
                                      meta=artifact_meta(ArtifactRef(path)))

    caps = [second, first, Copycat(artifacts)]       # registry order ≠ plan order
    llm = FakeLLM(
        {"planner": plan_json("chart_two", "chart", "copycat",
                              deps={"chart_two": ["chart"], "copycat": ["chart_two"]})},
        default="A sufficiently long final answer for the artifact test.",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    mgr = JobManager(graph, store, reports_dir=tmp_path / "artifacts")
    done = await mgr.run_job((await mgr.create_job("two charts")).job_id)

    # runs chart → chart_two → copycat, but the plan says the other way round
    assert list(done.results) == ["chart", "chart_two", "copycat"]      # arrival
    annexes = [o for o in done.outputs if o.role == "annex"]
    assert [o.produced_by for o in annexes] == ["chart_two", "chart"]   # plan
    assert [Path(o.path).name for o in annexes] == ["second.svg", "first.svg"]
    # copycat's re-declaration of first.svg is recorded once, attributed to the
    # step that declared it first — and a duplicate is not a missing file
    assert done.error is None


async def test_the_composition_root_hands_a_capability_a_store(tmp_path):
    """A third-party agent gets the port from `AgentContext`, rooted where the
    manager keeps deliverables — no shared code, no knowledge of the layout."""
    from jobsmith.agents import AGENTS
    from jobsmith.agents.base import AgentContext, AgentDefinition
    from jobsmith.app.agent import build_app
    from jobsmith.core.profile import AgentProfile

    seen: dict[str, ArtifactStore | None] = {}

    def capabilities(ctx: AgentContext):
        seen["store"] = ctx.artifacts
        assert ctx.artifacts is not None, "build_app must supply an artifact store"
        return [ChartCapability(ctx.artifacts)]

    mine = AgentDefinition(name="drawer", description="draws things",
                           capabilities=capabilities, profile=AgentProfile())
    AGENTS[mine.name] = mine
    try:
        llm = FakeLLM({"planner": plan_json("chart")},
                      default="A sufficiently long final answer for this run.")
        app = await build_app(agent="drawer", llm=llm, chat_model=object(),
                              reports_dir=str(tmp_path))
        try:
            done = await app.manager.run_job(
                (await app.manager.create_job("draw me something")).job_id)
        finally:
            await app.aclose()
    finally:
        del AGENTS[mine.name]

    assert isinstance(seen["store"], LocalArtifactStore)
    [_, annex] = done.outputs
    assert annex.path == str(tmp_path / done.job_id / "chart.svg")
    assert Path(annex.path).is_file()


# ------------------------------------------------- a run that did not finish

class HalfChartCapability(ChartCapability):
    """Writes its file, then fails — the case `_emit_failure` had no channel for.

    The chart is on disk exactly as if the step had succeeded; only what came
    after it went wrong. `meta=` is what lets it say so.
    """

    spec = CapabilitySpec(name="half_chart", description="draws a chart, then breaks")

    async def draw(self, state: CapabilityBaseState) -> dict:
        self.seen_job_id = state.get("job_id", "")
        path = await self.artifacts.write(self.seen_job_id, self.filename, SVG)
        return self._emit_failure(
            "the export died after the file was written",
            meta=artifact_meta(ArtifactRef(path, title="Half a chart")),
        )


class SlowCapability(Capability):
    """A step that hangs, so a run can be cancelled while it is inside one."""

    spec = CapabilitySpec(name="slow", description="takes its time")

    def __init__(self, delay: float = 30.0):
        self.delay = delay
        self.runs = 0

    async def work(self, state: CapabilityBaseState) -> dict:
        self.runs += 1
        await asyncio.sleep(self.delay)
        return self._emit_success({"echo": "slow"})

    def render_context(self, result):
        return "the slow step finished"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


def make_two_step_manager(store, checkpointer, tmp_path, caps, deps):
    """A manager whose plan runs `caps` in order, with `deps` between them."""
    llm = FakeLLM(
        {"planner": plan_json(*[c.spec.name for c in caps], deps=deps)},
        default="A sufficiently long final answer for the artifact test.",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts")


def test_a_failed_step_can_declare_the_file_it_wrote():
    """The first gate: `_emit_failure` takes a `meta`, like `_emit_success`.

    Without it a capability that wrote a chart and then hit an error has no
    way to say so, and the file is orphaned at birth.
    """
    cap = ChartCapability(LocalArtifactStore("/nowhere"))
    emitted = cap._emit_failure("boom", meta=artifact_meta(ArtifactRef("of/a/chart.svg")))
    result = emitted["results"]["chart"]

    assert result["ok"] is False and result["error"] == "boom"
    assert artifact_refs(result["meta"]) == [ArtifactRef("of/a/chart.svg", format="svg")]


async def test_a_job_that_failed_still_lists_the_files_its_steps_produced(
    store, checkpointer, tmp_path
):
    """The second gate: declarations were only ever read for a DONE job.

    The run never reached an answer, so there is no report — but the file is
    on disk, and a file recorded nowhere is a file nobody can find.
    """
    class ExplodingGenLLM(FakeLLM):
        """Generation cannot answer — the run ends escalated, not DONE."""

        async def chat(self, messages, **kwargs):
            if "planner" in self._system_of(messages):
                return plan_json("half_chart")
            raise RuntimeError("llm down")

    half = HalfChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    graph = build_agent(Deps(llm=ExplodingGenLLM()), CapabilityRegistry([half]),
                        checkpointer=checkpointer)
    mgr = JobManager(graph, store, reports_dir=tmp_path / "artifacts")
    done = await mgr.run_job((await mgr.create_job("draw me something")).job_id)

    assert done.status is JobStatus.FAILED and done.terminal_kind != "answer"
    [annex] = done.outputs
    assert (annex.role, annex.produced_by, annex.title) == ("annex", "half_chart",
                                                            "Half a chart")
    assert Path(annex.path).read_text() == SVG
    # no answer means no report: no main output, and the Reporter never ran
    assert done.report_path is None
    assert not list((tmp_path / "artifacts").glob("*.md"))
    # and the failure message still says why the job failed, nothing else
    assert done.error and "chart" not in done.error
    # persisted, so `jobsmith outputs` and GET /jobs/{id}/outputs find it
    fetched = await mgr.get_job(done.job_id)
    assert [(o.role, o.path) for o in fetched.outputs] == [("annex", annex.path)]


async def test_a_cancelled_job_lists_what_its_finished_steps_produced(
    store, checkpointer, tmp_path
):
    """A cancellation lands wherever the run happened to be; the steps that
    did finish still wrote their files."""
    chart = ChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    slow = SlowCapability()
    mgr = make_two_step_manager(store, checkpointer, tmp_path, [chart, slow],
                                {"slow": ["chart"]})
    job = await mgr.create_job("draw, then take forever")
    mgr.start_job(job.job_id)
    for _ in range(500):                     # wait until `slow` is actually running
        await asyncio.sleep(0.01)
        if slow.runs:
            break
    assert slow.runs == 1, "the second step never started"

    stopped = await mgr.cancel_job(job.job_id)
    assert stopped.status is JobStatus.CANCELLED
    [annex] = stopped.outputs
    assert (annex.role, annex.produced_by) == ("annex", "chart")
    assert Path(annex.path).read_text() == SVG
    assert stopped.report_path is None
    assert [o.path for o in (await mgr.get_job(job.job_id)).outputs] == [annex.path]


async def test_a_resumed_job_lists_each_file_exactly_once(store, checkpointer, tmp_path):
    """`job.outputs` is assigned, never appended to: a job that collected when
    it was cancelled must not list the same file twice when it finishes."""
    chart = ChartCapability(LocalArtifactStore(tmp_path / "artifacts"))
    slow = SlowCapability()
    mgr = make_two_step_manager(store, checkpointer, tmp_path, [chart, slow],
                                {"slow": ["chart"]})
    job = await mgr.create_job("draw, then take forever")
    mgr.start_job(job.job_id)
    for _ in range(500):
        await asyncio.sleep(0.01)
        if slow.runs:
            break
    stopped = await mgr.cancel_job(job.job_id)
    assert [o.path for o in stopped.outputs] == [
        str(tmp_path / "artifacts" / job.job_id / "chart.svg")]

    slow.delay = 0.0                          # let the interrupted step finish now
    resumed = await mgr.resume_job(job.job_id)

    assert resumed.status is JobStatus.DONE
    assert [o.role for o in resumed.outputs] == ["main", "annex"]
    assert [o.path for o in resumed.outputs].count(
        str(tmp_path / "artifacts" / job.job_id / "chart.svg")) == 1
    assert resumed.report_path.endswith(".md")


async def test_a_run_that_blew_up_mid_stream_still_lists_what_landed(store, tmp_path):
    """The third terminal in `_drive`: the run itself raised.

    Driven through the runner port (no graph needed) because that is the one
    terminal a real graph will not produce on demand — a node that raises is
    routed to `escalate` instead. The file its finished step wrote is on disk
    all the same.
    """
    from jobsmith.jobs.runner import PlanReady, StepFinished

    chart = tmp_path / "artifacts" / "landed.svg"
    chart.parent.mkdir(parents=True)
    chart.write_text(SVG)

    class ExplodingRunner:
        async def stream(self, job_id, query, inputs):
            yield PlanReady({"rationale": "r",
                             "steps": [{"capability": "chart", "depends_on": []}]})
            # one file that landed, one that did not — a run killed mid-write
            yield StepFinished("chart", {"ok": True, "data": {},
                                         "meta": artifact_meta(
                                             ArtifactRef(str(chart)),
                                             ArtifactRef(str(chart.parent / "half.svg")))})
            raise RuntimeError("the graph blew up")

        async def pending(self, job_id):
            return ()

    mgr = JobManager(store=store, runner=ExplodingRunner(),
                     reports_dir=tmp_path / "artifacts")
    done = await mgr.run_job((await mgr.create_job("draw me something")).job_id)

    assert done.status is JobStatus.FAILED
    # `job.error` still says why the run stopped, and ONLY that: on a run that
    # did not answer, a declared file that is not there is a consequence of
    # stopping, not the capability defect it would be on a job that answered.
    assert done.error == "the graph blew up"
    assert [(o.role, o.path) for o in done.outputs] == [("annex", str(chart))]
    assert done.report_path is None
