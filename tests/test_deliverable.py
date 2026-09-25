"""What a job delivers: whether a file, called what, in which formats — and
why there is none when there is none.

`Job.formats` holds three facts: `None` (said nothing — no file, still
fillable by the document step), `[]` (no file, never overridden), a list
(those files, the first is `main`). → 0055, 0096, 0028
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import FakeLLM, plan_json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import END
from support import (
    ANSWER,
    CFG,
    SlowEcho,
    client_for,
    direct_llm,
    files_under,
    launch_call,
    make_app,
    make_manager,
    make_session,
    no_libraries,
    wait_done,
)

from jobsmith.chat import ChatRunner, JobStarted
from jobsmith.core.artifacts import ArtifactRef, LocalArtifactStore, artifact_meta
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.report import (
    NAME_MAX,
    available_formats,
    compose_reporters,
    deliverable_filenames,
    document_stem,
)
from jobsmith.jobs.repository import StoreJobRepository

NAME = "chair_comparison"


def document_step(reply: dict, **script) -> FakeLLM:
    return FakeLLM({"document step": json.dumps(reply), "planner": plan_json("alpha"),
                    **script}, default=ANSWER)


def with_document_step(store, checkpointer, tmp_path, llm):
    return make_manager(store, checkpointer, tmp_path, llm=llm,
                        document_formats=("html", "markdown", "pdf"))


async def run(mgr, query="compare the chairs", **create):
    return await mgr.run_job((await mgr.create_job(query, **create)).job_id)


# ======================================================== whether there is a file

async def test_no_file_unless_asked_whatever_the_run_did(store, checkpointer, tmp_path):
    """A greeting, a full plan, an emptied plan: silence is no file, and no error."""

    class NeedsAnImage(SlowEcho):
        def __init__(self):
            super().__init__("vision")
            self.spec = CapabilitySpec(name="vision", description="reads an image",
                                       requires_inputs=("image_s3_keys",))

    greeting = make_manager(store, checkpointer, tmp_path, llm=direct_llm())
    planned = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("a"), SlowEcho("b")])
    emptied = make_manager(store, checkpointer, tmp_path, caps=[NeedsAnImage()], llm=FakeLLM(
        {"triage step": '{"route": "plan", "rationale": "r"}', "planner": plan_json("vision")},
        default=ANSWER))

    for done in (await run(greeting, "bonjour"), await run(planned), await run(emptied)):
        assert done.status is JobStatus.DONE and done.final_answer == ANSWER
        assert done.formats is None and done.deliverable_expected is False
        assert done.outputs == [] and done.report_path is None and done.error is None
    assert files_under(tmp_path) == []


@pytest.mark.parametrize("route", ["direct", "plan"])
async def test_formats_asked_for_are_written_whatever_the_route(
    store, checkpointer, tmp_path, route
):
    llm = direct_llm() if route == "direct" else None
    mgr = make_manager(store, checkpointer, tmp_path, llm=llm)
    done = await run(mgr, formats=["markdown"])

    assert done.deliverable_expected is True
    assert files_under(tmp_path) == [f"{done.job_id}.md"]


async def test_no_file_asked_for_is_obeyed_and_known_before_the_run(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("a"), SlowEcho("b")])
    job = await mgr.create_job("compare the chairs", formats=[])
    assert job.deliverable_expected is False and job.formats == []

    done = await mgr.run_job(job.job_id)
    assert set(done.results) == {"a", "b"}
    assert done.outputs == [] and done.error is None and files_under(tmp_path) == []


async def test_silence_and_no_file_stay_two_facts_through_the_store(
    store, checkpointer, tmp_path
):
    """`None` is fillable, `[]` is final: never `formats or []`. → 0096"""
    mgr = make_manager(store, checkpointer, tmp_path)
    silent = await mgr.create_job("compare the chairs")
    refused = await mgr.create_job("compare the chairs", formats=[])

    assert (await mgr.get_job(silent.job_id)).formats is None
    assert (await mgr.get_job(refused.job_id)).formats == []


async def test_a_record_older_than_the_field_reads_as_unstated(store):
    """Before #84 an empty `formats` meant "the deployment decides", and such a
    record has no `deliverable_expected` key — that is how it is recognised."""
    await store.aput(("jobs", "index"), "old1", {
        "status": "cancelled", "query": "q", "formats": [],
        "created_at": "2026-01-01T00:00:00+00:00"})
    job = await StoreJobRepository(store).load("old1")
    assert job is not None and job.formats is None and job.deliverable_expected is True


# ------------------------------------------------ decided by the document step

@pytest.mark.parametrize(("reply", "formats", "suffix"), ids=["named", "requested", "none", "unspecified"], argvalues=[
    ({"document": "named", "formats": ["html"]}, ["html"], ".html"),
    ({"document": "requested"}, ["markdown"], ".md"),    # the deployment's format
    ({"document": "none"}, [], None),
    ({"document": "unspecified"}, None, None),
])
async def test_the_request_s_own_words_decide_the_file(
    store, checkpointer, tmp_path, reply, formats, suffix
):
    """No caller spoke — `jobsmith run`, `/bg`, `POST /jobs` — and the sentence decides."""
    mgr = with_document_step(store, checkpointer, tmp_path, document_step(reply))
    done = await run(mgr, "compare X and Y")

    assert done.status is JobStatus.DONE and done.formats == formats
    assert done.deliverable_expected is (suffix is not None) and done.error is None
    assert done.report_path is None if suffix is None else done.report_path.endswith(suffix)
    index = await store.aget(("jobs", "index"), done.job_id)
    assert index.value["formats"] == formats


async def test_a_caller_who_named_formats_is_not_second_guessed(store, checkpointer, tmp_path):
    llm = document_step({"document": "named", "formats": ["html"]})
    mgr = with_document_step(store, checkpointer, tmp_path, llm)
    done = await run(mgr, "as an html page", formats=["markdown"])

    assert done.report_path is not None and done.report_path.endswith(".md")
    assert not [c for c in llm.calls if "document step" in c["messages"][0]["content"]]


@pytest.mark.parametrize("reply", [{"document": "none"}, {"document": "unspecified"}])
async def test_no_file_is_recorded_when_decided_even_if_the_run_then_fails(
    store, checkpointer, tmp_path, reply
):
    mgr = with_document_step(store, checkpointer, tmp_path,
                             document_step(reply, planner="not json at all"))
    done = await run(mgr, "compare X and Y")

    assert done.status is JobStatus.FAILED and done.deliverable_expected is False
    index = await store.aget(("jobs", "index"), done.job_id)
    assert index.value["deliverable_expected"] is False


# ======================================================== its name and title

async def test_a_named_deliverable_lands_in_the_job_s_folder(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    named = await run(mgr, document_name=NAME, formats=["markdown"])
    again = await run(mgr, "compare them again", document_name=NAME, formats=["markdown"])
    unnamed = await run(mgr, formats=["markdown"])

    assert named.report_path == str(tmp_path / "artifacts" / named.job_id / f"{NAME}.md")
    assert [o.name for o in named.outputs] == [f"{NAME}.md"]
    assert again.report_path != named.report_path          # a name is not unique
    assert unnamed.report_path == str(tmp_path / "artifacts" / f"{unnamed.job_id}.md")


@pytest.mark.parametrize("refused", [
    "../escape", "sub/dir/report", "/etc/passwd", ".", "x" * (NAME_MAX + 1)])
async def test_a_name_that_is_not_a_filename_is_refused_before_a_job_exists(
    store, checkpointer, tmp_path, refused
):
    mgr = make_manager(store, checkpointer, tmp_path)
    with pytest.raises(ValueError):          # PathRefused is one (→ 0060)
        await mgr.create_job("compare the chairs", document_name=refused)
    assert await mgr.list_jobs() == []


def test_a_known_extension_is_dropped_and_an_unknown_suffix_is_not():
    assert document_stem("rapport.md") == document_stem("rapport.pdf") == "rapport"
    assert document_stem("notes v1.2") == "notes v1.2"
    assert document_stem("  rapport  ") == "rapport"


@pytest.mark.parametrize(("name", "formats", "shown"), [
    (NAME, ["markdown", "pdf"], [f"{NAME}.md", f"{NAME}.pdf"]),
    ("", ["markdown"], []),              # no name yet
    (NAME, None, []),                    # no file: never a guessed filename
    (NAME, [], []),
])
def test_the_filenames_shown_are_the_filenames_written(name, formats, shown):
    assert deliverable_filenames(name, formats) == shown


async def test_the_heading_is_the_title_asked_for_else_one_derived_from_the_request(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    titled = await run(mgr, document_name=NAME, document_title="Comparatif des chaises",
                       formats=["markdown"])
    derived = await run(mgr, "compare the chairs for a home office", formats=["markdown"])

    assert open(titled.report_path).readline() == "# Comparatif des chaises\n"
    assert open(derived.report_path).readline().startswith(
        "# compare the chairs for a home office")


# ======================================================== its formats

async def test_each_format_asked_for_is_written_and_the_first_is_main(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    done = await run(mgr, document_name=NAME, formats=["html", "markdown"])

    expected = [("main", "html", f"{NAME}.html"), ("alternate", "markdown", f"{NAME}.md")]
    assert [(o.role, o.format, o.name) for o in done.outputs] == expected
    reloaded = await mgr.get_job(done.job_id)
    assert [(o.role, o.format, o.name) for o in reloaded.outputs] == expected
    assert reloaded.report_path == done.report_path and done.report_path.endswith(".html")


async def test_default_is_resolved_to_the_deployment_s_names_before_the_record(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    mgr.default_formats = ["html", "markdown"]
    job = await mgr.create_job("compare the chairs", formats=["default"])
    assert job.formats == ["html", "markdown"]

    mgr.default_formats = []
    with pytest.raises(ValueError, match="default"):
        await mgr.create_job("compare the chairs", formats=["default"])


async def test_a_job_that_wants_no_file_never_composes_a_reporter(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    mgr.reporter_factory = lambda formats: pytest.fail("should not be consulted")
    await run(mgr)
    with pytest.raises(ValueError, match="no report format"):
        compose_reporters([])


async def test_a_reporter_that_cannot_be_composed_is_a_failed_write(
    store, checkpointer, tmp_path
):
    """DONE, answer kept, cause in `job.error` — never a job left RUNNING. → 0028"""
    mgr = make_manager(store, checkpointer, tmp_path)

    def broken(formats):
        raise RuntimeError("renderer unavailable")

    mgr.reporter_factory = broken
    done = await run(mgr, formats=["markdown"])

    assert done.status is JobStatus.DONE and done.final_answer
    assert "renderer unavailable" in (done.error or "")
    assert (await mgr.get_job(done.job_id)).status is JobStatus.DONE


def test_markdown_is_always_offered():
    assert "markdown" in available_formats()


# ======================================================== why there is none

async def test_the_three_reasons_there_is_no_report_are_told_apart(
    store, checkpointer, tmp_path
):
    """Nothing asked, the write failed, the run never answered — each on its
    own field (`deliverable_expected`, `error`, `status`), never prose."""

    class Boom:
        format, extension, binary = "markdown", "md", False

        def write(self, job, directory):
            raise OSError("No space left on device")

    none_wanted = await run(make_manager(store, checkpointer, tmp_path, llm=direct_llm()),
                            "bonjour")
    broken = make_manager(store, checkpointer, tmp_path)
    broken.reporter = Boom()
    write_failed = await run(broken, formats=["markdown"])
    stopped = await run(make_manager(store, checkpointer, tmp_path), "   ")

    assert all(j.report_path is None for j in (none_wanted, write_failed, stopped))
    assert (none_wanted.status, none_wanted.deliverable_expected, none_wanted.error) == (
        JobStatus.DONE, False, None)
    assert (write_failed.status, write_failed.deliverable_expected) == (JobStatus.DONE, True)
    assert "markdown" in (write_failed.error or "")
    assert stopped.status is JobStatus.FAILED and stopped.deliverable_expected is True


async def test_no_document_still_lists_the_files_its_steps_wrote(
    store, checkpointer, tmp_path
):
    """"No deliverable" is not "no outputs": an annex is kept, and never `main`. → 0041"""

    class WritesAFile(Capability):
        spec = CapabilitySpec(name="charts", description="draws a chart")

        def __init__(self, artifacts):
            self.artifacts = artifacts

        async def work(self, state: CapabilityBaseState) -> dict:
            path = await self.artifacts.write(state.get("job_id", ""), "chart.svg", b"<svg/>")
            return self._emit_success({"drawn": True},
                                      meta=artifact_meta(ArtifactRef(path, title="Chart")))

        def build(self):
            g = self.state_graph(CapabilityBaseState)
            g.add_node("work", self.work)
            g.set_entry_point("work")
            g.add_edge("work", END)
            return g.compile()

    mgr = make_manager(store, checkpointer, tmp_path,
                       caps=[WritesAFile(LocalArtifactStore(tmp_path / "artifacts"))])
    done = await run(mgr, "draw me one", formats=[])

    assert done.deliverable_expected is False and done.report_path is None
    [annex] = done.outputs
    assert (annex.role, annex.produced_by, annex.name) == ("annex", "charts", "chart.svg")


async def test_every_surface_says_no_file_rather_than_not_yet(store, checkpointer, tmp_path):
    from jobsmith.cli.repl import job_lines, no_document_note
    from jobsmith.tui.render import outputs_block

    note = no_document_note({"job_id": "abcdef1234"})
    assert "none was asked for" in note and "abcdef12" in note
    assert "no file" in outputs_block([], expected=False)
    assert "no file yet" in outputs_block([], expected=True)
    assert any("no file" in line for line in job_lines({"query": "q", "formats": []}))
    assert any("writes" in line and "no file" in line for line in job_lines({"query": "q"}))
    assert any("markdown" in line for line in job_lines({"query": "q", "formats": ["markdown"]}))

    app, _ = make_app(store, checkpointer, tmp_path, [])
    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "just answer me", "formats": []})).json()["job_id"])
        refused = await client.get(f"/jobs/{job['job_id']}/report")
        assert refused.status_code == 404
        assert "none was asked for" in refused.json()["detail"]


async def test_a_silent_run_that_failed_reads_as_failed_not_as_unwanted(
    store, checkpointer, tmp_path, capsys
):
    """Both facts hold; every surface leads with the failure."""
    from jobsmith.cli.main import cmd_report
    from jobsmith.service import LocalAgentService

    mgr = make_manager(store, checkpointer, tmp_path,
                       llm=FakeLLM({"planner": "not json at all"}, default=ANSWER))
    failed = await run(mgr, "compare them")
    assert failed.status is JobStatus.FAILED and failed.deliverable_expected is False

    service = LocalAgentService(mgr, lambda session_id=None: None)
    assert await cmd_report(service, SimpleNamespace(job_id=failed.job_id[:8])) == 1
    printed = capsys.readouterr().out
    assert "none was asked for" not in printed and "is the job done" in printed

    app, api_mgr = make_app(store, checkpointer, tmp_path, [])
    api_mgr.runner = mgr.runner
    async with client_for(app) as client:
        job = await wait_done(client, (await client.post(
            "/jobs", json={"query": "compare them"})).json()["job_id"])
        detail = (await client.get(f"/jobs/{job['job_id']}/report")).json()["detail"]
        assert job["status"] == "failed" and "none was asked for" not in detail


# ======================================================== from the chat

async def test_the_notice_shows_the_document_the_run_then_creates(
    store, checkpointer, tmp_path
):
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "several steps",
                    document_name="chair_comparison.md",   # the model wrote an extension
                    document_title="Comparatif des chaises", formats=["markdown", "html"]),
        AIMessage(content="done"),
    ])
    events = [e async for e in ChatRunner(session.build()).stream(
        session.session_id, "compare the chairs please")]

    (started,) = [e for e in events if isinstance(e, JobStarted)]
    (job,) = await session.manager.list_jobs()
    assert started.job_id == job.job_id
    assert (started.document_name, started.document_title, started.formats) == (
        NAME, "Comparatif des chaises", ["markdown", "html"])
    assert (job.document_name, job.document_title, job.formats) == (
        NAME, "Comparatif des chaises", ["markdown", "html"])


@pytest.mark.parametrize(("fmt", "cause"), [("docx", "unknown report format"),
                                            ("pdf", "pango")])
async def test_a_format_that_cannot_be_written_is_not_launched_and_says_what_can(
    store, checkpointer, tmp_path, monkeypatch, fmt, cause
):
    """Refused in `create_job`, in front of whoever asked. → 0055, 0108"""
    from jobsmith.jobs import report_pdf

    monkeypatch.setattr(report_pdf, "_probed", None)
    if fmt == "pdf":
        no_libraries(monkeypatch)
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "several steps", formats=[fmt]),
        AIMessage(content="cannot."),
    ])
    out = await session.build().ainvoke(
        {"messages": [HumanMessage(f"compare the chairs as {fmt}")]}, CFG)

    assert await session.manager.list_jobs() == []
    reply = next(m for m in out["messages"] if isinstance(m, ToolMessage)).content
    assert "NOT launched" in reply and cause in reply
    possible = reply.split("formats available here:")[1]
    assert "markdown" in possible and (fmt != "pdf" or "pdf" not in possible)


async def test_the_chat_can_ask_for_no_file_and_the_model_is_told_so(
    store, checkpointer, tmp_path
):
    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "multi-step", formats=[]),
        AIMessage(content="Here you go."),
    ])
    await session.build().ainvoke({"messages": [HumanMessage("compare them, no file")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.formats == [] and job.report_path is None and job.error is None
    (result,) = [m for m in model.calls[-1]
                 if isinstance(m, ToolMessage) and m.name == "launch_job"]
    assert "none was asked for" in result.content
    assert "No deliverable file was saved" not in result.content
