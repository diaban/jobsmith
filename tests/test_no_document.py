"""A run leaves a document because the request asked for one (#84).

Say hello to the engine and you got a document. `/bg "bonjour"` — or any
request the router sends `direct` — walked
`validate_input → router → direct_answer → validate_output → post_process`,
reached a DELIVERED terminal, and the manager wrote a file: a title, a
provenance section, the request quoted back, a plan table with no rows and a
mermaid diagram of nothing. The graph had a route for *this needs no
capability*; it had no outcome for *this needs no document*, because the write
asked one question — did the run reach DELIVERED.

One gate, and it is not a guess about duration or mode: **the request**.
`Job.formats` gained a third state, and `[]` means no file — #55 built that
channel and stopped one field short of it. #84 left the *silent* request to
the run's shape (a run that planned filed its answer); #96 closed that too,
once #85 gave every answer a verbatim way back — so a request that said
nothing gets no file whatever the run did. `tests/test_silent_request.py`
pins that half; this file pins the ones that spoke and the absences.

The third thing pinned here is that the absence is *legible*: `report_path is
None` already meant "the run did not answer" and "the write failed", and a
caller that cannot tell those from "there was nothing to write" tells the user
the wrong one.
"""
from __future__ import annotations

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END
from test_jobs import SlowEcho, make_manager

from jobsmith.core.artifacts import ArtifactRef, artifact_meta
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.jobs.models import Job, JobStatus
from jobsmith.jobs.report import compose_reporters
from jobsmith.jobs.repository import StoreJobRepository

ANSWER = "A sufficiently long final answer for the job test."


def direct_llm() -> FakeLLM:
    """A model whose triage always answers `direct` — the greeting's route."""
    return FakeLLM({"triage step": '{"route": "direct", "rationale": "trivial"}'},
                   default=ANSWER)


def files_under(tmp_path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "artifacts").rglob("*") if p.is_file())


# ------------------------------------------------- the run that has nothing to file

async def test_a_greeting_leaves_no_document_and_no_error(store, checkpointer, tmp_path):
    """The defect, end to end: DONE, answered, and nothing on disk.

    Not FAILED and not an error either — nothing went wrong. `job.error` is
    the channel for a write that broke (#28), and putting "no file was asked
    for" there would be the same conflation this test exists to undo.
    """
    mgr = make_manager(store, checkpointer, tmp_path, llm=direct_llm())
    done = await mgr.run_job((await mgr.create_job("bonjour")).job_id)

    assert done.status is JobStatus.DONE and done.terminal_kind == "answer"
    assert done.final_answer == ANSWER          # the answer is not the casualty
    assert done.plan is None                    # nothing was planned: nothing to report on
    assert done.outputs == [] and done.report_path is None
    assert done.error is None
    assert done.deliverable_expected is False   # ...and it says why there is no file
    assert files_under(tmp_path) == []


async def test_an_empty_plan_leaves_no_document_either(store, checkpointer, tmp_path):
    """The empty plan joins the direct route here exactly as it does in the
    graph: every step dropped as inapplicable is a run with nothing to report
    on, answered by the same node."""

    class NeedsAnImage(SlowEcho):
        def __init__(self):
            super().__init__("vision")
            self.spec = CapabilitySpec(name="vision", description="reads an image",
                                       requires_inputs=("image_s3_keys",))

    llm = FakeLLM({"triage step": '{"route": "plan", "rationale": "looks complex"}',
                   "planner": plan_json("vision")}, default=ANSWER)
    mgr = make_manager(store, checkpointer, tmp_path, caps=[NeedsAnImage()], llm=llm)
    done = await mgr.run_job((await mgr.create_job("describe the picture")).job_id)

    assert done.plan == {"steps": [], "rationale": "test plan"}
    assert done.status is JobStatus.DONE and done.terminal_kind == "answer"
    assert done.deliverable_expected is False and done.outputs == []


async def test_a_run_that_planned_writes_the_document_it_was_asked_for(
    store, checkpointer, tmp_path
):
    """The half that must not move: a request that asked for a document gets
    one. What changed in #96 is only who has to ask — the plan no longer
    does it on the request's behalf."""
    mgr = make_manager(store, checkpointer, tmp_path)
    done = await mgr.run_job(
        (await mgr.create_job("compare the chairs", formats=["markdown"])).job_id)

    assert done.deliverable_expected is True
    assert done.report_path == str(tmp_path / "artifacts" / f"{done.job_id}.md")


# ------------------------------------------------- the request decides first

async def test_the_request_wins_over_the_route(store, checkpointer, tmp_path):
    """A request that asked for a file gets one, whatever the router made of
    the sentence. Withholding it because of how a triage step read the request
    would be a second silent decision — the thing #55 exists to stop."""
    mgr = make_manager(store, checkpointer, tmp_path, llm=direct_llm())
    done = await mgr.run_job(
        (await mgr.create_job("bonjour", formats=["markdown"])).job_id)

    assert done.deliverable_expected is True
    assert done.report_path is not None and done.plan is None
    assert files_under(tmp_path) == [f"{done.job_id}.md"]


async def test_a_request_for_no_document_is_obeyed_by_a_run_that_planned(
    store, checkpointer, tmp_path
):
    """`formats=[]` is the channel #55 stopped short of: a full plan, four
    steps, an answer — and no file, because none was asked for."""
    mgr = make_manager(store, checkpointer, tmp_path,
                       caps=[SlowEcho("alpha"), SlowEcho("beta")])
    job = await mgr.create_job("compare the chairs", formats=[])
    # known before the run, and recorded before the run: a QUEUED job that
    # reads as expecting a document it will never get is the same confusion.
    assert job.deliverable_expected is False and job.formats == []

    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.plan is not None and set(done.results) == {"alpha", "beta"}
    assert done.outputs == [] and done.error is None
    assert files_under(tmp_path) == []


async def test_saying_nothing_is_not_saying_no(store, checkpointer, tmp_path):
    """The two silences the record must keep apart. `None` is "the request
    said nothing" and `[]` is "no document". Since #96 they END the same way,
    and they are still two facts: `None` is what the engine's document step
    may still fill from the sentence ("write me a report"), `[]` is what it
    must never override — collapsing them (`formats or []`) would deafen that
    step to every silent request."""
    mgr = make_manager(store, checkpointer, tmp_path)
    silent = await mgr.create_job("compare the chairs")
    refused = await mgr.create_job("compare the chairs", formats=[])

    assert silent.formats is None and silent.deliverable_expected is True
    assert refused.formats == [] and refused.deliverable_expected is False
    # and it survives the store, which is where the collapse used to happen
    reloaded = await mgr.get_job(silent.job_id)
    assert reloaded is not None and reloaded.formats is None
    assert (await mgr.get_job(refused.job_id)).formats == []


async def test_a_record_written_before_this_change_still_means_unstated(store):
    """Between #55 and #84 an empty `formats` meant "the deployment decides",
    and such a record carries no `deliverable_expected` key at all. That is
    what tells the two vintages apart, so an old job resumed today is not read
    as having refused a file it never refused."""
    old = {"status": "cancelled", "query": "q", "formats": [],
           "created_at": "2026-01-01T00:00:00+00:00"}
    await store.aput(("jobs", "index"), "old1", old)
    job = await StoreJobRepository(store).load("old1")

    assert job is not None and job.formats is None and job.deliverable_expected is True


# ------------------------------------------------- telling the absences apart

async def test_the_three_reasons_there_is_no_report_are_told_apart(
    store, checkpointer, tmp_path
):
    """`report_path is None` is three different facts, and a front-end that
    says "no report available (is the job done?)" for all of them is wrong
    twice — it tells one reader to wait for a file that failed, and another to
    wait for one nobody ever asked for. Each fact has its own channel, and
    none of them is prose: `status`, `error`, `deliverable_expected`."""

    class Boom:
        format, extension, binary = "markdown", "md", False

        def write(self, job, directory):
            raise OSError("No space left on device")

    # 1. nothing was asked for
    quiet = make_manager(store, checkpointer, tmp_path, llm=direct_llm())
    none_wanted = await quiet.run_job((await quiet.create_job("bonjour")).job_id)

    # 2. the write failed
    broken = make_manager(store, checkpointer, tmp_path)
    broken.reporter = Boom()
    write_failed = await broken.run_job(
        (await broken.create_job("compare them", formats=["markdown"])).job_id)

    # 3. the run never answered (input validation rejects, terminal user_error)
    failing = make_manager(store, checkpointer, tmp_path)
    stopped = await failing.run_job((await failing.create_job("   ")).job_id)

    assert all(j.report_path is None for j in (none_wanted, write_failed, stopped))
    assert (none_wanted.status, none_wanted.deliverable_expected, none_wanted.error) == (
        JobStatus.DONE, False, None)
    assert write_failed.status is JobStatus.DONE
    assert write_failed.deliverable_expected is True
    assert "markdown" in (write_failed.error or "")
    assert stopped.status is JobStatus.FAILED and stopped.deliverable_expected is True


async def test_a_run_with_no_document_still_records_what_its_steps_produced(
    store, checkpointer, tmp_path
):
    """#41 in the one place it could have been lost: annexes are collected at
    every terminal, and "no deliverable" must not become "no outputs". A file
    a step wrote exists whether or not anyone asked for a report of the run —
    and it is still an annex, never the `main` #28 allows exactly one of."""

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

    from jobsmith.core.artifacts import LocalArtifactStore

    mgr = make_manager(store, checkpointer, tmp_path,
                       caps=[WritesAFile(LocalArtifactStore(tmp_path / "artifacts"))])
    done = await mgr.run_job(
        (await mgr.create_job("draw me one", formats=[])).job_id)

    assert done.deliverable_expected is False and done.report_path is None
    [annex] = done.outputs
    assert (annex.role, annex.produced_by, annex.name) == ("annex", "charts", "chart.svg")
    assert done.error is None


def test_a_reporter_is_never_asked_to_compose_nothing():
    """`compose_reporters([])` used to mean markdown, because empty meant "the
    caller said nothing". Empty now means "no document", and a Reporter is the
    last place that can be honoured — silently writing markdown for a job that
    asked for no file is the same mistake as writing it for one that asked for
    HTML."""
    with pytest.raises(ValueError, match="no report format"):
        compose_reporters([])
    assert compose_reporters("markdown").format == "markdown"


def test_the_record_says_it_where_every_front_end_can_read_it():
    """The fact travels on the job, not as a fourth exception on the port: the
    four surfaces that answer "where is the report" (`jobsmith report`,
    `/report`, the REPL, the TUI) all hold the record already."""
    job = Job(job_id="j1", status=JobStatus.DONE, query="bonjour",
              deliverable_expected=False)

    assert job.to_dict()["deliverable_expected"] is False
    assert job.summary()["deliverable_expected"] is False


# ------------------------------------------------- the two front doors

async def test_the_chat_can_ask_for_no_file_and_the_model_is_told_so(
    store, checkpointer, tmp_path
):
    """`[]` is sayable from the conversation, and the tool result says what
    happened rather than leaving the model to guess.

    Without the branch, a turn whose job wrote nothing carries no mention of a
    file at all — and a model that was told to name the deliverable either
    invents a path or apologises for a failure that did not happen. Neither is
    true, and both are read by the user as the product misbehaving.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from test_chat import CFG, launch_call, make_session

    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "multi-step", formats=[]),
        AIMessage(content="Here you go."),
    ])
    await session.build().ainvoke({"messages": [HumanMessage("compare them, no file")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.formats == [] and job.deliverable_expected is False
    assert job.report_path is None and job.error is None

    (result,) = [m for m in model.calls[-1]
                 if isinstance(m, ToolMessage) and m.name == "launch_job"]
    assert "none was asked for" in result.content
    assert "No deliverable file was saved" not in result.content


async def test_the_api_404_says_which_absence_it_is(store, checkpointer, tmp_path):
    """A 404 is the right code — there is no such resource — and "not DONE
    yet?" is the wrong reason for a run that finished, answered, and was asked
    for no file. The machine-readable half of the same fact is on the job."""
    from test_api import client_for, make_app, wait_done

    app, _ = make_app(store, checkpointer, tmp_path, [])
    async with client_for(app) as client:
        launched = (await client.post(
            "/jobs", json={"query": "just answer me", "formats": []})).json()
        job = await wait_done(client, launched["job_id"])
        assert job["deliverable_expected"] is False

        refused = await client.get(f"/jobs/{job['job_id']}/report")
        assert refused.status_code == 404
        assert "none was asked for" in refused.json()["detail"]
        assert "not DONE yet" not in refused.json()["detail"]


# ------------------------------------------------- what the screens say

def test_a_front_end_says_no_file_rather_than_not_yet():
    """"no file yet" and "no report available (is the job done?)" both read as
    *not yet*, and for a run that was asked for no document there is no yet.

    Wording lives in the presentation layers by standing rule (the fact is
    `deliverable_expected`), so this checks each of them says something, not
    that they say the same thing.
    """
    from jobsmith.cli.repl import job_lines, no_document_note
    from jobsmith.tui.render import outputs_block

    assert "none was asked for" in no_document_note({"job_id": "abcdef1234"})
    assert "abcdef12" in no_document_note({"job_id": "abcdef1234"})
    assert "no file" in outputs_block([], expected=False)
    assert "no file yet" in outputs_block([], expected=True)

    # ...and the notice shown BEFORE the run says it too, in both shapes of
    # "no file": asked for none (#84), and said nothing (#96) — which is no
    # file as well now, so leaving the line out would read as a file nobody
    # mentioned.
    asked_none = job_lines({"query": "q", "formats": []})
    assert any("no file" in line for line in asked_none)
    said_nothing = job_lines({"query": "q"})
    assert any("writes" in line and "no file" in line for line in said_nothing)
    assert any("markdown" in line for line in job_lines({"query": "q", "formats": ["markdown"]}))


def test_a_filename_is_never_shown_for_a_file_nobody_will_write():
    """The trap in `deliverable_filenames`: it guessed markdown for an empty
    `formats`, because empty used to mean "the deployment decides". A job that
    named its document and asked for no file would then be announced as
    writing `chair_notes.md`, which is the promise #55 exists to stop — and
    since #96 the same is true of `None`, which it went on guessing for."""
    from jobsmith.jobs.report import deliverable_filenames

    assert deliverable_filenames("chair_notes", None) == []
    assert deliverable_filenames("chair_notes", []) == []
    assert deliverable_filenames("chair_notes", ["html"]) == ["chair_notes.html"]
