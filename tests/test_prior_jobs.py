"""A follow-up references the JOB, not the file it wrote (#74).

Three layers, one property each: the adapter turns a stored run into the
port's vocabulary, the capability turns that into material the run can use,
and `research` reads it — because material that reaches only the final
generator is the defect #81 was opened for.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeLLM, ScriptedChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_default_pack import PACK_SCRIPT, notes_call
from test_jobs import make_manager

import jobsmith.agents.default.prior_jobs as prior_jobs_module
from jobsmith.agents.default.prior_jobs import PriorJobsCapability, referenced_jobs
from jobsmith.agents.default.research import ResearchCapability
from jobsmith.app import build_app
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.chat import ChatRunner, ChatSession
from jobsmith.chat.runner import JobStarted
from jobsmith.core.prior_jobs import PriorJob, PriorJobUnavailable, PriorStep
from jobsmith.core.state import FROM_JOBS_INPUT_KEY
from jobsmith.jobs.models import Job, JobStatus
from jobsmith.jobs.prior import RepositoryPriorJobs
from jobsmith.jobs.repository import StoreJobRepository


class FakePriorJobs:
    """A `PriorJobSource` scripted by job id; anything else is unavailable."""

    def __init__(self, jobs: dict[str, PriorJob] | None = None):
        self.jobs = dict(jobs or {})
        self.asked: list[str] = []

    async def load(self, job_id: str) -> PriorJob:
        self.asked.append(job_id)
        if job_id not in self.jobs:
            raise PriorJobUnavailable(f"{job_id!r}: no such job")
        return self.jobs[job_id]


def prior(job_id: str = "aaaaaaaa1111", *, answer: str = "The answer.",
          steps: tuple[PriorStep, ...] = (), status: str = "done") -> PriorJob:
    return PriorJob(job_id=job_id, query="compare X and Y", answer=answer,
                    status=status, steps=steps)


async def run(capability: PriorJobsCapability, *refs: str, query: str = "one-pager") -> dict:
    out = await capability.build().ainvoke(
        {"query": query, "inputs": {FROM_JOBS_INPUT_KEY: list(refs)}})
    return out["results"]["prior_jobs"]


# ---------------- the port's own shape ----------------


def test_nothing_in_the_capability_reaches_the_jobs_layer():
    """The layering the port exists for, asserted where it can be: the module
    the capability lives in imports `core` and nothing else of ours. A
    capability that knew `JobRepository` would be a capability that knows a
    store namespace, and the port would be decoration."""
    source = Path(prior_jobs_module.__file__ or "").read_text()
    assert "from ...jobs" not in source
    assert "jobsmith.jobs" not in source


async def test_the_adapter_reads_the_record_in_plan_order(store):
    """Plan order is the only deterministic order a run has — `results` is
    filled by parallel waves — so the adapter is where it is imposed, once."""
    repository = StoreJobRepository(store)
    job = Job(job_id="j1", status=JobStatus.DONE, query="compare X and Y",
              final_answer="X wins on price.")
    job.plan = {"steps": [{"capability": "web_search", "depends_on": []},
                          {"capability": "research", "depends_on": ["web_search"]}],
                "rationale": "r"}
    await repository.save_summary(job)
    await repository.save_plan("j1", job.plan)
    # saved in the OTHER order, as a wave that landed second would
    await repository.save_result("j1", "research", {"ok": True, "data": {"notes": "N"}})
    await repository.save_result("j1", "web_search", {
        "ok": True, "data": {"documents": [{"id": "d1", "text": "a page"}]}})

    loaded = await RepositoryPriorJobs(repository).load("j1")
    assert loaded.answer == "X wins on price."
    assert loaded.status == "done"
    assert [s.capability for s in loaded.steps] == ["web_search", "research"]
    assert "N" in loaded.steps[1].text


async def test_the_adapter_carries_a_failed_step_as_what_it_said(store):
    """"That run's web search failed" is material for whoever builds on it;
    dropping the step would read as "that run never searched"."""
    repository = StoreJobRepository(store)
    await repository.save_summary(Job(job_id="j2", status=JobStatus.DONE, query="q"))
    await repository.save_result("j2", "web_search",
                                 {"ok": False, "error": "the provider refused"})
    loaded = await RepositoryPriorJobs(repository).load("j2")
    assert loaded.steps[0].ok is False
    assert "provider refused" in loaded.steps[0].text


async def test_an_unknown_job_is_refused_rather_than_answered_empty(store):
    """"There is no such job" and "that job established nothing" are two
    different facts; an empty answer flattens them into the second."""
    with pytest.raises(PriorJobUnavailable):
        await RepositoryPriorJobs(StoreJobRepository(store)).load("nope")


# ---------------- the capability ----------------


async def test_the_answer_and_every_step_arrive_as_quotable_material():
    source = FakePriorJobs({"aaaaaaaa1111": prior(steps=(
        PriorStep("research", "Notes on X and Y."),
        PriorStep("analysis", "X wins on price."),
    ))})
    result = await run(PriorJobsCapability(source), "aaaaaaaa1111")
    assert result["ok"] is True
    documents = result["data"]["documents"]
    # the answer first — it is what a re-read of the file would have given —
    # then the material behind it, which the file never carried
    assert [d["id"] for d in documents] == [
        "aaaaaaaa#answer", "aaaaaaaa#research", "aaaaaaaa#analysis"]
    assert all(d["source"] == "job aaaaaaaa1111" for d in documents)
    assert "Notes on X and Y." in documents[1]["text"]


async def test_a_reference_that_cannot_be_served_is_material_not_silence():
    """One unloadable job among two does not cost the other, and the refusal
    travels — a model told nothing about it writes over the hole."""
    source = FakePriorJobs({"good": prior("good")})
    result = await run(PriorJobsCapability(source), "good", "gone")
    assert result["ok"] is True
    assert any("gone" in why for why in result["data"]["unavailable"])
    context = PriorJobsCapability(source).render_context(result) or ""
    assert "could NOT be read" in context and "gone" in context


async def test_a_job_that_produced_nothing_is_said_to_be_empty_not_missing():
    """It exists and its status says why; calling that "no such job" would be
    the one reading that is never true."""
    source = FakePriorJobs({"queued1": PriorJob(job_id="queued1", status="queued")})
    result = await run(PriorJobsCapability(source), "queued1")
    assert result["ok"] is False                       # nothing to work from
    assert "queued" in result["error"]


async def test_every_reference_failing_fails_the_step():
    result = await run(PriorJobsCapability(FakePriorJobs()), "gone")
    assert result["ok"] is False
    assert "gone" in result["error"]


async def test_the_material_is_bounded_and_says_where_it_was_cut():
    """A previous run's results measured 44k characters across four steps, and
    this block is re-sent by every downstream step that reads the merged
    context. So it is bounded, nothing is dropped whole, and each cut is
    written into the text — material silently shortened reads as material that
    is whole."""
    source = FakePriorJobs({"j": prior("j", answer="answer " * 400, steps=(
        PriorStep("research", "alpha " * 400),
        PriorStep("analysis", "beta " * 400),
    ))})
    result = await run(PriorJobsCapability(source, max_material_chars=600), "j")
    texts = [d["text"] for d in result["data"]["documents"]]
    assert len(texts) == 3                                    # none dropped whole
    assert all("truncated" in t for t in texts)               # every cut is declared
    assert sum(len(t) for t in texts) < 600 + 3 * 80          # the bound plus its notes


async def test_the_answer_is_not_squeezed_out_by_the_steps():
    """The answer is the one thing re-reading the file WOULD have given, so a
    run with many long steps must not reduce it to a sentence."""
    source = FakePriorJobs({"j": prior("j", answer="answer " * 200, steps=tuple(
        PriorStep(f"step{i}", "filler " * 400) for i in range(6)))})
    result = await run(PriorJobsCapability(source, max_material_chars=1400), "j")
    answer = next(d for d in result["data"]["documents"] if d["id"].endswith("#answer"))
    assert len(answer["text"]) > 500


async def test_only_so_many_jobs_are_read_and_the_rest_are_named():
    source = FakePriorJobs({f"j{i}": prior(f"j{i}") for i in range(5)})
    result = await run(PriorJobsCapability(source, max_jobs=2), *[f"j{i}" for i in range(5)])
    assert source.asked == ["j0", "j1"]
    assert len(result["data"]["unavailable"]) == 3


def test_a_request_that_references_no_job_drops_the_step():
    """`requires_inputs` gets the key; the key being present is not the same
    as referencing something (`read_files` overrides `is_applicable` for the
    same reason)."""
    capability = PriorJobsCapability(FakePriorJobs())
    assert capability.spec.requires_inputs == (FROM_JOBS_INPUT_KEY,)
    assert capability.is_applicable({"query": "q", "inputs": {}}) is False
    assert capability.is_applicable({"query": "q", "inputs": {FROM_JOBS_INPUT_KEY: []}}) is False
    assert capability.is_applicable({"query": "q", "inputs": {FROM_JOBS_INPUT_KEY: ["j"]}})


def test_a_single_id_where_a_list_was_documented_is_read_generously():
    assert referenced_jobs({FROM_JOBS_INPUT_KEY: "j1"}) == ["j1"]
    assert referenced_jobs({FROM_JOBS_INPUT_KEY: [" j1 ", ""]}) == ["j1"]
    assert referenced_jobs({FROM_JOBS_INPUT_KEY: 3}) == []
    assert referenced_jobs(None) == []


async def test_the_report_names_the_runs_it_read_and_never_a_path():
    source = FakePriorJobs({"aaaaaaaa1111": prior()})
    result = await run(PriorJobsCapability(source), "aaaaaaaa1111")
    report = PriorJobsCapability(source).render_report(result) or ""
    assert "job aaaaaaaa1111" in report


# ---------------- and it reaches the reasoning ----------------


async def test_research_writes_its_notes_from_the_earlier_run(store):
    """The whole point of the referencing, and the defect #81 named: material
    that reaches only the final generator is material three steps too late."""
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({
        "query": "one-pager", "inputs": {},
        "results": {"prior_jobs": {"ok": True, "data": {
            "documents": [{"id": "aaaaaaaa#analysis", "title": "analysis",
                           "source": "job aaaaaaaa1111", "text": "X wins on price"}],
            "unavailable": ["'gone': no such job"]}}},
    })
    call = notes_call(llm)
    assert call["messages"][0]["content"].startswith(
        ResearchCapability.GROUNDED_NOTES_SYSTEM)      # sourced, not recalled
    user = call["messages"][1]["content"]
    assert "X wins on price" in user
    # the refusal crosses too, under its own label and after the material
    assert user.index("X wins on price") < user.index("could NOT be read")
    assert out["results"]["research"]["meta"]["grounded_on"] == ["prior_jobs"]


# ---------------- the scope, where every other job tool puts it -------------


async def test_a_reference_is_resolved_against_this_session_and_becomes_an_id(
    store, checkpointer, tmp_path
):
    """A prefix in, a full id of THIS session's jobs out. `chat/tools.py` is
    the one place that knows who is asking — the job engine never sees the
    conversation — so it is the one place the scope can be enforced."""
    manager = make_manager(store, checkpointer, tmp_path)
    earlier = await manager.run_job(
        (await manager.create_job("the first task", session_id="s-mine")).job_id)

    model = ScriptedChatModel(responses=[
        launch_call("make a one-pager out of the earlier comparison", "builds on it",
                    from_jobs=[earlier.job_id[:8], "  "]),
        AIMessage(content="Done."),
    ])
    session = ChatSession(manager, model, session_id="s-mine", checkpointer=MemorySaver())
    runner = ChatRunner(session.build())

    [e async for e in runner.stream(session.session_id, "one-pager out of that")]

    launched = next(j for j in await manager.list_jobs(session_id="s-mine")
                    if j.job_id != earlier.job_id)
    assert launched.inputs[FROM_JOBS_INPUT_KEY] == [earlier.job_id]


async def test_a_job_referenced_twice_is_handed_over_once(store, checkpointer, tmp_path):
    """A prefix and the full id of the same job are one reference: the run
    loads it once (its 24 000-character budget is not spent twice on the same
    material) and the notice names it once — the two must agree (#104)."""
    manager = make_manager(store, checkpointer, tmp_path)
    first = await manager.create_job("the first task", session_id="s-mine")
    second = await manager.create_job("the second task", session_id="s-mine")

    model = ScriptedChatModel(responses=[
        launch_call("compare them", "builds on both",
                    from_jobs=[second.job_id[:8], first.job_id, second.job_id]),
        AIMessage(content="Done."),
    ])
    session = ChatSession(manager, model, session_id="s-mine", checkpointer=MemorySaver())
    runner = ChatRunner(session.build())

    events = [e async for e in runner.stream(session.session_id, "compare them")]

    launched = next(j for j in await manager.list_jobs(session_id="s-mine")
                    if j.job_id not in (first.job_id, second.job_id))
    assert launched.inputs[FROM_JOBS_INPUT_KEY] == [second.job_id, first.job_id]
    (notice,) = [e for e in events if isinstance(e, JobStarted)]
    assert [ref["job_id"] for ref in notice.from_jobs] == [second.job_id, first.job_id]


async def test_another_sessions_job_is_not_referenceable(store, checkpointer, tmp_path):
    """The scope is the point: a job reference must not become a way to read
    another conversation's work. It is refused BEFORE the run, in words the
    model can act on, exactly as an impossible format is."""
    manager = make_manager(store, checkpointer, tmp_path)
    theirs = await manager.run_job(
        (await manager.create_job("their task", session_id="s-theirs")).job_id)

    model = ScriptedChatModel(responses=[
        launch_call("build on that", "builds on it", from_jobs=[theirs.job_id]),
        AIMessage(content="I could not find that job."),
    ])
    session = ChatSession(manager, model, session_id="s-mine", checkpointer=MemorySaver())
    runner = ChatRunner(session.build())

    [e async for e in runner.stream(session.session_id, "build on that")]

    assert await manager.list_jobs(session_id="s-mine") == []      # nothing ran
    refusal = model.calls[-1][-1].content
    assert "NOT launched" in refusal and theirs.job_id[:8] in refusal


async def test_a_launch_that_references_no_job_carries_no_key(
    store, checkpointer, tmp_path
):
    """Absent, not empty: the planner drops the step rather than planning one
    that can only report that nothing was referenced."""
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=[
        launch_call("analyse the alpha data", "several steps"),
        AIMessage(content="Done."),
    ])
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    runner = ChatRunner(session.build())

    [e async for e in runner.stream(session.session_id, "analyse it")]

    (job,) = await manager.list_jobs(session_id=session.session_id)
    assert FROM_JOBS_INPUT_KEY not in job.inputs


# ---------------- the loop this issue is about ----------------


async def test_a_job_builds_on_the_previous_one_with_no_file_at_all(tmp_path):
    """The product's own loop, through the real path — and the premise #84
    made stronger: the first job writes NO document, and the second one still
    has everything it produced."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(),
                          db="memory", reports_dir=str(tmp_path / "artifacts"))
    try:
        assert "prior_jobs" in app.registry.names()
        first = await app.manager.run_job(
            (await app.manager.create_job("study the topic in depth", formats=[])).job_id)
        assert first.status is JobStatus.DONE
        assert first.report_path is None, "no file: nothing to re-read"

        second = await app.manager.create_job(
            "make a one-pager out of that job",
            {FROM_JOBS_INPUT_KEY: [first.job_id]},
        )
        second = await app.manager.run_job(second.job_id)
        assert second.status is JobStatus.DONE
        result = second.results["prior_jobs"]
        assert result["ok"] is True, result.get("error")
        texts = " ".join(d["text"] for d in result["data"]["documents"])
        assert first.final_answer and first.final_answer[:40] in texts
        # ...and the material each step produced, which the deliverable never
        # carried: the whole reason the reference beats the file
        assert {d["id"].split("#")[1] for d in result["data"]["documents"]} > {"answer"}
    finally:
        await app.aclose()
