"""The engine reads the request for what document it asked for (#90).

`jobsmith run "compare X and Y, give me that as a PDF"` wrote markdown. The
sentence was interpreted in exactly one place — the chat model filling
`launch_job`'s arguments — so `jobsmith run`, `/bg` and `POST /jobs` were all
deaf to it: #61's defect surviving one door over from where #55 fixed it.

So the decision moved into the graph, as a node of its own with an entry in
the path map. What is pinned here is not that a model can read a sentence —
it is the four properties that make such a node safe to run mid-flight:

- **it fills silence and never overrides**, and the gate is structural: a
  caller who named formats costs no model call at all;
- **it writes to state, never to the store** — the fact reaches the record
  through `jobs/runner.py` and `JobManager._apply`, like every other;
- **it cannot refuse**: it chooses from what this deployment can render, and
  anything else degrades to silence;
- **it is fail-open**, so every error lands on silence — which, since #96,
  is no file: a missing document is said on the record, an invented one is not.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, plan_json
from support import SlowEcho

from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.document import DocumentIntent
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.runner import FormatsChosen, GraphRunner, PlanReady

RENDERABLE = ("html", "markdown", "pdf")
ANSWER = "A sufficiently long final answer for the job test."

# What the prompt asks for, in the three shapes it accepts.
NAMED_HTML = json.dumps({"document": "named", "formats": ["html"]})
NO_FILE = json.dumps({"document": "none"})
UNSPECIFIED = json.dumps({"document": "unspecified"})
REQUESTED = json.dumps({"document": "requested"})


def make_node(reply: str, *, formats: tuple[str, ...] = RENDERABLE) -> DocumentIntent:
    return DocumentIntent(Deps(llm=FakeLLM({"document step": reply})), formats)


def asked(llm: FakeLLM) -> list[dict]:
    """The calls that went to the document step, and no others."""
    return [c for c in llm.calls
            if "document step" in c["messages"][0].get("content", "")]


# ------------------------------------------------------------ what it reads

async def test_a_format_named_in_the_request_becomes_the_decision():
    node = make_node(NAMED_HTML)
    assert await node.run({"query": "compare X and Y, as an html page"}) == {
        "document_formats": ["html"]}


async def test_a_request_that_says_nothing_writes_nothing():
    """Silence, not a default. Since #96 that silence IS the decision — no
    file — and it is written as nothing so the fail-open path and the
    ordinary one are the same path.
    """
    node = make_node(UNSPECIFIED)
    assert await node.run({"query": "compare X and Y"}) == {}


async def test_a_document_without_a_format_becomes_the_deployment_s_formats():
    """"write me a report" wants a file and names none (#96): the node answers
    with the default it was handed, as names, so nothing downstream learns a
    fourth state."""
    node = DocumentIntent(Deps(llm=FakeLLM({"document step": REQUESTED})), RENDERABLE,
                          default_formats=("html", "markdown"))
    assert await node.run({"query": "write me a report on X"}) == {
        "document_formats": ["html", "markdown"]}


async def test_a_document_without_a_format_is_silence_where_no_default_was_given():
    """A node nobody told the default cannot pick one — the same rule as a
    node told no renderable format. Silence, i.e. no file: never markdown by
    guess, which is the fallback #96 removed everywhere else."""
    for default in ((), ("docx",)):
        node = DocumentIntent(Deps(llm=FakeLLM({"document step": REQUESTED})),
                              RENDERABLE, default_formats=default)
        assert await node.run({"query": "write me a report on X"}) == {}


def _reads(reply: dict) -> DocumentIntent:
    return DocumentIntent(Deps(llm=FakeLLM({"document step": json.dumps(reply)})),
                          ("markdown", "html"), default_formats=("markdown",))


QUERY_HTML = {"query": "compare two schedulers and give me the result as an html page"}


async def test_formats_the_reply_carries_win_over_a_requested_label():
    """Measured on gpt-5-nano (#97 review): 1 call in 12 answered
    `requested` WITH `["html"]`, and the label alone resolved to the default
    — markdown, for a request that said html. The names are the more
    specific thing the model said."""
    node = _reads({"document": "requested", "formats": ["html"]})
    assert await node.run(QUERY_HTML) == {"document_formats": ["html"]}


async def test_a_format_name_used_as_the_label_names_that_format():
    """3 calls in 12 on the #90 prompt answered `{"document": "html",
    "formats": ["html"]}` — read as silence, so no file at all."""
    node = _reads({"document": "html", "formats": ["html"]})
    assert await node.run(QUERY_HTML) == {"document_formats": ["html"]}


async def test_a_format_name_as_the_label_names_it_even_with_no_formats():
    node = _reads({"document": "HTML"})
    assert await node.run(QUERY_HTML) == {"document_formats": ["html"]}


async def test_no_document_is_not_overruled_by_a_stray_list():
    """The label says no file: an explicit "no document" is the one answer a
    list next to it must never turn into a file."""
    node = _reads({"document": "none", "formats": ["html"]})
    assert await node.run({"query": "just answer here, no file"}) == {
        "document_formats": []}


async def test_an_unsure_label_does_not_become_a_file_because_a_list_came_with_it():
    node = _reads({"document": "unspecified", "formats": ["html"]})
    assert await node.run({"query": "compare two schedulers"}) == {}


async def test_requested_with_nothing_renderable_still_gets_the_default():
    node = _reads({"document": "requested", "formats": ["docx"]})
    assert await node.run({"query": "write me a report"}) == {
        "document_formats": ["markdown"]}


async def test_a_request_for_no_file_at_all_is_a_decision_and_says_so():
    """`[]` is not "nothing to say": it is the third state `Job.formats` has."""
    node = make_node(NO_FILE)
    assert await node.run({"query": "just answer here, do not write a file"}) == {
        "document_formats": []}


def test_the_prompt_offers_only_what_this_deployment_can_render():
    prompt = make_node(UNSPECIFIED, formats=("html", "markdown")).system_prompt()
    assert "- html" in prompt and "- markdown" in prompt
    assert "- pdf" not in prompt


# --------------------------------------------- the gate, before any call

async def test_a_caller_who_already_named_formats_is_never_second_guessed():
    """The structural gate: free, deterministic, and no second interpreter.

    Two readers of one sentence that can disagree is the failure mode; the
    node exists to fill silence. `document_formats` is seeded at entry with
    what the caller asked for (`GraphRunner.stream`), so a value there ends
    the question before a model is asked anything.
    """
    llm = FakeLLM({"document step": NAMED_HTML})
    node = DocumentIntent(Deps(llm=llm), RENDERABLE)
    state = {"query": "compare X and Y, as an html page",
             "document_formats": ["markdown"]}

    assert await node.run(state) == {}
    assert asked(llm) == []


async def test_a_caller_who_asked_for_no_file_is_not_second_guessed_either():
    """`[]` is an answer, so the gate is `is not None` and not truthiness."""
    llm = FakeLLM({"document step": NAMED_HTML})
    node = DocumentIntent(Deps(llm=llm), RENDERABLE)

    assert await node.run({"query": "as a pdf", "document_formats": []}) == {}
    assert asked(llm) == []


async def test_a_deployment_that_can_render_nothing_asks_nothing():
    """Same rule as a capability nothing can serve: it stays out of the way.

    `core/` never learns what a Reporter is, so a builder nobody told cannot
    name a real format — and a node with nothing to choose from must not
    invent one.
    """
    llm = FakeLLM({"document step": NAMED_HTML})
    node = DocumentIntent(Deps(llm=llm), ())

    assert await node.run({"query": "as an html page"}) == {}
    assert asked(llm) == []


# ------------------------------------------------------------- fail-open

class Exploding:
    async def chat(self, messages, **kwargs):
        raise RuntimeError("provider down")


@pytest.mark.parametrize(
    ("why", "reply"),
    [
        ("not json at all", "sure, a PDF!"),
        ("an answer the prompt never offered", json.dumps({"document": "maybe"})),
        ("named, with no format in it", json.dumps({"document": "named"})),
        ("named, with a format this deployment cannot render",
         json.dumps({"document": "named", "formats": ["docx"]})),
        ("formats that are not even a list",
         json.dumps({"document": "named", "formats": "pdf"})),
    ],
)
async def test_every_answer_it_cannot_use_degrades_to_silence(why, reply):
    """It cannot refuse — which is what lets it run where nobody is listening.

    #55 put both document refusals in `create_job`, where whoever asked can
    still fix them. A node three minutes into a run has no one to tell, so an
    answer it cannot honour is dropped and the run carries on exactly as it
    would have without it.
    """
    assert await make_node(reply).run({"query": "give me that as a docx"}) == {}, why


async def test_a_provider_that_blows_up_costs_the_run_nothing():
    node = DocumentIntent(Deps(llm=Exploding()), RENDERABLE)
    assert await node.run({"query": "as an html page"}) == {}


# ------------------------------------------------ the fact reaches the record

def intent_llm(reply: str) -> FakeLLM:
    return FakeLLM(
        {"document step": reply, "planner": plan_json("alpha")}, default=ANSWER)


def make_engine(store, checkpointer, tmp_path, llm: FakeLLM) -> JobManager:
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry([SlowEcho("alpha")]),
        checkpointer=checkpointer, document_formats=RENDERABLE,
        default_document_formats=("markdown",),
    )
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts",
                      default_formats=("markdown",))


async def test_a_request_that_names_its_format_gets_that_file(
    store, checkpointer, tmp_path
):
    """The defect, end to end, through the door the chat model never sees.

    No caller said anything about a format — `create_job` is called exactly
    as `jobsmith run` and `POST /jobs` call it — and the file that lands is
    the one the sentence asked for.
    """
    mgr = make_engine(store, checkpointer, tmp_path, intent_llm(NAMED_HTML))
    job = await mgr.create_job("compare X and Y, give me that as an html page")
    assert job.formats is None

    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.formats == ["html"]
    assert done.report_path is not None and done.report_path.endswith(".html")
    assert [o.format for o in done.outputs] == ["html"]
    # ...and it is on the record, not only on the file: a watcher reading the
    # summary sees what this job is going to leave behind.
    index = await store.aget(("jobs", "index"), job.job_id)
    assert index.value["formats"] == ["html"]


async def test_the_caller_stays_authoritative_and_is_not_even_asked(
    store, checkpointer, tmp_path
):
    """The same sentence, with a caller who already spoke: markdown wins.

    This is the gate seen from the outside — the chat model's answer is not
    re-litigated by the engine, and no token is spent finding that out.
    """
    llm = intent_llm(NAMED_HTML)
    mgr = make_engine(store, checkpointer, tmp_path, llm)
    job = await mgr.create_job(
        "compare X and Y, give me that as an html page", formats=["markdown"])

    done = await mgr.run_job(job.job_id)
    assert done.formats == ["markdown"]
    assert done.report_path is not None and done.report_path.endswith(".md")
    assert asked(llm) == []


async def test_a_request_for_no_file_leaves_none_and_says_it_was_meant(
    store, checkpointer, tmp_path
):
    """The engine reaching #84's other state — and keeping it coherent.

    `deliverable_expected` only ever goes True → False, so the decision is
    recorded when it is taken, exactly as `create_job` records it for a
    caller who asked for no file. Nothing went wrong, so `job.error` stays
    empty: a caller printing "no report available" for all three meanings of
    `report_path is None` tells the user the wrong one.
    """
    mgr = make_engine(store, checkpointer, tmp_path, intent_llm(NO_FILE))
    job = await mgr.create_job("compare X and Y, but do not write any file")

    done = await mgr.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.formats == []
    assert done.deliverable_expected is False
    assert done.report_path is None
    assert done.error is None
    assert not list((tmp_path / "artifacts").rglob("*.md"))


async def test_a_run_that_stops_still_records_that_it_owed_no_document(
    store, checkpointer, tmp_path
):
    """Known when it is decided, recorded when it is decided.

    The terminal path already turns `deliverable_expected` off for a DONE run
    that owed no file (#84), so this is the case that needs the fold to do it
    too: a run that never reaches a terminal at all. A FAILED record claiming
    a document was expected is the confusion the field exists to remove —
    `create_job` states it for a caller who asked for no file, and the engine
    asking for none is the same fact learned one layer later.
    """
    llm = FakeLLM({"document step": NO_FILE, "planner": "not json at all"},
                  default=ANSWER)
    mgr = make_engine(store, checkpointer, tmp_path, llm)
    done = await mgr.run_job((await mgr.create_job("answer me here, no file")).job_id)

    assert done.status is JobStatus.FAILED
    assert done.deliverable_expected is False
    index = await store.aget(("jobs", "index"), done.job_id)
    assert index.value["deliverable_expected"] is False


async def test_a_silent_request_gets_no_document_even_though_it_planned(
    store, checkpointer, tmp_path
):
    """#96, end to end: the node found nothing in the sentence, the run
    planned and executed a step — and no file, because nobody asked for one.

    Until #96 the plan's shape answered what the request did not, and this
    run wrote markdown. `formats` stays `None` on the record: "said nothing"
    is kept apart from "said no file" even though both end without one.
    """
    mgr = make_engine(store, checkpointer, tmp_path, intent_llm(UNSPECIFIED))
    job = await mgr.create_job("compare X and Y")

    done = await mgr.run_job(job.job_id)
    assert done.plan is not None and "alpha" in done.results
    assert done.formats is None
    assert done.deliverable_expected is False
    assert done.report_path is None and done.error is None
    assert not list((tmp_path / "artifacts").rglob("*.md"))


async def test_silence_is_recorded_when_the_node_answers_not_at_the_end(
    store, checkpointer, tmp_path
):
    """Known when it is decided, recorded when it is decided — the rule the
    `[]` answer already follows, applied to silence now that silence is a
    decision (#96). A run that stops after the document step and before its
    terminal must not leave a record promising a file it never owed."""
    llm = FakeLLM({"document step": UNSPECIFIED, "planner": "not json at all"},
                  default=ANSWER)
    mgr = make_engine(store, checkpointer, tmp_path, llm)
    done = await mgr.run_job((await mgr.create_job("compare X and Y")).job_id)

    assert done.status is JobStatus.FAILED
    assert done.formats is None and done.deliverable_expected is False


async def test_a_document_asked_for_without_a_format_gets_the_deployment_s(
    store, checkpointer, tmp_path
):
    """The one question `$JOBSMITH_REPORT_FORMAT` still answers (#96): which
    format, for a request that wants a document and named none. The node
    resolves it to names, so the record says what was written."""
    mgr = make_engine(store, checkpointer, tmp_path,
                      intent_llm(json.dumps({"document": "requested"})))
    done = await mgr.run_job((await mgr.create_job("write me a report on X")).job_id)

    assert done.formats == ["markdown"] and done.deliverable_expected is True
    assert done.report_path is not None and done.report_path.endswith(".md")


# --------------------------------------------------------- the translation

class FakeGraph:
    """A graph that publishes the updates it was given, LangGraph-shaped."""

    def __init__(self, *updates):
        self.updates = updates
        self.inputs: list[dict] = []

    async def astream(self, state, config=None, stream_mode=None):
        self.inputs.append(state)
        for update in self.updates:
            yield update


async def test_only_the_node_s_own_write_is_announced():
    """The node NAME says whose decision this is, and the key's presence says
    that there was one — the channel was seeded at entry, so a value in it is
    not news. Same discipline as reading a finished step off `cap_<name>`
    rather than off `results` (#53).
    """
    graph = FakeGraph(
        {"document_intent": {"document_formats": []}},
        {"router": {"route": "plan", "document_formats": ["html"]}},
        {"planner": {"plan": {"steps": [], "rationale": "r"}}},
    )
    runner = GraphRunner(graph)

    updates = [u async for u in runner.stream("j1", "q", {}, ["markdown"])]

    assert updates == [FormatsChosen([]), PlanReady({"steps": [], "rationale": "r"})]
    # and what the caller already asked for is what the graph was entered with
    assert graph.inputs[0]["document_formats"] == ["markdown"]


async def test_a_node_that_decided_nothing_announces_that_it_finished():
    """Silence is a decision since #96, so its moment is news: the runner
    says the step finished having written nothing, in both shapes LangGraph
    publishes it — `None` for a node that returned `{}`, and a dict without
    the key. Whether that is silence or a caller who had already spoken is
    the manager's to tell; it holds the record the graph was seeded from."""
    for published in ({}, None):
        runner = GraphRunner(FakeGraph({"document_intent": published}))
        assert [u async for u in runner.stream("j1", "q", {}, None)] == [
            FormatsChosen(None)]


# ------------------------------------------------------- the graph's shape

async def test_a_rejected_query_never_reaches_the_document_step(
    store, checkpointer, tmp_path
):
    """Input validation still gates everything: an empty query costs no call."""
    llm = intent_llm(NAMED_HTML)
    mgr = make_engine(store, checkpointer, tmp_path, llm)
    done = await mgr.run_job((await mgr.create_job("   ")).job_id)

    assert done.terminal_kind == "user_error"
    assert asked(llm) == []


async def test_a_direct_answer_is_filed_when_the_request_asked_for_a_file(
    store, checkpointer, tmp_path
):
    """Why the node sits BEFORE triage, and what that placement costs.

    #84 already honours a format a caller named on a `direct` run: a reader
    who asked for a PDF gets one whether the router sent their sentence to
    the planner or answered it on the spot. A node that only ran on the plan
    path would make that promise depend on how the sentence was triaged —
    which is the second silent decision this issue is about. The price is one
    model call before triage on every run whose caller named no format.
    """
    llm = FakeLLM(
        {"document step": NAMED_HTML,
         "triage": '{"route": "direct", "rationale": "trivial"}'},
        default=ANSWER,
    )
    mgr = make_engine(store, checkpointer, tmp_path, llm)
    done = await mgr.run_job(
        (await mgr.create_job("what can you do? give me an html page")).job_id)

    assert done.plan is None or not done.plan["steps"]
    assert done.report_path is not None and done.report_path.endswith(".html")
