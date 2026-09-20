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
- **it is fail-open**, so every error lands on exactly the behaviour that
  existed before it.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, plan_json
from test_jobs import SlowEcho

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


async def test_a_request_that_names_no_format_writes_nothing():
    """Silence, not a default: whatever already decides keeps deciding.

    That is the whole of the fail-open contract — this answer is the state
    every run was in before the node existed.
    """
    node = make_node(UNSPECIFIED)
    assert await node.run({"query": "compare X and Y"}) == {}


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
    )
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts")


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


async def test_a_silent_request_still_decides_the_way_it_always_did(
    store, checkpointer, tmp_path
):
    """Fail-open, end to end: the plan shape answers what the request did not.

    The two rules compose on disjoint questions (#84 + #90) — the node reads
    what a request can say, the run's shape answers the rest — so a run the
    node said nothing about is byte-for-byte the run that existed before it.
    """
    mgr = make_engine(store, checkpointer, tmp_path, intent_llm(UNSPECIFIED))
    job = await mgr.create_job("compare X and Y")

    done = await mgr.run_job(job.job_id)
    assert done.formats is None
    assert done.deliverable_expected is True
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


async def test_a_node_that_decided_nothing_announces_nothing():
    runner = GraphRunner(FakeGraph({"document_intent": {}}))
    assert [u async for u in runner.stream("j1", "q", {}, None)] == []


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
