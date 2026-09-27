"""The document step (`dag/document.py`): what file the request asked for.

It fills silence and never overrides, writes only to state, chooses only
what this deployment renders, and fails open to silence (= no file). → 0090, 0096
What a job then delivers is `test_deliverable.py`.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM
from langgraph.graph import END, START, StateGraph
from support import SlowEcho, make_manager

from jobsmith.dag.deps import Deps
from jobsmith.dag.document import DocumentIntent
from jobsmith.dag.profile import FILE_REQUEST_RULE
from jobsmith.dag.state import AgentState
from jobsmith.engine.facts import FACT_KEY
from jobsmith.engine.models import JobStatus

RENDERABLE = ("html", "markdown", "pdf")
NAMED_HTML = json.dumps({"document": "named", "formats": ["html"]})


def node(reply: str | dict, *, formats=RENDERABLE, default=(), llm=None) -> DocumentIntent:
    if isinstance(reply, dict):
        reply = json.dumps(reply)
    return DocumentIntent(Deps(llm=llm or FakeLLM({"document step": reply})), formats,
                          default_formats=default)


def asked(llm: FakeLLM) -> list[dict]:
    """The calls that went to the document step, and no others."""
    return [c for c in llm.calls if "document step" in c["messages"][0].get("content", "")]


# ------------------------------------------------------------ reading the reply

# `{}` is silence (no file, and `None` stays fillable); `[]` is "no file" said.
NOTHING, NO_FILE = {}, {"document_formats": []}


def _id(value) -> str:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, tuple)) else str(value)


@pytest.mark.parametrize(("reply", "default", "decided"), ids=_id, argvalues=[
    ({"document": "named", "formats": ["html"]}, (), {"document_formats": ["html"]}),
    ({"document": "unspecified"}, (), NOTHING),
    ({"document": "none"}, (), NO_FILE),
    # a document wanted, no format named: the deployment's default, as names
    ({"document": "requested"}, ("html", "markdown"), {"document_formats": ["html", "markdown"]}),
    ({"document": "requested"}, (), NOTHING),
    ({"document": "requested"}, ("docx",), NOTHING),
    ({"document": "requested", "formats": ["docx"]}, ("markdown",),
     {"document_formats": ["markdown"]}),
    # names beat the label: 1 and 3 calls in 12 on gpt-5-nano (→ 0110)
    ({"document": "requested", "formats": ["html"]}, ("markdown",), {"document_formats": ["html"]}),
    ({"document": "html", "formats": ["html"]}, ("markdown",), {"document_formats": ["html"]}),
    ({"document": "HTML"}, ("markdown",), {"document_formats": ["html"]}),
    # ...but a list never turns "no file" or "unsure" into a file
    ({"document": "none", "formats": ["html"]}, ("markdown",), NO_FILE),
    ({"document": "unspecified", "formats": ["html"]}, ("markdown",), NOTHING),
    # what it cannot use degrades to silence: it never refuses
    ("sure, a PDF!", (), NOTHING),
    ({"document": "maybe"}, (), NOTHING),
    ({"document": "named"}, (), NOTHING),
    ({"document": "named", "formats": ["docx"]}, (), NOTHING),
    ({"document": "named", "formats": "pdf"}, (), NOTHING),
])
async def test_the_reply_becomes_a_decision_or_silence(reply, default, decided):
    assert await node(reply, default=default).run({"query": "compare X and Y"}) == decided


async def test_a_provider_that_blows_up_is_silence():
    class Exploding:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("provider down")

    assert await node("", llm=Exploding()).run({"query": "as an html page"}) == {}


def test_the_prompt_offers_only_what_this_deployment_can_render():
    prompt = node("", formats=("html", "markdown")).system_prompt()
    assert "- html" in prompt and "- markdown" in prompt and "- pdf" not in prompt


async def test_a_file_asked_for_in_words_is_a_document_asked_for():
    """"save it to a file" names no format and still asks for one, whatever
    else the request asks; the word alone asks for nothing (→ 0125)."""
    llm = FakeLLM({"document step": NAMED_HTML})
    await node("", llm=llm).run({"query": "summarise X, and save that as a file"})
    assert FILE_REQUEST_RULE in asked(llm)[0]["messages"][0]["content"]


# ------------------------------------------------------------ asked nothing

@pytest.mark.parametrize(("seeded", "formats"), [
    (["markdown"], RENDERABLE),     # the caller named formats
    ([], RENDERABLE),               # the caller asked for no file: `is not None`, not truthiness
    (None, ()),                     # this deployment renders nothing
])
async def test_it_asks_nothing_when_there_is_nothing_to_decide(seeded, formats):
    llm = FakeLLM({"document step": NAMED_HTML})
    state = {"query": "as an html page"}
    if seeded is not None:
        state["document_formats"] = seeded
    assert await node("", formats=formats, llm=llm).run(state) == {}
    assert asked(llm) == []


# ------------------------------------------------------------ what the job hears

async def facts_of(step: DocumentIntent, state: dict) -> list[dict]:
    """What the node tells the job running it, run as a graph would run it."""
    g = StateGraph(AgentState)
    g.add_node("document_intent", step.run)
    g.add_edge(START, "document_intent")
    g.add_edge("document_intent", END)
    return [c async for c in g.compile().astream(state, stream_mode="custom")]


@pytest.mark.parametrize(("reply", "seeded", "heard"), ids=_id, argvalues=[
    ({"document": "named", "formats": ["html"]}, None, ["html"]),
    ({"document": "none"}, None, []),                   # "no file", said
    ({"document": "unspecified"}, None, None),          # silence
    ({"document": "named", "formats": ["html"]}, ["markdown"], None),   # a caller spoke
])
async def test_the_job_hears_the_question_settled_whichever_way(reply, seeded, heard):
    """A list, `[]`, or `None` for "wrote nothing" — silence and a caller who
    had spoken alike: the job tells those two apart from the record it seeded
    the run with (→ 0090, 0096)."""
    state = {"query": "a page about chairs"}
    if seeded is not None:
        state["document_formats"] = seeded
    assert await facts_of(node(reply), state) == [{FACT_KEY: "formats", "value": heard}]


# ------------------------------------------------------------ in the graph

def engine(store, checkpointer, tmp_path, reply: str, **script):
    llm = FakeLLM({"document step": reply, **script}, default="A sufficiently long answer.")
    mgr = make_manager(store, checkpointer, tmp_path, caps=[SlowEcho("alpha")], llm=llm,
                       document_formats=RENDERABLE)
    return mgr, llm


async def test_a_rejected_query_never_reaches_the_document_step(store, checkpointer, tmp_path):
    mgr, llm = engine(store, checkpointer, tmp_path, NAMED_HTML)
    done = await mgr.run_job((await mgr.create_job("   ")).job_id)
    assert done.terminal_kind == "user_error" and asked(llm) == []


async def test_it_runs_before_triage_so_a_direct_answer_gets_its_file(
    store, checkpointer, tmp_path
):
    mgr, _ = engine(store, checkpointer, tmp_path, NAMED_HTML,
                    **{"triage": '{"route": "direct", "rationale": "trivial"}'})
    done = await mgr.run_job((await mgr.create_job("what can you do? as html")).job_id)
    assert done.status is JobStatus.DONE and not (done.plan or {}).get("steps")
    assert done.report_path is not None and done.report_path.endswith(".html")
