"""The document step (`core/document.py`): what file the request asked for.

It fills silence and never overrides, writes only to state, chooses only
what this deployment renders, and fails open to silence (= no file). → 0090, 0096
What a job then delivers is `test_deliverable.py`.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLM
from support import SlowEcho, make_manager

from jobsmith.core.deps import Deps
from jobsmith.core.document import DocumentIntent
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.runner import FormatsChosen, GraphRunner, PlanReady

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


# ------------------------------------------------------------ the runner's translation

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
    """Read off the node name, never the seeded channel (→ 0053)."""
    graph = FakeGraph(
        {"document_intent": {"document_formats": []}},
        {"router": {"route": "plan", "document_formats": ["html"]}},
        {"planner": {"plan": {"steps": [], "rationale": "r"}}},
    )
    updates = [u async for u in GraphRunner(graph).stream("j1", "q", {}, ["markdown"])]

    assert updates == [FormatsChosen([]), PlanReady({"steps": [], "rationale": "r"})]
    assert graph.inputs[0]["document_formats"] == ["markdown"]


@pytest.mark.parametrize("published", [{}, None])
async def test_a_node_that_decided_nothing_still_announces_it(published):
    runner = GraphRunner(FakeGraph({"document_intent": published}))
    assert [u async for u in runner.stream("j1", "q", {}, None)] == [FormatsChosen(None)]


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
