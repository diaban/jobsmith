"""The default pack: research → analysis → critique, and what each step reads.

A prompt is asserted only through the named rule it must carry
(`SUBJECT_ONLY_RULE`, `CAVEATS_RULE`, `BRIEF_RULE`, `DELIVERED_FILES_RULE`,
`UNREADABLE_RULE`) — never by its wording, which the evals judge (→ 0003, 0110).
"""
from __future__ import annotations

import pytest
from conftest import FakeLLM, plan_json
from support import PACK_SCRIPT, notes_call

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import default_capabilities
from jobsmith.agents.default._step import SUBJECT_ONLY_RULE
from jobsmith.agents.default.analysis import AnalysisCapability
from jobsmith.agents.default.critique import CritiqueCapability
from jobsmith.agents.default.profile import (
    BRIEF_RULE,
    CAVEATS_RULE,
    DELIVERED_FILES_RULE,
    GLOBAL_GENERATOR_PROMPT,
)
from jobsmith.agents.default.read_files import ReadFilesCapability
from jobsmith.agents.default.research import UNREADABLE_RULE, ResearchCapability
from jobsmith.agents.default.sources import Document
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.state import SOURCE_FILES_INPUT_KEY

CHAIN = {"analysis": ["research"], "critique": ["analysis"]}
FINAL = "A sufficiently long final answer built from the pack context."


def pack_llm(*steps: str, deps=None, llm_class=FakeLLM, **script: str) -> FakeLLM:
    steps = steps or ("research", "analysis", "critique")
    return llm_class({"planner": plan_json(*steps, deps=CHAIN if deps is None else deps),
                      **PACK_SCRIPT, "ONLY the provided": FINAL, **script})


async def run_pack(checkpointer, llm, query="study X in depth", *, before=(), inputs=None):
    """The whole graph over the default pack (plus any capability in `before`)."""
    registry = CapabilityRegistry([*before, *default_capabilities(AgentContext(llm))])
    graph = build_agent(Deps(llm=llm), registry, checkpointer=checkpointer)
    return await graph.ainvoke({"query": query, "job_id": "p1", "inputs": inputs or {}},
                               config={"configurable": {"thread_id": "p1"}})


async def research(results=None, **ctor) -> tuple[FakeLLM, dict]:
    """Run `research` alone over these upstream results; its llm and its result."""
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm, **ctor).build().ainvoke(
        {"query": "study X", "inputs": {}, "results": results or {}})
    return llm, out["results"]["research"]


def retrieved(step: str, *docs: tuple[str, str]) -> dict:
    """A finished retrieval step's entry in `results`, as the pack emits it."""
    return {step: {"ok": True, "data": {"documents": [
        {"id": doc_id, "title": doc_id, "source": f"/{doc_id}", "text": text}
        for doc_id, text in docs
    ]}}}


# ------------------------------------------------------------ the chain

async def test_the_chain_runs_and_each_step_reads_the_one_before(checkpointer):
    llm = pack_llm()
    out = await run_pack(checkpointer, llm)

    assert out["terminal_kind"] == "answer"
    assert all(out["results"][n]["ok"] for n in ("research", "analysis", "critique"))
    analyst = next(c for c in llm.calls if "You are an analyst" in c["messages"][0]["content"])
    assert "history is long" in analyst["messages"][1]["content"]
    ctx = out["merged_context"]
    assert ctx.index("# Research notes") < ctx.index("# Analysis")   # plan order


async def test_a_failed_step_degrades_the_run_rather_than_failing_it(checkpointer):
    class FailingAnalyst(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "You are an analyst" in self._system_of(messages):
                raise RuntimeError("llm down")
            return await super().chat(messages, **kwargs)

    llm = pack_llm("research", "analysis", deps={"analysis": ["research"]},
                   llm_class=FailingAnalyst)
    out = await run_pack(checkpointer, llm, "study X")
    assert out["results"]["analysis"]["ok"] is False
    assert out["terminal_kind"] == "answer"
    assert "# Analysis" not in out["merged_context"]


async def test_research_decomposes_and_degrades_on_bad_json():
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    result = out["results"]["research"]
    assert result["data"]["aspects"] == ["history", "impact"]
    assert result["meta"]["aspect_count"] == 2 and "history is long" in result["data"]["notes"]

    llm = FakeLLM({**PACK_SCRIPT, "key aspects": "not json"})
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    assert out["results"]["research"]["data"]["aspects"] == ["study X"]   # degraded, not failed


# ------------------------------------------------------------ the subject, not the task → 0058, 0073

async def test_every_material_step_carries_the_subject_only_rule():
    llm = FakeLLM(PACK_SCRIPT)
    state = {"query": "study X and make me a deck", "inputs": {}}
    for capability in (ResearchCapability, AnalysisCapability, CritiqueCapability):
        await capability(llm).build().ainvoke(state)

    systems = [c["messages"][0]["content"] for c in llm.calls]
    assert len(systems) == 4                       # decompose, notes, analysis, critique
    assert all(SUBJECT_ONLY_RULE in s for s in systems)


def test_the_generator_carries_the_rule_and_still_obeys_the_brief():
    assert SUBJECT_ONLY_RULE + BRIEF_RULE in GLOBAL_GENERATOR_PROMPT


def test_the_generator_says_how_to_read_the_files_it_is_listed():
    """→ 0077, 0126: a listed file other than the answer is delivered apart."""
    assert DELIVERED_FILES_RULE in GLOBAL_GENERATOR_PROMPT


# ------------------------------------------------------------ the critique → 0082

async def test_the_caveats_reach_the_generator_as_subject_material(checkpointer):
    """Wired, under a heading about the subject; whether the deliverable uses
    them well is the evals' question, not this one's."""
    llm = pack_llm()
    out = await run_pack(checkpointer, llm)
    caveats = PACK_SCRIPT["checking the findings"]

    assert caveats in out["merged_context"] and CritiqueCapability.HEADING in out["merged_context"]
    generator = next(c for c in llm.calls if "ONLY the provided" in c["messages"][0]["content"])
    assert caveats in generator["messages"][-1]["content"]
    assert CAVEATS_RULE in GLOBAL_GENERATOR_PROMPT       # the pack says where they go
    report = CritiqueCapability(llm).render_report(out["results"]["critique"])
    assert report is not None and CritiqueCapability.HEADING in report and caveats in report


@pytest.mark.parametrize(("results", "sees"), [
    ({"analysis": {"ok": True, "data": {"analysis": "impact outweighs history"}},
      "research": {"ok": True, "data": {"notes": "the history is long [a.md]"}}},
     ["impact outweighs history", "the history is long [a.md]"]),
    ({"analysis": {"ok": False, "error": "llm down"},
      "research": {"ok": True, "data": {"notes": "the notes"}}},
     ["the notes"]),
    ({}, ["no upstream material"]),
])
async def test_the_check_reads_the_findings_and_the_notes_behind_them(results, sees):
    """Both, in that order — not `_material`'s first match — degrading to
    whatever survived."""
    llm = FakeLLM(PACK_SCRIPT)
    await CritiqueCapability(llm).build().ainvoke(
        {"query": "study X", "inputs": {}, "results": results})
    user = llm.calls[0]["messages"][-1]["content"]
    assert all(text in user for text in sees)
    assert [user.index(t) for t in sees] == sorted(user.index(t) for t in sees)


# ------------------------------------------------------------ research reads the retrieval → 0081

async def test_research_writes_from_what_the_retrieval_found():
    llm, result = await research(retrieved("web_search", ("u1", "the index peaks at 7 GB")))
    call = notes_call(llm)
    assert "the index peaks at 7 GB" in call["messages"][1]["content"]
    assert "[u1]" in call["messages"][1]["content"], "the id it can quote travels with it"
    assert ResearchCapability.GROUNDED_NOTES_SYSTEM in call["messages"][0]["content"]
    assert result["meta"]["grounded_on"] == ["web_search"]


@pytest.mark.parametrize("results", [None, {"documents": {"ok": False, "error": "no match"}}],
                         ids=["nothing-retrieved", "retrieval-failed"])
async def test_with_no_material_research_writes_from_its_own_knowledge(results):
    llm, result = await research(results)
    call = notes_call(llm)
    assert call["messages"][0]["content"].startswith(ResearchCapability.NOTES_SYSTEM)
    assert "Retrieved material" not in call["messages"][1]["content"]
    assert result["meta"]["grounded_on"] == []


async def test_every_retrieval_step_reaches_it_not_only_the_first():
    """Sources are complementary, not restatements: unlike `_material`, all of them."""
    llm, result = await research({
        **retrieved("read_files", ("mine.md", "the note the user handed over")),
        **retrieved("web_search", ("u1", "what the web says today")),
    })
    material = notes_call(llm)["messages"][1]["content"]
    assert "the note the user handed over" in material and "what the web says today" in material
    assert "could NOT be read" not in material, "nothing was refused, so nothing is declared"
    assert result["meta"]["grounded_on"] == ["read_files", "web_search"]


async def test_a_file_that_could_not_be_read_is_declared_after_the_material():
    llm, _ = await research({"read_files": {"ok": True, "data": {
        "documents": [{"id": "a.md", "title": "a.md", "text": "the file that opened"}],
        "unreadable": ["'gone.md': no such file"],
    }}})
    call = notes_call(llm)
    user = call["messages"][1]["content"]
    assert user.index("the file that opened") < user.index("could NOT be read")
    assert "gone.md" in user.split("could NOT be read")[1]
    assert UNREADABLE_RULE in call["messages"][0]["content"]


async def test_only_a_step_pointed_at_something_can_refuse():
    """A file or a run that was named can be missing; a search that matched
    nothing has no refusal to declare."""
    assert {step for step, _ in ResearchCapability.REFUSALS} == {"read_files", "prior_jobs"}
    llm, _ = await research({
        **retrieved("web_search", ("u1", "what the web says")),
        "documents": {"ok": True, "data": {"documents": [], "unreadable": ["ignored"]}},
    })
    assert "could NOT be read" not in notes_call(llm)["messages"][1]["content"]


async def test_the_material_is_bounded_and_says_where_it_was_cut():
    """The budget is shared so no document is dropped whole; each cut is written. → 0075"""
    llm, _ = await research({**retrieved("documents", ("a", "alpha " * 400)),
                             **retrieved("web_search", ("b", "beta " * 400))},
                            max_material_chars=200)
    material = notes_call(llm)["messages"][1]["content"]
    assert "alpha" in material and "beta" in material
    assert material.count("truncated") == 2
    assert len(material) < 700, "the bound holds, notes plus one line per cut"


async def test_the_result_says_which_kind_of_notes_it_holds():
    capability = ResearchCapability(FakeLLM(PACK_SCRIPT))
    _, grounded = await research(retrieved("documents", ("a.md", "a measured figure")))
    _, recalled = await research()
    for rendered in (capability.render_context(grounded), capability.render_report(grounded)):
        assert rendered is not None and "retrieved by documents" in rendered
    for rendered in (capability.render_context(recalled), capability.render_report(recalled)):
        assert rendered is not None and "own knowledge" in rendered


def test_the_planner_is_told_which_steps_research_reads():
    description = ResearchCapability.spec.description
    assert all(name in description for name in ("read_files", "documents", "web_search"))


class _HandedOver:
    """A `DocumentReader` over one canned text — no filesystem."""

    def __init__(self, text: str):
        self.text = text

    async def read(self, ref: str):
        return Document(id=ref, text=self.text, title=ref, source=f"/{ref}")


async def test_the_edge_the_plan_draws_carries_the_material(checkpointer):
    """`read_files → research → analysis`: `depends_on` is a data flow."""
    llm = pack_llm("read_files", "research", "analysis",
                   deps={"research": ["read_files"], "analysis": ["research"]},
                   **{"research notes": "Notes: the note says the index peaks at 7 GB."})
    out = await run_pack(checkpointer, llm, "study the note",
                         before=[ReadFilesCapability(_HandedOver("the index peaks at 7 GB"))],
                         inputs={SOURCE_FILES_INPUT_KEY: ["note.md"]})

    assert [s["capability"] for s in out["plan"]["steps"]] == [
        "read_files", "research", "analysis"]
    assert "the index peaks at 7 GB" in notes_call(llm)["messages"][1]["content"]
    assert out["results"]["research"]["meta"]["grounded_on"] == ["read_files"]
