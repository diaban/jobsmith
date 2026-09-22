"""Default capability pack: research → analysis → critique (LLM-only)."""
from __future__ import annotations

from conftest import FakeLLM, plan_json

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import default_capabilities
from jobsmith.agents.default._step import SUBJECT_ONLY_RULE
from jobsmith.agents.default.analysis import AnalysisCapability
from jobsmith.agents.default.critique import CritiqueCapability
from jobsmith.agents.default.profile import GLOBAL_GENERATOR_PROMPT
from jobsmith.agents.default.read_files import ReadFilesCapability
from jobsmith.agents.default.research import ResearchCapability
from jobsmith.agents.default.sources import Document
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.state import SOURCE_FILES_INPUT_KEY

PACK_SCRIPT = {
    "key aspects": '{"aspects": ["history", "impact"]}',
    "research notes": "Notes: the history is long; the impact is broad.",
    "You are an analyst": "Findings: impact outweighs history.",
    "checking the findings": "- The impact claim: the notes give no numbers for it.",
}


async def test_research_decomposes_and_emits_notes():
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    result = out["results"]["research"]
    assert result["ok"] is True
    assert result["data"]["aspects"] == ["history", "impact"]
    assert "history is long" in result["data"]["notes"]
    assert result["meta"]["aspect_count"] == 2


async def test_research_lenient_on_bad_decompose_json():
    llm = FakeLLM({**PACK_SCRIPT, "key aspects": "not json"})
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    assert out["results"]["research"]["data"]["aspects"] == ["study X"]  # degraded, not failed


async def test_pack_chain_end_to_end(checkpointer):
    llm = FakeLLM({
        "planner": plan_json(
            "research", "analysis", "critique",
            deps={"analysis": ["research"], "critique": ["analysis"]},
        ),
        **PACK_SCRIPT,
        "ONLY the provided": "A sufficiently long final answer built from the pack context.",
    })
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry(default_capabilities(AgentContext(llm))), checkpointer=checkpointer
    )
    out = await graph.ainvoke(
        {"query": "study X in depth", "job_id": "p1"},
        config={"configurable": {"thread_id": "p1"}},
    )
    assert out["terminal_kind"] == "answer"
    assert all(out["results"][name]["ok"] for name in ("research", "analysis", "critique"))

    # analysis actually consumed the research notes
    analyst_call = next(
        c for c in llm.calls if "You are an analyst" in c["messages"][0]["content"]
    )
    assert "history is long" in analyst_call["messages"][1]["content"]

    # merged context follows plan order with each capability's heading
    ctx = out["merged_context"]
    assert ctx.index("# Research notes") < ctx.index("# Analysis")


async def test_the_caveats_reach_the_generator_as_subject_material(checkpointer):
    """#82: the step has a consumer that acts on it again — because what it
    produces changed.

    #58 kept `critique` in the generator's material and labelled the block
    ("Internal review of the work — not of the subject"); #73 withdrew it,
    on the evidence of a run where the model wrote the review instead of the
    answer — a label says what a block *is* and does not stop a weak model
    copying the best-structured thing it can see. What that left was a step
    making an LLM call per run whose output nothing read.

    So it stopped reviewing the work: it checks the findings against the
    material, which is ordinary material about the subject, with the ordinary
    consumer. What is asserted here is the wiring — the block reaches the
    generator under a heading that no longer claims to be about the work, and
    the human's copy is unchanged. Whether the deliverable then copies it is
    not a unit test's question; it is `report_answers_request` and
    `report_reader_facing` in `evals/`, measured before and after.
    """
    llm = FakeLLM({
        "planner": plan_json(
            "research", "analysis", "critique",
            deps={"analysis": ["research"], "critique": ["analysis"]},
        ),
        **PACK_SCRIPT,
        "ONLY the provided": "A sufficiently long final answer built from the pack context.",
    })
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry(default_capabilities(AgentContext(llm))),
        checkpointer=checkpointer,
    )
    out = await graph.ainvoke(
        {"query": "study X in depth", "job_id": "p3"},
        config={"configurable": {"thread_id": "p3"}},
    )
    caveats = PACK_SCRIPT["checking the findings"]
    assert out["results"]["critique"]["ok"] is True
    assert caveats in out["merged_context"]
    assert "Internal review" not in out["merged_context"]
    assert CritiqueCapability.HEADING in out["merged_context"]
    assert "OF THE WORK" not in CritiqueCapability.HEADING.upper()

    generator_call = next(
        c for c in llm.calls if "ONLY the provided" in c["messages"][0]["content"]
    )
    assert caveats in generator_call["messages"][-1]["content"]
    # and the pack's own generator prompt says what to do with such a block:
    # on the statement it bears on, never as a section of its own. (This graph
    # is built without a profile, so the call above carries the CORE default —
    # the rule belongs to the pack that produces the block, not to the
    # framework, which has no notion of a `critique` step.)
    assert "block of caveats" in GLOBAL_GENERATOR_PROMPT
    assert "never collect them into a section" in GLOBAL_GENERATOR_PROMPT

    # the human's copy is unchanged, under the same label
    report = CritiqueCapability(llm).render_report(out["results"]["critique"])
    assert report is not None
    assert CritiqueCapability.HEADING in report and caveats in report


async def test_the_check_sees_the_findings_and_the_notes_behind_them():
    """You cannot say a claim is unsupported while seeing only the claim.

    `SingleStepCapability._material` takes the FIRST upstream that matches —
    a priority chain over restatements of one thing (the analysis, else the
    notes it came from), which is right for a step reasoning onward and wrong
    for one checking one against the other. Same distinction `research`'s
    `GROUNDING` draws against the same base class (#81).
    """
    llm = FakeLLM(PACK_SCRIPT)
    await CritiqueCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {
            "analysis": {"ok": True, "data": {"analysis": "impact outweighs history"}},
            "research": {"ok": True, "data": {"notes": "the history is long [a.md]"}},
        },
    })
    user = llm.calls[0]["messages"][-1]["content"]
    assert "impact outweighs history" in user
    assert "the history is long [a.md]" in user, "the claim without its source is uncheckable"
    assert user.index("[analysis") < user.index("[research")


async def test_the_check_degrades_to_whatever_upstream_survived():
    """One block, no block: the base class's fallback still applies."""
    llm = FakeLLM(PACK_SCRIPT)
    capability = CritiqueCapability(llm)
    await capability.build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {"analysis": {"ok": False, "error": "llm down"},
                    "research": {"ok": True, "data": {"notes": "the notes"}}},
    })
    assert "the notes" in llm.calls[0]["messages"][-1]["content"]

    await capability.build().ainvoke({"query": "study X", "inputs": {}})
    assert "no upstream material" in llm.calls[-1]["messages"][-1]["content"]


async def test_every_material_step_is_told_to_work_on_the_subject_alone():
    """#58, one step upstream of the deliverable.

    A request carries a subject *and* instructions about the document ("a
    printable one-pager, plus a deck for the team"). A step that treats both
    as its material analyses the task: the observed report about chairs grew
    a section of advice on building the deck and an invented slide structure.
    Every prompt of the pack that produces material carries the rule, so a
    capability added later inherits it from `SingleStepCapability.work`.
    """
    llm = FakeLLM(PACK_SCRIPT)
    state = {"query": "study X and make me a deck", "inputs": {}}
    await ResearchCapability(llm).build().ainvoke(state)
    await AnalysisCapability(llm).build().ainvoke(state)
    await CritiqueCapability(llm).build().ainvoke(state)

    systems = [c["messages"][0]["content"] for c in llm.calls]
    assert len(systems) == 4                       # decompose, notes, analysis, critique
    assert all(SUBJECT_ONLY_RULE in s for s in systems)


def test_the_generator_works_on_the_subject_alone_and_still_obeys_the_brief():
    """#73, one step downstream of the material.

    The rule went to every step that produces material (#58) and not to the
    step that produces the document, which duly recited the delivery
    instructions back at its reader ("Document name: …") — noise, since #55
    made those structured fields on the `Job`. The generator is also the one
    consumer that must still *carry out* what the request asks of the
    document, so the shared rule alone would be a lie here: the sentence that
    follows it is what keeps "not part of the subject" from reading as
    "ignore the length you were asked for".
    """
    assert SUBJECT_ONLY_RULE in GLOBAL_GENERATOR_PROMPT
    after = GLOBAL_GENERATOR_PROMPT.split(SUBJECT_ONLY_RULE, 1)[1]
    assert "honour" in after and "never restate it" in after


async def test_pack_degrades_when_one_step_fails(checkpointer):
    class FailingAnalystLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "You are an analyst" in self._system_of(messages):
                raise RuntimeError("llm down")
            return await super().chat(messages, **kwargs)

    llm = FailingAnalystLLM({
        "planner": plan_json("research", "analysis", deps={"analysis": ["research"]}),
        **PACK_SCRIPT,
        "ONLY the provided": "A sufficiently long final answer from research alone.",
    })
    graph = build_agent(
        Deps(llm=llm), CapabilityRegistry(default_capabilities(AgentContext(llm))), checkpointer=checkpointer
    )
    out = await graph.ainvoke(
        {"query": "study X", "job_id": "p2"},
        config={"configurable": {"thread_id": "p2"}},
    )
    assert out["results"]["analysis"]["ok"] is False       # recoverable failure
    assert out["terminal_kind"] == "answer"                # run still completes
    assert "# Analysis" not in out["merged_context"]       # failed step renders nothing


# ------------------------------------------- the grounding reaches the notes


def retrieved(step: str, *docs: tuple[str, str]) -> dict:
    """A finished retrieval step's entry in `results`, as the pack emits it."""
    return {step: {"ok": True, "data": {"documents": [
        {"id": doc_id, "title": doc_id, "source": f"/{doc_id}", "text": text}
        for doc_id, text in docs
    ]}}}


def notes_call(llm: FakeLLM) -> dict:
    """The call that wrote the notes — by its system prompt, either mode.

    Not by a substring of the script: the planner's own prompt renders every
    capability's description, and this one's says "research notes".
    """
    return next(c for c in llm.calls if c["messages"][0]["content"].startswith(
        (ResearchCapability.NOTES_SYSTEM, ResearchCapability.GROUNDED_NOTES_SYSTEM)))


async def test_research_writes_from_what_the_retrieval_found():
    """#81: the step at the far end of `web_search → research` reads it.

    Before this, `investigate` built its messages from the query and the
    aspects alone. The measured cost was a run where 14.7k characters of
    sourced specifications sat in `results` while this step wrote product
    sheets from the model's memory and `analysis` concluded from those.
    """
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        **{"results": retrieved("web_search", ("u1", "the index peaks at 7 GB"))},
    })
    call = notes_call(llm)
    assert "the index peaks at 7 GB" in call["messages"][1]["content"]
    assert "[u1]" in call["messages"][1]["content"], "the id it can quote travels with it"
    # and it is held to the standard of a step with a source, not to the one
    # written for a step with only its memory
    assert ResearchCapability.GROUNDED_NOTES_SYSTEM in call["messages"][0]["content"]
    result = out["results"]["research"]
    assert result["meta"]["grounded_on"] == ["web_search"]


async def test_research_falls_back_to_its_own_knowledge_with_nothing_retrieved():
    """The LLM-only agent is unchanged, to the prompt: no material, no change."""
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({"query": "study X", "inputs": {}})
    call = notes_call(llm)
    assert call["messages"][0]["content"].startswith(ResearchCapability.NOTES_SYSTEM)
    assert "Retrieved material" not in call["messages"][1]["content"]
    assert out["results"]["research"]["meta"]["grounded_on"] == []


async def test_every_retrieval_step_reaches_it_not_only_the_first():
    """The rule that is NOT `_material`'s (`_step.py`), and why.

    There, the upstream tuple is a priority chain over restatements of one
    thing — the analysis, else the notes it was drawn from — and reading the
    second as well would hand the model the same content twice. Here the
    entries are *sources*: a file the user named and what the web says today
    are complementary, and dropping either because the other matched first
    loses material nothing downstream can recover.
    """
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {
            **retrieved("read_files", ("mine.md", "the note the user handed over")),
            **retrieved("web_search", ("u1", "what the web says today")),
        },
    })
    material = notes_call(llm)["messages"][1]["content"]
    assert "the note the user handed over" in material
    assert "what the web says today" in material
    assert "could NOT be read" not in material, "nothing was refused, so nothing is declared"
    assert out["results"]["research"]["meta"]["grounded_on"] == ["read_files", "web_search"]


async def test_a_failed_retrieval_step_is_not_material():
    llm = FakeLLM(PACK_SCRIPT)
    out = await ResearchCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {"documents": {"ok": False, "error": "nothing matched"}},
    })
    assert "Retrieved material" not in notes_call(llm)["messages"][1]["content"]
    assert out["results"]["research"]["meta"]["grounded_on"] == []


async def test_a_file_that_could_not_be_read_travels_with_the_material():
    """`read_files`'s own rule, one step earlier than it was written for.

    Its module docstring already says it: "a refusal is material, not
    silence" — one unreadable file among three does not fail the step, and
    the refusal goes into the generation context "because a model told
    nothing about the missing file writes confidently over the hole". #81 put
    a step that writes *sourced* notes between the two, and the refusal has
    to cross it or the deliverable inherits a gap nothing in the run
    mentions.
    """
    llm = FakeLLM(PACK_SCRIPT)
    await ResearchCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {"read_files": {"ok": True, "data": {
            "documents": [{"id": "a.md", "title": "a.md", "text": "the file that opened"}],
            "unreadable": ["'gone.md': no such file"],
        }}},
    })
    call = notes_call(llm)
    user = call["messages"][1]["content"]
    assert "gone.md" in user
    # under its own label, after the material and not inside it: what is
    # missing is not something to reason from, it is something to declare
    assert user.index("the file that opened") < user.index("could NOT be read")
    assert "gone.md" not in user.split("could NOT be read")[0]
    # and the prompt says what to do with it
    assert "never write it up from your own knowledge" in call["messages"][0]["content"]


async def test_a_search_that_found_nothing_invents_no_refusal():
    """The two ports are not symmetrical, and this is where that shows.

    A file the user named and could not be opened is missing from the answer
    they expect. A query that matched nothing returned nothing — the step
    fails, and "the sources are silent about X" is not a document anyone
    asked for. So `REFUSALS` names only the steps that were POINTED AT
    something — `read_files` (a file) and, since #74, `prior_jobs` (a run) —
    rather than giving the search an equivalent it does not have.
    """
    named = {step for step, _ in ResearchCapability.REFUSALS}
    assert named == {"read_files", "prior_jobs"}
    assert ("read_files", "unreadable") in ResearchCapability.REFUSALS
    llm = FakeLLM(PACK_SCRIPT)
    await ResearchCapability(llm).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {
            **retrieved("web_search", ("u1", "what the web says")),
            # a shape a search could produce, and must not be read as a refusal
            "documents": {"ok": True, "data": {"documents": [], "unreadable": ["ignored"]}},
        },
    })
    assert "could NOT be read" not in notes_call(llm)["messages"][1]["content"]


async def test_the_material_is_bounded_and_says_where_it_was_cut():
    """Since #75 a single web page can be 8 000 characters and there can be
    ten of them, and every one of those blocks is re-sent to the generator
    afterwards. The budget is shared out so no document is dropped whole, and
    a cut is written into the text — a document silently shortened reads to
    the model as a complete one."""
    llm = FakeLLM(PACK_SCRIPT)
    await ResearchCapability(llm, max_material_chars=200).build().ainvoke({
        "query": "study X", "inputs": {},
        "results": {
            **retrieved("documents", ("a", "alpha " * 400)),
            **retrieved("web_search", ("b", "beta " * 400)),
        },
    })
    material = notes_call(llm)["messages"][1]["content"]
    assert "alpha" in material and "beta" in material     # neither is dropped
    assert material.count("truncated") == 2
    assert len(material) < 700, "the bound holds, notes plus one line per cut"


async def test_the_result_says_which_kind_of_notes_it_holds():
    """A reader of the job record can tell a sourced figure from a recalled
    one — the whole reason the two modes are not one prompt (#81)."""
    llm = FakeLLM(PACK_SCRIPT)
    capability = ResearchCapability(llm)
    grounded = (await capability.build().ainvoke({
        "query": "study X", "inputs": {},
        "results": retrieved("documents", ("a.md", "a measured figure")),
    }))["results"]["research"]
    recalled = (await capability.build().ainvoke(
        {"query": "study X", "inputs": {}}))["results"]["research"]

    for rendered in (capability.render_context(grounded), capability.render_report(grounded)):
        assert rendered is not None and "retrieved by documents" in rendered
    for rendered in (capability.render_context(recalled), capability.render_report(recalled)):
        assert rendered is not None and "own knowledge" in rendered


async def test_the_planner_is_told_the_step_reads_what_was_retrieved():
    """The description is the whole of what the planner reads. It used to say
    'from the model's own knowledge (no external sources)', which was true and
    is now false — a planner choosing on it would keep the step away from the
    material it is supposed to read."""
    description = ResearchCapability.spec.description
    assert "no external sources" not in description
    assert all(name in description for name in ("read_files", "documents", "web_search"))


async def test_the_edge_the_plan_draws_carries_the_material(checkpointer):
    """End to end, on the chain the deliverable renders as a graph.

    `depends_on` is a scheduling constraint and every report draws it as a
    data flow; on `read_files → research → analysis` it now is one.
    """
    llm = FakeLLM({
        "planner": plan_json(
            "read_files", "research", "analysis",
            deps={"research": ["read_files"], "analysis": ["research"]},
        ),
        **PACK_SCRIPT,
        "research notes": "Notes: the note says the index peaks at 7 GB.",
        "ONLY the provided": "A sufficiently long final answer built from the pack context.",
    })
    registry = CapabilityRegistry([
        ReadFilesCapability(_HandedOver("the index peaks at 7 GB")),
        *default_capabilities(AgentContext(llm)),
    ])
    graph = build_agent(Deps(llm=llm), registry, checkpointer=checkpointer)
    out = await graph.ainvoke(
        {"query": "study the note", "job_id": "g1",
         "inputs": {SOURCE_FILES_INPUT_KEY: ["note.md"]}},
        config={"configurable": {"thread_id": "g1"}},
    )
    assert [s["capability"] for s in out["plan"]["steps"]] == [
        "read_files", "research", "analysis"]
    # the retrieved text reached the step the plan points at, not only the
    # generator three steps later
    notes_input = notes_call(llm)["messages"][1]["content"]
    assert "the index peaks at 7 GB" in notes_input
    assert out["results"]["research"]["meta"]["grounded_on"] == ["read_files"]


class _HandedOver:
    """A `DocumentReader` over one canned text — no filesystem, no port leak."""

    def __init__(self, text: str):
        self.text = text

    async def read(self, ref: str):
        return Document(id=ref, text=self.text, title=ref, source=f"/{ref}")
