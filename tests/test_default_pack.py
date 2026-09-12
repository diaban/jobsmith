"""Default capability pack: research → analysis → critique (LLM-only)."""
from __future__ import annotations

from conftest import FakeLLM, plan_json

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import default_capabilities
from jobsmith.agents.default._step import SUBJECT_ONLY_RULE
from jobsmith.agents.default.analysis import AnalysisCapability
from jobsmith.agents.default.critique import CritiqueCapability
from jobsmith.agents.default.profile import GLOBAL_GENERATOR_PROMPT
from jobsmith.agents.default.research import ResearchCapability
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry

PACK_SCRIPT = {
    "key aspects": '{"aspects": ["history", "impact"]}',
    "research notes": "Notes: the history is long; the impact is broad.",
    "You are an analyst": "Findings: impact outweighs history.",
    "critical reviewer": "Gap: no numbers back the impact claim.",
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


async def test_the_internal_review_does_not_reach_the_generator(checkpointer):
    """#73 reverses #58's second decision, on the evidence of one run.

    #58 kept `critique` in the generator's material and labelled the block
    ("Internal review of the work — not of the subject"), the generator's
    prompt saying to use it as evidence and never as voice. A later run
    falsified that: handed hedged notes and one impeccably structured
    methodology review, the model wrote the review — the deliverable's
    sections matched `critique`'s nearly one for one, and the sourced
    specifications upstream never crossed. A label says what a block *is*; it
    does not stop a model copying the best-structured thing it can see.

    So the step still runs, still reports ok, still reaches the human through
    `render_report` — and contributes nothing the generator can copy.
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
    review = PACK_SCRIPT["critical reviewer"]
    assert out["results"]["critique"]["ok"] is True         # it ran, and it is kept
    assert review not in out["merged_context"]              # and the generator never saw it
    assert "Internal review" not in out["merged_context"]

    generator_call = next(
        c for c in llm.calls if "ONLY the provided" in c["messages"][0]["content"]
    )
    assert review not in generator_call["messages"][-1]["content"]

    # the human still gets it, under the label that says what it is
    report = CritiqueCapability(llm).render_report(out["results"]["critique"])
    assert report is not None
    assert CritiqueCapability.HEADING in report and review in report


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
