"""The comparison harness runs one task set through the DAG agent and the ReAct
baseline, on the same material, and scores both the same way
(→ docs/design/compiler-v1.md, "Measurement (step −1)", 0208)."""
from __future__ import annotations

import json

import pytest
from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage, ToolMessage

from evals.compare import (
    BASELINE,
    CASES,
    DAG,
    SITES,
    Run,
    compare,
    missing_terms,
    site_note,
    summarize,
)


@pytest.mark.parametrize("answer, missing", [
    ("Alder and CORRAN both run on solar.", ["Glenmoor"]),
    ("Alderney is not a site; Corran and Glenmoor are.", ["Alder"]),
    ("alder, corran, glenmoor.", []),
])
def test_a_term_counts_only_as_a_whole_word_in_any_case(answer, missing):
    assert missing_terms(answer, ("Alder", "Corran", "Glenmoor")) == missing


def test_every_case_can_be_satisfied_by_the_material():
    """A term no note contains would make a case unwinnable for both agents."""
    notes = " ".join(site_note(name, facts) for name, facts in SITES.items())
    for case in CASES:
        assert missing_terms(notes, case.must_mention) == [], case.id


def test_success_needs_an_answer_that_misses_nothing_and_cost_needs_every_price():
    runs = [Run(DAG, "c", 1, answered=True, cost_usd=0.01),
            Run(DAG, "c", 2, answered=True, missing=["Farrow"], cost_usd=0.03),
            Run(BASELINE, "c", 1, answered=True, cost_usd=None),
            Run(BASELINE, "c", 2, answered=False, cost_usd=0.02)]
    table = summarize(runs)
    assert table["c"][DAG]["success"] == 0.5 and table["c"][DAG]["cost_usd"] == 0.02
    assert table["c"][BASELINE]["success"] == 0.5 and table["c"][BASELINE]["cost_usd"] is None


async def test_both_agents_run_the_case_on_the_same_material(monkeypatch):
    for key in ("TAVILY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    everyone = ", ".join(SITES)
    model = ScriptedChatModel(responses=[
        AIMessage("", tool_calls=[{"id": "c1", "name": "search_documents",
                                   "args": {"query": "field site uplink power"}}]),
        AIMessage(f"Sites: {everyone}. Corran is the most constrained.")])
    runs = await compare([CASES[0]], provider="fake", chat_model=model)

    assert [(r.agent, r.case, r.error) for r in runs] == [
        (DAG, "width_every_site", None), (BASELINE, "width_every_site", None)]
    baseline = next(r for r in runs if r.agent == BASELINE)
    assert baseline.success and baseline.status == "done"
    found = json.loads(next(m for m in model.calls[-1] if isinstance(m, ToolMessage)).content)
    assert any("field site" in item["text"].lower() for item in found["items"])
    assert next(r for r in runs if r.agent == DAG).calls > 0
