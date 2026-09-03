"""Profile-driven validators: rule folding + default/banking behaviors."""
from __future__ import annotations

from jobsmith.agents.banking.profile import BANKING_PROFILE, rule_citations_when_search
from jobsmith.core.profile import AgentProfile
from jobsmith.core.validate import InputValidator, OutputValidator


async def test_default_profile_rejects_empty_query_in_english():
    out = await InputValidator(AgentProfile()).run({"query": "  "})
    assert out["input_valid"] is False
    assert out["user_error_message"] == "Your query is empty."


async def test_banking_profile_rejects_empty_query_in_french():
    out = await InputValidator(BANKING_PROFILE).run({"query": "  "})
    assert out["user_error_message"] == "Votre requête est vide."


async def test_output_rules_collect_all_issues():
    out = await OutputValidator(BANKING_PROFILE).run({
        "draft_answer": "",
        "results": {"search": {"ok": True, "data": {}}},
    })
    assert set(out["validation_issues"]) == {"empty_answer", "answer_too_short", "missing_citations"}
    assert out["output_valid"] is False


def test_citation_rule_only_fires_on_successful_search():
    state = {"draft_answer": "long answer without citations at all, definitely"}
    assert rule_citations_when_search(state) is None                       # no search ran
    state["results"] = {"search": {"ok": False}}
    assert rule_citations_when_search(state) is None                       # search failed
    state["results"] = {"search": {"ok": True}}
    assert rule_citations_when_search(state) == "missing_citations"        # fires
    state["draft_answer"] = "answer citing [doc_1]"
    assert rule_citations_when_search(state) is None                       # citation present


async def test_custom_max_refine_flows_from_profile():
    profile = AgentProfile(max_refine=5)
    out = await InputValidator(profile).run({"query": "hello"})
    assert out["max_refine"] == 5
