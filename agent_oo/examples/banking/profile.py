"""Banking profile: prompts, French user-facing messages, and domain rules.

This is the domain surface that used to be hardcoded in the framework.
"""
from __future__ import annotations

from ...core.profile import (
    AgentProfile,
    rule_min_answer_len,
    rule_nonempty_answer,
)
from ...core.state import AgentState

MAX_QUERY_LEN = 4000

BANKING_PLANNER_TEMPLATE = """You are the planner of a banking-assistant agent.
Given a banker's query, decide which of the available capabilities are needed
and in what order. Output a JSON object describing a DAG.

Available capabilities:
{capabilities}

Schema:
{{
  "steps": [
    {{"capability": "<name>", "depends_on": [<other capability names>]}}
  ],
  "rationale": "<short explanation>"
}}

Rules:
- Include only capabilities that are actually needed.
- depends_on values must refer to other steps in the same plan.
- The DAG must be acyclic.
- Return ONLY the JSON object, no prose, no markdown fences."""

BANKING_GENERATOR_PROMPT = (
    "You are a banking assistant. Answer the banker's query using ONLY the "
    "provided context. Cite sources inline as [doc_id] when relevant. "
    "If the context is insufficient, say so explicitly. Be concise and precise."
)


# ---------- Rules (French user-facing messages) ----------

def rule_nonempty_query_fr(state: AgentState) -> str | None:
    return "Votre requête est vide." if not (state.get("query") or "").strip() else None


def rule_max_query_len_fr(state: AgentState) -> str | None:
    if len((state.get("query") or "").strip()) > MAX_QUERY_LEN:
        return f"Requête trop longue (max {MAX_QUERY_LEN} caractères)."
    return None


def rule_citations_when_search(state: AgentState) -> str | None:
    """Search succeeded but the draft cites nothing → refine."""
    draft = state.get("draft_answer") or ""
    search = state.get("results", {}).get("search")
    if search and search.get("ok") and "[" not in draft and "(" not in draft:
        return "missing_citations"
    return None


BANKING_PROFILE = AgentProfile(
    planner_prompt_template=BANKING_PLANNER_TEMPLATE,
    generator_system_prompt=BANKING_GENERATOR_PROMPT,
    user_error_message="Une erreur est survenue.",
    escalation_message="Votre demande a été transmise à un analyste. Vous serez recontacté.",
    input_rules=(rule_nonempty_query_fr, rule_max_query_len_fr),
    output_rules=(rule_nonempty_answer, rule_min_answer_len(), rule_citations_when_search),
)
