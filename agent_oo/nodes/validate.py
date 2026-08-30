"""Input / Output validators — object-oriented version."""
from __future__ import annotations

from ..deps import Deps
from ..state import AgentState


class InputValidator:
    MAX_QUERY_LEN = 4000

    def __init__(self, deps: Deps):
        self.deps = deps  # kept for symmetry / future compliance checks

    async def run(self, state: AgentState) -> dict:
        query = (state.get("query") or "").strip()
        if not query:
            return {
                "input_valid": False,
                "rejection_reason": "empty_query",
                "user_error_message": "Votre requête est vide.",
            }
        if len(query) > self.MAX_QUERY_LEN:
            return {
                "input_valid": False,
                "rejection_reason": "query_too_long",
                "user_error_message": (
                    f"Requête trop longue (max {self.MAX_QUERY_LEN} caractères)."
                ),
            }
        return {"input_valid": True, "max_refine": state.get("max_refine", 2)}


class OutputValidator:
    MIN_ANSWER_LEN = 20

    def __init__(self, deps: Deps):
        self.deps = deps

    async def run(self, state: AgentState) -> dict:
        draft = state.get("draft_answer") or ""
        issues: list[str] = []

        if not draft:
            issues.append("empty_answer")
        if len(draft) < self.MIN_ANSWER_LEN:
            issues.append("answer_too_short")
        if state.get("search_result") and "[" not in draft and "(" not in draft:
            issues.append("missing_citations")

        return {
            "output_valid": len(issues) == 0,
            "validation_issues": issues,
        }
