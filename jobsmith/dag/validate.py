"""Input / Output validators — fold over the profile's rule tuples."""
from __future__ import annotations

from .profile import AgentProfile
from .state import AgentState


class InputValidator:
    def __init__(self, profile: AgentProfile):
        self.profile = profile

    async def run(self, state: AgentState) -> dict:
        for rule in self.profile.input_rules:
            message = rule(state)
            if message is not None:
                return {
                    "input_valid": False,
                    "rejection_reason": getattr(rule, "__name__", "input_rule"),
                    "user_error_message": message,
                }
        return {
            "input_valid": True,
            "max_refine": state.get("max_refine", self.profile.max_refine),
        }


class OutputValidator:
    def __init__(self, profile: AgentProfile):
        self.profile = profile

    async def run(self, state: AgentState) -> dict:
        issues = [
            issue for rule in self.profile.output_rules
            if (issue := rule(state)) is not None
        ]
        return {
            "output_valid": len(issues) == 0,
            "validation_issues": issues,
        }
