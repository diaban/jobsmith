"""Terminal error nodes."""
from __future__ import annotations

from .profile import AgentProfile
from .state import AgentState


class ExecutionError:
    """Aggregation node — no work, just a hook for the conditional router."""

    async def run(self, state: AgentState) -> dict:
        return {}


class Escalator:
    """Marks the run as escalated. The escalation payload (errors, plan,
    partial results) is already in state — JobManager persists it."""

    def __init__(self, profile: AgentProfile):
        self.message = profile.escalation_message

    async def run(self, state: AgentState) -> dict:
        return {"terminal_kind": "escalated", "user_error_message": self.message}


class UserErrorEmitter:
    def __init__(self, profile: AgentProfile):
        self.default_message = profile.user_error_message

    async def run(self, state: AgentState) -> dict:
        msg = state.get("user_error_message") or self.default_message
        return {"terminal_kind": "user_error", "user_error_message": msg}
