"""Terminal error nodes."""
from __future__ import annotations

from typing import Any

from .profile import AgentProfile
from .state import AgentState


class ExecutionError:
    """Aggregation node — no work, just a hook for the conditional router."""

    async def run(self, state: AgentState) -> dict:
        return {}


class Escalator:
    def __init__(self, store: Any, profile: AgentProfile):
        self.store = store
        self.message = profile.escalation_message

    async def run(self, state: AgentState) -> dict:
        await self.store.aput(
            ("escalations", state["job_id"]),
            state["job_id"],
            {
                "query": state.get("query"),
                "errors": state.get("errors", []),
                "plan": state.get("plan"),
                "partial_results": state.get("results", {}),
            },
        )
        return {"terminal_kind": "escalated", "user_error_message": self.message}


class UserErrorEmitter:
    def __init__(self, profile: AgentProfile):
        self.default_message = profile.user_error_message

    async def run(self, state: AgentState) -> dict:
        msg = state.get("user_error_message") or self.default_message
        return {"terminal_kind": "user_error", "user_error_message": msg}
