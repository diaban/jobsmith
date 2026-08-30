"""Terminal error nodes."""
from __future__ import annotations

from typing import Any

from .state import AgentState


class ExecutionError:
    """Aggregation node — no work, just a hook for the conditional router."""

    async def run(self, state: AgentState) -> dict:
        return {}


class Escalator:
    DEFAULT_MESSAGE = (
        "Votre demande a été transmise à un analyste. Vous serez recontacté."
    )

    def __init__(self, store: Any, *, message: str | None = None):
        self.store = store
        self.message = message or self.DEFAULT_MESSAGE

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
    DEFAULT_MESSAGE = "Une erreur est survenue."

    async def run(self, state: AgentState) -> dict:
        msg = state.get("user_error_message") or self.DEFAULT_MESSAGE
        return {"terminal_kind": "user_error", "user_error_message": msg}
