"""Terminal error nodes — object-oriented version."""
from __future__ import annotations

from typing import Any

from ..deps import Deps
from ..state import AgentState


class ExecutionError:
    """Aggregation node — no work, just a hook for the conditional router."""

    async def run(self, state: AgentState) -> dict:
        return {}


class Escalator:
    DEFAULT_MESSAGE = (
        "Votre demande a été transmise à un analyste. Vous serez recontacté."
    )

    def __init__(self, deps: Deps, store: Any, *, message: str | None = None):
        self.deps = deps
        self.store = store
        self.message = message or self.DEFAULT_MESSAGE

    async def run(self, state: AgentState) -> dict:
        await self.store.aput(
            namespace=("escalations", state["thread_id"]),
            key=state["thread_id"],
            value={
                "query": state.get("query"),
                "errors": state.get("errors", []),
                "plan": state.get("plan"),
                "partial_results": {
                    "search": state.get("search_result"),
                    "vision": state.get("vision_result"),
                    "refs": state.get("refs_result"),
                },
            },
        )
        return {"terminal_kind": "escalated", "user_error_message": self.message}


class UserErrorEmitter:
    DEFAULT_MESSAGE = "Une erreur est survenue."

    async def run(self, state: AgentState) -> dict:
        msg = state.get("user_error_message") or self.DEFAULT_MESSAGE
        return {"terminal_kind": "user_error", "user_error_message": msg}
