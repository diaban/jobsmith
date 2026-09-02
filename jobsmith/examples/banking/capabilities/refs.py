"""REFERENCES capability — retrieve past slides / reference decks."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.constants import END

from ....core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ....core.state import CapabilityResult
from ..deps import SearchEngine


class RefsState(CapabilityBaseState, total=False):
    raw_refs: list[dict[str, Any]]
    filtered_refs: list[dict[str, Any]]


class RefsCapability(Capability):
    """Retrieve past slides / reference decks."""

    spec = CapabilitySpec(
        name="refs",
        description="Retrieve past slides and reference decks",
        output_schema={
            "type": "object",
            "properties": {
                "refs": {"type": "array", "items": {"type": "object"}},
            },
        },
    )

    def __init__(self, search: SearchEngine, *, top_k: int = 20, keep_top: int = 5):
        self.search = search
        self.top_k = top_k
        self.keep_top = keep_top

    # -------------------- Nodes --------------------

    async def retrieve(self, state: RefsState) -> dict:
        try:
            refs = await self.search.search(
                state["query"] + " past_slides", top_k=self.top_k
            )
            return {"raw_refs": refs}
        except Exception:
            return {"raw_refs": []}

    async def filter_refs(self, state: RefsState) -> dict:
        # TODO: LLM-based relevance filter; simple top-N for now.
        return {"filtered_refs": state.get("raw_refs", [])[: self.keep_top]}

    async def emit_success(self, state: RefsState) -> dict:
        return self._emit_success({"refs": state.get("filtered_refs", [])})

    async def emit_failure(self, state: RefsState) -> dict:
        return self._emit_failure("no refs retrieved")

    # -------------------- Router --------------------

    def route_after_filter(self, state: RefsState) -> Literal["ok", "fail"]:
        return "ok" if state.get("filtered_refs") else "fail"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        refs = result.get("data", {}).get("refs", [])
        if not refs:
            return None
        parts = ["# References"]
        for i, r in enumerate(refs):
            parts.append(f"[{r.get('id', f'ref_{i}')}] {r.get('summary', '')}")
        return "\n\n".join(parts)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(RefsState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("filter", self.filter_refs)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "filter")
        g.add_conditional_edges("filter", self.route_after_filter, {
            "ok": "emit_success",
            "fail": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
