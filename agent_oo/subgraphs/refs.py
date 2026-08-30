"""REFERENCES sub-graph — object-oriented version."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.constants import END
from langgraph.graph import StateGraph

from ..deps import Deps
from ..state import NodeError, RefsResult, SubgraphName


class RefsSubState(TypedDict, total=False):
    query: str
    raw_refs: list[dict[str, Any]]
    filtered_refs: list[dict[str, Any]]
    refs_result: Any
    completed_subgraphs: Any
    errors: Any


class RefsSubgraph:
    def __init__(self, deps: Deps, *, top_k: int = 20, keep_top: int = 5):
        self.deps = deps
        self.top_k = top_k
        self.keep_top = keep_top

    async def retrieve(self, state: RefsSubState) -> dict:
        try:
            refs = await self.deps.search.search(
                state["query"] + " past_slides", top_k=self.top_k
            )
            return {"raw_refs": refs}
        except Exception:
            return {"raw_refs": []}

    async def filter_refs(self, state: RefsSubState) -> dict:
        # TODO: LLM-based relevance filter; simple top-N for now.
        return {"filtered_refs": state.get("raw_refs", [])[: self.keep_top]}

    async def emit_success(self, state: RefsSubState) -> dict:
        result: RefsResult = {"refs": state.get("filtered_refs", [])}
        return {
            "refs_result": result,
            "completed_subgraphs": [SubgraphName.REFS.value],
        }

    async def emit_failure(self, state: RefsSubState) -> dict:
        err: NodeError = {
            "subgraph": SubgraphName.REFS.value,
            "kind": "refs_fail",
            "detail": "no refs retrieved",
            "recoverable": True,
        }
        return {
            "completed_subgraphs": [SubgraphName.REFS.value],
            "errors": [err],
        }

    def route_after_filter(self, state: RefsSubState) -> Literal["ok", "fail"]:
        return "ok" if state.get("filtered_refs") else "fail"

    def build(self):
        g = StateGraph(RefsSubState)
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
