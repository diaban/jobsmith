"""Executor step — object-oriented version.

Owns:
- the mapping subgraph-name → parent-graph node name
- the wave-computation logic (which subgraphs are ready)
- the dispatch node (pass-through) and the router function

Usage in the parent graph:
    executor = Executor()
    g.add_node("executor_dispatch", executor.dispatch)
    g.add_conditional_edges("executor_dispatch", executor.route, {...})
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from ..state import AgentState


class Executor:

    # Default mapping; override in the constructor if you rename parent nodes.
    DEFAULT_SUBGRAPH_NODE = {
        "search": "subgraph_search",
        "vision": "subgraph_vision",
        "refs":   "subgraph_refs",
    }

    def __init__(self, subgraph_node_map: dict[str, str] | None = None):
        self.subgraph_node = subgraph_node_map or self.DEFAULT_SUBGRAPH_NODE

    # -------- Helpers --------

    @staticmethod
    def _has_unrecoverable(state: AgentState) -> bool:
        return any(not e["recoverable"] for e in state.get("errors", []))

    @staticmethod
    def _ready_subgraphs(state: AgentState) -> list[str]:
        plan = state.get("plan")
        if not plan:
            return []
        done = set(state.get("completed_subgraphs", []))
        ready: list[str] = []
        for step in plan["steps"]:
            sg = step["subgraph"]
            if sg in done:
                continue
            if all(dep in done for dep in step["depends_on"]):
                ready.append(sg)
        return ready

    @staticmethod
    def _all_done(state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        done = set(state.get("completed_subgraphs", []))
        return len(done) >= len(plan["steps"])

    # -------- Node + Router --------

    async def dispatch(self, state: AgentState) -> dict:
        """Pass-through node. Real work happens in `route`."""
        return {}

    def route(self, state: AgentState) -> Any:
        """Return either a list[Send] to fan out the next wave, or a string
        node name for a normal transition.
        """
        if self._has_unrecoverable(state):
            return "execution_error"
        if self._all_done(state):
            return "merge_results"
        ready = self._ready_subgraphs(state)
        if not ready:
            return "execution_error"  # deadlock — shouldn't happen
        return [Send(self.subgraph_node[sg], state) for sg in ready]
