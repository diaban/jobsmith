"""Executor step — wave-based dispatch over registered capabilities.

Owns:
- the naming convention capability-name → parent-graph node name
- the wave-computation logic (which capabilities are ready)
- the dispatch node (pass-through) and the router function

Each capability sub-graph edges back to `executor_dispatch` on completion, so
the router can compute the next wave — this executes an arbitrary dependency
DAG without baking a topological schedule into the graph.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from .registry import CapabilityRegistry
from .state import AgentState


class Executor:

    @staticmethod
    def node_name(cap_name: str) -> str:
        """Parent-graph node name for a capability."""
        return f"cap_{cap_name}"

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    # -------- Helpers --------

    @staticmethod
    def _has_unrecoverable(state: AgentState) -> bool:
        return any(not e["recoverable"] for e in state.get("errors", []))

    @staticmethod
    def _ready_capabilities(state: AgentState) -> list[str]:
        plan = state.get("plan")
        if not plan:
            return []
        done = set(state.get("completed_capabilities", []))
        ready: list[str] = []
        for step in plan["steps"]:
            cap = step["capability"]
            if cap in done:
                continue
            if all(dep in done for dep in step["depends_on"]):
                ready.append(cap)
        return ready

    @staticmethod
    def _all_done(state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        done = set(state.get("completed_capabilities", []))
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
        ready = self._ready_capabilities(state)
        if not ready:
            return "execution_error"  # deadlock — shouldn't happen
        return [Send(self.node_name(cap), state) for cap in ready]
