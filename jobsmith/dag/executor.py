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

from collections import Counter
from typing import Any

from langgraph.types import Send

from .capability import CAP_NODE_PREFIX
from .registry import CapabilityRegistry
from .state import AgentState

# The append-only channels a capability emits back (`CapabilityOutputState`).
# A sub-graph is entered with what it is Sent and returns those channels as they
# stand at its end, so it must be Sent none of them: it would hand the parent
# back what the parent already has, appended again (#194) — and a step's runs,
# which bound its retries, are counted in one of them (#191).
_APPENDED = ("completed_capabilities", "errors")


class Executor:

    @staticmethod
    def node_name(cap_name: str) -> str:
        """Parent-graph node name for a capability."""
        return CAP_NODE_PREFIX + cap_name

    def __init__(self, registry: CapabilityRegistry, *, max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries

    # -------- Helpers --------

    @staticmethod
    def _has_unrecoverable(state: AgentState) -> bool:
        return any(not e["recoverable"] for e in state.get("errors", []))

    def _to_retry(self, state: AgentState) -> set[str]:
        """Steps whose last run failed saying another try is worth it, while
        they have run at most `max_retries` times (#191). Each run appends the
        step to `completed_capabilities`, so the count is already there."""
        results = state.get("results", {})
        runs = Counter(state.get("completed_capabilities", []))
        return {cap for cap, result in results.items()
                if not result.get("ok") and result.get("retryable")
                and runs[cap] <= self.max_retries}

    def _done(self, state: AgentState) -> set[str]:
        """Finished for good: a step to retry is not, so it is dispatched
        again and nothing that depends on it runs before it."""
        return set(state.get("completed_capabilities", [])) - self._to_retry(state)

    def _ready_capabilities(self, state: AgentState) -> list[str]:
        plan = state.get("plan")
        if not plan:
            return []
        done = self._done(state)
        ready: list[str] = []
        for step in plan["steps"]:
            cap = step["capability"]
            if cap in done:
                continue
            if all(dep in done for dep in step["depends_on"]):
                ready.append(cap)
        return ready

    def _all_done(self, state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        return len(self._done(state)) >= len(plan["steps"])

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
        sent = {key: value for key, value in state.items() if key not in _APPENDED}
        return [Send(self.node_name(cap), sent) for cap in ready]
