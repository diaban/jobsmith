"""Executor step — wave-based dispatch over registered capabilities.

Owns:
- the naming convention capability-name → parent-graph node name
- the wave-computation logic (which capabilities are ready)
- the dispatch node (pass-through) and the router function

Each capability sub-graph edges back to `executor_dispatch` on completion, so
the router can compute the next wave — this executes an arbitrary dependency
DAG without baking a topological schedule into the graph.

It is the compiler's interpreter (docs/design/compiler-v1.md): before a step
is Sent, its references are resolved against the results so far and the job's
inputs (`dag/ir.py`) and its op's `input_model`, when it declares one, checks
them. A step whose arguments cannot be had fails as a step — recoverable, so
the run degrades — never the job; the dispatch node writes that failure, so
the router only ever Sends a step it can run.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from langgraph.types import Send
from pydantic import ValidationError

from ..engine.facts import publish
from .capability import CAP_NODE_PREFIX
from .ir import UnresolvedReference, resolve
from .registry import CapabilityRegistry
from .state import AgentState, CapabilityResult, NodeError, PlanStep, step_id

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
        """Steps (by id) whose last run failed saying another try is worth
        it, while they have run at most `max_retries` times (#191). Each run
        appends the step's id to `completed_capabilities`, so the count is
        already there."""
        results = state.get("results", {})
        runs = Counter(state.get("completed_capabilities", []))
        return {step for step, result in results.items()
                if not result.get("ok") and result.get("retryable")
                and runs[step] <= self.max_retries}

    def _done(self, state: AgentState) -> set[str]:
        """Finished for good: a step to retry is not, so it is dispatched
        again and nothing that depends on it runs before it."""
        return set(state.get("completed_capabilities", [])) - self._to_retry(state)

    def _ready_steps(self, state: AgentState) -> list[PlanStep]:
        plan = state.get("plan")
        if not plan:
            return []
        done = self._done(state)
        return [step for step in plan["steps"]
                if step_id(step) not in done
                and all(dep in done for dep in step["depends_on"])]

    def _all_done(self, state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        return len(self._done(state)) >= len(plan["steps"])

    # -------- Node + Router --------

    def _arguments(self, state: AgentState, step: PlanStep) -> tuple[dict[str, Any], str | None]:
        """The step's arguments with every reference resolved, or why they
        cannot be had."""
        try:
            args = resolve(step.get("args") or {}, results=state.get("results") or {},
                           inputs=state.get("inputs") or {})
        except UnresolvedReference as exc:
            return {}, str(exc)
        model = self.registry.get(step["capability"]).spec.input_model
        if model is not None:
            try:
                model.model_validate(args)
            except ValidationError as exc:
                error = exc.errors()[0]
                where = ".".join(str(part) for part in error["loc"])
                return {}, f"arguments refused at {where}: {error['msg']}"
        return args, None

    async def dispatch(self, state: AgentState) -> dict:
        """Fail, as steps, the ready steps whose arguments cannot be had —
        then their dependents that became ready by it, until none is left —
        so `route` Sends only steps that can run."""
        failed: dict[str, CapabilityResult] = {}
        errors: list[NodeError] = []
        while True:
            view: AgentState = {**state, "results": {**state.get("results", {}), **failed},
                                "completed_capabilities": [
                                    *state.get("completed_capabilities", []), *failed]}
            refused = [(step, why) for step in self._ready_steps(view)
                       if (why := self._arguments(view, step)[1])]
            if not refused:
                break
            for step, why in refused:
                sid = step_id(step)
                failed[sid] = {"ok": False, "error": why, "meta": {}}
                publish(f"step:{sid}", failed[sid])
                errors.append({"source": step["capability"], "kind": "arguments_fail",
                               "detail": f"{sid}: {why}", "recoverable": True})
        if not failed:
            return {}
        return {"results": failed, "completed_capabilities": list(failed), "errors": errors}

    def route(self, state: AgentState) -> Any:
        """Return either a list[Send] to fan out the next wave, or a string
        node name for a normal transition.
        """
        if self._has_unrecoverable(state):
            return "execution_error"
        if self._all_done(state):
            return "merge_results"
        ready = self._ready_steps(state)
        if not ready:
            return "execution_error"  # deadlock — shouldn't happen
        sent = {key: value for key, value in state.items() if key not in _APPENDED}
        # Each step is sent with its identity, which is what its result, its
        # fact and its run count are keyed by (`Capability._step_id`), and its
        # resolved arguments (`dispatch` failed every step that has none).
        return [Send(self.node_name(step["capability"]),
                     {**sent, "step": {"id": step_id(step), "capability": step["capability"],
                                       "args": self._arguments(state, step)[0]}})
                for step in ready]
