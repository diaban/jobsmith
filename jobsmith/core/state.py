"""Generic agent state schemas.

Design notes:
- AgentState is the *global* state for the parent graph; capability sub-graphs
  declare their own private schemas (extending CapabilityBaseState) to keep
  intermediate values out of the global state.
- Capability results live in a single `results` dict keyed by capability name.

Fan-in safety of the `results` reducer:
  Each capability sub-graph writes exactly {"results": {its_own_name: result}}.
  The registry enforces unique capability names and the planner forbids
  duplicate plan steps, so the keys written by parallel Send branches are
  provably disjoint — dict-union merging is therefore order-independent.
  (The old invariant "single writer per field" becomes "single writer per key".)

Determinism caveat:
  Insertion order of `results` depends on wave arrival order. Consumers that
  need stable ordering (e.g. context merging) must iterate in *plan order*,
  never in dict order.

Totality, and why `query` is the exception:
  These schemas are `total=False` because a LangGraph node returns a *partial*
  update — that is right for writes. It is wrong for reads: pyright's
  `reportTypedDictNotRequiredAccess` then rejects `state["query"]` even though
  the graph is only ever entered with a query (`jobs/runner.py` invokes it with
  `{"query", "inputs", "job_id"}`, and a resume replays that same checkpoint).
  `Required[str]` states that truthfully, and costs nothing on the write side
  because no node is annotated `-> AgentState`: they all return plain `dict`.
  Every other key is genuinely absent until some node writes it, so it stays
  NotRequired and must be read with `.get()` and a default that says what
  missing means — see PostProcessor, DocumentsCapability, ResearchCapability.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Required, TypedDict

# ---------- Plan ----------

class PlanStep(TypedDict):
    capability: str         # registered capability name
    depends_on: list[str]   # other capability names; [] = ready immediately


class Plan(TypedDict):
    steps: list[PlanStep]
    rationale: str          # for observability / debugging


# ---------- Capability results ----------

class CapabilityResult(TypedDict, total=False):
    ok: bool
    data: dict[str, Any]        # capability-specific payload (matches its output_schema)
    error: str | None
    meta: dict[str, Any]        # via_fallback, timings, artifact refs, ...


def merge_results(
    left: dict[str, CapabilityResult] | None,
    right: dict[str, CapabilityResult] | None,
) -> dict[str, CapabilityResult]:
    """Fan-in reducer for parallel Send branches: dict union, right wins per key.

    Kept total (never raises) — key disjointness is enforced upstream by the
    registry (unique names) and the planner (no duplicate steps).
    """
    return {**(left or {}), **(right or {})}


# ---------- Reserved `inputs` keys ----------

# `inputs` is an open dict of domain material (image keys, file refs, ...), but
# one key is a framework convention: the excerpt of the conversation a job was
# launched from. The chat layer fills it (chat/tools.py), the planner reads it
# to resolve what a request refers to ("analyse that"). It is background, never
# the request itself — `query` stays authoritative.
CONVERSATION_INPUT_KEY = "conversation"


# ---------- Errors ----------

class NodeError(TypedDict):
    source: str         # capability name or framework node ("planner", "generation", ...)
    kind: str           # e.g. "search_fail", "planner_fail", "generation_fail"
    detail: str
    recoverable: bool   # False → routes to execution_error / escalation


# ---------- Global state ----------

class AgentState(TypedDict, total=False):
    # --- Input ---
    query: Required[str]        # guaranteed at entry: the graph is invoked with it
    inputs: dict[str, Any]      # arbitrary domain inputs (image keys, file refs, ...)
                                # plus CONVERSATION_INPUT_KEY when chat-launched
    job_id: str

    # --- Validation ---
    input_valid: bool
    rejection_reason: str | None

    # --- Routing (triage decision) ---
    route: str | None           # "plan" | "direct" (see core/router.py)

    # --- Planner output ---
    plan: Plan | None

    # --- Capability execution (fan-in safe) ---
    completed_capabilities: Annotated[list[str], add]
    results: Annotated[dict[str, CapabilityResult], merge_results]

    # --- Generation pipeline ---
    merged_context: str | None
    draft_answer: str | None
    output_valid: bool
    validation_issues: list[str]
    final_answer: str | None

    # --- Control ---
    errors: Annotated[list[NodeError], add]
    refine_count: int
    max_refine: int

    # --- Terminal status (for routing to user_error / escalate) ---
    terminal_kind: str | None  # "answer" | "user_error" | "escalated"
    user_error_message: str | None
