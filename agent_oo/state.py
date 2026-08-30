"""Agent state schemas.

Design notes:
- AgentState is the *global* state for the parent graph.
- Each sub-graph declares its own input/output schema to avoid polluting
  the global state with intermediate values (retry counters, raw API
  responses, etc).
- Fields written by parallel sub-graphs have explicit reducers.
"""
from __future__ import annotations

from enum import Enum
from operator import add
from typing import Annotated, Any, Optional, TypedDict


# ---------- Plan ----------

class SubgraphName(str, Enum):
    SEARCH = "search"
    VISION = "vision"
    REFS = "refs"


class PlanStep(TypedDict):
    subgraph: str           # SubgraphName value
    depends_on: list[str]   # list of SubgraphName values; [] = ready immediately


class Plan(TypedDict):
    steps: list[PlanStep]
    rationale: str          # for observability / debugging


# ---------- Sub-graph results (each has ONE writer → no reducer needed) ----------

class SearchResult(TypedDict):
    docs: list[dict[str, Any]]
    query_used: str
    via_fallback: bool


class VisionResult(TypedDict):
    description: str
    image_s3_key: str


class RefsResult(TypedDict):
    refs: list[dict[str, Any]]


# ---------- Errors ----------

class NodeError(TypedDict):
    subgraph: str
    kind: str           # e.g. "search_fail", "vision_fail", "refs_fail",
                        #      "validation_fail", "planner_fail"
    detail: str
    recoverable: bool   # False → routes to execution_error / escalation


# ---------- Global state ----------

class AgentState(TypedDict, total=False):
    # --- Input ---
    query: str
    image_s3_keys: list[str]          # may be empty
    thread_id: str

    # --- Validation ---
    input_valid: bool
    rejection_reason: Optional[str]

    # --- Planner output ---
    plan: Optional[Plan]

    # --- Sub-graph completion tracking (fan-in safe) ---
    completed_subgraphs: Annotated[list[str], add]

    # --- Sub-graph results (single writer each) ---
    search_result: Optional[SearchResult]
    vision_result: Optional[VisionResult]
    refs_result: Optional[RefsResult]

    # --- Generation pipeline ---
    merged_context: Optional[str]
    draft_answer: Optional[str]
    output_valid: bool
    validation_issues: list[str]
    final_answer: Optional[str]

    # --- Control ---
    errors: Annotated[list[NodeError], add]
    refine_count: int
    max_refine: int

    # --- Terminal status (for routing to user_error / escalate) ---
    terminal_kind: Optional[str]  # "answer" | "user_error" | "escalated"
    user_error_message: Optional[str]


# ---------- Search sub-graph private state ----------

class SearchSubState(TypedDict, total=False):
    # Inherited from parent
    query: str
    # Local
    generated_query: str
    raw_docs: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    # Output
    search_result: Optional[SearchResult]
    completed_subgraphs: Annotated[list[str], add]
    errors: Annotated[list[NodeError], add]
