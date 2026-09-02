"""Capability: a pluggable agentic sub-graph with a self-describing spec.

Pattern (same OO idiom as the rest of the framework):
- A Capability subclass owns its deps and configuration; its constructor takes
  exactly the clients it needs — the framework never introspects them.
- Node logic is async instance methods; `build()` returns the compiled
  sub-graph that the parent graph mounts as a single node.
- Terminal nodes call `_emit_success` / `_emit_failure` so every capability
  reports uniformly into `results` / `completed_capabilities` / `errors`.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph

from .state import AgentState, CapabilityResult, NodeError, merge_results

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Self-description used for the planner prompt and job metadata.

    Schemas are plain JSON-schema dicts: they exist to be rendered into the
    planner prompt and stored alongside jobs — enforcement is advisory.
    """
    name: str                                   # unique; doubles as node-name suffix & results key
    description: str                            # one paragraph, feeds the planner prompt
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_inputs: tuple[str, ...] = ()       # keys that must exist in state["inputs"]

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"invalid capability name {self.name!r} (must match {_NAME_RE.pattern})"
            )


class CapabilityOutputState(TypedDict, total=False):
    """The ONLY channels a capability sub-graph emits back to the parent.

    All three have reducers, so parallel Send branches finishing in the same
    superstep merge safely. Capability `build()` implementations MUST pass this
    as `output_schema` — otherwise the sub-graph echoes its whole final state
    (including plain channels like `query`) and concurrent branches collide
    with InvalidUpdateError.
    """
    results: Annotated[dict[str, CapabilityResult], merge_results]
    completed_capabilities: Annotated[list[str], add]
    errors: Annotated[list[NodeError], add]


class CapabilityBaseState(CapabilityOutputState, total=False):
    """Base for private capability states — inputs plus the output channels.

    Sub-graph private schemas extend this with their own intermediate fields.
    The reducers MUST match AgentState's so parent fan-in works.
    """
    query: str
    inputs: dict[str, Any]


class Capability(ABC):
    """A pluggable agentic graph. Subclasses own deps + config."""

    spec: CapabilitySpec  # set as a class attribute (or instance attribute) by subclasses

    @abstractmethod
    def build(self):
        """Return the compiled sub-graph (no checkpointer — the parent owns it)."""
        ...

    @staticmethod
    def state_graph(private_schema: type) -> StateGraph:
        """StateGraph pre-configured with the mandatory capability output schema."""
        return StateGraph(private_schema, output_schema=CapabilityOutputState)

    # ---- Planner integration ----

    def is_applicable(self, state: AgentState) -> bool:
        """Whether this capability can run for the given request.

        Default: all `requires_inputs` keys are present in state["inputs"].
        Override for richer conditions.
        """
        inputs = state.get("inputs") or {}
        return all(k in inputs for k in self.spec.requires_inputs)

    # ---- Generation integration ----

    def render_context(self, result: CapabilityResult) -> str | None:
        """Format own successful result for the generation context.

        Return None (default) or empty to be omitted from the merged context.
        """
        return None

    def render_report(self, result: CapabilityResult) -> str | None:
        """Format own result as markdown, for a human reading the job's report.

        Twin of `render_context` (which targets the model). The default is a
        readable rendering of the payload — override when a capability knows
        better: a link to a file it produced, a table, an embedded image.
        Return None to be left out of the report entirely.
        """
        if not result.get("ok"):
            return f"_{result.get('error') or 'no detail'}_"
        return default_result_markdown(result.get("data") or {})

    # ---- Emit helpers (used by terminal sub-graph nodes) ----

    def _emit_success(self, data: dict[str, Any], meta: dict[str, Any] | None = None) -> dict:
        result: CapabilityResult = {"ok": True, "data": data, "meta": meta or {}}
        return {
            "results": {self.spec.name: result},
            "completed_capabilities": [self.spec.name],
        }

    def _emit_failure(self, detail: str, *, recoverable: bool = True) -> dict:
        err: NodeError = {
            "source": self.spec.name,
            "kind": f"{self.spec.name}_fail",
            "detail": detail,
            "recoverable": recoverable,
        }
        result: CapabilityResult = {"ok": False, "error": detail}
        return {
            "results": {self.spec.name: result},
            "completed_capabilities": [self.spec.name],
            "errors": [err],
        }


def default_result_markdown(data: dict[str, Any]) -> str:
    """Best-effort markdown for a payload nobody described.

    Deliberately dumb: prose stays prose, a list of strings becomes bullets,
    and only genuinely structured values fall back to JSON. A capability that
    cares about its presentation overrides `render_report` instead of growing
    this function — the framework must not learn the shape of every payload.
    """
    if not data:
        return "_(empty)_"
    if len(data) == 1 and isinstance(next(iter(data.values())), str):
        return next(iter(data.values()))
    parts: list[str] = []
    for key, value in data.items():
        parts.append(f"**{key}**\n")
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            parts.append("\n".join(f"- {v}" for v in value))
        else:
            parts.append("```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```")
        parts.append("")
    return "\n".join(parts).strip()
