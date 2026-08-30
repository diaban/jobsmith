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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

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


class CapabilityBaseState(TypedDict, total=False):
    """Base for private capability states — carries the shared output channels.

    Sub-graph private schemas extend this with their own intermediate fields.
    The reducers MUST match AgentState's so parent fan-in works.
    """
    query: str
    inputs: dict[str, Any]
    results: Annotated[dict[str, CapabilityResult], merge_results]
    completed_capabilities: Annotated[list[str], add]
    errors: Annotated[list[NodeError], add]


class Capability(ABC):
    """A pluggable agentic graph. Subclasses own deps + config."""

    spec: CapabilitySpec  # set as a class attribute (or instance attribute) by subclasses

    @abstractmethod
    def build(self):
        """Return the compiled sub-graph (no checkpointer — the parent owns it)."""
        ...

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
