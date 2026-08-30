"""Compose the parent agent graph from class instances + a capability registry.

Pattern:
- Each step is a class instance owning its deps and config.
- Nodes registered as `instance.run` (bound method, async).
- Routers registered as `instance.route` (bound method, sync).
- Capabilities registered as `capability.build()` (compiled sub-graphs), one
  node per registry entry, each edging back to `executor_dispatch`.

Static vs dynamic trade-off: the capability set of a compiled graph is FIXED —
Send targets and conditional-edge path maps must name real nodes. `build()`
therefore freezes the registry. Registering a new capability means creating a
fresh AgentBuilder with a new registry; compilation costs milliseconds.
"""
from __future__ import annotations

from typing import Any

from langgraph.constants import END
from langgraph.graph import StateGraph

from .deps import Deps
from .errors import Escalator, ExecutionError, UserErrorEmitter
from .executor import Executor
from .generation import ContextMerger, Generator, PostProcessor, Refiner
from .planner import Planner
from .profile import AgentProfile
from .registry import CapabilityRegistry
from .state import AgentState
from .validate import InputValidator, OutputValidator


class AgentBuilder:
    """Builds the parent graph and holds references to every step instance.

    Holding references is useful for tests / observability (you can inspect
    `builder.planner.system_prompt()`, swap an instance before `.build()`, etc).
    """

    def __init__(
        self,
        deps: Deps,
        registry: CapabilityRegistry,
        *,
        profile: AgentProfile | None = None,
        checkpointer: Any = None,
    ):
        self.deps = deps
        self.registry = registry
        self.profile = profile or AgentProfile()
        self.checkpointer = checkpointer

        # --- Step instances ---
        self.input_validator  = InputValidator(self.profile)
        self.planner          = Planner(deps, registry,
                                        prompt_template=self.profile.planner_prompt_template)
        self.executor         = Executor(registry)
        self.context_merger   = ContextMerger(registry, self.profile)
        self.generator        = Generator(deps, self.profile)
        self.output_validator = OutputValidator(self.profile)
        self.refiner          = Refiner(deps, self.profile)
        self.post_processor   = PostProcessor()
        self.execution_error  = ExecutionError()
        self.escalator        = Escalator(self.profile)
        self.user_error       = UserErrorEmitter(self.profile)

    # ---- Conditional edge functions ----

    @staticmethod
    def _route_validate_input(state: AgentState) -> str:
        return "planner" if state.get("input_valid") else "user_error"

    @staticmethod
    def _route_after_planner(state: AgentState) -> str:
        if state.get("plan") is None or any(
            not e["recoverable"] for e in state.get("errors", [])
        ):
            return "execution_error"
        return "executor_dispatch"

    @staticmethod
    def _route_validate_output(state: AgentState) -> str:
        # Unrecoverable generation/refine failure — bail out instead of looping
        # the refine cycle (the refine counter only advances on success).
        if any(not e["recoverable"] for e in state.get("errors", [])):
            return "execution_error"
        if state.get("output_valid"):
            return "post_process"
        if state.get("refine_count", 0) >= state.get("max_refine", 2):
            return "execution_error"
        return "refine"

    @staticmethod
    def _route_execution_error(state: AgentState) -> str:
        has_partial = any(
            r.get("ok") for r in state.get("results", {}).values()
        )
        return "escalate" if has_partial else "user_error"

    # ---- Build ----

    def build(self):
        self.registry.freeze()
        g = StateGraph(AgentState)

        # Static nodes
        g.add_node("validate_input",    self.input_validator.run)
        g.add_node("planner",           self.planner.run)
        g.add_node("executor_dispatch", self.executor.dispatch)
        g.add_node("merge_results",     self.context_merger.run)
        g.add_node("generation",        self.generator.run)
        g.add_node("validate_output",   self.output_validator.run)
        g.add_node("refine",            self.refiner.run)
        g.add_node("post_process",      self.post_processor.run)
        g.add_node("execution_error",   self.execution_error.run)
        g.add_node("escalate",          self.escalator.run)
        g.add_node("user_error",        self.user_error.run)

        # One node per registered capability, each looping back to the dispatcher
        cap_targets: dict[str, str] = {}
        for cap in self.registry:
            node = Executor.node_name(cap.spec.name)
            g.add_node(node, cap.build())
            g.add_edge(node, "executor_dispatch")
            cap_targets[node] = node

        # Edges
        g.set_entry_point("validate_input")
        g.add_conditional_edges("validate_input", self._route_validate_input, {
            "planner": "planner",
            "user_error": "user_error",
        })
        g.add_conditional_edges("planner", self._route_after_planner, {
            "executor_dispatch": "executor_dispatch",
            "execution_error": "execution_error",
        })

        # Executor routes via list[Send] or string targets
        g.add_conditional_edges("executor_dispatch", self.executor.route, {
            "merge_results": "merge_results",
            "execution_error": "execution_error",
            **cap_targets,
        })

        # Generation pipeline
        g.add_edge("merge_results", "generation")
        g.add_edge("generation", "validate_output")
        g.add_conditional_edges("validate_output", self._route_validate_output, {
            "post_process": "post_process",
            "refine": "refine",
            "execution_error": "execution_error",
        })
        g.add_edge("refine", "generation")
        g.add_edge("post_process", END)

        # Error routing
        g.add_conditional_edges("execution_error", self._route_execution_error, {
            "escalate": "escalate",
            "user_error": "user_error",
        })
        g.add_edge("escalate", END)
        g.add_edge("user_error", END)

        return g.compile(checkpointer=self.checkpointer)


# Convenience function
def build_agent(
    deps: Deps,
    registry: CapabilityRegistry,
    *,
    profile: AgentProfile | None = None,
    checkpointer: Any = None,
):
    return AgentBuilder(deps, registry, profile=profile, checkpointer=checkpointer).build()
