"""Compose the parent agent graph from class instances.

Pattern differences vs the factory-functions version:
- Each step is a class instance owning `deps` and config.
- Nodes registered as `instance.run` (bound method, async).
- Routers registered as `instance.route` (bound method, sync).
- Sub-graphs registered as `instance.build()` (compiled CompiledGraph).

We expose a single `AgentBuilder` so the wiring lives next to the instances.
"""
from __future__ import annotations

from typing import Any

from langgraph.constants import END
from langgraph.graph import StateGraph

from .deps import Deps
from .nodes.errors import Escalator, ExecutionError, UserErrorEmitter
from .nodes.executor import Executor
from .nodes.generation import ContextMerger, Generator, PostProcessor, Refiner
from .nodes.planner import Planner
from .nodes.validate import InputValidator, OutputValidator
from .state import AgentState
from .subgraphs.refs import RefsSubgraph
from .subgraphs.search import SearchSubgraph
from .subgraphs.vision import VisionSubgraph


class AgentBuilder:
    """Builds the parent graph and holds references to every step instance.

    Holding references is useful for tests / observability (you can inspect
    `builder.planner.SYSTEM_PROMPT`, swap an instance before `.build()`, etc).
    """

    def __init__(self, deps: Deps, checkpointer: Any, store: Any):
        self.deps = deps
        self.checkpointer = checkpointer
        self.store = store

        # --- Step instances ---
        self.input_validator  = InputValidator(deps)
        self.planner          = Planner(deps)
        self.executor         = Executor()
        self.search_subgraph  = SearchSubgraph(deps)
        self.vision_subgraph  = VisionSubgraph(deps)
        self.refs_subgraph    = RefsSubgraph(deps)
        self.context_merger   = ContextMerger(deps)
        self.generator        = Generator(deps)
        self.output_validator = OutputValidator(deps)
        self.refiner          = Refiner(deps)
        self.post_processor   = PostProcessor(deps, store)
        self.execution_error  = ExecutionError()
        self.escalator        = Escalator(deps, store)
        self.user_error       = UserErrorEmitter()

    # ---- Conditional edge functions (kept as methods for symmetry) ----

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
        if state.get("output_valid"):
            return "post_process"
        if state.get("refine_count", 0) >= state.get("max_refine", 2):
            return "execution_error"
        return "refine"

    @staticmethod
    def _route_execution_error(state: AgentState) -> str:
        has_partial = any(
            state.get(k) is not None
            for k in ("search_result", "vision_result", "refs_result")
        )
        return "escalate" if has_partial else "user_error"

    # ---- Build ----

    def build(self):
        g = StateGraph(AgentState)

        # Nodes — methods are valid LangGraph callables
        g.add_node("validate_input",    self.input_validator.run)
        g.add_node("planner",           self.planner.run)
        g.add_node("executor_dispatch", self.executor.dispatch)
        g.add_node("subgraph_search",   self.search_subgraph.build())
        g.add_node("subgraph_vision",   self.vision_subgraph.build())
        g.add_node("subgraph_refs",     self.refs_subgraph.build())
        g.add_node("merge_results",     self.context_merger.run)
        g.add_node("generation",        self.generator.run)
        g.add_node("validate_output",   self.output_validator.run)
        g.add_node("refine",            self.refiner.run)
        g.add_node("post_process",      self.post_processor.run)
        g.add_node("execution_error",   self.execution_error.run)
        g.add_node("escalate",          self.escalator.run)
        g.add_node("user_error",        self.user_error.run)

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
            "subgraph_search": "subgraph_search",
            "subgraph_vision": "subgraph_vision",
            "subgraph_refs": "subgraph_refs",
        })
        g.add_edge("subgraph_search", "executor_dispatch")
        g.add_edge("subgraph_vision", "executor_dispatch")
        g.add_edge("subgraph_refs",   "executor_dispatch")

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


# Backwards-compatible convenience function
def build_agent(deps: Deps, checkpointer: Any, store: Any):
    return AgentBuilder(deps, checkpointer, store).build()
