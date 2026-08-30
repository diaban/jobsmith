# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`agent_oo` is a LangGraph agent for a banking assistant, built with an **object-oriented** node pattern (as opposed to the more common factory-function style). Every graph step — nodes, routers, and sub-graphs — is a class instance that owns its own `Deps` and configuration, and exposes bound methods that LangGraph registers directly as callables.

There is no build system, lint config, or test suite in this directory yet (no `pyproject.toml`, `requirements.txt`, or `tests/`). Treat any tooling commands as something to set up, not something to assume exists.

## Architecture

### The OO pattern

- A class's `__init__` takes `Deps` (and any tunables like `max_retries`, `top_k`) and stores them as instance attributes.
- Node logic is an **async instance method** (e.g. `self.planner.run`), registered with `g.add_node("planner", self.planner.run)`.
- Conditional-edge routers are **sync instance methods** (or `@staticmethod`s), registered with `g.add_conditional_edges(...)`.
- Sub-graphs are classes whose instances are *not* callable themselves — they expose `.build()`, which returns a compiled `CompiledGraph` that the parent graph adds as a single node (`g.add_node("subgraph_search", self.search_subgraph.build())`).
- `AgentBuilder` (`graph.py`) is the composition root: it instantiates every step, wires nodes/edges, and holds references to each instance (useful for tests/observability — e.g. `builder.planner.SYSTEM_PROMPT`, or swapping an instance before `.build()`).

### Dependency injection (`deps.py`)

`Deps` is a frozen dataclass aggregating three `Protocol`-typed clients: `SearchEngine`, `OpenAIClient` (chat + vision), `S3Client`. No DI framework — factories/classes just close over a `Deps` instance, which makes mocking trivial in tests. The LangGraph `checkpointer` and `store` are passed separately (not part of `Deps`) since their types come directly from `langgraph`.

### State design (`state.py`)

- `AgentState` is the **global** parent-graph state (`TypedDict, total=False`).
- Each sub-graph declares its **own** private state (`SearchSubState`, `VisionSubState`, `RefsSubState`) to keep intermediate values (retry counters, raw API responses) out of the global state.
- Fields written by parallel/fanned-out branches use explicit reducers: `completed_subgraphs: Annotated[list[str], add]` and `errors: Annotated[list[NodeError], add]`. Every other field has exactly one writer, so no reducer is needed.
- `NodeError["recoverable"]` is the key control-flow signal: `False` forces a hard stop into `execution_error`; `True` just gets logged and the graph continues.

### Graph flow (`graph.py`)

```
validate_input → planner → executor_dispatch ⇄ {subgraph_search, subgraph_vision, subgraph_refs}
                                  ↓ (all done)
                            merge_results → generation → validate_output
                                                            ↓ fail (retries left)   ↓ pass
                                                          refine → generation    post_process → END
```

- **Planner** (`nodes/planner.py`) asks the LLM for a JSON DAG of which sub-graphs to run (`search` / `vision` / `refs`) and their dependencies, then validates it: allowed subgraph names, no duplicates, `vision` dropped if no image was supplied, and a Kahn's-algorithm cycle check.
- **Executor** (`nodes/executor.py`) does *not* execute anything itself — `dispatch` is a pass-through node; the real logic is in `route`, a router that computes which subgraph steps are "ready" (all `depends_on` satisfied, not yet in `completed_subgraphs`) and returns a `list[langgraph.types.Send]` to fan them out in parallel. Each sub-graph, on completion, loops back to `executor_dispatch` so the next wave can be computed — this is how a dependency DAG is executed wave-by-wave without a fixed topological schedule baked into the graph.
- **Sub-graphs** (`subgraphs/search.py`, `vision.py`, `refs.py`) each follow the same internal shape: do the work → on failure, retry with backoff → fall back to a degraded path → always end at one of `emit_success` / `emit_failure`, which appends to `completed_subgraphs` (and `errors` if failed, marked `recoverable: True`). This means a single sub-graph failing does not halt the parent graph; it degrades gracefully and is aggregated later.
- **Generation pipeline** (`nodes/generation.py`): `ContextMerger` (deterministic, formats search/vision/refs results into one context string) → `Generator` (LLM call) → `OutputValidator` (nodes/validate.py; checks for empty/too-short answers and missing `[doc_id]` citations when search results exist) → on failure, `Refiner` rewrites the draft against the same context and loops back to `generation`, up to `state["max_refine"]` times → `PostProcessor` persists the final answer to the long-term `store` and marks `terminal_kind: "answer"`.
- **Error/escalation path** (`nodes/errors.py`): any unrecoverable error or exhausted refine budget routes to `execution_error`, which then routes to `escalate` (if any partial sub-graph result exists — persists context to the store for a human analyst) or `user_error` (nothing recoverable at all). Both are terminal.

### Notes for future work

- User-facing error/escalation strings are in **French** (e.g. `"Votre requête est vide."`, `"Une erreur est survenue."`) — match that when adding new user-facing messages.
- `nodes/__init__.py` and `subgraphs/__init__.py` are currently empty — nothing is re-exported at the package level; import directly from the submodule (`from .nodes.planner import Planner`).
- `graph.py` keeps a `build_agent(deps, checkpointer, store)` free function for backwards compatibility with a pre-OO factory-function API; prefer `AgentBuilder` directly in new code.
