# 0211 — A plan step is known by its id, and a capability learns it from the node wrapper, not from its own code

- **Issue:** #211 · **PR:** #212 · **Status:** partially superseded by [0217](0217-two-steps-one-capability.md) (the channel is not renamed)
- **Rule in `CLAUDE.md`:** "`dag/state.py` — results … keyed by step id" (Core concepts); "`step:<id>` fact's key" (Jobs layer)

**Context.** Step 0a of `docs/design/compiler-v1.md`: a step was its capability's name everywhere (`results`, the `step:` fact, the run count, `step_finished_at`), which rules out two steps of one capability (`map`, recompilation). The results are written by `Capability._emit_success`/`_emit_failure`, which receive the step's data and nothing else, from ~50 call sites across both agents and the tests.

**Decision.** `PlanStep` has an `id` (the planner writes the capability's name while the duplicate ban holds; `step_id()` reads a plan checkpointed without one as ids = names). The executor Sends each step with `step = {id, capability}`; `Capability.state_graph` returns a `StateGraph` whose `add_node` wraps each function node to put that id in a `ContextVar` for the node's call (async and sync, signature kept by `functools.wraps`); `_emit_*` and the annexes' `produced_by` key by it, or by the capability's name when it runs outside a plan. Readers key by id: waves and retries, the context merger, plan order in the view and the document. The channel keeps its name `completed_capabilities` (it holds ids): renaming it breaks in-flight checkpoints, and belongs with 0d, where an id first differs from a name.

**Alternatives and why not.** Passing `state` to `_emit_*`: every capability, shipped or future, would have to, and one that forgot would key by name silently. Reading the id off the run's config: a Send carries no config of its own, and `checkpoint_ns` gives a task id, not a step. Re-keying in the parent's reducer: two instances collide before any reducer sees them.

**Measured.** `tests/test_step_identity.py`: ids that are not names flow executor → Send → sub-graph → results, run count and `step:<id>` fact, from an async and a sync node; a plan without ids and a capability outside a plan key by name; the view orders and drops by id. Falsified: keying `_emit_*` by name fails 1; sending no identity fails 2; leaving sync nodes unwrapped fails 1. Full suite unchanged (911 passed).

**Consequences.** No behaviour change until 0d lifts the ban. `drop_steps`, the TUI, the REPL and the chat still read `capability` (0b); `UPSTREAM` reading still goes by name (0c); usage per instance still collides on `cap_<name>` (0d).
