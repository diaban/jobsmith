# 0217 — One capability may run as several steps, told apart by id, each counting its own usage through a scope the node names

- **Issue:** #217 · **PR:** #218 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "scope = a node's `usage_scope` (a step's: `cap_<id>`)" (Core concepts, `engine/usage.py`); "**ids unique** (default: the name)" (Graph flow, Planner); partially supersedes [0211](0211-step-identity.md) (the channel's rename)

**Context.** Step 0d of `docs/design/compiler-v1.md`, settled point 1. After 0a–0c every reader keys by step id, but usage is booked under the root node of `checkpoint_ns`, which drops the task id: two steps of one capability share `cap_<name>` and each reads back the sum. The planner refused a capability twice. The planner prompt asks for no ids and no arguments until step 1, so two steps of one capability from a model today would be the same work twice.

**Decision.** The engine offers `usage_scope(name)` (a `ContextVar` `current_scope()` reads first; generic, G4 clean), and the node wrapper of 0211 enters `usage_scope(cap_<step id>)` for a step in a plan; `_usage_meta` reads that scope. The ban moves from capabilities to ids: a step with no id has its capability's name, so a plan with none still refuses a capability twice ("duplicate step id"); a plan that gives distinct ids runs it as several steps. Ids match `^[a-z][a-z0-9_]*$`; `depends_on` names ids. The channel `completed_capabilities` keeps its name: renaming buys a name and breaks resuming a checkpoint in flight (settled point 2).

**Alternatives and why not.** Keeping the task id in the scope: changes the ledger's shape for every graph. Per-capability usage only: drops the per-step figure. Lifting the ban outright: a model repeating a step by mistake would run it twice instead of being refused. Renaming with "start clean": detecting a pre-upgrade checkpoint to say so is code for a label.

**Measured.** Gate C0 (`tests/test_step_identity.py`): through the engine, a plan with two `spend` steps (`small`, `large`) runs both; `step_finished_at`, results and usage kept apart — 6 and 100 output tokens, 2 calls each, booked once by `record_usage` and once by the runner's LangChain callback: the note's *(to check)* holds, the scope reaches a callback inside a sub-graph run by a Send. A plan with no ids still refuses `echo` twice. `tests/test_usage.py`: two interleaved branches book apart. Falsified: `current_scope` ignoring the named scope fails 1; the wrapper entering no scope fails 1; the ban back on capabilities fails 1.

**Consequences.** No behaviour change for a model that writes no ids. Step 1's IR gives the planner ids; step 2's `map` instances (`reads[3]`) will need `[`/`]` in an id, which `_ID_RE` refuses today — widened there.
