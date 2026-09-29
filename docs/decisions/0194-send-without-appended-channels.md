# 0194 — A capability is Sent the parent state without the append-only channels

- **Issue:** #194 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "`Send` also strips the append-only channels" (Critical invariant: capability output schema) — added by the scribe, 2026-09-29

**Context.** The executor fanned out with `Send(node, state)`. A capability sub-graph is entered with what it is sent, and returns its output channels as they stand at its end. `completed_capabilities` and `errors` are `add` channels (`CapabilityOutputState`), so each step handed the parent back what it had been given plus its own entry, and the parent appended all of it. Measured on a plain `research → analysis → critique` run: `['research', 'research', 'analysis', 'research', 'research', 'analysis', 'critique']`. It was harmless while every reader took `set(...)`. Since 0191 a step's runs are counted there to bound its retries: with `max_step_retries=2`, one retry already counted as three runs.

**Decision.** `Executor.route` sends each capability the state without `_APPENDED` (`completed_capabilities`, `errors`). No capability reads either. `results` stays in the payload: its reducer is a dict union, so an echo changes nothing.

**Alternatives and why not.**
- Deduplicate in the reducer: the count of runs is the information 0191 needs.
- Drop the two channels from the sub-graph's input schema: `CapabilityBaseState` extends the output schema on purpose (the reducers must match the parent's), and every capability's private state inherits from it.

**Measured.** `tests/test_executor.py::test_each_run_of_a_step_is_counted_once` (a → b → c, b retryable, `max_step_retries=2`) failed before the fix, with 10 extra entries, and passes after: `['a', 'b', 'b', 'b', 'c']`, 3 runs, 3 errors. `make check`: green.
