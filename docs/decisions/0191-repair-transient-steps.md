# 0191 — A step that failed transiently runs again before its dependents; no replanning

- **Issue:** #191 (the "repair or replan" loop of #167) · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it: `_emit_failure(retryable=True)` only for a transient failure; the executor re-runs it within `max_step_retries`, before its dependents)

**Context.** A failed capability lands in `results` with `ok: False`, and its dependents run on what is left. That happens even when the failure was transient. The default steps (`_step.py`, `research`) turned a model call that raised (network, rate limit, timeout) into "produced no output".

**Decision.** `_emit_failure(retryable=True)` marks a failure worth another try (`CapabilityResult.retryable`). The executor counts a step's runs in `completed_capabilities` (each run appends it, so no new state is needed). A retryable failure with at most `AgentProfile.max_step_retries` runs (default 1) is not done: it is dispatched again, and nothing that depends on it runs before it. The default steps mark only a model call that raised. An empty answer, nothing found or a refused file stays final.

**Alternatives and why not.**
- Retry every failure: a deterministic one comes back the same and costs another call.
- LangGraph's `RetryPolicy` on the capability node: capabilities catch their errors and emit a failure instead of raising, so it never fires. Letting them raise would lose the declared files and usage a failed step records (#41, 0002).
- Replan around a failed step: the run already degrades around it. The generator is told what failed, and `critique` checks the subject. A second planner call would need a plan diff, a policy for steps already run, and a probe budget of its own. Nothing measured asks for it, so it is left out.

**Measured.** `tests/test_executor.py`: retried before `b` and `c`, bounded, a non-transient failure not retried; end to end, the dependent reads the second result; the default step marks only a raised call. Disabling the retry fails 2 tests; removing the bound fails 1. `make check`, the structural eval tier included: green.

**Consequences.** With this, #167's four loops are all done or decided (0167, 0187, 0189, here).
