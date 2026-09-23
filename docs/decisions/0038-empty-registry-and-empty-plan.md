# 0038 — Nothing to run is answered, not failed

- **Issue:** #38 · **PR:** #39
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.

## From “Graph flow”

**Fail-open holds only while there is something to plan with.** An **empty registry** is routed `"direct"` *structurally* — before the LLM call, so it is deterministic and free — and the fallback goes there too (#38): `"plan"` against nothing is not a wider door but a wall, the planner can only raise and the run ends in `user_error`. That is not hypothetical, it is the price of the rule below that a capability nothing can serve stays out of the registry: an agent whose capabilities are **all** conditionally registered composes an empty one legitimately.

Validation answers one of three things: a plan with steps, an **empty plan**, or an unrecoverable `NodeError` — never two at once. Empty is reachable only by every step being dropped as inapplicable (every other defect raises, and `{"steps": []}` straight from the model is still an error — indistinguishable from a truncated response). That is a fact about the request, not a broken plan, so it travels as data and **`AgentBuilder._route_after_planner` sends it to `direct_answer`**: the decision lives in the path map, never as a rescue inside the planner. Note the executor would *also* reach a terminal on an empty plan (`_all_done` is vacuously true) — but through the generator, whose prompt forbids answering outside the context it was given, so "no context" there means "I cannot answer". Which node answers is the point of the edge.
