# 0215 — A capability reads another's material by capability, from every step that ran it, in plan order

- **Issue:** #215 · **PR:** #216 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "A capability reads another's material by `results_of` … never `results[name]`" (Core concepts)

**Context.** Step 0c of `docs/design/compiler-v1.md`. After 0a/0b (0211, 0213) results are keyed by step id, but five readers took another step's result as `results[name]`: `SingleStepCapability._material` (`UPSTREAM`), `research`'s grounding and refusals, `critique`, `slide_deck`, the banking citation rule. With two steps of one capability they would read one key and miss the other; with an id that is not a name, nothing.

**Decision.** `dag.state.results_of(state, capability)` returns every successful result of a step that runs it, in plan order, with the step's id; a result keyed by the name that no plan step accounts for (a capability run outside a plan) still counts, last. Every reader goes through it and labels each block by step id; `_material` still takes the first upstream capability, in priority order, that left material — now from all its steps. `CapabilityBaseState` declares `plan`: a sub-graph is entered with the keys its schema declares and no other, so without it a step sees no plan and `results_of` finds nothing. This is the note's by-op fallback; explicit references replace it at step 1.

**Alternatives and why not.** The latest step only (the note's first wording): drops material silently, the one thing the pack's readers are written never to do (0060, 0081). Recording the capability inside each result and filtering `results`: loses plan order, since `results` fills in waves. Reading `plan` from the parent: a capability sub-graph has no parent state, only what it was sent.

**Measured.** `tests/test_step_identity.py`: plan order and the ok filter; the fallback outside a plan; a `Send` from the executor carries the plan into the sub-graph, which reads both upstream steps; `analysis` reads two `research` steps labelled `notes_a`, `notes_b`. Falsified: removing `plan` from the base state fails 1; `_material` keeping the first block fails 1; `results_of` in dict order fails 1. Full suite unchanged.

**Consequences.** No behaviour change while ids are names (labels read as before). The duplicate ban holds until 0d.
