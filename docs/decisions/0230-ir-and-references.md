# 0230 — The program is an IR stored as JSON beside today's fields; the interpreter resolves references, and a step whose arguments cannot be had fails alone

- **Issue:** #230 · **PR:** #231 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "**`dag/ir.py`** — `Program` = `steps` … + `result.answer`" and "It resolves references before the Send; an unresolvable one fails its step, recoverably (`dispatch`)" (Core concepts, Graph flow)

**Context.** Step 1b of `docs/design/compiler-v1.md`, after 0228 (a program is `steps` and `result`). Today's plans are `capability` + `depends_on`, read by the executor, the views, the TUI, the REPL, the chat and `drop_steps`; capabilities take no arguments.

**Decision.** `dag/ir.py`: `Program`, `Step` (`id`, `op`, `args`, `after`), `AnswerSlot` (`mode`, `instruction`, `material`), references parsed by one pattern and never interpolated, dependencies derived from references then `after`. Models validate; the `plan` channel holds JSON (`Program.stored()`), each step also carrying `capability` (its op) and the derived `depends_on`, so no reader changes in this step. Today's plans read as programs with no args, so the planner's prompt is untouched (1c). The interpreter resolves a ready step's references (results, inputs) and its op's `input_model`, when declared, checks them; a step whose arguments cannot be had is failed by the `dispatch` node, recoverably, and its dependents with it, so the router Sends only a step that can run. The answer slot's `material`, when it names steps, is what generation reads, in that order. `dag/ops.py`: `OpSpec` (alias `CapabilitySpec`), `Effects` declared, not enforced. `pydantic` becomes a declared dependency (it was transitive).

**Alternatives and why not.** Pydantic objects in the state: the checkpoint, the `plan` fact and HTTP would carry classes, against "the IR is data". Moving every reader to `op` and derived dependencies now: a wide diff for no behaviour; they move when they need `args`. Failing the job on an unresolvable reference: one bad edge would sink every other result (the run degrades today, 0191). Sending the step and letting it fail itself: every op would have to check its own arguments.

**Measured.** No model involved. Falsified, one at a time against `tests/test_ir.py`, `test_executor.py`, `test_planner.py`: `dispatch` failing nothing fails 3; generation ignoring the slot's material fails 1; references giving no dependency fail 5; dropping a step leaving `after` fails 1; `input_model` unchecked fails 1.

**Consequences.** Through the engine, a step's argument comes from a reference (gate C1, its first half). Step 1c makes the planner write the IR; `step_args(state)` is how an op reads its arguments; renaming `plan`/`PlanStep` waits until their readers move.
