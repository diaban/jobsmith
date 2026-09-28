# 0167 — A run paused at an interrupt waits for an answer, with its own status

- **Issue:** #167 (the "needs input" loop; the three others stay open there) · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it: an `interrupt()` is `needs_input` + `asked`, answered by `answer_job`; not an ending, not delivered)

**Context.** A graph paused at a LangGraph `interrupt()` ended FAILED with "its result is not JSON". The runner took the run's last root `values` for its result, and that chunk carries the `Interrupt` objects (checked on langgraph 1.2.11: `updates` yields `{"__interrupt__": (Interrupt(value, id),)}`, then `values` repeats it).

**Decision.** `runner.py` reads `__interrupt__` and yields `Interrupted(asked)` in place of an `Output`. `_conclude` makes the job `needs_input`, with `Job.asked` holding the interrupt values; a value that is not JSON is kept as its `repr`, so the job stays readable and can still be answered. `answer_job` / `start_answer` re-enter the thread with `Command(resume=answer)` through `_begin_resume(expect=(NEEDS_INPUT,))`, as a new attempt, with `asked` cleared. `resume_job` does not take a paused job. `cancel_job` stops one where it stands, since no process is running it. The engine port has `answer_job` and the API `POST /engine/jobs/{id}/answer`.

**Alternatives and why not.**
- Deliver the pause to the return address as if it were an ending: the told-by-attempt key (#170) would then need a second ending inside one attempt, and a cancelled pause would never be told. The pause is read from the record (`run_for`, `get_job`); pushing it to the address is left open.
- Resume with `None` as the answer path: the interrupted node runs again and asks the same thing again.

**Measured.** `tests/test_needs_input.py`: pause, then answer, then DONE with the answer in the result, at attempt 2 and delivered. Also tested: the refusals both ways, a cancelled pause that is told, and the engine door end to end. With the runner's `Interrupted` branch disabled, all 4 tests fail with the old "is not JSON".

**Consequences.** The DAG never interrupts, so the chat, CLI and TUI never see `needs_input`; they only render the DAG's statuses.
