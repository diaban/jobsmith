# 0202 — A task a stop caught between its writes is pending, and the resume forks its checkpoint first

- **Issue:** #202 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "A task a stop caught between its writes is pending..." (Jobs layer, under Resume)

**Context.** Found while doing #196 (0196): when a job was cancelled during the planner, 60 of 60 ended CANCELLED with nothing to resume. The cause is in LangGraph 1.2.11, and 1.2.12 does not change it. A node writes in two parts: first its state, then its route (the trigger of its conditional edge). When a task is cancelled, `PregelRunner.commit` persists the writes the task has made so far, plus `(ERROR, CancelledError)`. On resume, `_reapply_writes_to_succeeded_nodes` restores the non-ERROR writes. The task now counts as done, so its route is never written, and `next` is empty. Our facts widen the window: the manager awaits the store while the node finishes.

**Decision.** The fix lives only in `GraphRunner`, the only module that knows LangGraph. `_half_written` names the tasks whose pending writes hold both an ERROR and some output. LangGraph's control channels are private, so they are copied in with the version that was checked. `pending()` counts these tasks, so resume, `failure.retryable`, relaunch and amend all see work left. Before `resume`, `answer` and `update`, `_repaired` forks the checkpoint without its pending writes (`aupdate_state(None, as_node="__copy__")`), so the task runs again. A task stopped mid-work leaves an ERROR and nothing else; it is left alone, because LangGraph already runs it again and keeps its finished siblings.

**Alternatives and why not.**
- Detection alone, so that `pending` sees the task: measured, the resume then ends DONE with a wrong result. The route is still missing, so the next node never runs (`{"n": 2}` instead of `{"n": 20}`). That is worse than a job that cannot be resumed.
- Drop only the stopped task's writes: no checkpointer offers this publicly, so it needs code for each backend (memory, SQLite, Postgres). The fork runs the whole superstep again, finished siblings included, but only in this rare case.
- Stop only at a superstep boundary: the stop would wait for the running step, which 0177 rejected.
- Fix it in LangGraph: that is not ours to ship. Reporting it upstream is left to the owner.

**Measured.** An engine test whose conditional edge waits (`tests/test_jobs.py`), on memory and SQLite, with Postgres in CI. Before the fix, `pending()` was `()`; after it, `("first",)`, and the resume ends with `{"n": 20}`. A second test checks that a task stopped mid-work does not run its finished sibling again. The #196 race, replayed with a plain `cancel_job` as the plan appears on 60 concurrent jobs: all 60 were refused before, and all 60 ended DONE with every step after. `make mutate` killed 13 of 17 mutants at first; the sibling test killed the other 4. Falsified: without the repair, the test ends with `{"n": 2}`.

**Consequences.** The guard #196 put in `drop_steps` stays for its other reason: amended before the planner's checkpoint, the planner would run again over the change.
