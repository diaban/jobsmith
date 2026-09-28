# 0168 — Re-running part of a finished DAG job is a new job built on it, never a new attempt

- **Issue:** #168 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it: a partial re-run is a new job with `from_jobs`; DONE stays unresumable)

**Context.** `resume_job` refuses a DONE job (`_begin_resume`). Pushing a finished run further, for example "redo the analysis with this correction" or "add a critique", needed a decision: a new attempt of the same job, or a new job built on the old one.

**Decision.** A new job, with the old one in `from_jobs` (0074). `prior_jobs` hands the new run the old answer and each step's material in plan order, bounded and quotable by id, so the new plan may run only what is stale (e.g. `prior_jobs → analysis → critique`) and does not pay again for retrieval. Its ending is delivered as any job's is. Nothing is built: the path exists.

**Alternatives and why not.** A new attempt of the same job, re-entering its checkpoint with some results invalidated (`aupdate_state`, then run from `executor_dispatch`):
- The record says what the job was given. A correction is a new request, and `attempt` counts tries at one request: a resume, an answer (0167). It is also the key an ending is told by (#170), so "a retry" and "another question" would share it.
- `completed_capabilities` is append-only. The executor counts retries in it (0191), `results`/`final_answer`/`refine_count` would need rewriting past their reducers, and dependents would have to be computed from the plan.
- The deliverable would be overwritten in place under the same name, so the answer the first run gave would be lost.

**Measured.** Nothing new: the path is `launch_job(..., from_jobs=[id])`, tested in `tests/test_prior_jobs.py` and `tests/test_references.py`.

**Consequences.** Saying which steps of the old run to carry (all of them travel today, the stale analysis included, marked `<id>#analysis`) is not supported. That becomes an issue if a run shows the stale material misleading the new one. A graph agent (0165) builds on an earlier job through its own input.
