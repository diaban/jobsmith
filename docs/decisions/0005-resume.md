# 0005 — A stopped job resumes from its checkpoint

- **Issue:** #5 · **PR:** #22
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0010](0010-cross-process-ownership.md)

## From “Jobs layer (`jobs/`)”

**Resume** (`resume_job` / `start_resume`, `POST /jobs/{id}/resume`, `jobsmith resume <id-prefix>`, `/resume` in the REPL): `runner.resume()` re-enters the thread with `None` as input, LangGraph's "carry on" — the last completed superstep is replayed from the checkpoint and only the still-pending tasks run. Finished steps are neither re-run nor re-emitted, so the manager keeps the results the repository loaded; the plan likewise comes back from the store, not the stream. Two gates, both needed: the status must be CANCELLED or FAILED (DONE is the *other* half of #5; QUEUED wants `run_job`), **and** `runner.pending(job_id)` must be non-empty. Status alone is not enough — a job that FAILED because a node raised reached `escalate`/`user_error`, so its frontier is empty and re-entering would replay one superstep and silently do nothing; ditto a job cancelled before it ever started (no checkpoint at all — `astream(None)` there raises `EmptyInputError`). Both are refused with `ValueError`, as `run_job` refuses a wrong status. A resumed attempt is driven by the same `_drive` as a first one (same persistence, events, reporting), and its usage ledger is **seeded with what earlier attempts spent**, so `Job.usage` stays the job's total cost rather than the last attempt's. `_begin_resume` clears **two** stale facts about the stopped attempt: `job.error` (its message no longer describes the job) and `job.announced` (a job picked back up is news again). The second is load-bearing now that a cancellation is announced: without it, a job announced as stopped and then resumed to DONE is filtered out of `list_finished_unannounced` and its answer never reaches the conversation that asked for it.

What is **not** implemented: re-running a subset of the DAG of a job that already finished. That needs a way to say which results are stale and how the plan is amended — see the note on issue #5.
