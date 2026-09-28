# 0187 — A FAILED job says why as data, and "retryable" is what a resume answers

- **Issue:** #187 (the "structured failure" loop of #167) · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it: every FAILED goes through `_fail`; `retryable` = a live frontier, the resume gate's own test)

**Context.** A FAILED job carried only `error`, a string. The only way to learn whether it could be resumed, and where it stopped, was to try `resume_job` or read the prose.

**Decision.** `Job.failure = {"kind", "pending", "retryable"}` sits next to `error`. Every FAILED written by the engine goes through `_fail`. The kind (`FailureKind`) is one of `raised` (with `exception`, its type), `interrupted` (via `recover_interrupted` → `_settle`), `declared` (`JobFailed`), `no_result` or `unreadable`. `pending` is what `runner.pending()` says a resume would run, and `()` when the graph cannot say. `retryable` is `bool(pending)`, the same test `_begin_resume` applies, so the two cannot disagree. A resume clears it. Also fixed on the way: an exception with an empty message made `error` empty. It is now the exception's type name.

**Alternatives and why not.**
- A retryable flag decided per kind (`raised` → yes): a node that raised with no checkpointer has nothing to resume, and the flag would promise a resume the gate then refuses.
- A single `step`: one superstep can fail several nodes at once. `pending` lists them, in the gate's own words.

**Measured.** `tests/test_failure.py` covers raised, declared and unreadable, each checked against what `start_resume` answers, plus an interrupted job whose checkpoint stops before `second` and that resumes to DONE. Forcing `retryable` to False fails 2 tests. Returning no `pending` fails 2 tests.

**Consequences.** Orphan relaunch (the next loop of #167) can read `failure.kind == "interrupted"` and `retryable` instead of parsing `error`. The DAG's own dead ends (`escalate`, `user_error`) are DAG results rather than engine failures. They are not changed here.
