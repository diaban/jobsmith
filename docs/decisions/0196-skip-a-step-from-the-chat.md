# 0196 — The chat skips a step with a tool of its own; the port and the API answer it like a resume

- **Issue:** #196 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "Reached as `drop_steps` on `JobService`..." (Jobs layer, under `amend_job`)

**Context.** Only Python could call `DagJobs.drop_steps` (0177). A probe on `main` (gpt-5-nano, 5 phrasings of "skip the critique" × 5, with a job running in the background) found that the chat **cancelled the whole job** 15/25 times and launched a new job 7/25 times: `cancel_job` was the only tool that could touch a running job.

**Decision.** The chat gets a tool, `skip_steps(job_id_prefix, steps)`, scoped by `_find` like the other job tools. When the DAG refuses with a `ValueError`, the model gets "Nothing was skipped: " plus the DAG's own words (the plan's steps, "cancel it instead"), which it can act on. On success it gets the plan that is left. The system prompt gains one line: to leave part of a running task out, call `skip_steps`, since `cancel_job` stops all of it. The banking prompt gains the same line in French. On the port, `JobService.drop_steps` answers the way `resume_job` does, through the same `_resumed`: `{"job_id", "status"}` on success, `{"status", "error"}` on a refusal. The API serves it as `POST /jobs/{id}/drop {steps}`: 404 for an unknown job, 409 for a refusal. `DaemonClient` maps both codes back through `_refusable`, which it now shares with resume.

**A plan the checkpoint does not hold yet is not amended.** The planner publishes the `plan` fact from inside its node, before its checkpoint is written. So `drop_steps` refuses while `JobManager.pending` still names `planner`, and it refuses before anything is stopped. In that window, 60 of 60 concurrent jobs that were amended ended CANCELLED with nothing left to resume. The stop lands while the manager is persisting the fact; the planner finishes meanwhile, and LangGraph records it with both its result and a `CancelledError`. With the guard, all 60 are refused and run on untouched. Amended after one step had finished, 60 of 60 ended DONE without the dropped step. The same race can hit any cancel that lands while a node's fact is being persisted: that is #202.

**Alternatives and why not.**
- Call the tool `drop_steps`: users say "skip", so the tool uses their word. The port keeps the DAG's name.
- A refusal raised as an exception across the port: `resume_job` already answers refusals as a body, and a front-end should handle one shape.
- Name the pending steps in the progress notice so the model need not look them up: that is a second prompt change. When the model misses on its first call, it calls `job_status`, which is a correct first step, and a refusal lists the plan anyway.

**Measured.** `make probe NODE=chat READ=tool CASES=evals/probes/chat.json`. The probe gained a `chat` pseudo-node: the session's prompt and tools, plus a notice and a history. `probe-compare.sh` now runs the branch's probe on both sides. Results on the branch, n=10: the skip phrasings got 38/50 `skip_steps`, 12 `job_status` and 0 `cancel_job`. "Cancel that job" and "stop it all" got 20/20 `cancel_job`. A new task got 6/10 `launch_job`; `main`'s side of this case errored in the parallel run, and run alone `main` scored 3/5. "How is it going?" never got `skip_steps`. Tests: the tool's success and refusal (`test_chat.py`), both backings (`test_service.py`), 409 and 404 (`test_api.py`). `make mutate` killed 10 of 12 mutants. The two survivors remove `@tool` and `@abstractmethod`, and neither changes behaviour.

**Consequences.** There is no CLI command or REPL `/skip`: in the terminal, the chat covers it. `EngineService` still has no door to amend a job.
