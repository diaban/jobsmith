# 0172 — The chat's launch_job stays its own shell over the adapter's contract

- **Issue:** #172 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "the chat's stays a shell over `run_for` + `told_in_thread`" (Chat layer, Tools)

**Context.** Core split step 9 (0161) put a generic `launch_tool` in `adapters/langchain/launch.py`: create, `run_for`, then either the promoted text or `told_in_thread`. The chat's `launch_job` (`chat/tools.py`, 230 lines) kept its own shell and only ends through `told_in_thread`. The design note had said "move … generic over a `GraphSpec` plus an input builder".

**Decision.** The shell stays. What is contractual is shared and lives in one place each: promotion (`JobManager.run_for`), telling in the turn (`told_in_thread`), telling later (`JobDeliveryMiddleware`). What the chat adds is the DAG's own business around a run, and it stays in `chat/`.

**Alternatives and why not.** Building `launch_job` on `launch_tool` needs six more hooks:
- `prepare` and `create`: refusals before a job exists (`from_jobs` resolution, formats nothing renders), the approval `interrupt`, and creation through `DagJobs.create_job`, which checks the document and loads PDF in a thread;
- `before_run` and `watching`: the `job_started` notice, and the plan watch, subscribed before the start, drained after the end, released in `finally` (0086);
- `on_crash` and `on_ended`: the "crashed" message, and the verbatim answer (0083, 0085).

That turns the tool into a framework of callbacks, the thing this project does not build around LangChain's tools.

**Measured.** Both paths end in the same contract and are tested there: G5 (`tests/test_contract.py`) for the generic tool, and the in-turn and crash tests in `tests/test_chat.py` for the chat's.

**Consequences.** A second application that needs a notice or a watch around its launch revisits this. The hook it needs then is one, not six.
