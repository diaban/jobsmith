# 0177 — A running job is amended by stop, checkpoint update, run on; the DAG drops steps with it

- **Issue:** #177 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it: `JobManager.amend_job` = stop + `runner.update` + resume, the stop never delivered; `DagJobs.drop_steps` is the DAG's meaning)

**Context.** 0086 left interjection on an announced plan ("skip the critique") out of scope. #177 asked two questions: is it the DAG's feature or the engine's, and what happens to a step that is already running?

**Decision.** Both, split by who knows what. The engine offers `amend_job(job_id, update, facts=)`. It stops the job where it stands, writes `update` into its checkpoint through the graph's reducers (`GraphRunner.update` = `aupdate_state`), records `facts`, and runs the job on as a new attempt. The DAG gives it a meaning: `DagJobs.drop_steps(job_id, names)` writes the plan without those steps and with every `depends_on` on them pruned (`planner.without_steps`), and republishes the `plan` fact. It refuses a step not in the plan, a step already finished, and a plan left empty. A step running when the amendment arrives is stopped. It runs again from its start unless the change removed it: that is the price of a cancel, and LangGraph offers no stop at the next superstep boundary without the graph's cooperation. The stop is not an ending. `_amending` keeps it from being delivered or listed as pending, since a chat or webhook told "cancelled" would be told something false.

**Alternatives and why not.**
- Write the checkpoint of a run that is still going: its next checkpoint write overwrites the change.
- DAG only, with the executor polling an amendment channel between waves: graph nodes stay job-agnostic ("they do not know a Job exists"), and the engine would need a second input channel beside the checkpoint.
- Wait for the running wave to end before amending: no signal to stop at a boundary exists short of `interrupt_before` on every node.

**Measured.** Probe on the real DAG (stub steps): cancelled while `analysis` ran, the plan updated without `critique`, resumed; the run ended with an answer and no `critique`. `tests/test_amend.py`: the engine re-runs the stopped step on the change and delivers only `done`; `drop_steps` gives its three refusals, `critique` never runs, `analysis` runs twice, and the plan fact has two steps. Removing the `_amending` guard fails a test; amending with `{}` fails a test.

**Consequences.** The chat and the API do not reach `drop_steps` yet: #196. The probe found #194, where every capability echoed `completed_capabilities` back.
