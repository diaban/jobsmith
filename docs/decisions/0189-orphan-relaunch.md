# 0189 — An interrupted job is relaunched only where its graph allows it, by the daemon, once

- **Issue:** #189 (the "orphan relaunch" loop of #167) · **Status:** accepted
- **Rule in `CLAUDE.md`:** "`relaunch_interrupted()`..." (composition root, under Interrupted jobs) — added by the scribe, 2026-09-29

**Context.** A job whose process died is settled FAILED at the next start (`failure.kind == "interrupted"`, 0187) and waits for someone to resume it. A resume runs the interrupted node again from its start. The engine cannot tell whether that is harmless.

**Decision.** `GraphSpec.relaunch: int = 0`: how many times the graph lets such a job be resumed with nobody asking. `JobManager.relaunch_interrupted()` resumes every FAILED `interrupted` retryable job whose attempt is at most that bound, so a job that kills its process stops being relaunched. It is called by `jobsmith serve` only. `recover_interrupted` still only settles, at every start. On a shared store each candidate is claimed first, in the same way as `_take_over`: the lease is written, one heartbeat waited, then read again, and a process that wrote after us owns it.

**Alternatives and why not.**
- Relaunch inside `recover_interrupted`: every short-lived embedded command (`jobsmith jobs`) would resume the job, then stop it again on exit, spending an attempt and re-running a node for nothing.
- Relaunch by default, or decided from the failure kind: only the graph knows whether its nodes may run twice (an LLM call may, a payment may not).
- No claim: a resume checks the status, then writes RUNNING, with no compare-and-set in between. Measured with that window widened (a slow `pending()`): without the claim both managers relaunched (the test failed 2 of 2, and a third run hung on two resumes of one checkpoint); with it, one did, 5 of 5.

**Measured.** `tests/test_relaunch.py`: relaunched to DONE at attempt 2; left FAILED when the graph says never, or past its bound (the bound removed fails it); two managers on one SQLite file relaunch it once; `serve` relaunches before it serves.

**Consequences.** Neither agent in the tree opts in (the DAG keeps `relaunch=0`); a graph agent does, in its `GraphSpec`.
