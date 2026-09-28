# 0161 — The job engine is the product: it runs any LangGraph graph and tells each ending once

- **Issue:** #161 · **PR:** #143 (design note), #144–#160 and the 9c/9d PRs · **Status:** accepted
- **Rule in `CLAUDE.md`:** the engine's lines (the `engine/` layer, delivery, usage, the G4 leakage gate, an agent that is a graph)

**Context.** On 2026-09-27, three external reviews found that the bench (chat, planner DAG, documents) had shaped the core. Two consumers run against `main` 68b21db, a `create_agent` job and a structured job with no chat, hit nine workarounds, from input to coupling. The job ↔ caller contract, the differentiator, was locked inside the chat.

**Decision.** Core v1 is the existing code made generic, with nothing added. `engine/` runs any graph through a `GraphSpec`: the input goes in as given, the result is what `ainvoke` returns, and `JobFailed` declares a failure. It records facts and root steps, promotes on the clock (`run_for`), delivers to a JSON `reply_to` (deliverers `none` and `Pulled`), and counts LangChain calls. The DAG, files (`artifacts/`), the chat and the contract (`adapters/langchain/`) sit on top. Design and an as-built line per step: [`docs/design/core-v1.md`](../design/core-v1.md).

**Alternatives and why not.**
- A return address as a callable: it does not survive a restart.
- Nested or dotted filters on the address: nested fails on SQLite, dotted finds nothing in memory (measured). Hence the flat `reply_key`.
- Marking inside the model call: a crash after the mark loses the ending. Hence `delivered_jobs`, checkpointed with the answer, with the mark written after.
- A shim for old records: the user decided to start clean under `jobs_v1`.

**Measured.** The gates are in the suite, each falsified by breaking its property. The Postgres paths never ran: there is no DSN locally or in CI.
- G1: ReAct via `build_app` + `run_for`, with usage counted. G2: a structured graph in a fresh interpreter, delivered as it settles.
- G3: imports per layer, empty allowlist. G4: `ENGINE_WORDS`. G5: the adapter and the engine alone; promoted; a crash on each side of the checkpoint; told once in the thread.

**Consequences.** Out of v1: serving a graph agent over the daemon or the API (`JobService` speaks the DAG's shape), push deliverers, and recovery loops. The DAG keeps registering `session`, because `POST /jobs?session_id=` needs it with no chat. The process unfreezes.
