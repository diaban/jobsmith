# 0165 — A graph agent is served through the engine's own port, on its own paths

- **Issue:** #165 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "A graph agent has its own port" (The inbound port) and "Engine" (HTTP API) — added by the scribe, 2026-09-29

**Context.** Since 0161 an agent may be a graph of its own, but only in Python: `AgentApp.service()` needed `.manager`, which raises for a graph agent, so `jobsmith serve --agent <graph agent>` could not start. The only port, `JobService`, speaks the DAG's request (query, document name/title/formats).

**Decision.** `EngineService` / `LocalEngineService` (`service.py`) is the engine's port as dicts: `launch_job(input, graph, label, reply_to, wait)`, get, list (`status`, `reply_to`), cancel, resume, subscribe. `AgentApp.service()` answers it for a graph agent and the DAG + chat port otherwise, so `serve` is unchanged. The API serves it on `/engine/jobs…` plus the shared `/health` and `/events`; a capability pack's routes are untouched.

**Alternatives and why not.**
- The engine's shape on `/jobs` for a graph agent: two bodies on one path, and a client of the wrong kind gets a body read the wrong way instead of a 404.
- The engine door on every deployment, the DAG's included: `DagJobs.create_job` is where a document name is checked (`document_stem`), and `write_document` does not check it again. A raw door onto the DAG graph would write a file wherever its input says.
- `EngineService` as a third half of `AgentService`: the method names are the same with other shapes (`get_job` is `DagJob.to_dict` on one side, `Job.to_dict` on the other).

**Measured.** `tests/test_any_graph.py`: a graph agent through the API (launch with `?wait`, get, list, 409 resume, 400 unknown graph, no `/jobs`), and `jobsmith serve --agent count` with uvicorn replaced by one request through the app it was given. Both fail when `service()` always answers the DAG's port. `make mutate`: 251 mutants, 125 killed; the survivors were type annotations, `@abstractmethod`, `/events` (tested under uvicorn in `test_service.py`), and the status filter, cancel and 404 routes, which the test now covers.

**Consequences.** No remote backing yet: the CLI's `DaemonClient` still speaks only the DAG's port, so `jobsmith run`/`jobs` do not reach a graph agent's daemon.
