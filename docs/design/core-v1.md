# Core v1: the job engine runs any LangGraph graph

Design note for the core split, written 2026-09-27 (step 3). It works through the nine findings
from step 2, where two consumers were run against `main` 68b21db. **Split without adding**: core
v1 is existing code made generic. Recovery loops (repair/replan, needs-input, orphan relaunch),
push deliverers and partial re-runs come later. Through every step the bench (chat, CLI, API,
TUI, the DAG, both agents) keeps its behaviour: `make check` stays green and the structural
evals stay at 100% on each PR.

## Layers

| package | holds | may import |
|---|---|---|
| `jobsmith/engine/` | `jobs/{models,manager,runner,repository,ownership,events}`, `core/{usage,paths,artifacts}`, `delivery.py` | nothing from `jobsmith` outside itself |
| `jobsmith/dag/` | the rest of `core/` (router, planner, executor, generation, document intent, registry, capability, profile, state, deps, `prior_jobs`), `jobs/report*.py`, `jobs/prior.py`, `clients.py` | `engine` |
| `jobsmith/agents/` | capability packs, unchanged | `dag`, `engine` |
| the bench | `chat/`, `service.py`, `app/`, `api/`, `cli/`, `tui/` | everything |

## The contract a graph signs

```python
@dataclass(frozen=True)
class GraphSpec:                          # engine/graph.py
    name: str                             # recorded on the job; resume finds its graph by it
    graph: Pregel                         # compiled with the engine's checkpointer
    result: Callable[[Any], Any] = identity   # output → JSON result; may raise JobFailed(reason)
```

- **Input.** `create_job(graph, input, *, label="", reply_to=NONE)`. `input` is whatever the graph's input schema takes, as JSON, and it is passed through untouched. A node that needs its job id calls `current_job_id()`, which reads the config's `thread_id`; that id is visible inside a sub-graph too (checked).
- **Output.** A run's output is exactly what `ainvoke` would return: the last root `values` chunk, streamed with `output_keys=graph.output_channels`. On langgraph 1.2.11 the two are identical (checked). Without `output_keys`, `values` carries the whole state. `spec.result(output)` turns that output into the job's `result`. A result that isn't JSON makes the job FAILED with the reason stated; it never crashes the persist. `create_agent` passes `lambda out: out["messages"][-1].text`.
- **Ending.** A graph that completes is DONE. One that raises is FAILED, with `error=str(e)`. A cancelled one is CANCELLED. For a *declared* failure, `result` raises `JobFailed(reason)`: the job is FAILED with that reason and an empty frontier, so it can't be resumed, as with `escalate`/`user_error` today. **Invariant: FAILED ⇒ `error` is a non-empty string.**
- **Progress.** Each node that finishes in the root namespace becomes a step: `steps[node] = ts` plus one event. The engine attaches no meaning to node names.
- **Facts.** During the run, a node calls `publish(key, value)`, a thin wrapper over `get_stream_writer()`. The engine stores each fact at `("jobs", id, "facts")/key`, last write wins, and emits an event. The runner streams with `subgraphs=True` so that facts from a capability sub-graph reach it; without that flag they are lost (checked). The DAG publishes `plan`, `formats` and `step:<cap>`. These replace `PlanReady`, `FormatsChosen`, `StepFinished`, `_TERMINAL_NODES` and the parsing of `cap_*` node names.
- **Files.** `ArtifactStore.write` writes under the job's directory and declares the file on the same channel. The engine appends it to `job.outputs` right away, so a cancel keeps it (0041's "every terminal" without collecting at each one). `role` is the graph's own word (main/alternate/annex for the DAG) and means nothing to the engine.
- **Usage.** The per-run ledger stays as it is. The runner adds a LangChain callback handler to the run config, which books the `usage_metadata` of every LangChain model call, so a ReAct graph's calls get counted. `clients.py` calls the provider SDKs directly, so nothing is counted twice. The scope is the root node name; the `cap_` prefix is stripped by the DAG view, not by the engine.

## The record

`Job v1`: `job_id · graph · status · label · input · result · error · reply_to · delivered_at ·
created_at · updated_at · steps · facts (full view only) · outputs · usage`. `label` is the one
line that listings and events show; the DAG passes the query.

What leaves the engine, and where it goes:

- `query`, `inputs`, `document_name`, `document_title`, `formats` → the DAG's input.
- `final_answer`, `terminal_kind`, `deliverable_expected`, write errors → the DAG's result.
- `plan`, `results` → facts.
- `session_id`, `announced` → `reply_to`, `delivered_at`.
- `report_path`, `ordered_results`, `step_usage` → the DAG's view of the record.

## Use cases

These don't change: create, run, start, resume, start_resume, get, list, cancel,
recover_interrupted, subscribe. Cancel, resume, lease, events and the repository were shown
generic at step 2. Two more come in by extraction:

- **`run_for(job_id, timeout) -> Job`**: `start_job` then `asyncio.wait`, moved out of `chat/tools.launch_job` (0083). The chat and `POST /jobs?wait=S` both call it. `$JOBSMITH_SYNC_TIMEOUT` remains a bench setting.
- **Delivery.** `reply_to` is JSON, `{"kind": …, …}` (step 1). When a job settles, the manager calls `deliverers[kind].deliver(job)`; a `True` answer writes `delivered_at`. There are two v1 kinds:
  - `none` is delivered at settle, since nobody is waiting. This ends "announceable to session None".
  - `session` is registered by the chat. Its `deliver` answers False; the conversation pulls with `pending_deliveries(reply_to)`, then calls `mark_delivered(id)` once the model has seen the job (today's middleware).

  `create_job` refuses an unknown kind, and resume clears `delivered_at`. The guarantee as it stands: **at least once, keyed by job id**. A crash between handoff and `delivered_at` delivers again, and the receiver dedups on the id. Push kinds need no schema change.

## The nine findings

| # | today | v1 |
|---|---|---|
| 1 input | the runner sends `{query, inputs, job_id, document_formats}` | `input` as is, plus `current_job_id()` |
| 2 end | DONE only through a `_TERMINAL_NODES` `terminal_kind`; otherwise FAILED with `error=None` | completed = DONE; `spec.result`; `JobFailed`; FAILED ⇒ error |
| 3 composition | `build_app` always builds `AgentBuilder`, and a chat model is required | `AgentDefinition.graph` (a `GraphSpec` factory) *or* a capability pack (→ `dag`); chat only when served |
| 4 promotion | only in `launch_job` | `run_for` |
| 5 delivery | `session_id` + `announced` + middleware | `reply_to` + deliverers + `delivered_at` |
| 6 progress | planner and `cap_*` only | every root node, plus facts |
| 7 usage | `LLMClient` only | plus the LangChain callback |
| 8 fields | product fields on every job | the record above |
| 9 coupling | `service.py` imports `chat.runner`; `AgentApp` = manager + chat | `JobService` + `ChatService` (the bench's `AgentService` is both); `AgentApp.chat` optional |

## Gates: tests that land with the step that makes them pass, and stay

- **G1, ReAct job.** `create_agent(ScriptedChatModel)` is wrapped in a `GraphSpec` by the test and launched through `build_app` and `run_for`. Expected: DONE, the result is the final text, usage > 0. The diff touches no file under `engine/`.
- **G2, structured job with no chat and no document.** A `StateGraph` with input `{a, b}` and output `{sum}`, `reply_to` none. Expected: DONE, `result == {"sum": 3}`, delivered at settle, no outputs. It runs in a subprocess where `jobsmith.chat`, `jobsmith.dag` and `langchain` are absent from `sys.modules`.
- **G3, imports.** An AST test enforces the "may import" column. It lands at step 4 with today's violations on an allowlist, shrinks at every step, and is empty at step 8.
- **G4, shape leak-check.** `make leak-check` also greps `engine/` for the product vocabulary (`query`, `session`, `document`, `formats`, `capabilit`, `plan`, `report`, `announc`, `terminal_kind`, `final_answer`, `deliverable`, `chat`), docstrings included. A false positive is renamed, never allowlisted.

## Order: one PR per step, each `make check` + structural evals at 100%

4. **Move.** `engine/` and `dag/` are created by `git mv` plus import rewrites, nothing else. G3 lands with its allowlist.
5. **The DAG writes its own document.** `_write_outputs` becomes a final DAG node that calls the Reporters and declares through `ArtifactStore`, so the manager stops calling Reporters. The fields that served them leave at step 6.
6. **Contract and record.** Lands `GraphSpec`, the generic runner, `Job v1` and `JobFailed`. The DAG publishes its facts and its result, and the bench's view maps the new record. G2 lands.
7. **Promotion and delivery.** Lands `run_for` and `reply_to`/deliverers/`delivered_at`; the chat and `POST /jobs?wait` switch to them.
8. **Composition and usage.** Lands `AgentDefinition.graph`, optional chat, `JobService`/`ChatService` and the LangChain callback. G1 and G4 land and the allowlist is empty. After that the process unfreezes: the single refonte record points here, and the scribe rewrites `CLAUDE.md`.

## Open questions

1. **Existing `jobs.db` records.** Start clean (recommended): the namespace moves from `jobs` to `jobs_v1`, and old rows are left untouched and unread. The alternative is a read shim of about 20 lines, plus `prior_jobs` reading the legacy `meta`.
2. **JSON shape of a job over HTTP.** Only our own CLI and TUI read it, so change it in place (recommended), without versioning.
3. **Several graphs per engine** (`graph` on the record). Recommended: yes, since G1 and G2 already make two graphs. The alternative is one graph per engine and no field.
