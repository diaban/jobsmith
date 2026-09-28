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
| `jobsmith/engine/` | `jobs/{models,manager,runner,repository,ownership,events}`, `core/usage`, `delivery.py` | nothing from `jobsmith` outside itself |
| `jobsmith/artifacts/` | **optional, outside the core** (decided 2026-09-27): the per-job file store, path safety (`core/{artifacts,paths}`), and `declare`, which publishes a file as a fact | `engine` |
| `jobsmith/dag/` | the rest of `core/` (router, planner, executor, generation, document intent, registry, capability, profile, state, deps, `prior_jobs`), `jobs/report*.py`, `jobs/prior.py`, `clients.py` | `engine` |
| `jobsmith/adapters/langchain/` | **the job ↔ conversation contract, built at step 9**: the launch tool (promotion within the turn, verbatim answer 0083/0085, plan notice 0086), the notification middleware (0006), the delivered-ids channel | `engine`, `langchain` |
| `jobsmith/agents/` | capability packs, unchanged | `dag`, `engine` |
| the bench | `chat/`, `service.py`, `app/`, `api/`, `cli/`, `tui/`: persona, prompt, `create_agent` assembly, the DAG's tool arguments (`source_files`, `from_jobs`, formats) | everything |

That contract is what sets the framework apart: a job handed back to someone, in a precise
shape, not just a job run durably. The split does not stop until step 9 has taken it out of
the bench.

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
- **Files are facts** (decided 2026-09-27, following the external reviews: a job engine that knows files and deliverables is less agnostic). Declaring a file publishes the fact `artifact:<declarer>:<path>` (title, media type, role, and `missing` when the file was not on disk as it was declared) through `jobsmith/artifacts/`, which also holds the store and path safety, outside the core. Facts persist as they arrive, so a cancel keeps them (0041's "every terminal" without collecting at each one). Listing a job's files, ordering them and saying one is missing belong to the reader, which for the DAG is its view. The engine has no `outputs` and no word for a file.
- **Usage.** The per-run ledger stays as it is. The runner adds a LangChain callback handler to the run config, which books the `usage_metadata` of every LangChain model call, so a ReAct graph's calls get counted. `clients.py` calls the provider SDKs directly, so nothing is counted twice. The scope is the root node name; the `cap_` prefix is stripped by the DAG view, not by the engine.

## The record

`Job v1`: `job_id · graph · status · label · input · result · error · reply_to · delivered_at ·
created_at · updated_at · steps · facts (full view only) · usage`. `label` is the one
line that listings and events show; the DAG passes the query.

What leaves the engine, and where it goes:

- `query`, `inputs`, `document_name`, `document_title`, `formats` → the DAG's input.
- `final_answer`, `terminal_kind`, `deliverable_expected`, write errors → the DAG's result.
- `plan`, `results` → facts.
- `session_id`, `announced` → `reply_to`, `delivered_at`.
- `outputs`, `report_path`, `ordered_results`, `step_usage` → the DAG's view of the record, `outputs` from the `artifact:` facts.

## Use cases

These don't change: create, run, start, resume, start_resume, get, list, cancel,
recover_interrupted, subscribe. Cancel, resume, lease, events and the repository were shown
generic at step 2. Two more come in by extraction:

- **`run_for(job_id, timeout) -> Job`**: `start_job` then `asyncio.wait`, moved out of `chat/tools.launch_job` (0083). The chat and `POST /jobs?wait=S` both call it. `$JOBSMITH_SYNC_TIMEOUT` remains a bench setting.
- **Delivery.** `reply_to` is JSON, `{"kind": …, …}` (step 1). When a job settles, the manager calls `deliverers[kind].deliver(job)`; a `True` answer writes `delivered_at`. There are two v1 kinds:
  - `none` is delivered at settle, since nobody is waiting. This ends "announceable to session None".
  - `session` is registered by the chat. Its `deliver` answers False; the conversation pulls with `pending_deliveries(reply_to)`, then calls `mark_delivered(id)` once the model has seen the job (today's middleware).

  `create_job` refuses an unknown kind, and resume clears `delivered_at`. Push kinds need no schema change.

  The summary stores `reply_to` together with a flat `reply_key` that the kind's deliverer derives (`session:S1`). `pending_deliveries` filters on that key only, because every nested form fails on one backend or another (checked):

  | filter | `InMemoryStore` | `AsyncSqliteStore` |
  |---|---|---|
  | nested `{"reply_to": {"kind": …, "id": …}}` | finds the job | `ValueError: Unsupported operator: kind` |
  | dotted `{"reply_to.id": …}` | finds nothing | finds the job |
  | flat `{"reply_key": "session:S1"}` | finds the job | finds the job |

  The delivery tests run on all three backends, like `test_job_history.py` (0141, the same family of bug).
- **The guarantee, as it really is.** The engine promises **at least once, keyed by job id**; deduplication is the receiver's job. For `session`, before step 9 the guarantee is weaker, both today and throughout steps 4 to 8: the notices are transient (0006), so the thread keeps no trace to deduplicate on. `mark_delivered` is also written *inside* the model call, before the model node finishes. A crash before the mark delivers the job again, and a crash after the mark but before the checkpoint loses the announcement from the thread. Step 9 closes both windows. The middleware declares a `delivered_jobs` channel in the thread state and returns `ExtendedModelResponse(response, Command(update={"delivered_jobs": ids}))`. LangChain applies that command in the **same update as the response**, and so in the same checkpoint (checked on langchain 1.4.0, `_build_commands`). A job whose id is already in the channel is never injected again. `delivered_at` becomes the index, and a late one gets repaired. The outcome is exactly once **in the thread**. The only thing left is a turn streamed to the screen that died before its checkpoint, which the next turn rewrites.

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
- **G5, the contract out of the bench** (step 9). A `create_agent` conversation built only from `adapters/langchain` and `engine`, with no import from the bench, launches the G2 graph. The job is promoted and delivered. After a simulated crash on each side of the mark, it appears in the thread exactly once.
- **G4, shape leak-check.** `make leak-check` also greps `engine/` for the product vocabulary (`query`, `session`, `document`, `formats`, `capabilit`, `plan`, `report`, `announc`, `terminal_kind`, `final_answer`, `deliverable`, `artifact`, `annex`, `chat`), docstrings included. A false positive is renamed, never allowlisted.

## Order: one PR per step, each `make check` + structural evals at 100%

4. **Move.** `engine/` and `dag/` are created by `git mv` plus import rewrites, nothing else. G3 lands with its allowlist.
5. **The DAG writes its own document.** `write_document` is a final DAG node, after `post_process` and `unanswered`, that calls the Reporters, so the manager stops calling them. As built, it differs from the plan in two ways. It reports what it wrote in its state update, which the runner turns into `DocumentWritten`; that becomes a declaration at 6a. Deliverable paths keep their place, which is not `ArtifactStore`'s `<job_id>/<name>`. Reporters read a `ReportSubject` protocol, which the old `Job` still satisfies. The fields that served them leave at step 6.
6a. **Generic runner and facts.** The runner passes the input through and reads the output from the root `values`, steps from the root `updates` and facts from `custom`. The DAG publishes `plan`, `formats` and `step:<cap>`, and the manager rebuilds the **old** `Job` from those facts and from the output. Neither the data model nor the bench changes, so the graph's output contract settles first. As built: the DAG has no output schema, so its output is its whole final state (terminal kind, answer, errors, the document it wrote). Until 6b, the manager still builds the DAG's input (`job_id` included), and `NodeFinished` is not recorded.
6b. **`Job v1` and `JobFailed`**, re-split in three to stay within the budget of one PR per step (0130). **6b.1**: the DAG's façade `DagJobs` (`dag/jobs.py`) carries today's job API over the engine. The bench, the evals and the tests call it, and it checks what a request's document is to be. **6b.2**: the engine persists facts and root steps. Files become `artifact:` facts, declared through `jobsmith/artifacts/` (store and path safety move there). The façade's view derives plan, results, step times, the file list and its order (main, alternates, annexes in plan order), and says when a file is missing. `outputs` leaves the engine. As built: a path declared twice is listed once (the first in plan order); a progress event carries only what changed (id, status, times, usage), so readers re-read; a listing is summaries, with no plan, results or files. **6b.3**: `Job v1`, `GraphSpec` and `JobFailed`, the `jobs_v1` namespace, and the DAG's result extractor taking over from the manager's mapping. The façade renders the v1 record in today's shape, so CLI, API and TUI need no change. G2 lands. As built: `engine/graph.py` holds `GraphSpec` and `JobFailed(reason, result=)`, and a run that never returned, or whose result is not JSON, is FAILED and says why. Nodes read their job id with `current_job_id()` (the DAG's `job_id_of`). A progress event carries `label`. `jobsmith/__init__.py` imports on first use, so the engine loads alone. G2 checks that in a fresh interpreter; its delivery half joins at step 7. G3 has no crossing left out of `engine/`.
7. **Promotion and delivery.** Lands `run_for` and `reply_to`/deliverers/`delivered_at`; the chat and `POST /jobs?wait` switch to them. Split in two to stay within budget. **7a**: `run_for`. As built: it answers with the record as it stands after the wait, and raises a crash that `_drive` could not settle (the chat's tool turns it into its "crashed" message, as before). `POST /jobs?wait=S` answers with the whole job; without `wait`, the answer is still `{job_id, status}`. **7b**: delivery.
8. **Composition and usage.** Lands `AgentDefinition.graph`, optional chat, `JobService`/`ChatService` and the LangChain callback. G1 and G4 land and the allowlist is empty.
9. **The contract leaves the bench.** The launch tool, the notification middleware and the delivered-ids channel move to `adapters/langchain/`, generic over a `GraphSpec` plus an input builder given by the app. `chat/` keeps the persona and the DAG's arguments. G5 lands. After that the process unfreezes: the single refonte record points here, and the scribe rewrites `CLAUDE.md`.

## Decided (2026-09-27, the three recommendations approved)

1. **Existing `jobs.db` records: start clean.** The namespace moves from `jobs` to `jobs_v1`, and old rows are left untouched and unread. No read shim.
2. **JSON shape of a job over HTTP: changed in place**, without versioning. Only our own CLI and TUI read it.
3. **Several graphs per engine: yes.** `graph` goes on the record, and resume finds its graph by that name.
