# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`jobsmith` is both a **product** — a general-purpose conversational agent (`python -m jobsmith`) that answers simple messages directly and launches complex tasks as **background jobs** (human-in-the-loop approval), keeps chatting while they run, then surfaces finished-job markdown reports back into the conversation — and the **domain-agnostic framework** it is built on (LangGraph, object-oriented node pattern): a registry-driven planner emits a DAG of pluggable capabilities (self-describing agentic sub-graphs); a wave-based executor fans them out in parallel; a generation pipeline merges their results; each run is a persistent, trackable, cancellable **Job**; a chat layer (`jobsmith/chat/`, LangGraph prebuilt ReAct agent) sits on top. **`jobsmith/agents/` holds the agent definitions** (a capability pack + a profile: `default` = research→analysis→critique, `banking` = the domain example) and **`jobsmith/app/` is the composition root that runs any of them** (provider selection, persistence, `build_app(agent=...)`).

`README.md` is the human-facing counterpart of this file: product pitch, quickstart,
CLI/API surface, limits. Keep it in sync when a command or a limit changes.

## Commands

A Makefile wraps the common ones: `make help` lists them (`install`, `install-all`, `test [T=kw]`, `lint`, `fix`, `check` = lint+leak-gate+tests, `serve`/`chat`/`jobs` = the global agent, `chat-banking`/`api-banking`/`demo-banking` = the example, `clean`). Raw equivalents:

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,api,anthropic]"  # setup
.venv/bin/python -m pytest tests/ -q                        # all tests
.venv/bin/python -m pytest tests/test_planner.py::test_cycle_rejected  # one test
.venv/bin/ruff check .                                      # lint
jobsmith serve [--port 8000]                                # ★ the daemon: it owns the job engine
jobsmith chat [--session ID]                                # ★ converse (daemon if up, else embedded)
jobsmith run "<task>" [--wait] | jobs | job <id> | report <id> | cancel <id>
# `python -m jobsmith …` is the same entrypoint when the venv's bin isn't on PATH
jobsmith --agent banking chat | serve                       # any agent, same shell
.venv/bin/python -m jobsmith.agents.banking.demo            # scripted banking demo (fakes)
```

The REPL is a CONVERSATION by default (chat agent; complex asks → job proposal → y/N approval → background job → synthesis + report path on a later turn); `/bg <text>` bypasses the chat and runs a job directly. It auto-selects the provider for BOTH stacks (`--llm=anthropic|openai|fake` overrides): the job engine uses `jobsmith/clients.py` adapters, the chat agent uses LangChain models (`langchain-anthropic`/`langchain-openai`, extras `chat-anthropic`/`chat-openai`; fake mode needs neither). Job-engine LLM selection: `AnthropicLLMClient` (`claude-opus-5`) when `ANTHROPIC_API_KEY` is set, else `OpenAILLMClient` (`gpt-5.1`; honors `OPENAI_BASE_URL` for Ollama/vLLM/gateways) when `OPENAI_API_KEY` is set, else the deterministic `KeywordLLM` fake. Both real adapters live in `jobsmith/clients.py`, implement chat+vision, take an injectable `client` for tests, and raise `RuntimeError` on refusals so they flow through the framework's NodeError path. Provider quirks handled there: the Anthropic adapter drops `temperature` (removed on Claude Opus 5 — sending it 400s), hoists `system` messages to the top-level param, and enables server-side refusal fallbacks; the OpenAI adapter drops `temperature` and uses `max_completion_tokens` for reasoning models (gpt-5*/o*).

Domain-leakage gate (`make leak-check`, must return nothing): `grep -ri --include="*.py" "banking\|banquier\|votre\|analyste" jobsmith/core jobsmith/jobs jobsmith/chat jobsmith/api jobsmith/app jobsmith/cli jobsmith/agents/default jobsmith/agents/base.py` — note it scans `agents/default` and `agents/base.py`, **not** `agents/banking`, which is allowed to be as domain-specific as it likes.

### CLI + daemon (`cli/`) — where jobs actually run

The point of this layer: **a job must outlive the command that launched it**. `jobsmith serve` is a long-lived process owning the JobManager; every other command is a *client*.

- `cli/client.py` — one `AgentClient` interface, two backings: `DaemonClient` (HTTP, `persistent=True`) and `EmbeddedClient` (in-process, `persistent=False`). Both return the API's dict shapes, so `cli/repl.py` and every command are written once. `open_client()` probes `GET /health` and falls back to embedded, **printing the trade-off on stderr**.
- **All diagnostics go to stderr** (provider/persistence/daemon banners) — stdout stays pipeable (`jobsmith jobs | cut -d' ' -f1`).
- `cli/main.py` — argparse; `--llm` is exported as `$JOBSMITH_LLM` so `pick_provider` sees it from both stacks. Bare `jobsmith` == `jobsmith chat`.
- Sessions are **rebuildable by id**: the API's registry is only a cache, the conversation lives in the checkpointer under `thread_id=session_id`, so `jobsmith chat --session <id>` resumes across a daemon restart (and gets the finished-job announcement). `session_factory` therefore takes an optional `session_id`.
- Embedded mode is honest about its limit: jobs stop when the process exits (`recover_interrupted` settles them on the next start).

### Agents (`agents/`) — what an agent *is*

An agent is **a capability pack + a profile (+ an optional chat persona)**, and nothing else — `AgentDefinition` in `agents/base.py`. The runtime, job engine, chat, CLI and API are shared by all of them, so **adding an agent touches no shared code**: define the capabilities, register the definition in `agents/__init__.py`, done. `tests/test_agents.py` pins that property (it composes a third-party agent from scratch).

- `agents/default/`: the LLM-only pack — `research` (decompose into aspects, lenient JSON, then structured notes) → `analysis` → `critique`. The latter two subclass `SingleStepCapability` (`_step.py`): one LLM node reading the best upstream `results` entry, degrading to the bare request when the upstream failed/was skipped.
- `agents/banking/`: the domain example — capabilities, its **own ports** (`deps.py`: `SearchEngine`/`VisionClient`/`S3Client` Protocols) and **its own adapters** (`fakes.py`), plus a French profile. The ports live next to the capabilities that consume them, never in a central `ports/` package: that is what keeps them scaling with their consumers.
- Selection: `--agent NAME` (CLI, applies to whichever process owns the engine — so pass it to `serve`), `build_app(agent=...)`, `make chat AGENT=banking`.

### The composition root (`app/`)

Wiring only, no content — everything here is domain-neutral:
- `providers.py`: `pick_provider` (one `--llm=` flag / key auto-detect shared by both LLM stacks), `make_llm` (job engine, `clients.py` adapters), `make_chat_model` (LangChain), `load_dotenv`, and the keyless fakes — `KeywordLLM` plans by **parsing capability names out of the rendered planner prompt** (works with any registry), `KeywordChatModel` proposes a job on analysis-ish keywords.
- `agent.py`: `await build_app(agent=..., **overrides) -> AgentApp(manager, session_factory, agent_name, aclose)`. It composes ONE agent definition into a running product; the REPL and entrypoints live in `cli/`.
- `persistence.py`: `pick_db()` (arg > `--db=` > `$JOBSMITH_DB` > `memory`) + `open_persistence(spec, stack)` → `(checkpointer, store)`, teardown on the caller's `AsyncExitStack`. Backends: `memory` (default), a SQLite path (`.[sqlite]`), or a Postgres DSN (`.[postgres]`, one shared `AsyncConnectionPool` for saver + store, `autocommit/prepare_threshold=0/dict_row` as those backends require). Chat sessions share the job graph's checkpointer, so conversations persist too (namespaced by `thread_id`: `session_id` vs `job_id`).

**Why `build_app` is async**: real backends must be opened in the event loop that will use them. `python -m jobsmith api` therefore serves with `await uvicorn.Server(config).serve()` inside that same loop — `uvicorn.run()` would start its own loop and strand the pool. **SQLite gotcha** (cost an hour): the *store* needs `isolation_level=None` (it drives its own `BEGIN`/`COMMIT`; under implicit transactions its first write leaves one open and the next `BEGIN` raises "cannot start a transaction within a transaction"), the *saver* keeps the default; never run a stray `PRAGMA` on those live connections (it opens a transaction and deadlocks the other connection) — WAL is set once on a throwaway connection, and the busy timeout via `connect(timeout=...)`.
- **Interrupted jobs**: `JobManager.recover_interrupted()` runs in `build_app` — RUNNING records with no in-process task are leftovers from a dead process, marked FAILED (checkpoint retained for a future resume); QUEUED jobs stay runnable.

## Architecture

### The OO pattern

Every graph step is a class instance owning its deps and config. Node logic is **async instance methods** registered directly (`g.add_node("planner", self.planner.run)`); routers are **sync methods**; capabilities expose `.build()` returning a compiled sub-graph mounted as one parent node. `AgentBuilder` (`core/builder.py`) is the composition root and holds references to every step instance (swap one before `.build()` in tests).

### Core concepts (read these files first)

- **`core/capability.py`** — `Capability` ABC + `CapabilitySpec` (name, description, JSON-schema dicts, `requires_inputs`). Capabilities take *exactly the clients they need* in their constructors; the framework never introspects them. Terminal sub-graph nodes call `_emit_success`/`_emit_failure` so every capability reports uniformly.
- **`core/registry.py`** — `CapabilityRegistry`: single source of truth for what the agent can do. The planner prompt, executor Send targets, and builder node map all derive from it. **Frozen at `build()`** — a compiled graph's capability set is fixed; new capability ⇒ new `AgentBuilder` (compilation is milliseconds).
- **`core/state.py`** — capability results live in one `results: dict[str, CapabilityResult]` with a dict-union reducer. Fan-in safety: each capability writes only its own key; registry-unique names + no-duplicate plan steps ⇒ disjoint keys. **Determinism caveat:** consumers must iterate in *plan order*, never dict order (ContextMerger does).
- **`core/profile.py`** — `AgentProfile` is the entire domain surface: prompt templates, user-facing messages, input/output validation rules (plain callables), `max_refine`. Core defaults are neutral English; the banking example overrides them (French messages live *only* in `agents/banking/profile.py`).

### Graph flow

```
validate_input → router ─(direct)→ direct_answer ────────────────┐
                   └(plan)→ planner → executor_dispatch ⇄ {cap_<name> × registry}
                                ↓ (all done)                     ↓
                          merge_results → generation → validate_output
                                              ↑ refine ←┘ (≤ max_refine)  → post_process → END
errors: execution_error → escalate (some ok result) | user_error (none) → END
```

- **Router** (`core/router.py`) is a dedicated triage node — the planner never decides *whether* to plan. LLM picks a route from `Router.routes` (`"plan"` → planner, `"direct"` → `DirectResponder`, which renders the registry into its prompt so "what can you do?" is answerable, then joins at `validate_output`). **Fail-open**: any LLM/parse error or unknown route falls back to `"plan"`. New route = entry in `Router.routes` + node + `AgentBuilder.route_targets` entry before `.build()`.
- **Planner** (`core/planner.py`) renders its prompt from `registry.specs()`, validates the LLM's JSON DAG: names against the registry, drops steps whose `is_applicable(state)` is false (generalizes "vision only if image" via `spec.requires_inputs`), **prunes dropped names from surviving `depends_on`**, Kahn cycle check.
- **Executor** (`core/executor.py`) is a pass-through node + router: computes ready capabilities each wave and returns `list[Send]`; capability nodes edge back to `executor_dispatch`. This executes an arbitrary DAG without a baked-in schedule.
- **Two error channels**: `NodeError.recoverable=False` (planner/generation failures) hard-stops into `execution_error`; capability failures are recoverable — they land in `results` with `ok: False` and the run degrades gracefully.

### Critical invariant: capability output schema

Capability `build()` MUST use `self.state_graph(PrivateState)` (which sets `output_schema=CapabilityOutputState`). Without it, the sub-graph echoes its full state (including `query`) to the parent, and two capabilities finishing in the same superstep collide with `InvalidUpdateError`. Private states extend `CapabilityBaseState`.

### Jobs layer (`jobs/`)

`JobManager` holds **only the use cases** — `create_job` / `run_job` (awaitable) / `start_job` (background task) / `get_job` / `list_jobs` / `cancel_job` / `recover_interrupted`. Everything else is a collaborator behind a port, so each changes for its own reason:

| collaborator | responsibility | file |
|---|---|---|
| `JobRepository` | where records live + **the store schema** | `jobs/repository.py` |
| `GraphRunner` | drives the run, translates it to domain updates | `jobs/runner.py` |
| `JobEvents` | broadcasts progress | `jobs/events.py` |
| `Reporter` | produces the deliverable | `jobs/report.py` |

Defaults wire the v1 stack, so `JobManager(graph, store)` still works; pass `repository=`/`runner=`/`events=` to swap one. `tests/test_job_seams.py` drives the whole lifecycle with **no graph and no store** — if that stops being possible, a responsibility has leaked back in.

- **Only `runner.py` knows LangGraph's stream shape** (`{node: update}`, `cap_<name>` nodes, terminal node names). It yields `PlanReady` / `StepFinished` / `NodeErrors` / `Terminal`; the manager folds those into the Job. Graph nodes stay job-agnostic — `PostProcessor`/`Escalator` do NOT write to the store; new persistence goes in the manager or the repository, never in a node.
- **Only `repository.py` knows the schema**: `("jobs","index")/job_id` → summary; `("jobs",job_id,"meta")` → plan/errors; `("jobs",job_id,"results")/cap_name` → per-capability result. Fine-grained state stays in the checkpointer under `thread_id == job_id`. Moving job records to SQL is another implementation of this port.
- Cross-process progress (Postgres LISTEN/NOTIFY, Redis) is another `JobEvents`, not a manager change; the same goes for `resume_job()`, which belongs on the manager.
- Cancellation is in-process (asyncio.Task cancel → CANCELLED persisted, checkpoint retained for future resume). Cross-process cancel is a best-effort tombstone — documented v1 limit.
- **Vocabulary (was ambiguous, now fixed)**: an **artifact/output** is what the job produces FOR THE HUMAN (`Job.outputs: list[JobOutput]` — path, format, title, role `main`|`annex`); a capability's intermediate payload is a **result** (state channel `results`, store namespace `("jobs",id,"results")`). `Job.report_path` is a property = the main output's path.
- **Producing the deliverable is a Reporter's job**, not the manager's (`jobs/report.py`): `build_document(job, registry)` makes a format-independent `JobDocument`, `MarkdownReport.write()` serializes it and returns a `JobOutput`. Other formats (HTML/PDF/PPTX) = other Reporters over the same document; `JobManager(..., reporter=...)` takes any of them. The composition root injects `MarkdownReport(registry)`.
- **The report is a deliverable, not a trace**: title + answer first, then provenance (request, timings, plan table, mermaid DAG). Per-step material is NOT inlined — it lives in the store and is served by `GET /jobs/{id}` and `jobsmith job <id>`. `MarkdownReport(with_annexes=True)` re-adds it as collapsible `<details>` for a self-contained archive.
- **Capabilities present their own results**: `Capability.render_report(result)` (twin of `render_context`, which targets the model) with `default_result_markdown` as the base implementation — prose stays prose, list[str] becomes bullets, only structured values fall back to JSON. Never grow that default to learn payload shapes: override `render_report` in the capability instead.
- `Job` also carries `session_id` (chat session that launched it) and an `announced` flag (`list_finished_unannounced`/`mark_announced` drive chat notifications).

### Chat layer (`chat/`)

`ChatSession(manager, model, *, session_id, system_prompt, checkpointer).build()` → a `langchain.agents.create_agent` (LangChain model, NOT the framework's LLMClient — deliberate two-stack split: LangChain handles per-provider tool formats; the job engine stays dependency-light).

- **Tools** (`chat/tools.py`) wrap JobManager use-cases, scoped to the session's own jobs. `launch_job` is **human-in-the-loop**: it `interrupt()`s with `{action, query, rationale}` before creating anything; resume with `Command(resume={"approved": bool})`. Approved → `create_job(session_id=...)` + `start_job` (background, non-blocking).
- **Notifications**: `JobNotificationMiddleware.awrap_model_call` appends a `SystemMessage` ("background jobs finished" marker) with final answer + report path for unannounced session jobs, then marks them announced *after* the model call succeeds. Wrapping the request (rather than writing state) is what keeps the notice out of the persisted thread — `create_agent`'s state has only `messages`/`jump_to`/`structured_response`, and the old `llm_input_messages` ephemeral channel no longer exists.
- Tests script the model with `conftest.ScriptedChatModel` (AIMessages with `tool_calls`, `bind_tools` no-op, records inputs in `.calls`).

### HTTP API (`api/`)

`create_api(manager, session_factory) -> FastAPI` (extra `.[api]`; served by `jobsmith serve`, whichever agent is composed). Domain-agnostic — everything arrives via the injected manager/factory.

- **Chat**: `POST /sessions` → id; `POST /sessions/{id}/messages` returns `{"type": "message"}` or `{"type": "proposal"}` (HITL interrupt surfaced over HTTP); client answers with `POST /sessions/{id}/approval {"approved": bool}`.
- **Jobs**: `GET /jobs[?session_id&status]`, `GET /jobs/{id}` (plan/step_finished_at/results — the UI's DAG data), `POST /jobs` (direct launch), `POST /jobs/{id}/cancel`.
- **Outputs**: `GET /jobs/{id}/outputs` lists the produced files, `/outputs/{name}` downloads one, `/report` is the inline shortcut to the main deliverable.
- **Live**: `GET /events` — SSE over `JobManager.subscribe()`: every `_persist_summary` emits `{job_id, status, steps_done, report_path, ...}` to subscriber queues (`put_nowait`, drops on full — never blocks a run; in-process only, same v1 scope as cancellation). SSE can't be exercised through httpx `ASGITransport` (it buffers the whole body) — the pub/sub is unit-tested in `test_jobs.py`, endpoints in `test_api.py`.

### Adding a capability

1. Subclass `Capability`; define `spec` (unique snake_case name), constructor taking needed clients, async node methods, `build()` via `self.state_graph(...)`, terminal nodes returning `self._emit_success(...)`/`self._emit_failure(...)`, and `render_context()` if its result should feed generation.
2. Register it in the composition root before `AgentBuilder(...).build()`. Nothing else: planner prompt, dispatch, merging all pick it up from the registry.

## Testing conventions

- `tests/conftest.py` — `FakeLLM` scripts responses by **substring of the system prompt** (`{"planner": ..., "ONLY the provided": ...}`); `plan_json()` builds planner responses. Fixtures: `checkpointer` (MemorySaver), `store` (InMemoryStore).
- `tests/test_banking_example.py` is the behavior-parity suite for the pre-refactor agent (French rejection messages, citation rule, vision-dropped-without-image).
- Tests import capabilities/stubs directly and assert on the final state dict (`terminal_kind`, `results`, `completed_capabilities`).
