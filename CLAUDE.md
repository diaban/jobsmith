# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`agent_oo` is a **domain-agnostic agent planner/executor framework** on LangGraph, written in an **object-oriented** node pattern. A user message comes in; a registry-driven planner emits a DAG of pluggable capabilities (self-describing agentic sub-graphs); a wave-based executor fans them out in parallel where dependencies allow; a generation pipeline merges their results into an answer. Each run is a persistent, trackable, cancellable **Job**. Above the job engine sits a **conversational chat layer** (`agent_oo/chat/`): a LangGraph prebuilt ReAct agent that answers simple messages directly and launches complex tasks as background jobs (human-in-the-loop approval), then surfaces finished-job reports back into the conversation. The original banking-specific agent survives as an example under `agent_oo/examples/banking/`.

## Commands

A Makefile wraps the common ones: `make help` lists them (`install`, `install-all`, `test [T=kw]`, `lint`, `fix`, `check` = lint+leak-gate+tests, `demo`, `chat [LLM=...]`, `api [PORT=...]`, `clean`). Raw equivalents:

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,anthropic]"  # setup
.venv/bin/python -m pytest tests/ -q                        # all tests
.venv/bin/python -m pytest tests/test_jobs.py -q            # one file
.venv/bin/python -m pytest tests/test_planner.py::test_cycle_rejected  # one test
.venv/bin/ruff check .                                      # lint
.venv/bin/python -m agent_oo.examples.banking.main          # runnable demo (fakes)
.venv/bin/python -m agent_oo.examples.banking.chat          # interactive REPL
.venv/bin/python -m agent_oo.examples.banking.api [port]    # HTTP API (needs .[api]); docs at /docs
```

The REPL is a CONVERSATION by default (chat agent; complex asks → job proposal → y/N approval → background job → synthesis + report path on a later turn); `/bg <text>` bypasses the chat and runs a job directly. It auto-selects the provider for BOTH stacks (`--llm=anthropic|openai|fake` overrides): the job engine uses `agent_oo/clients.py` adapters, the chat agent uses LangChain models (`langchain-anthropic`/`langchain-openai`, extras `chat-anthropic`/`chat-openai`; fake mode needs neither). Job-engine LLM selection: `AnthropicLLMClient` (`claude-opus-5`) when `ANTHROPIC_API_KEY` is set, else `OpenAILLMClient` (`gpt-5.1`; honors `OPENAI_BASE_URL` for Ollama/vLLM/gateways) when `OPENAI_API_KEY` is set, else the deterministic `KeywordLLM` fake. Both real adapters live in `agent_oo/clients.py`, implement chat+vision, take an injectable `client` for tests, and raise `RuntimeError` on refusals so they flow through the framework's NodeError path. Provider quirks handled there: the Anthropic adapter drops `temperature` (removed on Claude Opus 5 — sending it 400s), hoists `system` messages to the top-level param, and enables server-side refusal fallbacks; the OpenAI adapter drops `temperature` and uses `max_completion_tokens` for reasoning models (gpt-5*/o*).

Domain-leakage gate (must return nothing): `grep -ri "banking\|banquier\|Votre\|analyste" agent_oo/core agent_oo/jobs`

## Architecture

### The OO pattern

Every graph step is a class instance owning its deps and config. Node logic is **async instance methods** registered directly (`g.add_node("planner", self.planner.run)`); routers are **sync methods**; capabilities expose `.build()` returning a compiled sub-graph mounted as one parent node. `AgentBuilder` (`core/builder.py`) is the composition root and holds references to every step instance (swap one before `.build()` in tests).

### Core concepts (read these files first)

- **`core/capability.py`** — `Capability` ABC + `CapabilitySpec` (name, description, JSON-schema dicts, `requires_inputs`). Capabilities take *exactly the clients they need* in their constructors; the framework never introspects them. Terminal sub-graph nodes call `_emit_success`/`_emit_failure` so every capability reports uniformly.
- **`core/registry.py`** — `CapabilityRegistry`: single source of truth for what the agent can do. The planner prompt, executor Send targets, and builder node map all derive from it. **Frozen at `build()`** — a compiled graph's capability set is fixed; new capability ⇒ new `AgentBuilder` (compilation is milliseconds).
- **`core/state.py`** — capability results live in one `results: dict[str, CapabilityResult]` with a dict-union reducer. Fan-in safety: each capability writes only its own key; registry-unique names + no-duplicate plan steps ⇒ disjoint keys. **Determinism caveat:** consumers must iterate in *plan order*, never dict order (ContextMerger does).
- **`core/profile.py`** — `AgentProfile` is the entire domain surface: prompt templates, user-facing messages, input/output validation rules (plain callables), `max_refine`. Core defaults are neutral English; the banking example overrides them (French messages live *only* in `examples/banking/profile.py`).

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

`JobManager(graph, store)` — `create_job` / `run_job` (awaitable) / `start_job` (background task) / `get_job` / `list_jobs` / `cancel_job`. `job_id` doubles as the LangGraph `thread_id`.

- **Graph nodes are job-agnostic**: `run_job` drives `graph.astream(stream_mode="updates")` and persists plan/artifacts/status as node updates arrive. `PostProcessor`/`Escalator` do NOT write to the store — if you add persistence, put it in the JobManager, not in nodes.
- Store schema: `("jobs","index")/job_id` → summary; `("jobs",job_id,"meta")` → plan/errors; `("jobs",job_id,"artifacts")/cap_name` → per-capability result. Fine-grained state stays in the checkpointer under the same thread_id.
- Cancellation is in-process (asyncio.Task cancel → CANCELLED persisted, checkpoint retained for future resume). Cross-process cancel is a best-effort tombstone — documented v1 limit.
- On DONE the manager writes a **markdown report** to `reports_dir` (default `artifacts/`, gitignored): answer, plan table with per-step status/timestamps (`step_finished_at`, recorded as `cap_*` updates stream), mermaid DAG, artifact payloads. `Job` also carries `session_id` (chat session that launched it) and an `announced` flag (`list_finished_unannounced`/`mark_announced` drive chat notifications).

### Chat layer (`chat/`)

`ChatSession(manager, model, *, session_id, system_prompt, checkpointer).build()` → a `create_react_agent` (LangChain model, NOT the framework's LLMClient — deliberate two-stack split: LangChain handles per-provider tool formats; the job engine stays dependency-light). Deprecation of `create_react_agent` in langgraph v2 → migrate to `langchain.agents.create_agent` then (warning filtered in pyproject).

- **Tools** (`chat/tools.py`) wrap JobManager use-cases, scoped to the session's own jobs. `launch_job` is **human-in-the-loop**: it `interrupt()`s with `{action, query, rationale}` before creating anything; resume with `Command(resume={"approved": bool})`. Approved → `create_job(session_id=...)` + `start_job` (background, non-blocking).
- **Notifications**: `notify_finished_jobs` pre-model hook injects a `SystemMessage` ("background jobs finished" marker) with final answer + report path for unannounced session jobs, marks them announced. **Gotcha**: `llm_input_messages` is a persisted channel — the hook must reset it every turn (return current messages) or a stale notice replays.
- Tests script the model with `conftest.ScriptedChatModel` (AIMessages with `tool_calls`, `bind_tools` no-op, records inputs in `.calls`).

### HTTP API (`api/`)

`create_api(manager, session_factory) -> FastAPI` (extra `.[api]`; banking runner: `examples/banking/api.py`). Domain-agnostic — everything arrives via the injected manager/factory.

- **Chat**: `POST /sessions` → id; `POST /sessions/{id}/messages` returns `{"type": "message"}` or `{"type": "proposal"}` (HITL interrupt surfaced over HTTP); client answers with `POST /sessions/{id}/approval {"approved": bool}`.
- **Jobs**: `GET /jobs[?session_id&status]`, `GET /jobs/{id}` (plan/step_finished_at/results — the UI's DAG data), `POST /jobs` (direct launch), `POST /jobs/{id}/cancel`, `GET /jobs/{id}/report` (markdown).
- **Live**: `GET /events` — SSE over `JobManager.subscribe()`: every `_persist_summary` emits `{job_id, status, steps_done, report_path, ...}` to subscriber queues (`put_nowait`, drops on full — never blocks a run; in-process only, same v1 scope as cancellation). SSE can't be exercised through httpx `ASGITransport` (it buffers the whole body) — the pub/sub is unit-tested in `test_jobs.py`, endpoints in `test_api.py`.

### Adding a capability

1. Subclass `Capability`; define `spec` (unique snake_case name), constructor taking needed clients, async node methods, `build()` via `self.state_graph(...)`, terminal nodes returning `self._emit_success(...)`/`self._emit_failure(...)`, and `render_context()` if its result should feed generation.
2. Register it in the composition root before `AgentBuilder(...).build()`. Nothing else: planner prompt, dispatch, merging all pick it up from the registry.

## Testing conventions

- `tests/conftest.py` — `FakeLLM` scripts responses by **substring of the system prompt** (`{"planner": ..., "ONLY the provided": ...}`); `plan_json()` builds planner responses. Fixtures: `checkpointer` (MemorySaver), `store` (InMemoryStore).
- `tests/test_banking_example.py` is the behavior-parity suite for the pre-refactor agent (French rejection messages, citation rule, vision-dropped-without-image).
- Tests import capabilities/stubs directly and assert on the final state dict (`terminal_kind`, `results`, `completed_capabilities`).
