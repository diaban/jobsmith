# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`jobsmith` is both a **product** — a general-purpose conversational agent (`python -m jobsmith`) that answers simple messages directly and launches complex tasks as **background jobs** (human-in-the-loop approval), keeps chatting while they run, then surfaces finished-job markdown reports back into the conversation — and the **domain-agnostic framework** it is built on (LangGraph, object-oriented node pattern): a registry-driven planner emits a DAG of pluggable capabilities (self-describing agentic sub-graphs); a wave-based executor fans them out in parallel; a generation pipeline merges their results; each run is a persistent, trackable, cancellable **Job**; a chat layer (`jobsmith/chat/`, LangGraph prebuilt ReAct agent) sits on top. **`jobsmith/agents/` holds the agent definitions** (a capability pack + a profile: `default` = research→analysis→critique, `banking` = the domain example) and **`jobsmith/app/` is the composition root that runs any of them** (provider selection, persistence, `build_app(agent=...)`).

`README.md` is the human-facing counterpart of this file: product pitch, quickstart,
CLI/API surface, limits. Keep it in sync when a command or a limit changes.

## Commands

A Makefile wraps the common ones: `make help` lists them (`install`, `install-all`, `test [T=kw]`, `lint`, `fix`, `check` = lint+leak-gate+tests, `eval`/`eval-llm` = score the prompts on the golden set, `serve`/`chat`/`jobs` = the global agent, `chat-banking`/`api-banking`/`demo-banking` = the example, `clean`). Raw equivalents:

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

The REPL is a CONVERSATION by default (chat agent; complex asks → job proposal → y/N approval → background job → synthesis + report path on a later turn); `/bg <text>` bypasses the chat and runs a job directly. It auto-selects the provider for BOTH stacks (`--llm=anthropic|openai|fake` overrides): the job engine uses `jobsmith/clients.py` adapters, the chat agent uses LangChain models (`langchain-anthropic`/`langchain-openai`, extras `chat-anthropic`/`chat-openai`; fake mode needs neither). Job-engine LLM selection: `AnthropicLLMClient` (`claude-opus-5`) when `ANTHROPIC_API_KEY` is set, else `OpenAILLMClient` (`gpt-5.1`; honors `OPENAI_BASE_URL` for Ollama/vLLM/gateways) when `OPENAI_API_KEY` is set, else the deterministic `KeywordLLM` fake. Both real adapters live in `jobsmith/clients.py`, implement chat+vision, take an injectable `client` for tests, and raise `RuntimeError` on refusals so they flow through the framework's NodeError path. Both adapters also book each response's tokens into the usage ledger (`core/usage.py`) instead of discarding the SDK's `usage`, pricing it by the model the *response* reports (so a server-side fallback is billed as what actually served); `$JOBSMITH_PRICES` overrides the price table. Provider quirks handled there: the Anthropic adapter drops `temperature` (removed on Claude Opus 5 — sending it 400s), hoists `system` messages to the top-level param, and enables server-side refusal fallbacks; the OpenAI adapter drops `temperature` and uses `max_completion_tokens` for reasoning models (gpt-5*/o*).

### Working on this repo

- **One short-lived branch per issue**, off `main`: `feat/<n>-<slug>`, `fix/<n>-<slug>`, `chore/<slug>` (`gh issue develop <n>` creates one already linked). Open a PR, let CI run, merge, delete. **No `develop` branch**: there are no releases yet, so it would only add a merge — the PR + CI is the integration point it used to provide. Releases, when they come, are tags.
- `main` stays green. CI (`.github/workflows/ci.yml`) runs `make check` on push and PR across Python 3.11 and 3.12, plus `uv lock --check` so the lockfile cannot silently drift from pyproject.
- **`main` is protected**: PR required, the three checks must pass, admins included, no force-push. **Status checks are strict** — a branch must contain the current `main` before it can merge, so CI validates the *post-merge* state rather than a stale snapshot. When several PRs are in flight, each merge invalidates the rest: bring them up to date with `gh pr update-branch <n>` (or a rebase) and let CI re-run.

  This exists because green checks on a stale base do not mean the merge is green. Git only sees *textual* conflicts; two PRs can merge cleanly and still break each other — one renames what the other calls, one reshapes an output the other asserts on. Three PRs in the first parallel batch merged on stale CI and survived only because the combination was verified by hand each time.
- **`uv.lock` is committed** and must be regenerated (`uv lock`) in the same commit as any dependency change. This project has already been bitten by version drift (`create_react_agent` deprecation, the removed `llm_input_messages` channel, checkpoint-sqlite's `isolation_level`), which is exactly what the lockfile prevents across sessions.
- **Parallel sessions use git worktrees**, one per issue — separate checkouts of the same repo, so two sessions never fight over the working tree or the current branch:

  ```bash
  make worktree B=feat/1-grounding      # checkout + venv + .env, ~1s (uv cache)
  cd .claude/worktrees/feat-1-grounding
  make check                            # each worktree has its OWN .venv
  make worktree-rm B=feat/1-grounding   # after the PR is merged
  ```

  Gotchas, both verified: a venv is **path-specific** (its shebangs are absolute), so a worktree needs its own — never symlink or copy one; and `.env`, `agent.db`, `artifacts/` are gitignored, so a fresh worktree has **no API key** until it is copied (the `make worktree` target does it). `.claude/worktrees/` is gitignored, which is also where Claude Code's own `EnterWorktree` puts them.
- `make coverage` reports per-module coverage (84% overall; `jobs/` and most of `core/` at 100%). The thin areas are the interactive layers — `cli/repl.py` 29%, `cli/main.py` 42%, `chat/tools.py` 61% — so a change landing there needs its tests written *with* it, not after.

Domain-leakage gate (`make leak-check`, must return nothing): `grep -ri --include="*.py" "banking\|banquier\|votre\|analyste" jobsmith/core jobsmith/jobs jobsmith/chat jobsmith/api jobsmith/app jobsmith/cli jobsmith/agents/default jobsmith/agents/base.py` — note it scans `agents/default` and `agents/base.py`, **not** `agents/banking`, which is allowed to be as domain-specific as it likes.

### The inbound port (`service.py`)

`AgentService` is **the** use-case surface of the application — sessions and jobs — and `LocalAgentService` is its in-process implementation over a composed `AgentApp`. Every entrypoint is an *adapter* over it, never a second implementation:

| adapter | backing |
|---|---|
| `cli/repl.py`, `cli/main.py` | `AgentService` (either backing) |
| `api/app.py` | `LocalAgentService` — each route is serialization + one call |
| `cli/client.py` `DaemonClient` | the same port, backed by HTTP |

`LocalAgentService` additionally exposes what only an in-process service can (`subscribe`/`unsubscribe`, `list_outputs`, `find_output`); the HTTP adapter re-exposes those as `/events` and `/jobs/{id}/outputs/...` so remote callers get them too. `tests/test_service.py` drives the same sequence through both backings and asserts **identical answers**, and greps the API module for leaked use-case code (`create_job(`, `ainvoke(`, `__interrupt__`, …). Adding a UI or a bot = one adapter, zero new use cases.

### CLI + daemon (`cli/`) — where jobs actually run

The point of this layer: **a job must outlive the command that launched it**. `jobsmith serve` is a long-lived process owning the JobManager; every other command is a *client*.

- `cli/client.py` — two backings for that one port: `DaemonClient` (HTTP, `persistent=True`) and `EmbeddedClient` (`persistent=False`), the latter being *nothing but* `LocalAgentService` owning the app it composed. `open_client()` probes `GET /health` and falls back to embedded, **printing the trade-off on stderr**. `AgentClient` remains as the CLI's alias for `AgentService`.
- **All diagnostics go to stderr** (provider/persistence/daemon banners) — stdout stays pipeable (`jobsmith jobs | cut -d' ' -f1`).
- `cli/main.py` — argparse; `--llm` is exported as `$JOBSMITH_LLM` so `pick_provider` sees it from both stacks. Bare `jobsmith` == `jobsmith chat`.
- Sessions are **rebuildable by id**: the API's registry is only a cache, the conversation lives in the checkpointer under `thread_id=session_id`, so `jobsmith chat --session <id>` resumes across a daemon restart (and gets the finished-job announcement). `session_factory` therefore takes an optional `session_id`.
- Embedded mode is honest about its limit: jobs stop when the process exits (`recover_interrupted` settles them on the next start).

### Agents (`agents/`) — what an agent *is*

An agent is **a capability pack + a profile (+ an optional chat persona, + whatever it needs open)**, and nothing else — `AgentDefinition` in `agents/base.py`. The runtime, job engine, chat, CLI and API are shared by all of them, so **adding an agent touches no shared code**: define the capabilities, register the definition in `agents/__init__.py`, done. `tests/test_agents.py` pins that property (it composes a third-party agent from scratch).

```python
AgentDefinition(
    open_resources=...,   # async (stack) -> anything: pools, sessions, clients
    capabilities=...,     # (AgentContext) -> list[Capability]   ctx.llm / ctx.resources
    profile=..., chat_prompt=...,
)
```

**Resources — who opens, who closes.** The agent knows *what* to open (which backend, which collection); the composition root owns the *lifetime* and the event loop. So `open_resources` is handed `build_app`'s own `AsyncExitStack` and returns whatever it built: teardown happens in reverse order on `AgentApp.aclose()`, **including when startup itself failed** (`tests/test_resources.py` pins that). `build_app(resources=...)` injects them instead (the demo and tests use it; the caller then owns them).

**Several capabilities on one backend**: share the *connection*, give each capability **its own adapter** for the port it declared — never one fat client exposing both sets of methods. Capabilities run in parallel waves, so the shared thing must be a **pool**, not a raw connection. `tests/test_resources.py` demonstrates the exact shape (two capabilities, two adapters, one pool).

- `agents/default/`: `documents` → `research` (decompose into aspects, lenient JSON, then structured notes) → `analysis` → `critique`. The latter two subclass `SingleStepCapability` (`_step.py`): one LLM node reading the best upstream `results` entry, degrading to the bare request when the upstream failed/was skipped.
  - **`documents` is the grounding step** — without it a job is the model talking to itself. It depends on the `DocumentSource` **port** (`sources.py`), never on what backs it: `search(query, limit) -> list[Document]`, shaped by the need so a web or vector backend is another adapter, not a rewrite. `LocalFiles` is the first adapter (term-overlap ranking over a directory — *keyword*, not semantic; no key, no network, so tests and CI can use it).
  - **`web_search` is the same capability pointed at the web** (`web.py`, `TavilySource`): subclassed from `DocumentsCapability` because the ONLY difference is the spec — and the spec is what the planner reads to choose between the user's own files and what the web says today. Gaining the web required no change to the consuming capability, which is what the port was shaped for.
  - Both are registered **only when something backs them** (`--docs PATH` / `$JOBSMITH_DOCS`; `$TAVILY_API_KEY` + extra `.[web]`), read by `open_default_resources`. A capability nothing can serve must stay out of the registry, or the planner will plan a step that can only fail. With neither, the agent still runs LLM-only.
  - `TavilySource` is the **first adapter that must be closed**: its `httpx.AsyncClient` is entered on the app's `AsyncExitStack`, so the connection pool is released with the app — cleanly or on a failed startup. It raises rather than returning `[]` on an HTTP error, because the capability already isolates one failing query and a silent empty list would read as "the web knows nothing about this".
  - Retrieved passages carry a **quotable id** (`path#chunk`); `render_context` gives the model the material, `render_report` gives the human the provenance only.
- `agents/banking/`: the domain example — capabilities, its **own ports** (`deps.py`: `SearchEngine`/`VisionClient`/`S3Client` Protocols), **its own adapters** (`fakes.py`, assembled in `open_banking_resources`), and a French profile. The ports live next to the capabilities that consume them, never in a central `ports/` package: that is what keeps them scaling with their consumers, and a port is shaped by the *need*, not by the vendor's API. `demo.py` runs the whole product with richer fakes injected through `build_app(resources=...)`.
- Selection: `--agent NAME` (CLI, applies to whichever process owns the engine — so pass it to `serve`), `build_app(agent=...)`, `make chat AGENT=banking`.

### The composition root (`app/`)

Wiring only, no content — everything here is domain-neutral:
- `providers.py`: `pick_provider` (one `--llm=` flag / key auto-detect shared by both LLM stacks), `make_llm` (job engine, `clients.py` adapters), `make_chat_model` (LangChain), `load_dotenv`, and the keyless fakes — `KeywordLLM` plans by **parsing capability names out of the rendered planner prompt** (works with any registry), `KeywordChatModel` proposes a job on analysis-ish keywords.
- `agent.py`: `await build_app(agent=..., **overrides) -> AgentApp(manager, session_factory, agent_name, resources, aclose)`. It composes ONE agent definition into a running product: providers, persistence, the agent's resources, registry, graph, JobManager, chat-session factory — all on one `AsyncExitStack`. The REPL and entrypoints live in `cli/`.
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
- **`core/usage.py`** — token/cost accounting. `LLMClient.chat` still returns `str`: usage travels on an **ambient ledger** (a `ContextVar` the JobManager installs per run) that adapters push into with `record_usage(...)`, because what a call cost belongs to the run, not to every call site's signature. Attribution is read from LangGraph's runtime config — the root segment of `checkpoint_ns` IS the responsible parent node (`planner`, `cap_research` → `research`) — so a capability nobody wrote for this feature is still attributed, and node signatures are untouched. Failure to resolve a scope degrades to `unattributed`, never to a wrong total. `DEFAULT_PRICES` is a dated snapshot, overridable with `$JOBSMITH_PRICES` (inline JSON or a file path, longest-prefix match); an unpriced model reports tokens with `cost_usd: None` rather than an invented number. **Not covered**: the chat layer's LangChain calls (the two-stack split) — conversation tokens are not counted.
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

`JobManager` holds **only the use cases** — `create_job` / `run_job` (awaitable) / `start_job` (background task) / `resume_job` (awaitable) / `start_resume` (background) / `get_job` / `list_jobs` / `cancel_job` / `recover_interrupted`. Everything else is a collaborator behind a port, so each changes for its own reason:

| collaborator | responsibility | file |
|---|---|---|
| `JobRepository` | where records live + **the store schema** | `jobs/repository.py` |
| `GraphRunner` | drives the run, translates it to domain updates | `jobs/runner.py` |
| `JobEvents` | broadcasts progress | `jobs/events.py` |
| `Reporter` | produces the deliverable | `jobs/report.py` |

Defaults wire the v1 stack, so `JobManager(graph, store)` still works; pass `repository=`/`runner=`/`events=` to swap one. `tests/test_job_seams.py` drives the whole lifecycle with **no graph and no store** — if that stops being possible, a responsibility has leaked back in.

- **Only `runner.py` knows LangGraph's stream shape** (`{node: update}`, `cap_<name>` nodes, terminal node names). It yields `PlanReady` / `StepFinished` / `NodeErrors` / `Terminal`; the manager folds those into the Job. Two ways in — `stream()` (from the query) and `resume()` (from the thread's checkpoint) — share one translation, so the manager folds a resumed run exactly like a first one. Graph nodes stay job-agnostic — `PostProcessor`/`Escalator` do NOT write to the store; new persistence goes in the manager or the repository, never in a node.
- **Only `repository.py` knows the schema**: `("jobs","index")/job_id` → summary; `("jobs",job_id,"meta")` → plan/errors; `("jobs",job_id,"results")/cap_name` → per-capability result. Fine-grained state stays in the checkpointer under `thread_id == job_id`. Moving job records to SQL is another implementation of this port.
- Cross-process progress (Postgres LISTEN/NOTIFY, Redis) is another `JobEvents`, not a manager change.
- Cancellation is in-process (asyncio.Task cancel → CANCELLED persisted, checkpoint retained for resume). Cross-process cancel is a best-effort tombstone — documented v1 limit.
- **Resume** (`resume_job` / `start_resume`, `POST /jobs/{id}/resume`, `jobsmith resume <id-prefix>`, `/resume` in the REPL): `runner.resume()` re-enters the thread with `None` as input, LangGraph's "carry on" — the last completed superstep is replayed from the checkpoint and only the still-pending tasks run. Finished steps are neither re-run nor re-emitted, so the manager keeps the results the repository loaded; the plan likewise comes back from the store, not the stream. Two gates, both needed: the status must be CANCELLED or FAILED (DONE is the *other* half of #5; QUEUED wants `run_job`), **and** `runner.pending(job_id)` must be non-empty. Status alone is not enough — a job that FAILED because a node raised reached `escalate`/`user_error`, so its frontier is empty and re-entering would replay one superstep and silently do nothing; ditto a job cancelled before it ever started (no checkpoint at all — `astream(None)` there raises `EmptyInputError`). Both are refused with `ValueError`, as `run_job` refuses a wrong status. A resumed attempt is driven by the same `_drive` as a first one (same persistence, events, reporting), and its usage ledger is **seeded with what earlier attempts spent**, so `Job.usage` stays the job's total cost rather than the last attempt's.
- What is **not** implemented: re-running a subset of the DAG of a job that already finished. That needs a way to say which results are stale and how the plan is amended — see the note on issue #5.
- **Vocabulary (was ambiguous, now fixed)**: an **artifact/output** is what the job produces FOR THE HUMAN (`Job.outputs: list[JobOutput]` — path, format, title, role `main`|`annex`); a capability's intermediate payload is a **result** (state channel `results`, store namespace `("jobs",id,"results")`). `Job.report_path` is a property = the main output's path.
- **Producing the deliverable is a Reporter's job**, not the manager's (`jobs/report.py`): `build_document(job, registry)` makes a format-independent `JobDocument`, a Reporter serializes it and returns the `JobOutput`s describing what it wrote. Two ship — `MarkdownReport` (default) and `HtmlReport` (`jobs/report_html.py`) — sharing `FileReporter` (build the document, write one file, describe it); a subclass only supplies `format`, `extension` and `render(doc)`. PDF/PPTX would be more of the same. `make_reporter(format, registry)` selects one (unknown name ⇒ `ValueError`).
- **One job, several deliverables**: `Reporter.write` returns `list[JobOutput]` and the manager assigns `job.outputs = list(that)`, so asking for markdown *and* HTML is a composition choice, not a protocol change. `pick_report_formats()` (argument > `$JOBSMITH_REPORT_FORMAT` > markdown) reads a **comma-separated** list — `JOBSMITH_REPORT_FORMAT=markdown,html` — and `compose_reporters(formats, registry)` turns it into one reporter: a single name gives that Reporter unchanged (a one-format run is byte-for-byte what it was), several give a `MultiReporter` that writes each and concatenates the lists. Aliases of one format collapse (`markdown,md` is one file); an unknown name anywhere still raises. **Exactly one output is `role="main"`** — the FIRST format asked for, since `Job.report_path`, `jobsmith report` and `/report` (whose content type follows that output's `format`) all read it; the rest are `role="alternate"`, the same document rendered again. Deliberately *not* `annex`, which stays what it says it is: per-step material a capability produced. No CLI/API flag yet, deliberately.
- **A failed write does not fail the job**: the graph answered and that answer is persisted, so a job whose deliverable could not be written stays DONE with `job.error` naming the format and the cause (FAILED would misreport the work, and is a dead end anyway — the checkpoint has nothing pending, so `resume_job` refuses it). `JobManager._write_outputs` wraps the write in its *own* `try` — widening the run's, whose `except` sets FAILED, would conflate "the graph blew up" with "the disk was full" — and the final `_persist_summary` therefore always runs; before that, an escaping exception left the store holding the RUNNING row the last finished step wrote. `ReportWriteError` (`jobs/report.py`) carries the outputs already on disk and `MultiReporter` raises it, so markdown written before HTML failed is still recorded — a file with no `JobOutput` is a deliverable nobody can find.
- **The report is a deliverable, not a trace**: title + answer first, then provenance (request, timings, plan table, DAG). Per-step material is NOT inlined — it lives in the store and is served by `GET /jobs/{id}` and `jobsmith job <id>`. `with_annexes=True` re-adds it (collapsible `<details>` in both formats) for a self-contained archive.
- **The HTML report takes no dependency and makes no request** (`jobs/report_html.py`): inline CSS (light/dark), and since a browser renders neither mermaid nor a fenced block, the same DAG edges are drawn as an inline SVG (`dag_svg`, longest-path columns left-to-right — the plan's step order is not assumed topological). The answer and the annexes are markdown by contract, so a deliberately small subset renderer (`markdown_to_html`: headings, lists, fences, rules, emphasis) converts them — **everything is `html.escape`d first and only our own tags are added afterwards**, which is why model output cannot open a tag; links are not rendered on purpose (an anchor means sanitizing `javascript:` URLs). Grow that renderer no further: if a report needs tables or links, take a dependency then.
- **Capabilities present their own results**: `Capability.render_report(result)` (twin of `render_context`, which targets the model) with `default_result_markdown` as the base implementation — prose stays prose, list[str] becomes bullets, only structured values fall back to JSON. Never grow that default to learn payload shapes: override `render_report` in the capability instead.
- **What a job cost is part of its record**: `Job.usage` is the run's aggregate (`core/usage.py`), refreshed on *every* summary persist — so a job still running already shows its spend, and `/events` carries it live. The per-step breakdown lives where the step's own material does, in `CapabilityResult.meta["usage"]` (stamped by `Capability._emit_success`/`_emit_failure`, including on failure — a step that burned 40k tokens and failed is exactly the one worth seeing). The report shows the total in *About this job* and a per-step column in the plan table; `jobsmith job <id>` shows both.
- `Job` also carries `session_id` (chat session that launched it) and an `announced` flag (`list_finished_unannounced`/`mark_announced` drive chat notifications).

### Chat layer (`chat/`)

`ChatSession(manager, model, *, session_id, system_prompt, checkpointer).build()` → a `langchain.agents.create_agent` (LangChain model, NOT the framework's LLMClient — deliberate two-stack split: LangChain handles per-provider tool formats; the job engine stays dependency-light).

- **Tools** (`chat/tools.py`) wrap JobManager use-cases, scoped to the session's own jobs. `launch_job` is **human-in-the-loop**: it `interrupt()`s with `{action, query, rationale, context}` before creating anything; resume with `Command(resume={"approved": bool})`. Approved → `create_job(session_id=...)` + `start_job` (background, non-blocking).
- **The job engine never sees the thread**, so "analyse that" would reach the planner with its referent gone — a silent failure (the report looks fine, it answers a slightly different question). Two guards: the tool docstring *is* the model's instruction and demands a self-contained `query` (that wording is also what the user approves), and `recent_conversation()` attaches a bounded excerpt of the last turns as `inputs[CONVERSATION_INPUT_KEY]` (`core/state.py`, key `"conversation"`) as the safety net. Bounds live in `chat/tools.py` (`MAX_CONTEXT_TURNS`/`MAX_TURN_CHARS`/`MAX_CONTEXT_CHARS`) — this rides on every launch; only human/assistant prose travels (tool calls, tool results and the `[job update]`/`[job progress]` notices are dropped). `Planner.user_message` renders it as clearly-labelled background above the request; the request stays authoritative.
- **Notifications**: `JobNotificationMiddleware.awrap_model_call` injects two kinds of transient `SystemMessage` into the model *request* — never into state, which is what keeps them out of the persisted thread (`create_agent`'s state has only `messages`/`jump_to`/`structured_response`, and the old `llm_input_messages` ephemeral channel no longer exists).
  - **Completion** ("background jobs finished" marker): final answer + report path for unannounced session jobs, marked announced *after* the model call succeeds. First of the notices — it is an instruction to act on now.
  - **Progress** ("background jobs still running" marker): one `progress_line()` per in-flight job ("2/4 steps done (research, documents) · running analysis · 3m elapsed"), derived from `plan` + `step_finished_at` alone — the job engine gained no bookkeeping for it. **Pushed, but only when it moved**: `progress_signature()` (status, plan size, steps landed — *not* elapsed time) is remembered in memory per session, so a stalled job costs nothing on the next turn and five turns of progress leave zero notices behind. Pull-only (`job_status`) was rejected: the model only calls it when the user thinks to ask, and the opaque wait is the bug. Cost: ~80 tokens on a turn that has news, 0 otherwise; `MAX_PROGRESS_JOBS` bounds both the tokens and the reloads (`list_jobs` returns summaries without the plan, so shown jobs are re-read with `get_job`).
- **Where the notices go is a provider constraint, not taste** (`_inject`): `langchain_anthropic._format_messages` raises *"Received multiple non-consecutive system messages"* for any `SystemMessage` not adjacent to the leading system block, so both notices go **directly after the system prompt** — appending last, or slotting before the final turn, breaks every turn on Claude (silently, since scripted-model tests never format anything). Anthropic hoists all system content into the top-level `system` param, so adjacency costs nothing; the conversation is left untouched, so the user's message stays the last message and a status line is never mistaken for the turn being answered.
- Tests script the model with `conftest.ScriptedChatModel` (AIMessages with `tool_calls`, `bind_tools` no-op, records inputs in `.calls`) — which is exactly why the assembled message list must ALSO be run through the real provider formatters (`langchain_anthropic.chat_models._format_messages`, `langchain_openai.chat_models.base._convert_message_to_dict`, `pytest.importorskip`-guarded: the chat extras are optional). No network needed, and it is the only thing that catches a message list a provider will refuse.

### HTTP API (`api/`)

`create_api(service) -> FastAPI` (extra `.[api]`; served by `jobsmith serve`, whichever agent is composed). A pure adapter: every route is serialization plus one `AgentService` call — if job or chat logic reappears here, it belongs in `service.py`.

- **Chat**: `POST /sessions` → id; `POST /sessions/{id}/messages` returns `{"type": "message"}` or `{"type": "proposal"}` (HITL interrupt surfaced over HTTP); client answers with `POST /sessions/{id}/approval {"approved": bool}`.
- **Jobs**: `GET /jobs[?session_id&status]`, `GET /jobs/{id}` (plan/step_finished_at/results — the UI's DAG data), `POST /jobs` (direct launch), `POST /jobs/{id}/cancel`, `POST /jobs/{id}/resume` (409 when the job has nothing left to run — `AgentService.resume_job` answers refusals as `{"status", "error"}` on both backings, so `DaemonClient` maps the code back into that dict).
- **Outputs**: `GET /jobs/{id}/outputs` lists the produced files, `/outputs/{name}` downloads one, `/report` is the inline shortcut to the main deliverable. `/report`'s content type is derived from the main output's `format` (`REPORT_MEDIA_TYPES` in `api/app.py`, unknown ⇒ `text/plain`) — mapping domain vocabulary to an HTTP header is the adapter's job, which is why the table is not in `jobs/report.py`. It serves *text* formats; a binary deliverable belongs on `/outputs/{name}`, whose `FileResponse` infers the type from the extension.
- **Live**: `GET /events` — SSE over `JobManager.subscribe()`: every `_persist_summary` emits `{job_id, status, steps_done, report_path, ...}` to subscriber queues (`put_nowait`, drops on full — never blocks a run; in-process only, same v1 scope as cancellation). SSE can't be exercised through httpx `ASGITransport` (it buffers the whole body) — the pub/sub is unit-tested in `test_jobs.py`, endpoints in `test_api.py`.

### Adding a capability

1. Subclass `Capability`; define `spec` (unique snake_case name), constructor taking needed clients, async node methods, `build()` via `self.state_graph(...)`, terminal nodes returning `self._emit_success(...)`/`self._emit_failure(...)`, and `render_context()` if its result should feed generation.
2. Register it in the composition root before `AgentBuilder(...).build()`. Nothing else: planner prompt, dispatch, merging all pick it up from the registry.

## Testing conventions

- `tests/conftest.py` — `FakeLLM` scripts responses by **substring of the system prompt** (`{"planner": ..., "ONLY the provided": ...}`); `plan_json()` builds planner responses. Fixtures: `checkpointer` (MemorySaver), `store` (InMemoryStore).
- `tests/test_banking_example.py` is the behavior-parity suite for the pre-refactor agent (French rejection messages, citation rule, vision-dropped-without-image).
- Tests import capabilities/stubs directly and assert on the final state dict (`terminal_kind`, `results`, `completed_capabilities`).

## Evaluating prompts (`evals/`)

A prompt change (router, planner, generator, a capability's own instructions) is
not judged by eye here: `make eval` scores it on a golden set of requests and
prints a table comparable with the previous run.

- **Properties, not expected text.** Nothing asserts an answer. The checks are
  structural: the plan names only registered capabilities, is duplicate-free,
  acyclic and has satisfiable dependencies; an obviously simple message is
  triaged `direct` and a compound one `plan`; the run reaches the expected
  terminal; every planned step ran and reported ok; the deliverable has a title,
  the answer and its provenance (job id + request). `scoring.py` holds one
  function per property, each returning pass / fail / **skip** — skipped checks
  leave the denominator, so a direct-route case never dilutes the plan checks.
- **The report checks are format-independent, and now really are.** They read
  the deliverable through `deliverable.extract(text, format)`, which returns
  the title and the visible text with the markup stripped, and compare needle
  and haystack after the same `normalize()` — so `- **web_search**` and
  `<li><strong>web_search</strong></li>` are one string. Before that they were
  markdown-shaped while the docstring claimed otherwise (`# ` is not how HTML
  opens; escaping moved the strings they searched for), and the golden set
  scored 13 checks lower in HTML purely on the format. `--report-format html`
  scores the other Reporter, and `tests/test_evals.py` pins the two runs to
  identical per-check tallies. An unknown format is read as plain text: a new
  Reporter is scored on its content from day one, only its *title* needs an
  extractor here.
- **Two tiers.** `structural` runs on `KeywordLLM` — no key, no variance — and
  is expected to be 100%: `tests/test_evals.py` asserts exactly that, so a
  prompt edit that breaks the machinery fails `make check`. `llm` needs a real
  provider, tolerates variance and **never gates CI** (`python -m evals` exits
  non-zero only on the deterministic fakes — the provider is the condition, not
  the tier label — or with an explicit `--fail-under`).
  A case declares which tiers it is meaningful in: one a keyword fake would pass
  or fail *by accident* is llm-only, otherwise the fake is what gets measured.
- **The harness runs the real product path** (`harness.py`): `build_app` →
  `create_job` → `run_job`, persistence forced to `memory`, reports into a
  scratch dir, a `KeywordChatModel` injected only so composition does not go
  looking for chat extras. The triage decision is read back from the
  checkpointer (`graph.aget_state`), because a planner rescuing a message the
  router should have sent direct is invisible from the outside.
- **Runs are stored, not just printed** — `evals/results/*.json` (gitignored),
  tagged with agent, provider, tier, case set, report format and git rev; the
  next run matching the first four is picked up as a baseline automatically and
  rendered as a Δ column. A `--case` slice is therefore never a baseline for the
  full set. The report format is recorded but deliberately NOT part of that
  match: the checks score the same property either way, so an HTML run is a
  legitimate baseline for a markdown one.
- Adding a case is one `EvalCase` in `cases.py`; adding a property is one
  function in `scoring.py` plus its name in `CHECK_NAMES` (a test pins the two
  together). Cases stay **domain-neutral** — `make leak-check` scans `evals/`
  too; an agent-specific golden set would live with that agent.
- **Say what it is worth.** Eleven cases sampled once from a stochastic model is
  a smoke signal, not a benchmark; the structural tier is the only part that is
  actually reliable, and it only proves the machinery holds.
