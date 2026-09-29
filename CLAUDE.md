# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file holds the rules; `docs/decisions/` holds why.** A rule that came out of a decision ends with `→ NNNN`, meaning `docs/decisions/NNNN-*.md`: the measurements, the alternatives that lost and why. Read the record before changing the rule; the index is `docs/decisions/README.md`. → 0102

**Size budget: 40 000 characters** (≈ 10k tokens), enforced by `tests/test_claude_md_budget.py`. 40k — under a quarter of the pre-migration size — fits the code map and the rules, not narrative. When the test fails, move history into a record; do not raise the number. → 0102

## Project overview

`jobsmith` is a **job engine** (`jobsmith/engine/`) — the product since 2026-09-27: it runs any LangGraph graph as a durable, trackable, cancellable **Job** and delivers each ending exactly once to a return address (`reply_to`). → 0161. On top sits one **reference graph**, the planner DAG (`jobsmith/dag/`): a registry-driven planner emits a DAG of capabilities, a wave-based executor fans them out, a generation pipeline merges results. **`jobsmith/agents/` holds the agent definitions** (a capability pack + a profile — `default` and `banking` ship — **or a graph of its own**, `AgentDefinition.graph`: no DAG, no chat); **`jobsmith/app/`** composes any of them (`build_app(agent=...)`). The bench — `chat/`, `cli/`, `api/`, `tui/` — drives and tests the engine; it is not the differentiator.

`README.md` is the human-facing counterpart of this file: product pitch, quickstart,
CLI/API surface, limits. Keep it in sync when a command or a limit changes.

## Commands

A Makefile wraps the common ones: `make help` lists them (`install`, `install-all`, `test [T=kw]`, `test-fast` = skip `slow`-marked, `lint`, `fix`, `types`, `check` = lint+types+leak-gate+tests, `eval`/`eval-llm` = score the golden set, `serve`/`chat`/`ui`/`jobs` = the global agent, `chat-banking`/`serve-banking`/`demo-banking` = the example, `clean`). Raw equivalents:

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,api,anthropic]" # setup
.venv/bin/python -m pytest tests/ -q # all tests
.venv/bin/python -m pytest tests/ -q -m "not slow" # the inner loop (make test-fast)
.venv/bin/python -m pytest tests/test_planner.py::test_cycle_rejected # one test
.venv/bin/ruff check . # lint
.venv/bin/pyright # types (config: [tool.pyright])
jobsmith serve [--port 8000] # ★ the daemon: it owns the job engine
jobsmith chat [--session ID] # ★ converse (daemon if up, else embedded)
jobsmith ui [--session ID] [--theme NAME] # ★ the same conversation on screen (.[tui])
jobsmith run "<task>" [--wait] | jobs | job <id> | report <id> | cancel <id>
# `python -m jobsmith …` is the same entrypoint when the venv's bin isn't on PATH
jobsmith --agent banking chat | serve # any agent, same shell
.venv/bin/python -m jobsmith.agents.banking.demo # scripted banking demo (fakes)
```

`/bg <text>` in the REPL runs a task in the background without waiting. One provider choice serves both stacks (`--llm=anthropic|openai|fake`, else by key: Anthropic, OpenAI, then the `KeywordLLM` fake). → 0000

### Working on this repo

- **A PR writes its decision record** in `docs/decisions/` (`NNNN` = issue number, `TEMPLATE.md`, a line in the index) whenever it takes a decision a later reader could undo without knowing why — the agent that decided writes it, in the same PR. **`CLAUDE.md` gains a line only when a rule is created or changed**: a short statement plus `→ NNNN`. Never narrative.
- **Run the `scribe` agent** (`.claude/agents/scribe.md`) every ~10 merges, or when this file nears its budget: it writes missing records, keeps the index, fixes stale claims and writes a brief in `docs/briefs/` covering `main` since the last one. → 0121
- **Effort is proportionate to the risk, never to the rules' maximum**: ≤ 45 min per issue, ≤ 15 min measuring; measure a prompt at its node with `make probe` (`evals/probes/<node>.json`: ≈10 phrasings, half controls, n=10; main vs branch in parallel, ≤ 300 calls), one variant, full-job evals only if that is inconclusive; record ≤ 20 lines; a small fix is done directly, not delegated; over budget → stop and report. → 0130
- **The protocol runs as commands, not by hand**: `make check` (parallel, `pytest -n auto`), `make probe`, `make mutate TESTS=…` (mutants on the diff's lines), `make combo PRS="a b"` for PRs in flight, `make hooks` (ruff + `uv lock --check` at commit), `gh pr merge --auto --squash`; `tests/test_decision_records.py` checks the index and every `→ NNNN`. → 0133, 0134
- **Falsify with the targeted test** (file or `-k`), not the whole suite: `make test-fast` (`-m "not slow"`) while iterating, the full suite once before the PR — a delegated task that re-runs everything per falsification pays the slow tests' cost every time. → 0109
- **One short-lived branch per issue**, off `main`: `feat/<n>-<slug>`, `fix/<n>-<slug>`, `chore/<slug>` (`gh issue develop <n>` creates one already linked). Open a PR, let CI run, merge, delete. **No `develop` branch** (no releases yet; PR + CI is the integration point). Releases, when they come, are tags.
- `main` stays green. CI (`.github/workflows/ci.yml`) runs what `make check` runs — lint, **types**, the leakage gate, tests — on push and PR across Python 3.11 and 3.12, **Postgres included** (a service sets `$JOBSMITH_TEST_PG`, → 0169), plus `uv lock --check`. CI installs **every** extra, since optional providers only resolve and type-check there. **A PR touching only `docs/**` and top-level `*.md` runs only the two docs tests**, under the same check names. → 0198
- **The type gate (`make types`, pyright)** is the only check that sees a signature that lies; `[tool.pyright]` configures editor and gate at once. → 0031
  - **`reportTypedDictNotRequiredAccess` is on**: `query` is `Required[str]` (guaranteed at entry); every other state key is guaranteed only by graph order and is read with `.get()` plus a default, next to a comment naming the node that guarantees it. **Never blanket-`# pyright: ignore` it** — a site where neither is honest is a finding about the graph. → 0031
  - `reportMissingImports` is a **warning**, not an error (lazy optional extras); scope is `jobsmith/` only, `typeCheckingMode: basic`. → 0031
- **`main` is protected**: PR required, the three checks must pass, admins included, no force-push. **Status checks are strict** — a branch must contain the current `main`, so CI validates the *post-merge* state. Each merge invalidates PRs in flight: `gh pr update-branch <n>` (or a rebase) and let CI re-run. → 0000
- **`uv.lock` is committed**; regenerate it (`uv lock`) in the same commit as any dependency change. → 0000
- **Parallel sessions use git worktrees**, one per issue — separate checkouts of the same repo, so two sessions never fight over the working tree or the current branch:

  ```bash
  make worktree B=feat/1-grounding # checkout + venv + .env, ~1s (uv cache)
  cd .claude/worktrees/feat-1-grounding
  make check # each worktree has its OWN .venv
  make worktree-rm B=feat/1-grounding # after the PR is merged
  ```

  **Gotchas**: a venv is path-specific — never symlink or copy one across worktrees; `.env`/`agent.db`/`artifacts/` are gitignored, so a fresh worktree has no API key until `make worktree` copies it. → 0000
- `make coverage`: the interactive layers (`cli/`, `chat/tools.py`) are the thin ones — a change there brings its tests with it. → 0000

Leakage gates (`make leak-check`, must return nothing): no `banking|banquier|votre|analyste` in shared code, `agents/default`, `agents/base.py` or `evals/` — **not** `agents/banking`, which may be as domain-specific as it likes; no product word (`ENGINE_WORDS`) in `engine/` (G4). → 0161

### The inbound port (`service.py`)

`AgentService` is **the** use-case surface — `JobService` + `ChatService`; the jobs half loads no chat — and `LocalAgentService` its in-process implementation over an `AgentApp`. Every entrypoint is an *adapter* over it, never a second implementation:

| adapter | backing |
|---|---|
| `cli/repl.py`, `cli/main.py` | `AgentService` (either backing) |
| `api/app.py` | `LocalAgentService` — each route is serialization + one call |
| `cli/client.py` `DaemonClient` | the same port, backed by HTTP |

- **Live progress and produced files are on the port**: `subscribe`/`unsubscribe`, `list_outputs`, `find_output`. `DaemonClient.subscribe` drops on a full queue like `InProcessEvents`; **`None` on the queue means the stream is over** and is never dropped. `tests/test_service.py` asserts identical answers from both backings. → 0048
- **The port throws exactly `BinaryDeliverable`, `ChatStreamError`, `ServiceUnavailable`** — that list is the whole contract; a 500 or an unparsable body is a defect, never translated. Front-ends catch `ServiceUnavailable` **by name**, never a broad `except`. → 0064
- **A graph agent has its own port**: `EngineService`/`LocalEngineService`; `AgentApp.service()` answers it for a graph agent, the DAG's otherwise. → 0165
- **A turn is a flow**: `stream`/`stream_approval` are abstract; `send`/`approve` are `terminal_of(self.stream(...))`. Seven events. **`Message.content` is the concatenation of the turn's `Token`s**, never the model's last message. Terminals: `{"type": "message", "content", "usage"}`, `{"type": "proposal", "query", "rationale", "sources", …, "usage"}`. → 0050, 0083, 0086
- **The chat stream never drops a token**: no queue on the path (runner generator → `StreamingResponse` → `aiter_lines`); an undecodable SSE line or a stream with no terminal raises `ChatStreamError`. `/events` is the opposite and sheds. → 0050

### CLI + daemon (`cli/`) — where jobs actually run

The point of this layer: **a job must outlive the command that launched it**. `jobsmith serve` is a long-lived process owning the JobManager; every other command is a *client*.

- `cli/client.py` — two backings for that one port: `DaemonClient` (HTTP) and `EmbeddedClient` (nothing but `LocalAgentService`). `open_client()` probes `GET /health`, falls back to embedded. → 0000
- **All diagnostics go to stderr** (banners, tool activity); stdout stays pipeable and carries the answer's tokens; both are flushed per write. → 0000
- The REPL renders the flow with no fallback branch and never prints the terminal (the tokens delivered it); a mid-turn `ChatStreamError` is said on stderr and the loop continues. → 0050
- **The job notice is on stdout**, one `job_lines` renderer for notice and proposal, ending `stop it : /cancel <id>`. → 0083
- **The plan is activity, not record**: `job_planned` goes to stderr as `… plan: a + b → c` (waves from `dag.state.plan_waves`); the TUI says it on the activity line, never in the conversation. → 0086
- `run_repl` wraps each command in one `except ServiceUnavailable`; `run_command` likewise (exit 1). → 0064
- `cli/main.py` — argparse; `--llm` is exported as `$JOBSMITH_LLM` so `pick_provider` sees it from both stacks. Bare `jobsmith` == `jobsmith chat`.
- Sessions are **rebuildable by id**: the API's registry is only a cache, the conversation lives in the checkpointer under `thread_id=session_id`, so `--session <id>` resumes across a daemon restart (and gets the finished-job announcement); `session_factory` takes an optional `session_id`.
- Embedded mode: a *running* job stops with the process (`recover_interrupted` settles it next start); records are kept. → 0063

### Agents (`agents/`) — what an agent *is*

An agent is **a capability pack + a profile (+ an optional persona, + whatever it needs open)**, or **a graph of its own** (`graph=`, a `GraphSpec` factory: `AgentApp.engine` runs it, no DAG, no chat) — `AgentDefinition` in `agents/base.py`. Shared by all: **adding an agent touches no shared code** — define the capabilities, register in `agents/__init__.py`, done. `tests/test_agents.py` pins that.

```python
AgentDefinition(
    open_resources=..., # async (stack) -> anything: pools, sessions, clients
    capabilities=..., # (AgentContext) -> list[Capability] ctx.llm / ctx.resources
    profile=..., chat_prompt=...,
)
```

- `open_resources` gets `build_app`'s `AsyncExitStack`; teardown is reverse order on `AgentApp.aclose()`, even after a failed startup. Several capabilities on one backend share a **pool**, never a fat client. → 0000

- `agents/default/`: `read_files`/`prior_jobs`/`documents`/`web_search` → `research` → `analysis` → `critique`, plus `slide_deck`. `analysis`/`critique` subclass `SingleStepCapability` (`_step.py`); `critique` overrides `_material` to read both. → 0000
  - **`documents` is the grounding step**, over the `DocumentSource` port (`sources.py`; `LocalFiles` is keyword ranking, no key, no network). → 0000
  - **`read_files` reads a named file** (`documents` searches): path in `inputs["source_files"]`, never parsed from the query; **a refusal is material, not silence**. → 0060
  - **`prior_jobs` reads an earlier RUN, not its file** (`inputs["from_jobs"]`, bounded, session scope in `chat/tools.py`); it and `read_files` are **first in the registry** (`KeywordLLM` chains in order). → 0074
  - **`web_search`** = `DocumentsCapability` over `TavilySource`: pages, not snippets, cut per document; an HTTP error raises. → 0075
  - **A capability nothing can serve stays out of the registry** — every conditional step is registered only when something backs it (`open_default_resources`); an empty registry is then answered directly. → 0000, 0038
  - **`slide_deck` is a generation, not a report format**: deck structure is asked of the model; only `pptx_deck.py` imports `python-pptx`; 16:9; refused without a job before the LLM call; the deck is an `annex`. **Its description says what it is NOT.** → 0035, 0061
  - **The deliverable is written for its reader, and answers**: prompts oblige the answer first, from the material, doubt marked where it bears; `SUBJECT_ONLY_RULE` names the deck too (a separate step builds it, → 0129); `NO_ANSWER_INSTRUCTION` sets a high bar and shape for a refusal. → 0058, 0073
  - **The generator is told which files the run delivers** and names no other; when the list names the answer itself, `ANSWER_FILE_RULE` says that entry **is** the text — never "delivered separately". → 0077, 0126
  - Retrieved passages carry a **quotable id** (`path#chunk`).
  - **`research` reads every retrieval step's material** and `read_files`' refusals, and says so in `meta["grounded_on"]`; **`critique` checks the subject, not the work**, and feeds the generator. → 0081, 0082
- `agents/banking/`: the domain example, with its **own ports** next to its capabilities (`deps.py`) and its own adapters (`fakes.py`); `vision` is registered only when the LLM satisfies `VisionClient`. → 0000
- Selection: `--agent NAME` (CLI, applies to whichever process owns the engine — so pass it to `serve`), `build_app(agent=...)`, `make chat AGENT=banking`.

### The composition root (`app/`)

Wiring only, no content — everything here is domain-neutral:
- `providers.py`: `pick_provider` (one `--llm=` flag / key auto-detect for both stacks), `make_llm`/`make_chat_model`, `load_dotenv`, and the keyless fakes `KeywordLLM`/`KeywordChatModel`. → 0000
- `agent.py`: `build_app(agent=..., **overrides) -> AgentApp` composes ONE agent on one `AsyncExitStack`. **`reports_dir` resolves once, to an absolute path** (`pick_reports_dir`); `AgentContext(readable_roots=(reports_dir,))`; one `StoreJobRepository` serves the manager and the `PriorJobSource`. → 0063
- `persistence.py`: `pick_db()` (arg > `--db=` > `$JOBSMITH_DB` > `<data dir>/jobs.db`) + `open_persistence(spec, stack)` → `(checkpointer, store)`, teardown on the caller's `AsyncExitStack`.
  - **Default: a SQLite file under `data_dir()`**, never the cwd; SQLite is a core dependency. **`memory` must be asked for by name**: tests, `evals/harness.py` and the demo pass `db="memory"`; `conftest.sandboxed_data_dir` fails the run if the suite wrote to the data dir. → 0063
  - **Every SQLite writer takes the lock up front** (`_ImmediateBegin`), so a contended batch waits rather than fails "database is locked". → 0063
  - **Postgres setup runs under a *tried* advisory lock** (`_took_setup_lock`): a waited one deadlocks with the migrations' `CREATE INDEX CONCURRENTLY`. → 0169
  - Backends: `memory`, a SQLite path (the default, or any path you name), or a Postgres DSN (`.[postgres]`, one shared `AsyncConnectionPool` for saver + store, `autocommit`/`prepare_threshold=0`/`dict_row`). Chat sessions share the job graph's checkpointer (`thread_id`: `session_id` vs `job_id`).

**`build_app` is async**: real backends must open in the loop that uses them — `jobsmith serve` runs `await uvicorn.Server(config).serve()`, never `uvicorn.run()`. **SQLite gotcha**: never run a stray `PRAGMA` on the live connections (deadlock); the store needs `isolation_level=None`, the saver keeps the default. → 0000
- **Interrupted jobs**: `JobManager.recover_interrupted()` (in `build_app`) marks FAILED a RUNNING record whose owner is provably gone; a live owner's job is left alone; QUEUED stays runnable. → 0010
  - **`relaunch_interrupted()`** (called by `serve` only) then resumes such a job again, up to `GraphSpec.relaunch` tries (0 = never), claimed on a shared store like ownership. → 0189

## Architecture

### Layers

`engine` imports nothing of ours; `artifacts`/`dag` may import `engine` (`dag` also `artifacts`); `adapters/langchain` may import `engine`; the bench (`agents/`, `chat/`, `service.py`, `app/`, `api/`, `cli/`, `tui/`) may import anything — gate G3, `tests/test_layers.py`'s `ALLOWED` allowlist, empty since the split; a needed new crossing is added there, with a reason. → 0161

### The OO pattern

Every step of the reference graph (`dag/`) is a class instance owning its deps and config: node logic is **async instance methods** (`g.add_node("planner", self.planner.run)`), routers are **sync methods**, capabilities expose `.build()` as one compiled sub-graph node. `AgentBuilder` (`dag/builder.py`) holds every step instance (swap one before `.build()` in tests).

### Core concepts (read these files first)

- **`dag/capability.py`** — `Capability` ABC + `CapabilitySpec` (name, description, JSON-schema dicts, `requires_inputs`). Capabilities take *exactly the clients they need*; the framework never introspects them. Terminal sub-graph nodes call `_emit_success`/`_emit_failure` so every capability reports uniformly.
- **`dag/registry.py`** — `CapabilityRegistry`: single source of truth for what the agent can do. The planner prompt, executor targets and builder node map all derive from it. **Frozen at `build()`** — a compiled graph's capability set is fixed; new capability ⇒ new `AgentBuilder` (compilation is milliseconds).
- **`dag/state.py`** — capability results live in one `results: dict[str, CapabilityResult]` with a dict-union reducer. Fan-in safety: each capability writes only its own key; registry-unique names + no-duplicate plan steps ⇒ disjoint keys. **Determinism caveat:** consumers must iterate in *plan order*, never dict order (ContextMerger does).
- **`engine/usage.py`** — an **ambient ledger** (`ContextVar` per run) adapters push into with `record_usage`, plus LangChain calls via the runner's `ModelCallUsage` callback; scope = the root node of `checkpoint_ns` (a capability's is `cap_<name>`), else `unattributed`; `$JOBSMITH_PRICES`; unpriced ⇒ `cost_usd: None`; the conversation's own calls are summed per turn (terminal `usage`, #173). → 0002, 0161
- **`artifacts/paths.py`** — `safe_name` (one component) and `resolve_within` (refused unless it **lands** in a declared root; resolves before comparing). → 0060
- **`dag/prior_jobs.py`** — the `PriorJobSource` port, in `dag/` because the composition root supplies it. → 0074
- **`dag/profile.py`** — `AgentProfile` is the entire domain surface: prompt templates, user-facing messages, input/output validation rules (plain callables), `max_refine`. Core defaults are neutral English; the banking example overrides them (French messages live *only* in `agents/banking/profile.py`).

### Graph flow

```
validate_input → document_intent → router ─(direct | empty registry)→ direct_answer ┐
                                     └(plan)→ planner ─(nothing applicable)→ ───────┤
                              └→ executor_dispatch ⇄ {cap_<name> × registry}
                                ↓ (all done)                     ↓
                          merge_results → generation → validate_output
                                              ↑ refine ←┘ (≤ max_refine)  → post_process → write_document → END
                                                         └(could not answer)→ unanswered → write_document → END
errors: execution_error → escalate (some ok result) | user_error (none) → END
```

- **Router** (`dag/router.py`) is a dedicated triage node — the planner never decides *whether* to plan. Routes live in `Router.routes` (`plan`, `direct` → `DirectResponder`); **fail-open** to `plan`. New route = `Router.routes` entry + node + `AgentBuilder.route_targets` entry. → 0000
  - An **empty registry** routes `"direct"` structurally, before the LLM call, fallback included. → 0038
  - **The direct route is told the files the run delivers**; asked for as a file, its reply is that document (`DIRECT_DOCUMENT_RULE`) — the request is never moved to the planner for it. → 0080
- **Document intent** (`dag/document.py`), a decision node **before triage**: fills silence, never overrides (a seeded `document_formats` returns before any model call), chooses from `available_formats(registry)`, proves it (`confirm`) so it cannot refuse, fail-open, and publishes the `formats` fact. → 0090, 0108
  - **No document unless asked**: a file only when `formats` is non-empty (`DagJob.deliverable_expected`); silence is the fact `formats = None`. **`None` (may be filled) and `[]` (never overridden) are two facts** — never `formats or []`. → 0096
  - **A file asked for in words is asked for**: "save it to a file" is `requested` whatever else the request asks; a file it is only about is not (`FILE_REQUEST_RULE`). → 0125
- **Planner** (`dag/planner.py`) renders its prompt from `registry.specs()`, validates the LLM's JSON DAG: names against the registry, drops steps whose `is_applicable(state)` is false (generalizes "vision only if image" via `spec.requires_inputs`), **prunes dropped names from surviving `depends_on`**, Kahn cycle check.
  - A plan emptied by applicability routes to `direct_answer` (`_route_after_planner`); `{"steps": []}` from the model is still an error. → 0038
- **A refusal is data**: `split_declaration` reads a first-line marker only; `_route_validate_output` sends it to `unanswered`, never to refine; the job stays DONE, `terminal_kind` says so. → 0059
- **Executor** (`dag/executor.py`) is a pass-through node + router: computes ready capabilities each wave and returns `list[Send]`; capability nodes edge back to `executor_dispatch`. This executes an arbitrary DAG without a baked-in schedule.
- **Two error channels**: `NodeError.recoverable=False` (planner/generation failures) hard-stops into `execution_error`; capability failures are recoverable — they land in `results` with `ok: False` and the run degrades gracefully.
  - **A transient one is retried**: `_emit_failure(retryable=True)` (a raised model call, never an empty answer); the executor re-dispatches it, before its dependents, within `AgentProfile.max_step_retries`; no replanning. → 0191

### Critical invariant: capability output schema

Capability `build()` MUST use `self.state_graph(PrivateState)` (sets `output_schema=CapabilityOutputState`); without it the sub-graph echoes its full state (including `query`) and two capabilities finishing in the same superstep collide with `InvalidUpdateError`. Private states extend `CapabilityBaseState`. **`Send` also strips the append-only channels** (`completed_capabilities`, `errors`) — never let a sub-graph echo them back and miscount a retry (→ 0191). → 0194

### Jobs layer (`engine/`)

`JobManager` holds **only the use cases** — `create_job` / `run_job` / `start_job` / `run_for` (waits, then promotes) / `resume_job` / `start_resume` / `answer_job` / `start_answer` / `amend_job` / `get_job` / `list_jobs` / `cancel_job` / `recover_interrupted` / `relaunch_interrupted` (paired awaitable/background-task twins). Everything else is a collaborator behind a port, so each changes for its own reason:

| collaborator | responsibility | file |
|---|---|---|
| `JobRepository` | where records live + **the store schema** | `engine/repository.py` |
| `GraphRunner` | drives the run, translates it to domain updates | `engine/runner.py` |
| `JobEvents` | broadcasts progress | `engine/events.py` |
| — | the deliverable: the run writes it (`write_document`, Reporters) and declares it as `artifact:` facts | `dag/deliver.py` |

Defaults wire the v1 stack, so `JobManager(graph, store)` still works; pass `repository=`/`runner=`/`events=` to swap one. `tests/test_job_seams.py` drives the whole lifecycle with **no graph and no store** — if that stops being possible, a responsibility has leaked back in.

- **Only `runner.py` knows LangGraph's stream shape**; it yields `NodeFinished`/`Fact`/`Output` and knows nothing of the graph. Graph nodes stay job-agnostic: **new persistence goes in the manager or the repository, never in a node.** → 0000
  - The `plan` fact persists a summary as soon as the plan exists. **Which step finished is the `step:<cap>` fact's key**, never read from `results`. → 0048, 0053
- **Only `repository.py` knows the schema**: `("jobs_v1","index")/job_id` → summary (graph, label, input, result…); `("jobs_v1",job_id,"facts")/key` → a fact the run published (full view has values; summary has `facts_at`). Fine-grained state stays in the checkpointer under `thread_id == job_id`; moving records to SQL is another implementation of this port.
- **`load_all` is complete**: filters applied by the store, one query, nothing cut; `list_jobs` orders newest first, **then** cuts; what must see everything (announcements, an id, orphans, events) passes `limit=None` — never a bigger number. `tests/test_job_history.py` seeds 250 jobs on every backend. → 0141
- **`dag/prior.py`** (`RepositoryPriorJobs`) is the only place that knows how a finished run's material is reached. → 0074
- **Ownership is on the record** (`engine/ownership.py`, `("jobs_v1", id, "control")`): a lease written **before** RUNNING, heartbeat 2 s, TTL 30 s; **a cancel is a request the owner acts on**, never a status written over its run. → 0010
- **Events cross processes on a shared database** (`WatchedEvents`, watching **only while someone is subscribed**, announcing a job whose `updated_at` moved): SQLite polls `PRAGMA data_version` on its own read-only connection, re-reading only when it moved; Postgres `NOTIFY`s on publish and `LISTEN`s off the pool; memory stays `InProcessEvents`. `$JOBSMITH_TEST_PG` gates the Postgres tests. → 0100, 0138, 0169
- **Resume** re-enters with `None`, gated on CANCELLED/FAILED **and** non-empty `runner.pending()`; seeds usage; `_begin_resume` clears `error`, `result`, `delivered_at`, counts `attempt`. A task a stop caught between its writes is pending, and the resume forks the checkpoint first (`GraphRunner._repaired`, → 0202). **No partial re-run of a finished DAG**: that request is a new job with `from_jobs` on the old one, never a new attempt. → 0005, 0168
- **A run paused at an `interrupt()` waits for an answer**: `needs_input` + `Job.asked`; `answer_job` resumes it with `Command(resume=…)` as a new attempt, clearing `asked`; not an ending, never delivered. → 0167
- **A FAILED job says why, as data**: `Job.failure = {kind, pending, retryable}`; `retryable` is the same test the resume gate applies (`bool(pending)`), so the two cannot disagree. → 0187
- **`amend_job`** stops a running job, writes an update into its checkpoint (`runner.update`), and resumes it as a new attempt — the stop is never delivered. The DAG's meaning is `DagJobs.drop_steps`: a finished step stays, a running one restarts unless dropped. → 0177. The chat's `skip_steps` (refusals returned as text) and `POST /jobs/{id}/drop` (409) reach it. → 0196
- **Vocabulary**: an **output** is what the job produces for the human (`DagJob.outputs`, role `main`|`alternate`|`annex`); a **result** is a capability's payload (`results`). `DagJob.report_path` = the main output's path; the engine's record has neither (`dag/jobs.py` derives them from facts). → 0000
- **Reporters** (`dag/report.py`): `build_document` → `JobDocument` → `FileReporter` subclasses (`render`, or `serialize` for bytes); `is_binary_format`. → 0009
- **Exactly one output is `role="main"`** (the first format; the rest `alternate`, never `annex`); `compose_reporters` refuses two Reporters on one extension. `pick_report_formats()` only says *which* file when one is wanted and none was named. A failed write leaves the job DONE; the view's `error` names the format. → 0028, 0096
- **An annex is a file a step declared** via `ArtifactStore` + `artifact_meta(...)`: `_emit_*` publishes it as an `artifact:` fact, kept at **every** terminal since facts persist as they arrive; the DAG view lists it in plan order, once per path; missing when declared shows in the view's `error`. → 0035, 0041
- **Name, title and formats are facts on the DAG job's input**, refused in `DagJobs.create_job`; a title is cut on a word (`document_title`). → 0055, 0054
- **The report is title + answer + one `job_reference` line**; the record holds the rest (`with_provenance` to recite it). → 0085
- **HTML takes no dependency**: escape everything first, add only our own tags, no links, nested lists clamped to the open-list stack. **The PDF is that page printed**; `.[pdf]` needs pango/cairo, **offered when installed** (`find_spec`), **loaded on first need** (eager only when PDF is the deployment default). → 0009, 0076, 0034, 0108
- **Capabilities present their own results**: `Capability.render_report(result)` (twin of `render_context`, targeting the model) with `default_result_markdown` as base — prose stays prose, list[str] becomes bullets, structured values fall back to JSON. Never grow it to learn payload shapes: override `render_report` instead.
- `Job.usage` refreshes on every persist; per-step usage is in `meta["usage"]`, failures included. → 0002
- **Delivery** (`engine/delivery.py`): `reply_to` is JSON, indexed by a flat `reply_key`; an ending asks the kind's deliverer, True stamps `delivered_at`. `none` delivers as it settles; a pulled kind waits for `pending_deliveries`/`mark_delivered`. The DAG registers `session`. → 0161
  - **A pushed kind (`Pushed`) never delivers inline**: saved first, then pushed by its own task (backoff, one per `job_id#attempt`), stamped only after it lands; `recover_interrupted` retries what is pending. `Webhook(allowed)` is the first one. → 0166

### Chat layer (`chat/`)

`ChatSession(manager, model, *, session_id, system_prompt, checkpointer).build()` → a `langchain.agents.create_agent` (LangChain model, NOT the DAG's `LLMClient` (`dag/deps.py`): the two-stack split).

- **`chat/runner.py` alone knows what `astream` emits**: `Token`, `ToolStarted`/`ToolFinished` (real tool name; wording is the front-end's), `JobStarted`, `JobPlanned`, then exactly one terminal, `Message` or `Proposal`. → 0050, 0083, 0086
- **Tools** (`chat/tools.py`) wrap JobManager use-cases, scoped to the session's own jobs; the generic launch tool is `adapters/langchain.launch_tool` (G5); the chat's stays a shell over `run_for` + `told_in_thread`. → 0172 `launch_job` **runs the task**: `create_job(session_id=...)` + `run_for(job_id, pick_sync_timeout())`.
- **Synchronous by default, promoted by the clock**: `JobManager.run_for` = `start_job` then `asyncio.wait` (never `wait_for`) for `$JOBSMITH_SYNC_TIMEOUT` (20 s; `0` = never); no classification; a job finished in the turn goes into `delivered_jobs` with the tool's result. → 0083
- **The plan is announced while the turn waits**, from `manager.subscribe()` (subscribed before `start_job`, drained with `None` when the run ends, released in `finally`); never after promotion; a one-step plan is not announced. → 0086
- **The answer is written verbatim into the turn**, never returned through the model; a promoted answer uses the same channel up to `$JOBSMITH_INLINE_ANSWER_MAX` (2 000), or at any length when no file was written. → 0083, 0085
- **The approval card is a notice** (`job_started`: query, sources, `from_jobs` as short id + start of query, name/title/formats, job id); the gate survives behind `$JOBSMITH_APPROVE_JOBS`. → 0083, 0104
- **The engine never sees the thread**: a self-contained `query`, plus `recent_conversation()` as `inputs[CONVERSATION_INPUT_KEY]`. `source_files` and `from_jobs` (resolved against **this session's** jobs) are `launch_job` arguments. → 0004, 0060, 0074
- **Notifications are transient `SystemMessage`s in the model request, never state**: completion = `adapters/langchain.JobDeliveryMiddleware` (chat: `JobNotificationMiddleware`), progress = `adapters/langchain.JobProgressMiddleware` (root steps; chat's: the plan), both **directly after the system prompt** (`inject`), where Anthropic hoists them. → 0006
  - **Completion is told once per ending**: `delivered_jobs` (id → attempt, never a time: #170) enters the thread with the answer (`ExtendedModelResponse`); `delivered_at` marks after, in `aafter_model`. → 0161 (G5)
- **`conftest.ScriptedChatModel` implements `_astream`** with several chunks; the message list is also run through the real provider formatters, which only run with the chat extras (CI). → 0006

### HTTP API (`api/`)

`create_api(service) -> FastAPI` (extra `.[api]`; served by `jobsmith serve`, whichever agent is composed). A pure adapter: every route is serialization plus one `AgentService` call — if job or chat logic reappears here, it belongs in `service.py`.

- **Chat**: `POST /sessions`, `/sessions/{id}/messages`, `/sessions/{id}/approval`, each with a `/stream` SSE twin (no queue behind it). → 0050
- **Jobs**: `GET /jobs[?session_id&status]`, `GET /jobs/{id}` (plan/step_finished_at/results, the UI's DAG data), `POST /jobs[?wait=S]` (`wait` = `run_for`), `POST /jobs/{id}/cancel`, `POST /jobs/{id}/resume` (409 when nothing is left to run; `resume_job` answers refusals as `{"status", "error"}` on both backings, `DaemonClient` maps the code back).
- **Engine**: a graph agent's own routes, untouched by the DAG's — `POST /engine/jobs[?wait=S]`, `GET /engine/jobs[/{id}]`, `.../cancel`, `.../resume`, `.../answer` (a paused `needs_input` job). → 0165, 0167
- **Outputs**: `/jobs/{id}/outputs[/{name}]`; `/report` serves text (type from `REPORT_MEDIA_TYPES`), **415** for a binary deliverable. → 0034
- **Live**: `GET /events`, in-process, drops on full; untestable through `ASGITransport` (hangs), so `DaemonClient.subscribe` is tested under uvicorn. → 0048

### Terminal UI (`tui/`)

`jobsmith ui` (`.[tui]`) renders the same port and events beside `jobsmith chat`; `subscribe()` drives every repaint, no poll, no timer. → 0048

- **Stylesheet: standard theme roles only** (an invented one fails at parse time); CSS says `$text-muted`, markup `$foreground-muted`; an undefined markup variable is silently dropped. → 0048
- An event says *something changed*: re-read, join a refresh in flight, remember a skipped one. A turn in flight refuses input and keeps the text. F8 arms before cancelling; bindings are function keys. Model text is `escape`d. → 0048
- **A snapshot never proves a fact** — assert what a pane says by name; `pytest-textual-snapshot` is pinned exactly. → 0048

### Adding a capability

1. Subclass `Capability`; define `spec` (unique snake_case name), constructor taking needed clients, async node methods, `build()` via `self.state_graph(...)`, terminal nodes returning `self._emit_success(...)`/`self._emit_failure(...)`, and `render_context()` if its result should feed generation.
2. Register it in the composition root before `AgentBuilder(...).build()`. Nothing else: planner prompt, dispatch, merging all pick it up from the registry.
3. To **read a named file**: `requires_inputs=(SOURCE_FILES_INPUT_KEY,)`, a port in the constructor, never `Path.read_text`; let the port refuse. → 0060
4. To read an **earlier job**: `requires_inputs=(FROM_JOBS_INPUT_KEY,)`, `ctx.prior_jobs`, never reach into `engine/`; bound what travels and write every cut into the text. → 0074
5. To produce a **file**: `ctx.artifacts.write(job_id_of(state), name, data)` and `meta=artifact_meta(ArtifactRef(path, title=...))` on `_emit_success` (or `_emit_failure`); never build a path. → 0035

## Testing conventions

- **Tests are organised by purpose, never by issue**: a file per component or rule (`test_deliverable.py`, not `test_<issue>.py`); a change extends the file of the rule it touches. A docstring states the property in a sentence, `→ NNNN` for the why — history lives in the record. Cases that differ only in data are one parametrized test. → 0110
- **A prompt is asserted through the named rule it must carry** (`SUBJECT_ONLY_RULE`, `CAVEATS_RULE`, `BRIEF_RULE`, `UNREADABLE_RULE`: `RULE in prompt`), never by its wording — wording is the evals' to judge. A rule worth a test is a constant in the product. → 0110
- **Shared builders live in `tests/support.py`** (`OneStep` — a one-node capability: write `work` only —, `SlowEcho`, `ChartCapability`, `make_manager`, `planning`, `make_session`, `chat_turn`, `service_over`, `make_app`, `wait_done`, `until` — wait on state, never a fixed sleep —, `StubPdf`); **a test file never imports from another test file**. → 0110
- `tests/conftest.py` — `FakeLLM` scripts responses by **substring of the system prompt** (`{"planner": ..., "ONLY the provided": ...}`); `plan_json()` builds planner responses. Fixtures: `checkpointer` (MemorySaver), `store` (InMemoryStore).
- `tests/test_banking_example.py` pins the banking agent's behavior (French rejection messages, citation rule, vision dropped without an image).
- Tests import capabilities/stubs directly and assert on the final state dict (`terminal_kind`, `results`, `completed_capabilities`).
- **The default registry is configuration-dependent**: ask `conftest.registered_capabilities(app)`, never hardcode it (CI installs `.[pptx]`); its **order** is load-bearing for `KeywordLLM`. → 0035

## Evaluating prompts (`evals/`)

A prompt change (router, planner, generator, a capability's own instructions) is
not judged by eye here: `make eval` scores it on a golden set of requests and
prints a table comparable with the previous run.

- **Properties, not expected text**: one function per property, pass / fail / **skip**. Report checks read through `deliverable.extract`; a binary first format is refused. → 0003, 0025
- **Two tiers**: `structural` (`KeywordLLM`) must be 100% and gates `make check`; `llm` never gates. **Blank the keys for the fake tier** — `.env` refills with `setdefault`: `TAVILY_API_KEY= ANTHROPIC_API_KEY= OPENAI_API_KEY=`. → 0003
- Checks are scored against the **case**, never the job record; answer checks through `_answer_applies`, file checks through `_report_applies`; retrieval is recognised by the **shape** of a result. → 0073, 0081, 0090, 0096
- Adding a case is one `EvalCase` in `cases.py`; adding a property is one
  function in `scoring.py` plus its name in `CHECK_NAMES` (a test pins the two
  together). Cases stay **domain-neutral** — `make leak-check` scans `evals/`
  too; an agent-specific golden set would live with that agent.
- **`answer_invents_no_file` is the one check read against the record**: whether the answer tells the truth about `Job.outputs` is a fact, not a decision the case holds; a file kind the request or material already names is skipped. → 0077
- **The structural tier must exercise every check.** The llm tier is a smoke signal, not a benchmark. → 0003
