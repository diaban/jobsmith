# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file holds the rules; `docs/decisions/` holds why.** A rule that came out of a decision ends with `→ NNNN`, meaning `docs/decisions/NNNN-*.md`: the measurements, the alternatives that lost and why. Read the record before changing the rule; the index is `docs/decisions/README.md`. → 0102

**Size budget: 40 000 characters** (≈ 10k tokens), enforced by `tests/test_claude_md_budget.py`. Every session carries this file through every turn; at 174k it was the largest fixed cost of a delegated task and went stale unowned. 40k — under a quarter — fits the code map and the rules, not narrative. When the test fails, move history into a record; do not raise the number. → 0102

## Project overview

`jobsmith` is both a **product** — a general-purpose conversational agent (`python -m jobsmith`) that answers simple messages directly and **runs** everything else on the job engine: a task runs inside the turn and answers there, and one that outlives a clock is promoted to a **background job** and surfaced back into the conversation when it lands (#83) — and the **domain-agnostic framework** it is built on (LangGraph, object-oriented node pattern): a registry-driven planner emits a DAG of pluggable capabilities (self-describing agentic sub-graphs); a wave-based executor fans them out in parallel; a generation pipeline merges their results; each run is a persistent, trackable, cancellable **Job**; a chat layer (`jobsmith/chat/`, LangGraph prebuilt ReAct agent) sits on top. **`jobsmith/agents/` holds the agent definitions** (a capability pack + a profile: `default` = grounding (`read_files`/`prior_jobs`/`documents`)→research→analysis→critique (+ `slide_deck`), `banking` = the domain example) and **`jobsmith/app/` is the composition root that runs any of them** (provider selection, persistence, `build_app(agent=...)`).

`README.md` is the human-facing counterpart of this file: product pitch, quickstart,
CLI/API surface, limits. Keep it in sync when a command or a limit changes.

## Commands

A Makefile wraps the common ones: `make help` lists them (`install`, `install-all`, `test [T=kw]`, `test-fast` = skip what's marked `slow`, `lint`, `fix`, `types`, `check` = lint+types+leak-gate+tests, `eval`/`eval-llm` = score the prompts on the golden set, `serve`/`chat`/`ui`/`jobs` = the global agent, `chat-banking`/`api-banking`/`demo-banking` = the example, `clean`). Raw equivalents:

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,api,anthropic]"  # setup
.venv/bin/python -m pytest tests/ -q                        # all tests
.venv/bin/python -m pytest tests/ -q -m "not slow"          # the inner loop (make test-fast)
.venv/bin/python -m pytest tests/test_planner.py::test_cycle_rejected  # one test
.venv/bin/ruff check .                                      # lint
.venv/bin/pyright                                           # types (config: [tool.pyright])
jobsmith serve [--port 8000]                                # ★ the daemon: it owns the job engine
jobsmith chat [--session ID]                                # ★ converse (daemon if up, else embedded)
jobsmith ui [--session ID] [--theme NAME]                   # ★ the same conversation on screen (.[tui])
jobsmith run "<task>" [--wait] | jobs | job <id> | report <id> | cancel <id>
# `python -m jobsmith …` is the same entrypoint when the venv's bin isn't on PATH
jobsmith --agent banking chat | serve                       # any agent, same shell
.venv/bin/python -m jobsmith.agents.banking.demo            # scripted banking demo (fakes)
```

`/bg <text>` in the REPL runs a task in the background without waiting. One provider choice serves both stacks (`--llm=anthropic|openai|fake`, else by key: Anthropic, OpenAI, then the `KeywordLLM` fake); the engine's adapters are `jobsmith/clients.py` (they raise `RuntimeError` on refusals and drop `temperature`), the chat's are LangChain's. → 0000

### Working on this repo

- **A PR writes its decision record** in `docs/decisions/` (`NNNN` = issue number, `TEMPLATE.md`, a line in the index) whenever it takes a decision a later reader could undo without knowing why — the agent that measured and chose writes it, in the same PR. **`CLAUDE.md` gains a line only when a rule is created or changed**: a short statement plus `→ NNNN`. Never narrative.
- **Run the `scribe` agent** (`.claude/agents/scribe.md`) every ~10 merges, or when this file nears its budget: it writes missing records, keeps the index, fixes stale claims and writes a brief in `docs/briefs/` covering `main` since the last one. Each pass is a PR paying a full CI round. → 0121
- **Effort is proportionate to the risk, never to the rules' maximum**: ≤ 45 min per issue, ≤ 15 min measuring; measure a prompt at its node with `evals/probe.py` (≈10 phrasings, half controls, n=10, before/after, ≤ 300 calls), one variant, full-job evals only if that is inconclusive; record ≤ 20 lines; a small fix is done directly, not delegated; over budget → stop and report. → 0130
- **Falsify with the targeted test** (file or `-k`), not the whole suite: `make test-fast` (`-m "not slow"`) while iterating, the full suite once before the PR — a delegated task that re-runs everything per falsification pays the slow tests' cost every time. → 0109
- **One short-lived branch per issue**, off `main`: `feat/<n>-<slug>`, `fix/<n>-<slug>`, `chore/<slug>` (`gh issue develop <n>` creates one already linked). Open a PR, let CI run, merge, delete. **No `develop` branch**: there are no releases yet, so it would only add a merge — the PR + CI is the integration point it used to provide. Releases, when they come, are tags.
- `main` stays green. CI (`.github/workflows/ci.yml`) runs what `make check` runs — lint, **types**, the leakage gate, tests — on push and PR across Python 3.11 and 3.12, plus `uv lock --check` so the lockfile cannot silently drift from pyproject. CI installs **every** extra, so it is the stricter reading: the optional providers' imports resolve there and are type-checked, where a plain `make install` (`.[dev,api]`) leaves them unresolved.
- **The type gate (`make types`, pyright)** is the only check that sees a signature that lies; pyright rather than mypy because Pylance is pyright, so `[tool.pyright]` configures editor and gate at once. → 0031
  - **`reportTypedDictNotRequiredAccess` is on**: `query` is `Required[str]` (guaranteed at entry); every other state key is guaranteed only by graph order and is read with `.get()` plus a default, next to a comment naming the node that guarantees it. **Never blanket-`# pyright: ignore` it** — a site where neither is honest is a finding about the graph. → 0031
  - `reportMissingImports` is a **warning**, not an error (lazy optional extras); scope is `jobsmith/` only, `typeCheckingMode: basic`; pyright needs `node` on `PATH` (silently downloaded on first run if missing, which is why a fresh machine's first `make types` is slow). → 0031
- **`main` is protected**: PR required, the three checks must pass, admins included, no force-push. **Status checks are strict** — a branch must contain the current `main` before it can merge, so CI validates the *post-merge* state rather than a stale snapshot. When several PRs are in flight, each merge invalidates the rest: bring them up to date with `gh pr update-branch <n>` (or a rebase) and let CI re-run. Green checks on a stale base do not mean the merge is green. → 0000
- **`uv.lock` is committed**; regenerate it (`uv lock`) in the same commit as any dependency change — version drift has bitten this project three times. → 0000
- **Parallel sessions use git worktrees**, one per issue — separate checkouts of the same repo, so two sessions never fight over the working tree or the current branch:

  ```bash
  make worktree B=feat/1-grounding      # checkout + venv + .env, ~1s (uv cache)
  cd .claude/worktrees/feat-1-grounding
  make check                            # each worktree has its OWN .venv
  make worktree-rm B=feat/1-grounding   # after the PR is merged
  ```

  Gotchas, both verified: a venv is **path-specific** (its shebangs are absolute), so a worktree needs its own — never symlink or copy one; and `.env`, `agent.db`, `artifacts/` are gitignored, so a fresh worktree has **no API key** until it is copied (the `make worktree` target does it). `.claude/worktrees/` is gitignored, which is also where Claude Code's own `EnterWorktree` puts them.
- `make coverage`: the interactive layers (`cli/`, `chat/tools.py`) are the thin ones — a change there brings its tests with it. → 0000

Domain-leakage gate (`make leak-check`, must return nothing): `grep -ri --include="*.py" "banking\|banquier\|votre\|analyste" jobsmith/core jobsmith/jobs jobsmith/chat jobsmith/api jobsmith/app jobsmith/cli jobsmith/tui jobsmith/agents/default jobsmith/agents/base.py` — note it scans `agents/default` and `agents/base.py`, **not** `agents/banking`, which is allowed to be as domain-specific as it likes.

### The inbound port (`service.py`)

`AgentService` is **the** use-case surface of the application — sessions and jobs — and `LocalAgentService` is its in-process implementation over a composed `AgentApp`. Every entrypoint is an *adapter* over it, never a second implementation:

| adapter | backing |
|---|---|
| `cli/repl.py`, `cli/main.py` | `AgentService` (either backing) |
| `api/app.py` | `LocalAgentService` — each route is serialization + one call |
| `cli/client.py` `DaemonClient` | the same port, backed by HTTP |

- **Live progress and produced files are on the port**: `subscribe`/`unsubscribe`, `list_outputs`, `find_output`. `DaemonClient.subscribe` drops on a full queue like `InProcessEvents`; **`None` on the queue means the stream is over** and is never dropped. `tests/test_service.py` asserts identical answers from both backings. → 0048
- **The port throws exactly `BinaryDeliverable`, `ChatStreamError`, `ServiceUnavailable`** — that list is the whole contract; a 500 or an unparsable body is a defect, never translated. Front-ends catch `ServiceUnavailable` **by name**, never a broad `except`. → 0064
- **A turn is a flow**: `stream`/`stream_approval` are abstract; `send`/`approve` are `terminal_of(self.stream(...))`. Seven events (`job_started`/`job_planned` since #83/#86). **`Message.content` is the concatenation of the turn's `Token`s**, never the model's last message. Terminals: `{"type": "message", "content"}`, `{"type": "proposal", "query", "rationale", "sources"}`. → 0050, 0083, 0086
- **The chat stream never drops a token**: no queue on the path (runner generator → `StreamingResponse` → `aiter_lines`); an undecodable SSE line or a stream with no terminal raises `ChatStreamError`. `/events` is the opposite and sheds. → 0050

### CLI + daemon (`cli/`) — where jobs actually run

The point of this layer: **a job must outlive the command that launched it**. `jobsmith serve` is a long-lived process owning the JobManager; every other command is a *client*.

- `cli/client.py` — two backings for that one port: `DaemonClient` (HTTP) and `EmbeddedClient` (nothing but `LocalAgentService`). `open_client()` probes `GET /health` and falls back to embedded, printing the trade-off on stderr. → 0000
- **All diagnostics go to stderr** (banners, tool activity); stdout stays pipeable and carries the answer's tokens; both are flushed per write. → 0000
- The REPL renders the flow with no fallback branch and never prints the terminal (the tokens delivered it); a mid-turn `ChatStreamError` is said on stderr and the loop continues. → 0050
- **The job notice is on stdout**, one `job_lines` renderer for notice and proposal, ending `stop it : /cancel <id>`. → 0083
- **The plan is activity, not record**: `job_planned` goes to stderr as `… plan: a + b → c` (waves from `core.state.plan_waves`); the TUI says it on the activity line, never in the conversation. → 0086
- `run_repl` wraps each command in one `except ServiceUnavailable`; `run_command` likewise (exit 1). → 0064
- `cli/main.py` — argparse; `--llm` is exported as `$JOBSMITH_LLM` so `pick_provider` sees it from both stacks. Bare `jobsmith` == `jobsmith chat`.
- Sessions are **rebuildable by id**: the API's registry is only a cache, the conversation lives in the checkpointer under `thread_id=session_id`, so `jobsmith chat --session <id>` resumes across a daemon restart (and gets the finished-job announcement). `session_factory` therefore takes an optional `session_id`.
- Embedded mode: a *running* job stops with the process (`recover_interrupted` settles it next start); records are kept. → 0063

### Agents (`agents/`) — what an agent *is*

An agent is **a capability pack + a profile (+ an optional chat persona, + whatever it needs open)**, and nothing else — `AgentDefinition` in `agents/base.py`. The runtime, job engine, chat, CLI and API are shared by all of them, so **adding an agent touches no shared code**: define the capabilities, register the definition in `agents/__init__.py`, done. `tests/test_agents.py` pins that property (it composes a third-party agent from scratch).

```python
AgentDefinition(
    open_resources=...,   # async (stack) -> anything: pools, sessions, clients
    capabilities=...,     # (AgentContext) -> list[Capability]   ctx.llm / ctx.resources
    profile=..., chat_prompt=...,
)
```

- `open_resources` gets `build_app`'s `AsyncExitStack`; teardown is reverse order on `AgentApp.aclose()`, even after a failed startup. Several capabilities on one backend share a **pool**, never a fat client. → 0000

- `agents/default/`: `read_files`/`prior_jobs`/`documents`/`web_search` → `research` → `analysis` → `critique`, plus `slide_deck`. `analysis` and `critique` subclass `SingleStepCapability` (`_step.py`, first-match over `UPSTREAM`, degrading to the bare request); `critique` overrides `_material` to read both. → 0000
  - **`documents` is the grounding step**, over the `DocumentSource` port (`sources.py`; `LocalFiles` is keyword ranking, no key, no network). → 0000
  - **`read_files` reads a named file** (`documents` searches): port `DocumentReader`, path in `inputs["source_files"]` (`SOURCE_FILES_INPUT_KEY`), never parsed from the query; dropped when nothing was named; no model call; **a refusal is material, not silence**. → 0060
  - **`prior_jobs` reads an earlier RUN, not its file**: `inputs["from_jobs"]`, port `PriorJobSource`; no model call; refusal is material; bounded at 24 000 characters, ≤ 3 jobs; session scope enforced in `chat/tools.py`. → 0074
  - **`prior_jobs` and `read_files` are first in the registry list**: `KeywordLLM` chains it in order, and a gated step pruned from the middle severs grounding→reasoning. → 0074
  - **`web_search`** = `DocumentsCapability` over `TavilySource`: page over snippet, `$TAVILY_SEARCH_DEPTH` default `advanced`, `max_chars=8_000` per document, cut written into the text. `TavilySource`'s client is closed on the app's stack; an HTTP error raises. → 0075
  - **A capability nothing can serve stays out of the registry** — every conditional step is registered only when something backs it (`open_default_resources`); an empty registry is then answered directly. → 0000, 0038
  - **`slide_deck` is a generation, not a report format**: deck structure is asked of the model; only `pptx_deck.py` imports `python-pptx`; 16:9; refused without a job before the LLM call; non-JSON salvaged as `meta["via_fallback"]`; a failed write declares nothing; the deck is an `annex`. **Its description says what it is NOT.** → 0035, 0061
  - **The deliverable is written for its reader**, and **answers**: the prompts that produce it name the reader, rule out the state of the work, and oblige the answer first, from the material, with doubt marked where it bears; `SUBJECT_ONLY_RULE` is on every material prompt and the generator; `NO_ANSWER_INSTRUCTION` sets a high bar and a shape for a refusal. → 0058, 0073
  - **The generator is told which files the run delivers** (`delivered_files_note`: requested formats + declared annexes, or "none") and names no other; no prompt offers a file by example. When the list names the answer itself, `ANSWER_FILE_RULE` says that entry **is** the text being written — never described, saved by hand or "delivered separately". → 0077, 0126
  - Retrieved passages carry a **quotable id** (`path#chunk`); `render_context` gives the model the material, `render_report` gives the human the provenance only.
  - **`research` reads every retrieval step's material** (`GROUNDING`, not first-match) and `read_files`' refusals (`REFUSALS`), in its own prompt, bounded at 32 000 characters, and says so in `meta["grounded_on"]`. → 0081
  - **`critique` checks the subject, not the work** (≤ 8 bullets, reads analysis *and* notes) and feeds the generator. Watch for an *Open questions* section appearing. → 0082
- `agents/banking/`: the domain example, with its **own ports** next to its capabilities (`deps.py`) and its own adapters (`fakes.py`); `vision` is registered only when the LLM satisfies `VisionClient`. → 0000
- Selection: `--agent NAME` (CLI, applies to whichever process owns the engine — so pass it to `serve`), `build_app(agent=...)`, `make chat AGENT=banking`.

### The composition root (`app/`)

Wiring only, no content — everything here is domain-neutral:
- `providers.py`: `pick_provider` (one `--llm=` flag / key auto-detect for both stacks), `make_llm`/`make_chat_model`, `load_dotenv`, and the keyless fakes — `KeywordLLM`, `KeywordChatModel` (runs a job on analysis-ish keywords, synchronous by default). → 0000
- `agent.py`: `build_app(agent=..., **overrides) -> AgentApp` composes ONE agent on one `AsyncExitStack`. **`reports_dir` is resolved once, to an absolute path** (`pick_reports_dir`); `AgentContext(readable_roots=(reports_dir,))`; one `StoreJobRepository` serves both the manager and the `PriorJobSource`. → 0063
- `persistence.py`: `pick_db()` (arg > `--db=` > `$JOBSMITH_DB` > `<data dir>/jobs.db`) + `open_persistence(spec, stack)` → `(checkpointer, store)`, teardown on the caller's `AsyncExitStack`.
  - **Default: a SQLite file under `data_dir()`**, never the cwd; SQLite is a core dependency. **`memory` must be asked for by name**: tests, `evals/harness.py` and the demo pass `db="memory"`; `conftest.sandboxed_data_dir` fails the run if the suite wrote to the data dir. → 0063
  - **Every SQLite writer takes the lock up front** (`_ImmediateBegin`), so a contended batch waits rather than fails "database is locked". → 0063
  - Backends: `memory`, a SQLite path (the default, or any path you name), or a Postgres DSN (`.[postgres]`, one shared `AsyncConnectionPool` for saver + store, `autocommit/prepare_threshold=0/dict_row` as those backends require). Chat sessions share the job graph's checkpointer, so conversations persist too (namespaced by `thread_id`: `session_id` vs `job_id`).

**`build_app` is async**: real backends must open in the loop that uses them — `jobsmith serve` runs `await uvicorn.Server(config).serve()`, never `uvicorn.run()`. **SQLite gotcha**: never run a stray `PRAGMA` on the live connections (deadlock); the store needs `isolation_level=None`, the saver keeps the default. → 0000
- **Interrupted jobs**: `JobManager.recover_interrupted()` (in `build_app`) marks FAILED a RUNNING record whose owner is provably gone; a live owner's job is left alone; QUEUED stays runnable. → 0010

## Architecture

### The OO pattern

Every graph step is a class instance owning its deps and config. Node logic is **async instance methods** registered directly (`g.add_node("planner", self.planner.run)`); routers are **sync methods**; capabilities expose `.build()` returning a compiled sub-graph mounted as one parent node. `AgentBuilder` (`core/builder.py`) is the composition root and holds references to every step instance (swap one before `.build()` in tests).

### Core concepts (read these files first)

- **`core/capability.py`** — `Capability` ABC + `CapabilitySpec` (name, description, JSON-schema dicts, `requires_inputs`). Capabilities take *exactly the clients they need* in their constructors; the framework never introspects them. Terminal sub-graph nodes call `_emit_success`/`_emit_failure` so every capability reports uniformly.
- **`core/registry.py`** — `CapabilityRegistry`: single source of truth for what the agent can do. The planner prompt, executor Send targets, and builder node map all derive from it. **Frozen at `build()`** — a compiled graph's capability set is fixed; new capability ⇒ new `AgentBuilder` (compilation is milliseconds).
- **`core/state.py`** — capability results live in one `results: dict[str, CapabilityResult]` with a dict-union reducer. Fan-in safety: each capability writes only its own key; registry-unique names + no-duplicate plan steps ⇒ disjoint keys. **Determinism caveat:** consumers must iterate in *plan order*, never dict order (ContextMerger does).
- **`core/usage.py`** — an **ambient ledger** (`ContextVar` per run) adapters push into with `record_usage`; scope from the root of `checkpoint_ns`, else `unattributed`; `$JOBSMITH_PRICES`; unpriced ⇒ `cost_usd: None`; chat-layer calls are not counted. → 0002
- **`core/paths.py`** — `safe_name` (one component) and `resolve_within` (refused unless it **lands** in a declared root; resolves before comparing). → 0060
- **`core/prior_jobs.py`** — the `PriorJobSource` port, in `core/` because the composition root supplies it. → 0074
- **`core/profile.py`** — `AgentProfile` is the entire domain surface: prompt templates, user-facing messages, input/output validation rules (plain callables), `max_refine`. Core defaults are neutral English; the banking example overrides them (French messages live *only* in `agents/banking/profile.py`).

### Graph flow

```
validate_input → document_intent → router ─(direct | empty registry)→ direct_answer ┐
                                     └(plan)→ planner ─(nothing applicable)→ ───────┤
                              └→ executor_dispatch ⇄ {cap_<name> × registry}
                                ↓ (all done)                     ↓
                          merge_results → generation → validate_output
                                              ↑ refine ←┘ (≤ max_refine)  → post_process → END
                                                         └(could not answer)→ unanswered → END
errors: execution_error → escalate (some ok result) | user_error (none) → END
```

- **Router** (`core/router.py`) is a dedicated triage node — the planner never decides *whether* to plan. Routes live in `Router.routes` (`plan`, `direct` → `DirectResponder`); **fail-open** to `plan`. New route = `Router.routes` entry + node + `AgentBuilder.route_targets` entry. → 0000
  - An **empty registry** routes `"direct"` structurally, before the LLM call, fallback included. → 0038
  - **The direct route is told the files the run delivers**; asked for as a file, its reply is that document (`DIRECT_DOCUMENT_RULE`) — the request is never moved to the planner for it. → 0080
- **Document intent** (`core/document.py`), a decision node **before triage**: fills silence, never overrides (a seeded `document_formats` returns before any model call), chooses from `available_formats(registry)` and proves a choice (`confirm`) so it cannot refuse, fail-open (writes nothing = no file), writes only to state (`FormatsChosen`, folded by `JobManager._apply`). → 0090, 0108
  - **No document unless asked**: `_deliverable_wanted` is `bool(job.formats)`; silence is `FormatsChosen(None)`. **`None` (may be filled) and `[]` (never overridden) are two facts** — never `formats or []`. → 0096
  - **A file asked for in words is asked for**: "save it to a file" is `requested` whatever else the request asks; a file it is only about is not (`FILE_REQUEST_RULE`). → 0125
- **Planner** (`core/planner.py`) renders its prompt from `registry.specs()`, validates the LLM's JSON DAG: names against the registry, drops steps whose `is_applicable(state)` is false (generalizes "vision only if image" via `spec.requires_inputs`), **prunes dropped names from surviving `depends_on`**, Kahn cycle check.
  - A plan emptied by applicability routes to `direct_answer` (`_route_after_planner`); `{"steps": []}` from the model is still an error. → 0038
- **A refusal is data**: `split_declaration` reads a first-line marker only; `_route_validate_output` sends it to `unanswered`, never to refine; the job stays DONE, `terminal_kind` says so. → 0059
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

- **Only `runner.py` knows LangGraph's stream shape**; it yields `PlanReady`/`StepFinished`/`NodeErrors`/`Terminal`, for `stream()` and `resume()` alike. Graph nodes stay job-agnostic: **new persistence goes in the manager or the repository, never in a node.** → 0000
  - `PlanReady` persists a summary, so the stream says the job moved as soon as the plan exists. **Which step finished is read from the `cap_*` node name**, never from `results` (the whole channel). → 0048, 0053
- **Only `repository.py` knows the schema**: `("jobs","index")/job_id` → summary; `("jobs",job_id,"meta")` → plan/errors; `("jobs",job_id,"results")/cap_name` → per-capability result. Fine-grained state stays in the checkpointer under `thread_id == job_id`. Moving job records to SQL is another implementation of this port.
- **`jobs/prior.py`** (`RepositoryPriorJobs`) is the only place that knows how a finished run's material is reached. → 0074
- **Ownership is on the record** (`jobs/ownership.py`, `("jobs", id, "control")`): a lease written **before** RUNNING, heartbeat 2 s, TTL 30 s; **a cancel is a request the owner acts on**, never a status written over its run. Events stay in-process (#100). → 0010
- **Resume** re-enters with `None`, gated on CANCELLED/FAILED **and** non-empty `runner.pending()`; seeds usage; `_begin_resume` clears `job.error` and `job.announced`. No partial re-run of a finished DAG. → 0005
- **Vocabulary**: an **output** is what the job produces for the human (`Job.outputs`, role `main`|`alternate`|`annex`); a **result** is a capability's payload (`results`). `Job.report_path` is the main output's path. → 0000
- **Reporters** (`jobs/report.py`): `build_document` → `JobDocument` → `FileReporter` subclasses (`render`, or `serialize` for bytes); `is_binary_format`. → 0009
- **Exactly one output is `role="main"`** (the first format; the rest `alternate`, never `annex`); `compose_reporters` refuses two Reporters on one extension. `pick_report_formats()` only says *which* file when one is wanted and none was named. A failed write leaves the job DONE with `job.error`. → 0028, 0096
- **An annex is a file a step declared** via `ArtifactStore` + `artifact_meta(...)`, collected in plan order at **every** terminal, assigned never appended; a declared-but-absent file goes in `job.error`. → 0035, 0041
- **Name, title and formats are facts on the job**, refused in `create_job`; a title is cut on a word (`document_title`). → 0055, 0054
- **The report is title + answer + one `job_reference` line**; the record holds the rest (`with_provenance` to recite it). → 0085
- **HTML takes no dependency**: escape everything first, add only our own tags, no links, nested lists clamped to the open-list stack. **The PDF is that page printed**; `.[pdf]` needs pango/cairo, **offered when installed** (`find_spec`) and **loaded on first need** (at startup only when the deployment default is PDF). → 0009, 0076, 0034, 0108
- **Capabilities present their own results**: `Capability.render_report(result)` (twin of `render_context`, which targets the model) with `default_result_markdown` as the base implementation — prose stays prose, list[str] becomes bullets, only structured values fall back to JSON. Never grow that default to learn payload shapes: override `render_report` in the capability instead.
- `Job.usage` refreshes on every persist; per-step usage is in `meta["usage"]`, failures included. → 0002
- `Job` also carries `session_id` (chat session that launched it) and an `announced` flag (`list_finished_unannounced`/`mark_announced` drive chat notifications).

### Chat layer (`chat/`)

`ChatSession(manager, model, *, session_id, system_prompt, checkpointer).build()` → a `langchain.agents.create_agent` (LangChain model, NOT the framework's LLMClient — deliberate two-stack split: LangChain handles per-provider tool formats; the job engine stays dependency-light).

- **`chat/runner.py` alone knows what `astream` emits**: `Token`, `ToolStarted`/`ToolFinished` (real tool name; wording is the front-end's), `JobStarted`, `JobPlanned`, then exactly one terminal, `Message` or `Proposal`. → 0050, 0083, 0086
- **Tools** (`chat/tools.py`) wrap JobManager use-cases, scoped to the session's own jobs. `launch_job` **runs the task**: `create_job(session_id=...)` + `start_job`, then it *waits* on that task for `pick_sync_timeout()`.
- **Synchronous by default, promoted by the clock**: `start_job` then `asyncio.wait` (never `wait_for`) for `$JOBSMITH_SYNC_TIMEOUT` (20 s; `0` = never); no classification; a job finished in the turn is `mark_announced`. → 0083
- **The plan is announced while the turn waits**, from `manager.subscribe()` (subscribed before `start_job`, drained with `None` when the run ends, released in `finally`); never after promotion; a one-step plan is not announced. → 0086
- **The answer is written verbatim into the turn**, never returned through the model; a promoted answer uses the same channel up to `$JOBSMITH_INLINE_ANSWER_MAX` (2 000), or at any length when no file was written. → 0083, 0085
- **The approval card is a notice** (`job_started`: query, sources, `from_jobs` as short id + start of query, name/title/formats, job id); the gate survives behind `$JOBSMITH_APPROVE_JOBS`. → 0083, 0104
- **The engine never sees the thread**: a self-contained `query`, plus `recent_conversation()` as `inputs[CONVERSATION_INPUT_KEY]`. `source_files` and `from_jobs` (resolved against **this session's** jobs) are `launch_job` arguments. → 0004, 0060, 0074
- **Notifications** (`JobNotificationMiddleware.awrap_model_call`) are transient `SystemMessage`s in the model *request*, never in state. → 0006
  - Completion (every terminal, `ANNOUNCEABLE`) and progress (only when `progress_signature()` moved) notices go **directly after the system prompt** (`_inject`), where Anthropic hoists them. → 0006
- **`conftest.ScriptedChatModel` implements `_astream`** with several chunks; the message list is also run through the real provider formatters, which only run with the chat extras (CI). → 0006

### HTTP API (`api/`)

`create_api(service) -> FastAPI` (extra `.[api]`; served by `jobsmith serve`, whichever agent is composed). A pure adapter: every route is serialization plus one `AgentService` call — if job or chat logic reappears here, it belongs in `service.py`.

- **Chat**: `POST /sessions`, `/sessions/{id}/messages`, `/sessions/{id}/approval`, each with a `/stream` SSE twin (no queue behind it). → 0050
- **Jobs**: `GET /jobs[?session_id&status]`, `GET /jobs/{id}` (plan/step_finished_at/results — the UI's DAG data), `POST /jobs` (direct launch), `POST /jobs/{id}/cancel`, `POST /jobs/{id}/resume` (409 when the job has nothing left to run — `AgentService.resume_job` answers refusals as `{"status", "error"}` on both backings, so `DaemonClient` maps the code back into that dict).
- **Outputs**: `/jobs/{id}/outputs[/{name}]`; `/report` serves text (type from `REPORT_MEDIA_TYPES`), **415** for a binary deliverable. → 0034
- **Live**: `GET /events`, in-process, drops on full; untestable through `ASGITransport` (hangs), so `DaemonClient.subscribe` is tested under uvicorn. → 0048

### Terminal UI (`tui/`) — the first UI, and why it is a TUI

`jobsmith ui` (`.[tui]`) renders the same port and events beside `jobsmith chat`; `subscribe()` drives every repaint, no poll, no timer. → 0048

- **Stylesheet: standard theme roles only** (an invented one fails at parse time); CSS says `$text-muted`, markup `$foreground-muted`; an undefined markup variable is silently dropped. → 0048
- An event says *something changed*: re-read, join a refresh in flight, remember a skipped one. A turn in flight refuses input and keeps the text. F8 arms before cancelling; bindings are function keys. Model text is `escape`d. → 0048
- **A snapshot never proves a fact** — assert what a pane says by name; `pytest-textual-snapshot` is pinned exactly. → 0048

### Adding a capability

1. Subclass `Capability`; define `spec` (unique snake_case name), constructor taking needed clients, async node methods, `build()` via `self.state_graph(...)`, terminal nodes returning `self._emit_success(...)`/`self._emit_failure(...)`, and `render_context()` if its result should feed generation.
2. Register it in the composition root before `AgentBuilder(...).build()`. Nothing else: planner prompt, dispatch, merging all pick it up from the registry.
3. To **read a named file**: `requires_inputs=(SOURCE_FILES_INPUT_KEY,)`, a port in the constructor, never `Path.read_text`; let the port refuse. → 0060
4. To read an **earlier job**: `requires_inputs=(FROM_JOBS_INPUT_KEY,)`, `ctx.prior_jobs`, never reach into `jobs/`; bound what travels and write every cut into the text. → 0074
5. To produce a **file**: `ctx.artifacts.write(state.get("job_id", ""), name, data)` and `meta=artifact_meta(ArtifactRef(path, title=...))` on `_emit_success` (or `_emit_failure`); never build a path. → 0035

## Testing conventions

- **Tests are organised by purpose, never by issue**: a file per component or rule (`test_deliverable.py`, not `test_<issue>.py`); a change extends the file of the rule it touches. A docstring states the property in a sentence, `→ NNNN` for the why — history lives in the record. Cases that differ only in data are one parametrized test. → 0110
- **A prompt is asserted through the named rule it must carry** (`SUBJECT_ONLY_RULE`, `CAVEATS_RULE`, `BRIEF_RULE`, `UNREADABLE_RULE`: `RULE in prompt`), never by its wording — wording is the evals' to judge. A rule worth a test is a constant in the product. → 0110
- **Shared builders live in `tests/support.py`** (`OneStep` — a one-node capability: write `work` only —, `SlowEcho`, `ChartCapability`, `make_manager`, `planning`, `make_session`, `chat_turn`, `service_over`, `make_app`, `wait_done`, `until` — wait on state, never a fixed sleep —, `StubPdf`); **a test file never imports from another test file**. → 0110
- `tests/conftest.py` — `FakeLLM` scripts responses by **substring of the system prompt** (`{"planner": ..., "ONLY the provided": ...}`); `plan_json()` builds planner responses. Fixtures: `checkpointer` (MemorySaver), `store` (InMemoryStore).
- `tests/test_banking_example.py` is the behavior-parity suite for the pre-refactor agent (French rejection messages, citation rule, vision-dropped-without-image).
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
