# jobsmith

A conversational agent that answers simple messages directly and turns complex
ones into **background jobs** — with your approval — then keeps chatting while
they run and hands you a written report when they finish.

Underneath, it is a domain-agnostic framework: a planner emits a DAG of
pluggable **capabilities**, an executor fans them out in parallel, and every run
is a persistent, trackable, cancellable **Job**. One runtime serves any number
of agents — a new one supplies only its capabilities and its voice.

```
you> what can you do?
     → answered inline, no job

you> compare hexagonal and layered architectures for an LLM agent
     → the agent proposes a background job, explains its approach, waits for y/N
     → runs research → analysis → critique in the background
     → later, in the same conversation: a synthesis + the path to the report
```

---

## Quickstart

No API key needed to try it — a deterministic fake provider ships in the box.

```bash
make install                 # venv + dev/API deps
make chat LLM=fake           # or: .venv/bin/python -m jobsmith --llm fake chat
```

With a real model, drop a key in `.env` (see `.env.example`) and the provider is
auto-detected:

```bash
cp .env.example .env         # ANTHROPIC_API_KEY or OPENAI_API_KEY
make chat
```

A session looks like this:

```
$ jobsmith chat
[jobs llm: Claude via AnthropicLLMClient — claude-opus-5]
[chat llm: ChatAnthropic — claude-opus-5]
[persistence: sqlite — agent.db]
[session 1635d9410abb494a8d2c7c6c31bf558b]

agent> compare hexagonal and layered architectures for an LLM agent

  the agent proposes a background job:
    task     : compare hexagonal and layered architectures for an LLM agent
    approach : Three steps — gather the trade-offs of each style, analyse them
               against an agent's constraints, then critique the conclusion.
  launch it? [y/N] y

  Started in the background (17abcd66). Ask me anything meanwhile.

agent> /jobs
  job 17abcd66  [running]  'compare hexagonal and layered architectures for a'

agent> and which one does LangGraph itself use?
  ...

  [the job finishes — the next turn carries the synthesis and the report path]
```

---

## Driving it

### Chat (the default)

`jobsmith chat` is a conversation. The agent decides whether to answer or to
propose a job; you approve. In-REPL commands:

| command | |
|---|---|
| `/jobs` | list this session's jobs |
| `/job <id>` | plan, steps, results, answer |
| `/report <id>` | print the finished deliverable |
| `/bg <text>` | skip the chat, run it as a job now |
| `/image <key>` | attach an image input to the next `/bg` job |
| `/cancel <id>` | cancel a running job |
| `/resume <id>` | restart a stopped job from its checkpoint |
| `/quit` | leave (jobs keep running if a daemon owns them) |

### CLI

```bash
jobsmith serve [--port 8000]     # the daemon: it owns the job engine
jobsmith run "<task>" [--wait]   # launch a job directly, no chat
jobsmith jobs [--status running] # list
jobsmith job <id-prefix>         # plan, steps, results, answer
jobsmith report <id-prefix>      # print the markdown deliverable
jobsmith outputs <id-prefix>     # list the files the job produced
jobsmith cancel <id-prefix>
jobsmith resume <id-prefix>       # restart a stopped job from its checkpoint
```

Ids can be given by prefix. All diagnostics go to **stderr**, so stdout stays
pipeable: `jobsmith jobs | cut -d' ' -f1`.

### Daemon or embedded — the one thing worth knowing

**A job must outlive the command that launched it.** `jobsmith serve` is a
long-lived process owning the job engine; every other command is a *client*
that talks to it over HTTP. If no daemon is running, commands fall back to
running the agent **embedded** in their own process — convenient, but jobs then
die with the command (`run` compensates by waiting, and says so on stderr).

```bash
jobsmith serve &                 # jobs now survive everything else
jobsmith run "long analysis"     # returns immediately
jobsmith jobs                    # another process sees it
```

Conversations are rebuildable by id: `jobsmith chat --session <id>` resumes
across a daemon restart, and picks up the announcement of any job that finished
while you were away.

### Grounding jobs in real material

Point the agent at a directory and its jobs start from **your** files rather
than from the model's recollection:

```bash
jobsmith --docs ./docs chat
jobsmith --docs . run "how does the job engine persist state?" --wait
```

The `documents` capability then joins the plan, retrieves the relevant
passages and hands them to the rest of the DAG with a quotable id each
(`path#chunk`), so the report can point at where something came from. Ranking
is keyword overlap, not semantics — honest and dependency-free.

For material the model cannot have — recent events, third-party facts, current
versions — set a Tavily key and a `web_search` step joins the same plan, citing
URLs a reader can open:

```bash
uv pip install -e ".[web]"
export TAVILY_API_KEY=tvly-...
jobsmith run "compare the current LangGraph and LlamaIndex agent APIs" --wait
```

Both are the same port with different adapters, so the capability consuming
them is identical — and each stays out of the registry entirely when nothing
backs it, rather than being planned and failing.

### Which agent

An **agent** is a pack of capabilities plus a profile — what the thing can do
and how it speaks. Two ship: `default` (research → analysis → critique, needs
nothing but a key) and `banking` (a domain example: document search, slide
vision, French). Everything else — job engine, chat, CLI, API, persistence — is
shared, so they run on exactly the same commands:

```bash
jobsmith --agent banking chat    # or: make chat AGENT=banking
jobsmith --agent banking serve
```

`--agent` applies to whichever process owns the engine, so pass it to `serve`
when a daemon is running. Writing your own is covered under
[Capabilities](#capabilities).

---

## What a job produces

The deliverable is a markdown file — **the answer first**, provenance after:

````markdown
# compare hexagonal and layered architectures for an LLM agent

<the synthesis>

---

## About this job

- **Request**: …
- **Job**: `a803205bea59412bae2e376a8555ee62`
- **Started** / **Finished**: …
- **Usage**: 7 LLM calls — 21,430 in / 5,120 out tokens — ~$0.2352 est. — claude-opus-5

### Steps

| step | depends on | status | usage | finished at |
|---|---|---|---|---|
| research | — | ok | 12.4k tok · ~$0.1120 | … |
| analysis | research | ok | 8.1k tok · ~$0.0790 | … |
| critique | analysis | ok | 4.9k tok · ~$0.0442 | … |

```mermaid
flowchart LR
  research --> analysis
  analysis --> critique
```
````

**What it cost is part of the deliverable.** Every LLM call is booked to the
step that made it, so the report (and `jobsmith job <id>`, and the `/events`
stream, live while it runs) answers both *what did this cost* and *which step
spent it*. Prices are a dated snapshot — override them with `$JOBSMITH_PRICES`
(inline JSON or a path: `{"gpt-5.1": {"input": 1.25, "output": 10.0}}`); a model
with no price is reported in tokens rather than in an invented dollar figure.

Per-step material is deliberately **not** inlined — it lives in the store, and
`jobsmith job <id>` or `GET /jobs/{id}` serves it. For a self-contained archive,
`MarkdownReport(with_annexes=True)` folds it back in as collapsible sections.

A job carries a **list** of outputs (`role: main | annex`, a `format`, the
capability that produced it), so annexes and other formats are a matter of
adding Reporters, not of reshaping the model.

---

## Architecture

### The graph

```
validate_input → router ─(direct)→ direct_answer ────────────────┐
                   └(plan)→ planner → executor_dispatch ⇄ {cap_<name> × registry}
                                ↓ (all done)                     ↓
                          merge_results → generation → validate_output
                                              ↑ refine ←┘ (≤ max_refine)  → post_process → END
errors: execution_error → escalate (some result ok) | user_error (none) → END
```

- **Router** — a dedicated triage node. The planner never decides *whether* to
  plan; the router picks `plan` or `direct`, and **fails open to `plan`** on any
  LLM or parse error. A new route is one entry in `Router.routes` plus a node.
- **Planner** — renders its prompt from the registry, validates the LLM's JSON
  DAG (names, applicability, dangling dependencies, Kahn cycle check).
- **Executor** — computes the ready capabilities of each wave and returns
  `Send`s; capability nodes edge back to it. Any DAG, no baked-in schedule.
- **Two error channels** — planner/generation failures hard-stop; capability
  failures land in `results` with `ok: False` and the run degrades gracefully.

### Object-oriented nodes

Every graph step is a class instance owning its dependencies and config; node
logic is registered as bound methods (`g.add_node("planner", self.planner.run)`).
`AgentBuilder` is the composition root and holds every step instance, so a test
can swap one out before `.build()`.

### Capabilities

A capability is a self-describing agentic sub-graph, mounted as a single parent
node. It declares a `CapabilitySpec` (name, description, JSON schemas, required
inputs), takes exactly the clients it needs in its constructor, and presents its
own results twice: `render_context()` for the model, `render_report()` for the
human reading the deliverable. The framework never introspects a payload.

Adding one:

1. Subclass `Capability`, define `spec`, write async node methods, and `build()`
   the sub-graph with `self.state_graph(...)`.
2. Return it from an agent's capability pack in `agents/`.

That is all — the planner prompt, the dispatch map and the merging step all
derive from the registry.

**External dependencies** — a vector store, an HTTP API, an MCP server — are
declared as Protocols next to the capability that consumes them, and opened by
the agent's `open_resources(stack)` on the app's `AsyncExitStack`, so they are
closed in reverse order when the app closes. When several capabilities need the
same backend differently, share the *pool* and give each one its own adapter for
the port it declared: capabilities run in parallel waves, so a raw shared
connection is a bug waiting to happen.

> **Invariant:** `build()` must use `self.state_graph(...)`, which pins the
> sub-graph's `output_schema`. Without it, two capabilities finishing in the
> same superstep collide with `InvalidUpdateError`.

### Layout

```
jobsmith/
  core/         the engine: router, planner, executor, generation, registry
  jobs/         the job use cases + their ports (repository, runner, events, reporter)
  chat/         conversational layer (LangChain create_agent) + job tools
  service.py    ★ the inbound port: what any front-end can ask of a running app
  api/          adapter — FastAPI: sessions, jobs, outputs, SSE
  cli/          adapter — daemon, clients, REPL, argparse entrypoint
  agents/       ★ what each agent IS — a capability pack + a profile
    default/      research → analysis → critique (LLM-only)
    banking/      a domain agent: its own capabilities, ports and adapters
  app/          composition: providers, persistence, build_app(agent=...)
evals/          the golden set + the property checks that score a prompt change
```

Two boundaries carry the design. **`agents/`** is the only place a domain
lives: adding one means writing its capabilities and registering an
`AgentDefinition` — the planner, job engine, chat, CLI and API all serve it
with no shared code touched. **`service.py`** is the only place the use cases
live: the HTTP API and the CLI are adapters over it, and the CLI cannot tell
whether the work runs in this process or in a daemon — which is what makes a
UI or a bot one more adapter rather than a rewrite.

Two LLM stacks, deliberately: the chat layer uses **LangChain** models (they
handle per-provider tool formats), the job engine uses a dependency-light
`LLMClient` protocol (`jobsmith/clients.py`).

---

## HTTP API

`jobsmith serve` exposes the daemon (`.[api]`):

| | |
|---|---|
| `POST /sessions` · `POST /sessions/{id}/messages` | chat; a reply is `{"type": "message"}` or `{"type": "proposal"}` |
| `POST /sessions/{id}/approval` | answer a proposal — `{"approved": bool}` |
| `GET /jobs` · `GET /jobs/{id}` | listing and full detail (plan, timings, results) |
| `POST /jobs` · `POST /jobs/{id}/cancel` | direct launch, cancellation |
| `GET /jobs/{id}/outputs[/{name}]` · `/report` | the deliverables |
| `GET /events` | SSE stream of job progress |

---

## Configuration

| | |
|---|---|
| `--llm anthropic\|openai\|fake` | provider for **both** stacks (default: auto-detected from keys) |
| `--agent NAME` | which agent to run — `default` or `banking` (applies to whichever process owns the engine, so pass it to `serve`) |
| `--docs DIR` | ground jobs in the files under `DIR` (default: `$JOBSMITH_DOCS`); without it the agent runs on the model's own knowledge |
| `TAVILY_API_KEY` | enables the `web_search` step (extra `.[web]`); absent, the capability is not registered |
| `--db memory\|<file.db>\|<postgres DSN>` | persistence (default: `$JOBSMITH_DB`, else memory) |
| `$JOBSMITH_PRICES` | per-model prices for the cost estimate, as inline JSON or a path to a JSON file (USD per million tokens) |
| `--url` / `--local` | point at another daemon / never use one |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | key auto-detection; Anthropic wins if both are set |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | model override; the base URL points at Ollama, vLLM or a gateway |

Persistence is opt-in and backs both jobs and conversations:

```bash
uv pip install -e ".[sqlite]"    && jobsmith --db agent.db chat
uv pip install -e ".[postgres]"  && jobsmith --db postgresql://user:pass@localhost/agent chat
```

Without a backend everything is in-memory: only the `.md` reports survive.

---

## Development

```bash
make help          # every target
make check         # lint + domain-leakage gate + tests
make test T=router # one keyword's worth
make coverage      # per-module coverage report
make fix           # ruff --fix
```

### Judging a prompt change

The router, the planner and the generator are prompts, and a prompt change used
to be judged by eye on one example. `evals/` turns that into a number.

```bash
make eval                     # deterministic tier — fakes, no API key, runs in CI
make eval-llm                 # the same golden set against a real provider (opt-in)
make eval ARGS='--repeat 3'   # sample the same cases repeatedly to see the variance
python -m evals --list        # what the golden set contains
```

It scores **properties, never expected text**: the plan only names registered
capabilities, its DAG is acyclic with satisfiable dependencies, an obviously
simple message is triaged `direct` and a compound one `plan`, the run reaches
the terminal it should, every planned step ran and reported success, and the
deliverable carries a title, the answer and its provenance. Wording may vary
freely; structure may not.

Two tiers, because only one of them can be trusted to gate anything:

| | provider | variance | gates CI |
|---|---|---|---|
| `structural` | the deterministic fakes | none | **yes** — `tests/test_evals.py` requires 100% |
| `llm` | a real model | real | never |

Every run is written to `evals/results/` (gitignored) tagged with agent,
provider, tier and git revision, and the next comparable run prints the delta
per check — that is how you tell whether an edit to a prompt helped.

**What it is not.** Eleven hand-written cases sampled once against a stochastic
model is a smoke signal, not a benchmark. A few points of movement in the LLM
tier is noise — use `--repeat` before believing a delta — and the golden set
only covers failure modes somebody thought of. The structural tier is the part
that is genuinely reliable, and it only proves the machinery still holds
together, not that the answers got better.

CI runs `make check` on every push and pull request, across Python 3.11 and
3.12, and verifies that `uv.lock` still matches `pyproject.toml`.

Contributions use one short-lived branch per issue (`feat/12-thing`,
`fix/13-thing`) with a PR onto `main` — there is no `develop` branch, and
releases will be tags. Dependency changes must include a regenerated
`uv.lock` in the same commit.

`make leak-check` is a gate, not a formality: the shared code and the default
agent must contain no domain-specific vocabulary — `agents/banking/` is exempt,
and is meant to be as domain-specific as it likes. It proves a domain can be
carried (French user messages, a citation rule, a vision capability dropped when
no image is supplied) without any of it leaking upward. `make demo-banking` runs
it on fakes; `make chat AGENT=banking` opens it in the normal REPL.

---

## Status and known limits

Working end to end: chat with HITL job launch, DAG planning and parallel
execution, persistence (memory/SQLite/Postgres), the daemon/client split, the
HTTP API with SSE, markdown deliverables.

Honest v1 boundaries:

- **Cancellation and SSE are in-process.** A client can cancel a job the daemon
  runs; cross-process preemption writes a best-effort tombstone.
- **Resume restarts, it does not re-plan.** `jobsmith resume <id>` re-enters a
  cancelled or interrupted job's checkpoint and runs only the steps that never
  finished — the ones already paid for are kept as they are. A job that
  finished, or that failed at its last step, has nothing to re-enter and is
  refused: pushing a *finished* job further (redo one step, extend the
  analysis) is a separate feature.
- **Cost accounting covers jobs, not conversations.** The job engine books
  every LLM call; the chat layer talks to LangChain models on the other side of
  the two-stack split and is not counted yet. Dollar figures are estimates from
  a local price table, never a bill. A resumed job reports the *total* it cost
  across attempts, not just the resumed portion — the interrupted attempt's
  tokens were spent all the same.
- **Markdown is the only Reporter.** HTML/PDF/PPTX are additional Reporters over
  the same `JobDocument`.
- **No web UI.** Everything is terminal or HTTP for now; the API already serves
  what a chat / jobs-DAG / artifacts interface would need.
