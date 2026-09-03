# jobsmith

A conversational agent that answers simple messages directly and turns complex
ones into **background jobs** — with your approval — then keeps chatting while
they run and hands you a written report when they finish.

Underneath, it is a domain-agnostic framework: a planner emits a DAG of
pluggable **capabilities**, an executor fans them out in parallel, and every run
is a persistent, trackable, cancellable **Job**.

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

## Two ways to drive it

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

### Steps

| step | depends on | status | finished at |
|---|---|---|---|
| research | — | ok | … |
| analysis | research | ok | … |
| critique | analysis | ok | … |

```mermaid
flowchart LR
  research --> analysis
  analysis --> critique
```
````

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

**External dependencies** — a vector store, an HTTP API, an MCP server — are
declared as Protocols next to the capability that consumes them, and opened by
the agent's `open_resources(stack)` on the app's `AsyncExitStack`, so they are
closed in reverse order when the app closes. When several capabilities need the
same backend differently, share the *pool* and give each one its own adapter for
the port it declared: capabilities run in parallel waves, so a raw shared
connection is a bug waiting to happen.

That is all — the planner prompt, the dispatch map and the merging step all
derive from the registry.

> **Invariant:** `build()` must use `self.state_graph(...)`, which pins the
> sub-graph's `output_schema`. Without it, two capabilities finishing in the
> same superstep collide with `InvalidUpdateError`.

### Layout

```
jobsmith/
  core/         the engine: router, planner, executor, generation, registry
  jobs/         the job use cases + their ports (repository, runner, events, reporter)
  chat/         conversational layer (LangChain create_agent) + job tools
  api/          FastAPI: sessions, jobs, outputs, SSE
  cli/          daemon, clients, REPL, argparse entrypoint
  agents/       ★ what each agent IS — a capability pack + a profile
    default/      research → analysis → critique (LLM-only)
    banking/      a domain agent: its own capabilities, ports and adapters
  app/          composition: providers, persistence, build_app(agent=...)
```

Everything but `agents/` is shared. **Adding an agent touches no shared code**:
write its capabilities, register an `AgentDefinition`, and the planner, job
engine, chat, CLI and API all serve it — `jobsmith --agent <name> chat`.

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
| `--db memory\|<file.db>\|<postgres DSN>` | persistence (default: `$JOBSMITH_DB`, else memory) |
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
make fix           # ruff --fix
```

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
- **No `resume_job()` yet.** A job interrupted by a dead process is marked
  failed on the next start — its checkpoint is retained, so resuming is a
  feature away, not a redesign.
- **Markdown is the only Reporter.** HTML/PDF/PPTX are additional Reporters over
  the same `JobDocument`.
- **No web UI.** Everything is terminal or HTTP for now; the API already serves
  what a chat / jobs-DAG / artifacts interface would need.
