# jobsmith

A conversational agent that answers simple messages directly and **runs real
tasks** for everything else — planning them into a DAG of capabilities,
executing it, and answering — with a document when you ask for one. A task
runs in the conversation and answers there; one that turns out to be slow moves to the background on its
own, and comes back in the same conversation when it is done.

Underneath, it is a domain-agnostic framework: a planner emits a DAG of
pluggable **capabilities**, an executor fans them out in parallel, and every run
is a persistent, trackable, cancellable **Job**. One runtime serves any number
of agents — a new one supplies only its capabilities and its voice.

```
you> what can you do?
     → answered inline, no job (the chat model, talking on its own)

you> summarise the two files I gave you
     → says what it is about to run, what it will read, what it will write
     → runs it now: read_files → analysis
     → answers in this turn, word for word — and writes no file, since none was asked for

you> compare hexagonal and layered architectures for an LLM agent, as a report
     → same start — but research → analysis → critique takes minutes
     → after 20 seconds it says so and carries on in the background
     → later, in the same conversation: the answer + the path to the report
```

Nothing predicts which of the two a task will be. It starts, and the clock
decides — a prediction would be wrong in both directions, because what a run
costs is set by the plan, and the plan does not exist yet when the guess would
have to be made.

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
[persistence: sqlite — jobs kept in /home/you/.local/share/jobsmith/jobs.db  (default; --db=memory keeps nothing)]
[session 1635d9410abb494a8d2c7c6c31bf558b]

agent> compare hexagonal and layered architectures for an LLM agent, as a report

  running this as job 17abcd66:
      task     : compare hexagonal and layered architectures for an LLM agent
      approach : Three steps — gather the trade-offs of each style, analyse them
                 against an agent's constraints, then critique the conclusion.
      writes   : architecture_comparison.md
      stop it  : /cancel 17abcd66

  … running the task
  It is taking longer than 20s, so it is now running in the background —
  I will report back here when it lands.

agent> /jobs
  job 17abcd66  [running]  'compare hexagonal and layered architectures for a'

agent> and which one does LangGraph itself use?
  ...

  [the job finishes — the next turn carries the synthesis and the report path]
```

Three things are on that notice on purpose, and they used to be on a y/N card:
the **query as the engine will see it** (the job never sees the conversation,
so reading it is what catches a "that" whose referent has gone), the **files it
may open**, and **what it will write**. What changed is that they are stated
rather than asked — and `stop it` is the line that replaces the gate:
cancelling is now the undo.

### Where your jobs live

**An unconfigured jobsmith keeps its jobs.** Jobs, their per-step results and
your conversations go to a SQLite file in your user data directory, and the
documents they write go next to it — so `jobsmith jobs`, `job <id>`, `resume`
and `chat --session <id>` find them from any later process, whichever directory
you run it from:

| platform | data directory |
|---|---|
| Linux & co | `$XDG_DATA_HOME/jobsmith`, else `~/.local/share/jobsmith` |
| macOS | `~/Library/Application Support/jobsmith` (or `$XDG_DATA_HOME/jobsmith` if you set it) |
| Windows | `%LOCALAPPDATA%\jobsmith` (or `$XDG_DATA_HOME\jobsmith` if you set it) |

Inside it: `jobs.db` (the default `--db`) and `reports/` (the default reports
directory). The first run creates both.

This used to be the other way round — the default was in-memory, and a job
disappeared with the process that ran it, leaving its report orphaned in an
`artifacts/` folder under whatever directory you had run it from. To get that
behaviour back, ask for it: `--db=memory` (or `JOBSMITH_DB=memory`); to keep
the reports where you were, `JOBSMITH_REPORTS_DIR=artifacts` (a relative value
is resolved once, at startup, so every path a job records is absolute).
SQLite is a core dependency now, not the `.[sqlite]` extra — that extra still
exists, empty, so old install commands keep working.

---

## Driving it

### Chat (the default)

`jobsmith chat` is a conversation. The agent decides whether to answer from its
own knowledge or to run a task on the engine; it does not ask permission, it
says what it is doing. The answer is printed **as it is written** — and when a
task answers, its answer is printed *verbatim*, never a summary of it — while
what the agent is doing meanwhile (`… running the task`) shows on stderr, so
stdout stays the conversation and nothing else. In-REPL commands:

| command | |
|---|---|
| `/jobs` | list this session's jobs |
| `/job <id>` | plan, steps, results, answer |
| `/report <id>` | print the finished deliverable |
| `/bg <text>` | run it as a background job and do not wait for it |
| `/image <key>` | attach an image input to the next `/bg` job |
| `/cancel <id>` | cancel a running job |
| `/resume <id>` | restart a stopped job from its checkpoint |
| `/quit` | leave (jobs keep running if a daemon owns them) |

### The terminal UI

`jobsmith ui` (extra `.[tui]`) is the same conversation with the jobs on
screen. It exists because the REPL cannot show them: it blocks on `input()`,
so nothing repaints until you type and a running job is invisible until then.
Textual owns the event loop, so the tab bar carries a live count, `F3` opens
the job list, and the detail pane draws the plan, the per-step cost and the
files produced.

It follows a job **as it runs**, with no poll and nothing to press: the job
event stream drives the repaint, so the plan appears when the planner answers
and each step lands on screen when it lands. If that stream ends — a daemon
shutting down — the screen says so rather than quietly going still. The files
pane names each deliverable and annex and says whose disk it is on: against a
daemon those paths are the daemon's, and `GET /jobs/{id}/outputs/{name}` is
what fetches the bytes.

```bash
uv pip install -e ".[tui]"
jobsmith ui                      # F2 chat · F3 jobs · F5 refresh · F8 F8 cancel
jobsmith ui --theme tide-dark    # or $JOBSMITH_THEME; ctrl+p switches live
```

Cancelling asks twice, and only from the jobs pane where the row is on
screen — which is also the undo for a task running in the turn. While a turn is
being written the prompt refuses new input rather than cutting it; a task
running in the turn is exactly when that matters, and it is why this UI shows
the plan filling in while you wait instead of a still screen.

It sits **beside** `jobsmith chat`, never instead of it: a TUI takes the whole
screen and cannot be piped, and every other command here keeps stdout
pipeable. Four themes ship (`ember-dark`, `ember-light`, `tide-dark`,
`tide-light`) and join Textual's 21 in the same `ctrl+p` picker. Without the
extra the command prints what to install and exits.

### CLI

```bash
jobsmith serve [--port 8000]     # the daemon: it owns the job engine
jobsmith ui                      # the terminal UI (extra .[tui])
jobsmith run "<task>" [--wait]   # launch a job directly, no chat
jobsmith jobs [--status running] # list
jobsmith job <id-prefix>         # plan, steps, results, answer
jobsmith report <id-prefix>      # print the deliverable (text formats)
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
running the agent **embedded** in their own process — convenient, but a job
still running then stops with the command (`run` compensates by waiting, and
says so on stderr). Its record is kept either way: the next process marks it
interrupted, and `jobsmith resume <id>` carries on from its checkpoint.

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

What it grounds on is the **page**, not the search engine's snippet of it: the
adapter asks Tavily for the extracted text, searches at `advanced` depth and
falls back to the excerpt only for a page that could not be fetched. Each
document is capped at 8 000 characters, and a page that was cut says so in its
own text. Set `$TAVILY_SEARCH_DEPTH=basic` to search cheaper and shallower.

Both are the same port with different adapters, so the capability consuming
them is identical — and each stays out of the registry entirely when nothing
backs it, rather than being planned and failing. An agent left with no
capabilities at all still answers: the router sees an empty registry and
replies directly instead of planning.

What is retrieved is read by the **run**, not only by whoever writes the
report: with a retrieval step before it in the plan, `research` writes its
notes from those passages and quotes their ids, and it falls back to the
model's own knowledge only when the plan retrieved nothing. Which of the two
it did is on the job record (`grounded_on`) and in the report, because a
figure read off a source and the same figure recalled are not the same
claim. A file that was named and could not be opened travels with the
passages, as a gap the notes have to declare rather than fill in.

### Pointing a request at a file

`documents` searches; **`read_files` reads**. When a request names a document —
a path you give — the file itself travels with the job as an input, and the
step opens it:

```
you : make a one-pager out of the report from this morning
      (the notice says which files the run will read, before it reads them)
      task     : condense /home/you/.local/share/jobsmith/reports/8aea26ec.md into a one-pager
      reads    : /home/you/.local/share/jobsmith/reports/8aea26ec.md
      stop it  : /cancel 4f21b0aa
```

**What a job may open is decided, not inherited**: a named path is resolved (symlinks followed, `..` collapsed) and
then has to land inside the directory jobs write their own files into, or the
`--docs` directory when there is one. Anything else is refused with a message
saying so, whether it was spelled `../../etc/passwd`, hidden behind a symlink,
or written as an absolute path to somewhere else. Nothing widens that set at
runtime, and the files are named in the notice before the run opens them —
you are the one handing them over, so you are the one who gets to see the
list.

### Building on an earlier job

*"Make a one-pager out of that"* points at something else entirely: a **job**,
not a file. jobsmith references it instead of re-reading the document it
wrote — the run's own records are in the store, so the follow-up gets the
answer **and** the material each step gathered, where the report carries only
the prose:

```
you : make a one-pager out of that comparison
      task     : condense the chair comparison into a one-page brief
      stop it  : /cancel 4f21b0aa
```

A report is a deliverable, not a trace: it never carried the research notes or
the retrieved pages, it may be a PDF nobody can read back, and since a run only
writes a document when the request asked for one, it may not exist at all.
Only jobs of the **same conversation** can be referenced, and a reference that
matches nothing is refused before the run starts rather than quietly ignored.

### Naming what comes out

The person who knows what a document is for is the one asking for it, so the
request decides what it is **called**, what it is **titled**, and which
**formats** are written — and the notice shows all three before anything is
written:

```
you : compare the chairs for a home office, one page. Call it
      rapport_chaises_gabarit, markdown and PDF.
      (the notice says what the run will leave behind)
      task     : compare ergonomic chairs for a home-office workstation …
      titled   : Comparatif des chaises ergonomiques
      writes   : rapport_chaises_gabarit.md, rapport_chaises_gabarit.pdf
      stop it  : /cancel 9c04e17b
```

Say nothing and nothing is silently decided for you either: say nothing about
a file and there is none (see [No document unless you ask for one](#no-document-unless-you-ask-for-one));
ask for "a report" without a format and it comes in the deployment's format
(`$JOBSMITH_REPORT_FORMAT`); the model proposes a short name from the subject,
and the title falls back to the request. And you do not have to go through
the conversation to be heard — `jobsmith run "compare X and Y, give me that as
a PDF"` gets a PDF, because the **engine** reads the request for a format when
whoever launched the job named none. It only ever fills that silence: a format
you did ask for is never second-guessed, and one this deployment cannot render
is never invented. **A name is not a title and
neither is a format** — asking for one never quietly answers the others.

The name is a *filename*, never a location: no directories, no traversal, no
absolute paths, bounded in length, refused (not flattened) when it is anything
else. A named deliverable lands in the job's own folder next to the files its
steps produced, so two jobs called `rapport` keep two files. And a format
nothing can render *here* is refused before the job exists — where you can
still ask for another one — rather than at the end of a run that spent three
minutes first.

`POST /jobs` takes the same three (`document_name`, `document_title`,
`formats`), and refuses them the same way, with a 400. `formats: ["default"]`
asks for a document in this deployment's format without naming it.

### Asking for a deck

Install one extra and a `slide_deck` step joins the registry: ask for a
presentation and the job leaves a `.pptx` next to its report.

```bash
uv pip install -e ".[pptx]"
jobsmith run "summarise our retrieval options and give me a deck for Monday" --wait
jobsmith outputs <id>            # the report, and the deck as an annex
```

It is a **capability, not a report format**, and the difference is the point.
A report is prose and a Reporter only serializes it; a deck is a different
document — sections, one idea per slide, bullets, speaker notes — so the deck
is *designed* by the model as its own step, which is why its tokens are booked
to it like every other step's. The written report is still delivered:
the deck is one more thing the job produced, never a substitute for the
answer. Like the two above, the step is registered only when something can
render it (`python-pptx`, pure Python, no system libraries).

### Which agent

An **agent** is a pack of capabilities plus a profile — what the thing can do
and how it speaks. Two ship: `default` (research → analysis → critique, plus a
slide deck when the request asks for one, needs nothing but a key) and `banking` (a domain example: document search, slide
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

### No document unless you ask for one

> **Contract change (#96).** Until now every run that planned something wrote
> a report file, whether or not anyone asked for one. **It no longer does.**
> `jobsmith run "<task>"`, `POST /jobs`, `/bg` and the conversation all follow
> one rule: a request that says nothing about a document gets **its answer and
> no file**. If you relied on a `.md` appearing next to every job, ask for it.

Every job answers; only a request that asks for a document gets one. The
answer is never harder to reach for it: it is on the job record
(`jobsmith job <id>`, `GET /jobs/{id}` → `final_answer`), `jobsmith run --wait`
prints it, and the conversation delivers it word for word. A run that writes
no file says so — `deliverable_expected: false` on the record, "none was asked
for" from `jobsmith report` and `/report` — rather than looking like one that
has not finished.

Asking is done in words or in arguments, on every door:

| you want | say | what is written |
|---|---|---|
| an answer | nothing about a file | no file |
| a document | "…as a report", "…in a document I can keep" — or `formats: ["default"]` | the deployment's format (`$JOBSMITH_REPORT_FORMAT`, markdown by default) |
| a given format | "…as a PDF", "…as an html page" — or `formats: ["pdf"]` | that format |
| explicitly no file | "just answer here" — or `formats: []` | no file |

The words are read by the engine itself, so `jobsmith run "… as a report"` is
heard exactly like the same sentence in the conversation. Which step planned,
how long the run took and which door it came through decide nothing: a
four-minute comparison with no file is an answer, a ten-second request for a
PDF is a PDF.

### The deliverable

The deliverable, when one is asked for, is a markdown file by default, and it
is **an answer, not a record of the run**:

````markdown
# compare hexagonal and layered architectures for an LLM agent

<the synthesis>

---

_Produced by job a803205bea59412bae2e376a8555ee62 — jobsmith job a803205b shows
the request, the plan, the steps and what it cost._
````

That last line is the whole of what the file says about itself. The rest — the
request quoted in full, the timestamps, the session, the token bill, the plan
table with a per-step cost column, the DAG — is **recorded, not recited**:
`jobsmith job <id>` and `GET /jobs/{id}` serve it whole, and duplicating it
into the document was the product talking about itself inside the thing you
opened to read an answer.

**What it cost is part of the record.** Every LLM call is booked to the step
that made it, so `jobsmith job <id>` (and the `/events` stream, live while it
runs) answers both *what did this cost* and *which step spent it*. Prices are a
dated snapshot — override them with `$JOBSMITH_PRICES` (inline JSON or a path:
`{"gpt-5.1": {"input": 1.25, "output": 10.0}}`); a model with no price is
reported in tokens rather than in an invented dollar figure.

Per-step material is deliberately **not** inlined either — it lives in the
store, and the same two commands serve it. For a self-contained archive, both
switch back on: `MarkdownReport(with_provenance=True, with_annexes=True)` folds
the record and the step material into one file.

**Or the same thing as a web page.** Ask for html, or set
`JOBSMITH_REPORT_FORMAT=html` to make it what "a report" means here: the
deliverable becomes a self-contained HTML file instead — same document, same order
(the answer, then one line back to the run), no dependency and no network:
inline CSS, and — with `with_provenance` — the plan drawn as an inline SVG,
since a browser renders no mermaid.

**Or as a PDF.** Asking for a PDF (or `JOBSMITH_REPORT_FORMAT=pdf`) prints that very same page:
`PdfReport` renders what the HTML Reporter renders and hands the string to
WeasyPrint, so there is one layout and no Reporter waiting on another's file.
It is the one extra with a dependency **outside Python** — WeasyPrint draws
through pango/cairo, which must be installed where the agent runs:

```bash
uv pip install -e ".[pdf]"
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0   # Debian/Ubuntu
```

(`ubuntu-latest` already has them — CI renders a PDF with no extra step. A
container built `FROM python:3.12-slim` does not.) A format nothing can
render refuses at startup rather than at the end of the first job that asked
for one.

**Or several at once.** The variable takes a comma-separated list —
`JOBSMITH_REPORT_FORMAT=markdown,html` — and a run asked for a document then
writes one file per format, each recorded as an output of the job. The **first** name is the main
deliverable: `report_path`, `jobsmith report <id>` and `GET /jobs/{id}/report`
point at it, the others are the same report rendered again.

A job carries a **list** of outputs (`role: main | alternate | annex`, a
`format`, the capability that produced it): `main` is the deliverable,
`alternate` the same report in another format, `annex` a **file a step
produced** — a chart, an exported table. `jobsmith outputs <id>` and
`GET /jobs/{id}/outputs` list them all, `/outputs/{name}` downloads one.
`/report` is the inline shortcut to the main one and serves **text**: when
that deliverable is a PDF it answers `415` naming the `/outputs/{name}` to
download instead, because "no report" would be false — the job has one, on
disk. `jobsmith report <id>` says the same thing in the terminal.

**A capability can hand back a file** — `slide_deck` is the one that ships.
It writes through the `ArtifactStore` port (`core/artifacts.py`) — `write(job_id, name, data) -> path`, backed by a
directory today and by object storage the day that matters — and names what it
wrote in its result's `meta`; the job records each one as an annex, attributed
to the step. Annexes never disturb the report: they come after the
deliverables and are never `main`, so `report_path` still points at the
report. A declared file that is not on disk is dropped rather than listed —
an output nobody can open is worse than none — and the job says which one, the
same way it does for a report it could not write.

**And they survive a run that did not finish.** A step can declare a file it
wrote even when it then fails, and a job that ends **failed** or **cancelled**
still lists what its steps left on disk — those files are exactly what is
often worth having, and a file recorded nowhere is a file nobody can find.
Such a job gets no *report*, though: there is no answer to write one about, so
it has no `main` output and `report_path` stays empty. The conversation is
told either way — a job that failed **or one you cancelled** is announced with
what happened and the files it left, never in silence. (Cancelling is
something the assistant itself can do on your behalf, which is exactly why it
must not be the one ending that goes unmentioned.)

If writing a file fails (a full disk, a read-only directory), the job is
still **done** — the answer was produced and is stored — and says so: the
error names the format that failed, and any file that did get written is
still listed as an output.

---

## Architecture

### The graph

```
validate_input → document_intent → router ─(direct | empty registry)→ direct_answer ┐
                                     └(plan)→ planner ─(nothing applicable)→ ───────┤
                              └→ executor_dispatch ⇄ {cap_<name> × registry}
                                ↓ (all done)                     ↓
                          merge_results → generation → validate_output
                                              ↑ refine ←┘ (≤ max_refine)  → post_process → END
errors: execution_error → escalate (some result ok) | user_error (none) → END
```

- **Router** — a dedicated triage node. The planner never decides *whether* to
  plan; the router picks `plan` or `direct`, and **fails open to `plan`** on any
  LLM or parse error — except with an empty registry, where there is nothing to
  plan with and `direct` is chosen structurally, without an LLM call. A new
  route is one entry in `Router.routes` plus a node.
- **Document intent** — a second dedicated decision node, answering what file
  the request asked for. It runs **only when the caller named no format**, that
  gate being structural (no model call at all otherwise), chooses among the
  formats this deployment can actually render (a document asked for without a
  format gets the deployment's default), and writes nothing when the request
  said nothing — which means **no file**, so any error or answer it cannot use
  costs a document at worst, never invents one. What it decides reaches the job record through the runner
  and the manager, never from inside the node.
- **Planner** — renders its prompt from the registry, validates the LLM's JSON
  DAG (names, applicability, dangling dependencies, Kahn cycle check).
- **Executor** — computes the ready capabilities of each wave and returns
  `Send`s; capability nodes edge back to it. Any DAG, no baked-in schedule.
- **Two error channels** — planner/generation failures hard-stop; capability
  failures land in `results` with `ok: False` and the run degrades gracefully.
  A plan left empty because every step was inapplicable is not a failure: it
  joins the `direct` route and the agent answers anyway.

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
3. If it produces a **file**, take an `ArtifactStore` in the constructor
   (`ctx.artifacts`), write with `state.get("job_id", "")` and a filename, and
   declare it: `meta=artifact_meta(ArtifactRef(path, title="Revenue chart"))`.

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
  tui/          adapter — Textual: chat pane, job list, job detail
  agents/       ★ what each agent IS — a capability pack + a profile
    default/      read_files/prior_jobs/documents → research → analysis
                  → critique, + slide_deck (a .pptx annex)
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
| `POST /sessions` · `POST /sessions/{id}/messages` | chat; a reply is `{"type": "message"}`, or `{"type": "proposal"}` where the approval gate was kept. A task runs inside the turn, so this can take as long as the task |
| `POST /sessions/{id}/approval` | answer a proposal — `{"approved": bool}` |
| `.../messages/stream` · `.../approval/stream` | the same turn as SSE: `token`, `tool_started`, `tool_finished`, `job_started`, then that same reply |
| `GET /jobs` · `GET /jobs/{id}` | listing and full detail (plan, timings, results) |
| `POST /jobs` · `POST /jobs/{id}/cancel` | direct launch, cancellation |
| `GET /jobs/{id}/outputs[/{name}]` · `/report` | the deliverables (`/report` is text-only: `415` on a PDF, pointing at the download) |
| `GET /events` | SSE stream of job progress |

---

## Configuration

| | |
|---|---|
| `--llm anthropic\|openai\|fake` | provider for **both** stacks (default: auto-detected from keys) |
| `--agent NAME` | which agent to run — `default` or `banking` (applies to whichever process owns the engine, so pass it to `serve`) |
| `--docs DIR` | ground jobs in the files under `DIR` (default: `$JOBSMITH_DOCS`); without it the agent runs on the model's own knowledge. It also becomes readable by name: a request may point `read_files` at a file inside it |
| `TAVILY_API_KEY` | enables the `web_search` step (extra `.[web]`); absent, the capability is not registered |
| `$TAVILY_SEARCH_DEPTH` | `advanced` (default) or `basic` — how hard `web_search` digs; `basic` costs less per call and retrieves less. Anything else is refused at startup |
| extra `.[tui]` | enables `jobsmith ui`; absent, the command says what to install |
| `$JOBSMITH_THEME` | the UI's theme (default `ember-dark`); `--theme NAME` overrides it, `ctrl+p` switches it for the session |
| `$JOBSMITH_SYNC_TIMEOUT` | seconds a task may hold the conversation before it is promoted to the background (default `20`). `0` never waits — every task goes to the background, which is what jobsmith did before |
| `$JOBSMITH_INLINE_ANSWER_MAX` | characters an answer may have and still be written into the conversation word for word when a background job lands (default `2000`). Past it you get the path instead. `0` never writes one — except for a run that produced no file, where the conversation is the only channel there is |
| `$JOBSMITH_APPROVE_JOBS` | `1` restores the y/N approval card before a task runs. Off by default: the agent says what it is doing, and cancelling is the undo |
| extra `.[pptx]` | enables the `slide_deck` step — a `.pptx` annex next to the report; absent, the capability is not registered |
| `--db memory\|<file.db>\|<postgres DSN>` | persistence (default: `$JOBSMITH_DB`, else `jobs.db` in the data directory — see [Where your jobs live](#where-your-jobs-live)). `memory` keeps nothing past the process |
| `$JOBSMITH_REPORTS_DIR` | where deliverables and annexes are written (default: `reports/` in the data directory). Resolved to an absolute path at startup; it is also the directory `read_files` may read a report back from |
| `$XDG_DATA_HOME` | relocates the data directory (`$XDG_DATA_HOME/jobsmith`), on every platform |
| `$JOBSMITH_PRICES` | per-model prices for the cost estimate, as inline JSON or a path to a JSON file (USD per million tokens) |
| `$JOBSMITH_REPORT_FORMAT` | `markdown` (default), `html` or `pdf` (extra `.[pdf]` + pango/cairo) — the format of a document **asked for without naming one** ("…as a report", `formats: ["default"]`). It never causes a file to be written: a request that says nothing about a document gets none (#96). A comma-separated list (`markdown,pdf`) writes one file per format, the first being the main one; a format nothing can render here is refused at startup |
| `--url` / `--local` | point at another daemon / never use one |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | key auto-detection; Anthropic wins if both are set |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | model override; the base URL points at Ollama, vLLM or a gateway |

Persistence is on by default and backs both jobs and conversations. Pick
another file, Postgres, or nothing at all:

```bash
jobsmith --db ./project.db chat                                       # another SQLite file
uv pip install -e ".[postgres]"  && jobsmith --db postgresql://user:pass@localhost/agent chat
jobsmith --db memory chat                                             # keep nothing
```

---

## Development

```bash
make help          # every target
make check         # lint + types + domain-leakage gate + tests
make types         # pyright over jobsmith/
make test T=router # one keyword's worth
make coverage      # per-module coverage report
make fix           # ruff --fix
```

`make types` runs **pyright**, configured once in `[tool.pyright]` — the same
block Pylance reads, so VS Code and CI agree instead of each flagging what the
other ignores. It is the only gate here that can see a bug the tests cannot: a
signature that lies is invisible at runtime, and the one that prompted this
(a factory promising `Reporter` and returning `Reporter | MultiReporter`) was
noticed by accident in an editor. Scope is `jobsmith/` — tests and `evals/` are
deliberately out.

`reportTypedDictNotRequiredAccess` is on, and it is a read-discipline gate
rather than a lint: the graph state schemas are `total=False` because a node
returns a *partial* update, which is right for writes and wrong for reads. Of
its 31 hits, 17 were `query` — guaranteed at entry, so it is now `Required` in
`AgentState` and `CapabilityBaseState` — 12 were keys guaranteed only by graph
order, now read with a default that says what missing means, and 2 were reads
of `CapabilityResult`, whose keys really are all optional.

### Judging a prompt change

The router, the planner and the generator are prompts, and a prompt change used
to be judged by eye on one example. `evals/` turns that into a number.

```bash
make eval                     # deterministic tier — fakes, no API key, runs in CI
make eval-llm                 # the same golden set against a real provider (opt-in)
make eval ARGS='--repeat 3'   # sample the same cases repeatedly to see the variance
make eval ARGS='--report-format html'   # score the other deliverable format
python -m evals --list        # what the golden set contains
```

It scores **properties, never expected text**: the plan only names registered
capabilities, its DAG is acyclic with satisfiable dependencies, an obviously
simple message is triaged `direct` and a compound one `plan`, the run reaches
the terminal it should, every planned step ran and reported success, and the
deliverable carries a title, the answer and the job that produced it. Wording may vary
freely; structure may not — and neither does the deliverable's format: the
report checks read the file through a markup stripper, so a markdown run and an
HTML one score identically. A format whose file is bytes (`pdf`) is refused
before the first case runs: there is no text in it to score, and a run that
failed every report check on the format would become the next run's baseline.

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

CI runs what `make check` runs — lint, types, the leakage gate, tests — on
every push and pull request, across Python 3.11 and 3.12, and verifies that
`uv.lock` still matches `pyproject.toml`. It installs every extra, so the
optional providers' imports are type-checked there too.

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

Working end to end: chat that runs tasks in the turn and promotes the slow
ones, DAG planning and parallel execution, persistence
(SQLite by default, Postgres, or memory on request), the daemon/client split, the HTTP API with SSE,
markdown deliverables.

Honest v1 boundaries:

- **Cancellation and SSE are in-process.** A client can cancel a job the daemon
  runs; cross-process preemption writes a best-effort tombstone. This matters
  more than it used to: cancelling is the undo that replaced the approval card,
  so it is the only control over a task already running.
- **The answer lives in the turn and in a file, and nothing yet decides which.**
  A task that finishes in the conversation delivers its answer there word for
  word *and* writes the report; a promoted one only writes it. That is a
  deliberate seam, not a settled question.
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
- **Three Reporters ship: markdown, HTML, PDF.** PPTX would be a fourth over
  the same `JobDocument` — but a generation rather than a rendering, so it is
  not one of these. The PDF is the only deliverable with a **deployment**
  constraint: a daemon that produces one needs pango/cairo where it runs.
- **`/report` serves text only.** A binary deliverable is fetched whole from
  `/jobs/{id}/outputs/{name}`; the shortcut refuses with a `415` that names
  it rather than pretending the job has no report.
- **The terminal UI shows what the record holds, live.** `jobsmith ui` follows
  the event stream, so plan, per-step status, cost and files move on their own
  — but a step's `took` is still derived from when its dependencies landed,
  because the engine records when a step *finished* and never when it started.
  Its file pane locates what a job produced and does not promise you can open
  it: with a daemon, those paths are on the daemon's machine.
- **No web UI.** Everything is terminal or HTTP for now; the API already serves
  what a chat / jobs-DAG / artifacts interface would need.
