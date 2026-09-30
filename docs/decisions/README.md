# Decision records

`CLAUDE.md` holds the **rules** — what an agent editing this code must know not
to break something. This directory holds **why** those rules are what they are:
one record per decision, carrying what was observed, what was measured, the
options that lost and the reasons they lost. A rule in `CLAUDE.md` points here
(`→ docs/decisions/0085-answer-in-the-conversation.md`); a record is read by
whoever is about to touch that area, not loaded into every session.

## What a record is

A markdown file, `NNNN-<slug>.md`, following [`TEMPLATE.md`](TEMPLATE.md):
Context · Decision · Alternatives and why not · Measured · Consequences ·
Supersedes / superseded by. The headers are short; the substance is prose, and
numbers are kept as numbers — "measured, 40/40 runs green" is the part a later
reader cannot reconstruct from a diff.

**Numbering**: `NNNN` is the issue number, zero-padded (`0096-…` for #96). A
decision taken without an issue gets one first. A record whose number was a PR
rather than an issue (0028) keeps the number it has always been cited by.

## When to write one

**The PR that takes the decision writes its record**, in the same PR — the agent
that measured, falsified and chose is the only one that knows why; rebuilding
that from a diff afterwards loses exactly the part worth keeping. A decision is
anything a later reader could plausibly undo without knowing the reason: a
default, a refusal, a port's shape, a trade-off with a number behind it. A
bug fix with no alternative worth recording needs none.

`CLAUDE.md` gains a line only when the decision creates or changes a **rule**,
and that line points at the record.

## Superseding

A record is history: it is never edited to say something new. A later decision
writes its own record, and both are marked — `Status: superseded by NNNN` (or
*partially superseded*) on the old one, `Supersedes NNNN` on the new one — and
the index below says it too. Fixing a typo, a dead link or a pronoun is fine;
changing what a record claims is not.

The records migrated from `CLAUDE.md` in #102 carry their original text,
headed by the section of `CLAUDE.md` it lived in; newer records follow the
template. The [`scribe`](../../.claude/agents/scribe.md) agent keeps this index
current and writes the records a merged PR forgot, marked as reconstructed.

## Index

| # | Record | Decision | Status |
|---|---|---|---|
| 0000 | [Foundations: decisions older than the issue that would carry them](0000-foundations.md) | the chat-first REPL and provider selection, strict status checks, resources' lifetimes, the banking example's ports, the rule that a capability nothing can serve stays out of the registry | accepted; partially superseded by 0085; partially superseded by 0096 |
| 0002 | [What a job cost is part of its record](0002-usage-ledger.md) | token/cost usage rides an ambient ledger attributed by `checkpoint_ns`, and is kept per job and per step | accepted |
| 0003 | [Prompt changes are scored, not eyeballed](0003-evals-harness.md) | a golden set scored on structural properties, in a deterministic tier that gates CI and an LLM tier that never does | accepted; partially superseded by 0085; partially superseded by 0096 |
| 0004 | [A launched job carries the conversation's referent](0004-conversation-referent.md) | a self-contained `query` plus a bounded excerpt of the thread as an input, since the engine never sees the thread | accepted |
| 0005 | [A stopped job resumes from its checkpoint](0005-resume.md) | resume re-enters the thread with `None`, gated on status and on a non-empty frontier; re-running part of a finished DAG is not implemented | accepted |
| 0006 | [Job notices in the conversation](0006-job-notices.md) | completion and progress notices are transient system messages, pushed only when a job moved, placed right after the system prompt | accepted; partially superseded by 0085 |
| 0009 | [The Reporter seam, and an HTML deliverable with no dependency](0009-reporters-and-html.md) | a format-independent `JobDocument` serialized by Reporters; HTML is escaped-first with a deliberately small markdown subset | accepted |
| 0010 | [Cross-process ownership and cancellation](0010-cross-process-ownership.md) | a lease on the record says which process owns a run, and a cancel is a request the owner acts on; events stay in-process | accepted; partially superseded by 0063 |
| 0025 | [The report checks are format-independent for real](0025-format-independent-report-checks.md) | evals read the deliverable through `deliverable.extract`, and refuse a binary one rather than score it (#45) | accepted |
| 0028 | [One run hands back several deliverables](0028-several-deliverables.md) | `Reporter.write` returns a list, exactly one output is `main`, and a failed write leaves the job DONE | accepted |
| 0031 | [pyright is a gate, and reads of partial state justify themselves](0031-type-gate.md) | pyright in `make check` and CI; `reportTypedDictNotRequiredAccess` on, `query` Required, everything else read with `.get()` | accepted |
| 0034 | [A PDF deliverable: a peer Reporter, not a post-processor](0034-pdf-deliverable.md) | the PDF is the HTML page printed in memory; the engine is probed at startup; `/report` answers 415 for a binary deliverable | accepted; partially superseded by 0108 |
| 0035 | [A capability can produce a file; a slide deck is a generation](0035-capability-artifacts-and-slide-deck.md) | an `ArtifactStore` port and annexes recorded in plan order; `slide_deck` asks the model for deck-shaped structure | accepted |
| 0038 | [Nothing to run is answered, not failed](0038-empty-registry-and-empty-plan.md) | an empty registry routes `direct` structurally, and a plan emptied by applicability goes to `direct_answer` | accepted |
| 0041 | [A file outlives the run that produced it](0041-files-outlive-the-run.md) | artifacts are collected at every terminal, a declared-but-absent file is named in `job.error` | accepted |
| 0048 | [A terminal UI, and live progress on the port](0048-terminal-ui.md) | `subscribe`/`list_outputs`/`find_output` joined the port; the TUI renders the same flow, re-reads on events, speaks theme roles | accepted |
| 0050 | [The chat streams its turn](0050-streamed-turn.md) | a turn is a flow of events with exactly one terminal; `send` drains `stream`; the chat stream never drops a token | accepted |
| 0053 | [A step's node name says which step finished](0053-step-timestamps.md) | a `cap_*` update's `results` is the whole channel, so the finished step is read from the node name | accepted |
| 0054 | [A report's title is cut on a word](0054-title-on-a-word.md) | `document_title` collapses whitespace and cuts at a word boundary with an ellipsis | accepted |
| 0055 | [The request decides what the document is called, titled and written as](0055-document-name-title-format.md) | `document_name`/`document_title`/`formats` are facts on the job, refused in `create_job`, shown before the run | accepted; partially superseded by 0096 |
| 0058 | [The deliverable is written for its reader](0058-reader-facing-deliverable.md) | the prompts that produce a deliverable name their reader; `SUBJECT_ONLY_RULE` on every material step | accepted; partially superseded by 0073; partially superseded by 0082 |
| 0059 | [A refusal is not an answer, and says so as data](0059-refusal-as-data.md) | a first-line marker routes a declared refusal to `unanswered`; fail-open; never refined; the job stays DONE | accepted |
| 0060 | [A request can name a file, and the job reads it](0060-requests-name-files.md) | `read_files` reads a named path through a `DocumentReader` port; `core/paths.py` resolves before it compares | accepted |
| 0061 | [A printable document is not a presentation](0061-printable-is-not-a-presentation.md) | a capability that can be mistaken for another says what it is not; `must_exclude` measures it | accepted |
| 0063 | [An unconfigured jobsmith keeps its jobs](0063-persistent-by-default.md) | the default persistence is a SQLite file in the user data dir, and every writer takes the lock up front | accepted; partially supersedes 0010 |
| 0064 | [A backing that went away is one exception](0064-service-unavailable.md) | the port throws `ServiceUnavailable` for a transport failure and nothing else; front-ends catch it by name | accepted |
| 0073 | [A deliverable answers with the material it has](0073-deliverable-answers.md) | the generator gets an obligation, a refusal gets a high bar and a shape, and `critique` stopped feeding the generator | accepted; partially supersedes 0058; partially superseded by 0082 |
| 0074 | [A follow-up job references the job, not the file it wrote](0074-prior-job-references.md) | a `PriorJobSource` port hands back an earlier run's answer and per-step material, scoped to the session in `chat/tools.py` | accepted; partially superseded by 0085; gap closed by 0104 |
| 0075 | [web_search grounds on the page, not the snippet](0075-web-pages-not-snippets.md) | Tavily's `raw_content` over the snippet, `advanced` depth by default, 8 000 characters per document | accepted |
| 0076 | [Nested lists nest in the HTML and PDF deliverables](0076-nested-lists.md) | one level every two spaces, depth clamped to the open-list stack, a nested list opened inside its item | accepted |
| 0077 | [The generator is told which files the run delivers](0077-generator-knows-the-files.md) | the generator and refiner get the list of delivered files (or "none") and no prompt offers a file by example; `answer_invents_no_file` checks the answer against `Job.outputs` | accepted; partially superseded by 0126 |
| 0080 | [A direct reply asked for as a file is written as that document](0080-direct-answer-as-document.md) | `DirectResponder` gets the delivered-files note always and `DIRECT_DOCUMENT_RULE` when a file is written; the request stays on the direct route; `report_reader_facing` scores that file | accepted |
| 0081 | [The grounding reaches the reasoning](0081-grounding-reaches-reasoning.md) | `research` reads every retrieval step's material (bounded), says so in `meta["grounded_on"]`, and an eval checks it | accepted |
| 0082 | [critique reviews the material, not the run](0082-critique-checks-the-subject.md) | `critique` checks claims against their sources, at most 8 bullets, and feeds the generator again | accepted; partially supersedes 0073; partially supersedes 0058 |
| 0083 | [A task runs synchronously by default; the background is a promotion](0083-synchronous-by-default.md) | `launch_job` starts the job and waits `$JOBSMITH_SYNC_TIMEOUT`; the answer is written into the turn; the approval card became a notice | accepted; partially superseded by 0085 |
| 0085 | [The answer comes back in the conversation, and the file stops being about the run](0085-answer-in-the-conversation.md) | a promoted answer uses the same verbatim channel below `$JOBSMITH_INLINE_ANSWER_MAX`; the report is title, answer, job reference | accepted; partially supersedes 0083; partially supersedes 0074; partially supersedes 0003; partially supersedes 0000 |
| 0086 | [The plan is announced in the turn that waits on it, as activity](0086-plan-announced-in-the-turn.md) | a seventh event, `job_planned {job_id, steps}`, written by `launch_job` from the manager's own events while it waits; REPL stderr, TUI activity line; nothing after promotion; no one-step plan | accepted |
| 0090 | [The engine reads the request for what document it asked for](0090-document-intent-node.md) | `document_intent` is a dedicated decision node before triage that fills silence, never overrides, cannot refuse | accepted |
| 0096 | [A request that says nothing about a document gets none, on every door](0096-no-document-unless-asked.md) | `_deliverable_wanted` is `bool(job.formats)`; silence is recorded as `FormatsChosen(None)`; the answer checks leave the file gate | accepted; partially supersedes 0003; partially supersedes 0055; partially supersedes 0000 |
| 0100 | [Job events cross processes on a SQLite file](0100-cross-process-events-sqlite.md) | `SqliteWatchEvents`: `PRAGMA data_version` on its own read-only connection, only while subscribed; re-reads the index only when it moved; Postgres is 0138 | accepted |
| 0102 | [Decisions are recorded in `docs/decisions/`, and `CLAUDE.md` keeps the rules](0102-decision-records.md) | `CLAUDE.md` holds only rules with a `→ NNNN` pointer; every decision gets its own record, written by the agent that took it; a scribe keeps the whole under budget | accepted (reconstructed) |
| 0104 | [The job notice names the earlier jobs a run builds on](0104-notice-names-prior-jobs.md) | `job_started`/`proposal` carry `from_jobs` as `{job_id, query}` (query cut on a word at 60), rendered by the one renderer per front-end; nothing when none | accepted; closes the gap in 0074 |
| 0108 | [The PDF engine is probed on the first request that needs it, not to compose the app](0108-pdf-engine-probed-on-first-need.md) | PDF is offered when installed (`find_spec`); the engine is loaded, cached, in `create_job` or when the document step chose it; eager only for a PDF deployment default; an unloadable engine is a `ValueError` refusal | accepted; partially supersedes 0034 |
| 0109 | [Fewer themes, fewer hammer rounds, a `slow` marker](0109-faster-suite.md) | the colour test renders 7 representative themes, not 25; the SQLite hammer runs 10 rounds, not 150; `make test-fast` skips tests marked `slow` | accepted |
| 0110 | [Tests are organised by purpose and share builders through one module](0110-tests-by-purpose.md) | one file per component or rule, never per issue; builders in `tests/support.py`, no test imports another; data-only variants parametrized; docstrings state the property | accepted (all 5 steps landed) |
| 0121 | [The scribe runs every ~10 merges, not every ~3](0121-scribe-every-ten-merges.md) | each pass is a CI-gated PR; ~10 merges or the CLAUDE.md budget, no per-batch trigger | accepted; partially supersedes 0102 |
| 0125 | [A file asked for in words is a document asked for](0125-a-file-asked-for-in-words.md) | `FILE_REQUEST_RULE` in the document step's prompt: "save it to a file" is `requested` whatever else the request asks; a file it is only about is not | accepted |
| 0126 | [An answer written to a file is that file, not a text about it](0126-answer-is-the-file.md) | when the list of delivered files names the answer, `ANSWER_FILE_RULE` says that entry is the text being written; "delivered separately" is scoped to the other files; `SUBJECT_ONLY_RULE` names a file to save in; `answer_is_the_file` scores it | accepted; partially supersedes 0077 |
| 0129 | [The material steps write no slides](0129-material-not-slides.md) | `SUBJECT_ONLY_RULE` names the deck: a separate step builds it, so no slides, slide titles, outline or speaker notes in the material | accepted |
| 0130 | [Effort is proportionate to the risk](0130-proportionate-effort.md) | a budget per issue (≤45 min, node probe ≤300 calls with `evals/probe.py`, one variant, record ≤20 lines); small fixes done directly | accepted |
| 0133 | [The PR protocol runs as commands, not by hand](0133-protocol-as-commands.md) | `make probe` (main vs branch, parallel), `make combo`, `make hooks`, parallel `make check` with `pytest -n auto`, repo auto-merge, a test for the index and `→ NNNN` | accepted |
| 0134 | [Mutation testing on the diff replaces falsifying by hand](0134-mutation-on-the-diff.md) | `make mutate TESTS=…`: cosmic-ray + `cr-filter-git` on a scratch worktree, only the changed lines, survivors listed by line | accepted |
| 0138 | [Job events cross processes on Postgres, by LISTEN/NOTIFY](0138-postgres-notify-events.md) | `PostgresNotifyEvents`: NOTIFY on publish, LISTEN while subscribed, both on connections off the pool; `WatchedEvents` shares the rest with SQLite | accepted |
| 0141 | [Complete queries filter in the store; human listings cut after ordering](0141-complete-queries-at-scale.md) | `load_all` is complete (filters in the store, one query, no cut); `list_jobs` newest first then cut; `limit=None` for what must see everything | accepted |
| 0161 | [The job engine is the product: it runs any LangGraph graph and tells each ending once](0161-core-v1-job-engine.md) | `engine/` knows no product (G4); `GraphSpec`, facts, `run_for`, `reply_to` + deliverers, LangChain usage; the DAG, files, chat and `adapters/langchain` sit on top; details in `docs/design/core-v1.md` | accepted |
| 0165 | [A graph agent is served through the engine's own port, on its own paths](0165-graph-agent-served.md) | `EngineService`/`LocalEngineService`: launch (input, graph, label, `reply_to`, wait), get, list, cancel, resume, events; `AgentApp.service()` answers it for a graph agent; API on `/engine/jobs`, the DAG's routes untouched | accepted |
| 0166 | [A pushed return address is delivered off the persist path, retried until it lands](0166-push-deliverers.md) | `Pushed.push` never inline: saved, then a task per `job_id#attempt` with capped backoff, stamped after, only for that attempt; pending pushes resume in `recover_interrupted`; `Webhook(allowed)` | accepted |
| 0167 | [A run paused at an interrupt waits for an answer, with its own status](0167-needs-input.md) | runner yields `Interrupted`; `needs_input` + `Job.asked`; `answer_job` = `Command(resume=…)`, a new attempt; not an ending, not delivered; `POST /engine/jobs/{id}/answer` | accepted |
| 0168 | [Re-running part of a finished DAG job is a new job built on it, never a new attempt](0168-partial-rerun-is-a-new-job.md) | `from_jobs` (0074) carries the old answer and per-step material; an attempt stays a try at one request; DONE stays unresumable | accepted |
| 0169 | [CI runs the Postgres paths; Postgres setup takes a tried advisory lock](0169-postgres-in-ci.md) | a `postgres:16-alpine` service in the `check` job sets `$JOBSMITH_TEST_PG`; both setups under `pg_try_advisory_lock` in a loop — a waited lock deadlocks with `CREATE INDEX CONCURRENTLY` | accepted |
| 0172 | [The chat's launch_job stays its own shell over the adapter's contract](0172-chat-launch-stays-a-shell.md) | shared: `run_for`, `told_in_thread`, `JobDeliveryMiddleware`; the chat's approval, notice, plan watch and verbatim answer stay in `chat/` rather than six hooks on `launch_tool` | accepted |
| 0177 | [A running job is amended by stop, checkpoint update, run on; the DAG drops steps with it](0177-amend-a-running-job.md) | `JobManager.amend_job` (stop, `aupdate_state`, resume; the stop never delivered); `DagJobs.drop_steps` + `without_steps`; a running step re-runs unless dropped; chat/API: #196 | accepted |
| 0187 | [A FAILED job says why as data, and "retryable" is what a resume answers](0187-structured-failure.md) | `Job.failure` = kind (`raised`/`interrupted`/`declared`/`no_result`/`unreadable`), `pending`, `retryable` = `bool(pending)`, the resume gate's test; every FAILED through `_fail` | accepted |
| 0189 | [An interrupted job is relaunched only where its graph allows it, by the daemon, once](0189-orphan-relaunch.md) | `GraphSpec.relaunch` (0 = never) bounds attempts; `relaunch_interrupted` from `serve` only; claimed on a shared store (lease, one heartbeat, re-read) | accepted |
| 0191 | [A step that failed transiently runs again before its dependents; no replanning](0191-repair-transient-steps.md) | `_emit_failure(retryable=True)` (a raised model call, never an empty answer); executor re-dispatches within `max_step_retries`, counted in `completed_capabilities`; replan left out | accepted |
| 0194 | [A capability is Sent the parent state without the append-only channels](0194-send-without-appended-channels.md) | `Send` strips `completed_capabilities`/`errors`: a sub-graph echoed them back and the parent appended them again, miscounting the runs 0191 bounds | accepted |
| 0196 | [The chat skips a step with a tool of its own; the port and the API answer it like a resume](0196-skip-a-step-from-the-chat.md) | chat `skip_steps` (refusals = the DAG's words, returned to the model); `JobService.drop_steps` shaped like `resume_job`; `POST /jobs/{id}/drop`, 409; probe gains a `chat` node: `main` cancelled the job on "skip the critique" 15/25 | accepted |
| 0198 | [A docs-only pull request runs only the docs tests, under the same check names](0198-docs-only-ci.md) | a `Scope` step diffs the merge commit against its base (`--no-renames`); only `docs/**` and top-level `*.md` → the two docs tests in a bare venv, `--noconftest`; anything else, or any doubt, runs it all | accepted |
| 0202 | [A task a stop caught between its writes is pending, and the resume forks its checkpoint first](0202-resume-a-half-written-task.md) | LangGraph keeps a cancelled task's partial writes and counts it done on resume (no route: `next` empty); `GraphRunner._half_written` + `pending()` + `_repaired` (`__copy__` fork) before resume/answer/update; detection alone ends DONE with a wrong result | accepted |
| 0206 | [The compiler's baseline is a ReAct agent over the retrieval ports only, reasoning alone](0206-react-baseline.md) | `agents/react/`: `create_agent` over `read_prior_job`/`read_file`/`search_documents`/`web_search`, each only when a port backs it; one item shape within `research`'s budget; no reasoning capability as a tool, since before step 1 they read an empty state | accepted; partially superseded by 0208 |
| 0208 | [The DAG and the ReAct baseline are compared on run-time-width tasks, same material and bounds, scored on coverage](0208-compare-harness.md) | `evals/compare.py` (`make compare`): one local fixture as both agents' `DocumentSource`, success = every `must_mention` term; one baseline search reads one DAG query (6 hits, pinned). First measure: equal coverage, DAG ~4× the tokens, ~4-5× the time | accepted |
| 0211 | [A plan step is known by its id, and a capability learns it from the node wrapper, not from its own code](0211-step-identity.md) | `PlanStep.id` (= the name while duplicates are banned; `step_id()` for plans without one); the executor Sends `step`; `state_graph` wraps nodes to set a `ContextVar` `_emit_*` key by; results, `step:<id>`, run count, `step_finished_at` by id | accepted; partially superseded by 0217 |
| 0213 | [Whoever reads a plan speaks step ids; a capability's name designates its only step and is refused when it runs as several](0213-plan-readers-speak-ids.md) | `step_id` reads any mapping; plan event and notice carry `id`; chat, REPL, TUI key and label by id; `drop_steps` takes ids, a name resolves to its only step, ambiguous refused with the ids | accepted |
| 0215 | [A capability reads another's material by capability, from every step that ran it, in plan order](0215-material-by-capability.md) | `results_of(state, capability)` (plan order, ok only, a name outside a plan counts); five readers go through it, blocks labelled by step id; `CapabilityBaseState` declares `plan` so a sub-graph sees it | accepted |
| 0217 | [One capability may run as several steps, told apart by id, each counting its own usage through a scope the node names](0217-two-steps-one-capability.md) | engine `usage_scope(name)` read first by `current_scope`; a step enters `cap_<id>`; the planner's ban moves from capabilities to ids (none given = the name); the channel keeps its name | accepted |
| 0219 | [The planner's IR is sent in strict mode whole; a keyword a provider refuses falls to the analysis, never the arguments](0219-strict-mode-probe.md) | `evals/strict_probe.py`: OpenAI accepts the normalised step-1 IR (closed objects, every key required, `oneOf` → `anyOf`); first-call validity 37/40 strict vs 34/40 plain vs 30/40 args-as-a-string (gpt-5-nano); Anthropic unmeasured, owed before 1c | accepted |

### Issues cited without a record of their own

- **#1** — the default agent's grounding steps (`documents`, `LocalFiles`, quotable ids) — rules kept in `CLAUDE.md`
- **#5** — resume landed as [0005](0005-resume.md) and the issue was closed; its other half never got an issue, although [0086](0086-plan-announced-in-the-turn.md) says "remains #5": re-running part of a finished job is #168, interjection on a running plan is #177 (found by #174)
- **#8** — the web UI, still open; named as the future consumer of `POST /sessions/{id}/messages` in [0050](0050-streamed-turn.md)
- **#13** — `web_search` behind `DocumentSource`, its adapter `TavilySource` — see [0075](0075-web-pages-not-snippets.md)
- **#36** — one agent, one purpose: the final text's brief is the agent's profile — referenced from [0035](0035-capability-artifacts-and-slide-deck.md)
- **#45** — evals refuse a deliverable they cannot read — recorded in [0025](0025-format-independent-report-checks.md)
- **#49** — a PR (the remote client hears the same events) — see [0048](0048-terminal-ui.md) and [0074](0074-prior-job-references.md)
- **#57** — a PR (#48 step 3, the TUI follows a job) — recorded in [0048](0048-terminal-ui.md) and [0064](0064-service-unavailable.md)
- **#62** — decks are 16:9 — recorded in [0035](0035-capability-artifacts-and-slide-deck.md)
- **#80** — the plan guard landed with [0096](0096-no-document-unless-asked.md); the direct reply written as a file is [0080](0080-direct-answer-as-document.md)
- **#84** — a run leaves a document because the request asked — its narrative was replaced by [0096](0096-no-document-unless-asked.md), which supersedes it; PR #89
- **#89** — the PR for #84 — see [0090](0090-document-intent-node.md) and [0096](0096-no-document-unless-asked.md)
- **#100** — cross-process events: SQLite [0100](0100-cross-process-events-sqlite.md), Postgres [0138](0138-postgres-notify-events.md)
