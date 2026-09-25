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
| 0077 | [The generator is told which files the run delivers](0077-generator-knows-the-files.md) | the generator and refiner get the list of delivered files (or "none") and no prompt offers a file by example; `answer_invents_no_file` checks the answer against `Job.outputs` | accepted |
| 0081 | [The grounding reaches the reasoning](0081-grounding-reaches-reasoning.md) | `research` reads every retrieval step's material (bounded), says so in `meta["grounded_on"]`, and an eval checks it | accepted |
| 0082 | [critique reviews the material, not the run](0082-critique-checks-the-subject.md) | `critique` checks claims against their sources, at most 8 bullets, and feeds the generator again | accepted; partially supersedes 0073; partially supersedes 0058 |
| 0083 | [A task runs synchronously by default; the background is a promotion](0083-synchronous-by-default.md) | `launch_job` starts the job and waits `$JOBSMITH_SYNC_TIMEOUT`; the answer is written into the turn; the approval card became a notice | accepted; partially superseded by 0085 |
| 0085 | [The answer comes back in the conversation, and the file stops being about the run](0085-answer-in-the-conversation.md) | a promoted answer uses the same verbatim channel below `$JOBSMITH_INLINE_ANSWER_MAX`; the report is title, answer, job reference | accepted; partially supersedes 0083; partially supersedes 0074; partially supersedes 0003; partially supersedes 0000 |
| 0086 | [The plan is announced in the turn that waits on it, as activity](0086-plan-announced-in-the-turn.md) | a seventh event, `job_planned {job_id, steps}`, written by `launch_job` from the manager's own events while it waits; REPL stderr, TUI activity line; nothing after promotion; no one-step plan | accepted |
| 0090 | [The engine reads the request for what document it asked for](0090-document-intent-node.md) | `document_intent` is a dedicated decision node before triage that fills silence, never overrides, cannot refuse | accepted |
| 0096 | [A request that says nothing about a document gets none, on every door](0096-no-document-unless-asked.md) | `_deliverable_wanted` is `bool(job.formats)`; silence is recorded as `FormatsChosen(None)`; the answer checks leave the file gate | accepted; partially supersedes 0003; partially supersedes 0055; partially supersedes 0000 |
| 0102 | [Decisions are recorded in `docs/decisions/`, and `CLAUDE.md` keeps the rules](0102-decision-records.md) | `CLAUDE.md` holds only rules with a `→ NNNN` pointer; every decision gets its own record, written by the agent that took it; a scribe keeps the whole under budget | accepted (reconstructed) |
| 0104 | [The job notice names the earlier jobs a run builds on](0104-notice-names-prior-jobs.md) | `job_started`/`proposal` carry `from_jobs` as `{job_id, query}` (query cut on a word at 60), rendered by the one renderer per front-end; nothing when none | accepted; closes the gap in 0074 |
| 0108 | [The PDF engine is probed on the first request that needs it, not to compose the app](0108-pdf-engine-probed-on-first-need.md) | PDF is offered when installed (`find_spec`); the engine is loaded, cached, in `create_job` or when the document step chose it; eager only for a PDF deployment default; an unloadable engine is a `ValueError` refusal | accepted; partially supersedes 0034 |
| 0109 | [Fewer themes, fewer hammer rounds, a `slow` marker](0109-faster-suite.md) | the colour test renders 7 representative themes, not 25; the SQLite hammer runs 10 rounds, not 150; `make test-fast` skips tests marked `slow` | accepted |
| 0110 | [Tests are organised by purpose and share builders through one module](0110-tests-by-purpose.md) | one file per component or rule, never per issue; builders in `tests/support.py`, no test imports another; data-only variants parametrized; docstrings state the property | accepted (first step: the document cluster) |

### Issues cited without a record of their own

- **#1** — the default agent's grounding steps (`documents`, `LocalFiles`, quotable ids) — rules kept in `CLAUDE.md`
- **#8** — the web UI, still open; named as the future consumer of `POST /sessions/{id}/messages` in [0050](0050-streamed-turn.md)
- **#13** — `web_search` behind `DocumentSource`, its adapter `TavilySource` — see [0075](0075-web-pages-not-snippets.md)
- **#36** — one agent, one purpose: the final text's brief is the agent's profile — referenced from [0035](0035-capability-artifacts-and-slide-deck.md)
- **#45** — evals refuse a deliverable they cannot read — recorded in [0025](0025-format-independent-report-checks.md)
- **#49** — a PR (the remote client hears the same events) — see [0048](0048-terminal-ui.md) and [0074](0074-prior-job-references.md)
- **#57** — a PR (#48 step 3, the TUI follows a job) — recorded in [0048](0048-terminal-ui.md) and [0064](0064-service-unavailable.md)
- **#62** — decks are 16:9 — recorded in [0035](0035-capability-artifacts-and-slide-deck.md)
- **#80** — open: `report_reader_facing` scores a chat turn — see [0096](0096-no-document-unless-asked.md)
- **#84** — a run leaves a document because the request asked — its narrative was replaced by [0096](0096-no-document-unless-asked.md), which supersedes it; PR #89
- **#89** — the PR for #84 — see [0090](0090-document-intent-node.md) and [0096](0096-no-document-unless-asked.md)
- **#100** — open: cross-process job events — the decision to keep them in-process is in [0010](0010-cross-process-ownership.md)
