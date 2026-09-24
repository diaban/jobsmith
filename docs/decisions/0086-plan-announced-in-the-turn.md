# 0086 — The plan is announced in the turn that waits on it, as activity

- **Issue:** #86 · **PR:** #PR
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "A turn is a flow … a seventh, `job_planned` (#86)" → 0050, 0083, 0086; "The plan is activity" (CLI) → 0086
- **See also:** [0083](0083-synchronous-by-default.md), [0050](0050-streamed-turn.md), [0048](0048-terminal-ui.md), [0006](0006-job-notices.md), [0104](0104-notice-names-prior-jobs.md)

## Context

Since #83 a task runs inside the turn and the user watches `… running the
task` for up to 20 s with nothing saying what is running. The plan is the
first thing a run has that is worth showing, and `PlanReady` already persists
it (and publishes a summary) the moment the planner answers (0048). Nothing
brought it into the conversation.

Out of scope, by decision: **interjection** ("skip the critique" means
amending a running plan, the hard half of #5) and **any gate**. This is a
notice; nothing waits on it.

## Decision

**1. What travels: a seventh event, `job_planned {job_id, steps}`.**
`steps` is the plan as the planner left it — `{"capability", "depends_on"}`
per step, plan order, plain lists. It rides the tool's custom channel like
`job_started` (`CUSTOM_JOB_PLANNED`), `chat/runner.py` alone reads it
(`JobPlanned`), and both backings carry the same dict.
- **Not text on an existing channel.** A token is the answer: it goes into
  `Message.content`, the reply `send` / `POST /sessions/{id}/messages` hands a
  caller that waits (0050). A plan written as text would become part of the
  answer, and every front-end would print it in the answer's place.
- **Not a sentence either.** How a DAG reads on one line is wording, and each
  front-end words it (the rule `ToolStarted`/`TOOL_ACTIVITY` set). The line's
  *columns* are shared — `core.state.plan_waves`, built on `plan_depths`, the
  two drawings' own columns — so the line and the DAG never disagree.

**How the tool learns it: the manager's own events, while it waits.**
`launch_job` subscribes to `manager.subscribe()` **before** `start_job` (so the
event that says the plan landed cannot go to nobody), and `announce_plan`
re-reads the job on each event for it; the first read with a plan writes it
and ends the watch. No polling (the event already says the job moved), no
engine change. When the run finishes inside the wait, `None` is put on the
queue and the watch drains up to it, so a run faster than its watcher still
has its plan said, in order, before its answer. The watch is a courtesy: its
exception is retrieved and goes no further; it is cancelled and unsubscribed
in a `finally`.

**2. Where it renders: with the activity, not with the record.**
- **REPL: stderr**, `  … plan: web_search + documents → research → analysis`,
  between `… running the task` and `✓ running the task`. The plan is neither
  what the user handed over (the notice, stdout) nor the answer (stdout): it
  is the engine's decision about how it is working, i.e. what it is doing
  right now, stated precisely — and it is superseded the moment the answer
  lands. A piped transcript keeps the question, what was handed, and the
  answer; a run narrating its own shape is the "state of the work" 0058
  keeps out of the deliverable.
- **TUI: the activity line**, `… running the task: <plan>`, overwritten when
  the task ends. Nothing in the conversation: the panes are switched (F2/F3),
  so the jobs pane's live DAG is not visible from the chat — but it is the
  exact, durable drawing, and the conversation keeps the same record the
  REPL's stdout does. The activity line is where "right now" is said on
  that screen, and it is what the user was staring at.

**3. The promoted case: nothing travels once the turn has stopped waiting.**
The watch lives exactly as long as the wait. A plan that landed before
promotion was already said in the turn — which is where the issue wanted it,
next to "this keeps going in the background". One that lands after (a promoted
run with a slow planner, `$JOBSMITH_SYNC_TIMEOUT=0`, `/bg`) reaches the user
the way the rest of that run's progress does: the progress notice (0006: its
`progress_signature` already includes the plan's size, so the plan landing is
news on the next turn), the jobs pane, `/job`. A message of its own would be a
turn nobody asked for, costing a model call, to deliver something already one
keystroke away.

**4. Shape.** The waves in order, `→` between them, `+` inside one:
`web_search + documents → research → analysis`. A step sits after the longest
chain of what it waits on, whatever order the plan lists it in. **A one-step
plan is not announced**, decided once in the tool rather than in every
front-end: it would repeat the name of the only step and tell nothing the
`job_started` notice did not. A direct answer runs no job and has no plan; a
job whose router answers directly never plans — neither says anything.

## Alternatives and why not

- **The plan as a token / a line in the answer.** Rejected above: it enters
  `Message.content`, and the no-drop, transcript-is-the-reply rule would then
  hand it to every caller that waits.
- **Poll the record.** The event stream already says the job moved; polling
  would be a second mechanism for a fact the first reveals (the same argument
  0083 made against a plan-shape promotion trigger).
- **A hook in the job engine** (a callback from `PlanReady` into the tool):
  couples the engine to the chat; the subscription is already the port.
- **stdout in the REPL, next to the notice.** It would make the plan part of
  the turn's record. It is the run's current state, and stderr is where the
  REPL puts state.
- **A line (or a card update) in the TUI conversation.** Duplicates the jobs
  pane's DAG in a less exact form, in the pane that is the record.
- **The exact edges on one line** (`web_search, documents → research`,
  `research → analysis`, …). Longer, and no easier to read than waves; the
  exact DAG is drawn in two places already. What waves lose — an edge that
  skips a wave reads as depending on the whole previous wave — is stated in
  `plan_waves`.
- **Announcing after promotion** through the completion-notice path: see 3.

## Measured

Tests: `test_the_plan_crosses_either_backing_between_the_notice_and_the_answer`
(local/http), `test_the_plan_is_shown_while_the_run_is_still_going`,
`test_a_run_faster_than_its_watch_still_has_its_plan_said_first` (the first
step is held until the turn has shown the plan, so a late plan hangs rather
than passes), `test_a_one_step_plan_and_a_direct_answer_announce_nothing`,
`test_a_promoted_run_says_nothing_of_its_plan_once_the_turn_is_over`,
`test_the_repl_says_the_plan_as_activity_not_as_the_answer`,
`test_a_plan_reads_as_its_waves`, and in the TUI
`test_the_plan_is_on_the_activity_line_while_the_task_runs` (asserted by name,
while the run is RUNNING). Each break below was applied on purpose, the tests
run, the code restored:

| broken on purpose | failed | which |
|---|---|---|
| tool never writes job_planned | 6 | crosses[local], crosses[http], while-running, faster-than-watch, repl, tui |
| no drain: watch cancelled as soon as the run is done | 1 | faster-than-watch |
| the watch reads a queue the manager never publishes to | 6 | crosses[local], crosses[http], while-running, faster-than-watch, repl, tui |
| subscription not released | 2 | while-running, promoted |
| a one-step plan is announced | 1 | one-step/direct |
| runner drops the payload | 6 | crosses[local], crosses[http], while-running, faster-than-watch, repl, tui |
| runner keeps depends_on as a tuple | 1 | crosses[local] |
| plan sent as a Token | 6 | crosses[local], crosses[http], while-running, faster-than-watch, repl, tui |
| REPL prints the plan on stdout | 1 | repl |
| TUI ignores job_planned | 1 | tui |
| TUI draws it in the conversation | 1 | tui |
| waves = plan order (no depth) | 3 | repl, waves, tui |

Short names: `crosses` is the parity test, `faster-than-watch` is
`test_a_run_faster_than_its_watch_still_has_its_plan_said_first`, the rest
name the tests above. The drain survived every other test until
`faster-than-watch` was written: the fake run never outran its watch, so the
drain was untested code until a lagging queue made it so.

## Consequences

- The port carries seven events; a front-end that ignores `job_planned` loses
  nothing it had.
- The announcement exists only where the waiting process owns the
  `JobManager` — always true today (the tool runs in the daemon or the
  embedded app). Cross-process events (#100) are not needed for it.
- Interjection on the announced plan remains #5.

## Supersedes / superseded by

None.
