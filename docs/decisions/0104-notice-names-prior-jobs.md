# 0104 — The job notice names the earlier jobs a run builds on, by short id and the start of their query

- **Issue:** #104 · **PR:** #105
- **Status:** accepted · closes the gap recorded in [0074](0074-prior-job-references.md)
- **Rule in `CLAUDE.md`:** "The approval card is a notice (`job_started`: query, sources, from_jobs, …)" → 0083, 0104
- **See also:** [0083](0083-synchronous-by-default.md), [0060](0060-requests-name-files.md), [0055](0055-document-name-title-format.md), [0050](0050-streamed-turn.md)

## Context

The `job_started` notice (#83) shows what a run is handed before it runs: the
reformulated query, the files it may open (#60), the document it will write
(#55). `from_jobs` (#74) was a fourth thing a run can be handed and was not
shown — recorded in 0074 as a gap, not a decision. A model that picked the
wrong one of three recent jobs, or a prefix that resolved to another, started
a run nothing on screen described.

## Decision

`launch_job`'s payload carries `from_jobs`: one `{"job_id", "query"}` per
referenced job, in the order given, and `[]` when none. It rides the same path
as `sources`: `JobStarted` and `Proposal` (`chat/runner.py`), the event dict,
both backings — a list of fresh `dict[str, str]`, never a tuple, so the local
and HTTP backings answer the same JSON.

- **What is shown per job: the short id and the start of its query.** The
  short id is what `/job`, `/cancel` and the jobs pane use; the query is what
  the user recognises — an id alone names a job nobody remembers by id.
- **The query is cut on a word at 60 characters** (`REFERENCE_QUERY_MAX`,
  through `document_title`, ellipsis when cut). The notice names the job; it
  does not restate it, and three referenced jobs at full length would bury the
  query the notice exists to show.
- **The full id travels**; shortening is the renderer's job, as for
  `job_started.job_id`.
- **The tool builds it** from the `Job` `_find` already returned: no second
  lookup, no front-end reaching into the job store to label a reference.
- **Nothing is rendered when nothing is referenced** — unlike `writes`, where
  "no file" is itself a decision, an empty "builds on" line would read as a
  claim that something was referenced.
- One renderer per front-end, shared by notice and proposal: `job_lines`
  (REPL: `builds on: job 01234567 — compare the two chairs`, one line per job)
  and `_job_body` (TUI: `builds on job 01234567 — …`, query `escape`d).

## Alternatives and why not

- **Ids only** (the input as it stands): cheapest, but an 8-hex id is not how
  anyone remembers a job; the issue asks for recognition.
- **The job's title or document name**: most jobs since #96 have neither.
  The query is the one thing every job has.
- **The full query**: long, and a wrong reference is caught at a glance, not
  by reading.
- **A pre-formatted string per job**: would fix one front-end's layout in the
  payload and leave the TUI unable to escape the query apart from its markup.

## Measured

Tests: `test_the_jobs_a_run_builds_on_cross_either_backing` (local/http ×
notice/proposal), `test_the_repl_names_the_jobs_a_run_builds_on`,
`test_the_card_names_the_jobs_a_run_builds_on` (notice/proposal). Each break
below was applied on purpose and failed those tests:

| broken on purpose | failed |
|---|---|
| tool payload omits `from_jobs` | 6 |
| query not cut (limit 1000) | 4 |
| `JobStarted` drops `from_jobs` | 3 |
| `Proposal` drops `from_jobs` | 3 |
| `_job_references` returns a tuple | 2 (local backing) |
| TUI does not `escape` the query | 2 |
| TUI omits the line | 2 |
| REPL prints a line when nothing is referenced | 1 |
| REPL prints the full id | 1 |

## Consequences

The notice now shows everything `launch_job` hands a run besides the
conversation excerpt. A layout snapshot does not change: the snapshot runs
reference no job, and nothing is drawn then.

## Supersedes / superseded by

Closes the gap written into [0074](0074-prior-job-references.md) ("What the
notice does NOT yet carry is the reference itself").
