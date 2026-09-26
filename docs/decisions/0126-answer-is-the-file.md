# 0126 — An answer written to a file is that file, not a text about it

- **Issue:** #126 · **PR:** (this one)
- **Status:** accepted · partially supersedes [0077](0077-generator-knows-the-files.md)
- **Rule in `CLAUDE.md`:** "The generator is told which files the run delivers … When the list names the answer itself, `ANSWER_FILE_RULE` says that entry is the text being written" (Agents)

## Context

Measured while working on #80, on the plan route (gpt-5-nano, default agent):
asked "how many days are there in a leap year? save the answer to a file",
the run wrote the file, and the file talked about itself instead of simply
being the answer — hand-saving instructions (`echo 366 > leap_year_days.txt`,
`open('leap_year_days.txt', 'w')`), "Delivery note — The answer file is
provided separately as a markdown file with this run. It is not reproduced
here", "File saving note — The answer is saved to a file as markdown". Every
eval check was green: the file existed, in the right format, with the answer
in it.

The issue named two instructions working against each other, and the probe
below found a third, upstream of the generator:

1. **The list** (#77) names the document as `this answer itself, written to a
   file as markdown`, and the default pack's prompt said of *every* entry on
   it that it "is delivered separately, not inside this text … never
   reproduce or summarise its contents". True of a deck; false of the
   answer's own file. The self-notes are that bullet obeyed.
2. **`BRIEF_RULE`**, "carry it out, and never restate it", read as "carry out
   the saving".
3. **The material steps.** `SUBJECT_ONLY_RULE` tells `research`, `analysis`
   and `critique` that a request naming "a report, a deck, a page, a
   language" is not the subject — and "save the answer to a file" is none of
   those. In the probe's first run the `echo … >` section was `analysis`'s,
   word for word; the generator, told to use what the material holds,
   delivered it. `research` did the same ("To save this as a file, copy the
   content above into a file named tcp_vs_udp.md").

## Decision

1. **`ANSWER_FILE_RULE`** (`core/generation.py`) follows the list in
   `delivered_files_note` whenever the list names the answer itself: that
   entry is the text being written, saved exactly as written; write the
   document, and say nothing about the file — not that it is saved, how,
   where or in what format. It is in the note, not in a profile, because
   every route that writes the answer to a file reads the note: the
   generator whatever its profile (`DEFAULT_GENERATOR_PROMPT` needed no
   edit), the refiner, and the direct reply — where it sits beside
   `DIRECT_DOCUMENT_RULE` (#80), whose last sentence says the same thing.
2. **The pack's bullet is scoped** (`DELIVERED_FILES_RULE`, now a named
   constant): *any file on it other than this answer itself* is delivered
   separately. This is the part of 0077's decision that changes; the rest
   of it — the list, "none" said plainly, no file by example, name no
   other file — holds as it was.
3. **`SUBJECT_ONLY_RULE` names "a file to save it in"**, and "never advise …
   on how to save it". Same rule, same placement (every material step and the
   generator, #58); one more kind of document instruction is named.
   `BRIEF_RULE` is unchanged: it lists length, structure and language, and
   with the file named in the rule it follows, "carry it out" has nothing to
   do with saving.
4. **An eval check, `answer_is_the_file`**, and a golden case,
   `plan_answer_saved_as_a_file`. The check reads the answer of a run that
   wrote a file (`_report_applies`, so a direct reply written as one is
   scored too) for the two shapes the issue reported — instructions for
   saving it by hand, and a note on its own file — named narrowly
   (`SELF_DELIVERY_MARKERS`). A shape the **request** asks about is not
   scored ("how do I write it with echo?"); the material is no exemption,
   since that is where the instructions were born. Nothing in it says
   "separately" or "delivery note": a deck named as delivered apart, under
   that heading or not, is #77's correct behaviour. It does not see a
   sentence that merely restates the request ("saving the comparison to a
   file is not covered"), which no marker tells apart from the request's own
   words — the keyword fake echoes them verbatim.

## Alternatives and why not

- **Drop the answer from the list.** The list would then say "none" of a run
  that writes a file, and the generator could no longer say "this document"
  truthfully; #77 made the list the whole truth about files on purpose.
- **Say it only in the default pack's prompt.** `DEFAULT_GENERATOR_PROMPT`
  (any other profile) and the refiner would keep the contradiction's second
  half ("carry it out"), and the direct route would be told something
  different from the generator about the same file.
- **Grow `BRIEF_RULE`** ("a file it asks for is written for you"). A third
  place saying the same thing to a model that under-follows multi-rule
  prompts; the fact belongs beside the list it explains, and the upstream
  cause is the material steps, which never see `BRIEF_RULE`.
- **Only the generator.** It is told to use what the material holds; with
  save instructions in the material, the fix would be a tug of war between
  two of its own rules. Fixed where they are born.
- **Tell the material steps that the run writes the file** (`SUBJECT_ONLY_RULE`
  ending "— the run writes any file itself"), to stop them remarking that
  the saving "cannot be done here". Measured on `analysis` alone, ten
  samples a request: 2/10 vs 2/10 (leap year), 1/10 vs 1/10 (TCP/UDP)
  mentions of the saving. No effect, so not added.
- **Grow `PRODUCER_FACING_MARKERS`** with these shapes. It is a floor about
  register, deliberately not grown for style (0080); this is one fact about
  one kind of run, so it is a check of its own.

## Measured

gpt-5-nano (the `.env`'s), default agent, `build_app → create_job → run_job`,
`db="memory"`, `TAVILY_API_KEY` blank (no `web_search`: the defect is in how
the prompts read the request, and the probes were run under a 200k-TPM
limit shared with another session). **Triage was forced to `plan`** in the
probes — the property is the plan-route generator's, and the leap-year
question routes direct half the time; the golden case measures the real
path. Five requests, six runs each, before (`main`) and after:

- **leap** "how many days are there in a leap year? save the answer to a file"
- **tcp** "compare TCP and UDP and save that as a file"
- **heat** "summarise the pros and cons of heat pumps, as a markdown document"
- **deck** (annex control) "make a short slide deck comparing TCP and UDP, and also save a written summary as a markdown file"
- **nofile** (control) "compare TCP and UDP for the networking of a multiplayer game"

Counted over the runs that planned, answered **and wrote a file** — the
document step missed the file twice in each arm (#125's side), and one
after-run on leap ended `unanswered` because its only step, `analysis`,
failed under the rate limit — every hit read by hand. *Instructions*: commands or steps for saving
it (`echo 366 > leap_year_days.txt`, "To save this as a file, copy the
content above into a file named tcp_vs_udp.md"). *Self-note*: the text
describing its own file ("Delivery note — This answer is written to a file
as markdown", "Note about file-saving — This content is prepared to be saved
as a file named tcp_vs_udp_notes.md"). *Restated*: the file request recited
as a limitation ("The request to save the answer to a file cannot be
addressed within the answer content").

| request | before: file runs | instructions | self-note | restated | any | after: file runs | instructions | self-note | restated | any |
|---|---|---|---|---|---|---|---|---|---|---|
| leap | 5 | 2 | 3 | 0 | **5** | 5 | 0 | 0 | 1 | **1** |
| tcp | 5 | 1 | 1 | 1 | **3** | 5 | 0 | 0 | 1 | **1** |
| heat | 6 | 0 | 0 | 0 | **0** | 6 | 0 | 0 | 0 | **0** |
| **all** | 16 | 3 | 4 | 1 | **8** | 16 | 0 | 0 | 2 | **2** |

One-sided Fisher on "any": p ≈ 0.027. `answer_is_the_file` on the same
runs: 7/16 fail before, 0/16 after (p ≈ 0.003) — it misses the restated
shape by design, both before (1) and after (2). In 6 of the 8 before-hits the
generator's material already held the instructions — `analysis` or
`research` had written them — and in all 10 hits, before and after, the
material mentioned the saving.

**Where each part acts**, one model call per sample, so the variable that
changed is the only one:

| replay | before | after |
|---|---|---|
| generator alone, on the before-runs' own material (2 per run: leap, tcp, heat) — any shape, read by hand | 11/32 | 4/32 |
| … leap samples whose material held instructions: the answer copied them | 4/8 | 0/8 |
| … tcp samples whose material held instructions: the answer copied them | 1/4 | 1/4 |
| `analysis` alone on the bare leap request — instructions for saving | 7/11 | 0/9 |
| `analysis` alone on the bare TCP/UDP request — instructions for saving | 2/6 | 0/10 |

The generator's side alone cuts the rate by about two thirds, and does not
reliably refuse instructions the material hands it (tcp: 1/4 either way);
the rest is the material's, which is why `SUBJECT_ONLY_RULE` moved too. The
`analysis` replays lost 10 of 66 samples to rate-limit errors (all in the
first batch, before retries were added); they are excluded, not counted as
clean.

**Controls.**

- *deck*: `answer_invents_no_file` 6/6 pass in both arms, and every answer
  names the deck as a separate file. Both arms also paste the deck's outline
  slide by slide (1/6 before, 6/6 after end to end), which #77 forbids. It is
  not this change: `analysis` designs the slides on the bare request 8/8
  under the old rule and 8/8 under the new one, and the generator replayed
  on identical material reproduces them 6/12 under the old prompt and 6/12
  under the new. Left open (see Consequences).
- *nofile*: `answer_invents_no_file` 6/6 pass in both arms, no marker hit;
  one after-run wrote "No files are delivered with this answer" (the "none"
  note, whose text this change does not touch, recited once in 12).

**Golden case** (`plan_answer_saved_as_a_file`, llm tier, real triage, after
only, 4 runs): routed `plan` 4/4, a markdown file 4/4, `answer_is_the_file`
4/4, 84/84 checks. It pins route, file and shape on the real path; the probe
above is the measurement, as in 0080.

**Falsification** — each break made, the targeted tests run, the file
restored:

| broken on purpose | tests that failed |
|---|---|
| the note without `ANSWER_FILE_RULE` | 6 (note, both profiles' generator, refiner, direct route) |
| `ANSWER_FILE_RULE` on every non-empty list | 1 (`…naming_the_answer…[deck alone]`) |
| the pack's prompt without `DELIVERED_FILES_RULE` | 1 (`test_the_generator_says_how_to_read_the_files_it_is_listed`) |
| `answer_is_the_file` never fails | 7 (the violation row, the six observed shapes) |
| no request exemption | 1 (the requested `echo` command) |
| gated on the answer instead of the file | 1 (the run asked for no file) |

Structural tier: 211/211, `answer_is_the_file` applying on 5 runs (the four
cases that ask for a file, the direct one included, and the new one).

## Consequences

- Every profile's generator, refiner and the direct reply now read the same
  sentence when the answer is a file; a greeting and a run with no file see
  the note exactly as before.
- The direct route carries the idea twice (`ANSWER_FILE_RULE` in the note,
  the last sentence of `DIRECT_DOCUMENT_RULE`). Consistent, and left so:
  trimming a rule measured in 0080 was not worth an unmeasured change.
- `answer_is_the_file` would score a request about writing files in code that
  names no command on the code it rightly gets back; no golden case asks
  that, and such a case would have to be scored without it.
- Left open, seen here: (1) the restated shape — a material step remarks that
  the saving "cannot be done in this environment" and the generator carries
  the remark (2 in 16 after); (2) on gpt-5-nano the material steps still
  design a requested deck despite `SUBJECT_ONLY_RULE` (8/8), and the
  generator pastes that outline about half the time — #58's and #77's rules,
  unchanged by this record; (3) the document step missing "save … to a
  file" is #125.

## Supersedes / superseded by

Partially supersedes [0077](0077-generator-knows-the-files.md): its bullet no
longer says that every listed file is delivered separately. Builds on
[0058](0058-reader-facing-deliverable.md) (`SUBJECT_ONLY_RULE`'s scope and
placement unchanged), [0073](0073-deliverable-answers.md) (`BRIEF_RULE`) and
[0080](0080-direct-answer-as-document.md) (`DIRECT_DOCUMENT_RULE`, unchanged).
