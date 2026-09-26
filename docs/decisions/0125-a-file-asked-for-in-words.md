# 0125 — A file asked for in words is a document asked for

- **Issue:** #125 · **PR:** #TBD
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "A file asked for in words is asked for: "save it to a file" is `requested` whatever else the request asks, a file it is only about is not (`FILE_REQUEST_RULE`)" (Graph flow, under Document intent)

## Context

Measured during #80 on gpt-5-nano: a request that asks in so many words for
its answer to be written to a file ("save the answer to a file", "write it to
a file", "summarise … and save that as a file") was sometimes read by the
document step as `{"document": "unspecified"}`. `document_formats` stayed
`None`, the run wrote no file, and nothing on the record said one had been
asked for: `status: done`, `terminal: answer`, `job.error: null`. The issue's
probe missed 2 runs in 15; the probe below, at the node, 43 in 698 (6.2%).

`DEFAULT_DOCUMENT_INTENT_TEMPLATE` gave the model three reasons to miss it:
the `"requested"` shape was illustrated by "write me a report", "put it in a
document" and "something I can print", never by "a file", the most literal way
to ask for one; `"unspecified"` is "the right one whenever you are unsure"
(#96's deliberate bias); and "a summary … is asking for an answer, not for a
file" sat next to it, which "summarise X and save that as a file" matches
word for word. Of the 43 misses on `main`, 42 were labelled `unspecified`.

## Decision

The document step's prompt carries a named rule, `FILE_REQUEST_RULE`
(`core/profile.py`, composed into `DEFAULT_DOCUMENT_INTENT_TEMPLATE`):

- asking for the answer to be saved, written or put in a file is asking for a
  document, **whatever else the request asks for**: `"requested"`, or
  `"named"` if it names a format this deployment renders;
- a file the request is **only about**, or gives to be read, is not one.

The `"requested"` shape also lists "save it to a file" among its examples. The
summary rule, the `"unspecified"` bias and the node's code are unchanged:
the rule takes a request that names a file out of the uncertain zone rather
than making that zone smaller, so #96's contract (silence gets no file) is
untouched. The second sentence is that contract's side of the same rule: the
word "file" alone ("how do I save a file in vim?", "the file I sent") asks for
nothing to be left behind.

`KeywordLLM` recognises "to a file" / "as a file" as a document asked for,
the same crude way it recognises "a report": a destination, never the bare
word. Three golden cases: `plan_summary_saved_as_a_file` (both tiers, the
summary-plus-file collision), `trivial_fact_saved_to_a_file` (llm tier, the
issue's own request, no route claimed since the step runs before triage) and
`plan_file_is_the_subject` (both tiers, `expect_document=False`, a file the
request is about).

## Alternatives and why not

Measured on the same probe (below), 40 runs per request each:

- **The rule alone** (`v1`, no example in the `"requested"` shape): 7 misses
  in 420 (1.7%), controls 1 in 660. Indistinguishable from the chosen prompt
  within noise; the shape that defines `"requested"` naming "a file" is the
  issue's first cause, and it costs four words.
- **The rule plus "on its own" in the summary rule** (`v2`: "a summary … is
  asking for an answer, not for a file: on its own, that is unspecified"): 4
  misses in 420 (1.0%), controls 2 in 660. No better, and it weakens #96's
  sentence for every compound request that names no file, which is where the
  controls' false positives already come from (below). The collision is
  resolved from the file's side instead ("whatever else the request asks
  for"), leaving the summary rule as strong as it was.
- **Soften `"unspecified"` ("whenever you are unsure")**. That is #96's
  fail-open bias toward no file, and it is right for vague requests; the
  issue is about a request that names a file, which a named rule takes out of
  the uncertain zone without moving it.
- **Match "file" in code** (a keyword gate before or after the model). It
  cannot tell "save that to a file" from "how do I save a file in vim?" or
  "the file I sent", which is exactly the distinction the second sentence of
  the rule draws; the node reads the request through a model for that reason
  (#90).
- **Treat `"named"` with no renderable format as `"requested"`** (2 of the 7
  misses after the fix were `{"document": "named"}` with no format). It would
  also turn "as a docx" into the default format here, the substitution the
  prompt forbids and #90 ruled out; left as is.

## Measured

gpt-5-nano, the `DocumentIntent` node composed as `build_app` composes it
(formats offered: html, markdown; default: markdown), called directly on each
request and read back as `document_formats`; a call that hit the rate limit is
excluded, not counted as silence. Cells are runs that decided a file / runs.
`main` pools up to four probe rounds (fewer for the rows added later), the
fix two. The fix's prompt quotes "save it to a file", "write that to a
file" and "as a file"; the rows marked *held out* use phrasings it does not
quote.

| request | expects | `main` | fix |
|---|---|---|---|
| how many days are there in a leap year? save the answer to a file | file | 76/79 | 50/50 |
| what is the boiling point of water at sea level? write it to a file | file | 75/79 | 49/50 |
| summarise the difference between TCP and UDP, and save that as a file | file | 74/80 | 49/50 |
| list the planets of the solar system and save it to a file | file | 75/79 | 49/50 |
| explain what a hash table is, and put that in a file | file | 70/80 | 49/50 |
| *held out:* give me the common HTTP status codes in a file I can keep | file | 76/82 | 50/50 |
| *held out:* what are the SOLID principles? I'd like the answer as a file | file | 75/79 | 50/50 |
| *held out:* draft a short checklist … and drop it into a file | file | 39/40 | 49/50 |
| *held out:* explain the CAP theorem; keep it in a file for me | file | 36/40 | 48/50 |
| *held out:* give me a packing list for a weekend hike, saved as a file | file | 40/40 | 50/50 |
| golden `plan_summary_saved_as_a_file` | file | 19/20 | 50/50 |
| compare REST and GraphQL for a small team and export it | *(no claim)* | 69/78 | 46/50 |
| how many days are there in a leap year? | no file | 0/82 | 0/50 |
| what is the boiling point of water at sea level? | no file | 0/82 | 0/50 |
| summarise the difference between TCP and UDP | no file | 0/81 | 0/50 |
| hello there | no file | 0/82 | 0/50 |
| golden `plan_survey_with_failure_modes` (summarise RAG approaches) | no file | 1/80 | 0/50 |
| golden `plan_compare_and_recommend` | no file | 2/82 | 1/50 |
| golden `plan_tradeoff_analysis` | no file | 0/20 | 1/50 |
| golden `plan_named_file_grounds_the_steps` (the note I named) | no file | 4/82 | 0/50 |
| golden `unanswerable_missing_material` (the attached report) | no file | 1/80 | 1/50 |
| research … locking, and summarise the trade-offs (the golden case minus its file) | no file | 0/20 | 2/50 |
| how do I save a file in vim? | no file | 0/81 | 0/50 |
| what is a file system, and how does it store a file on disk? | no file | 0/80 | 0/50 |
| how do I export a pandas dataframe to a CSV file? | no file | 0/79 | 0/50 |
| what is the maximum file size on FAT32? | no file | 0/40 | 0/50 |
| compare JSON and YAML as file formats for configuration | no file | 0/40 | 0/50 |
| golden `plan_file_is_the_subject` (ext4/btrfs store a file) | no file | 0/20 | 0/50 |
| summarise the file I sent you and list its main points | no file | 0/40 | 1/50 |
| just answer here, no file: what is the capital of Peru? | none | 0/79 | 0/50 |

- **Requests that name a file: 43 missed in 698 (6.2%) → 7 in 550 (1.3%).**
  The held-out phrasings move with the quoted ones (15 missed in 281 → 3 in
  250), so the rule is read, not matched.
- **Controls: 8 given a file in 1150 (0.7%) → 6 in 900 (0.7%).** Unchanged in
  rate. The false positives predate the fix and sit on compound
  research/compare requests read as `"requested"`; per request the counts move
  both ways within noise (4/82 → 0/50 on the named-note case, 0/20 → 2/50 on
  the locking pair). No request whose file is the subject or the material
  crossed more than once in 50.
- "export it" names no file and the rule does not claim it; it is shown for
  what it is.

The llm tier on the three new golden cases, 5 repeats each: EVAL_TABLE

The structural tier: 220/220 on 17 runs (was 15 runs before the two new
structural cases).

Falsified: composing the template without `FILE_REQUEST_RULE` fails
`test_a_file_asked_for_in_words_is_a_document_asked_for`; removing "to a
file" / "as a file" from `KeywordLLM.DOCUMENT_WORDS` fails
`document_as_requested` on `plan_summary_saved_as_a_file` in the structural
tier (29/30).

## Consequences

- A request that names a file now gets one on every door, in the deployment's
  default format, whichever route answers it; on the direct route that reply
  is written as the document (0080).
- Seen, left open: the controls' ~0.7% false positives (a compound
  research/compare request read as `"requested"`) are #96's leak and predate
  this; and a request for "a file" is sometimes answered `"named"` with both
  offered formats, so the main output is html where the default is markdown
  (7 of 50 on "I'd like the answer as a file"; 8 of 79 on `main`).

## Supersedes / superseded by

None. Extends [0090](0090-document-intent-node.md) and
[0096](0096-no-document-unless-asked.md) without changing either.
