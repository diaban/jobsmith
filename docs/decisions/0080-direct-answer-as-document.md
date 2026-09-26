# 0080 — A direct reply asked for as a file is written as that document

- **Issue:** #80 · **PR:** #124
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "A direct reply is told the files the run delivers; asked for as a file, it is written as that document (`DIRECT_DOCUMENT_RULE`)"

## Context

#80 was opened on one red cell: `report_reader_facing` scored a `DirectResponder`
reply ("who are you?") as a deliverable and caught `would you like`. It asked
two things: that the check get the plan guard its neighbours had, and whether a
direct answer that *is* written to a file should hold a deliverable's register.

The first half was settled on the way: #84 stopped the direct route from
writing a file nobody asked for, and #96 moved the answer checks behind
`_answer_applies`, which skips a run with no plan ([0096](0096-no-document-unless-asked.md)).

The second half stayed open. #84 deliberately honours a file asked for in words
whichever route answered, so "what can you do? put the answer in a markdown
file" is answered by `DirectResponder` and written to disk. That prompt asks for
a chat turn and was never told a file exists: not the list of delivered files
the generator gets since #77 ([0077](0077-generator-knows-the-files.md)), nor
who reads it. On gpt-5-nano, the file then read like a turn: `[Team Name]`,
`[Manager Email]` left to fill in; "if you want…, tell me"; once, with the file
missing, `echo 366 > leap_year_days.txt`.

## Decision

`DirectResponder.system_prompt(state)` carries, after the profile's template:

- `delivered_files_note(state)`, always, exactly as the generator gets it — a
  reply that promises or denies a file contradicts something it can see;
- `DIRECT_DOCUMENT_RULE` (`core/generation.py`) **only when `document_formats`
  names a file**: the reply is that document, read apart from the conversation —
  content first, nothing asked back or offered, no placeholders (write around an
  unknown detail), never how to save it.

The rule is in `core/`, not the profile, because it is conditional on the run's
state, like the note it follows; the profile's template is untouched, and so is
a greeting.

The eval follows: `report_reader_facing` scores a direct reply that was written
as a file (gated by `_report_applies` instead of `_answer_applies`), and
`direct_capabilities_as_file` is a golden case for it.

## Alternatives and why not

- **Send a request that wants a file to the planner** (the issue's other reading:
  "a request the planner should have taken"). Measured against: when the planner
  took "give me your list of capabilities as a document", `analysis` answered
  "ChatGPT Capabilities — Shareable Overview", because only `DirectResponder` is
  shown the registry. It also cannot be complete — an empty registry and an
  empty plan reach the direct route anyway.
- **Withhold the file on the direct route.** A second silent decision over how
  triage read the sentence; #84 already refused it.
- **Rewrite the profile's direct template in a deliverable's register.** Wrong
  for the common case: a greeting is a turn and should read like one.
- **Grow `PRODUCER_FACING_MARKERS`** to catch "tell me", "if you want",
  brackets. Out of scope by the issue's own terms; the marker list is a floor.

## Measured

gpt-5-nano, four requests likely to route direct while asking for a file
(capabilities as markdown, as a shareable document, a welcome note as markdown,
the days in a leap year saved to a file), 4 runs each, before and after, markers
counted over the full answer (`[Placeholder]`, would you like, let me know, tell
me, if you want, `echo`, `Out-File`):

| | runs that answered direct and wrote a file | with a marker |
|---|---|---|
| before | 12 | 7 |
| after | 13 | 0 |

The golden case on the llm tier is green before and after (4/4 on
`report_reader_facing`): what failed before is not on the marker list, so the
case pins route and file, and the probe above is the measurement. Triage missed
the case once in the 4 runs after (router prompt unchanged; noise).

Falsified: dropping the rule from the prompt fails
`test_a_direct_reply_asked_for_as_a_file_is_written_as_that_document`.

## Consequences

- A greeting's prompt now also says "Files this run delivers: none" — harmless,
  and the same fact the generator is given.
- Seen during the measurement, left open: `document_intent` twice missed "save
  the answer to a file" (no file written); on the plan route the generator
  recited save-to-file instructions or the file note itself ("File saving note").

## Supersedes / superseded by

None.
