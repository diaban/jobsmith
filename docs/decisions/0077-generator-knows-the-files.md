# 0077 — The generator is told which files the run delivers, and names no other

- **Issue:** #77 · **PR:** #NN
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "The generator is told which files the run delivers" (Agents) and "`answer_invents_no_file` is the one check read against the record" (Evaluating prompts)

## Context

Job `2afcb24e…` delivered, twice: *« un fichier slide deck existe
parallèlement à ce document, mais son contenu n'est pas reproduit ici »*. The
plan was `web_search → research → analysis → critique`; no deck was planned,
none existed. The issue named two candidate sources and asked which before any
fix: the conversation excerpt attached to every chat launch
(`inputs["conversation"]`), or invention. And it asked for the measurement to
be taken after #96 (a file only when asked) and #85 (the file is title +
answer + one line), which it was.

**The excerpt cannot be the source, by construction.** `inputs["conversation"]`
is read in exactly one place, `Planner.user_message`. No capability and no
generation node reads it; the generator's input is `query` + `merged_context`.
And in every probe run below, the generator's material contained no
deck-ish word at all (recorded per run).

**The generator prompt was.** `GLOBAL_GENERATOR_PROMPT` carried: *"The material
may say that the run produced a file (a slide deck, say). It is delivered
alongside this document, not inside it: mention that it exists if that helps
the reader, and never reproduce or summarise its contents."* The observed
sentence is that bullet paraphrased — *a slide deck*, *alongside this
document*, *its contents not reproduced here* — and every instance measured
below carries the same fingerprint. Meanwhile nothing told the generator which
files existed: a run that produced none left the model a conditional about a
deck, an example of one, and no fact to check it against.

## Decision

1. **The generator is given the list** (`core/generation.py`,
   `delivered_files_note`), between the request and the material, in the
   generator's and the refiner's user message: the requested document with its
   formats (`document_formats`, as decided by the document step or seeded from
   `Job.formats`), plus every file a step declared through `artifact_meta`, in
   plan order — or, when there is none, *"Files this run delivers: none. The
   answer is read as text, with nothing attached to it or delivered alongside
   it"* — closed by *"Name no other file"*. An invented file becomes a
   contradiction of something the model can see, and "no file" is said as
   plainly as a list of two. It lives in `core/` because it is framework state
   and applies to every profile, not only the default one.
2. **No prompt offers a file by example.** The bullet now points at the list:
   a listed file is delivered separately, mention it if it helps, never
   reproduce it, never mention one that is not listed. `slide_deck`'s
   `render_context` said *"delivered alongside this report"* — false since #96
   whenever no report was asked for — and now says *"delivered as a separate
   file"*; its planner description said *"the written report is still
   delivered alongside it"*, now *"the written answer"*.
3. **`KeywordLLM` reads the request as what precedes `FILES_HEADING`**: its
   missing-material words include "attached", which the list itself says.
   Without that the structural tier fell to two false refusals.
4. **An eval check, `answer_invents_no_file`**, applies wherever the run
   answered from a plan (`_answer_applies`) and fails when the answer names a
   kind of file — deck, PDF, spreadsheet, or a generic "attached / alongside
   this document / ci-joint / pièce jointe" — that `Job.outputs` does not hold
   (the harness now records `output_formats`). A kind the request or the
   material already names is not scored: "compare PDF libraries" or a
   datasheet cited as "the PDF" is not a claim about this run's outputs.

## Alternatives and why not

- **Label the excerpt as background for the generator.** It never reaches the
  generator, so there is nothing to label; this was the issue's first
  hypothesis and the code falsifies it before any run does.
- **Only delete the example from the prompt.** Removes the seed but leaves the
  model with no fact about files; a weaker model generalises "a produced file
  may exist" on its own. The list costs one short paragraph per call.
- **Put the list in the system prompt.** The system prompt is the profile's
  and is static; the list is per run, so it goes where the per-run facts go.
- **Score the check against the case** (the evals rule). The case says what
  the request warranted; whether the prose tells the truth about what exists
  is a fact on the record, and `Job.outputs` is the only place it is written.
  Hence the one stated exception in `CLAUDE.md`.
- **A broad file vocabulary** (`slides`, `annexe`, `file`). Real runs write
  *"les travaux annexes"* and *"the price slides"*; only words that name a
  file as something the reader has are listed, and a test pins the ordinary
  ones as passing.

## Measured

Model: `gpt-5-nano` (the `.env`'s), default agent with `web_search` (Tavily) and
`slide_deck` registered, `build_app → create_job → run_job`, `db="memory"`.
One subject (air-to-water heat pump vs gas condensing boiler, 120 m², cold
climate) in English and French, four shapes:

- **a** silent request + a conversation excerpt that discusses a slide deck
  and a printable PDF;
- **b** the same request, no excerpt;
- **c** the request + "write it up as a document";
- **d** the request + "deliver it as an HTML page".

Seven repeats per shape and language before, seven after. Scored with
`answer_invents_no_file` (every hit also read by hand); "planned answers" are
runs that answered from a plan (the only ones the check applies to; some
before-runs escalated on OpenAI rate limits at 12-way concurrency, re-run at 6).

| shape | before: invented / planned answers | after |
|---|---|---|
| a — excerpt about a deck | 1 / 12 | 0 / 14 |
| b — silent, no excerpt | 0 / 12 | 0 / 13 |
| c — document in words | 1 / 12 | 0 / 13 |
| d — named format | 0 / 13 | 0 / 14 |
| **all** | **2 / 49 (4 %)** | **0 / 54** |

The two before-hits: *"A slide deck with the 120 m² case exists alongside this
document"* (a, EN, no output at all) and *"Aucun contenu de diaporama n'est
reproduit ici"* after *"Remarque sur les contenus livrés avec ce document"*
(c, FR, output: the markdown file alone). Neither shape predicts it — (a) ≈ (c),
and (c) has no excerpt — so the source is not the excerpt; both are the
prompt's own bullet. At this rate the end-to-end comparison is not significant
on its own (2/49 vs 0/54).

**Generator-only A/B**, to get power on the variable that changed: the 26
materials the after-runs handed the generator for runs with no file, each sent
to the generator three times with the old prompt and no list, and three times
with the new prompt and the list.

| arm | invented / answers |
|---|---|
| old prompt, no list | AB_OLD |
| new prompt + list | AB_NEW |

**Falsification** — each break made, the suite run, the file restored:

| broken on purpose | tests that failed |
|---|---|
| generator's user message without the list | 1 (`test_the_generator_sees_the_list_…`) |
| refiner's user message without the list | 1 (`test_the_refiner_sees_the_same_list`) |
| the note always says "none" | 2 |
| the "(a slide deck, say)" bullet put back | 1 (`test_the_generator_prompt_offers_no_file_by_example`) |
| `answer_invents_no_file` never fails | 2 (the violation row, the verbatim #77 sentence) |
| no exemption for a kind the request/material names | 2 (the borrowed-word cases) |
| `KeywordLLM` scans the whole message for "attached" | 5 (structural tier, …) |

Structural tier: 176/176, `answer_invents_no_file` applying on 8 runs.

## Consequences

- Any profile's generator now sees the list, including banking's; it is
  neutral English and names no domain.
- A declared annex that turns out to be missing is still listed (the step did
  write it as far as the run can tell); the job layer drops it later and says
  so in `job.error` (#41).
- The check does not catch a request that asked for a deck the run then failed
  to make and an answer that claims it anyway — that is #55's side, where the
  request names the kind.
- Seen in passing, not addressed here: one after-run wrote *"conformément à
  votre demande de ne pas proposer de plan de calcul"* — the generator's own
  rules restated to the reader as the user's request; the #58/#73 register,
  not a file.

## Supersedes / superseded by

None. Builds on [0058](0058-reader-facing-deliverable.md) (the bullet it
replaces was #58's), [0073](0073-deliverable-answers.md),
[0085](0085-answer-in-the-conversation.md) and
[0096](0096-no-document-unless-asked.md) (a run may deliver no file at all,
which is why "none" has to be said).
