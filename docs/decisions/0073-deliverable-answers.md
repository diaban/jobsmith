# 0073 — A deliverable answers with the material it has

- **Issue:** #73 · **PR:** #79
- **Status:** accepted · partially supersedes [0058](0058-reader-facing-deliverable.md) · partially superseded by [0082](0082-critique-checks-the-subject.md)
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0059](0059-refusal-as-data.md), [0081](0081-grounding-reaches-reasoning.md)

## From “Agents (`agents/`) — what an agent *is*”

**A deliverable answers, and a refusal is a last resort with a shape** (#73). #58 gave the generator a reader and a list of banned registers; nothing said what the document must *contain*, and the run that showed the difference is the one worth keeping. A comparison of ergonomic chairs: four steps ok, 14.7k characters of sourced specifications from `web_search`, 14.6k of per-item sheets, 8.3k of analysis with real conclusions — and a deliverable whose sections were *Données à collecter et plan de sourcing*, *Plan de vérification*, a literal blank template (`Nom exact du produit :`), closing with an offer to prepare it. `terminal_kind` was `unanswered`. Not one line of it was about chairs: **the deliverable was `critique`'s output rendered as a document**, section for section. Every link in that chain was ours. **The generator gets an obligation, not only prohibitions** (`GLOBAL_GENERATOR_PROMPT`, `DEFAULT_GENERATOR_PROMPT`, and both refiner templates, which restate the same contract): the answer to the request, first, in the subject's own terms, from the material — a model handed prohibitions and no substance fills the gap with what is at hand, which is the working material. **Uncertainty is marked where it bears**, never in a preamble: "capacity given as X, unconfirmed" is an answer, "the values will all need checking" at the top is a refusal wearing an answer's clothes — and that hedge is born one step earlier, so `research.NOTES_SYSTEM` no longer says "flag any uncertainty explicitly" but says where the mark goes and that what the model is sure of must not be marked. **`unanswered` needs a much higher bar**: `NO_ANSWER_INSTRUCTION` now states it — material that says NOTHING about the subject, never material that is partial, indicative or unverified — because a demand for rigour against a pack that flags its own doubt otherwise refuses every time. The #59 machinery is untouched: same marker, same `split_declaration`, same path map, same fail-open. **A refusal has a specified shape** (what was asked, what the material did and did not support, what is known anyway; a few short paragraphs; no work plan, no template, no offer of service) and the same line says outright that it is the one path on which the rules above are lifted — the old prompt forbade exactly the register a refusal needs while asking for a refusal, nothing arbitrated, and the model produced the maximal version of the forbidden thing. **`critique` stops feeding the generator** (`CritiqueCapability.render_context` returned `None`): #58 chose the label over withholding, and this run falsifies that choice for a weak model. It does not become decorative — `render_report` carries it to the human under the same label (the job record, `jobsmith job <id>`, the annexes), and `slide_deck` keeps its own labelled block, because no observed run shows a deck copying the review and changing a decision on analogy rather than on evidence is what put it here in the first place. (**That last claim did not survive the arithmetic**: withholding left a step spending an LLM call per run for a consumer that only exists when a deck is planned — #82 below, which is what put it back in the generator's material by changing what it produces rather than by re-arguing the label.) **`SUBJECT_ONLY_RULE` goes on the generator too**, plus one sentence the material steps do not need: it is the step that actually produces the document, so "not part of the subject" must not read as "ignore the length you were asked for". Measured, not judged: two new checks in `evals/scoring.py` — `report_answers_request` and `refusal_is_bare` — and the llm tier was 100% on the twenty checks that existed before, which is the whole reason this needed new ones.

## From “Evaluating prompts (`evals/`)”

**Two checks read the deliverable's substance, and one of them reads its
input** (#73). `report_reader_facing` catches a document addressed to the
wrong reader; nothing caught a document with no answer in it, which is how
a run scored 100% here while delivering a data-collection plan.
`report_answers_request` asks two things at once, because either alone is
free: the answer names most of the request's own content words *minus the
vocabulary of producing a document* (`compare X and write me a report` is
about X), **and** it carries a third of what the material keeps coming back
to — the half a restatement of the brief cannot fake, which is exactly why
it is measured against the generator's input and not against the request.
`refusal_is_bare` is the twin of `refusal_declared` and scores a shape:
short, no blank form, no plan or offer of service. Both read the *answer*,
never the file — a Reporter's provenance is about the run by design and
quotes the request. The term machinery (`terms`, `frequent_terms`,
`STOPWORDS`, `PROCESS_WORDS`) lives in `deliverable.py` with `normalize`,
and is deliberately blunt: a bag of words with no stemming, thresholded at
a third, so a missed stopword makes a check more lenient and never wrong.
What neither detects is the opposite failure — a generator that dumps its
context scores perfectly on both.
