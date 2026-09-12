"""Profile of the global agent — prompts fitted to the default capability pack.

The core defaults are written for retrieval-style capabilities (they ask for
inline [doc_id] citations). The default pack is LLM-only: there are no
documents to cite, so the model invents markers like "[Context]" on every
line. This profile states what the material actually is and asks for a
written deliverable rather than a chat turn.

**And it says who reads it** (#58). Every step upstream of this prompt writes
for the *run*: `research` produces notes, `analysis` produces findings, and
`critique` is agent-facing by design — "challenge the material, suggest
improvements" is a memo about the work, addressed to whoever will redo it.
Handed to a generator that was told only to answer, that material comes back
out as its own shape: a deliverable whose sections were *Contraintes de
livrable*, *État actuel et risques*, *Prochaines étapes*, given to someone who
asked about a subject and does not know a DAG exists.

**And it says what the document must contain** (#73). Naming the audience and
banning a register was not enough, and the run that showed it is worth
keeping: four steps produced 14.7k characters of sourced specifications, 14.6k
of per-item sheets and 8.3k of analysis with real conclusions, and the
deliverable was `critique`'s methodology review rendered as a document — a
data-collection plan and a blank template, closing with an offer to prepare
the template. Not one line of it was about the subject. A model given
prohibitions and no obligation fills the gap with the best-structured thing it
can see, so the obligation is now stated first, and `critique` is no longer
one of the things it can see (`CritiqueCapability.render_context`).
"""
from __future__ import annotations

from ...core.profile import NO_ANSWER_INSTRUCTION, AgentProfile
from ._step import SUBJECT_ONLY_RULE

GLOBAL_GENERATOR_PROMPT = (
    "You are writing the final deliverable of a background job.\n"
    "Who reads it: the person who made the request. They were not part of the "
    "work, do not know which steps produced this, and want the subject they "
    "asked about — not a report on the work that was done for them.\n"
    "What it must contain: the answer to their request, first, in the "
    "subject's own terms — the things, the figures, the findings the material "
    "actually holds — then the supporting sections that matter. A document "
    "that describes what would be needed in order to answer has not answered.\n"
    "Use ONLY the material provided below, and use what it holds: state what "
    "it establishes. Write in the language of the request.\n"
    "- Structure it as a short written report: a direct answer first, then the "
    "supporting sections that matter.\n"
    "- Where the material is partial, indicative or unverified, mark the doubt "
    "on the statement it bears on ('capacity given as X, unconfirmed'), never "
    "as a preamble that disqualifies everything below it. A qualified answer "
    "is an answer.\n"
    "- Write about the subject only. No section on the state of the work, what "
    "is still missing, what would be needed to go further, or options for the "
    "reader to choose between; no placeholders or templates to fill in; no "
    "requests for input or confirmation.\n"
    "- The material may say that the run produced a file (a slide deck, say). "
    "It is delivered alongside this document, not inside it: mention that it "
    "exists if that helps the reader, and never reproduce or summarise its "
    "contents section by section.\n"
    "- Do NOT add citation markers: the material has no sources to cite.\n"
    "- Do NOT end with questions or offers of further help — this is a "
    "document, not a chat turn.\n"
    # The pack's shared rule, plus the one sentence the generator needs that a
    # material-producing step does not (#73): it is the step that actually
    # produces the document, so "not part of the subject" must not read as
    # "ignore the length the request asked for".
    + SUBJECT_ONLY_RULE
    + " Any length, structure or language it asks for is still yours to "
    "honour: carry it out, and never restate it in the prose.\n"
    # Appended, not folded into the bullets above: the declaration is the
    # framework's protocol (#59), and a profile that wants it says so by
    # adding this one line rather than by re-wording it. It also carries the
    # bar and the shape of a refusal (#73) — including the sentence that says
    # it is the one path on which the rules above are lifted, which is why
    # nothing here says "if the material is insufficient, say so plainly".
    + NO_ANSWER_INSTRUCTION
)

GLOBAL_REFINER_TEMPLATE = (
    "The deliverable you produced failed validation.\n"
    "Validation issues: {issues}\n"
    "Produce a corrected version, fixing these issues and keeping to the "
    "provided material. Same reader as before: the person who made the "
    "request, who was not part of the work. It must still answer that request "
    "first, in the subject's own terms, from the material. No citation "
    "markers, no closing questions, nothing addressed to whoever produced the "
    "document."
)

DEFAULT_APP_PROFILE = AgentProfile(
    generator_system_prompt=GLOBAL_GENERATOR_PROMPT,
    refiner_prompt_template=GLOBAL_REFINER_TEMPLATE,
)
