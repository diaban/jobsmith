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
asked about a subject and does not know a DAG exists. Naming the audience is
what turns working material into a document; saying that `critique` is
evidence and never voice is what keeps the review out of the reader's hands.
"""
from __future__ import annotations

from ...core.profile import AgentProfile

GLOBAL_GENERATOR_PROMPT = (
    "You are writing the final deliverable of a background job.\n"
    "Who reads it: the person who made the request. They were not part of the "
    "work, do not know which steps produced this, and want the subject they "
    "asked about — not a report on the work that was done for them.\n"
    "Use ONLY the material provided below (research notes, analysis, "
    "critique). Write in the language of the request.\n"
    "- Structure it as a short written report: a direct answer first, then the "
    "supporting sections that matter.\n"
    "- The material is working material, written for the run and not for the "
    "reader. The critique in particular reviews the work itself: use it as "
    "evidence — correct what it corrects, drop what it undermines — and never "
    "let it become the shape or the voice of the document.\n"
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
    "- If the material is insufficient, say so plainly."
)

GLOBAL_REFINER_TEMPLATE = (
    "The deliverable you produced failed validation.\n"
    "Validation issues: {issues}\n"
    "Rewrite it, fixing these issues and keeping to the provided material. "
    "Same reader as before: the person who made the request, who was not part "
    "of the work. No citation markers, no closing questions, nothing addressed "
    "to whoever produced the document."
)

DEFAULT_APP_PROFILE = AgentProfile(
    generator_system_prompt=GLOBAL_GENERATOR_PROMPT,
    refiner_prompt_template=GLOBAL_REFINER_TEMPLATE,
)
