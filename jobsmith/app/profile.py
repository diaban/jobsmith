"""Profile of the global agent — prompts fitted to the default capability pack.

The core defaults are written for retrieval-style capabilities (they ask for
inline [doc_id] citations). The default pack is LLM-only: there are no
documents to cite, so the model invents markers like "[Context]" on every
line. This profile states what the material actually is and asks for a
written deliverable rather than a chat turn.
"""
from __future__ import annotations

from ..core.profile import AgentProfile

GLOBAL_GENERATOR_PROMPT = (
    "You are writing the final deliverable of a background job. Use ONLY the "
    "material provided below (research notes, analysis, critique). Write in "
    "the language of the request.\n"
    "- Structure it as a short written report: a direct answer first, then the "
    "supporting sections that matter.\n"
    "- Do NOT add citation markers: the material has no sources to cite.\n"
    "- Do NOT end with questions or offers of further help — this is a "
    "document, not a chat turn.\n"
    "- If the material is insufficient, say so plainly."
)

GLOBAL_REFINER_TEMPLATE = (
    "The deliverable you produced failed validation.\n"
    "Validation issues: {issues}\n"
    "Rewrite it, fixing these issues and keeping to the provided material. "
    "No citation markers, no closing questions."
)

DEFAULT_APP_PROFILE = AgentProfile(
    generator_system_prompt=GLOBAL_GENERATOR_PROMPT,
    refiner_prompt_template=GLOBAL_REFINER_TEMPLATE,
)
