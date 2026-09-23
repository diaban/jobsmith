"""Document intent: which file, if any, the request asked to be left behind.

A dedicated decision node, for the same reason the router is one: what a run
is *for* is decided in a node of its own with an entry in the path map, never
folded into another node's prompt. That is what makes the decision reachable
from every door at once — `jobsmith run`, `/bg`, `POST /jobs` and the chat all
go through this graph, and only the chat has an interpreter in front of it
(#90). A request saying "give me that as a PDF" got markdown everywhere else,
because nothing outside `chat/tools.py` had ever read the sentence.

It answers in the three states `Job.formats` already has, and in no others:

- a list of format names — the request named them, **or it asked for a
  document without naming a format** ("write me a report"), in which case
  the names are the deployment's default (`default_formats`, handed down by
  `app/agent.py` from `$JOBSMITH_REPORT_FORMAT`, #96);
- `[]` — the request said there is to be **no** file;
- nothing at all (`{}`, no write) — the request said nothing, which is the
  ordinary case. Since #96 it means **no file**, on every door: nothing else
  reads the request for a document, so this silence is where that decision
  is taken, and `jobs/manager.py` records it as such.

The fourth answer the prompt offers ("a document, no format named") is not a
fourth state: it resolves to names here, so the record says what will be
written and nothing downstream learns a new value. That is also why the
default travels as names, like `formats` — `core/` never learns what a
Reporter is.

**It fills silence and never overrides.** The gate is structural, decided
before any model call: the graph is entered with `document_formats` seeded
from what the caller already asked for (`jobs/runner.py`), and a value there
means somebody has spoken — the node returns immediately, free and
deterministic, exactly as the router decides an empty registry without asking
a model. Two interpreters that can disagree is the failure mode; a redundancy
is not a harmless one.

**It cannot refuse.** #55 put both document refusals in `create_job`, where
whoever asked is still listening; this node runs mid-run, where nobody is. It
is allowed to run there precisely because it has nothing to refuse: it
chooses from the formats this deployment can actually render, which is handed
to it at composition (`app/agent.py`, `available_formats`), and a name
outside that list is dropped rather than raised on. With no renderable format
at all it never runs.

**It is FAIL-OPEN, like the router**: any LLM error, bad JSON, unknown answer
or unrenderable format degrades to writing nothing — "the request said
nothing". Since #96 that is no longer the state that wrote a file by default:
failing open now costs a document someone asked for in words, where it used
to cost a format. That is the right side to fail on — a missing file is said
on the record (`deliverable_expected`) and the answer is delivered in full
either way (#85), where an invented one is a file nobody asked for — and it
is still the only side this node can fail on without refusing.

What it deliberately does NOT decide: the document's *title* and *name*. A
title is already derived from the request mechanically (#54) and a second,
interpretive derivation beside it would need a rule saying which wins.
"""
from __future__ import annotations

import json

from .deps import Deps
from .profile import DEFAULT_DOCUMENT_INTENT_TEMPLATE
from .state import AgentState

#: The three answers the prompt asks for. Anything else is treated as silence.
NAMED = "named"
REQUESTED = "requested"     # a document, no format named (#96)
NO_DOCUMENT = "none"
UNSPECIFIED = "unspecified"


class DocumentIntent:
    """Reads the request for what document it asked for, and nothing else."""

    DEFAULT_TEMPLATE = DEFAULT_DOCUMENT_INTENT_TEMPLATE

    def __init__(
        self,
        deps: Deps,
        formats: tuple[str, ...] | list[str] = (),
        *,
        default_formats: tuple[str, ...] | list[str] = (),
        prompt_template: str | None = None,
    ):
        self.deps = deps
        # What this deployment can actually render, canonical. Empty by
        # default: a graph composed without it (a test, a hand-assembled
        # builder) cannot know, and a node that cannot name a real format has
        # nothing to say — so it says nothing rather than guessing.
        self.formats = tuple(formats)
        # What "a document" means here when the request named no format —
        # the deployment's answer, handed down like `formats`. Empty leaves a
        # request for an unnamed document silent, i.e. no file: this node
        # does not pick a format on the deployment's behalf.
        self.default_formats = tuple(default_formats)
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    # -------- Prompt rendering --------

    def system_prompt(self) -> str:
        return self.prompt_template.format(
            formats="\n".join(f"- {name}" for name in self.formats)
        )

    # -------- Decision --------

    def _renderable(self, names: object) -> list[str]:
        """The names this deployment can actually write, in the order given.

        The whole of criterion "it cannot refuse": a model naming `docx` here
        is not an error to raise three minutes into a run, it is an answer
        this deployment cannot honour — so it is dropped, and an answer left
        with nothing in it is silence.
        """
        chosen: list[str] = []
        for name in names if isinstance(names, list) else []:
            canonical = str(name).strip().lower()
            if canonical in self.formats and canonical not in chosen:
                chosen.append(canonical)
        return chosen

    def _read(self, answer: str, formats: object) -> list[str]:
        """What a reply that is not "no document" asked for, or `[]` for silence.

        **Formats this deployment can render win over the label.** The label
        is the model's summary of its own answer, and the names are the
        answer; when they disagree, the names are the more specific thing it
        said. Measured on gpt-5-nano with the eval's own sentence ("…give me
        the result as an html page", 12 calls per prompt, #97 review): the
        #90 prompt answered `{"document": "html", "formats": ["html"]}` 3
        times in 12 — a format name as the label, read as silence — and the
        #96 prompt `{"document": "requested", "formats": ["html"]}` once in
        12, which resolved to the default and wrote markdown. One rule reads
        both right, and it is the node's own contract rather than a new
        policy: it never refuses, so it takes what it can honour.

        - `named`, `requested`, or a label that is itself a renderable format
          name, with renderable `formats` → those formats;
        - a label that is a renderable format name with no usable `formats`
          → that format (the label named it);
        - `requested` with nothing renderable → the deployment's default;
        - anything else (`unspecified`, unknown) → silence, even if a list
          came with it: a model unsure whether a file was asked for does not
          get to invent one.
        """
        label_format = self._renderable([answer])
        if answer in (NAMED, REQUESTED) or label_format:
            if chosen := self._renderable(formats):
                return chosen
            if label_format:
                return label_format
        if answer == REQUESTED:
            return self._renderable(list(self.default_formats))
        return []

    # -------- Node --------

    async def run(self, state: AgentState) -> dict:
        # Structural gates, both before any model call. `document_formats` is
        # seeded at entry by `jobs/runner.py` with what the caller asked for;
        # absent (`None`) means nobody has said anything yet, which is the
        # only case this node exists for.
        if not self.formats or state.get("document_formats") is not None:
            return {}
        try:
            raw = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            reply = json.loads(raw)
            answer = str(reply.get("document", "")).strip().lower()
            if answer == NO_DOCUMENT:
                # The label says no file, and a stray list next to it does
                # not overrule an explicit "no document".
                return {"document_formats": []}
            # The most specific thing the model said wins — see `_read`.
            if chosen := self._read(answer, reply.get("formats")):
                return {"document_formats": chosen}
        except Exception:  # fail-open by design, see the module docstring
            pass
        # Silence: the request said nothing this node could read — which,
        # since #96, is the decision "no file". Written as nothing, so the
        # fail-open path and the ordinary one are the same path.
        return {}
