"""SLIDE_DECK capability: ask the model for a deck, render it, hand back a file.

**Why this is a capability and not a Reporter.** A report is prose — a title,
the answer, its provenance — and a Reporter *serializes* that document: it
does not think. A deck is a different document: sections, one idea per slide,
bullets a room can read at a glance, speaker notes. Slicing a report's
headings into slides mechanically produces bad decks; asking the model for a
deck-shaped structure is **another generation**. Generations belong on this
side of the line, where the tokens they burn are attributed to a step and
booked in the usage ledger like every other call.

What it leaves behind is an **annex** (`core/artifacts.py`): the job's main
deliverable is still the report, and this is one more thing the run produced.
So the capability writes through the `ArtifactStore` port — it names a *file*,
never a path — and declares what it wrote in its result's `meta`.

`DeckRenderer` is the port, and it lives here — next to its one consumer —
rather than in a central `ports/` package: a port is shaped by the need that
declares it, and `pptx` is one adapter behind it (`pptx_deck.py`), not the
contract. That is also what makes the capability testable without
python-pptx installed, and what makes the registration rule literal — with no
renderer, nothing backs the capability and it stays out of the registry.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langgraph.constants import END

from ...core.artifacts import ArtifactRef, ArtifactStore, artifact_meta
from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.deps import LLMClient
from ...core.state import CapabilityResult

# Bounds on what the model is allowed to hand back. A deck is a document a
# human presents: past a dozen slides or half a dozen bullets it stops being
# one, and an unbounded structure is also an unbounded file.
MAX_SLIDES = 12
MAX_BULLETS = 6
MAX_CHARS = 300          # per bullet / per title — a slide is not a paragraph
MAX_NOTES_CHARS = 1200
MAX_MATERIAL_CHARS = 12000


# ---------------------------------------------------------------- the deck

@dataclass(frozen=True)
class Slide:
    """One slide: a claim, what supports it, and what the presenter says."""

    title: str
    bullets: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Deck:
    """A deck as a *structure* — the thing the model is asked to produce.

    Deliberately not a file and not markup: rendering it is the renderer's
    business, and the capability's own reasoning stays readable in the state
    and in the result it emits.
    """

    title: str
    subtitle: str = ""
    slides: tuple[Slide, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form — this travels through the checkpointer and the store."""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "slides": [
                {"title": s.title, "bullets": list(s.bullets), "notes": s.notes}
                for s in self.slides
            ],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Deck:
        """Read a deck back from anything, keeping only what is usable.

        Fed straight from an LLM reply, so every field is treated as a
        suggestion: wrong types are dropped, over-long text is cut, and a
        slide with no title and no bullets is not a slide.
        """
        if not isinstance(data, dict):
            return cls(title="")
        slides: list[Slide] = []
        for raw in _as_list(data.get("slides")):
            if len(slides) == MAX_SLIDES:      # bounded on what is KEPT, so a
                break                          # junk entry costs no slide
            if not isinstance(raw, dict):
                continue
            bullets = tuple(
                _clean(b, MAX_CHARS) for b in _as_list(raw.get("bullets"))[:MAX_BULLETS]
                if _clean(b, MAX_CHARS)
            )
            title = _clean(raw.get("title"), MAX_CHARS)
            if not title and not bullets:
                continue
            slides.append(Slide(title=title or "…", bullets=bullets,
                                notes=_clean(raw.get("notes"), MAX_NOTES_CHARS)))
        return cls(
            title=_clean(data.get("title"), MAX_CHARS),
            subtitle=_clean(data.get("subtitle"), MAX_CHARS),
            slides=tuple(slides),
        )


class DeckRenderer(Protocol):
    """The port: a deck structure, as the bytes of a presentation file.

    `extension` is what the file is called (the capability names its file and
    knows nothing about where it lands). Async because a renderer that is a
    remote service is a legitimate second adapter — the local one just moves
    the CPU work off the event loop.
    """

    extension: str

    async def render(self, deck: Deck) -> bytes: ...


# ---------------------------------------------------------------- capability

class DeckState(CapabilityBaseState, total=False):
    deck: dict           # the structure the model produced (JSON-safe)
    via_fallback: bool   # the structure was salvaged from prose, not designed
    path: str            # what the store returned
    failure: str         # why this step cannot finish


class SlideDeckCapability(Capability):
    """Turn the job's material into a slide deck, and leave the file behind."""

    spec = CapabilitySpec(
        name="slide_deck",
        description=(
            "produce a slide deck file (a presentation: title, one idea per "
            "slide, bullets and speaker notes) from the material the other "
            "steps produced — plan it when the request asks for slides, a "
            "deck or a presentation, and plan it LAST, after the steps that "
            "produce that material; the written report is still delivered "
            "alongside it, so this is never a substitute for analysis"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "deck": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                        "slides": {"type": "array", "items": {"type": "object"}},
                    },
                },
                "path": {"type": "string"},
            },
        },
    )

    DESIGN_SYSTEM = (
        "Design a slide deck about the SUBJECT of the request, for the person "
        "who asked and the room they will show it to. They were not part of "
        'the work that produced it. Return JSON: {"title": "<deck title>", '
        '"subtitle": "<one line>", "slides": [{"title": "<slide title>", '
        '"bullets": ["<short phrase>", ...], "notes": "<what the presenter '
        'says>"}, ...]}.\n'
        f"- Between 4 and {MAX_SLIDES} slides, ordered as an argument: what "
        "this is about, the substance, then what follows from it. One idea "
        "per slide.\n"
        f"- 3 to {MAX_BULLETS} bullets per slide, short phrases a room reads "
        "at a glance — never sentences, never paragraphs.\n"
        "- Use ONLY the provided material. If it is thin, make fewer slides "
        "rather than padding them.\n"
        "- Each block of material is labelled with what it is. A block that "
        "reviews the WORK is evidence: correct the slides with it, drop what "
        "it undermines, and never give it a slide of its own.\n"
        "- Every slide is about the subject. None is about the state of the "
        "work, what is still missing, what remains to be done, options for "
        "the reader to choose between, or a template to fill in; no slide "
        "asks the reader for input.\n"
        "- notes: two or three sentences the presenter says over that slide, "
        "carrying the detail the bullets left out.\n"
        "Write in the language of the request. No prose, no markdown, JSON only."
    )

    #: Where the deck's content comes from, in priority order, and **what
    #: each block is** — `(capability, data key, what this material is)`.
    #: Fixed rather than derived from `results`, which arrives in wave order
    #: (a consumer must never iterate that, see `core/state.py`).
    #:
    #: The third field is the fix for #58. `critique` is agent-facing by
    #: design — it reviews the work, not the subject — and a deck handed that
    #: block under a bare `[critique]` tag rendered it faithfully: two slides
    #: of gaps and next steps, shown to someone who asked about the subject.
    #: It stays in the material because a review that says a claim is
    #: unsupported is worth knowing before it reaches a slide; what changes is
    #: that the block now says what it is, and `DESIGN_SYSTEM` says what to do
    #: with a block of that kind.
    MATERIAL: tuple[tuple[str, str, str], ...] = (
        ("analysis", "analysis", "findings about the subject"),
        ("research", "notes", "research notes about the subject"),
        (
            "critique",
            "critique",
            "an internal review OF THE WORK, not of the subject — evidence "
            "only, never the subject of a slide",
        ),
    )

    def __init__(
        self,
        llm: LLMClient,
        artifacts: ArtifactStore,
        renderer: DeckRenderer,
        *,
        max_material_chars: int = MAX_MATERIAL_CHARS,
    ):
        self.llm = llm
        self.artifacts = artifacts
        self.renderer = renderer
        self.max_material_chars = max_material_chars

    # -------------------- Nodes --------------------

    def _material(self, state: DeckState) -> str:
        """Everything upstream worth putting on a slide, in a fixed order.

        Each block is labelled with what it *is*, not only with the step that
        wrote it: the deck is the one deliverable that reads this material
        directly, with no generation between it and the reader (#58).
        """
        results = state.get("results", {})
        blocks: list[str] = []
        for name, key, role in self.MATERIAL:
            result = results.get(name)
            if not result or not result.get("ok"):
                continue
            # every CapabilityResult key is NotRequired: a failed step has no
            # `data` at all
            text = (result.get("data") or {}).get(key)
            if text:
                blocks.append(f"[{name} — {role}]\n{text}")
        if not blocks:
            return "(no upstream material — build the deck from the request alone)"
        return "\n\n".join(blocks)[: self.max_material_chars]

    async def design(self, state: DeckState) -> dict:
        """Ask for a deck-shaped structure. This is the generation, not a render."""
        # `.get()`: job_id is NOT Required on CapabilityBaseState — a graph
        # driven outside a job has none. Checked BEFORE the model call: a deck
        # is a file, and there is nowhere to keep it, so composing one would
        # only spend tokens on something nobody could ever open.
        if not state.get("job_id", ""):
            return {"failure": "no job to write a deck for: a deck is a file and a "
                               "run outside a job has nowhere to keep it"}
        try:
            raw = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.DESIGN_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Request: {state['query']}\n\n"
                                   f"Material:\n{self._material(state)}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
        except Exception:
            raw = ""
        deck = Deck.from_dict(_loads(raw))
        if deck.slides:
            return {"deck": deck.to_dict(), "via_fallback": False}
        salvaged = _deck_from_prose(state["query"], raw)
        if salvaged is None:
            return {"failure": "the deck design produced nothing to put on a slide"}
        return {"deck": salvaged.to_dict(), "via_fallback": True}

    async def render(self, state: DeckState) -> dict:
        """Structure → bytes → a file the job can hand over."""
        # Both are guaranteed by `design`, which the router only leaves for
        # here once it has written a deck (and it refuses to run without a job).
        deck = Deck.from_dict(state.get("deck") or {})
        job_id = state.get("job_id", "")
        try:
            data = await self.renderer.render(deck)
            path = await self.artifacts.write(job_id, deck_filename(self.renderer), data)
        except Exception as e:
            # No path came back, so there is nothing to declare: a ref to a
            # file that may or may not exist is exactly what the manager
            # refuses to record (#40).
            return {"failure": f"the deck could not be written: {e}"}
        return {"path": path}

    async def emit_success(self, state: DeckState) -> dict:
        # Reached only through `route_after_render` == "success", i.e. with a
        # path the store returned; `design` wrote the deck before that.
        deck = state.get("deck") or {}
        path = state.get("path", "")
        title = str(deck.get("title") or "") or "Slide deck"
        return self._emit_success(
            {"deck": deck, "path": path},
            meta={
                **artifact_meta(ArtifactRef(path, title=title)),
                "slide_count": len(deck.get("slides") or []),
                "via_fallback": bool(state.get("via_fallback")),
            },
        )

    async def emit_failure(self, state: DeckState) -> dict:
        """Report the failure — with the file, if one already reached disk.

        Nothing between the store's write and this node can fail today, so the
        `meta` arm below is a guarantee about tomorrow rather than a live path:
        the day a node is added after `render`, the deck it wrote must still be
        declared, or it is a file recorded nowhere (#41).
        """
        detail = state.get("failure") or "the deck step produced nothing"
        path = state.get("path", "")
        return self._emit_failure(
            detail, meta=artifact_meta(ArtifactRef(path)) if path else None)

    # -------------------- Routers --------------------

    def route_after_design(self, state: DeckState) -> Literal["render", "failure"]:
        return "render" if state.get("deck") else "failure"

    def route_after_render(self, state: DeckState) -> Literal["success", "failure"]:
        return "success" if state.get("path") else "failure"

    # -------------------- Rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        """For the model: that a deck exists, and nothing it can copy out.

        It used to hand over the outline, on the reasoning that knowing the
        deck's shape is what lets the answer refer to it. Measured on a real
        run (#58), the answer did not refer to the deck — it *transcribed* it:
        a section listing the ten slide titles, then the same list again as a
        summary, and with them the deck's own register ("Prochaines étapes")
        inside a document whose prompt forbids exactly that. The report and
        the deck are two documents about one subject, not one quoting the
        other, so what travels is that the annex exists and how big it is.

        `render_report` still shows the outline: that is the provenance
        section, written for a human who wants to know what the run produced.
        """
        deck = Deck.from_dict((result.get("data") or {}).get("deck"))
        if not deck.slides:
            return None
        return (f"# Slide deck produced\n\n**{deck.title or 'Untitled'}** — "
                f"{len(deck.slides)} slides, delivered alongside this report.")

    def render_report(self, result: CapabilityResult) -> str | None:
        """For the human: the file it produced, and what is on it."""
        if not result.get("ok"):
            return f"_{result.get('error') or 'no detail'}_"
        data = result.get("data") or {}
        deck = Deck.from_dict(data.get("deck"))
        path = str(data.get("path") or "")
        if not deck.slides:
            return "_no deck was produced_"
        outline = "\n".join(f"{i}. {s.title}" for i, s in enumerate(deck.slides, 1))
        header = f"**{deck.title or 'Untitled'}** — {len(deck.slides)} slides"
        if path:
            header += f" · `{path}`"
        return f"{header}\n\n{outline}"

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(DeckState)
        g.add_node("design", self.design)
        g.add_node("render", self.render)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("design")
        g.add_conditional_edges("design", self.route_after_design, {
            "render": "render",
            "failure": "emit_failure",
        })
        g.add_conditional_edges("render", self.route_after_render, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()


# ---------------------------------------------------------------- helpers

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean(value: Any, limit: int) -> str:
    """One line of slide text: a string, trimmed, bounded."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit].strip()


def _loads(raw: str) -> Any:
    """The model's reply as JSON, tolerating the wrappers models add.

    Lenient by design, like `documents` and `research`: a reply that is JSON
    inside a code fence, or JSON with a sentence in front of it, is a deck the
    user can have. Anything else returns None and the caller falls back.
    """
    text = (raw or "").strip()
    if not text:
        return None
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _deck_from_prose(request: str, text: str) -> Deck | None:
    """Last resort: cut prose into slides.

    This is exactly the mechanical slicing #35 argues produces bad decks —
    which is why it is the fallback and not the design. It exists so a reply
    that came back as prose still leaves a file built from real material
    rather than nothing; the result says `via_fallback` so the difference is
    never invisible.
    """
    blocks: list[tuple[str, list[str]]] = []
    for chunk in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        heading = _HEADING.match(lines[0])
        title = _clean(heading.group(1) if heading else lines[0], MAX_CHARS)
        bullets = [
            _clean(m.group(1) if (m := _BULLET.match(ln)) else ln, MAX_CHARS)
            for ln in lines[1:]
        ]
        blocks.append((title, [b for b in bullets if b][:MAX_BULLETS]))
    slides = [Slide(title=t, bullets=tuple(b)) for t, b in blocks if t or b][:MAX_SLIDES]
    if not slides:
        return None
    return Deck(title=_clean(request, MAX_CHARS) or "Slide deck",
                subtitle="", slides=tuple(slides))


def deck_filename(renderer: DeckRenderer) -> str:
    """The file this capability asks the store to keep (a name, never a path)."""
    return f"{SlideDeckCapability.spec.name}.{renderer.extension}"


__all__ = [
    "Deck",
    "DeckRenderer",
    "Slide",
    "SlideDeckCapability",
    "deck_filename",
]
