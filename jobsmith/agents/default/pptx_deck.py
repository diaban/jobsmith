"""PowerPoint as a `DeckRenderer`.

The adapter behind the port `slides.py` defines, and the only module in the
tree that imports `python-pptx` (extra `.[pptx]`). Nothing in
`SlideDeckCapability` knows this format exists: a second renderer — Google
Slides, a reveal.js bundle — is another adapter, not a rewrite.

It uses the stock template's own layouts (title slide, then title-and-content)
rather than building shapes by hand: the placeholders carry the theme's fonts,
sizes and colours, so a deck opens looking like a deck instead of like text
dropped on a white rectangle. Speaker notes go where PowerPoint puts them —
the notes pane — because that is the half of a deck a reader never sees on the
slide.
"""
from __future__ import annotations

import asyncio
from io import BytesIO
from typing import cast

import pptx
from pptx.presentation import Presentation
from pptx.shapes.autoshape import Shape
from pptx.slide import Slide as PptxSlide

from .slides import Deck, Slide

TITLE_LAYOUT = 0            # "Title Slide" in the default template
CONTENT_LAYOUT = 1          # "Title and Content"


class PptxRenderer:
    """`DeckRenderer` producing a .pptx file, in memory."""

    extension = "pptx"

    async def render(self, deck: Deck) -> bytes:
        # to_thread for the same reason `LocalArtifactStore.write` uses it:
        # capability waves run in parallel and this is blocking CPU work, so
        # the other branches keep moving while a deck is assembled.
        return await asyncio.to_thread(self.build, deck)

    def build(self, deck: Deck) -> bytes:
        # `pptx.Presentation` is a factory function; the class it returns lives
        # in `pptx.presentation`, which is what the annotations above name.
        presentation = pptx.Presentation()
        self._title_slide(presentation, deck)
        for slide in deck.slides:
            self._content_slide(presentation, slide)
        buffer = BytesIO()
        presentation.save(buffer)
        return buffer.getvalue()

    # -------------------- slides --------------------

    def _title_slide(self, presentation: Presentation, deck: Deck) -> None:
        slide = presentation.slides.add_slide(presentation.slide_layouts[TITLE_LAYOUT])
        # the body placeholder of this layout is the subtitle line; an absent
        # subtitle takes the placeholder with it (see `_set_body`)
        _set_title(slide, deck.title or "Slide deck")
        _set_body(slide, [deck.subtitle] if deck.subtitle else [])

    def _content_slide(self, presentation: Presentation, slide_spec: Slide) -> None:
        slide = presentation.slides.add_slide(presentation.slide_layouts[CONTENT_LAYOUT])
        _set_title(slide, slide_spec.title)
        _set_body(slide, list(slide_spec.bullets))
        if not slide_spec.notes:
            # reading `notes_slide` CREATES one: a slide with nothing to say
            # must not get an empty notes page
            return
        notes = slide.notes_slide.notes_text_frame
        if notes is not None:
            notes.text = slide_spec.notes


def _set_title(slide: PptxSlide, text: str) -> None:
    """Fill the layout's title placeholder, if it has one."""
    title = slide.shapes.title
    if title is not None:
        title.text_frame.text = text


def _set_body(slide: PptxSlide, lines: list[str]) -> None:
    """Fill the layout's body placeholder with one paragraph per line.

    `python-pptx` has no "add bullet" call: a bullet IS a paragraph of the
    body placeholder, and the layout supplies the glyph. The first line reuses
    the paragraph the empty placeholder already has, or PowerPoint shows a
    blank bullet above the text.
    """
    # cast: `has_text_frame` is the runtime guard, and the type of
    # `placeholders` (a shape that may have no text) cannot express it.
    body = next(
        (cast(Shape, p) for p in slide.placeholders
         if p.placeholder_format.idx != 0 and p.has_text_frame),
        None,
    )
    if body is None:                       # a layout with nowhere to put them
        return
    if not lines:
        # An empty placeholder is not invisible: PowerPoint shows its "click to
        # add text" prompt to whoever opens the file. A slide with nothing to
        # bullet is a slide with no body.
        element = body.element
        element.getparent().remove(element)
        return
    frame = body.text_frame
    frame.text = lines[0]
    for line in lines[1:]:
        frame.add_paragraph().text = line


__all__ = ["PptxRenderer"]
