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
from pptx.util import Emu, Length

from .slides import Deck, Slide

TITLE_LAYOUT = 0            # "Title Slide" in the default template
CONTENT_LAYOUT = 1          # "Title and Content"

# 16:9, the size PowerPoint itself creates — the stock python-pptx template is
# 10 × 7.5 in, i.e. 4:3, which has been the default nowhere else for fifteen
# years and is pillarboxed on any screen a deck is shown on (#62). Note the
# HEIGHT is the template's own: PowerPoint's 16:9 is the same 7.5 inches with
# a wider canvas, so only the width really changes.
SLIDE_WIDTH = Emu(12192000)     # 13.333 in
SLIDE_HEIGHT = Emu(6858000)     # 7.5 in


class PptxRenderer:
    """`DeckRenderer` producing a .pptx file, in memory."""

    extension = "pptx"
    #: The page. Class attributes rather than constants inlined below, so a
    #: deployment that wants 4:3 back — or A4 landscape — is a subclass with
    #: two numbers and no copied render logic.
    slide_width: Length = SLIDE_WIDTH
    slide_height: Length = SLIDE_HEIGHT

    async def render(self, deck: Deck) -> bytes:
        # to_thread for the same reason `LocalArtifactStore.write` uses it:
        # capability waves run in parallel and this is blocking CPU work, so
        # the other branches keep moving while a deck is assembled.
        return await asyncio.to_thread(self.build, deck)

    def build(self, deck: Deck) -> bytes:
        # `pptx.Presentation` is a factory function; the class it returns lives
        # in `pptx.presentation`, which is what the annotations above name.
        presentation = pptx.Presentation()
        self._set_page(presentation)
        self._title_slide(presentation, deck)
        for slide in deck.slides:
            self._content_slide(presentation, slide)
        buffer = BytesIO()
        presentation.save(buffer)
        return buffer.getvalue()

    # -------------------- the page --------------------

    def _set_page(self, presentation: Presentation) -> None:
        """Resize the deck, and move the template's placeholders with it.

        Setting `slide_width` alone is the trap, and it is why this is six
        lines rather than one: the stock template positions its placeholders
        for a 10-inch canvas, and they do not follow. Measured on the naive
        version — a body placeholder ending at 8,686,800 EMU on a 12,192,000
        EMU slide, a 3.8-inch gutter down the right of every slide, which
        reads as a deck someone left-aligned by mistake.

        So the horizontal geometry of the master and of every layout is scaled
        by the same ratio the slide grew by. Proportional, so a margin stays a
        margin and the columns of the multi-content layouts keep their gaps;
        vertical geometry is untouched because the height did not change. The
        slides added afterwards inherit from these, so nothing per-slide needs
        to know the deck is not 4:3.
        """
        was = presentation.slide_width
        presentation.slide_width = self.slide_width
        presentation.slide_height = self.slide_height
        if not was or was == self.slide_width:
            return
        ratio = self.slide_width / was
        sources = [presentation.slide_master, *presentation.slide_layouts]
        # Read every value first, write after. A layout placeholder with no
        # geometry of its own reports the MASTER's, so scaling in one pass
        # scales those a second time — measured, 457,200 → 812,800 EMU and a
        # body 2.6 inches wider than the slide it sits on. Writing the scaled
        # value onto such a placeholder only materialises what it was
        # inheriting anyway.
        geometry = [(ph, ph.left, ph.width)
                    for source in sources for ph in source.placeholders]
        for placeholder, left, width in geometry:
            if left is not None:
                placeholder.left = Emu(round(left * ratio))
            if width is not None:
                placeholder.width = Emu(round(width * ratio))

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
