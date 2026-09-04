"""Reading a deliverable back, whatever format it was written in.

The report checks in `scoring.py` claim to be layout-independent: they look
for the title, the answer, the job id and the request — not for the headings
`MarkdownReport` happens to use. That claim was only true *within* markdown.
`# ` is not how an HTML report opens, and HTML escapes and wraps the very
text the checks search for, so scoring an HTML run failed on the format
rather than on the run.

This module is the missing step: turn a rendered deliverable into the two
things the checks actually ask about —

    title   what the document announces itself as
    text    everything a reader can see, markup gone

— so a check is written once and holds for every Reporter. A format the
extractor does not know is read as plain text, which degrades to today's
behaviour instead of pretending the document has no content.

Deliberately **not** a parser. It never validates the markup, only strips it,
and it is paired with `normalize()` so that the needle a check searches for
is flattened exactly like the haystack: `- **web_search**` in markdown and
`<li><strong>web_search</strong></li>` in HTML both come out as
`web_search`, which is the only reason one substring test can serve both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

# Markup that carries no text: dropped whole, contents included.
_DROP_BLOCKS = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_H1 = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.I | re.S)
_TITLE_TAG = re.compile(r"<title\b[^>]*>(.*?)</title>", re.I | re.S)

# Markdown markers a reader never sees, so neither should a check: heading
# hashes, list bullets, fences, table pipes, quotes and inline emphasis.
# `_` is left alone on purpose — it is inside `web_search` far more often
# than it is around an emphasised word.
_MD_LINE_PREFIX = re.compile(r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)")
_MD_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_MD_INLINE = re.compile(r"[*`|]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Flatten prose to what a reader would see, on one line.

    Applied to both sides of every containment check — the document's text
    and the needle — so a match no longer depends on which Reporter ran.
    """
    lines = []
    for line in (text or "").splitlines():
        if _MD_RULE.match(line):
            continue
        if line.strip().startswith("```"):
            continue
        lines.append(_MD_INLINE.sub(" ", _MD_LINE_PREFIX.sub("", line)))
    return _WHITESPACE.sub(" ", " ".join(lines)).strip()


@dataclass(frozen=True)
class Deliverable:
    """One rendered report, read as text."""

    format: str
    raw: str
    title: str
    text: str          # normalized: what the checks search

    def contains(self, needle: str) -> bool:
        """Is this text in the document — in any format's rendering of it?"""
        flat = normalize(needle)
        return bool(flat) and flat in self.text


def _from_html(raw: str) -> tuple[str, str]:
    body = _DROP_BLOCKS.sub(" ", raw)
    match = _H1.search(body) or _TITLE_TAG.search(raw)
    title = normalize(unescape(_TAG.sub(" ", match.group(1)))) if match else ""
    return title, unescape(_TAG.sub(" ", body))


def _from_markdown(raw: str) -> tuple[str, str]:
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    title = first[1:].strip() if first.startswith("# ") else ""
    return title, raw


_EXTRACTORS = {"html": _from_html, "markdown": _from_markdown}


def extract(raw: str | None, report_format: str | None) -> Deliverable:
    """Read a rendered deliverable as `(title, searchable text)`.

    An unknown format is read as markdown — plain text with no markup to
    strip is exactly what that does, so a new Reporter scores on its content
    from day one and only its *title* waits for an extractor here.
    """
    fmt = (report_format or "markdown").strip().lower()
    title, text = _EXTRACTORS.get(fmt, _from_markdown)(raw or "")
    return Deliverable(format=fmt, raw=raw or "", title=title, text=normalize(text))
