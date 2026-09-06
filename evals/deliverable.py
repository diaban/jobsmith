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

— so a check is written once and holds for every Reporter. A *text* format
the extractor does not know is read as plain text, which degrades to today's
behaviour instead of pretending the document has no content. A format whose
file is bytes has no such reading at all, and `ensure_readable` refuses it
before a suite runs rather than letting it score zero.

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

from jobsmith.jobs.report import is_binary_format

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

    Every *text* format is covered: an unknown one is read as markdown —
    plain text with no markup to strip is exactly what that does, so a new
    text Reporter scores on its content from day one and only its *title*
    waits for an extractor here.

    A format whose file is bytes is not covered and never arrives: there is
    no text in it to score, so `ensure_readable` refuses it at the entry.
    """
    fmt = (report_format or "markdown").strip().lower()
    title, text = _EXTRACTORS.get(fmt, _from_markdown)(raw or "")
    return Deliverable(format=fmt, raw=raw or "", title=title, text=normalize(text))


def ensure_readable(report_format: str) -> str:
    """The scored format, or a refusal saying why it cannot be scored.

    Asked of the *main* deliverable — the only file the checks ever read — at
    the entry of a suite run, before a case is composed. A binary deliverable
    fails every report check for a reason that has nothing to do with the
    agent, and the run it would store is worse than the wasted minute: the
    report format is deliberately not part of `load_baseline`'s notion of
    comparable, so that record becomes the next run's baseline and prints a
    regression nobody caused.

    Loud, like `make_reporter` on an unknown name and `BinaryDeliverable` on
    `get_report`: a refusal naming what to do instead beats a number that
    means nothing. `is_binary_format` is the lookup, so what counts as text
    stays defined once, by the Reporters themselves.
    """
    fmt = (report_format or "").strip().lower()
    if is_binary_format(fmt):
        raise ValueError(
            f"cannot score a {fmt} deliverable: the report checks read the "
            f"deliverable as text, and a {fmt} file is bytes — every one of "
            f"them would fail for a reason that is not about the agent. Score "
            f"a text format instead: --report-format markdown, or "
            f"markdown,{fmt} to write the {fmt} file as well."
        )
    return fmt
