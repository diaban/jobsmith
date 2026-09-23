# 0054 — A report's title is cut on a word

- **Issue:** #54 · **PR:** #71
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0055](0055-document-name-title-format.md)

## From “Jobs layer (`jobs/`)”

**A title is never cut mid-word** (#54): `document_title` (`jobs/report.py`) derives it from the request — whitespace collapsed so a multi-line request cannot break the markdown heading in two, cut at the last word boundary that fits and marked with an ellipsis, `DEFAULT_TITLE` when the request says nothing usable. It was `job.query[:80]`, a raw slice landing wherever the count landed (`…adaptées à un utilisate`) in the one line a reader sees before deciding whether to read the rest. One place decides it and `JobDocument.title` feeds all three Reporters; the day a job carries a title of its own (#55), this stays as the floor under it.
