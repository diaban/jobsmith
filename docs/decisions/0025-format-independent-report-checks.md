# 0025 — The report checks are format-independent for real

- **Issue:** #25 · **PR:** #27, #47
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.

## From “Evaluating prompts (`evals/`)”

**The report checks are format-independent, and now really are.** They read
the deliverable through `deliverable.extract(text, format)`, which returns
the title and the visible text with the markup stripped, and compare needle
and haystack after the same `normalize()` — so `- **web_search**` and
`<li><strong>web_search</strong></li>` are one string. Before that they were
markdown-shaped while the docstring claimed otherwise (`# ` is not how HTML
opens; escaping moved the strings they searched for), and the golden set
scored 13 checks lower in HTML purely on the format. `--report-format html`
scores the other Reporter, and `tests/test_evals.py` pins the two runs to
identical per-check tallies. An unknown *text* format is read as plain text:
a new text Reporter is scored on its content from day one, only its *title*
needs an extractor here. A format whose file is **bytes** has no such
reading, so it is refused at the entry (`deliverable.ensure_readable`, the
lookup being `is_binary_format` — evals grows no second list of what is
text) rather than scoring ~2/10 on the format and storing a record the next
run would pick up as its baseline (#45). It is asked of the *first* format,
which is the `main` output and therefore the only file the checks read, so
`--report-format markdown,pdf` still scores markdown honestly.
