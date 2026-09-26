# 0110 — Tests are organised by purpose, share builders through one module, and state properties rather than history

- **Issue:** #110 · **PR:** (this one)
- **Status:** accepted — all 5 steps landed (#117–#120, #122)
- **Rule in `CLAUDE.md`:** "Tests are organised by purpose, never by issue" and "Shared builders live in `tests/support.py`"

## Context

`tests/` had grown to 627k characters and 704 collected tests for a 16k-line
package. The ratio (≈ 0.9 test line per source line) was not the problem; the
shape was:

- **Files named after the PR that wrote them.** "What file does a job
  deliver?" was answered in `test_no_document.py` (#84), `test_document_request.py`
  (#55), `test_document_intent.py` (#90), `test_pdf_lazy_probe.py` (#108) and
  `test_delivered_files.py` (#77), plus tests in `test_jobs`, `test_report_html`,
  `test_service`. Each PR pinned its fact at its own layer and nothing merged
  them afterwards, so one fact was asserted three times: e.g. "a job asked for
  markdown writes `artifacts/<id>.md`" in `test_jobs`, `test_no_document` and
  `test_document_request`; "two formats, the first is `main`" in five tests.
- **Test files imported each other** (29 `from test_x import …`): `make_manager`
  lived in `test_jobs.py`, `launch_call`/`make_session` in `test_chat.py`, two
  different `wait_done` in `test_api.py` and `test_cli.py`, `StubPdf` in
  `test_report_pdf.py`, the broken-engine monkeypatches twice.
- **Docstrings narrated issue history** ("Until #96…", "#84 left…"), which since
  0102 is the records' job; ~14 % of the suite's lines are docstrings.

## Decision

1. **`tests/support.py`** holds every builder more than one file needs; no test
   file imports another. `make_manager` grew `document_formats`/`default_formats`
   so the document-step engine is the same builder, not a second one.
2. **One file per component or rule.** The document cluster became
   `test_document_intent.py` (the node: reply table, gate, runner translation)
   and `test_deliverable.py` (what a job delivers: whether, name/title, formats,
   why none, from the chat); the lazy-probe tests joined `test_report_pdf.py`.
3. **Data-only variations are one parametrized test**: the node's 17 reply
   shapes are one table; the direct/plan routes, the four document-step
   replies, the two "run fails after the decision" cases, the docx/pdf chat
   refusals are each one test.
4. **A docstring states the property**, with `→ NNNN` for the why. Before
   deleting narrative, each fact was checked against 0055, 0090, 0096, 0108;
   the two that no record held are kept here:
   - gpt-5-nano's replies to the #90 prompt (#97 review): 1 call in 12 answered
     `requested` **with** `["html"]`, 3 in 12 answered `{"document": "html"}` —
     why names beat the label.
   - a job record written between #55 and #84 has `formats: []` and **no**
     `deliverable_expected` key; that absence is how it is read as "unstated".

## Alternatives and why not

- **Delete tests by count.** The redundancy was ~15 % of the cluster's
  functions, less than first estimated (20–30 %); most tests defend a real rule
  (`None` vs `[]`, fail-open, lazy probe). Cutting harder would cut coverage.
- **Helpers in `conftest.py`.** It already holds fixtures and the fakes of the
  model; builders of the application are a different concern, and an explicit
  `from support import` says where a name comes from.
- **Parametrize the end-to-end layers together** (node + manager + HTTP). The
  layers answer different questions (the pyramid is legitimate); only
  duplicates *within* a layer were merged.

## Measured

- Document cluster: 5 files, 1 489 lines, 66k chars, 77 test functions →
  2 files (618 lines, 27k chars, 33 functions) + 8 functions in
  `test_report_pdf.py`; 7 duplicates in other files removed or merged.
- Whole suite: 627k → 594k characters, 704 → 695 collected cases (cases were
  kept where they were distinct data), 0 cross-test imports.
- Falsified: gating the document step on truthiness instead of `is not None`
  fails `test_it_asks_nothing_when_there_is_nothing_to_decide[[]…]`; writing a
  file for every DONE run fails 5 tests in `test_deliverable.py`; renaming the
  `none` label fails the reply table and the end-to-end `none` case.

## Consequences

Other clusters follow the same treatment in later PRs: chat/service/api
(the parity tests), `test_tui.py` (internal-state assertions, #110's third
point), `test_default_pack.py` (prompt-wording assertions). Records that cite
the removed files (0108, 0109) are history and stay as written.

## Supersedes / superseded by

None.
