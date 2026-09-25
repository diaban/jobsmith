# 0109 — Fewer themes, fewer hammer rounds, a `slow` marker

- **Issue:** #109 · **PR:** #110
- **Status:** accepted
- **Rule in `CLAUDE.md`:** falsify with the targeted test, `make test-fast` while iterating, the full suite once before the PR (Working on this repo)

## Context

672 tests, 83–93 s wall. Three files carried most of it: the TUI's colour test
rendered all 25 registered themes (12.2 s) though it only needs to catch two
failure modes (a stylesheet that fails to parse, a markup role that resolves
to nothing); `test_sqlite_concurrency.py` ran 5 real processes × 150 rounds
against `_ImmediateBegin` (≈12 s across the store/saver cases) with no
record of why 150; `test_cross_process.py` crosses a real process boundary
for two properties (≈4 s). Agents falsify a regression by re-running the
*whole* suite, so this cost is paid on every falsification of every
delegated task, not once in CI.

## Decision

**Theme sweep** (`tests/test_tui.py::test_every_registered_theme_resolves`):
split the registration check (`set(THEMES) <= available_themes`, one mount,
no render loop) from the render check, and render only the themes that carry
*distinct* risk, each its own `pytest.mark.parametrize` case so a failure
names the theme:

- `ember-dark`, `ember-light`, `tide-dark`, `tide-light` — the only themes
  this project authors, so the only ones a typo in `themes.py` can break.
- `ansi-dark`, `ansi-light` — the two built-ins with a reduced variable set
  (no `$foreground-disabled`; `$surface-active` misspelt `$surface-actrive`),
  which is why `ANSI_GAP` exists at all.
- `textual-dark` — standing in for the other 20 built-ins, which all ship the
  same variable set Textual defines; one of them is what "a foreign theme
  still parses and resolves" needs to prove, and the other 20 do not each
  carry a different answer to that question.

**SQLite hammer** (`tests/test_sqlite_concurrency.py`): `ROUNDS` dropped from
150 to 10, measured (below).

**A `slow` marker**, registered in `pyproject.toml` (`--strict-markers` also
turned on, so a typoed marker fails collection rather than silently matching
nothing): every process-spawning test, the hammer, the one test that starts a
real uvicorn server, the theme sweep, and every other test measured over
roughly a second that a real subprocess, a real job run through the graph, or
a real per-keystroke UI turn makes inherently that slow. `make test-fast`
(`-m "not slow"`) is the inner loop; `make check` and CI keep running `test`,
unfiltered.

## Alternatives and why not

- **Cut `PROCESSES` instead of, or as well as, `ROUNDS`** in the hammer: not
  measured here, out of scope for this issue, and `PROCESSES = 5` is what
  #10/#63 measured the failure against — changing it would need its own
  falsification.
- **Delete the excluded themes from the sweep** rather than list a subset:
  the point of the test is that *every registered* theme resolves; a
  representative sample inside one parametrized test says which themes were
  chosen and why, where deleting coverage would just look like fewer themes
  shipped.
- **One loop over the reduced theme list, not `parametrize`**: would still
  save the render time, but a failure would only name the theme in the
  assertion text, not in the test id — parametrize costs one app mount per
  theme (7 instead of 1) but that is small next to the render time saved by
  dropping 18 themes.

## Measured

Theme sweep: 25 themes in one test, 12.24 s → 7 parametrized cases,
~0.8–0.95 s each (~6 s total) plus a near-free registration check. Both
falsifications still fire: an invented markup role
(`MARKUP_ROLES = (..., "$this-role-does-not-exist")`) failed all 7 cases by
name; an invented CSS variable in `JobsmithApp.CSS` failed all 7 with
`UnresolvedVariableError` at parse time. Both reverted before commit.

Hammer round count — `_ImmediateBegin`'s effect removed locally (`BEGIN` left
un-upgraded to `BEGIN IMMEDIATE`), the `store` case run 10 times at each
count:

| rounds | fix removed | fix restored |
|---|---|---|
| 1   | 10/10 fail | 10/10 pass |
| 2   | 10/10 fail | — |
| 3   | 10/10 fail | — |
| 5   | 10/10 fail | 10/10 pass |
| 10  | 10/10 fail | 10/10 pass |
| 150 (previous) | 10/10 fail | (known-good) |

The race is decided by the first batch every process runs at the shared,
synchronized `start_at` — a deferred `BEGIN` reading a snapshot another
process is about to commit past — not by how many rounds follow, so round
count bought no reliability past the first one. 10 was kept rather than 1 so
the test still runs a real loop, not a single write; the fix restored before
committing. `tests/test_sqlite_concurrency.py::test_concurrent_processes_write_one_file_without_locking_out[store]`
dropped from 5.88 s to ~4.3 s (`[saver]` similarly), the remainder being the
subprocess start-up and the 4 s synchronization stagger this change did not
touch.

Suite total (rebased on `main` after #108 landed, which is what this PR
ships against): 689 tests / 76.13 s (pytest) / 88.9 s wall for the full,
unfiltered run (`make test`/`make check`/CI unchanged in what they run).
`make test-fast` (`-m "not slow"`): 650 tests, 22.9 s (pytest) / 28.4 s wall
— no `weasyprint` import anywhere in that run (confirmed by its absence from
the warnings summary, which fires from inside the library's own module body
on import).

**A real `weasyprint` import moves to whichever test is first to need it, and
that took real chasing.** #108 stopped `build_app` from probing the engine
eagerly, but several tests still exercise the *real* engine deliberately (a
render only pango/cairo can do correctly is not something a stub proves), and
the ~4 s import is paid once per test **process**, by whichever of those
tests pytest reaches first — so marking one specific test `slow` only moves
the tax to the next unmarked one, not off the `test-fast` run. Chased down to
two sources and fixed at the root of each rather than the symptom:

- `tests/test_report_pdf.py` and `tests/test_pdf_lazy_probe.py` share one
  `requires_pdf` marker (skip when `.[pdf]` is not installed); every test it
  guards turned out to genuinely invoke the real engine (`import weasyprint`
  directly, or `PdfReport.render`/`.write`), so `requires_pdf` itself now
  also applies `slow` (`requires_pdf = lambda func:
  pytest.mark.slow(_skip_without_pdf(func))`) — one definition, so a new
  `@requires_pdf` test is `slow` without a second decorator to remember.
- `tests/test_evals.py`'s structural tier (`structural_run`/
  `html_structural_run` fixtures, and the CLI-driven
  `test_cli_runs_and_gates_the_structural_tier`) resolves at least one golden
  case onto a PDF format even on `provider="fake"`, and #108's document step
  proves that choice by really loading the engine before it lets the run
  keep it. Marked `slow`; root-causing *which* case resolves to `pdf` and
  whether it should is #108/#90 territory, out of scope here (this PR does
  not touch `core/document.py`, `jobs/report*.py`, `app/`, or the PDF probe).

## Consequences

`make test-fast` is the loop CLAUDE.md now tells an agent to falsify with;
`make check`/CI still gate on the full, unfiltered suite, so nothing that
proved a property before this issue stopped proving it — only how often the
slow third of the suite is paid for changed. A theme added to `themes.py`
needs a line in `REPRESENTATIVE_THEMES` (`tests/test_tui.py`) only if it is
ours or otherwise known to ship a reduced variable set; a foreign Textual
theme is covered by `textual-dark` standing in for the rest, and by
`test_every_theme_is_registered` for being in the picker at all.

## Supersedes / superseded by

None.
