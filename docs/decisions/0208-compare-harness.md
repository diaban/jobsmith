# 0208 — The DAG and the ReAct baseline are compared on run-time-width tasks, same material and bounds, scored on coverage

- **Issue:** #208 · **PR:** see the issue · **Status:** accepted
- **Rule in `CLAUDE.md`:** "`make compare`: DAG vs `react` baseline…" (Evaluating prompts); partially supersedes [0206](0206-react-baseline.md) (hits per search)

**Context.** Step −1 of `docs/design/compiler-v1.md`: every compiler step is measured against the baseline of 0206 on one task set, or "compiled is better" stays a claim. The golden set (`evals/cases.py`) scores the DAG's own decisions (route, plan, file) and cannot score a graph agent.

**Decision.** `evals/compare.py` (`make compare`) runs each case through `default` and `react` with one provider and one `DocumentSource`: a local fixture (7 site notes, 2 distractors) written into a scratch root, so both read the same documents, reproducibly, with no web. Success = answered and every `must_mention` term present as a whole word; per run: calls, tokens, cost when priced, wall time. Cases: all 7 sites (width 7), the 4 solar ones (a filter: width known only after reading), one fact (control). **One search of the baseline reads what one query of the DAG reads** (`MAX_HITS == DocumentsCapability` `per_query`, 6, pinned by a test): 0206 had 10, `max_documents`, a whole step's total.

**Alternatives and why not.** A model judging answers: a model grading models, and the first question (did it read everything) needs none. The web: not reproducible, not the same pages twice. The golden set's checks: they read the DAG's record.

**Measured.** gpt-5-nano (`.env`), n=3, unpriced so no cost. First run, baseline at 10 hits: DAG 0/3 on 7 sites (Glenmoor, 7th of 7 tied hits, cut at 6 every time), baseline 3/3: retrieval, not orchestration. Aligned at 6: both 0/3 on 7 sites (same miss; the baseline searched once), both 3/3 on solar and on the control. Per run, DAG vs baseline: 8.0 vs 2.0 calls, 21.4k vs 5.7k tokens, 106 vs 30 s (7 sites); 6.0 vs 2.0 calls, 9.7k vs 3.2k tokens, 56 vs 11 s (control). Setting `MAX_HITS` back to 10 fails the pin; matching terms as substrings fails the whole-word test.

**Consequences.** The baseline to beat: equal coverage at a quarter of the DAG's tokens and a fifth of its time. The 7-site case fails for both until something reads past one page (step 2's `map`, or queries per item). One weak model, n=3: noise both ways. Results stay local (`evals/results/`, gitignored); this record keeps the numbers.
