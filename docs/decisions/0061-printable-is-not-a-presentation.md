# 0061 — A printable document is not a presentation

- **Issue:** #61 · **PR:** #68
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0035](0035-capability-artifacts-and-slide-deck.md), [0055](0055-document-name-title-format.md)

## From “Evaluating prompts (`evals/`)”

**`must_include` has a mirror.** `must_exclude` names capabilities the plan
must NOT contain, scored by `plan_excluded_steps` — the only way to measure
a spec that is too easy to reach (#61), since such a run is green end to
end and the user simply opens the wrong kind of file. It skips when none of
the named capabilities is registered, like its mirror. It is exercised in
the *structural* tier by `plan_report_request` excluding `read_files`: the
fake chains the whole registry, and plan validation drops that step as
inapplicable because no file was named — so the deterministic tier measures
the dropping, and a case excluding a step the fake would really plan
(`slide_deck`) is llm-only, or it would measure the fake.
