# 0003 — Prompt changes are scored, not eyeballed

- **Issue:** #3 · **PR:** #17
- **Status:** accepted · partially superseded by [0085](0085-answer-in-the-conversation.md) · partially superseded by [0096](0096-no-document-unless-asked.md)
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0081](0081-grounding-reaches-reasoning.md), [0073](0073-deliverable-answers.md), [0025](0025-format-independent-report-checks.md)

## From “Evaluating prompts (`evals/`)”

**Properties, not expected text.** Nothing asserts an answer. The checks are
structural: the plan names only registered capabilities, is duplicate-free,
acyclic and has satisfiable dependencies; an obviously simple message is
triaged `direct` and a compound one `plan`; the run reaches the expected
terminal; every planned step ran and reported ok; the deliverable has a title,
the answer and the job that produced it. `scoring.py` holds one
function per property, each returning pass / fail / **skip** — skipped checks
leave the denominator, so a direct-route case never dilutes the plan checks.

**Two tiers.** `structural` runs on `KeywordLLM` — no key, no variance — and
is expected to be 100%: `tests/test_evals.py` asserts exactly that, so a
prompt edit that breaks the machinery fails `make check`. `llm` needs a real
provider, tolerates variance and **never gates CI** (`python -m evals` exits
non-zero only on the deterministic fakes — the provider is the condition, not
the tier label — or with an explicit `--fail-under`).
A case declares which tiers it is meaningful in: one a keyword fake would pass
or fail *by accident* is llm-only, otherwise the fake is what gets measured.
**The fake tier is only deterministic if no key reaches the process**, and
`python -m evals` loads `.env` (which `make worktree` copies) with
`os.environ.setdefault` — so `env -u TAVILY_API_KEY` is refilled from the
file and `web_search` quietly joins every plan (measured: 147/160 instead of
153/153, the extra checks being web material the fake cannot trace). Blank
the keys instead: `TAVILY_API_KEY= ANTHROPIC_API_KEY= OPENAI_API_KEY=`.

**The harness runs the real product path** (`harness.py`): `build_app` →
`create_job` → `run_job`, persistence forced to `memory`, reports into a
scratch dir, a `KeywordChatModel` injected only so composition does not go
looking for chat extras. The triage decision is read back from the
checkpointer (`graph.aget_state`), because a planner rescuing a message the
router should have sent direct is invisible from the outside. The
generator's **merged context** is read back from the same snapshot (#73):
a `Job` records what came out, and "was the deliverable built from the
material" is a question about what went in.

**Runs are stored, not just printed** — `evals/results/*.json` (gitignored),
tagged with agent, provider, tier, case set, report format and git rev; the
next run matching the first four is picked up as a baseline automatically and
rendered as a Δ column. A `--case` slice is therefore never a baseline for the
full set. The report format is recorded but deliberately NOT part of that
match: the checks score the same property either way, so an HTML run is a
legitimate baseline for a markdown one — which stays true only because every
run that reaches storage scored a text deliverable, the refusal above being
what guarantees it.

**The structural tier must exercise every check** (`test_structural_tier_actually_exercises_every_check`): a check nothing applies to protects nothing. #73's two checks needed no new case — the fake's answer echoes the material it was handed, which is exactly what `report_answers_request` measures, and `unanswerable_missing_material` already reaches the `unanswered` terminal for `refusal_is_bare`. #81's needed one, because no existing case gave the run any material to trace (see the fixture bullet above); it is scored on the fake for the same reason #73's are — the fake echoes what it is handed, so a step that was handed the material says so and a step that was not cannot. `refusal_declared` (#59) is the case that made this bite — the fake has no notion of insufficient material, so `KeywordLLM` recognises the one shape it can (a request leaning on material nobody supplied, `MISSING_MATERIAL_WORDS`) exactly as `DIRECT_WORDS` lets it exercise triage, and one golden case reaches the `unanswered` terminal deterministically.

**Say what it is worth.** Eleven cases sampled once from a stochastic model is
a smoke signal, not a benchmark; the structural tier is the only part that is
actually reliable, and it only proves the machinery holds.
