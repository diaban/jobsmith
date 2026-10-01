# 0219 — The planner's IR is constrained as far as each provider's strict mode takes it; what it refuses falls to the analysis, never the arguments

- **Issue:** #219 · **PR:** #220, #221 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet: step 1c ships the normaliser and writes the rule

**Context.** Step 1a of `docs/design/compiler-v1.md`, its riskiest hypothesis: a strict mode holds the IR schema. `evals/strict_probe.py` writes the step-1 IR as Pydantic models (the default registry's 8 ops plus `analyze`/`extract`/`synthesize`, their args, `$…` references, `map`, `when`, `output`): `full`, a step shape per op discriminated by `op` (22 `$defs`, 9.4k characters); `shared`, one step shape with `args` typed by shape, the op/args pairing left to the analysis (4.5k).

**Decision.** For 1c, one normaliser (`strict()`, in draft in the probe): every object closed with every key required, `oneOf` → `anyOf` (OpenAI's refusals), `const` → a one-value `enum`. Then the fallback, in two levels, each measured: (1) a keyword a provider refuses moves into the description and the analysis enforces it (Anthropic refuses `minimum`, `maximum`, `maxItems`; it accepts `pattern`, so ids and references stay constrained); (2) a schema whose grammar a provider refuses as too large becomes `shared`. Arguments always stay in the schema: as a JSON string they measured worse than no schema on OpenAI, and gave nothing over `shared` on Anthropic. Whether 1c ships `shared` for both providers or `full` on OpenAI is open: `shared` on OpenAI is not measured.

**Measured.** 10 requests (4 of run-time width, 2 extraction/research, 4 controls), n=4, first call, no repair; valid = parses against the IR models and passes checks 1 and 3 (and, for `shared`, the op/args pairing). OpenAI refuses Pydantic's raw schema (`required`, then `oneOf`) and accepts both normalised. Anthropic refuses `full` even degraded ("the compiled grammar is too large") and accepts `shared` and `fallback` degraded. Valid programs out of 40, in the first form the provider accepts:

| variant | gpt-5-nano | gpt-6-luna | claude-haiku-4-5 |
|---|---|---|---|
| `full`, strict | 37: 3 references to no step or to itself | 40 | refused |
| `shared`, degraded | not measured | not measured | 37: 3 `analyze` with no `instruction` |
| args as a JSON string | 30: argument names | 27 | 36 |
| plain `json_object` | 34 | 36: `slide_deck` given an `instruction` | 33: a `map` inside `args`, `output` as a reference |

**Consequences.** A schema removes argument errors, not wrong programs: a reference to a missing step fits the pattern, so check 3 and the repair stay, and `shared` adds the op/args pairing to the analysis. Per call: nano 18 s, luna and Haiku 2–3 s. One model per row, n=4: noise both ways. Falsified: `strict()` without `required` fails 2 tests; no cycle check, no pairing check, `pattern` refused each fail 1.
