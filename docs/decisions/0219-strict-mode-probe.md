# 0219 — The planner's IR is sent in strict mode whole; a keyword a provider refuses falls to the analysis, never the arguments

- **Issue:** #219 · **PR:** #220 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet: step 1c ships the normaliser and writes the rule

**Context.** Step 1a of `docs/design/compiler-v1.md`, its riskiest hypothesis: a strict mode holds the IR schema. `evals/strict_probe.py` writes the step-1 IR as Pydantic models (the default registry's 8 ops plus `analyze`/`extract`/`synthesize`, their args, `$…` references, `map`, `when`, `output`): a union of 11 step shapes discriminated by `op`, 22 `$defs`, 90 properties, 32 `pattern`s, 9.4k characters.

**Decision.** For 1c: OpenAI gets the whole schema in strict mode. One normaliser (`strict()`, in draft in the probe): every object closed with every key required and `oneOf` → `anyOf` (OpenAI's two refusals), plus `const` → a one-value `enum` and no `default`/`title`/`discriminator` (Anthropic's SDK moves `const` out of the grammar). **The note's fallback changes**: where a provider refuses a keyword, that keyword moves into the description and the analysis enforces it, as Anthropic's `transform_schema` does; the arguments stay in the schema. Leaving them out cannot be an open object under strict mode (every object is closed), so it means arguments as a JSON string, which measured worse than no schema.

**Alternatives and why not.** Arguments as a JSON string (`fallback`): 30/40 valid, below plain JSON. Plain JSON (today's `json_object`): 6 argument errors in 40 that the schema removes. Arguments as name/value pairs: the same loss of types as the string; not measured.

**Measured.** gpt-5-nano, 10 requests (4 of run-time width, 2 extraction/research, 4 controls), n=4, first call, no repair; valid = parses against the IR models and passes checks 1 and 3. Pydantic's raw schema is refused by OpenAI (`required` must list every key, then `oneOf` is not permitted); normalised, it is accepted, `pattern`, bounds, `$defs` and `const` included. **Anthropic is not measured** (no key here); its SDK would keep 32 `pattern`s, 2 `maxItems` and 5 bounds out of the grammar, so ids and reference shapes are the analysis' there. Falsified: `strict()` without `required` fails 2 tests; no cycle check fails 1.

| variant | valid | schema or args errors | findings | mean s | output tokens |
|---|---|---|---|---|---|
| full, strict | 37/40 | 0 | 3, all on the deck request: a reference to no step, or to itself | 18.3 | 2 826 |
| fallback, args as a string | 30/40 | 9: wrong argument names (`materials`, `material` for `items`) | 1 | 18.6 | 2 787 |
| plain `json_object` | 34/40 | 6 | 0 | 22.4 | 3 331 |

**Consequences.** A strict schema removes argument errors, not wrong programs: a reference to a missing step still fits the pattern, so check 3 and the repair stay. Anthropic's row is owed before 1c ships its client (`python -m evals.strict_probe --provider anthropic -n 4`). One weak model, n=4: noise both ways.
