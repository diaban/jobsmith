# Compiler v1: the reference graph compiles a program and interprets it

Design note written 2026-09-29 on `main` b4f0be6. It follows an outside review of that
commit (in French, not checked in), which set the target and the reasons for it. It
turns that review into a design: the shape of the IR, the effects model, the gates and the
build order. **Status: proposed.** The owner has decided three things (listed under
"Decided"); everything else here is a proposal to accept or amend. "Open questions" lists
what the owner has to settle before step 0 lands. Claims about today's code were read on
b4f0be6 and carry a file reference; claims marked *(to check)* are hypotheses a step's
probe must confirm.

## Why

Today's planner **selects capabilities**; it does not orchestrate tasks. A capability
appears at most once (`dag/planner.py:108`), a step has no arguments (`PlanStep` is
`capability` + `depends_on`, `dag/state.py:43`), and the plan is fixed before any result
exists. The width of the DAG is therefore bounded by the size of the registry (about six).
"Find the GitHub repositories named jobsmith and compare them" needs one search, then **N
reads in parallel, where N is known only after the search**, then a synthesis. The planner
cannot express it.

The target is an **agent compiler**, in the line of LLMCompiler (arXiv 2312.04511) and
PlanCompiler (arXiv 2604.13092): the planner fills a constrained intermediate
representation (IR), deterministic checks run before anything executes, and a generic
interpreter runs the result on the engine. Orchestration is compiled; exploration stays
confined to bounded ReAct steps. The owner's principle: **the less the planner writes from
scratch, and the more it is held by schemas and small sets of allowed values, the less it
breaks.**

Three reservations shape the design:

- **Compiling moves errors, it does not remove them.** 59% of PlanCompiler's failures are
  valid-but-wrong plans. Hence verification points placed on purpose (step 4), not only
  static checks.
- **On a truly open task an autonomous agent stays better.** The compiled path targets
  structured, repeatable tasks; the ReAct agent stays the unit of exploration.
- **The lasting advantage is cost, latency, determinism and auditability**, not
  intelligence. Each step is measured against a ReAct baseline on the same tasks, or the
  claim stays a claim (step −1).

## Scope

jobsmith becomes a **compiler** (`dag/`) sitting on a **runtime** (`engine/`, delivered by
core v1). Two paths reach the engine, and both hand it the same `GraphSpec`:

- **Just in time**: a new request goes through the planner → IR → analysis → interpreter.
- **Calibrated program**: a known task starts from an IR that has already been validated,
  with no planner call.

**Non-goals (v1).** A Turing-complete IR, general loops or nested `map`; Python code as
the compile target (A1's approach: more expressive, much less checkable); replacing ReAct
agents; an optimiser beyond "independent steps run in parallel" (element 4 of the review
is deferred, see "Later"). **Engine changes are out of scope except the ones named in this
note**, each generic (G4) and each an open question.

## Layers

Unchanged from core v1 (`tests/test_layers.py`): `engine` imports nothing of ours, `dag`
may import `engine` and `artifacts`, the bench imports anything. The compiler lives in
`dag/`:

| module | holds | replaces |
|---|---|---|
| `dag/ir.py` | the IR types (`Program`, `Step`, `MapSpec`, `Ref`), parsing and the JSON schema the planner is constrained by | `Plan`/`PlanStep` in `dag/state.py` |
| `dag/ops.py` | `OpSpec` (the one notion of a step) and `Effects` | `CapabilitySpec` (kept as an alias during the migration) |
| `dag/analysis.py` | the static checks, each returning a list of `Finding(step_id, code, message)` | `Planner._validate_plan` |
| `dag/interpreter.py` | the wave router: ready steps, ref resolution, `map` expansion, retries | `dag/executor.py` (evolved in place, not rewritten) |
| `dag/programs.py` | loading, parameters and versions of calibrated programs | — |
| `agents/<agent>/programs/*.json` | an agent's calibrated programs, reviewed as diffs | — |

## Step identity (step 0: the blocker)

**Today a step is its capability's name, everywhere.** Checked on b4f0be6: the planner
refuses a duplicate; the fact is `step:{self.spec.name}` (`dag/capability.py:170,210`);
`results` is keyed by name and `completed_capabilities` appends names
(`dag/capability.py:174,214`); the executor's `_done` and `_to_retry` count runs by name
(`dag/executor.py:47-61`); `drop_steps` takes names and is a public contract since #203
(`dag/jobs.py:381`); `step_finished_at`, `ordered_results`, `step_usage`, the TUI's DAG
(`tui/render.py:176`) and the REPL's plan line read names. 22 source files match.

Two more couplings, which the review did not list, have to move with it:

1. **Capabilities read each other by name.** `SingleStepCapability._material` walks a
   fixed `UPSTREAM` list of capability names (`agents/default/_step.py:77`); `research`,
   `critique`, `slides` and `generation` do the same. This is implicit dataflow. With two
   `analysis` steps, `critique` cannot know which one to read.
2. **Usage is booked per root node.** `current_scope()` keeps the root segment of
   `checkpoint_ns` and drops the task id (`engine/usage.py:280-281`), and
   `Capability._usage_meta` reads `ledger.get("cap_<name>")`. Two instances of one op
   running in the same wave book into one scope, and each reports the sum.

**The change, compatible by construction.**

- `PlanStep` gains `id`, **defaulting to the capability name**, so every plan written
  today is unchanged and every name in the public contract is already a valid id.
- `results`, the `step:<id>` fact, the run count and `step_finished_at` are keyed by id.
  The channel `completed_capabilities` becomes `completed_steps` (ids); see open question 2
  on checkpoints in flight across the rename.
- The executor Sends each step with its identity in the payload: `step = {"id", "op",
  "args"}`. `Capability._emit_*` writes under `state["step"]["id"]` instead of
  `self.spec.name`. A capability never learns its id any other way.
- `depends_on` and `drop_steps` take ids. A name that is the op of several steps and the
  id of none is refused as ambiguous, with the ids listed.
- **Reading upstream material by name keeps working for a step with no `args`**: "the
  latest `ok` result of a step whose op is X", in plan order. It is the fallback the
  default agent runs on until its planner emits references (step 1), after which a step's
  material comes from its arguments. The fallback is removed when no shipped agent needs
  it; that removal is its own PR.
- **Usage per instance**: see open question 1. Until it is settled, the duplicate ban
  stays, so no two steps can share a scope.
- **Only then is the duplicate ban lifted.**

## The IR

A small JSON language, filled by the planner under a JSON schema generated from the
registry (the op names are an `enum`, each op's `args` are its input schema), so the
set of things the model can write is closed.

```json
{"version": 1,
 "steps": [
   {"id": "search", "op": "web_search", "args": {"query": "jobsmith github repository"}},
   {"id": "reads", "op": "read_repository",
    "map": {"over": "$search.hits", "max_items": 20, "concurrency": 5, "on_error": "partial"},
    "args": {"url": "$item.url"}},
   {"id": "compare", "op": "analysis",
    "args": {"material": "$reads", "instruction": "Compare the repositories on purpose and activity."}}
 ],
 "output": "compare"}
```

- **A step** is `id` (`^[a-z][a-z0-9_]*$`, unique in the program), `op` (a registered
  op), `args` (an object checked against the op's input schema) and, optionally, `after`
  (ids to wait for with no data passed: ordering for effects), `map` and `when`.
- **References** are strings of the form `$<id>`, `$<id>.<field>[.<field>…]`,
  `$input.<key>` (the job's `inputs`) and, inside a `map`, `$item[.<field>…]`. A
  reference is a whole argument value; there is no string interpolation (`"see $x"` is a
  literal). Dependencies are **derived** from references plus `after`; the planner no
  longer writes `depends_on`, which removes a class of inconsistent plans.
- **`output`** names the step whose result is the program's answer material. Absent, it
  is the last step (today's behaviour: generation reads everything, in plan order).
- **`instruction`** is not special: it is a string argument that ops calling a model
  declare in their input schema. An op that is deterministic has none.
- **`when`** (step 3, reserved in v1's schema): `{"ref": "$triage.kind", "in": ["a",
  "b"]}`. A step whose condition is false is **skipped**, its dependents too unless they
  also depend on a step that ran. The value must be typed by an `enum` in the producer's
  output schema, so the branch set is closed. No `else`: two guarded steps.
- **`verify`** steps: see recompilation (step 4).

**What the IR never gets**: loops, recursion, user-defined functions, string templates,
arithmetic. The day a program needs one, it is a sign the step belongs inside a ReAct op.

### `map`

- **Typing**: `over` resolves to `array<T>`; the body's args are checked with `$item: T`;
  the map step's output is `array<U>` where `U` is the op's output. Checked statically.
- **`max_items` is required**, capped by the op's `Effects.max_fanout` and the profile.
  Beyond it, the list is **cut and the cut is written** in the step's result meta (`{"cut":
  {"kept": 20, "of": 34}}`) and shown in the view; `"on_overflow": "fail"` fails instead.
  Never silent.
- **`concurrency`** bounds the instances in flight, separately from the width, for rate
  limits. *Caveat, checked by reading the executor*: the executor dispatches by
  superstep, and a superstep waits for its slowest branch. Concurrency `k` therefore
  means batches of `k`, not a sliding window, and a slow instance holds the next batch
  (and every other ready step). v1 accepts this; the latency measurement will say whether
  a sliding window (one Send per freed slot, possible because each instance edges back to
  the dispatcher) is worth building.
- **`on_error`**: `"all"` (any failed instance fails the step), `"partial"` (the step
  succeeds with the successes, failures listed in meta) or `{"min_ok": n}`.
- **Identity**: instance `i` is the step `reads[i]`, with its own result, fact, run count
  and retry. The map step's value `$reads` is not stored: the interpreter assembles it in
  item order when a reference reads it. A resume sends only the instances with no result
  (`reads[0..5]` done ⇒ `reads[6..8]` sent); progress reads `6/9` from the facts.
- **No nested `map`.** A body is one op.
- A "reduce" is not an operator: it is an ordinary step whose input is `array<U>`.

## The registry: one notion of a step

```python
@dataclass(frozen=True)
class Effects:
    read_only: bool = True        # False: writes, sends, pays — something outside the job changes
    idempotent: bool = True       # True: running it twice leaves the world as running it once
    cost: str = "llm"             # "free" | "llm" | "agent": an order of magnitude, for the bound
    approval: bool = False        # a human says yes before it runs
    max_fanout: int = 50          # the largest map over this op

@dataclass(frozen=True)
class OpSpec:                     # today's CapabilitySpec, plus effects, schemas now binding
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    effects: Effects = Effects()
    requires_inputs: tuple[str, ...] = ()
```

- **Schemas become binding** (today they are "advisory", `dag/capability.py:38`). They
  are checked statically on every reference, and at run time on each value crossing a
  step boundary. The checker is ours and covers a stated subset of JSON Schema (`type`,
  `properties`, `required`, `items`, `enum`, `format`): no new dependency. An op with an
  empty output schema produces `object` and cannot be referenced below the top level.
- **Behind the contract, several implementations**, each compiled to one parent-graph
  node, so the interpreter never sees the difference:
  - `Capability`: today's sub-graph, the **bounded ReAct/agentic step**. Kept as is.
  - `ToolOp`: a LangChain `BaseTool` (hence an MCP tool through its adapter) used as it
    is: its args schema is the input schema.
  - `FnOp`: deterministic Python, `cost="free"`.
  - `LlmOp`: one constrained model call whose answer must match the output schema.
  - `ReviewOp`: a human decision, through `interrupt()` → `needs_input` → `answer_job`
    (0167), which the DAG does not use yet.
- **The planner sees schemas, not a taxonomy.** It never knows whether an op is a tool
  or a sub-agent.
- The prefix `cap_` stays on every op's node: it is what the usage scope and the view
  strip today, and renaming it buys nothing.

### What effects decide

- An op with `approval=True`, or `read_only=False` under the profile's policy, is
  preceded by a review the interpreter inserts. A calibrated program can carry the
  approval as reviewed (step 5); a just-in-time one never does.
- **`GraphSpec.relaunch`** exists because "only the graph knows whether a node can run
  twice" (0189). With effects declared, the graph knows. But `relaunch` is **per graph**,
  not per job: the DAG's value can only be derived from its whole registry (every op
  idempotent ⇒ it may be > 0). Deriving it per program needs a per-job bound on the
  record, which is an engine change: open question 4.
- A step's cache key (later) is the hash of its op, version and resolved args, only for
  an op that is `read_only` and `idempotent`.

## Static analysis and repair

Run on every program, just-in-time or calibrated, before its first step. Each check is a
function returning findings; a finding names a step and a code, so a test asserts the
code, not the wording (0110).

1. **Well-formed**: parses against the IR schema; ids unique and well-formed.
2. **Ops exist**, and are applicable (`is_applicable`: today's drop-and-prune of
   inapplicable steps is kept, and a program emptied by it still routes to
   `direct_answer`, 0038).
3. **References resolve** to a step of the program, `$item` only inside a `map`;
   dependencies (derived) are acyclic (Kahn, as today).
4. **Types**: each reference's schema, followed down its field path, is assignable to the
   argument's schema.
5. **`map`**: `over` is an array, `max_items` present and within caps, not nested.
6. **Output reachable**: `output` exists; a step that feeds nothing and has no effect is
   a finding (dropped, not an error).
7. **Cost bounded**: Σ over steps of cost class × width (`max_items` for a map) is within
   the profile's budget.
8. **Effects policy**: as above.
9. **Free text where a structure exists**: a literal string for an argument whose schema
   has an `enum`, a `format` or an object type is refused. It is PlanCompiler's first
   cause of failure.

**A program with findings goes back to the planner** with the program and the findings
rendered as a list, **twice at most**; then it is today's unrecoverable `NodeError`. The
repair is a planner concern and its own node, so each attempt is a checkpoint and a
crash does not replay it. (Today's retry, 0191, is about a step that failed, not a plan.)

## The interpreter

`dag/executor.py` already is the review's element 5: it keeps the plan in state,
recomputes the ready steps each wave, and Sends them. It evolves:

- the plan channel holds a `Program`; ready = every dependency done (skipped counts as
  done) and `when` true;
- before a Send, the interpreter **resolves the step's references** against `results`
  and validates the resolved args; a step whose args do not validate fails as a step
  (recoverable), not the job;
- a `map` step expands into its instances, within `concurrency`;
- retries are counted per id, `_APPENDED` (0194) still stripped from what is Sent;
- the program's `output` feeds the generation pipeline, which stays as it is
  (`merge_results` → `generation` → `validate_output` → …): the compiler changes how
  material is gathered, not how the deliverable is written.

The `plan` fact publishes the program (with `version` and a `revision`, see below). The
DAG view derives what it shows today from it; the HTTP shape changes additively (`id`,
`op`, `args` on each step, `capability` kept as the op for readers until the TUI and
REPL move to ids in the same step).

## Recompilation (step 4)

A **`verify`** step is an interpreter node, not an op: `{"id": "check", "verify":
{"of": "$reads", "expect": …}}`, where `expect` is either a deterministic predicate
(count, non-empty field, an `enum` value) or a constrained model judgement with a
yes/no schema. On failure it calls the planner with the program, the results so far
and the finding, to **rewrite the steps that have not started**. Bounds:
`AgentProfile.max_recompile` (default 1). Rules:

- a finished id is immutable: a rewrite may not reuse it or change it; what gives this
  rule meaning is step 0;
- the rewrite goes through the same analysis; a rewrite with findings is dropped and the
  program runs on unchanged (the verify's failure is recorded);
- the `plan` channel has no reducer (`dag/state.py:207`) and is re-read each wave, so the
  node returns `{"plan": …}`; the `plan` fact is republished with `revision + 1`;
- it is checkpointed, so a recompiled program survives a crash. The engine knows nothing.

The alternative, recompiling from outside with `amend_job` (0177), exists already. The
step's probe compares the two on the same tasks and keeps one.

## Calibrated programs (step 5)

A calibrated program is an IR that has proved itself: **frozen, named, versioned,
reviewed as a diff, covered by golden cases** in `evals/`. It lives with its agent
(`agents/<agent>/programs/<name>.json`, with `params`: a JSON schema for `$input.*`).

- **The planner passes a supplied program through**, the way `document_intent` passes
  seeded formats through (0090): an input `program` (name + version + params) is loaded,
  analysed, and run with **zero planner calls**.
- **Promotion is manual in v1**: `jobsmith program export <job_id> NAME` writes a
  finished job's program as a file to commit. Mining the job history for recurring,
  successful programs is later.
- **Who chooses a program** is open question 3.

## Measurement (step −1)

"Compiled is better" is measured, per step, on one task set, against a ReAct baseline:
**first-try success** (the eval properties, 0003), **cost** (`Job.usage`), **latency**
(created → done), model calls.

What exists: `evals/` scores the DAG agent on a golden set; `make probe` measures a
prompt at a node; the engine serves a graph agent (0165) and counts LangChain calls. What
does not: **there is no ReAct agent to compare with** (G1 is a test with a scripted
model, not an agent in `agents/`), and no harness running one case set through two
agents. Step −1 builds both:

- `agents/react/`: `AgentDefinition(graph=…)` over `create_agent`, whose tools are **the
  same registry's ops**, wrapped as tools. The comparison is then orchestration against
  orchestration, not tools against tools.
- `evals/compare.py`: the same cases, both agents, one table; the llm tier only (the
  structural tier cannot measure a model), bounded like `make probe` (≤ 300 calls).
- New cases that need a width known at run time (the "compare the repositories" family),
  domain-neutral (leak gate).

## Gates

Each lands with the step that makes it pass, and stays. Throughout: `make check` green,
structural evals at 100% (the `KeywordLLM` fake planner learns to emit the IR in the step
that changes the planner's output), G3 and G4 unchanged.

- **C0, identity.** A plan with two steps of one op runs both; both results, both
  `step:<id>` facts and both run counts are kept apart; `drop_steps` by id; ambiguity
  refused; every existing test passes with plans that have no ids.
- **C1, references.** A step's argument comes from a reference; a type mismatch is a
  finding before any step runs; a planner that fixes it on the second call runs; one that
  never does fails as today.
- **C2, map.** 9 items ⇒ 9 instances, never more than `concurrency` in flight; the job
  killed after instance 6 resumes with 3 sends (counted); a list over `max_items` is cut
  and the cut is in the view; `partial` keeps the successes.
- **C3, effects.** A non-read-only op never runs before an answer (`needs_input`); the
  analysis refuses free text where an `enum` exists.
- **C4, recompilation.** A rewrite after a verify failure runs; a finished id is never
  re-run; the rewritten program survives a crash between waves.
- **C5, programs.** A supplied program runs with zero planner calls (counted on the
  ledger) and fails analysis the same way a planned one does.
- **M, measurement.** Each step's PR carries the comparison table against step −1's
  baseline; a regression is stated, not hidden.

## Order: one PR per step (split when over budget, 0130)

−1. **Baseline**: the ReAct agent over the registry's ops, `evals/compare.py`, the
    run-time-width cases. No change under `dag/`.
0. **Step identity** (C0), without lifting the duplicate ban until open question 1 is
   settled. Split: 0a `id` + keys + facts + executor; 0b `drop_steps`, view, TUI, REPL,
   chat; 0c capabilities read material through the by-op fallback; 0d usage per
   instance, then the ban lifted.
1. **Minimal IR**: ids, typed args, references, `output`; binding schemas; `OpSpec` +
   `Effects` declared (not yet enforced); the planner constrained by the generated
   schema; analysis checks 1-4 and 6; repair. C1.
2. **`map`** (C2), and the default agent's first op that makes sense per item.
3. **Effects enforced, `when`, checks 5, 7-9, `ReviewOp`** (C3).
4. **Recompilation**: `verify`, or `amend_job` from outside, whichever the probe keeps
   (C4).
5. **Calibrated programs** from a supplied plan (C5).

A decision record is written by each step that takes a decision (CLAUDE.md), and one
record for the compiler as a whole when step 5 lands, as 0161 did for core v1.

## Open questions for the owner

1. **Usage per instance needs the engine.** Options: (a) a generic `usage_scope(name)`
   context the engine offers and a node may enter, so an op instance books under its
   step id: a small engine addition with no product word; (b) the engine keeps the task
   id in the scope and the DAG's view sums per node: changes the ledger's shape for
   every graph; (c) no per-instance usage, only per op, and the per-step figure is
   dropped from the view. Proposal: (a), probed in 0d (*to check*: that the context
   reaches the LangChain callback inside a sub-graph run by a Send).
2. **Checkpoints in flight across the rename** of `completed_capabilities`. Proposal: as
   core v1 decided for records, start clean: a job interrupted before the upgrade is not
   resumable after it, and says so (`Job.failure`).
3. **Who picks a calibrated program?** Proposal for v1: explicitly (`--program`, an API
   field, a chat tool argument). The router choosing one from the request is later, and
   would be measured like any prompt.
4. **`relaunch` per job.** Proposal: not in v1; the DAG derives its per-graph value from
   its registry's effects.
5. **The superstep barrier.** Proposal: accept it in v1 and let the latency measurement
   decide on a sliding window.

## Later

The optimiser (an LLM step replaced by an equivalent deterministic op, caching by input
hash), mining calibrated programs from history, a sliding-window scheduler, a per-job
relaunch bound, nested `map`, and the router choosing a program.

## Decided (by the owner, before this note)

1. **A constrained IR**, filled by the planner.
2. **Calibrated (ahead-of-time) programs** for tasks already run in, alongside
   just-in-time compilation.
3. **Steps finer than a capability**: a step can be a tool or a single node, not only a
   sub-graph, and it carries an instruction and arguments.

## References

- Kim et al., *An LLM Compiler for Parallel Function Calling*: https://arxiv.org/abs/2312.04511
- *PlanCompiler*: https://arxiv.org/html/2604.13092v1
- *Agent JIT Compilation* (ICML 2026), reading note: https://en.papernotes.org/ICML2026/llm_agent/agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling/
- MightyBot, *What Is an Agent Compiler?*: https://mightybot.ai/blog/what-is-an-agent-compiler/
- stanford-mast/a1: https://github.com/stanford-mast/a1
