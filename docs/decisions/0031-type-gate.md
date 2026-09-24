# 0031 — pyright is a gate, and reads of partial state justify themselves

- **Issue:** #31 · **PR:** #32, #33
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added. The scribe moved three more bullets here verbatim from `CLAUDE.md` on 2026-09-24 (`reportMissingImports`, scope/`typeCheckingMode`, the `node` dependency), to hold the budget (→ `tests/test_claude_md_budget.py`); `CLAUDE.md` keeps a one-line rule for them with `→ 0031`.

## From “Working on this repo”

**The type gate (`make types`, pyright)** is the only check here that can see a bug no test can. A signature that lies is not observable at runtime — the one that prompted #31 (`compose_reporters` promising `Reporter`, returning `Reporter | MultiReporter`) was spotted by accident in VS Code and would never have failed CI. pyright rather than mypy *because* of that: they report the same findings, but Pylance **is** pyright, so `[tool.pyright]` in `pyproject.toml` is one configuration for the editor and the gate instead of a permanent split.

- **`reportTypedDictNotRequiredAccess` is on** (#31 phase 2) — it is a *read-discipline* gate on the state schemas. `AgentState`, `CapabilityBaseState` and `CapabilityOutputState` are `total=False` because a LangGraph node returns a **partial update**; that is right for writes and wrong for reads, so the rule asks every `state["k"]` to justify itself. Its 31 hits were by key — `query` 17, `aspects` 3, `generated_query` 3, `queries` 2, `found`/`notes`/`output`/`draft_answer`/`ok`/`data` 1 each — and only **two** touched `CapabilityResult`; an earlier note here claiming they concentrated there was simply wrong. Two guarantees, one expressible in a type: `query` is guaranteed **at entry** (`runner.stream()` invokes the graph with it, and the executor's `Send(node, state)` hands each sub-graph the whole parent state) and is now `Required[str]` in `AgentState` and `CapabilityBaseState` — free, because **no node is annotated `-> AgentState`**, they all return plain `dict`. Everything else is guaranteed only **by graph order** (`draft_answer` exists when `validate_output` runs, not before), where `Required` would be a lie: those read with `.get()` plus a default that states what missing means, next to a comment naming the node or router that guarantees it. **Never blanket-`# pyright: ignore` this rule** — a site where neither is honest is a finding about the graph, not noise.

- `reportMissingImports` is a **warning**, not an error: the optional extras are imported lazily behind `try/except ImportError`, and their absence under `.[dev,api]` is a fact about that environment, not a defect. A misspelled import stays visible and is loud at runtime anyway.

- Scope is `jobsmith/`. `tests/` and `evals/` are out, and `typeCheckingMode` is `basic`, not `strict` — the point is a gate that holds, not a maximal one.

- pyright is a Python wrapper around a bundled JS checker: it needs a `node` on `PATH`. GitHub runners have one; without one it silently downloads a node build on first run, which is why `make types` is fast here and may not be on a fresh machine.
