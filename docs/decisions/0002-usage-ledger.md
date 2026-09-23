# 0002 — What a job cost is part of its record

- **Issue:** #2 · **PR:** #16
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0085](0085-answer-in-the-conversation.md)

## From “Core concepts (read these files first)”

**`core/usage.py`** — token/cost accounting. `LLMClient.chat` still returns `str`: usage travels on an **ambient ledger** (a `ContextVar` the JobManager installs per run) that adapters push into with `record_usage(...)`, because what a call cost belongs to the run, not to every call site's signature. Attribution is read from LangGraph's runtime config — the root segment of `checkpoint_ns` IS the responsible parent node (`planner`, `cap_research` → `research`) — so a capability nobody wrote for this feature is still attributed, and node signatures are untouched. Failure to resolve a scope degrades to `unattributed`, never to a wrong total. `DEFAULT_PRICES` is a dated snapshot, overridable with `$JOBSMITH_PRICES` (inline JSON or a file path, longest-prefix match); an unpriced model reports tokens with `cost_usd: None` rather than an invented number. **Not covered**: the chat layer's LangChain calls (the two-stack split) — conversation tokens are not counted.

## From “Jobs layer (`jobs/`)”

**What a job cost is part of its record**: `Job.usage` is the run's aggregate (`core/usage.py`), refreshed on *every* summary persist — so a job still running already shows its spend, and `/events` carries it live. The per-step breakdown lives where the step's own material does, in `CapabilityResult.meta["usage"]` (stamped by `Capability._emit_success`/`_emit_failure`, including on failure — a step that burned 40k tokens and failed is exactly the one worth seeing). `jobsmith job <id>` and `GET /jobs/{id}` show both; the deliverable shows neither since #85, unless `with_provenance` asks it to recite the record.
