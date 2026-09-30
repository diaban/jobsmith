# 0213 — Whoever reads a plan speaks step ids; a capability's name designates its only step and is refused when it runs as several

- **Issue:** #213 · **PR:** see the issue · **Status:** accepted
- **Rule in `CLAUDE.md`:** "`DagJobs.drop_steps` (step ids; a name stands for its only step, → 0213)" (Jobs layer)

**Context.** Step 0b of `docs/design/compiler-v1.md`. After 0a (0211) the run keys everything by step id, but its readers still keyed by capability: `drop_steps` (a public contract since #203, through the chat's `skip_steps` and `POST /jobs/{id}/drop`), the chat's progress line, `job_status` and plan notice, the `JobPlanned` event, the REPL's plan line and `/job`, the TUI's DAG and steps table. With two steps of one capability they would merge two rows into one and drop the wrong step.

**Decision.** Every reader keys and labels steps by `step_id`, which now takes any mapping (a plan read off HTTP is plain dicts); the plan event and notice carry `id` beside `capability`; the TUI's row key is `step`. `drop_steps` takes ids, and a capability's name still works when exactly one step runs it — "skip the critique" keeps its meaning; when several do, it is refused with their ids listed, before anything is stopped.

**Alternatives and why not.** Ids only, names refused: breaks every caller that says "skip the critique", for no gain while one step runs each capability. A name dropping every step that runs it: a user naming a capability rarely means all its instances, and a silent broad drop cannot be taken back. Labelling by capability with the id in brackets: noise while ids are names, which is every plan until 0d.

**Measured.** `tests/test_step_identity.py`, `tests/test_tui.py`: with a plan where one capability runs as two steps, `drop_steps` resolves an id, a name with one step, refuses an unknown name and the ambiguous one; the REPL line, the chat's running steps and progress, the TUI's states and activity line name `notes_a`, `notes_b`, `review`. Falsified: no ambiguity check fails 1; the chat's running steps by name fails 1; the TUI's rows by name fails 1. Full suite unchanged.

**Consequences.** No behaviour change while ids are names. `UPSTREAM` still reads material by name (0c); the ban holds until 0d.
