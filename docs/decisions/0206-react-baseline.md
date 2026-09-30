# 0206 — The compiler's baseline is a ReAct agent over the retrieval ports only, reasoning alone

- **Issue:** #206 · **PR:** #207 · **Status:** partially superseded by [0208](0208-compare-harness.md) (hits per search)
- **Rule in `CLAUDE.md`:** "`agents/react/`: the compiler baseline…" (Agents)

**Context.** `docs/design/compiler-v1.md` (step −1) measures every compiler step against a ReAct agent on one task set; none existed (G1 is a scripted test). The first draft gave it "the registry's ops, wrapped as tools". But `analysis` and `critique` take no arguments and read their material from the DAG state by name (`_material`, `agents/default/_step.py`), so wrapped before step 1 they would find `results` empty and reason from the request alone.

**Decision.** `agents/react/` is a graph agent (`create_agent`) whose tools are the default agent's retrieval ports: `read_prior_job`, `read_file`, `search_documents`, `web_search`, each registered only when a port backs it (`open_default_resources`, `readable_roots`, the job history). The model analyses and critiques itself. A call returns `{"items": [{id, source, title, text}]}`, shared out within 32 000 characters (`research`'s material budget), ≤ 10 hits per search; a refusal travels as `refused`/`unavailable`, never as silence (0060).

**Alternatives and why not.**
- Every op as a tool: a baseline weakened by construction, the comparison rigged for the compiler.
- Wrapping the retrieval *capabilities* (their own query planning, a second model): the baseline would orchestrate less than it is meant to; the ports leave query writing to it.
- Comparing on the same reasoning ops: possible only after step 1 gives them arguments; then a second baseline, said to be one.

**Measured.** `tests/test_baseline_agent.py`: tools exist exactly when a port backs them and are the retrieval set; one item shape within the budget, a long item cut and marked, a short one untouched; a refused file and an unknown job come back as material; a scripted search-then-answer runs as a job through `build_app`/`run_for`, result = the answer. Removing the share-out fails the budget test; registering a tool with no port fails the first.

**Consequences.** The comparison harness and the run-time-width cases are #206's second half (`evals/compare.py`). `jobsmith --agent react chat` has no chat (a graph agent); it runs as a job (`/engine/jobs`). `KeywordChatModel` drives the chat's tool, not these, so a keyless run of the baseline answers without retrieving.
