# 0053 — A step's node name says which step finished

- **Issue:** #53 · **PR:** #56
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0048](0048-terminal-ui.md)

## From “Jobs layer (`jobs/`)”

**A `cap_*` update's `results` is the whole channel, not the step's own contribution.** The stream is incremental — a mounted sub-graph is published at the superstep it completes, measured, so nothing here wants `subgraphs=True` — but what it publishes is its output *state*, and `Send(node, state)` seeds it with the parent's, so `results` comes back as the union of every step so far. Which step just finished is therefore read from the **node name**, never from the payload's keys. Reading the keys re-announced every earlier step on every wave, and `_apply`'s `now_iso()` overwrote their stamps until a four-step chain claimed to have finished inside 121µs at the end of a 3½-minute run (#53) — it also re-saved every earlier result, quadratically. The property is pinned on a chain in `tests/test_jobs.py`, as **spacing** and not as "strictly increasing": the stamps are written in `results` order, so the defect produced increasing values too and passed that reading.
