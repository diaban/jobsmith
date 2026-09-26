# 0130 — Effort is proportionate to the risk, with a budget and a node probe

- **Issue:** #130 · **PR:** #NN
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "Effort is proportionate to the risk" (Working on this repo)

**Context.** #125 was a 14-line prompt fix. Its pass took ~1h30 and ~5,900 gpt-5-nano calls: 4 variants × 30 phrasings × 20–30 runs, plus ~30 min of full-job evals that showed nothing new, plus a 179-line record. The answer was clear 20 minutes in. Re-measure, falsify, record and evals each set a floor and none sets a ceiling, and the owner judged the cost out of proportion for a small personal project.

**Decision.**
- **A budget per issue:** ≤ 45 min, of which ≤ 15 min of measurement.
- **The measurement:** a node probe with `evals/probe.py`, before and after, ≈10 phrasings (half of them controls) × n=10, ≤ 300 calls. The probe refuses a run above that.
- **One variant** at a time; alternatives only if it fails.
- **Full-job evals** only when the probe is inconclusive.
- **A record ≤ 20 lines,** or none when no alternative is worth keeping.
- **A small fix is done directly** on a branch, not delegated.
- **Over budget:** stop and report.

**Alternatives.** A cap in memory only: it binds the coordinating session but not the agents it briefs, which read `CLAUDE.md`. Every agent writing its own probe: three probes in one day for #80, #125 and #126.

**Measured.** `evals/probe.py` on `document_intent`, gpt-5-nano, 4 requests × 5 runs = 20 calls in a few seconds. The file request was read as markdown 5/5; the three controls (the same request without the file, "save a file in vim", "hello") stayed silent 15/15.
