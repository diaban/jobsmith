# 0129 — The material steps write no slides: a separate step builds the deck

- **Issue:** #129 · **PR:** #NN · **Status:** accepted
- **Rule in `CLAUDE.md`:** "The deliverable is written for its reader" (`SUBJECT_ONLY_RULE`)

**Context.** On a deck request, `analysis` designed the slides itself ("Slide 1: …", speaker notes) 8/8 times despite `SUBJECT_ONLY_RULE` ("never design the document"). The generator then pasted that outline into the written answer about half the time (0126, *Controls*).

**Decision.** One variant: `SUBJECT_ONLY_RULE` now names the concrete shape: a deck is built by a separate step from the material, so write no slides, slide titles, slide-by-slide outline or speaker notes. The rule is on every material prompt and on the generator, so nothing else changed.

| `make probe` on `cap_analysis`, gpt-5-nano, n=10 | main | branch |
|---|---|---|
| 5 deck requests, no slide outline in the output | 15/50 | 49/50 |
| 5 controls (one-pager, PDF, plain comparison, file) | 50/50 | 50/50 |
| deck requests still about the subject (branch only, n=5) | — | 15/15 |

The generator was not re-measured: the outline it pasted came from this material.
