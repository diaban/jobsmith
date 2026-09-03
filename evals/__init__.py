"""Evaluation harness: make prompt changes measurable.

The router, the planner and the generator are prompts. Changing one used to be
judged by eye on a single example, which is tinkering, not engineering. This
package turns that judgement into numbers a human can compare between two runs.

**What it scores: properties, not text.** An LLM's wording varies from call to
call; the *properties* of what it produced do not. So nothing here asserts an
expected answer. It asserts that the plan only names registered capabilities,
that its DAG is acyclic and its dependencies are satisfiable, that the router
sends an obviously-simple message down the direct route and an obviously-complex
one down the planning route, that the run reached the terminal it should have,
and that the deliverable carries a title, an answer and its provenance.

**Two tiers.**

- ``structural`` runs on the deterministic fakes (``KeywordLLM``) — no API key,
  no variance, fast. Its job is to catch *structural* regressions: a prompt edit
  that drops the marker the router keys on, a planner template that no longer
  renders capability names, a report that loses its provenance. It is expected
  to score 100%, and ``tests/test_evals.py`` runs it inside ``make check`` so CI
  enforces exactly that.
- ``llm`` needs a real provider and tolerates variance. It never gates CI. Its
  numbers are the ones that actually say whether a prompt change helped.

**Honesty about what this is worth.** A handful of cases sampled once from a
stochastic model is a smoke signal, not a benchmark. A three-point move in the
LLM tier is noise; a check falling from 100% to 40% across every case is not.
Use ``--repeat`` to see how much a case wobbles before believing a delta, and
remember the golden set is small and hand-written: it measures the failure modes
someone thought of, not the ones nobody did.

Usage::

    make eval                    # structural tier, fakes, no key needed
    make eval LLM=anthropic      # llm tier against a real provider
    python -m evals --help
"""
from __future__ import annotations

from .cases import GOLDEN_CASES, EvalCase, cases_for
from .harness import Observation, run_suite
from .results import SuiteResult, load_baseline, render_summary, write_result
from .scoring import CHECKS, Check, score

__all__ = [
    "CHECKS",
    "GOLDEN_CASES",
    "Check",
    "EvalCase",
    "Observation",
    "SuiteResult",
    "cases_for",
    "load_baseline",
    "render_summary",
    "run_suite",
    "score",
    "write_result",
]
