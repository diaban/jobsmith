"""Running the golden set through the real product path.

Nothing is stubbed below `build_app`: a case becomes a real Job, driven by the
real graph (validate_input → router → planner → executor → generation), and
the deliverable is the file `MarkdownReport` actually wrote. That is the point
— a harness that reimplemented the pipeline would measure the harness.

Two deliberate choices:

- **Persistence is forced to `memory`** and reports go to a scratch directory,
  so an eval run never touches the developer's `agent.db` or `artifacts/`.
- **The chat stack is not exercised.** A `KeywordChatModel` is injected purely
  so `build_app` does not go looking for `langchain-anthropic`; no chat session
  is ever opened. Prompt quality in the chat layer is a different measurement.

The route a run took is read back from the checkpointer rather than inferred,
because the interesting failure — the planner rescuing a message the router
should have sent direct — is invisible from the outside.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jobsmith.app.agent import build_app, pick_report_format
from jobsmith.app.providers import KeywordChatModel, make_llm, pick_provider
from jobsmith.core.executor import Executor

from .cases import EvalCase


def resolve_provider(choice: str | None = None) -> str:
    """The provider this run uses: an explicit choice, else the usual auto-detection."""
    return choice or pick_provider()


@dataclass
class Observation:
    """What one run of one case produced — the raw material every check reads."""

    case_id: str
    query: str
    attempt: int = 1
    job_id: str = ""
    route: str | None = None
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    plan_rationale: str = ""
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal_kind: str | None = None
    final_answer: str | None = None
    report_path: str | None = None
    report_text: str | None = None
    report_format: str = "markdown"   # which Reporter wrote it (checks read through it)
    registry: tuple[str, ...] = ()
    duration_s: float = 0.0
    error: str | None = None          # the harness itself blew up (not a run failure)

    def compact(self) -> dict[str, Any]:
        """Serializable record — big payloads stay out of the results file."""
        return {
            "case": self.case_id,
            "attempt": self.attempt,
            "job_id": self.job_id,
            "route": self.route,
            "plan": [
                {"capability": s["capability"], "depends_on": list(s["depends_on"])}
                for s in self.plan_steps
            ],
            "steps_ok": {n: bool(r.get("ok")) for n, r in self.results.items()},
            "terminal_kind": self.terminal_kind,
            "answer_chars": len(self.final_answer or ""),
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


async def _route_of(app: Any, job_id: str) -> str | None:
    """The triage decision, read back from the run's checkpoint."""
    try:
        snapshot = await app.manager.graph.aget_state(
            {"configurable": {"thread_id": job_id}}
        )
        route = snapshot.values.get("route")
    except Exception:
        return None
    return route or None


async def run_case(
    app: Any, case: EvalCase, registry: tuple[str, ...], attempt: int = 1
) -> Observation:
    """One case, one job, through the whole graph."""
    obs = Observation(case_id=case.id, query=case.query, attempt=attempt, registry=registry)
    started = time.perf_counter()
    try:
        job = await app.manager.create_job(case.query, dict(case.inputs))
        obs.job_id = job.job_id
        job = await app.manager.run_job(job.job_id)
        obs.plan_steps = list((job.plan or {}).get("steps", []))
        obs.plan_rationale = (job.plan or {}).get("rationale", "")
        obs.results = dict(job.results)
        obs.terminal_kind = job.terminal_kind
        obs.final_answer = job.final_answer
        obs.report_path = job.report_path
        main = next((o for o in job.outputs if o.role == "main"), None)
        if main is not None:
            obs.report_format = main.format
        obs.route = await _route_of(app, job.job_id)
        if obs.report_path:
            try:
                obs.report_text = Path(obs.report_path).read_text(encoding="utf-8")
            except OSError:
                # A missing deliverable is a failed `report_written` check, not
                # a broken harness: leave it to the scorer to say so.
                obs.report_text = None
    except Exception as e:                      # a broken harness must not look like a bad prompt
        obs.error = f"{type(e).__name__}: {e}"
    obs.duration_s = time.perf_counter() - started
    return obs


async def run_suite(
    cases: list[EvalCase],
    *,
    agent: str | None = None,
    provider: str | None = None,
    repeat: int = 1,
    concurrency: int = 1,
    reports_dir: str | None = None,
    report_format: str | None = None,
) -> tuple[list[Observation], dict[str, Any]]:
    """Compose the agent once, then run every case (× `repeat`) through it.

    Returns the observations plus the run's context (provider, agent, registry)
    — everything a results file needs to be comparable with another one.
    """
    choice = resolve_provider(provider)
    llm = make_llm(choice)
    with TemporaryDirectory(prefix="jobsmith-eval-") as scratch:
        app = await build_app(
            agent=agent,
            llm=llm,
            chat_model=KeywordChatModel(),   # unused: no chat session is opened
            db="memory",
            reports_dir=reports_dir or scratch,
            report_format=report_format,
        )
        registry = registry_names(app)
        try:
            semaphore = asyncio.Semaphore(max(1, concurrency))

            async def one(case: EvalCase, attempt: int) -> Observation:
                async with semaphore:
                    return await run_case(app, case, registry, attempt)

            observations = await asyncio.gather(*[
                one(case, attempt)
                for attempt in range(1, max(1, repeat) + 1)
                for case in cases
            ])
        finally:
            await app.aclose()
    context = {
        "agent": app.agent_name,
        "provider": choice,
        "registry": list(registry),
        "repeat": max(1, repeat),
        # Recorded, but deliberately NOT part of what makes two runs
        # comparable: the checks read through `deliverable.extract`, so the
        # score is the same property whichever Reporter produced the file.
        "report_format": pick_report_format(report_format),
    }
    return list(observations), context


def registry_names(app: Any) -> tuple[str, ...]:
    """Capability names of the composed agent, read off the compiled graph.

    The registry is frozen into the graph at build time as one `cap_<name>`
    node per capability (`Executor.node_name`), so the graph is the honest
    source: it reflects what this configuration actually registered — with or
    without a document source, whichever agent was composed.
    """
    prefix = Executor.node_name("")
    nodes = getattr(app.manager.graph, "nodes", {}) or {}
    return tuple(sorted(n[len(prefix):] for n in nodes if n.startswith(prefix)))
