"""Property checks over one observed run.

Every check answers a yes/no question about *structure*, never about wording:
an LLM that phrases its answer differently must not move the score, an LLM
that emits a plan with a cycle must.

A check reports one of three statuses:

    pass / fail   the property was evaluated
    skip          the property does not apply to this run (a direct-route run
                  has no plan to validate; a rejected query has no report)

Skipped checks are excluded from the denominator, so adding a case that
exercises one path never dilutes the score of another.

The report checks are deliberately layout-independent — they look for the
title, the answer text, the job id and the request, not for the headings
`MarkdownReport` happens to use today. A reformatting of the deliverable
should not read as a regression; losing its provenance should.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .cases import EvalCase
from .harness import Observation

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def applicable(self) -> bool:
        return self.status != SKIP

    @property
    def passed(self) -> bool:
        return self.status == PASS


def _check(name: str, ok: bool, detail: str = "") -> Check:
    return Check(name, PASS if ok else FAIL, "" if ok else detail)


def _skip(name: str, why: str) -> Check:
    return Check(name, SKIP, why)


# ---------------------------------------------------------------- run-level

def check_run_completed(case: EvalCase, obs: Observation) -> Check:
    """The harness got a finished job back — an exception here invalidates the rest."""
    return _check("run_completed", obs.error is None, obs.error or "")


def check_terminal(case: EvalCase, obs: Observation) -> Check:
    """The run ended where the case says it should (answer / user_error / escalated)."""
    if obs.error:
        return _skip("terminal_kind", "run did not complete")
    return _check(
        "terminal_kind",
        obs.terminal_kind == case.expect_terminal,
        f"expected {case.expect_terminal!r}, got {obs.terminal_kind!r}",
    )


# ---------------------------------------------------------------- triage

def check_router_route(case: EvalCase, obs: Observation) -> Check:
    """Obviously-simple messages go direct; obviously-complex ones get planned."""
    if case.expect_route is None:
        return _skip("router_route", "case makes no triage claim")
    if obs.error:
        return _skip("router_route", "run did not complete")
    if obs.route is None:
        return _check("router_route", False, "no route recorded for this run")
    return _check(
        "router_route",
        obs.route == case.expect_route,
        f"expected {case.expect_route!r}, got {obs.route!r}",
    )


# ---------------------------------------------------------------- the plan

def check_plan_present(case: EvalCase, obs: Observation) -> Check:
    """A request routed to planning must yield a non-empty plan."""
    if case.expect_route != "plan":
        return _skip("plan_present", "case does not expect planning")
    if obs.error:
        return _skip("plan_present", "run did not complete")
    return _check("plan_present", bool(obs.plan_steps), "planner produced no plan")


def _plan_applies(obs: Observation, name: str) -> Check | None:
    if obs.error:
        return _skip(name, "run did not complete")
    if not obs.plan_steps:
        return _skip(name, "no plan in this run")
    return None


def check_plan_names_known(case: EvalCase, obs: Observation) -> Check:
    """Every step names a capability the registry actually holds."""
    name = "plan_names_known"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    unknown = [s["capability"] for s in obs.plan_steps if s["capability"] not in obs.registry]
    return _check(name, not unknown, f"not in the registry: {', '.join(unknown)}")


def check_plan_no_duplicates(case: EvalCase, obs: Observation) -> Check:
    """A capability appears at most once — the results channel is keyed by name."""
    name = "plan_no_duplicates"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    names = [s["capability"] for s in obs.plan_steps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    return _check(name, not dupes, f"repeated steps: {', '.join(dupes)}")


def check_plan_deps_satisfiable(case: EvalCase, obs: Observation) -> Check:
    """Every dependency refers to another step of the same plan."""
    name = "plan_deps_satisfiable"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    present = {s["capability"] for s in obs.plan_steps}
    dangling = sorted({
        d for step in obs.plan_steps for d in step["depends_on"]
        if d not in present or d == step["capability"]
    })
    return _check(name, not dangling, f"unsatisfiable dependencies: {', '.join(dangling)}")


def check_plan_acyclic(case: EvalCase, obs: Observation) -> Check:
    """Kahn's algorithm — a cycle would deadlock the wave executor."""
    name = "plan_acyclic"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    present = {s["capability"] for s in obs.plan_steps}
    indegree = {
        s["capability"]: len([d for d in s["depends_on"] if d in present])
        for s in obs.plan_steps
    }
    successors: dict[str, list[str]] = {s["capability"]: [] for s in obs.plan_steps}
    for step in obs.plan_steps:
        for dep in step["depends_on"]:
            if dep in successors:
                successors[dep].append(step["capability"])
    queue = [n for n, d in indegree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in successors[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return _check(name, visited == len(obs.plan_steps), "the plan contains a cycle")


def check_plan_size(case: EvalCase, obs: Observation) -> Check:
    """The DAG has as many steps as the request's shape calls for."""
    name = "plan_size"
    if case.min_steps <= 0 and case.max_steps is None:
        return _skip(name, "case sets no bounds")
    if (s := _plan_applies(obs, name)) is not None:
        return s
    size = len(obs.plan_steps)
    ok = size >= case.min_steps and (case.max_steps is None or size <= case.max_steps)
    bounds = f"{case.min_steps}..{case.max_steps if case.max_steps is not None else '*'}"
    return _check(name, ok, f"{size} step(s), expected {bounds}")


def check_plan_required_steps(case: EvalCase, obs: Observation) -> Check:
    """Capabilities the request names explicitly are in the plan (when registered)."""
    name = "plan_required_steps"
    wanted = [c for c in case.must_include if c in obs.registry]
    if not wanted:
        return _skip(name, "no required capability is registered here")
    if (s := _plan_applies(obs, name)) is not None:
        return s
    planned = {s["capability"] for s in obs.plan_steps}
    missing = [c for c in wanted if c not in planned]
    return _check(name, not missing, f"missing from the plan: {', '.join(missing)}")


# ---------------------------------------------------------------- execution

def check_steps_all_ran(case: EvalCase, obs: Observation) -> Check:
    """Every planned step produced a result — the executor drained the DAG."""
    name = "steps_all_ran"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    missing = [s["capability"] for s in obs.plan_steps if s["capability"] not in obs.results]
    return _check(name, not missing, f"never reported: {', '.join(missing)}")


def check_steps_all_ok(case: EvalCase, obs: Observation) -> Check:
    """No capability failed. Failures degrade gracefully by design — they are
    still the clearest signal that a capability's own prompt broke."""
    name = "steps_all_ok"
    if (s := _plan_applies(obs, name)) is not None:
        return s
    failed = sorted(n for n, r in obs.results.items() if not r.get("ok"))
    return _check(name, not failed, f"failed step(s): {', '.join(failed)}")


# ---------------------------------------------------------------- deliverable

def _report_applies(case: EvalCase, obs: Observation, name: str) -> Check | None:
    if case.expect_terminal != "answer":
        return _skip(name, "case expects no deliverable")
    if obs.error:
        return _skip(name, "run did not complete")
    if obs.terminal_kind != "answer":
        return _skip(name, "the run produced no answer to report")
    return None


def check_report_written(case: EvalCase, obs: Observation) -> Check:
    """A successful job leaves a deliverable on disk."""
    name = "report_written"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    return _check(name, bool(obs.report_text), "no report file was produced")


def _report_body(obs: Observation) -> str:
    return obs.report_text or ""


def check_report_title(case: EvalCase, obs: Observation) -> Check:
    """It opens with a non-empty top-level title."""
    name = "report_title"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    first = next((ln for ln in _report_body(obs).splitlines() if ln.strip()), "")
    return _check(
        name,
        first.startswith("# ") and len(first[2:].strip()) > 0,
        f"first line is {first[:60]!r}",
    )


def check_report_answer(case: EvalCase, obs: Observation) -> Check:
    """The answer the job settled on is actually in the deliverable."""
    name = "report_answer"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    answer = (obs.final_answer or "").strip()
    if len(answer) < 20:
        return _check(name, False, f"final answer is {len(answer)} chars")
    probe = answer[:80]
    return _check(name, probe in _report_body(obs), "the final answer is not in the report")


def check_report_provenance(case: EvalCase, obs: Observation) -> Check:
    """It says what was asked and which run produced it."""
    name = "report_provenance"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    body = _report_body(obs)
    missing = [
        label for label, needle in (("job id", obs.job_id), ("request", obs.query.strip()[:60]))
        if needle and needle not in body
    ]
    return _check(name, not missing, f"no {', no '.join(missing)} in the report")


def check_report_covers_plan(case: EvalCase, obs: Observation) -> Check:
    """Every executed step is accounted for in the deliverable."""
    name = "report_covers_plan"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    if not obs.plan_steps:
        return _skip(name, "no plan in this run")
    body = _report_body(obs)
    missing = [s["capability"] for s in obs.plan_steps if s["capability"] not in body]
    return _check(name, not missing, f"unmentioned step(s): {', '.join(missing)}")


CHECKS: tuple[Callable[[EvalCase, Observation], Check], ...] = (
    check_run_completed,
    check_router_route,
    check_terminal,
    check_plan_present,
    check_plan_names_known,
    check_plan_no_duplicates,
    check_plan_deps_satisfiable,
    check_plan_acyclic,
    check_plan_size,
    check_plan_required_steps,
    check_steps_all_ran,
    check_steps_all_ok,
    check_report_written,
    check_report_title,
    check_report_answer,
    check_report_provenance,
    check_report_covers_plan,
)

CHECK_NAMES: tuple[str, ...] = (
    "run_completed",
    "router_route",
    "terminal_kind",
    "plan_present",
    "plan_names_known",
    "plan_no_duplicates",
    "plan_deps_satisfiable",
    "plan_acyclic",
    "plan_size",
    "plan_required_steps",
    "steps_all_ran",
    "steps_all_ok",
    "report_written",
    "report_title",
    "report_answer",
    "report_provenance",
    "report_covers_plan",
)


def score(case: EvalCase, obs: Observation) -> list[Check]:
    """Run every check against one observation, in a stable order."""
    return [check(case, obs) for check in CHECKS]


def step_failure_rate(obs: Observation) -> float | None:
    """Fraction of executed capability steps that reported failure."""
    if not obs.results:
        return None
    failed = sum(1 for r in obs.results.values() if not r.get("ok"))
    return failed / len(obs.results)
