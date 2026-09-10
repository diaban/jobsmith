"""Property checks over one observed run.

Every check answers a yes/no question about *structure*, never about wording:
an LLM that phrases its answer differently must not move the score, an LLM
that emits a plan with a cycle must. One exception, and it declares itself:
`report_reader_facing` asks WHO the deliverable is addressed to, and a
document's register lives nowhere but in its words (#58). It is a floor under
the prompts, not a judgement of the prose.

A check reports one of three statuses:

    pass / fail   the property was evaluated
    skip          the property does not apply to this run (a direct-route run
                  has no plan to validate; a rejected query has no report)

Skipped checks are excluded from the denominator, so adding a case that
exercises one path never dilutes the score of another.

The report checks are deliberately layout-independent — they look for the
title, the answer text, the job id and the request, not for the headings
`MarkdownReport` happens to use today. A reformatting of the deliverable
should not read as a regression; losing its provenance should. That holds
across *formats* too: they read the file through `deliverable.extract`,
which hands back the title and the visible text with the markup stripped,
so the same check scores a markdown and an HTML report identically. It used
to be true only within markdown — `# ` is not how HTML opens, and escaping
moved the very strings the checks searched for.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jobsmith.core.state import TERMINAL_UNANSWERED
from jobsmith.jobs.report import UNANSWERED_NOTICE

from .cases import EvalCase
from .deliverable import Deliverable, extract, normalize
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


def check_plan_excluded_steps(case: EvalCase, obs: Observation) -> Check:
    """Capabilities the request does NOT call for stay out of the plan.

    The mirror of `check_plan_required_steps`, and it exists because a spec
    that is too easy to reach is invisible everywhere else: the run is green,
    every step reports ok, and the user opens the wrong kind of file. #61 was
    exactly that — a request for a printable one-page PDF planned `slide_deck`
    and delivered a PowerPoint, because a deck was the closest thing in the
    registry and its description did not say what it was not.

    Skipped when none of the named capabilities is registered here, for the
    same reason as its mirror: a registry is agent- and configuration-
    dependent, and "the plan omits a step this agent could never plan" is not
    a measurement.
    """
    name = "plan_excluded_steps"
    unwanted = [c for c in case.must_exclude if c in obs.registry]
    if not unwanted:
        return _skip(name, "no excluded capability is registered here")
    if (s := _plan_applies(obs, name)) is not None:
        return s
    planned = {s["capability"] for s in obs.plan_steps}
    present = [c for c in unwanted if c in planned]
    return _check(name, not present, f"should not have been planned: {', '.join(present)}")


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


def _deliverable(obs: Observation) -> Deliverable:
    """The report as title + visible text, whichever Reporter wrote it."""
    return extract(obs.report_text, obs.report_format)


def check_report_title(case: EvalCase, obs: Observation) -> Check:
    """It opens with a non-empty top-level title."""
    name = "report_title"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    title = _deliverable(obs).title
    return _check(name, bool(title.strip()), "the deliverable announces no title")


def check_report_answer(case: EvalCase, obs: Observation) -> Check:
    """The answer the job settled on is actually in the deliverable."""
    name = "report_answer"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    answer = (obs.final_answer or "").strip()
    if len(answer) < 20:
        return _check(name, False, f"final answer is {len(answer)} chars")
    probe = answer[:80]
    return _check(name, _deliverable(obs).contains(probe),
                  "the final answer is not in the report")


def check_report_provenance(case: EvalCase, obs: Observation) -> Check:
    """It says what was asked and which run produced it."""
    name = "report_provenance"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    doc = _deliverable(obs)
    missing = [
        label for label, needle in (("job id", obs.job_id), ("request", obs.query.strip()[:60]))
        if needle and not doc.contains(needle)
    ]
    return _check(name, not missing, f"no {', no '.join(missing)} in the report")


def check_report_covers_plan(case: EvalCase, obs: Observation) -> Check:
    """Every executed step is accounted for in the deliverable."""
    name = "report_covers_plan"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    if not obs.plan_steps:
        return _skip(name, "no plan in this run")
    doc = _deliverable(obs)
    missing = [s["capability"] for s in obs.plan_steps if not doc.contains(s["capability"])]
    return _check(name, not missing, f"unmentioned step(s): {', '.join(missing)}")


#: Phrases that only make sense if the document is addressed to whoever
#: PRODUCED it: a request for input, a decision left to the reader, a note on
#: what the work still needs. Taken from the two runs that opened #58 — a
#: deck whose slides were *Prochaines étapes* and *Option A/B selon la
#: préférence*, handed to someone who had asked about a subject — plus their
#: English equivalents, since the deliverable is written in the language of
#: the request and both were observed in French.
PRODUCER_FACING_MARKERS: tuple[str, ...] = (
    "next steps",
    "prochaines étapes",
    "open questions",
    "questions ouvertes",
    "to be confirmed",
    "à confirmer",
    "please confirm",
    "let me know",
    "would you like",
    "do you want me to",
    "option a or option b",
    "option a/b",
    "template to fill",
    "template prêt à remplir",
    "share the file",
    "partager le fichier",
)


def check_report_reader_facing(case: EvalCase, obs: Observation) -> Check:
    """The deliverable is written for its reader, not for its producer.

    The one check here that looks at *wording*, and it has to be: register is
    only observable in words. What it detects is not a phrasing preference but
    a document addressed to the wrong person — the failure of #58, where a
    request for a printable one-pager came back as a status report on the run,
    with next steps and an A/B option for the reader to pick. Every step
    upstream of the deliverable writes for the run (`critique` says so in its
    own prompt), so this is the register the prompts have to hold back, and a
    smoke detector for it is worth more than nothing.

    It reads the **answer**, not the file: the provenance the Reporter adds is
    *about* the run by design, and it quotes the request, so scanning the whole
    document would fire on the scaffolding and on the user's own words.
    `check_report_answer` already pins that this text is what the deliverable
    carries.

    Honest about its limit: a marker list catches what it lists. It is a floor
    under the prompts, not a proof of good register.
    """
    name = "report_reader_facing"
    if (s := _report_applies(case, obs, name)) is not None:
        return s
    answer = normalize(obs.final_answer or "").lower()
    hits = [m for m in PRODUCER_FACING_MARKERS if m in answer]
    return _check(name, not hits, f"addressed to the producer: {', '.join(hits)}")


def check_refusal_declared(case: EvalCase, obs: Observation) -> Check:
    """A run that could not answer says so in the file, not only in its prose.

    The generator declares an unanswerable request as data and the run gets
    its own terminal (#59); the deliverable is where that declaration reaches
    the person who waited for it. A run that answered has nothing to declare,
    so it skips — which is every case of the structural tier, the keyword fake
    having no notion of insufficient material. It scores where it matters: on
    a real provider, a report that reads like a report while the run gave up.
    """
    name = "refusal_declared"
    if obs.error:
        return _skip(name, "run did not complete")
    if obs.terminal_kind != TERMINAL_UNANSWERED:
        return _skip(name, "the run answered")
    if not obs.report_text:
        return _check(name, False, "no deliverable to declare it in")
    return _check(name, _deliverable(obs).contains(UNANSWERED_NOTICE),
                  "the deliverable does not say the run could not answer")


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
    check_plan_excluded_steps,
    check_steps_all_ran,
    check_steps_all_ok,
    check_report_written,
    check_report_title,
    check_report_answer,
    check_report_provenance,
    check_report_covers_plan,
    check_report_reader_facing,
    check_refusal_declared,
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
    "plan_excluded_steps",
    "steps_all_ran",
    "steps_all_ok",
    "report_written",
    "report_title",
    "report_answer",
    "report_provenance",
    "report_covers_plan",
    "report_reader_facing",
    "refusal_declared",
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
