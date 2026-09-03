"""`python -m evals` — run the golden set and print a comparable summary.

    python -m evals                          # tier inferred from the provider
    python -m evals --llm anthropic          # llm tier against a real model
    python -m evals --repeat 3 --case plan_compare_and_recommend
    python -m evals --baseline evals/results/<file>.json

Exit status is a gate only where a gate is meaningful. On the deterministic
fakes anything under 100% is a regression, so the run exits non-zero; against a
real model the run always exits 0 unless `--fail-under` is given explicitly.
The condition is the *provider*, not the tier label: determinism is what makes
a gate honest, and it is what lets CI run this at all.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from jobsmith.app.providers import load_dotenv

from .cases import GOLDEN_CASES, LLM, STRUCTURAL, cases_for
from .harness import resolve_provider, run_suite
from .results import (
    RESULTS_DIR,
    load_baseline,
    load_result,
    render_summary,
    summarize,
    write_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__.splitlines()[0])
    parser.add_argument("--llm", choices=["anthropic", "openai", "fake"],
                        help="provider (default: auto-detected from the available keys)")
    parser.add_argument("--agent", help="which agent to evaluate (default: the default one)")
    parser.add_argument("--tier", choices=[STRUCTURAL, LLM],
                        help="case tier (default: structural for the fake provider, llm otherwise)")
    parser.add_argument("--case", action="append", default=[], metavar="ID",
                        help="only this case id (repeatable)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run every case N times — how you tell a real move from noise")
    parser.add_argument("-j", "--concurrency", type=int, default=1,
                        help="cases to run in parallel (default: 1)")
    parser.add_argument("--fail-under", type=float, metavar="RATE",
                        help="exit non-zero below this overall pass rate (0..1)")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help=f"where run records are written (default: {RESULTS_DIR})")
    parser.add_argument("--reports-dir",
                        help="keep the generated job reports here (default: a scratch dir)")
    parser.add_argument("--baseline", help="compare against this results file instead of the latest")
    parser.add_argument("--no-write", action="store_true", help="print only, store nothing")
    parser.add_argument("--list", action="store_true", help="list the golden set and exit")
    return parser


def _list_cases() -> None:
    for case in GOLDEN_CASES:
        tiers = "+".join(case.tiers)
        route = case.expect_route or "—"
        print(f"{case.id:<32} route={route:<7} terminal={case.expect_terminal:<11} tiers={tiers}")
        if case.note:
            print(f"{'':<32} {case.note}")


async def _run(args: argparse.Namespace) -> int:
    # Resolve the provider first: the tier follows from it unless forced, and
    # the fakes are exactly the case where the deterministic tier applies.
    if args.llm is None:
        # Same as the CLI: a key in .env must reach the auto-detection, or
        # `make eval-llm` reports "fake" on a machine that has one. Only when
        # nothing was forced — an explicit choice must not touch the environment.
        load_dotenv()
    provider = resolve_provider(args.llm)
    tier = args.tier or (STRUCTURAL if provider == "fake" else LLM)
    cases = cases_for(tier, only=tuple(args.case))
    started = time.perf_counter()

    if not cases:
        print(f"no case matches tier={tier}", file=sys.stderr)
        return 2

    observations, context = await run_suite(
        cases,
        agent=args.agent,
        provider=provider,
        repeat=args.repeat,
        concurrency=args.concurrency,
        reports_dir=args.reports_dir,
    )
    result = summarize(cases, observations, tier=tier, context=context,
                       duration_s=time.perf_counter() - started)

    written: Path | None = None
    if not args.no_write:
        written = write_result(result, args.results_dir)
    baseline = (
        load_result(args.baseline) if args.baseline
        else load_baseline(result, args.results_dir, exclude=written)
    )

    print(render_summary(result, baseline))
    if written is not None:
        print(f"\nstored: {written}")

    floor = args.fail_under
    if floor is None and provider == "fake":
        floor = 1.0            # deterministic: anything less than perfect is a regression
    if floor is not None and (result.pass_rate is None or result.pass_rate < floor):
        print(f"\nFAIL: pass rate {result.pass_rate} is under {floor}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        _list_cases()
        return 0
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
