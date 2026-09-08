"""The evaluation harness: cases in, rates out.

    python -m eval.runner --model gpt-5.6-luna --runs 3
    python -m eval.runner --model gpt-5.6-sol --runs 3 --project-from eval/reports/<luna>.json

Offline and stochastic by design. Every case runs n times because temperature 0
is not determinism, and a single run is an anecdote. Excluded from CI.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from agentharness.domain import repository
from agentharness.harness.loop import AgentLoop, RunResult
from agentharness.harness.model_client import HarnessFatalError, ModelClient
from agentharness.harness.registry import build_registry
from agentharness.harness.tracer import SqliteTracer
from agentharness.store.runs import RunStore
from eval.metrics import (
    METRICS,
    OBSERVED_NOT_SCORED,
    ScoredRun,
    aggregate,
    percentile,
    score_run,
)
from eval.pricing import cost_of, price_note

CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
REPORTS = Path(__file__).resolve().parent / "reports"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def loop_factory(model: str, store: RunStore) -> Callable[[], AgentLoop]:
    registry = build_registry()

    def build() -> AgentLoop:
        client = ModelClient(OpenAI(), model=model)
        return AgentLoop(client, registry, tracer=SqliteTracer(store))

    return build


def execute(build_loop: Callable[[], AgentLoop], goal: str, store: RunStore):
    """One run, plus the trace rows it produced.

    A run that dies fatally is reported rather than crashing the sweep: losing
    ninety runs because the ninety-first hit a 400 would be its own failure.
    """
    loop = build_loop()
    try:
        result = loop.run(goal)
    except HarnessFatalError as error:
        return None, [], f"{type(error).__name__}: {error}"
    return result, store.get_steps(result.run_id), None


def run_case(
    case: dict[str, Any],
    build_loop: Callable[[], AgentLoop],
    store: RunStore,
    runs: int,
    model: str,
) -> dict[str, Any]:
    scored: list[ScoredRun] = []
    raw: list[dict[str, Any]] = []

    for index in range(runs):
        repository.reset()
        result, steps, error = execute(build_loop, case["goal"], store)
        if result is None:
            raw.append({"run_id": None, "error": error})
            print(f"  run {index + 1}/{runs}: ERROR {error}", file=sys.stderr)
            continue

        run_score = score_run(case, result, steps)
        scored.append(run_score)
        row = store.get_run(result.run_id) or {}
        usage = result.usage
        raw.append(
            {
                "run_id": result.run_id,
                "observed_outcome": run_score.observed_outcome,
                "outcome_reason": run_score.outcome_reason,
                "route": run_score.route,
                "terminal_reason": result.terminal_reason.name,
                "iterations": result.iterations,
                "duration_ms": row.get("duration_ms"),
                "tools_called": [
                    s["tool_name"] for s in steps if s["kind"] == "tool_call"
                ],
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost": cost_of(
                    model, usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens
                ),
                "metrics": {
                    name: {"value": r.value, "reason": r.reason}
                    for name, r in run_score.metrics.items()
                },
            }
        )
        print(
            f"  run {index + 1}/{runs}: {run_score.observed_outcome} "
            f"({run_score.route}, {result.iterations} iters)",
            file=sys.stderr,
        )

    latencies = [r["duration_ms"] for r in raw if r.get("duration_ms")]
    return {
        "case": case["name"],
        "goal": case["goal"],
        "runs": runs,
        "completed_runs": len(scored),
        "summary": aggregate(scored) if scored else {},
        "mean_iterations": statistics.fmean(
            [r["iterations"] for r in raw if r.get("iterations")]
        )
        if any(r.get("iterations") for r in raw)
        else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "tokens": {
            "prompt": sum(r.get("prompt_tokens", 0) or 0 for r in raw),
            "cached": sum(r.get("cached_tokens", 0) or 0 for r in raw),
            "completion": sum(r.get("completion_tokens", 0) or 0 for r in raw),
        },
        "cost": sum(r.get("cost") or 0.0 for r in raw) if any(r.get("cost") for r in raw) else None,
        "detail": raw,
    }


# --- reporting ----------------------------------------------------------------

RATE_METRICS = (
    "task_completion",
    "declared_insufficiency",
    "tool_precision",
    "tool_recall",
    "unnecessary_call_rate",
    "forbidden_tool_avoided",
    "within_call_budget",
    "grounding",
    "refusal_correctness",
    "fabricated_citation",
    "recovery",
)


def render_table(report: dict[str, Any]) -> str:
    lines = [
        f"{report['model']}  |  {report['runs_per_case']} runs per case  |  "
        f"{len(report['cases'])} cases  |  {report['price_note']}",
        "",
        "| metric | mean | stdev | scored | note |",
        "|---|---|---|---|---|",
    ]
    overall = report["overall"]
    for name in RATE_METRICS:
        entry = overall.get(name) or {}
        mean = entry.get("mean")
        if mean is None:
            lines.append(f"| {name} | n/a | - | 0 | never applicable |")
            continue
        note = ""
        if name in OBSERVED_NOT_SCORED:
            note = "REPORTED, not a pass criterion -- see DECISIONS"
        elif entry.get("reads_submit_fields"):
            share = entry.get("free_text_share") or 0.0
            note = f"reads submit-tool fields; {share:.0%} of runs were FREE_TEXT"
        lines.append(
            f"| {name} | {mean:.2f} | {entry.get('stdev', 0.0):.2f} | "
            f"{entry['scored_runs']}/{entry['total_runs']} | {note or '-'} |"
        )

    totals = report["totals"]
    lines += [
        "",
        f"mean iterations   {totals['mean_iterations']:.1f}"
        if totals.get("mean_iterations")
        else "mean iterations   n/a",
        f"latency p50/p95   {totals['p50_latency_ms']:.0f}ms / {totals['p95_latency_ms']:.0f}ms"
        if totals.get("p50_latency_ms")
        else "latency p50/p95   n/a",
        f"tokens            {totals['prompt_tokens']:,} prompt "
        f"({totals['cached_tokens']:,} cached, "
        f"{totals['prompt_tokens'] - totals['cached_tokens']:,} uncached), "
        f"{totals['completion_tokens']:,} completion",
        f"cost              ${totals['cost']:.4f}" if totals.get("cost") is not None else "cost              n/a",
        "",
        "outcomes          "
        + ", ".join(f"{k} {v}" for k, v in sorted(report["outcome_counts"].items()) if v),
        "routes            "
        + ", ".join(f"{k} {v}" for k, v in sorted(report["route_counts"].items())),
    ]
    return "\n".join(lines)


def render_case_table(report: dict[str, Any]) -> str:
    lines = ["| case | completion | outcome(s) observed | tools used |", "|---|---|---|---|"]
    for case in report["cases"]:
        completion = (case["summary"].get("task_completion") or {}).get("mean")
        outcomes: dict[str, int] = {}
        for run in case["detail"]:
            key = run.get("observed_outcome") or "ERROR"
            outcomes[key] = outcomes.get(key, 0) + 1
        tools = sorted(
            {
                tool
                for run in case["detail"]
                for tool in run.get("tools_called", [])
                if tool != "submit_final_answer"
            }
        )
        lines.append(
            f"| {case['case']} "
            f"| {'n/a' if completion is None else format(completion, '.2f')} "
            f"| {', '.join(f'{k} {v}' for k, v in sorted(outcomes.items()))} "
            f"| {', '.join(tools) or '-'} |"
        )
    return "\n".join(lines)


def project_cost(prior_report_path: Path, model: str) -> str:
    """What the same sweep would cost on another tier, before it is started."""
    prior = json.loads(prior_report_path.read_text(encoding="utf-8"))
    totals = prior["totals"]
    projected = cost_of(
        model, totals["prompt_tokens"], totals["cached_tokens"], totals["completion_tokens"]
    )
    return (
        f"projection from {prior_report_path.name} ({prior['model']}): "
        f"{totals['prompt_tokens']:,} prompt / {totals['completion_tokens']:,} completion "
        f"tokens would cost about ${projected:.2f} on {model}, against "
        f"${totals['cost']:.4f} actually spent on {prior['model']}"
    )


def main(argv: list[str] | None = None) -> int:
    cases = load_cases()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3, help="runs per case, n>=3")
    parser.add_argument("--db", default="./eval.db")
    parser.add_argument(
        "--case", action="append", choices=[case["name"] for case in cases],
        help="run only this case; repeatable",
    )
    parser.add_argument(
        "--project-from",
        type=Path,
        help="print what this sweep will cost, using a prior report's token counts",
    )
    arguments = parser.parse_args(argv)

    if arguments.project_from:
        print(project_cost(arguments.project_from, arguments.model))
        print()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in the environment", file=sys.stderr)
        return 1

    selected = [c for c in cases if not arguments.case or c["name"] in arguments.case]
    store = RunStore(arguments.db)
    build_loop = loop_factory(arguments.model, store)

    report: dict[str, Any] = {
        "model": arguments.model,
        "runs_per_case": arguments.runs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "price_note": price_note(arguments.model),
        "database": arguments.db,
        "cases": [],
    }
    for case in selected:
        print(case["name"], file=sys.stderr)
        report["cases"].append(run_case(case, build_loop, store, arguments.runs, arguments.model))

    every_run = []
    scored = []
    for case in report["cases"]:
        for run in case["detail"]:
            if not run.get("run_id"):
                continue
            every_run.append(run)
            scored.append(score_run_from_detail(case["case"], run))
    report["overall"] = aggregate(scored)
    latencies = [r["duration_ms"] for r in every_run if r.get("duration_ms")]
    report["totals"] = {
        "prompt_tokens": sum(r["prompt_tokens"] for r in every_run),
        "cached_tokens": sum(r["cached_tokens"] for r in every_run),
        "completion_tokens": sum(r["completion_tokens"] for r in every_run),
        "cost": sum(r["cost"] or 0.0 for r in every_run) if every_run and every_run[0]["cost"] is not None else None,
        "mean_iterations": statistics.fmean([r["iterations"] for r in every_run]) if every_run else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "runs": len(every_run),
    }
    report["outcome_counts"] = report["overall"]["outcomes"]
    report["route_counts"] = report["overall"]["routes"]

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"eval-{arguments.model}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(render_table(report))
    print()
    print(render_case_table(report))
    print()
    print(f"report  {path}")
    return 0


def score_run_from_detail(case_name: str, run: dict[str, Any]) -> ScoredRun:
    """Rebuild a ScoredRun from what was written, for the overall aggregate."""
    from eval.metrics import MetricResult

    return ScoredRun(
        run_id=run["run_id"],
        case=case_name,
        observed_outcome=run["observed_outcome"],
        outcome_reason=run["outcome_reason"],
        route=run["route"],
        metrics={
            name: MetricResult(value=entry["value"], reason=entry["reason"])
            for name, entry in run["metrics"].items()
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
