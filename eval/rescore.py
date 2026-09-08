"""Re-score a completed sweep from its stored traces, with no API calls.

    python -m eval.rescore --report eval/reports/eval-gpt-5.6-luna-<stamp>.json

A scorer change should not cost a sweep. Every input the metrics need was
persisted at run time -- the runs table holds the answer, route, sources,
insufficiency flag, terminal reason, iterations and token counts, and the steps
table holds every tool call with its outcome class -- so a corrected scorer can
be applied to runs that already happened.

This is also the strongest test of M5's claim that a run is reconstructable from
its trace alone. It either is, or this file cannot exist.

The one thing not reconstructable is the message array, which the trace holds no
column for by design. No metric reads it: the metrics score what was requested
and what happened, which is exactly what a trace records.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentharness.harness.loop import FinalAnswerRoute, RunResult
from agentharness.harness.model_client import TokenUsage
from agentharness.harness.policy import TerminalReason
from agentharness.store.runs import RunStore
from eval.metrics import ScoredRun, aggregate, score_run
from eval.runner import REPORTS, load_cases, percentile, render_case_table, render_table


def rebuild(row: dict[str, Any]) -> RunResult:
    """A RunResult from a trace row. Messages are not stored and not needed."""
    return RunResult(
        run_id=row["run_id"],
        answer=row["answer"],
        terminal_reason=TerminalReason[row["terminal_reason"]],
        iterations=row["iterations"] or 0,
        usage=TokenUsage(
            prompt_tokens=row["prompt_tokens"] or 0,
            completion_tokens=row["completion_tokens"] or 0,
            cached_tokens=row["cached_tokens"] or 0,
            cache_write_tokens=row["cache_write_tokens"] or 0,
        ),
        messages=[],
        route=FinalAnswerRoute[row["route"]] if row["route"] else FinalAnswerRoute.NONE,
        sources=json.loads(row["sources"] or "[]"),
        insufficient_information=bool(row["insufficient_information"]),
    )


def rescore(report: dict[str, Any], store: RunStore) -> dict[str, Any]:
    cases = {case["name"]: case for case in load_cases()}
    rebuilt: dict[str, Any] = dict(report)
    rebuilt["cases"] = []
    scored: list[ScoredRun] = []
    missing: list[str] = []

    for case_report in report["cases"]:
        case = cases.get(case_report["case"])
        if case is None:
            missing.append(f"{case_report['case']} is no longer in cases.yaml")
            continue

        detail = []
        case_scored: list[ScoredRun] = []
        for run in case_report["detail"]:
            run_id = run.get("run_id")
            row = store.get_run(run_id) if run_id else None
            if row is None:
                missing.append(f"{run_id} is not in the database")
                continue
            steps = store.get_steps(run_id)
            result = rebuild(row)
            run_score = score_run(case, result, steps)
            case_scored.append(run_score)
            scored.append(run_score)
            detail.append(
                dict(
                    run,
                    observed_outcome=run_score.observed_outcome,
                    outcome_reason=run_score.outcome_reason,
                    metrics={
                        name: {"value": r.value, "reason": r.reason}
                        for name, r in run_score.metrics.items()
                    },
                )
            )

        rebuilt["cases"].append(
            dict(case_report, detail=detail, summary=aggregate(case_scored) if case_scored else {})
        )

    rebuilt["overall"] = aggregate(scored)
    rebuilt["outcome_counts"] = rebuilt["overall"]["outcomes"]
    rebuilt["route_counts"] = rebuilt["overall"]["routes"]
    rebuilt["rescored_from"] = report.get("started_at")
    rebuilt["rescore_warnings"] = missing
    return rebuilt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--db", default="./eval.db")
    parser.add_argument("--write", action="store_true", help="save the rescored report")
    arguments = parser.parse_args(argv)

    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    store = RunStore(arguments.db)
    rebuilt = rescore(report, store)

    if rebuilt["rescore_warnings"]:
        print("warnings:", file=sys.stderr)
        for warning in rebuilt["rescore_warnings"]:
            print(f"  {warning}", file=sys.stderr)

    print(render_table(rebuilt))
    print()
    print(render_case_table(rebuilt))

    if arguments.write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / f"{arguments.report.stem}-rescored.json"
        path.write_text(json.dumps(rebuilt, indent=2), encoding="utf-8")
        print(f"\nreport  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
