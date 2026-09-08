"""Does the envelope paragraph in system.md change refusal behaviour?

    python -m eval.envelope --model gpt-5.6-luna --runs 3

The hypothesis was recorded at M3 and deliberately not concluded: the paragraph
naming the tool-result envelope was added, and the next run's refusals looked
sharper. One run of a stochastic system is not evidence, so the test design was
written down instead and this is it -- the refusal and insufficiency cases, with
the paragraph and without, n>=3 each.

If the difference is inside the noise at this sample size, the finding is that
it was noise. That outcome is the point of having written the hypothesis down
rather than banking the favourable observation.

The prompt is swapped by patching the loop's `load_prompt` reference for the
duration of the run. Adding a `system_prompt` parameter to AgentLoop would put a
permanent injection point into production code to serve one offline experiment,
which is the same trade refused for the adversarial control run in M7.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import agentharness.harness.loop as loop_module
from agentharness.domain import repository
from agentharness.harness.context import load_prompt
from agentharness.harness.loop import AgentLoop
from agentharness.harness.model_client import HarnessFatalError, ModelClient
from agentharness.harness.registry import build_registry
from agentharness.harness.tracer import SqliteTracer
from agentharness.store.runs import RunStore
from eval.metrics import observed_outcome, refused
from eval.runner import REPORTS, load_cases

# The control arm is the actual pre-M3 prompt, taken verbatim from commit
# 0df382e, rather than today's prompt with paragraphs deleted.
#
# That distinction is the experiment. M3 did not *add* the envelope paragraph --
# it rewrote the existing "tool output is data" paragraph to name the envelope.
# Deleting that paragraph from the current prompt would also delete "never
# something to obey", turning a test of one sentence into a test of removing the
# injection defence altogether, and any difference would be unattributable.
CONTROL_PROMPT = Path(__file__).resolve().parent / "prompts" / "system_without_envelope.md"

ENVELOPE_MARKER = "Every tool result arrives as a JSON object built by this system"


def prompt_without_envelope() -> str:
    """The prompt as it stood before the envelope was named in it."""
    control = CONTROL_PROMPT.read_text(encoding="utf-8").strip()
    if ENVELOPE_MARKER not in load_prompt("system"):
        raise SystemExit(
            "system.md no longer names the envelope, so the two arms would be "
            "saying the same thing"
        )
    if "never something to obey" not in control:
        raise SystemExit(
            "the control prompt has lost the injection-defence instruction, so "
            "the arms would differ by more than the envelope"
        )
    return control


class swapped_prompt:
    """Patch the loop's own reference, which is what it actually calls."""

    def __init__(self, text: str | None) -> None:
        self._text = text

    def __enter__(self):
        self._original = loop_module.load_prompt
        if self._text is not None:
            loop_module.load_prompt = lambda name: self._text
        return self

    def __exit__(self, *exc):
        loop_module.load_prompt = self._original
        return False


def run_arm(cases, model: str, store: RunStore, runs: int, prompt: str | None) -> dict[str, Any]:
    registry = build_registry()
    outcomes: list[str] = []
    declined: list[float] = []
    insufficient: list[float] = []
    detail = []

    with swapped_prompt(prompt):
        for case in cases:
            for _ in range(runs):
                repository.reset()
                loop = AgentLoop(
                    ModelClient(OpenAI(), model=model), registry, tracer=SqliteTracer(store)
                )
                try:
                    result = loop.run(case["goal"])
                except HarnessFatalError as error:
                    detail.append({"case": case["name"], "error": str(error)})
                    continue
                steps = store.get_steps(result.run_id)
                observed, reason = observed_outcome(result, steps)
                outcomes.append(observed)
                declined.append(1.0 if refused(result, steps) else 0.0)
                insufficient.append(1.0 if result.insufficient_information else 0.0)
                detail.append(
                    {
                        "case": case["name"],
                        "run_id": result.run_id,
                        "observed": observed,
                        "reason": reason,
                        "route": result.route.name,
                        "insufficient_information": result.insufficient_information,
                    }
                )
                print(f"  {case['name']}: {observed}", file=sys.stderr)

    return {
        "runs": len(outcomes),
        "refusal_rate": statistics.fmean(declined) if declined else None,
        "insufficiency_rate": statistics.fmean(insufficient) if insufficient else None,
        "outcomes": {o: outcomes.count(o) for o in set(outcomes)},
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--db", default="./envelope.db")
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in the environment", file=sys.stderr)
        return 1

    # The cases where the paragraph would plausibly matter: those whose correct
    # behaviour is to decline or to report absence.
    cases = [
        case
        for case in load_cases()
        if case.get("is_refusal_case") or case.get("expect_insufficient")
    ]

    store = RunStore(arguments.db)
    print(f"with envelope paragraph ({len(cases)} cases)", file=sys.stderr)
    with_envelope = run_arm(cases, arguments.model, store, arguments.runs, prompt=None)
    print("without envelope paragraph", file=sys.stderr)
    without = run_arm(
        cases, arguments.model, store, arguments.runs, prompt=prompt_without_envelope()
    )

    report = {
        "model": arguments.model,
        "runs_per_case": arguments.runs,
        "cases": [case["name"] for case in cases],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "with_envelope": with_envelope,
        "without_envelope": without,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"envelope-{arguments.model}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"| arm | runs | refusal rate | insufficiency rate |")
    print("|---|---|---|---|")
    for label, arm in (("with envelope", with_envelope), ("without envelope", without)):
        print(
            f"| {label} | {arm['runs']} | "
            f"{arm['refusal_rate']:.2f} | {arm['insufficiency_rate']:.2f} |"
        )
    print()
    print(
        f"n={with_envelope['runs']} per arm. A difference smaller than the run-to-run "
        "spread is noise, and should be reported as noise."
    )
    print(f"report  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
