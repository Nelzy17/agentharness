"""Render a stored run as readable terminal output.

    python -m agentharness.cli.trace <run_id>

Imports the store and the standard library, and nothing else. A trace that can
only be read by the code that wrote it is not a record, it is a cache -- so the
renderer works from rows and JSON text, with no harness type in sight.

The format supersedes the sketch referred to in DESIGN: same spine -- step,
tool, arguments, outcome, duration, totals -- with the terminal reason and the
cache figures added, because those are the two things you actually open a trace
to find.
"""

import argparse
import json
import os
import sys
from typing import Any

from agentharness.store.runs import RunStore

DEFAULT_DB_PATH = "./agentharness.db"
ARGUMENT_PREVIEW = 90

# ASCII only. This prints to a terminal, and a Windows console defaults to
# cp1252, where an arrow or a middot raises UnicodeEncodeError instead of
# rendering. A trace renderer that crashes on the platform it runs on is not a
# renderer.


def render(run: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    lines = [
        f"run {run['run_id']}  |  {run['model']}  |  {run['iterations'] or 0} iterations"
        f"  |  {_seconds(run['duration_ms'])}  |  {_tokens(run)}",
        f"goal    {run['goal']}",
        # Principle 9 says every terminal state has a human-readable reason.
        # This is the line that makes that pay off: it is why the trace was
        # opened, so it does not get abbreviated to an enum name.
        f"why     {run['terminal_reason'] or 'unfinished'}: {run['reason_text'] or ''}",
    ]
    if run["error"]:
        lines.append("error   " + "\n        ".join(str(run["error"]).splitlines()))
    lines.append("")

    for step in steps:
        if step["kind"] == "model_call":
            retry = f"  (attempt {step['attempts']})" if (step["attempts"] or 1) > 1 else ""
            lines.append(
                f"{step['iteration']:>3}  model  {_ms(step['duration_ms']):>8}"
                f"  {step['prompt_tokens']:>6} -> {step['completion_tokens']:<5}"
                f"  {_cached(step)}{retry}"
            )
        else:
            flag = "  [WRITE]" if step["is_write"] else ""
            lines.append(
                f"     tool   {_ms(step['duration_ms']):>8}  {step['tool_name']}"
                f"{flag}  {_arguments(step['arguments_json'])}"
            )
            lines.append(f"{'':>22}-> {step['outcome_class']}")
            if step["error_detail"]:
                first = str(step["error_detail"]).strip().splitlines()[-1]
                lines.append(f"{'':>22}  {first}")

    lines.append("")
    if run["answer"]:
        lines.append(f"answer  {run['answer']}")
    sources = json.loads(run["sources"] or "[]")
    lines.append(f"sources {', '.join(sources) if sources else '(none cited)'}")
    if run["insufficient_information"]:
        lines.append("        the answer declares the information insufficient")
    return "\n".join(lines)


def _seconds(duration_ms: float | None) -> str:
    return "-" if duration_ms is None else f"{duration_ms / 1000:.1f}s"


def _ms(duration_ms: float | None) -> str:
    """Enough precision that a fast call reads as fast rather than as nothing.

    A trace that prints 0ms for everything under a millisecond cannot answer
    "which tool was slow", which is most of why anyone reads the timings.
    """
    if duration_ms is None:
        return "-"
    if duration_ms >= 100:
        return f"{duration_ms:.0f}ms"
    if duration_ms >= 1:
        return f"{duration_ms:.1f}ms"
    return f"{duration_ms:.2f}ms"


def _tokens(run: dict[str, Any]) -> str:
    total = (run["prompt_tokens"] or 0) + (run["completion_tokens"] or 0)
    cached = run["cached_tokens"] or 0
    return f"{total:,} tokens ({cached:,} cached)" if cached else f"{total:,} tokens"


def _cached(step: dict[str, Any]) -> str:
    cached, written = step["cached_tokens"] or 0, step["cache_write_tokens"] or 0
    parts = []
    if cached:
        parts.append(f"{cached:,} cached")
    if written:
        parts.append(f"{written:,} written")
    return f"({', '.join(parts)})" if parts else ""


def _arguments(arguments_json: str | None) -> str:
    if not arguments_json:
        return ""
    text = " ".join(arguments_json.split())
    return text if len(text) <= ARGUMENT_PREVIEW else f"{text[:ARGUMENT_PREVIEW]}..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one run's trace.")
    parser.add_argument("run_id")
    parser.add_argument(
        "--db", default=os.environ.get("AGENTHARNESS_DB_PATH", DEFAULT_DB_PATH)
    )
    arguments = parser.parse_args(argv)

    store = RunStore(arguments.db)
    run = store.get_run(arguments.run_id)
    if run is None:
        print(f"no run with id {arguments.run_id}", file=sys.stderr)
        return 1
    print(render(run, store.get_steps(arguments.run_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
