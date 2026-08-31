"""Tracing, and specifically the measurements a scripted clock cannot check."""

import time
from dataclasses import replace

from agentharness.harness.loop import AgentLoop
from agentharness.harness.registry import ToolRegistry, build_registry
from agentharness.harness.tracer import InMemoryTracer
from agentharness.tools.definitions import TOOL_SPECS
from agentharness.tools.physician import get_physician_profile
from tests.conftest import FakeModelClient, assistant_text, assistant_tool_calls

GOAL = "Who is Dr. Chen?"
PROFILE = ("get_physician_profile", {"physician_name": "Evelyn Chen"})
SLOW_MS = 20


def run_with(script, registry=None) -> InMemoryTracer:
    """Deliberately uses the loop's real clocks. That is the point of this file."""
    tracer = InMemoryTracer()
    AgentLoop(FakeModelClient(script), registry or build_registry(), tracer=tracer).run(GOAL)
    return tracer


def tool_steps(tracer: InMemoryTracer) -> list[dict]:
    return [step for step in tracer.steps if step["kind"] == "tool_call"]


def test_a_slow_tool_is_measured_as_slow_by_the_real_clock():
    """The test a scripted clock cannot substitute for.

    A fake with distinct values makes every duration non-zero by construction,
    so it proves the plumbing and nothing about the measurement. This one runs
    the loop's own clock against an operation of known length.
    """

    def slow(**kwargs):
        time.sleep(SLOW_MS / 1000)
        return get_physician_profile(**kwargs)

    registry = ToolRegistry(
        [
            replace(spec, function=slow) if spec.name == "get_physician_profile" else spec
            for spec in TOOL_SPECS
        ]
    )
    tracer = run_with([assistant_tool_calls(PROFILE), assistant_text("done")], registry)

    measured = tool_steps(tracer)[0]["duration_ms"]
    assert measured >= SLOW_MS * 0.8, f"a {SLOW_MS}ms tool was measured as {measured}ms"


def test_a_fast_tool_is_measured_as_fast_rather_than_as_nothing():
    """The bug this file exists for.

    Every tool duration in the first real trace rendered as 0ms. The tools are
    genuinely fast -- dictionary lookups over in-memory fixtures -- but seven
    consecutive zeros is a measurement artefact, not a result. time.monotonic on
    Windows is GetTickCount64 with 15.625ms resolution, so it floors everything
    this harness does to zero. perf_counter does not.
    """
    tracer = run_with([assistant_tool_calls(PROFILE), assistant_text("done")])

    for step in tool_steps(tracer):
        assert step["duration_ms"] > 0, "a sub-millisecond tool call recorded as zero"
        assert step["duration_ms"] < SLOW_MS, "a dictionary lookup should not take 20ms"


def test_the_clock_used_for_intervals_can_resolve_them():
    """Asserts the property, so a future swap back to monotonic fails here.

    20,000 reads of time.monotonic on this platform return one value.
    """
    clock = AgentLoop(FakeModelClient([]), build_registry())._clock
    samples = {clock() for _ in range(2000)}

    assert len(samples) > 1000, (
        f"{clock.__name__} produced {len(samples)} distinct values in 2,000 reads; "
        "it cannot measure anything this harness does"
    )


def test_timestamps_are_real_time_and_durations_are_not():
    """Two clocks, because one number cannot be both.

    perf_counter has an arbitrary epoch, so storing it as started_at would
    record something that looks like a timestamp and is not one.
    """
    before = time.time()
    tracer = run_with([assistant_text("done")])
    after = time.time()

    started_at = next(iter(tracer.runs.values()))["started_at"]
    assert before <= started_at <= after


def test_the_model_call_duration_is_recorded_too():
    tracer = run_with([assistant_tool_calls(PROFILE), assistant_text("done")])
    model_steps = [step for step in tracer.steps if step["kind"] == "model_call"]

    assert len(model_steps) == 2
    assert all(step["duration_ms"] >= 0 for step in model_steps)


def test_the_renderer_shows_a_sub_millisecond_call_as_a_number():
    from agentharness.cli.trace import _ms

    assert _ms(0.0356) == "0.04ms"
    assert _ms(4.2) == "4.2ms"
    assert _ms(250.0) == "250ms"
    assert _ms(None) == "-"
