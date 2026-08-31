"""The public surface, end to end: post a goal, read back the trace.

Everything runs against FakeModelClient. Nothing here reaches a network.
"""

import json

import pytest
from fastapi.testclient import TestClient

from agentharness.api.app import create_app
from agentharness.cli.trace import render
from agentharness.harness.loop import AgentLoop
from agentharness.harness.model_client import HarnessFatalError
from agentharness.harness.registry import ToolRegistry, build_registry
from agentharness.harness.tracer import SqliteTracer
from agentharness.store.runs import RunStore
from agentharness.tools.definitions import TOOL_SPECS
from tests.conftest import FakeModelClient, assistant_text, assistant_tool_calls

GOAL = "Prepare me for tomorrow's meeting with Dr. Evelyn Chen about Nexovar."

SCRIPT = [
    assistant_tool_calls(
        ("get_physician_profile", {"physician_name": "Evelyn Chen"}), id_prefix="a"
    ),
    assistant_tool_calls(
        ("get_previous_meetings", {"physician_name": "Evelyn Chen"}), id_prefix="b"
    ),
    assistant_tool_calls(
        ("search_product_docs", {"query": "nexovar dosing", "product_name": "Nexovar"}),
        id_prefix="c",
    ),
    assistant_tool_calls(
        (
            "submit_final_answer",
            {
                "answer": "She asked about Nexovar dosing in renal impairment.",
                "sources": ["a_1", "b_1", "c_1"],
                "insufficient_information": False,
            },
        ),
        id_prefix="fin",
    ),
]


class Clock:
    """Distinct values, so a duration that reads zero is a bug and not a fake."""

    def __init__(self) -> None:
        self._tick = 0.0

    def __call__(self) -> float:
        self._tick += 0.25
        return self._tick


@pytest.fixture
def store() -> RunStore:
    return RunStore(":memory:")


def client_for(store, script=None, registry=None, model_client=None):
    def build_loop(tracer):
        return AgentLoop(
            model_client or FakeModelClient(list(script or SCRIPT)),
            registry or build_registry(),
            tracer=tracer,
            clock=Clock(),
        )

    return TestClient(create_app(build_loop=build_loop, store=store), raise_server_exceptions=False)


# --- the round trip -----------------------------------------------------------

def test_a_posted_goal_runs_and_returns_its_answer(store):
    response = client_for(store).post("/runs", json={"goal": GOAL})

    assert response.status_code == 200
    body = response.json()
    assert body["terminal_reason"] == "COMPLETED"
    assert body["reason_text"] == "the model produced a final answer"
    assert body["route"] == "SUBMIT_TOOL"
    assert body["sources"] == ["a_1", "b_1", "c_1"]
    assert body["insufficient_information"] is False
    assert body["iterations"] == 4


def test_the_trace_has_one_step_per_model_call_and_one_per_tool_call(store):
    client = client_for(store)
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    detail = client.get(f"/runs/{run_id}").json()
    steps = detail["steps"]

    assert len([s for s in steps if s["kind"] == "model_call"]) == 4
    assert len([s for s in steps if s["kind"] == "tool_call"]) == 4
    assert detail["run"]["iterations"] == 4
    assert [s["sequence"] for s in steps] == list(range(1, 9))


def test_every_step_carries_the_run_id_of_its_run(store):
    client = client_for(store)
    first = client.post("/runs", json={"goal": GOAL}).json()["run_id"]
    second = client.post("/runs", json={"goal": "What do I owe Dr. Patel?"}).json()["run_id"]

    assert first != second
    for run_id in (first, second):
        steps = client.get(f"/runs/{run_id}").json()["steps"]
        assert steps
        assert {step["run_id"] for step in steps} == {run_id}


def test_trace_token_totals_match_the_run(store):
    client = client_for(store)
    body = client.post("/runs", json={"goal": GOAL}).json()
    detail = client.get(f"/runs/{body['run_id']}").json()

    model_steps = [s for s in detail["steps"] if s["kind"] == "model_call"]
    assert sum(s["prompt_tokens"] for s in model_steps) == body["usage"]["prompt_tokens"]
    assert sum(s["completion_tokens"] for s in model_steps) == body["usage"]["completion_tokens"]
    assert sum(s["cached_tokens"] for s in model_steps) == body["usage"]["cached_tokens"]
    assert detail["run"]["prompt_tokens"] == body["usage"]["prompt_tokens"]


def test_a_retried_call_is_recorded_as_one_step_with_its_attempt_count(store):
    """Retries happen below the loop; the trace still has to show they happened."""
    from agentharness.harness.model_client import ModelResponse, TokenUsage

    class RetryingClient(FakeModelClient):
        model = "fake-model"

        def complete(self, messages, tools):
            response = super().complete(messages, tools)
            return ModelResponse(
                message=response.message,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                attempts=2,
            )

    client = client_for(store, model_client=RetryingClient([assistant_text("done")]))
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    steps = client.get(f"/runs/{run_id}").json()["steps"]
    assert [s["attempts"] for s in steps if s["kind"] == "model_call"] == [2]


def test_durations_are_recorded_and_not_zero(store):
    """The clock trap: a constant fake makes every duration zero and every
    assertion pass while measuring nothing."""
    client = client_for(store)
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["run"]["duration_ms"] > 0
    assert all(step["duration_ms"] > 0 for step in detail["steps"])


def test_an_unknown_run_id_is_a_404(store):
    assert client_for(store).get("/runs/nosuchrun").status_code == 404


def test_the_write_flag_distinguishes_a_write_from_a_read(store):
    """Unused until M6, populated now: adding a column to a populated table is
    worse than carrying one that waits."""
    script = [
        assistant_tool_calls(
            (
                "create_followup",
                {
                    "physician_name": "Raj Patel",
                    "description": "Send the dosing sheet.",
                    "due_date": "2026-09-15",
                },
            ),
            id_prefix="w",
        ),
        assistant_text("Done."),
    ]
    client = client_for(store, script=script)
    run_id = client.post("/runs", json={"goal": "follow up"}).json()["run_id"]

    steps = client.get(f"/runs/{run_id}").json()["steps"]
    writes = {s["tool_name"]: s["is_write"] for s in steps if s["kind"] == "tool_call"}
    assert writes == {"create_followup": 1}


# --- what the trace does and does not hold ------------------------------------

def test_no_column_can_hold_reasoning_or_a_transcript(store):
    """Asserted on the schema, not on data.

    Two things are being kept out. Reasoning, because a recorded rationale is a
    story told after the fact that cannot be checked against anything, and
    mixing it with observed facts makes the facts less trustworthy. And the
    assembled context, because "just store the messages for debugging" is the
    change that turns a trace into a transcript.
    """
    forbidden = (
        "reasoning",
        "rationale",
        "thought",
        "thinking",
        "explanation",
        "why",
        "messages",
        "message",
        "content",
        "context",
        "prompt_text",
        "transcript",
        "conversation",
    )
    for table in ("runs", "steps"):
        for column in store.column_names(table):
            assert column not in forbidden, f"{table}.{column} could hold a transcript"

    assert set(store.column_names("steps")) == {
        "step_id",
        "run_id",
        "sequence",
        "iteration",
        "kind",
        "duration_ms",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "attempts",
        "tool_call_id",
        "tool_name",
        "arguments_json",
        "outcome_class",
        "is_write",
        "error_detail",
    }


def test_a_tool_exception_is_whole_in_the_trace_and_sanitized_for_the_model(store):
    from dataclasses import replace

    secret = "/srv/secret/fixtures.db"

    def explode(**kwargs):
        raise RuntimeError(f"could not open {secret}")

    registry = ToolRegistry(
        [
            replace(spec, function=explode) if spec.name == "get_physician_profile" else spec
            for spec in TOOL_SPECS
        ]
    )
    script = [
        assistant_tool_calls(("get_physician_profile", {"physician_name": "Evelyn Chen"})),
        assistant_text("I could not retrieve the profile."),
    ]
    client = client_for(store, script=script, registry=registry)
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    steps = client.get(f"/runs/{run_id}").json()["steps"]
    failed = [s for s in steps if s["outcome_class"] == "ToolFailed"][0]

    # Whole in the trace.
    assert secret in failed["error_detail"]
    assert "Traceback" in failed["error_detail"]
    # And nowhere the model could read it.
    assert secret not in json.dumps(client.get(f"/runs/{run_id}").json()["run"])


def test_a_fatally_failed_run_is_still_persisted_and_retrievable(store):
    """The run that failed fatally is exactly the run someone wants to inspect."""

    class BrokenClient(FakeModelClient):
        model = "fake-model"

        def complete(self, messages, tools):
            raise HarnessFatalError("the API rejected our request: 400")

    client = client_for(store, model_client=BrokenClient([]))
    response = client.post("/runs", json={"goal": GOAL})

    assert response.status_code == 500
    run_id = response.json()["detail"]["run_id"]

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["run"]["terminal_reason"] == "HARNESS_ERROR"
    assert "400" in detail["run"]["error"]
    assert detail["run"]["duration_ms"] > 0


def test_the_500_body_points_at_the_trace_without_leaking_internals(store):
    class BrokenClient(FakeModelClient):
        model = "fake-model"

        def complete(self, messages, tools):
            raise HarnessFatalError("the API rejected our request at /srv/secret")

    response = client_for(store, model_client=BrokenClient([])).post(
        "/runs", json={"goal": GOAL}
    )

    assert "/srv/secret" not in response.text
    assert "/runs/" in response.json()["detail"]["message"]


# --- the renderer -------------------------------------------------------------

def test_a_stored_trace_round_trips_into_the_renderer(store):
    client = client_for(store)
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    output = render(store.get_run(run_id), store.get_steps(run_id))

    assert run_id in output
    assert "COMPLETED: the model produced a final answer" in output
    assert "get_physician_profile" in output
    assert "ToolSucceeded" in output
    assert "sources a_1, b_1, c_1" in output


def test_the_renderer_leads_with_the_reason_a_failed_run_stopped(store):
    script = [
        assistant_tool_calls(("get_open_followups", {"physician_name": n}), id_prefix=f"i{i}")
        for i, n in enumerate(["Raj Patel", "Evelyn Chen", "Sofia Lindqvist"], 1)
    ]

    def build_loop(tracer):
        return AgentLoop(
            FakeModelClient(script),
            build_registry(),
            max_iterations=3,
            tracer=tracer,
            clock=Clock(),
        )

    client = TestClient(create_app(build_loop=build_loop, store=store))
    run_id = client.post("/runs", json={"goal": GOAL}).json()["run_id"]

    output = render(store.get_run(run_id), store.get_steps(run_id))
    assert "MAX_ITERATIONS: the run reached the iteration cap without a final answer" in output


def test_the_renderer_shows_the_error_detail_for_a_failed_run(store):
    class BrokenClient(FakeModelClient):
        model = "fake-model"

        def complete(self, messages, tools):
            raise HarnessFatalError("the API rejected our request: 400")

    client = client_for(store, model_client=BrokenClient([]))
    run_id = client.post("/runs", json={"goal": GOAL}).json()["detail"]["run_id"]

    output = render(store.get_run(run_id), store.get_steps(run_id))
    assert "HARNESS_ERROR" in output
    assert "400" in output


def test_the_renderer_imports_no_harness_code():
    """A trace only its author can read is a cache, not a record."""
    import agentharness.cli.trace as module

    source = module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "agentharness.harness" not in text
    assert "agentharness.domain" not in text
    assert "agentharness.tools" not in text
