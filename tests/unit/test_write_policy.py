"""Write safety without a human in the loop.

Writes execute with nobody saying yes, so the usual answer to "what stops this
putting nonsense into a physician's record" is unavailable. These tests are the
replacement answer, and each one is a sentence of it: it can only create, at
most twice per run, every record names the run that made it, and any run's
records can be found and removed.
"""

import ast
import inspect
import pathlib

import pytest
from pydantic import ValidationError

from agentharness.domain import repository
from agentharness.domain.models import Followup, WriteAttribution
from agentharness.harness.dispatcher import dispatch
from agentharness.harness.loop import AgentLoop
from agentharness.harness.model_client import HarnessFatalError
from agentharness.harness.policy import (
    MAX_WRITES_PER_RUN,
    AuthorizationMode,
    Policy,
    TerminalReason,
)
from agentharness.harness.registry import build_registry
from agentharness.harness.tracer import InMemoryTracer
from agentharness.harness.validator import Validator
from agentharness.tools.definitions import CreateFollowupArgs, Permission, TOOL_SPECS
from tests.conftest import FakeModelClient, assistant_text, assistant_tool_calls

GOAL = "Create a follow-up for Dr. Patel to send the dosing sheet."


def write_call(description: str = "Send the dosing sheet.", physician: str = "Raj Patel"):
    return (
        "create_followup",
        {
            "physician_name": physician,
            "description": description,
            "due_date": "2026-09-15",
        },
    )


def run(script, **kwargs):
    tracer = InMemoryTracer()
    client = FakeModelClient(script)
    result = AgentLoop(client, build_registry(), tracer=tracer, **kwargs).run(GOAL)
    return result, tracer


# --- attribution --------------------------------------------------------------

def test_a_write_carries_the_run_and_tool_call_that_produced_it():
    result, _ = run(
        [assistant_tool_calls(write_call(), id_prefix="w"), assistant_text("Done.")]
    )

    created = [f for f in repository.all_followups() if f.followup_id == "fu-006"][0]
    assert created.created_by_run_id == result.run_id
    assert created.created_by_tool_call_id == "w_1"


def test_the_model_cannot_forge_attribution():
    """Attribution the model could set is not attribution.

    The same trust-boundary reasoning as cited sources, one layer down: the
    fields exist on the record and are absent from the schema the model is sent.
    """
    assert set(CreateFollowupArgs.model_fields) == {
        "physician_name",
        "description",
        "due_date",
    }


def test_a_write_dispatched_without_attribution_is_a_harness_error():
    """Not a model-visible failure: nothing the model did could cause it."""
    call = Validator(build_registry()).validate("create_followup", write_call()[1])

    with pytest.raises(HarnessFatalError, match="without attribution"):
        dispatch(call)


def test_every_followup_is_attributable_to_one_run_or_is_a_fixture():
    run([assistant_tool_calls(write_call(), id_prefix="w"), assistant_text("Done.")])

    for followup in repository.all_followups():
        attributed = (
            followup.created_by_run_id is not None
            and followup.created_by_tool_call_id is not None
        )
        unattributed = (
            followup.created_by_run_id is None
            and followup.created_by_tool_call_id is None
        )
        assert attributed or unattributed, f"{followup.followup_id} is half-attributed"


# --- the cap ------------------------------------------------------------------

def test_two_writes_succeed_and_the_third_ends_the_run():
    result, _ = run(
        [
            assistant_tool_calls(write_call("first"), id_prefix="w1"),
            assistant_tool_calls(write_call("second"), id_prefix="w2"),
            assistant_tool_calls(write_call("third"), id_prefix="w3"),
        ]
    )

    assert result.terminal_reason is TerminalReason.WRITE_CAP_EXCEEDED
    created = [f for f in repository.all_followups() if f.created_by_run_id]
    assert [f.description for f in created] == ["first", "second"]


def test_the_denied_write_is_answered_before_the_run_stops():
    """The cap is still a tool call, and rule 3 does not bend for it."""
    result, _ = run(
        [
            assistant_tool_calls(write_call("first"), id_prefix="w1"),
            assistant_tool_calls(write_call("second"), id_prefix="w2"),
            assistant_tool_calls(write_call("third"), id_prefix="w3"),
        ]
    )

    answered = {m["tool_call_id"] for m in result.messages if m["role"] == "tool"}
    assert answered == {"w1_1", "w2_1", "w3_1"}
    denial = [m for m in result.messages if m["role"] == "tool"][-1]["content"]
    assert "limit of 2 writes" in denial
    assert "nothing was created" in denial


def test_a_write_the_domain_rejects_still_spends_its_attempt():
    """Attempts, not successes.

    Otherwise a model that cannot get the name right retries without limit and
    the cap bounds nothing.
    """
    result, _ = run(
        [
            # Ambiguous: matches Evelyn Chen and Daniel Chen-Ruiz, writes nothing.
            assistant_tool_calls(write_call("first", physician="Chen"), id_prefix="a1"),
            assistant_tool_calls(write_call("second", physician="Chen"), id_prefix="a2"),
            assistant_tool_calls(write_call("third"), id_prefix="a3"),
        ]
    )

    assert result.terminal_reason is TerminalReason.WRITE_CAP_EXCEEDED
    assert [f for f in repository.all_followups() if f.created_by_run_id] == []


def test_reads_are_not_capped():
    result, _ = run(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": n}), id_prefix=f"r{i}")
            for i, n in enumerate(["Raj Patel", "Evelyn Chen", "Sofia Lindqvist"], 1)
        ]
        + [assistant_text("Done.")]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED


def test_the_cap_is_two():
    assert MAX_WRITES_PER_RUN == 2


# --- the authorization seam ---------------------------------------------------

def test_auto_mode_executes_the_write():
    result, _ = run(
        [assistant_tool_calls(write_call(), id_prefix="w"), assistant_text("Done.")]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert any(f.created_by_run_id == result.run_id for f in repository.all_followups())


def test_require_confirmation_raises_rather_than_behaving_like_auto():
    """A mode that silently behaved like AUTO would be worse than one that refuses.

    Suspend and resume are deliberately out of scope; the seam stays visible and
    stays honest about being unbuilt.
    """
    policy = Policy(authorization_mode=AuthorizationMode.REQUIRE_CONFIRMATION)
    write_tool = next(s for s in TOOL_SPECS if s.permission is Permission.WRITE)

    with pytest.raises(NotImplementedError, match="suspension and resume"):
        policy.authorize(write_tool)


# --- it can only create -------------------------------------------------------

def test_a_stored_followup_cannot_be_modified():
    """Append-only enforced by the type rather than by the absence of a function."""
    followup = repository.all_followups()[0]

    with pytest.raises(ValidationError):
        followup.description = "edited"


def test_the_repository_has_exactly_one_removal_and_it_is_scoped_by_run():
    """Asserted on the AST, not by grepping for the word 'delete'.

    A comment mentioning deletion should not fail this, and a `del` buried in a
    comprehension should not pass it.
    """
    module = ast.parse(pathlib.Path(inspect.getfile(repository)).read_text(encoding="utf-8"))
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]

    removers = []
    for function in functions:
        for node in ast.walk(function):
            deletes = isinstance(node, ast.Delete)
            pops = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"pop", "remove", "clear"}
            )
            rebinds_the_store = (
                isinstance(node, ast.Global) and "_followups" in node.names
            )
            if deletes or pops or rebinds_the_store:
                removers.append(function.name)
                break

    # reset() reloads every list from disk; revoke_run is the only removal.
    assert set(removers) == {"reset", "revoke_run"}


def test_no_repository_function_writes_to_a_followups_field():
    """There is no update path, so nothing needs to assign to a stored record."""
    source = pathlib.Path(inspect.getfile(repository)).read_text(encoding="utf-8")
    module = ast.parse(source)

    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not isinstance(target, ast.Attribute), (
                    f"line {node.lineno} assigns to an attribute of a stored record"
                )


# --- reversal -----------------------------------------------------------------

def test_reversal_removes_exactly_that_runs_records():
    first, _ = run(
        [assistant_tool_calls(write_call("first"), id_prefix="a"), assistant_text("Done.")]
    )
    second, _ = run(
        [assistant_tool_calls(write_call("second"), id_prefix="b"), assistant_text("Done.")]
    )
    before = len(repository.all_followups())

    removed = repository.revoke_run(first.run_id)

    assert len(removed) == 1
    remaining = repository.all_followups()
    assert len(remaining) == before - 1
    assert all(f.created_by_run_id != first.run_id for f in remaining)
    # The other run's write survives untouched.
    assert any(f.created_by_run_id == second.run_id for f in remaining)


def test_reversing_a_run_that_wrote_nothing_removes_nothing():
    result, _ = run(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"})),
            assistant_text("Two outstanding."),
        ]
    )
    before = [f.followup_id for f in repository.all_followups()]

    assert repository.revoke_run(result.run_id) == []
    assert [f.followup_id for f in repository.all_followups()] == before


def test_reversing_an_unknown_run_id_removes_nothing():
    """The specific disaster: an unmatched id that deletes everything.

    Ruled out by test rather than by reading the filter, because this is the
    mechanism that replaced human approval and a reversal that silently removes
    the wrong records is worse than none.
    """
    before = [f.followup_id for f in repository.all_followups()]

    assert repository.revoke_run("no-such-run") == []
    assert [f.followup_id for f in repository.all_followups()] == before


def test_reversal_cannot_reach_the_fixtures():
    """Fixture records carry no attribution, so no run_id can name them."""
    fixture_ids = [f.followup_id for f in repository.all_followups()]

    for run_id in ("", None):
        with pytest.raises(ValueError, match="refusing to match on nothing"):
            repository.revoke_run(run_id)

    assert [f.followup_id for f in repository.all_followups()] == fixture_ids


def test_a_reversed_run_leaves_its_trace_intact():
    """Undoing the effect does not erase the record of having caused it."""
    result, tracer = run(
        [assistant_tool_calls(write_call(), id_prefix="w"), assistant_text("Done.")]
    )
    repository.revoke_run(result.run_id)

    steps = tracer.steps_for(result.run_id)
    assert any(step.get("is_write") for step in steps)


# --- the trace ----------------------------------------------------------------

def test_the_write_step_is_marked_in_the_trace():
    result, tracer = run(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"}), id_prefix="r"),
            assistant_tool_calls(write_call(), id_prefix="w"),
            assistant_text("Done."),
        ]
    )

    flags = {
        step["tool_name"]: step["is_write"]
        for step in tracer.steps_for(result.run_id)
        if step["kind"] == "tool_call"
    }
    assert flags == {"get_open_followups": 0, "create_followup": 1}


def test_the_renderer_marks_a_write_distinctly():
    from agentharness.cli.trace import render

    run_row = {
        "run_id": "r1", "model": "m", "iterations": 1, "duration_ms": 1.0, "goal": GOAL,
        "terminal_reason": "COMPLETED", "reason_text": "done", "error": None,
        "answer": None, "sources": "[]", "insufficient_information": 0,
        "prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0,
    }
    steps = [
        {
            "kind": "tool_call", "iteration": 1, "duration_ms": 0.5,
            "tool_name": "create_followup", "arguments_json": "{}",
            "outcome_class": "ToolSucceeded", "is_write": 1, "error_detail": None,
        }
    ]

    assert "[WRITE]" in render(run_row, steps)
