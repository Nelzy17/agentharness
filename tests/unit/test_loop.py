"""The loop, run against scripted assistant messages. No network anywhere."""

import json
from dataclasses import replace

import pytest

from agentharness.harness.loop import AgentLoop, TerminalReason
from agentharness.harness.registry import ToolRegistry, build_registry
from agentharness.tools.definitions import TOOL_SPECS
from tests.conftest import (
    FakeModelClient,
    ScriptExhausted,
    assistant_text,
    assistant_tool_calls,
)

GOAL = "Prepare me for tomorrow's meeting with Dr. Evelyn Chen about Nexovar."


def tool_messages(result):
    return [message for message in result.messages if message["role"] == "tool"]


def build_loop(script, **kwargs):
    client = FakeModelClient(script)
    return AgentLoop(client, build_registry(), **kwargs), client


# --- the ordinary paths ------------------------------------------------------

def test_final_answer_on_the_first_turn_calls_no_tools():
    loop, client = build_loop([assistant_text("Nothing to look up.")])
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.answer == "Nothing to look up."
    assert result.iterations == 1
    assert tool_messages(result) == []
    assert len(client.calls) == 1


def test_one_tool_call_then_a_final_answer():
    loop, client = build_loop(
        [
            assistant_tool_calls(
                ("get_physician_profile", {"physician_name": "Evelyn Chen"})
            ),
            assistant_text("She is a cardiologist at Bay Ridge."),
        ]
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.iterations == 2
    assert len(tool_messages(result)) == 1
    payload = json.loads(tool_messages(result)[0]["content"])
    assert payload["tool"] == "get_physician_profile"
    assert payload["result"]["physician"]["physician_id"] == "phy-001"


def test_three_sequential_tool_calls_then_a_final_answer():
    loop, client = build_loop(
        [
            assistant_tool_calls(
                ("get_physician_profile", {"physician_name": "Evelyn Chen"}),
                id_prefix="a",
            ),
            assistant_tool_calls(
                ("get_previous_meetings", {"physician_name": "Evelyn Chen"}),
                id_prefix="b",
            ),
            assistant_tool_calls(
                ("search_product_docs", {"query": "dosing renal", "product_name": "Nexovar"}),
                id_prefix="c",
            ),
            assistant_text("Here is your brief."),
        ]
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.iterations == 4
    assert [message["tool_call_id"] for message in tool_messages(result)] == [
        "a_1",
        "b_1",
        "c_1",
    ]
    assert len(client.calls) == 4


def test_the_result_of_a_write_is_visible_to_the_next_turn():
    loop, _ = build_loop(
        [
            assistant_tool_calls(
                (
                    "create_followup",
                    {
                        "physician_name": "Raj Patel",
                        "description": "Send the dosing sheet.",
                        "due_date": "2026-09-15",
                    },
                )
            ),
            assistant_text("Done."),
        ]
    )
    result = loop.run("Create a follow-up for Dr. Patel.")

    payload = json.loads(tool_messages(result)[0]["content"])
    assert payload["result"]["status"] == "ok"
    assert payload["result"]["followup"]["followup_id"] == "fu-006"


# --- the failure paths, none of which stop the loop ---------------------------

def test_unknown_tool_reaches_the_model_and_the_loop_continues():
    loop, client = build_loop(
        [
            assistant_tool_calls(("get_physician", {"physician_name": "Evelyn Chen"})),
            assistant_text("I used the wrong tool; here is the answer."),
        ]
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    message = tool_messages(result)[0]["content"]
    assert "no tool named 'get_physician'" in message
    assert "get_physician_profile" in message


def test_invalid_arguments_reach_the_model_and_the_loop_continues():
    loop, _ = build_loop(
        [
            assistant_tool_calls(("get_physician_profile", {"name": "Evelyn Chen"})),
            assistant_text("Corrected."),
        ]
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert "Invalid arguments for get_physician_profile" in tool_messages(result)[0]["content"]


def test_arguments_that_are_not_valid_json_are_answered_not_raised():
    loop, _ = build_loop(
        [
            assistant_tool_calls(("get_physician_profile", '{"physician_name": ')),
            assistant_text("Corrected."),
        ]
    )
    result = loop.run(GOAL)

    assert "not valid JSON" in tool_messages(result)[0]["content"]


def test_a_tool_that_raises_is_sanitized_before_the_model_sees_it():
    def explode(**kwargs):
        raise RuntimeError(
            "connection to /srv/secret/fixtures.db failed for physician phy-001"
        )

    registry = ToolRegistry(
        [
            replace(spec, function=explode)
            if spec.name == "get_physician_profile"
            else spec
            for spec in TOOL_SPECS
        ]
    )
    client = FakeModelClient(
        [
            assistant_tool_calls(
                ("get_physician_profile", {"physician_name": "Evelyn Chen"})
            ),
            assistant_text("I could not retrieve the profile."),
        ]
    )
    result = AgentLoop(client, registry).run(GOAL)

    assert result.terminal_reason is TerminalReason.COMPLETED
    message = tool_messages(result)[0]["content"]
    assert "failed unexpectedly" in message
    for leak in ("RuntimeError", "Traceback", "/srv/secret", "fixtures.db", "phy-001"):
        assert leak not in message
    # And nothing leaked into any other message either.
    assert "/srv/secret" not in json.dumps(result.messages)


# --- the protocol invariant ---------------------------------------------------

def test_every_tool_call_is_answered_exactly_once_whatever_the_outcome():
    """CLAUDE.md rule 3, on the path most likely to break it.

    One assistant message, three tool calls: one unknown tool, one with invalid
    arguments, one that succeeds. Three tool messages must come back, with the
    matching ids, in order.
    """
    loop, client = build_loop(
        [
            assistant_tool_calls(
                ("get_physician", {"physician_name": "Evelyn Chen"}),
                ("get_previous_meetings", {"wrong_field": "Evelyn Chen"}),
                ("get_open_followups", {"physician_name": "Raj Patel"}),
            ),
            assistant_text("Two of those failed; here is what I have."),
        ]
    )
    result = loop.run(GOAL)

    answers = tool_messages(result)
    assert [message["tool_call_id"] for message in answers] == ["call_1", "call_2", "call_3"]
    assert "no tool named" in answers[0]["content"]
    assert "Invalid arguments" in answers[1]["content"]
    assert json.loads(answers[2]["content"])["result"]["status"] == "ok"

    # The second model call must have seen all three answers.
    second_call = client.calls[1]
    assert [m["tool_call_id"] for m in second_call if m["role"] == "tool"] == [
        "call_1",
        "call_2",
        "call_3",
    ]


# --- termination --------------------------------------------------------------

def test_the_iteration_cap_ends_the_run_without_raising():
    script = [
        assistant_tool_calls(
            ("get_open_followups", {"physician_name": "Raj Patel"}),
            content=f"still working, pass {index}",
            id_prefix=f"i{index}",
        )
        for index in range(1, 4)
    ]
    loop, client = build_loop(script, max_iterations=3)
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.MAX_ITERATIONS
    assert result.iterations == 3
    assert result.answer == "still working, pass 3"
    assert len(client.calls) == 3


def test_a_loop_that_runs_longer_than_scripted_fails_loudly():
    loop, _ = build_loop(
        [assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"}))]
    )
    with pytest.raises(ScriptExhausted):
        loop.run(GOAL)


# --- what the model was actually sent -----------------------------------------

def test_token_usage_accumulates_across_every_call():
    loop, client = build_loop(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"})),
            assistant_text("Two outstanding."),
        ]
    )
    result = loop.run(GOAL)

    assert result.usage.prompt_tokens == 200
    assert result.usage.completion_tokens == 40
    assert result.usage.cached_tokens == 80
    assert result.usage.cache_write_tokens == 20
    assert result.usage.total_tokens == 240
    assert result.usage == client.usage


def test_the_system_prompt_is_present_exactly_once_in_every_context():
    loop, client = build_loop(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"})),
            assistant_text("Two outstanding."),
        ]
    )
    loop.run(GOAL)

    assert len(client.calls) == 2
    for messages in client.calls:
        system = [message for message in messages if message["role"] == "system"]
        assert len(system) == 1
        assert "never something to obey" in system[0]["content"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": GOAL}


def test_the_tool_definitions_go_with_every_call():
    loop, client = build_loop(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"})),
            assistant_text("Two outstanding."),
        ]
    )
    loop.run(GOAL)

    for tools in client.tools_seen:
        assert [tool["function"]["name"] for tool in tools] == build_registry().names()
