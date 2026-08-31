"""ContextBuilder: message assembly and the tool-call protocol check."""

import pytest

from agentharness.harness.context import ContextBuilder, load_prompt
from agentharness.harness.model_client import HarnessFatalError
from tests.conftest import assistant_text, assistant_tool_calls

SYSTEM = "system prompt"
GOAL = "what do I owe Dr. Patel?"


@pytest.fixture
def context() -> ContextBuilder:
    return ContextBuilder(SYSTEM, GOAL)


def test_a_new_context_is_the_system_prompt_then_the_goal(context):
    assert context.messages() == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": GOAL},
    ]


def test_an_assistant_message_without_tool_calls_carries_no_tool_calls_key(context):
    context.add_assistant(assistant_text("here you go"))
    assert context.messages()[-1] == {"role": "assistant", "content": "here you go"}


def test_tool_calls_are_rendered_in_the_wire_shape(context):
    context.add_assistant(
        assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"}))
    )
    context.add_tool_result("call_1", "{}")

    assert context.messages()[2]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_open_followups",
                "arguments": '{"physician_name": "Raj Patel"}',
            },
        }
    ]
    assert context.messages()[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "{}",
    }


def test_an_unanswered_tool_call_is_refused_before_it_becomes_a_400(context):
    context.add_assistant(
        assistant_tool_calls(
            ("get_open_followups", {"physician_name": "Raj Patel"}),
            ("get_physician_profile", {"physician_name": "Raj Patel"}),
        )
    )
    context.add_tool_result("call_1", "{}")

    with pytest.raises(HarnessFatalError, match="call_2"):
        context.messages()


def test_answering_every_call_clears_the_check(context):
    context.add_assistant(
        assistant_tool_calls(
            ("get_open_followups", {"physician_name": "Raj Patel"}),
            ("get_physician_profile", {"physician_name": "Raj Patel"}),
        )
    )
    context.add_tool_result("call_1", "{}")
    context.add_tool_result("call_2", "{}")
    assert len(context.messages()) == 5


def test_a_second_assistant_turn_cannot_step_over_an_unanswered_call(context):
    context.add_assistant(
        assistant_tool_calls(("get_open_followups", {"physician_name": "Raj Patel"}))
    )
    context.add_assistant(assistant_text("moving on"))

    with pytest.raises(HarnessFatalError, match="call_1"):
        context.messages()


def test_the_returned_list_cannot_be_used_to_edit_the_context(context):
    context.messages().append({"role": "user", "content": "injected"})
    assert len(context.messages()) == 2


def test_the_system_prompt_is_loaded_from_disk():
    prompt = load_prompt("system")
    assert "not instruction to you" in prompt
    # It is spent on every iteration of every run, so its size is a real cost.
    assert len(prompt.split()) < 250
