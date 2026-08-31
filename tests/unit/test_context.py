"""ContextBuilder: message assembly and the tool-call protocol check."""

import hashlib
import json

import pytest
import tiktoken

from agentharness.harness.context import (
    ENCODING_NAME,
    ContextBudgetExceeded,
    ContextBuilder,
    count_tokens,
    load_prompt,
)
from agentharness.harness.model_client import HarnessFatalError
from agentharness.harness.registry import build_registry
from tests.conftest import assistant_text, assistant_tool_calls

SYSTEM = "system prompt"
GOAL = "what do I owe Dr. Patel?"
TOOLS = build_registry().tool_definitions()


@pytest.fixture
def context() -> ContextBuilder:
    return ContextBuilder(SYSTEM, GOAL, tools=[])


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


# --- the token budget ---------------------------------------------------------

def test_the_tool_definitions_are_counted_in_the_budget():
    """They are about 1,200 tokens: a budget ignoring them measures the smaller half."""
    without = count_tokens([{"role": "user", "content": GOAL}], tools=[])
    with_tools = count_tokens([{"role": "user", "content": GOAL}], tools=TOOLS)

    assert with_tools - without > 500


def test_a_context_within_budget_is_returned():
    context = ContextBuilder(SYSTEM, GOAL, TOOLS)
    assert context.messages_for_model_call() == context.messages()


def test_overflow_raises_rather_than_dropping_anything():
    """Nothing is discarded to fit. An agent that quietly forgets what it
    retrieved is worse than one that stops and says so."""
    context = ContextBuilder(SYSTEM, GOAL, TOOLS, token_budget=50)

    with pytest.raises(ContextBudgetExceeded, match="against a budget of 50"):
        context.messages_for_model_call()

    # And the context is intact afterwards: refused, not trimmed.
    assert len(context.messages()) == 2


def test_the_count_grows_with_the_conversation():
    context = ContextBuilder(SYSTEM, GOAL, TOOLS)
    before = context.token_count()
    context.add_assistant(assistant_text("a" * 400))

    assert context.token_count() > before


# --- calibration against the API's own count ----------------------------------

# Observed on the smoke run of 2026-08-31 against gpt-5.6-luna: the first call
# of the goal below reported this many prompt tokens. The three inputs to that
# number are pinned underneath, so a change to any of them fails here by name
# rather than surfacing later as unexplained token drift.
REPORTED_FIRST_CALL_PROMPT_TOKENS = 1585
SMOKE_GOAL = "Prepare me for tomorrow's meeting with Dr. Evelyn Chen about Nexovar."
SYSTEM_PROMPT_SHA256 = "e142c4ee5c8ceedccea28b5d447e670cd415f9bb80be282089cb87065617b1df"
TOOL_DEFINITIONS_SHA256 = "f85afdce58ae42bb13147240597cca72fcb1fefb970becdb9b40507a2a88adb1"
TOOL_NAMES = [
    "create_followup",
    "get_open_followups",
    "get_physician_profile",
    "get_previous_meetings",
    "search_product_docs",
    "submit_final_answer",
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_estimate_lands_within_five_percent_of_the_reported_count():
    """A budget enforced with a wildly wrong count is not a budget.

    The uncorrected count was 1,538 against 1,300 reported. 35 of that gap was
    ours to fix -- encoding the message dict rather than the text it carries --
    and the remaining ~200 is in the tool definitions, where the API bills its
    own representation of the schema rather than the JSON we send.

    To recalibrate after changing the prompt or the tools:
      1. run `python -m agentharness.cli.smoke --model <name>`
      2. take prompt_tokens from the first `usage payload:` line of the output
      3. update REPORTED_FIRST_CALL_PROMPT_TOKENS and the digests above, and
         record what changed and the new figure in DECISIONS.md
    """
    assert _sha256(load_prompt("system")) == SYSTEM_PROMPT_SHA256, (
        "system.md changed, so the recorded prompt-token count no longer "
        "describes this context. Recalibrate: see this test's docstring."
    )
    assert sorted(build_registry().names()) == TOOL_NAMES, (
        "the tool set changed. Recalibrate: see this test's docstring."
    )
    assert _sha256(json.dumps(TOOLS, sort_keys=True)) == TOOL_DEFINITIONS_SHA256, (
        "a tool description or schema changed, which moves the token count as "
        "surely as adding a tool does. Recalibrate: see this test's docstring."
    )

    context = ContextBuilder(load_prompt("system"), SMOKE_GOAL, TOOLS)
    estimate = context.token_count()
    drift = abs(estimate - REPORTED_FIRST_CALL_PROMPT_TOKENS) / REPORTED_FIRST_CALL_PROMPT_TOKENS

    assert drift < 0.05, f"estimated {estimate} against {REPORTED_FIRST_CALL_PROMPT_TOKENS} reported"


def test_the_estimate_errs_high_so_the_budget_fails_safe():
    context = ContextBuilder(load_prompt("system"), SMOKE_GOAL, TOOLS)
    assert context.token_count() >= REPORTED_FIRST_CALL_PROMPT_TOKENS


def test_message_counting_uses_the_text_not_the_serialized_dict():
    """The 35-token overcount came from escaping newlines and counting keys."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    content = """line one
line two
line three
""" * 10
    message = {"role": "user", "content": content}

    counted = count_tokens([message], tools=[])

    # Close to the text the message carries, and meaningfully below what
    # encoding the dict's JSON would have charged for the same content.
    assert counted - len(encoding.encode(content)) < 10
    assert counted < len(encoding.encode(json.dumps(message)))
