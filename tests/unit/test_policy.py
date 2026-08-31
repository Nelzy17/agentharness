"""Termination and authorization.

Every condition fires in isolation under a scripted model, with its own reason,
and a single test enumerates TerminalReason to prove none is unreachable.
"""

import json
from dataclasses import replace

import pytest

from agentharness.harness.loop import AgentLoop, FinalAnswerRoute
from agentharness.harness.model_client import TokenUsage
from agentharness.harness.outcomes import (
    DuplicateResult,
    InvalidArguments,
    PermissionDenied,
    RepeatedCall,
    ToolFailed,
    ToolOutcome,
    ToolSucceeded,
    UnknownTool,
)
from agentharness.harness.policy import (
    AuthorizationMode,
    Policy,
    TerminalReason,
    TerminationPolicy,
)
from agentharness.harness.registry import ToolRegistry, build_registry
from agentharness.tools.definitions import TOOL_SPECS, Permission
from tests.conftest import FakeModelClient, assistant_text, assistant_tool_calls

GOAL = "What do I owe Dr. Patel?"
FOLLOWUPS = ("get_open_followups", {"physician_name": "Raj Patel"})
# Sources are empty here on purpose: these tests script ids that vary per case,
# and citing a plausible-looking "call_1" is precisely what contextual
# validation now rejects. Cases that exercise real citation name real ids.
ANSWER = (
    "submit_final_answer",
    {"answer": "Two outstanding.", "sources": [], "insufficient_information": False},
)


class FakeClock:
    """A scripted clock. The policy reads it once at construction, so the first
    value is the start time and the rest is however far the test wants to jump."""

    def __init__(self, *times: float) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0) if len(self._times) > 1 else self._times[0]


def run_with(script, **kwargs):
    client = FakeModelClient(script)
    return AgentLoop(client, build_registry(), **kwargs).run(GOAL), client


# --- each condition, in isolation ---------------------------------------------

def test_a_final_answer_completes():
    result, _ = run_with([assistant_tool_calls(ANSWER)])
    assert result.terminal_reason is TerminalReason.COMPLETED


def test_the_iteration_cap_fires():
    script = [
        assistant_tool_calls(("get_open_followups", {"physician_name": name}), id_prefix=f"i{n}")
        for n, name in enumerate(["Raj Patel", "Evelyn Chen", "Sofia Lindqvist"], 1)
    ]
    result, client = run_with(script, max_iterations=3)

    assert result.terminal_reason is TerminalReason.MAX_ITERATIONS
    assert result.iterations == 3
    assert len(client.calls) == 3


def test_three_consecutive_errors_terminate():
    script = [
        assistant_tool_calls(("get_physician_profile", {"wrong": n}), id_prefix=f"e{n}")
        for n in range(1, 4)
    ]
    result, client = run_with(script)

    assert result.terminal_reason is TerminalReason.CONSECUTIVE_ERRORS
    assert result.iterations == 3
    # It stopped before spending a fourth call on a model that is not recovering.
    assert len(client.calls) == 3


def test_a_success_resets_the_strike_count():
    """Two failures then a success must not leave the run one failure from death."""
    script = [
        assistant_tool_calls(("get_physician_profile", {"wrong": 1}), id_prefix="e1"),
        assistant_tool_calls(("get_physician_profile", {"wrong": 2}), id_prefix="e2"),
        assistant_tool_calls(FOLLOWUPS, id_prefix="ok"),
        assistant_tool_calls(("get_physician_profile", {"wrong": 3}), id_prefix="e3"),
        assistant_tool_calls(("get_physician_profile", {"wrong": 4}), id_prefix="e4"),
        assistant_tool_calls(ANSWER, id_prefix="fin"),
    ]
    result, _ = run_with(script, max_iterations=10)

    assert result.terminal_reason is TerminalReason.COMPLETED


def test_the_third_identical_call_terminates():
    script = [assistant_tool_calls(FOLLOWUPS, id_prefix=f"r{n}") for n in range(1, 4)]
    result, _ = run_with(script)

    assert result.terminal_reason is TerminalReason.REPEATED_CALL
    assert result.iterations == 3


def test_the_wall_clock_fires_without_anyone_sleeping():
    clock = FakeClock(0.0, 61.0)
    client = FakeModelClient([assistant_tool_calls(FOLLOWUPS), assistant_tool_calls(ANSWER)])
    loop = AgentLoop(
        client,
        build_registry(),
        policy_factory=lambda: TerminationPolicy(wall_clock_seconds=60.0, clock=clock),
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.WALL_CLOCK_EXCEEDED
    assert client.calls == []


def test_the_token_budget_fires():
    # The fake reports 120 tokens per call, so a 200-token budget is spent
    # after two of them.
    script = [
        assistant_tool_calls(("get_open_followups", {"physician_name": name}), id_prefix=f"t{n}")
        for n, name in enumerate(["Raj Patel", "Evelyn Chen", "Sofia Lindqvist"], 1)
    ]
    client = FakeModelClient(script)
    loop = AgentLoop(
        client, build_registry(), policy_factory=lambda: TerminationPolicy(token_budget=200)
    )
    result = loop.run(GOAL)

    assert result.terminal_reason is TerminalReason.TOKEN_BUDGET_EXCEEDED
    assert result.usage.total_tokens > 200


def test_no_progress_fires_when_the_model_neither_calls_nor_answers():
    result, client = run_with([assistant_text("")])

    assert result.terminal_reason is TerminalReason.NO_PROGRESS
    assert len(client.calls) == 1


def test_the_context_budget_fires():
    result, client = run_with([assistant_text("never reached")], context_budget=10)

    assert result.terminal_reason is TerminalReason.CONTEXT_BUDGET_EXCEEDED
    assert client.calls == []


def test_every_terminal_reason_is_reachable():
    """No run ends in a state no test produces.

    Enumerating the enum means a reason added without a way to reach it, or a
    condition quietly made unreachable, fails here rather than sitting in the
    codebase looking implemented.
    """
    produced = set()

    produced.add(run_with([assistant_tool_calls(ANSWER)])[0].terminal_reason)
    produced.add(
        run_with(
            [
                assistant_tool_calls(("get_open_followups", {"physician_name": n}), id_prefix=f"c{i}")
                for i, n in enumerate(["Raj Patel", "Evelyn Chen", "Sofia Lindqvist"], 1)
            ],
            max_iterations=3,
        )[0].terminal_reason
    )
    produced.add(
        run_with(
            [
                assistant_tool_calls(("get_physician_profile", {"wrong": n}), id_prefix=f"e{n}")
                for n in range(1, 4)
            ]
        )[0].terminal_reason
    )
    produced.add(
        run_with([assistant_tool_calls(FOLLOWUPS, id_prefix=f"r{n}") for n in range(1, 4)])[
            0
        ].terminal_reason
    )
    produced.add(run_with([assistant_text("")])[0].terminal_reason)
    produced.add(
        run_with([assistant_text("x")], context_budget=10)[0].terminal_reason
    )

    clock = FakeClock(0.0, 999.0)
    produced.add(
        AgentLoop(
            FakeModelClient([assistant_text("x")]),
            build_registry(),
            policy_factory=lambda: TerminationPolicy(clock=clock),
        )
        .run(GOAL)
        .terminal_reason
    )
    produced.add(
        AgentLoop(
            FakeModelClient([assistant_tool_calls(FOLLOWUPS), assistant_tool_calls(ANSWER)]),
            build_registry(),
            policy_factory=lambda: TerminationPolicy(token_budget=1),
        )
        .run(GOAL)
        .terminal_reason
    )

    assert produced == set(TerminalReason)


# --- repeat detection ---------------------------------------------------------

def counting_registry(tool_name: str, counter: list[int]) -> ToolRegistry:
    """A registry whose named tool records how many times it actually ran."""
    original = next(spec for spec in TOOL_SPECS if spec.name == tool_name)

    def counted(**kwargs):
        counter.append(1)
        return original.function(**kwargs)

    return ToolRegistry(
        [replace(spec, function=counted) if spec.name == tool_name else spec for spec in TOOL_SPECS]
    )


def test_the_second_identical_call_does_not_re_execute_the_tool():
    calls: list[int] = []
    registry = counting_registry("get_open_followups", calls)
    client = FakeModelClient(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="a"),
            assistant_tool_calls(FOLLOWUPS, id_prefix="b"),
            assistant_tool_calls(ANSWER, id_prefix="fin"),
        ]
    )
    result = AgentLoop(client, registry).run(GOAL)

    assert len(calls) == 1, "the tool ran again for a call whose answer was already known"
    assert result.terminal_reason is TerminalReason.COMPLETED

    second = [m for m in result.messages if m["role"] == "tool"][1]
    assert "repeat_of" in second["content"]
    assert "a_1" in second["content"]
    # The prior result comes back, so the model is not left guessing.
    assert "fu-003" in second["content"]


def test_two_calls_to_one_tool_with_different_arguments_are_not_repeats():
    """The comparison case: different arguments are legitimate work."""
    result, _ = run_with(
        [
            assistant_tool_calls(
                ("search_product_docs", {"query": "dosing", "product_name": "Nexovar"}),
                id_prefix="a",
            ),
            assistant_tool_calls(
                ("search_product_docs", {"query": "dosing", "product_name": "Cardizyn"}),
                id_prefix="b",
            ),
            assistant_tool_calls(ANSWER, id_prefix="fin"),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED


def test_a_byte_identical_result_from_different_arguments_is_deduplicated():
    """"Patel" and "Raj Patel" reach the same records. The bytes are already here."""
    result, _ = run_with(
        [
            assistant_tool_calls(("get_open_followups", {"physician_name": "Patel"}), id_prefix="a"),
            assistant_tool_calls(FOLLOWUPS, id_prefix="b"),
            assistant_tool_calls(ANSWER, id_prefix="fin"),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    second = [m for m in result.messages if m["role"] == "tool"][1]
    assert "duplicate_of" in second["content"]
    assert "a_1" in second["content"]
    # Deduplicating means not resending: the records appear once.
    assert second["content"].count("fu-003") == 0


# --- authorization ------------------------------------------------------------

def test_auto_mode_denies_nothing():
    policy = Policy()
    for spec in TOOL_SPECS:
        assert policy.authorize(spec) is None


def test_require_confirmation_refuses_to_pretend_it_is_implemented():
    policy = Policy(authorization_mode=AuthorizationMode.REQUIRE_CONFIRMATION)
    write_tool = next(spec for spec in TOOL_SPECS if spec.permission is Permission.WRITE)

    with pytest.raises(NotImplementedError, match="suspension and resume"):
        policy.authorize(write_tool)


def test_a_denial_is_model_visible_and_names_no_internals():
    denied = PermissionDenied(name="create_followup", reason="writes need confirmation")

    assert denied.model_visible is True
    assert "create_followup" in denied.payload()


# --- the error taxonomy, read off the types -----------------------------------

@pytest.mark.parametrize(
    "outcome, is_error",
    [
        (ToolSucceeded(name="t", result_json="{}"), False),
        (RepeatedCall(name="t", prior_tool_call_id="call_1", prior_payload="{}"), False),
        (DuplicateResult(name="t", duplicate_of="call_1"), False),
        (ToolFailed(name="t"), True),
        (InvalidArguments(name="t", message="m"), True),
        (UnknownTool(requested_name="t", valid_names=("a",)), True),
        (PermissionDenied(name="t", reason="r"), True),
    ],
    ids=lambda value: getattr(value, "__class__", type(value)).__name__,
)
def test_strike_counting_reads_off_the_type(outcome, is_error):
    assert outcome.is_error is is_error


def test_every_outcome_is_model_visible_because_fatal_failures_raise():
    for outcome in ToolOutcome.__subclasses__():
        assert outcome.model_visible is True


def test_the_policy_counts_only_errors_towards_strikes():
    policy = TerminationPolicy()
    policy.record_outcome(InvalidArguments(name="t", message="m"))
    policy.record_outcome(InvalidArguments(name="t", message="m"))
    policy.record_outcome(ToolSucceeded(name="t", result_json="{}"))
    policy.record_outcome(InvalidArguments(name="t", message="m"))

    assert policy.check(iteration=1, usage=TokenUsage()) is None


# --- how a run ends with an answer --------------------------------------------

def test_submit_final_answer_terminates_and_its_fields_survive():
    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="a"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "You owe Dr. Patel two follow-ups.",
                        "sources": ["a_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="fin",
            ),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.route is FinalAnswerRoute.SUBMIT_TOOL
    assert result.answer == "You owe Dr. Patel two follow-ups."
    assert result.sources == ["a_1"]
    assert result.insufficient_information is False


def test_insufficient_information_survives_as_declared():
    result, _ = run_with(
        [
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "There is no physician by that name on record.",
                        "sources": [],
                        "insufficient_information": True,
                    },
                )
            )
        ]
    )

    assert result.insufficient_information is True
    assert result.sources == []


def test_free_text_still_ends_the_run_and_is_recorded_as_the_other_route():
    """The model may ignore the control tool, and that path has to stay live."""
    result, _ = run_with([assistant_text("Two outstanding.")])

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.route is FinalAnswerRoute.FREE_TEXT
    assert result.answer == "Two outstanding."
    # No sources, which is exactly the difference M8 is measuring.
    assert result.sources == []


def test_a_run_that_never_answers_records_no_route():
    result, _ = run_with([assistant_text("")])
    assert result.route is FinalAnswerRoute.NONE


def test_the_control_tools_acknowledgement_does_not_echo_the_answer():
    """Echoing it would double the cost of the answer for no benefit."""
    answer = "A long answer that should appear once in the context, not twice."
    result, _ = run_with(
        [
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {"answer": answer, "sources": [], "insufficient_information": False},
                )
            )
        ]
    )

    tool_message = [m for m in result.messages if m["role"] == "tool"][0]
    assert answer not in tool_message["content"]
    assert "Answer recorded" in tool_message["content"]


# --- the ordering that the protocol invariant depends on ----------------------

def test_the_final_answers_own_tool_call_is_answered_before_the_run_ends():
    """Terminating on submit_final_answer must not skip its tool message.

    The control tool dispatches like any other, so a run that ended the moment
    the answer was validated would leave its own tool_call.id unanswered in the
    final context.
    """
    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="a"),
            assistant_tool_calls(ANSWER, id_prefix="fin"),
        ]
    )

    answered = {m["tool_call_id"] for m in result.messages if m["role"] == "tool"}
    requested = {
        call["id"]
        for message in result.messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    }
    assert requested == answered
    assert "fin_1" in answered


def test_the_guard_fires_if_a_final_answer_call_goes_unanswered():
    """Proof that the ordering is enforced rather than merely observed.

    This is what the loop would produce if it terminated before emitting the
    control tool's message: ContextBuilder refuses to hand back the array.
    """
    from agentharness.harness.context import ContextBuilder
    from agentharness.harness.model_client import HarnessFatalError

    context = ContextBuilder("system", GOAL, tools=[])
    context.add_assistant(assistant_tool_calls(ANSWER, id_prefix="fin"))

    with pytest.raises(HarnessFatalError, match="fin_1"):
        context.messages()


def test_harness_fatal_failures_never_become_a_tool_message():
    """The taxonomy's other half: fatal failures raise and are not outcomes."""
    from agentharness.harness.context import ContextBudgetExceeded
    from agentharness.harness.model_client import HarnessFatalError

    assert issubclass(ContextBudgetExceeded, HarnessFatalError)
    assert not issubclass(HarnessFatalError, ToolOutcome)

    result, _ = run_with([assistant_text("x")], context_budget=10)
    assert [m for m in result.messages if m["role"] == "tool"] == []


# --- sources are checked against what this run actually issued ----------------

def test_a_fabricated_source_id_is_rejected_and_the_message_names_the_real_ones():
    """Observed for real: the model emitted call_1..call_5 when the ids were
    long opaque strings. The field was being generated in a plausible shape
    rather than reported from the conversation."""
    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="real"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "Two outstanding.",
                        "sources": ["call_1", "call_2"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="bad",
            ),
            assistant_text("giving up"),
        ]
    )

    rejection = [m for m in result.messages if m["role"] == "tool"][1]["content"]
    assert "call_1" in rejection
    assert "never issued" in rejection
    assert "real_1" in rejection, "the model was not told which ids it may cite"


def test_real_source_ids_are_accepted():
    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="real"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "Two outstanding.",
                        "sources": ["real_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="fin",
            ),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.sources == ["real_1"]


def test_an_empty_source_list_is_accepted_when_the_answer_admits_the_gap():
    result, _ = run_with(
        [
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "There is no physician by that name on record.",
                        "sources": [],
                        "insufficient_information": True,
                    },
                )
            )
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.insufficient_information is True


def test_the_run_recovers_when_the_model_corrects_its_citation():
    """The rejection is a message the model can act on, not the end of the run."""
    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="real"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "Two outstanding.",
                        "sources": ["call_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="bad",
            ),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "Two outstanding.",
                        "sources": ["real_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="good",
            ),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.route is FinalAnswerRoute.SUBMIT_TOOL
    assert result.sources == ["real_1"]
    assert result.iterations == 3


def test_a_rejected_submission_does_not_end_the_run_or_record_an_answer():
    result, _ = run_with(
        [
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "invented",
                        "sources": ["call_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="bad",
            ),
            assistant_text("recovered"),
        ]
    )

    assert result.answer == "recovered"
    assert result.route is FinalAnswerRoute.FREE_TEXT


def test_citing_an_id_from_the_same_turn_is_rejected():
    """A call cannot cite itself: its own message does not exist yet."""
    result, _ = run_with(
        [
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "x",
                        "sources": ["self_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="self",
            ),
            assistant_text("recovered"),
        ]
    )

    rejection = [m for m in result.messages if m["role"] == "tool"][0]["content"]
    assert "never issued" in rejection


# --- the id the model is asked to cite is now text it can read ----------------

def test_the_tool_call_id_appears_in_the_envelope():
    """The model had never seen one: ids live in protocol fields, not content."""
    result, _ = run_with(
        [assistant_tool_calls(FOLLOWUPS, id_prefix="real"), assistant_text("done")]
    )

    envelope = json.loads([m for m in result.messages if m["role"] == "tool"][0]["content"])
    assert envelope["tool_call_id"] == "real_1"


def test_a_citation_read_from_an_envelope_round_trips_into_the_result():
    """End to end: the id is emitted in a tool message, cited back, and kept."""
    first, _ = run_with([assistant_tool_calls(FOLLOWUPS, id_prefix="real"), assistant_text("x")])
    issued = json.loads([m for m in first.messages if m["role"] == "tool"][0]["content"])[
        "tool_call_id"
    ]

    result, _ = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="real"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "Two outstanding.",
                        "sources": [issued],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="fin",
            ),
        ]
    )

    assert result.sources == [issued]
    assert result.route is FinalAnswerRoute.SUBMIT_TOOL


def test_a_rejected_citation_is_recovered_from_end_to_end():
    """The whole path, as a run: gather, fabricate, get rejected, correct, finish.

    Observed twice against a real model -- first as call_1..call_5, then as five
    UUIDs after the field description was strengthened. The run must survive the
    mistake rather than spend its last iteration on it.
    """
    result, client = run_with(
        [
            assistant_tool_calls(FOLLOWUPS, id_prefix="gather"),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "You owe Dr. Patel two follow-ups.",
                        "sources": [
                            "5f2c8b1e-9d4a-4c7b-8e3f-1a2b3c4d5e6f",
                            "call_3",
                        ],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="fabricated",
            ),
            assistant_tool_calls(
                (
                    "submit_final_answer",
                    {
                        "answer": "You owe Dr. Patel two follow-ups.",
                        "sources": ["gather_1"],
                        "insufficient_information": False,
                    },
                ),
                id_prefix="corrected",
            ),
        ]
    )

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.route is FinalAnswerRoute.SUBMIT_TOOL
    assert result.answer == "You owe Dr. Patel two follow-ups."
    assert result.sources == ["gather_1"]
    assert result.iterations == 3
    assert len(client.calls) == 3

    rejection = [m for m in result.messages if m["role"] == "tool"][1]["content"]
    assert "never issued" in rejection
    assert "gather_1" in rejection


def test_the_raised_cap_leaves_room_to_recover_on_the_hardest_goal():
    """Five gathering iterations, a rejected submit, then a corrected one.

    At the old cap of six this run would have ended in MAX_ITERATIONS with no
    answer: the recovery would not have fitted.
    """
    gathering = [
        assistant_tool_calls(
            ("get_physician_profile", {"physician_name": "Evelyn Chen"}), id_prefix="g1"
        ),
        assistant_tool_calls(
            ("get_previous_meetings", {"physician_name": "Evelyn Chen"}), id_prefix="g2"
        ),
        assistant_tool_calls(
            ("search_product_docs", {"query": "nexovar dosing", "product_name": "Nexovar"}),
            id_prefix="g3",
        ),
        assistant_tool_calls(
            ("search_product_docs", {"query": "renal impairment", "product_name": "Nexovar"}),
            id_prefix="g4",
        ),
        assistant_tool_calls(
            ("get_open_followups", {"physician_name": "Evelyn Chen"}), id_prefix="g5"
        ),
    ]
    bad = assistant_tool_calls(
        ("submit_final_answer", {"answer": "a", "sources": ["call_1"], "insufficient_information": False}),
        id_prefix="bad",
    )
    good = assistant_tool_calls(
        ("submit_final_answer", {"answer": "a", "sources": ["g1_1"], "insufficient_information": False}),
        id_prefix="good",
    )
    result, _ = run_with(gathering + [bad, good])

    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.iterations == 7
