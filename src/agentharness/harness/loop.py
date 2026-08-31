"""The execution loop: context in, model call, tool dispatch, result back, repeat."""

import enum
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentharness.harness.context import (
    MAX_CONTEXT_TOKENS,
    ContextBudgetExceeded,
    ContextBuilder,
    load_prompt,
)
from agentharness.harness.dispatcher import dispatch
from agentharness.harness.model_client import TokenUsage, ToolCall
from agentharness.harness.outcomes import (
    DuplicateResult,
    InvalidArguments,
    RepeatedCall,
    ToolOutcome,
    ToolSucceeded,
)
from agentharness.harness.policy import (
    MAX_ITERATIONS,
    Policy,
    TerminalReason,
    TerminationPolicy,
)
from agentharness.harness.registry import ToolRegistry
from agentharness.harness.sanitizer import sanitize
from agentharness.harness.validator import ValidatedCall, Validator, validate_sources
from agentharness.tools.definitions import SubmitFinalAnswerArgs

FINAL_ANSWER_TOOL = "submit_final_answer"


class FinalAnswerRoute(enum.Enum):
    """How a run ended, when it ended with an answer.

    Both routes stay live. The model may ignore the control tool and simply stop
    calling tools, and that path must keep working -- but the difference is an
    answer with checkable sources versus one without, so M8 wants the rate.
    """

    SUBMIT_TOOL = "the model called submit_final_answer"
    FREE_TEXT = "the model stopped calling tools and answered in prose"
    NONE = "the run ended without an answer"


@dataclass(frozen=True)
class RunResult:
    answer: str | None
    terminal_reason: TerminalReason
    iterations: int
    usage: TokenUsage
    messages: list[dict[str, Any]]
    route: FinalAnswerRoute = FinalAnswerRoute.NONE
    sources: list[str] = field(default_factory=list)
    insufficient_information: bool = False


class AgentLoop:
    """Runs one goal to a terminal state.

    `model_client` is anything with `complete(messages, tools) -> ModelResponse`
    and a cumulative `usage`. The loop never learns which implementation it has,
    which is what makes every path here testable without a network.
    """

    def __init__(
        self,
        model_client: Any,
        registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
        context_budget: int = MAX_CONTEXT_TOKENS,
        policy: Policy | None = None,
        policy_factory: Callable[[], TerminationPolicy] | None = None,
    ) -> None:
        self._model_client = model_client
        self._registry = registry
        self._validator = Validator(registry)
        self._context_budget = context_budget
        self._policy = policy or Policy()
        self._policy_factory = policy_factory or (
            lambda: TerminationPolicy(max_iterations=max_iterations)
        )
        self._system_prompt = load_prompt("system")

    def run(self, goal: str) -> RunResult:
        context = ContextBuilder(
            self._system_prompt,
            goal,
            self._registry.tool_definitions(),
            token_budget=self._context_budget,
        )
        termination = self._policy_factory()
        last_content: str | None = None
        iteration = 0

        while True:
            iteration += 1

            reason = termination.check(iteration, self._model_client.usage)
            if reason is not None:
                return self._result(last_content, reason, iteration - 1, context)

            try:
                messages = context.messages_for_model_call()
            except ContextBudgetExceeded:
                # Stopping is the policy. Dropping older exchanges to fit would
                # forget something already retrieved, and would shift every byte
                # after the drop, destroying the cached prefix and re-billing
                # the remainder at full rate.
                return self._result(
                    last_content,
                    TerminalReason.CONTEXT_BUDGET_EXCEEDED,
                    iteration - 1,
                    context,
                )

            response = self._model_client.complete(messages, context.tool_definitions())
            context.add_assistant(response.message)
            last_content = response.message.content or last_content

            if not response.message.tool_calls:
                if response.message.content:
                    return self._result(
                        response.message.content,
                        TerminalReason.COMPLETED,
                        iteration,
                        context,
                        route=FinalAnswerRoute.FREE_TEXT,
                    )
                # Neither a tool call nor an answer. Recorded rather than
                # returned, so that every stop is decided in one place.
                termination.record_no_progress()
                continue

            # The ids this run has actually issued, which is what a cited
            # source is checked against.
            issued = context.answered_tool_call_ids()

            # Three phases, and the order is the invariant. Every tool call is
            # resolved to an outcome, then every outcome becomes a tool message,
            # and only then may the run end. Ending inside the resolution loop
            # would leave a tool_call.id unanswered in the final context, which
            # ContextBuilder refuses to build -- so the bug would surface here
            # rather than on a request that is never made.
            resolutions = [
                self._resolve(call, termination, issued)
                for call in response.message.tool_calls
            ]
            for call, (outcome, _) in zip(
                response.message.tool_calls, resolutions, strict=True
            ):
                termination.record_outcome(outcome)
                context.add_tool_result(call.id, sanitize(outcome, call.id))

            final = next((answer for _, answer in resolutions if answer), None)
            if final is not None:
                return self._result(
                    final.answer,
                    TerminalReason.COMPLETED,
                    iteration,
                    context,
                    route=FinalAnswerRoute.SUBMIT_TOOL,
                    sources=list(final.sources),
                    insufficient_information=final.insufficient_information,
                )

    def _resolve(
        self,
        call: ToolCall,
        termination: TerminationPolicy,
        issued_ids: frozenset[str],
    ) -> tuple[ToolOutcome, SubmitFinalAnswerArgs | None]:
        """One tool call, from raw model output to something the model can read.

        Returns the outcome and, for the control tool, the validated answer. The
        answer comes back as a return value rather than as a side effect so the
        loop can emit every tool message before acting on it.
        """
        try:
            raw_args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            # Arguments arrive as a string, so they can be malformed before any
            # schema is consulted. Same class of failure as a schema rejection:
            # the model wrote the arguments, and the model can fix them.
            return (
                InvalidArguments(
                    name=call.name, message="the arguments were not valid JSON"
                ),
                None,
            )

        validated = self._validator.validate(call.name, raw_args)
        if not isinstance(validated, ValidatedCall):
            return validated, None

        denial = self._policy.authorize(validated.spec)
        if denial is not None:
            return denial, None

        if call.name == FINAL_ANSWER_TOOL:
            # Contextual validation: the schema cannot know which ids this run
            # issued, so the check runs here, after the shape has passed and
            # before the answer is accepted.
            rejection = validate_sources(call.name, validated.args.sources, issued_ids)
            if rejection is not None:
                return rejection, None

        prior = termination.check_repeat(call.name, raw_args)
        if prior is not None:
            return (
                RepeatedCall(
                    name=call.name,
                    prior_tool_call_id=prior.tool_call_id,
                    prior_payload=prior.payload,
                ),
                None,
            )

        outcome = dispatch(validated)
        termination.record_call(call.name, raw_args, call.id, outcome.payload())

        final = validated.args if call.name == FINAL_ANSWER_TOOL else None
        if final is None and isinstance(outcome, ToolSucceeded):
            duplicate = termination.duplicate_of(outcome.payload(), call.id)
            if duplicate is not None:
                return DuplicateResult(name=call.name, duplicate_of=duplicate), None

        return outcome, final

    def _result(
        self,
        answer: str | None,
        reason: TerminalReason,
        iterations: int,
        context: ContextBuilder,
        route: FinalAnswerRoute = FinalAnswerRoute.NONE,
        sources: list[str] | None = None,
        insufficient_information: bool = False,
    ) -> RunResult:
        return RunResult(
            answer=answer,
            terminal_reason=reason,
            iterations=iterations,
            usage=self._model_client.usage,
            messages=context.messages(),
            route=route,
            sources=sources or [],
            insufficient_information=insufficient_information,
        )
