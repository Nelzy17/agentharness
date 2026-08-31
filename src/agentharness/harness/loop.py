"""The execution loop: context in, model call, tool dispatch, result back, repeat."""

import enum
import json
from dataclasses import dataclass
from typing import Any

from agentharness.harness.context import ContextBuilder, load_prompt
from agentharness.harness.dispatcher import dispatch
from agentharness.harness.model_client import TokenUsage, ToolCall
from agentharness.harness.outcomes import InvalidArguments, ToolOutcome
from agentharness.harness.registry import ToolRegistry
from agentharness.harness.validator import ValidatedCall, Validator

MAX_ITERATIONS = 6


class TerminalReason(enum.Enum):
    """Why a run stopped. The value is the human-readable reason.

    M4 adds the rest of the termination policy -- strikes, repeats, wall clock,
    token budget, no progress -- and will likely move this enum to policy.py
    with it. No run ends in a state that is not one of these.
    """

    COMPLETED = "the model produced a final answer"
    MAX_ITERATIONS = "the run reached the iteration cap without a final answer"


@dataclass(frozen=True)
class RunResult:
    answer: str | None
    terminal_reason: TerminalReason
    iterations: int
    usage: TokenUsage
    messages: list[dict[str, Any]]


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
    ) -> None:
        self._model_client = model_client
        self._registry = registry
        self._validator = Validator(registry)
        self._max_iterations = max_iterations
        self._system_prompt = load_prompt("system")

    def run(self, goal: str) -> RunResult:
        context = ContextBuilder(self._system_prompt, goal)
        tools = self._registry.tool_definitions()
        last_content: str | None = None

        for iteration in range(1, self._max_iterations + 1):
            response = self._model_client.complete(context.messages(), tools)
            context.add_assistant(response.message)
            last_content = response.message.content or last_content

            if not response.message.tool_calls:
                return self._result(
                    response.message.content,
                    TerminalReason.COMPLETED,
                    iteration,
                    context,
                )

            # Two phases, deliberately. Every tool call is resolved to an
            # outcome first, and only then are the tool messages emitted. The
            # outcome list is built from the tool call list, so it has the same
            # length by construction and no branch can skip an answer. Emitting
            # inside the resolution branch is how CLAUDE.md rule 3 gets broken.
            outcomes = [self._resolve(call) for call in response.message.tool_calls]
            for call, outcome in zip(response.message.tool_calls, outcomes, strict=True):
                context.add_tool_result(call.id, outcome.to_tool_message())

        return self._result(
            last_content, TerminalReason.MAX_ITERATIONS, self._max_iterations, context
        )

    def _resolve(self, call: ToolCall) -> ToolOutcome:
        """One tool call, from raw model output to something the model can read."""
        try:
            raw_args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            # Arguments arrive as a string, so they can be malformed before any
            # schema is consulted. Same class of failure as a schema rejection:
            # the model wrote the arguments, and the model can fix them.
            return InvalidArguments(
                tool_name=call.name, message="the arguments were not valid JSON"
            )

        outcome = self._validator.validate(call.name, raw_args)
        if isinstance(outcome, ValidatedCall):
            return dispatch(outcome)
        return outcome

    def _result(
        self,
        answer: str | None,
        reason: TerminalReason,
        iterations: int,
        context: ContextBuilder,
    ) -> RunResult:
        return RunResult(
            answer=answer,
            terminal_reason=reason,
            iterations=iterations,
            usage=self._model_client.usage,
            messages=context.messages(),
        )
