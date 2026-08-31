"""Deterministic test infrastructure.

Every scripted test in this and every later milestone runs through
FakeModelClient. It has the same surface as ModelClient -- `complete(messages,
tools)` and a cumulative `usage` -- so the loop cannot tell them apart.
"""

import copy
import json
from typing import Any

import pytest

from agentharness.domain import repository
from agentharness.harness.model_client import (
    AssistantMessage,
    ModelResponse,
    TokenUsage,
    ToolCall,
)

USAGE_PER_CALL = TokenUsage(
    prompt_tokens=100, completion_tokens=20, cached_tokens=40, cache_write_tokens=10
)


class ScriptExhausted(AssertionError):
    """The loop asked for more assistant messages than the test scripted.

    An assertion rather than a return of None: a loop that iterates further than
    the test expected should fail loudly and say how far it got, not hang or
    quietly answer with nothing.
    """


class FakeModelClient:
    # The tracer records which model produced a run; the fake says so plainly
    # rather than borrowing a real identifier.
    model = "fake-model"

    def __init__(
        self,
        script: list[AssistantMessage],
        usage_per_call: TokenUsage = USAGE_PER_CALL,
    ) -> None:
        self._script = list(script)
        self._scripted = len(script)
        self._usage_per_call = usage_per_call
        # The message array as it was at each call, copied so that later
        # mutation cannot rewrite what a test observed.
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []
        self.usage = TokenUsage()

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        self.calls.append(copy.deepcopy(messages))
        self.tools_seen.append(tools)
        if not self._script:
            raise ScriptExhausted(
                f"the loop made model call {len(self.calls)} but only "
                f"{self._scripted} assistant message(s) were scripted"
            )
        self.usage = self.usage + self._usage_per_call
        return ModelResponse(message=self._script.pop(0), usage=self._usage_per_call)


def assistant_text(content: str) -> AssistantMessage:
    """A final answer: content, no tool calls."""
    return AssistantMessage(content=content)


def assistant_tool_calls(
    *calls: tuple[str, Any], content: str | None = None, id_prefix: str = "call"
) -> AssistantMessage:
    """An assistant message requesting tools.

    Each call is a (tool_name, arguments) pair; arguments may be a dict, which
    is serialized, or a string, which is passed through untouched so that
    malformed JSON can be scripted. Ids are `<prefix>_1`, `<prefix>_2`, ... --
    unique within the message, which is what the protocol requires. Pass a
    distinct prefix when a test needs ids unique across several messages.
    """
    return AssistantMessage(
        content=content,
        tool_calls=tuple(
            ToolCall(
                id=f"{id_prefix}_{index}",
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            )
            for index, (name, arguments) in enumerate(calls, start=1)
        ),
    )


@pytest.fixture(autouse=True)
def fresh_repository():
    """The write tool mutates the in-memory store, so reload before each test."""
    repository.reset()
