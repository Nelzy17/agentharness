"""The only file in the codebase that touches the OpenAI SDK.

The SDK's types stop here. Everything above this line works with the small
types declared below, which is what lets the loop run against a fake client
without knowing it, and what keeps the message array our own problem rather
than the SDK's.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.5

logger = logging.getLogger(__name__)


class HarnessFatalError(RuntimeError):
    """The run cannot continue, and the cause is ours rather than the model's."""


@dataclass(frozen=True)
class TokenUsage:
    """What one or more model calls cost, in the units the API reports.

    `cached_tokens` and `cache_write_tokens` are subsets of `prompt_tokens`,
    not additions to it, so `total_tokens` stays prompt plus completion. They
    are carried separately because they bill at different rates -- cached input
    at roughly a tenth of standard -- and a cost figure computed from
    `prompt_tokens` alone overstates the bill substantially. Applying the rates
    is M8's job; counting honestly is this file's.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # The raw JSON string the model emitted. It is not parsed here: parsing is
    # trust-boundary work and belongs beside validation.
    arguments: str


@dataclass(frozen=True)
class AssistantMessage:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    message: AssistantMessage
    usage: TokenUsage


class ModelClient:
    """Wraps one chat completion call: retry, classification, usage accrual.

    The SDK client is injected rather than constructed here so that retries,
    fatal classification and usage accounting can be tested without a network.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_seconds: float = BACKOFF_SECONDS,
    ) -> None:
        self._client = client
        self._model = model
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self.usage = TokenUsage()

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    # v1 is sequential. The response still carries a list of
                    # tool calls, and the loop still handles it as one.
                    parallel_tool_calls=False,
                    # Not a performance setting. Chat Completions rejects
                    # function tools outright on this model family unless
                    # reasoning is disabled, returning a 400. The API's own
                    # suggestion is to move to /v1/responses, which would
                    # manage the message array server-side -- the one thing
                    # this harness exists to do by hand.
                    reasoning_effort="none",
                )
            except BadRequestError as error:
                # A 400 means we sent something malformed: a message array that
                # breaks the tool-call protocol, or a schema the API rejects.
                # Retrying sends the same broken request again.
                raise HarnessFatalError(
                    f"the API rejected our request: {error}"
                ) from error
            except (APITimeoutError, APIConnectionError) as error:
                last_error = error
            except APIStatusError as error:
                if error.status_code < 500:
                    raise HarnessFatalError(
                        f"the API returned {error.status_code}: {error}"
                    ) from error
                last_error = error
            else:
                # Accrued the moment a response exists and before anything
                # inspects it. A response that arrives and is then discarded
                # still cost the user tokens.
                usage = _usage_of(completion)
                self.usage = self.usage + usage
                return ModelResponse(message=_assistant_message(completion), usage=usage)

            if attempt < self._max_attempts:
                time.sleep(self._backoff_seconds * 2 ** (attempt - 1))

        raise HarnessFatalError(
            f"the model API failed {self._max_attempts} times, last error: {last_error}"
        )


def _assistant_message(completion: Any) -> AssistantMessage:
    message = completion.choices[0].message
    return AssistantMessage(
        content=message.content,
        tool_calls=tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ),
    )


def _usage_of(completion: Any) -> TokenUsage:
    usage = getattr(completion, "usage", None)
    # Logged rather than returned: the whole payload is an SDK object, and SDK
    # objects stop in this file. This is how a caller inspects what the API
    # actually reported without the type escaping upwards.
    logger.debug("usage payload: %r", usage)
    if usage is None:
        return TokenUsage()
    # The cache counters live on prompt_tokens_details, which is absent on
    # responses that cached nothing and on model families that do not cache at
    # all. cache_write_tokens has been seen at both levels, so both are read.
    details = getattr(usage, "prompt_tokens_details", None)
    return TokenUsage(
        prompt_tokens=_count(usage, "prompt_tokens"),
        completion_tokens=_count(usage, "completion_tokens"),
        cached_tokens=_count(details, "cached_tokens"),
        cache_write_tokens=(
            _count(details, "cache_write_tokens") or _count(usage, "cache_write_tokens")
        ),
    )


def _count(payload: Any, name: str) -> int:
    return getattr(payload, name, 0) or 0
