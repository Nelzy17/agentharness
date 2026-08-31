"""ModelClient: retry, fatal classification, and usage accrual.

Driven by a stub standing in for the SDK client, so no request leaves the
machine. The SDK's own exception types are constructed for real, because
classifying them correctly is the thing under test.
"""

from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError

from agentharness.harness.model_client import HarnessFatalError, ModelClient

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(code: int) -> APIStatusError:
    error_class = BadRequestError if code == 400 else APIStatusError
    return error_class(
        f"http {code}", response=httpx2.Response(code, request=REQUEST), body=None
    )


def completion(
    content="done",
    prompt_tokens=10,
    completion_tokens=5,
    tool_calls=None,
    cached_tokens=0,
    cache_write_tokens=0,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
            ),
        ),
    )


def tool_call(call_id="call_1", name="get_open_followups", arguments="{}"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class StubSDK:
    """Enough of the SDK surface to reach chat.completions.create."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_for(outcomes, **kwargs):
    sdk = StubSDK(outcomes)
    return ModelClient(sdk, model="test-model", backoff_seconds=0.0, **kwargs), sdk


# --- the ordinary path --------------------------------------------------------

def test_a_completion_becomes_an_assistant_message():
    client, _ = client_for([completion(content="here you go")])
    response = client.complete([{"role": "user", "content": "hi"}], tools=[])

    assert response.message.content == "here you go"
    assert response.message.tool_calls == ()


def test_tool_calls_are_converted_and_arguments_left_unparsed():
    client, _ = client_for(
        [completion(content=None, tool_calls=[tool_call(arguments='{"a": 1}')])]
    )
    response = client.complete([], tools=[])

    call = response.message.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_1", "get_open_followups", '{"a": 1}')


def test_parallel_tool_calls_is_disabled_on_the_wire():
    client, sdk = client_for([completion()])
    client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert sdk.requests[0]["parallel_tool_calls"] is False
    assert sdk.requests[0]["model"] == "test-model"


def test_reasoning_is_disabled_on_the_wire():
    """Chat Completions rejects function tools on this model family otherwise."""
    client, sdk = client_for([completion()])
    client.complete([], tools=[{"type": "function"}])

    assert sdk.requests[0]["reasoning_effort"] == "none"


# --- retry and classification -------------------------------------------------

@pytest.mark.parametrize(
    "error",
    [
        APITimeoutError(request=REQUEST),
        APIConnectionError(request=REQUEST),
        status_error(503),
        status_error(500),
    ],
    ids=["timeout", "connection", "503", "500"],
)
def test_transient_failures_are_retried_and_the_model_never_sees_them(error):
    client, sdk = client_for([error, completion(content="second attempt")])
    response = client.complete([], tools=[])

    assert response.message.content == "second attempt"
    assert len(sdk.requests) == 2


def test_a_400_is_fatal_and_is_not_retried():
    """A 400 is our bug -- a malformed message array or a rejected schema."""
    client, sdk = client_for([status_error(400), completion()])

    with pytest.raises(HarnessFatalError, match="rejected our request"):
        client.complete([], tools=[])
    assert len(sdk.requests) == 1


@pytest.mark.parametrize("code", [401, 404, 422])
def test_other_client_errors_are_fatal_and_are_not_retried(code):
    client, sdk = client_for([status_error(code), completion()])

    with pytest.raises(HarnessFatalError, match=str(code)):
        client.complete([], tools=[])
    assert len(sdk.requests) == 1


def test_retries_are_bounded_and_exhaustion_is_fatal():
    client, sdk = client_for([status_error(503)] * 3)

    with pytest.raises(HarnessFatalError, match="failed 3 times"):
        client.complete([], tools=[])
    assert len(sdk.requests) == 3


# --- usage ---------------------------------------------------------------------

def test_usage_accumulates_across_calls():
    client, _ = client_for(
        [
            completion(prompt_tokens=10, completion_tokens=5, cache_write_tokens=10),
            completion(prompt_tokens=30, completion_tokens=7, cached_tokens=10),
        ]
    )
    client.complete([], tools=[])
    client.complete([], tools=[])

    assert client.usage.prompt_tokens == 40
    assert client.usage.completion_tokens == 12
    assert client.usage.cached_tokens == 10
    assert client.usage.cache_write_tokens == 10
    # Cached and written tokens are part of the prompt count, not extra to it.
    assert client.usage.total_tokens == 52


def test_the_cache_counters_are_read_from_the_usage_payload():
    """Billed at a different rate, so counting only prompt_tokens overstates cost."""
    client, _ = client_for(
        [completion(prompt_tokens=2737, completion_tokens=100, cached_tokens=2298)]
    )
    response = client.complete([], tools=[])

    assert response.usage.cached_tokens == 2298
    assert response.usage.prompt_tokens == 2737


def test_a_response_with_no_cache_details_counts_zero_cached():
    """Absent on responses that cached nothing and on families that never cache."""
    client, _ = client_for(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=None))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )
        ]
    )
    client.complete([], tools=[])

    assert client.usage.cached_tokens == 0
    assert client.usage.prompt_tokens == 10


def test_usage_from_a_retried_call_counts_only_what_the_api_reported():
    """Accrual happens where the response arrives, not where it is returned.

    A 5xx carries no usage, so a retried call costs exactly what the successful
    attempt reported -- but the accrual point has to be inside the attempt loop
    for that to stay true once an attempt can be discarded.
    """
    client, sdk = client_for(
        [
            status_error(503),
            completion(prompt_tokens=30, completion_tokens=7, cached_tokens=20),
        ]
    )
    client.complete([], tools=[])

    assert len(sdk.requests) == 2
    assert client.usage.prompt_tokens == 30
    assert client.usage.completion_tokens == 7
    assert client.usage.cached_tokens == 20


def test_a_response_without_usage_does_not_break_accounting():
    client, _ = client_for([SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=None))], usage=None)])
    client.complete([], tools=[])

    assert client.usage.total_tokens == 0
