"""Context assembly and the token budget.

The only place in the codebase that builds a message. Architecture principle 5:
if a second file ever grows a dict with a "role" key, the ordering, the
truncation and the tool-call protocol all become everyone's problem instead of
this file's.

The stable prefix -- system prompt and tool definitions -- is owned here as one
thing, because it is one thing to the API: the bytes that get cached. Nothing
reorders it, rewrites it, or includes it conditionally.
"""

import functools
import json
from pathlib import Path
from typing import Any

import tiktoken

from agentharness.harness.model_client import AssistantMessage, HarnessFatalError

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

# Measured, at the M4 iteration cap of eight and with every result at the 2,000
# character cap, the largest context this harness can assemble is 6,289 tokens.
# The budget is a backstop that proves the bound, not a working constraint.
MAX_CONTEXT_TOKENS = 12_000

# Fixed rather than derived from the model name. tiktoken raises on identifiers
# it does not know, which would turn the budget into a configuration branch,
# and the count is an estimate used for a limit rather than for billing.
ENCODING_NAME = "o200k_base"

# Each message and each tool call costs a few tokens beyond its text for
# delimiters. These are properties of the serving format rather than of the
# tokenizer, so they are conventional approximations; the calibration test is
# what says whether they are close enough to be useful.
TOKENS_PER_MESSAGE = 4
TOKENS_PER_TOOL_CALL = 4

# Measured, not guessed. On the smoke run's first call the API reported 1,300
# prompt tokens where counting our serialized tool JSON gave 1,181 for the tools
# alone against an implied true cost of about 976 -- an overcount of roughly 21%,
# because the API bills its own representation of the schema rather than the JSON
# text we happen to send. The factor corrects that one component; 0.85 sits
# slightly above the measurement so the estimate stays on the conservative side.
# Re-measure if the tool set changes.
TOOL_SCHEMA_FACTOR = 0.85


class ContextBudgetExceeded(HarnessFatalError):
    """The assembled context is larger than the run is allowed to send."""


def load_prompt(name: str) -> str:
    """Read a prompt from disk at runtime. Prompts are never string literals."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


@functools.lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """Estimate the prompt tokens a request carrying these would cost.

    The tool definitions are counted. They are the larger half of a fresh
    context here, and a budget that ignored them would be measuring the smaller
    one.
    """
    encoding = _encoding()
    tool_json = json.dumps(tools, ensure_ascii=False)
    total = round(len(encoding.encode(tool_json)) * TOOL_SCHEMA_FACTOR)
    return total + sum(_message_tokens(message, encoding) for message in messages)


def _message_tokens(message: dict[str, Any], encoding: tiktoken.Encoding) -> int:
    """Count the text a message actually contributes, not our dict's JSON.

    Encoding `json.dumps(message)` counts the key names and escapes every
    newline into a two-character sequence, which overstated the two-message
    opening context by 35 tokens. Counting the parts the API actually transmits
    is both closer and easier to reason about.
    """
    total = TOKENS_PER_MESSAGE + len(encoding.encode(message["role"]))
    total += len(encoding.encode(message.get("content") or ""))
    total += len(encoding.encode(message.get("tool_call_id", "")))
    for call in message.get("tool_calls", []):
        total += TOKENS_PER_TOOL_CALL
        total += len(encoding.encode(call["function"]["name"]))
        total += len(encoding.encode(call["function"]["arguments"]))
    return total


class ContextBuilder:
    """Owns the message array, the stable prefix, and the budget for one run."""

    def __init__(
        self,
        system_prompt: str,
        goal: str,
        tools: list[dict[str, Any]],
        token_budget: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self._tools = tools
        self._token_budget = token_budget
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

    def tool_definitions(self) -> list[dict[str, Any]]:
        """The other half of the stable prefix, handed out rather than rebuilt."""
        return self._tools

    def add_assistant(self, message: AssistantMessage) -> None:
        entry: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        self._messages.append(entry)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def answered_tool_call_ids(self) -> frozenset[str]:
        """Every tool_call_id this context has answered.

        The harness knows these because it built them. That is what makes them
        usable as the authority against which a model-supplied id is checked.
        """
        return frozenset(
            message["tool_call_id"]
            for message in self._messages
            if message["role"] == "tool"
        )

    def token_count(self) -> int:
        return count_tokens(self._messages, self._tools)

    def messages(self) -> list[dict[str, Any]]:
        """The context as it stands. No budget check: used for inspection."""
        self._assert_every_tool_call_answered()
        return list(self._messages)

    def messages_for_model_call(self) -> list[dict[str, Any]]:
        """The context to send, or a refusal to send it.

        Nothing is dropped or rewritten to fit. Messages are appended and never
        revisited, so truncation only ever affects the tail -- which is what
        keeps the cached prefix intact, and what makes an overrun a stop rather
        than a silent loss of something already retrieved.
        """
        messages = self.messages()
        used = count_tokens(messages, self._tools)
        if used > self._token_budget:
            raise ContextBudgetExceeded(
                f"the assembled context is {used} tokens against a budget of "
                f"{self._token_budget}"
            )
        return messages

    def _assert_every_tool_call_answered(self) -> None:
        """CLAUDE.md rule 3, checked where the messages are built.

        An unanswered tool call is a 400 from the API on the next request, which
        reads like a model fault and is not one. Checking here means the bug
        surfaces in our own code, naming the call that went unanswered, instead
        of as an opaque rejection a second later.
        """
        pending: set[str] = set()
        for message in self._messages:
            if message["role"] == "assistant":
                if pending:
                    raise HarnessFatalError(
                        f"tool calls {sorted(pending)} were never answered with a "
                        "tool message"
                    )
                pending = {call["id"] for call in message.get("tool_calls", [])}
            elif message["role"] == "tool":
                pending.discard(message["tool_call_id"])
        if pending:
            raise HarnessFatalError(
                f"tool calls {sorted(pending)} were never answered with a tool message"
            )
