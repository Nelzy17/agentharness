"""Context assembly.

The only place in the codebase that builds a message. Architecture principle 5:
if a second file ever grows a dict with a "role" key, the ordering, the
truncation and the tool-call protocol all become everyone's problem instead of
this file's.
"""

from pathlib import Path
from typing import Any

from agentharness.harness.model_client import AssistantMessage, HarnessFatalError

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt from disk at runtime. Prompts are never string literals."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


class ContextBuilder:
    """Owns the message array for one run."""

    def __init__(self, system_prompt: str, goal: str) -> None:
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

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

    def messages(self) -> list[dict[str, Any]]:
        self._assert_every_tool_call_answered()
        return list(self._messages)

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
