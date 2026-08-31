"""Harness-level outcomes: whether a tool call could be executed at all.

A different axis from the domain statuses in `domain/models.py` (ok, empty,
not_found, ambiguous), which say what the data was. A member here exists for
every reason the model gets a message that is not the result of a tool running.

CLAUDE.md rule 3: every tool_call.id receives exactly one role="tool" message.
Every member of this union renders that message and knows nothing else about it.
"""

import abc
from dataclasses import dataclass
from typing import ClassVar


class ToolOutcome(abc.ABC):
    """One resolved tool call, rendered as the single tool message it produces."""

    # The taxonomy's first cut: model-visible or harness-fatal. Every member
    # added in M1 is visible; the harness-fatal ones arrive with the policy
    # layer in M4 and the write cap in M6. The flag lives here rather than in
    # the loop so the classification is a property of the outcome itself.
    model_visible: ClassVar[bool] = True

    @abc.abstractmethod
    def to_tool_message(self) -> str:
        """The content of the role="tool" message answering this call."""


@dataclass(frozen=True)
class UnknownTool(ToolOutcome):
    """The model named a tool that is not registered.

    Carries the valid names because this message is the model's only chance to
    correct itself before the strike is spent.
    """

    requested_name: str
    valid_names: tuple[str, ...]

    def to_tool_message(self) -> str:
        return (
            f"There is no tool named '{self.requested_name}'. "
            f"Available tools: {', '.join(self.valid_names)}."
        )


@dataclass(frozen=True)
class InvalidArguments(ToolOutcome):
    """The arguments did not pass the tool's schema, so the tool did not run."""

    tool_name: str
    message: str

    def to_tool_message(self) -> str:
        return f"Invalid arguments for {self.tool_name}: {self.message}"


@dataclass(frozen=True)
class ToolSucceeded(ToolOutcome):
    """A tool ran and returned a domain result.

    The result is carried already serialized. It is always wrapped before it
    reaches the model (CLAUDE.md rule 4), so nothing a tool returned can be read
    as though the harness had said it.
    """

    tool_name: str
    result_json: str

    def to_tool_message(self) -> str:
        # Composed rather than re-parsed: result_json is already valid JSON, and
        # tool_name is ours -- it comes from the registered spec, never from the
        # model -- so neither needs escaping here. Truncation arrives with the
        # sanitizer in M3.
        return f'{{"tool": "{self.tool_name}", "result": {self.result_json}}}'


@dataclass(frozen=True)
class ToolFailed(ToolOutcome):
    """A tool raised. The model is told that much and nothing more."""

    tool_name: str

    def to_tool_message(self) -> str:
        return (
            f"The tool {self.tool_name} failed unexpectedly and returned nothing. "
            "This is a fault in the system, not a problem with how you called it. "
            "Do not call it again: continue without it, and tell the user what "
            "you were unable to retrieve."
        )
