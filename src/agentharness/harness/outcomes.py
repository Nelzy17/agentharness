"""Harness-level outcomes: whether a tool call could be executed at all.

A different axis from the domain statuses in `domain/models.py` (ok, empty,
not_found, ambiguous), which say what the data was. A member here exists for
every reason the model gets a message that is not the result of a tool running.

CLAUDE.md rule 3: every tool_call.id receives exactly one role="tool" message.
Every member of this union carries the payload of that message. Turning a
payload into the string that goes on the wire -- envelope, truncation, marker --
belongs to the sanitizer, not here.
"""

import abc
from dataclasses import dataclass
from typing import ClassVar


class ToolOutcome(abc.ABC):
    """One resolved tool call and the payload it owes the model."""

    # The taxonomy's first cut: model-visible or harness-fatal. Every member
    # added in M1 is visible; the harness-fatal ones arrive with the policy
    # layer in M4 and the write cap in M6. The flag lives here rather than in
    # the loop so the classification is a property of the outcome itself.
    model_visible: ClassVar[bool] = True

    @property
    @abc.abstractmethod
    def tool_name(self) -> str:
        """The tool this outcome answers for. May be untrusted -- see UnknownTool."""

    @abc.abstractmethod
    def payload(self) -> str:
        """The content of this outcome, before the sanitizer wraps it."""


@dataclass(frozen=True)
class UnknownTool(ToolOutcome):
    """The model named a tool that is not registered.

    Carries the valid names because this message is the model's only chance to
    correct itself before the strike is spent.
    """

    requested_name: str
    valid_names: tuple[str, ...]

    @property
    def tool_name(self) -> str:
        # Model-supplied, so it is untrusted text that happens to sit in a
        # structural field. The sanitizer escapes and bounds it.
        return self.requested_name

    def payload(self) -> str:
        return (
            f"There is no tool named '{self.requested_name}'. "
            f"Available tools: {', '.join(self.valid_names)}."
        )


@dataclass(frozen=True)
class InvalidArguments(ToolOutcome):
    """The arguments did not pass the tool's schema, so the tool did not run."""

    name: str
    message: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return f"Invalid arguments for {self.name}: {self.message}"


@dataclass(frozen=True)
class ToolSucceeded(ToolOutcome):
    """A tool ran and returned a domain result, carried already serialized."""

    name: str
    result_json: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return self.result_json


@dataclass(frozen=True)
class ToolFailed(ToolOutcome):
    """A tool raised. The model is told that much and nothing more."""

    name: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return (
            f"The tool {self.name} failed unexpectedly and returned nothing. "
            "This is a fault in the system, not a problem with how you called it. "
            "Do not call it again: continue without it, and tell the user what "
            "you were unable to retrieve."
        )
