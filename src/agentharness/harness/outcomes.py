"""Harness-level outcomes: whether a tool call could be executed at all.

A different axis from the domain statuses in `domain/models.py` (ok, empty,
not_found, ambiguous), which say what the data was. A member here exists for
every reason the model gets a message that is not the plain result of a tool
running.

Two classifications are declared on the type rather than decided by whoever
handles it. `model_visible` is the error taxonomy's first cut: everything here
is visible, and harness-fatal failures are exceptions that never become an
outcome at all. `is_error` is what the strike counter reads, so "which failures
count towards termination" is answered by looking at a class rather than by
finding the list in the policy.

CLAUDE.md rule 3: every tool_call.id receives exactly one role="tool" message.
Every member carries the payload of that message. Wrapping it -- envelope,
truncation, marker -- belongs to the sanitizer.
"""

import abc
import json
from dataclasses import dataclass
from typing import ClassVar


class ToolOutcome(abc.ABC):
    """One resolved tool call and the payload it owes the model."""

    # Harness-fatal failures raise; they are never outcomes. Everything that
    # reaches the model is one of these.
    model_visible: ClassVar[bool] = True

    # Whether this counts towards the consecutive-error strike limit.
    is_error: ClassVar[bool] = True

    # "result" payloads are JSON documents and are nested as objects by the
    # sanitizer; "error" payloads are text. Declared here so the sanitizer reads
    # a class attribute instead of testing types it has to be kept in step with.
    envelope_key: ClassVar[str] = "error"

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
class PermissionDenied(ToolOutcome):
    """The policy layer refused the call before it reached the tool.

    In AUTO mode nothing is refused, so this does not arise in a v1 run. It
    exists because the authorization boundary is a real part of the design and
    a seam that is visible in the code is worth more than one described in a
    comment.
    """

    name: str
    reason: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return f"The tool {self.name} was not run: {self.reason}"


@dataclass(frozen=True)
class RepeatedCall(ToolOutcome):
    """The same tool with the same arguments, called again.

    The tool is not re-executed. The prior result comes back with a note, which
    answers the question the model was really asking -- whether there is more to
    find -- rather than letting it spend a turn discovering there is not.

    Not an error: it has its own counter, and counting it twice would terminate
    a run for the wrong reason.
    """

    is_error: ClassVar[bool] = False
    envelope_key: ClassVar[str] = "result"

    name: str
    prior_tool_call_id: str
    prior_payload: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return json.dumps(
            {
                "repeat_of": self.prior_tool_call_id,
                "note": (
                    "You already called this tool with these arguments. It was "
                    "not run again. The previous result is unchanged and is "
                    "repeated here. Do not call it a third time."
                ),
                "previous_result": json.loads(self.prior_payload),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class DuplicateResult(ToolOutcome):
    """Different arguments, byte-identical result to one already in context.

    The result is not repeated: the model is holding those bytes already, and
    resending them buys nothing. Terminates nothing -- this is a statement about
    whether the context is worth more tokens, not about whether the model is
    stuck.
    """

    is_error: ClassVar[bool] = False
    envelope_key: ClassVar[str] = "result"

    name: str
    duplicate_of: str

    @property
    def tool_name(self) -> str:
        return self.name

    def payload(self) -> str:
        return json.dumps(
            {
                "duplicate_of": self.duplicate_of,
                "note": (
                    "This returned exactly what an earlier call returned. The "
                    "content is already in this conversation and is not "
                    "repeated. Searching again the same way will not find more."
                ),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class ToolSucceeded(ToolOutcome):
    """A tool ran and returned a domain result, carried already serialized."""

    is_error: ClassVar[bool] = False
    envelope_key: ClassVar[str] = "result"

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
