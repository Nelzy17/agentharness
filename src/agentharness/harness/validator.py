"""The trust boundary.

Model-generated arguments become typed arguments here or they become nothing.
There is no other path from a tool call to a tool function.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agentharness.harness.outcomes import InvalidArguments, ToolOutcome, UnknownTool
from agentharness.harness.registry import ToolRegistry
from agentharness.tools.definitions import ToolSpec

# Leaves room for the "Invalid arguments for <tool>: " prefix inside the budget
# the message is tested against.
MAX_DETAIL_LENGTH = 200


@dataclass(frozen=True)
class ValidatedCall:
    """Arguments that passed the boundary, bound to the tool that will run them.

    Deliberately not a ToolOutcome. Nothing has happened yet, so there is no
    message for the model; the outcomes are what execution produces in M2.
    """

    spec: ToolSpec
    args: BaseModel


class Validator:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def validate(self, tool_name: str, raw_args: Any) -> ValidatedCall | ToolOutcome:
        """Resolve a tool call to either typed arguments or a model-visible error.

        `raw_args` is whatever came back from json.loads of the model's argument
        string, which is not necessarily a dict. model_validate handles that:
        anything that is not an object for this schema is a validation failure
        like any other, not a TypeError.
        """
        spec = self._registry.get(tool_name)
        if spec is None:
            return UnknownTool(
                requested_name=tool_name,
                valid_names=tuple(self._registry.names()),
            )
        try:
            args = spec.args_model.model_validate(raw_args)
        except ValidationError as error:
            return InvalidArguments(name=tool_name, message=_compact(error))
        return ValidatedCall(spec=spec, args=args)


def _compact(error: ValidationError) -> str:
    """One line per failed field: where it was, and what was expected.

    Pydantic's default rendering is several lines per error, carries a
    documentation URL, and echoes the input back. This text becomes a tool
    message and is the model's only chance to correct itself, so it is kept
    short, and it never repeats the input -- an argument the model was steered
    into producing should not be read back to it as though the harness endorsed
    it.
    """
    lines = [
        f"{'.'.join(str(part) for part in detail['loc']) or '(arguments)'}: {detail['msg']}"
        for detail in error.errors(
            include_url=False, include_input=False, include_context=False
        )
    ]
    message = "; ".join(lines)
    if len(message) > MAX_DETAIL_LENGTH:
        message = f"{message[: MAX_DETAIL_LENGTH - 3]}..."
    return message
