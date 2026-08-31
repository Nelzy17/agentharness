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

# Enough unknown ids to show the model the pattern of its mistake without
# spending the message on a list it does not need to read in full.
MAX_REPORTED_UNKNOWN = 3


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


def validate_sources(
    tool_name: str, sources: list[str], issued_ids: frozenset[str]
) -> InvalidArguments | None:
    """Check cited tool_call ids against the ids this run actually issued.

    This is a second kind of argument validation, and the distinction is worth
    naming. Everything the args models enforce is *structural*: it depends only
    on the schema, so a value is valid or not on its own terms and pydantic can
    decide it. This is *contextual*: whether 'call_x' is a valid source depends
    on what happened earlier in this run, which no schema can know. It therefore
    cannot live in the args model, and it runs here, after the schema has passed
    and before the tool executes.

    The trust-boundary principle is unchanged. An id the model emits is
    untrusted until checked against something the harness knows -- and the
    harness knows exactly which ids it issued.

    Unknown ids are rejected rather than dropped. Silently emptying the list
    would leave an answer that cites nothing looking like an answer that needed
    to cite nothing, which is the same failure wearing a different hat.
    """
    unknown = [source for source in sources if source not in issued_ids]
    if not unknown:
        return None

    shown = ", ".join(f"'{source}'" for source in unknown[:MAX_REPORTED_UNKNOWN])
    if len(unknown) > MAX_REPORTED_UNKNOWN:
        shown += f" and {len(unknown) - MAX_REPORTED_UNKNOWN} more"
    citable = ", ".join(sorted(issued_ids)) if issued_ids else "none yet"
    return InvalidArguments(
        name=tool_name,
        message=(
            f"sources cites {shown}, which this conversation never issued. "
            f"Cite only these ids, copied exactly: {citable}."
        ),
    )
