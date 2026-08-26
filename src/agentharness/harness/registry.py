"""Tool registration and the tool-definitions payload.

Registration is an explicit list, built at import from `TOOL_SPECS`. No
decorator populating a global on import, and no scanning of the filesystem:
five tools are readable by hand, and indirection here would hide the one place
that says what the model can do.
"""

from collections.abc import Sequence
from typing import Any

from agentharness.tools.definitions import TOOL_SPECS, ToolSpec, to_tool_definition


class ToolRegistry:
    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Two specs under one name means the model's call resolves to
            # whichever registered first. Fail at construction instead.
            raise ValueError(f"duplicate tool names: {', '.join(duplicates)}")
        self._specs = tuple(specs)

    def get(self, name: str) -> ToolSpec | None:
        for spec in self._specs:
            if spec.name == name:
                return spec
        return None

    def names(self) -> list[str]:
        return [spec.name for spec in self._specs]

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """The `tools` argument of a Chat Completions request."""
        return [to_tool_definition(spec) for spec in self._specs]


def build_registry() -> ToolRegistry:
    return ToolRegistry(TOOL_SPECS)
