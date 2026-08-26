"""Wording shared by the tools when a name does not resolve to exactly one record.

Shared strings only. Each tool still branches on all three resolution outcomes
itself, because each returns a different result type.
"""


def not_found_message(kind: str, name: str) -> str:
    return f"No {kind} matching '{name}' is on record."


def ambiguous_message(kind: str, name: str, candidates: tuple[str, ...]) -> str:
    return (
        f"'{name}' matches more than one {kind}: {', '.join(candidates)}. "
        "Ask which one is meant before continuing."
    )
