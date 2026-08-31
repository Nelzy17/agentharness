"""Termination and authorization: why a run stops, and what it is allowed to do.

Every condition that can end a run is evaluated in `TerminationPolicy.check`,
called once per iteration before anything else happens. State is recorded as
events occur; the decision is made in one ordered list. Adding a seventh
condition means adding a clause there, not another exit somewhere in the loop.

The one condition not evaluated here is the context budget. It is enforced in
ContextBuilder, where the context is assembled, because it is a property of the
assembled context rather than of the run. Its reason still belongs to the enum
below, so no run ends in a state this file does not name.
"""

import enum
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentharness.harness.model_client import TokenUsage
from agentharness.harness.outcomes import PermissionDenied, ToolOutcome
from agentharness.tools.definitions import Permission, ToolSpec

# Eight, not six. The hardest smoke goal needs five gathering iterations plus
# one to submit, which is exactly six -- leaving no room for any of the recovery
# paths the error taxonomy exists to provide. A rejected citation, a validation
# failure or a repeat would each guarantee MAX_ITERATIONS on that goal. A cap
# that makes recovery impossible is not a safety mechanism, it is a second
# failure mode.
MAX_ITERATIONS = 8
MAX_CONSECUTIVE_ERRORS = 3

# Two records per run. The number is small because it is a blast radius rather
# than a budget: writes execute with no human saying yes, so the question is not
# "how many does a reasonable run need" but "how much damage can one run do
# before something stops it".
MAX_WRITES_PER_RUN = 2

# The second identical call returns the prior result; the third ends the run.
MAX_IDENTICAL_CALLS = 3

WALL_CLOCK_SECONDS = 60.0

# A backstop, not a working constraint, and raised with the iteration cap.
# Measured at eight iterations against worst-case capped results: the context
# tops out at 6,289 tokens and the run costs about 32,600 in total. 40,000 would
# have left 1.2x headroom, which is close enough that a legitimately hard run
# could trip it -- and a backstop that ends good runs is a working constraint
# wearing a backstop's name. 65,000 restores roughly 2x.
#
# Worth noting why the increase is not proportional: raising the cap from six to
# eight is 33% more iterations but 55% more tokens, because every iteration
# re-sends the whole context. Iteration cost is quadratic in the cap, not linear.
MAX_TOTAL_TOKENS = 65_000


class TerminalReason(enum.Enum):
    """Why a run stopped. The value is the human-readable reason.

    No run ends in a state that is not one of these, and a test enumerates this
    enum and asserts every member is produced by some scripted run.
    """

    COMPLETED = "the model produced a final answer"
    MAX_ITERATIONS = "the run reached the iteration cap without a final answer"
    CONSECUTIVE_ERRORS = "the model made three failing tool calls in a row"
    REPEATED_CALL = "the model called the same tool with the same arguments three times"
    WALL_CLOCK_EXCEEDED = "the run exceeded its wall-clock budget"
    TOKEN_BUDGET_EXCEEDED = "the run exceeded its cumulative token budget"
    NO_PROGRESS = "the model returned neither a tool call nor an answer"
    WRITE_CAP_EXCEEDED = "the run attempted more writes than it is allowed to make"
    HARNESS_ERROR = (
        "the harness could not continue -- a fault on our side, not the model's"
    )
    CONTEXT_BUDGET_EXCEEDED = (
        "the assembled context exceeded the token budget, so the run stopped "
        "rather than discard what it had already retrieved"
    )


class AuthorizationMode(enum.Enum):
    AUTO = "auto"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass(frozen=True)
class PriorCall:
    tool_call_id: str
    payload: str


@dataclass(frozen=True)
class Policy:
    """What the model is allowed to do without a human.

    v1 executes writes autonomously (DECISIONS, Decision A). The safety story is
    bounded blast radius and attribution, not approval. REQUIRE_CONFIRMATION is
    defined and not implemented on purpose: it needs run suspension and resume,
    which is deliberately out of scope, and a mode that silently behaved like
    AUTO would be worse than one that refuses to pretend.
    """

    authorization_mode: AuthorizationMode = AuthorizationMode.AUTO

    def authorize(self, spec: ToolSpec) -> PermissionDenied | None:
        if self.authorization_mode is AuthorizationMode.AUTO:
            return None
        raise NotImplementedError(
            "REQUIRE_CONFIRMATION needs run suspension and resume, which v1 "
            "does not build. See DECISIONS.md, Decision A."
        )

    def writes(self, spec: ToolSpec) -> bool:
        return spec.permission is Permission.WRITE


class TerminationPolicy:
    """Every reason a run can stop, evaluated in one place."""

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
        max_identical_calls: int = MAX_IDENTICAL_CALLS,
        max_writes: int = MAX_WRITES_PER_RUN,
        wall_clock_seconds: float = WALL_CLOCK_SECONDS,
        token_budget: int = MAX_TOTAL_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_iterations = max_iterations
        self._max_consecutive_errors = max_consecutive_errors
        self._max_identical_calls = max_identical_calls
        self.max_writes = max_writes
        self._wall_clock_seconds = wall_clock_seconds
        self._token_budget = token_budget
        self._clock = clock
        self._started_at = clock()

        self._consecutive_errors = 0
        self._calls: dict[str, PriorCall] = {}
        self._call_counts: dict[str, int] = {}
        self._results: dict[str, str] = {}
        self._repeat_limit_reached = False
        self._no_progress = False
        self._write_attempts = 0
        self._write_cap_reached = False

    # --- the single decision point -------------------------------------------

    def check(self, iteration: int, usage: TokenUsage) -> TerminalReason | None:
        """The one place a run's continuation is decided.

        Ordered most specific first, so the reason reported is the one that
        explains the run rather than the one that happens to also be true. The
        iteration cap is last: it is the catch-all, and if it fires while
        something more diagnostic is true, the more diagnostic answer is better.
        """
        if self._no_progress:
            return TerminalReason.NO_PROGRESS
        if self._write_cap_reached:
            return TerminalReason.WRITE_CAP_EXCEEDED
        if self._repeat_limit_reached:
            return TerminalReason.REPEATED_CALL
        if self._consecutive_errors >= self._max_consecutive_errors:
            return TerminalReason.CONSECUTIVE_ERRORS
        if self.elapsed() > self._wall_clock_seconds:
            return TerminalReason.WALL_CLOCK_EXCEEDED
        if usage.total_tokens > self._token_budget:
            return TerminalReason.TOKEN_BUDGET_EXCEEDED
        if iteration > self._max_iterations:
            return TerminalReason.MAX_ITERATIONS
        return None

    def elapsed(self) -> float:
        return self._clock() - self._started_at

    # --- state, recorded as it happens ---------------------------------------

    def record_outcome(self, outcome: ToolOutcome) -> None:
        """Strikes count consecutive failures; anything that worked resets them."""
        if outcome.is_error:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0

    def record_write_attempt(self) -> bool:
        """Count one attempted write. False means it must not be executed.

        Attempts, not successes. A write the domain rejects -- an ambiguous
        physician, a name that matches nobody -- has still spent one, because
        otherwise a model that cannot get the name right could retry without
        limit and the cap would bound nothing.

        The boundary is execution. A call rejected by schema validation never
        reached here and does not count: it costs an error strike instead, and a
        mistyped date should not consume write budget. Nor does an identical
        repeat, which returns its earlier result without running and so cannot
        create a record.
        """
        self._write_attempts += 1
        if self._write_attempts > self.max_writes:
            self._write_cap_reached = True
            return False
        return True

    def record_no_progress(self) -> None:
        self._no_progress = True

    def check_repeat(self, tool_name: str, arguments: Any) -> PriorCall | None:
        """Has this exact call been made before?

        Identity is the tool name plus its arguments with object keys sorted, so
        that argument order cannot disguise a repeat. Two calls to the same tool
        with genuinely different arguments are legitimate and hash differently.
        """
        key = _call_key(tool_name, arguments)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        if self._call_counts[key] >= self._max_identical_calls:
            self._repeat_limit_reached = True
        return self._calls.get(key)

    def record_call(
        self, tool_name: str, arguments: Any, tool_call_id: str, payload: str
    ) -> None:
        self._calls.setdefault(
            _call_key(tool_name, arguments), PriorCall(tool_call_id, payload)
        )

    def duplicate_of(self, payload: str, tool_call_id: str) -> str | None:
        """Has this exact result already been put in the context?

        Different arguments reaching the same records -- "Patel" and "Raj Patel"
        -- produce identical bytes, and sending them twice buys nothing. This
        deduplicates; it never terminates.
        """
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        prior = self._results.get(digest)
        if prior is None:
            self._results[digest] = tool_call_id
            return None
        return prior


def _call_key(tool_name: str, arguments: Any) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
