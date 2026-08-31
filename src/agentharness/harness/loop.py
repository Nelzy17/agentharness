"""The execution loop: context in, model call, tool dispatch, result back, repeat."""

import enum
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentharness.harness.context import (
    MAX_CONTEXT_TOKENS,
    ContextBudgetExceeded,
    ContextBuilder,
    load_prompt,
)
from agentharness.harness.dispatcher import dispatch
from agentharness.harness.model_client import HarnessFatalError, TokenUsage, ToolCall
from agentharness.harness.outcomes import (
    DuplicateResult,
    InvalidArguments,
    RepeatedCall,
    ToolFailed,
    ToolOutcome,
    ToolSucceeded,
)
from agentharness.harness.policy import (
    MAX_ITERATIONS,
    Policy,
    TerminalReason,
    TerminationPolicy,
)
from agentharness.harness.registry import ToolRegistry
from agentharness.harness.sanitizer import sanitize
from agentharness.harness.tracer import InMemoryTracer, Tracer
from agentharness.harness.validator import ValidatedCall, Validator, validate_sources
from agentharness.tools.definitions import Permission, SubmitFinalAnswerArgs

FINAL_ANSWER_TOOL = "submit_final_answer"

# A model can emit an unbounded argument string. The trace records what was
# requested, which does not require recording all of it.
MAX_TRACED_ARGUMENTS = 4000


class FinalAnswerRoute(enum.Enum):
    """How a run ended, when it ended with an answer.

    Both routes stay live. The model may ignore the control tool and simply stop
    calling tools, and that path must keep working -- but the difference is an
    answer with checkable sources versus one without, so M8 wants the rate.
    """

    SUBMIT_TOOL = "the model called submit_final_answer"
    FREE_TEXT = "the model stopped calling tools and answered in prose"
    NONE = "the run ended without an answer"


@dataclass(frozen=True)
class Resolution:
    """One tool call resolved: what the model is told, and what it meant."""

    outcome: ToolOutcome
    final_answer: SubmitFinalAnswerArgs | None
    arguments_json: str | None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    answer: str | None
    terminal_reason: TerminalReason
    iterations: int
    usage: TokenUsage
    messages: list[dict[str, Any]]
    route: FinalAnswerRoute = FinalAnswerRoute.NONE
    sources: list[str] = field(default_factory=list)
    insufficient_information: bool = False


class AgentLoop:
    """Runs one goal to a terminal state.

    `model_client` is anything with `complete(messages, tools) -> ModelResponse`
    and a cumulative `usage`. The loop never learns which implementation it has,
    which is what makes every path here testable without a network. The tracer
    is injected the same way, so the loop equally cannot tell whether it is
    writing to SQLite or to memory.
    """

    def __init__(
        self,
        model_client: Any,
        registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
        context_budget: int = MAX_CONTEXT_TOKENS,
        policy: Policy | None = None,
        policy_factory: Callable[[], TerminationPolicy] | None = None,
        tracer: Tracer | None = None,
        clock: Callable[[], float] = time.perf_counter,
        timestamp: Callable[[], float] = time.time,
    ) -> None:
        self._model_client = model_client
        self._registry = registry
        self._validator = Validator(registry)
        self._context_budget = context_budget
        self._policy = policy or Policy()
        self._policy_factory = policy_factory or (
            lambda: TerminationPolicy(max_iterations=max_iterations)
        )
        self._tracer = tracer or InMemoryTracer()
        # Two clocks, because they answer different questions and only one of
        # them can answer either.
        #
        # `clock` measures intervals. perf_counter, not monotonic: on Windows
        # time.monotonic is GetTickCount64 with a resolution of 15.625ms, so
        # 20,000 consecutive reads return one value and every tool call this
        # harness makes floors to 0ms. perf_counter is QueryPerformanceCounter
        # at 100ns and is the correct clock for measuring an interval anywhere.
        #
        # `timestamp` says when a run happened. perf_counter cannot: its epoch
        # is arbitrary and has no relation to real time, so storing it as
        # started_at would record a number that looks like a timestamp and is
        # not one.
        self._clock = clock
        self._timestamp = timestamp
        self._system_prompt = load_prompt("system")

    def run(self, goal: str, run_id: str | None = None) -> RunResult:
        """Run one goal to a terminal state.

        The run id is generated here, or supplied by a caller that needs it
        before the run finishes -- the API does, so that a fatally failed run can
        still be pointed at. It is threaded explicitly through the call chain
        rather than held on the instance: one AgentLoop serves many runs, and
        per-run state on a shared object is wrong the moment two overlap, in the
        way hardest to debug -- a write attributed to the wrong run.
        """
        run_id = run_id or uuid.uuid4().hex
        started_at = self._clock()
        model = getattr(self._model_client, "model", "unknown")
        self._tracer.start_run(run_id, goal, model, self._timestamp())
        try:
            return self._run(run_id, goal, started_at)
        except HarnessFatalError as error:
            # Recorded before it propagates: a run that failed fatally is
            # exactly the run someone wants to inspect.
            self._tracer.fail_run(
                run_id,
                f"{type(error).__name__}: {error}",
                self._elapsed_ms(started_at),
            )
            raise

    def _run(self, run_id: str, goal: str, started_at: float) -> RunResult:
        context = ContextBuilder(
            self._system_prompt,
            goal,
            self._registry.tool_definitions(),
            token_budget=self._context_budget,
        )
        termination = self._policy_factory()
        last_content: str | None = None
        iteration = 0

        while True:
            iteration += 1

            reason = termination.check(iteration, self._model_client.usage)
            if reason is not None:
                return self._result(
                    run_id, started_at, last_content, reason, iteration - 1, context
                )

            try:
                messages = context.messages_for_model_call()
            except ContextBudgetExceeded:
                # Stopping is the policy. Dropping older exchanges to fit would
                # forget something already retrieved, and would shift every byte
                # after the drop, destroying the cached prefix and re-billing
                # the remainder at full rate.
                return self._result(
                    run_id,
                    started_at,
                    last_content,
                    TerminalReason.CONTEXT_BUDGET_EXCEEDED,
                    iteration - 1,
                    context,
                )

            call_started = self._clock()
            response = self._model_client.complete(messages, context.tool_definitions())
            self._tracer.record_model_call(
                run_id=run_id,
                iteration=iteration,
                duration_ms=self._elapsed_ms(call_started),
                usage=response.usage,
                attempts=response.attempts,
            )
            context.add_assistant(response.message)
            last_content = response.message.content or last_content

            if not response.message.tool_calls:
                if response.message.content:
                    return self._result(
                        run_id,
                        started_at,
                        response.message.content,
                        TerminalReason.COMPLETED,
                        iteration,
                        context,
                        route=FinalAnswerRoute.FREE_TEXT,
                    )
                # Neither a tool call nor an answer. Recorded rather than
                # returned, so that every stop is decided in one place.
                termination.record_no_progress()
                continue

            # The ids this run has actually issued, which is what a cited
            # source is checked against.
            issued = context.answered_tool_call_ids()

            # Three phases, and the order is the invariant. Every tool call is
            # resolved to an outcome, then every outcome becomes a tool message,
            # and only then may the run end. Ending inside the resolution loop
            # would leave a tool_call.id unanswered in the final context, which
            # ContextBuilder refuses to build -- so the bug would surface here
            # rather than on a request that is never made.
            resolutions: list[Resolution] = []
            for call in response.message.tool_calls:
                tool_started = self._clock()
                resolution = self._resolve(call, termination, issued)
                self._trace_tool_call(
                    run_id, iteration, call, resolution, self._elapsed_ms(tool_started)
                )
                resolutions.append(resolution)

            for call, resolution in zip(
                response.message.tool_calls, resolutions, strict=True
            ):
                termination.record_outcome(resolution.outcome)
                context.add_tool_result(call.id, sanitize(resolution.outcome, call.id))

            final = next(
                (
                    resolution.final_answer
                    for resolution in resolutions
                    if resolution.final_answer
                ),
                None,
            )
            if final is not None:
                return self._result(
                    run_id,
                    started_at,
                    final.answer,
                    TerminalReason.COMPLETED,
                    iteration,
                    context,
                    route=FinalAnswerRoute.SUBMIT_TOOL,
                    sources=list(final.sources),
                    insufficient_information=final.insufficient_information,
                )

    def _resolve(
        self,
        call: ToolCall,
        termination: TerminationPolicy,
        issued_ids: frozenset[str],
    ) -> Resolution:
        """One tool call, from raw model output to something the model can read.

        Carries the validated answer back for the control tool as a return value
        rather than as a side effect, so the loop can emit every tool message
        before acting on it.
        """
        raw = (call.arguments or "")[:MAX_TRACED_ARGUMENTS]
        try:
            raw_args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            # Arguments arrive as a string, so they can be malformed before any
            # schema is consulted. Same class of failure as a schema rejection:
            # the model wrote the arguments, and the model can fix them.
            return Resolution(
                InvalidArguments(
                    name=call.name, message="the arguments were not valid JSON"
                ),
                None,
                raw,
            )

        validated = self._validator.validate(call.name, raw_args)
        if not isinstance(validated, ValidatedCall):
            return Resolution(validated, None, raw)

        # From here the arguments are typed, so the trace records the validated
        # form rather than whatever the model wrote.
        arguments_json = validated.args.model_dump_json()

        denial = self._policy.authorize(validated.spec)
        if denial is not None:
            return Resolution(denial, None, arguments_json)

        if call.name == FINAL_ANSWER_TOOL:
            # Contextual validation: the schema cannot know which ids this run
            # issued, so the check runs here, after the shape has passed and
            # before the answer is accepted.
            rejection = validate_sources(call.name, validated.args.sources, issued_ids)
            if rejection is not None:
                return Resolution(rejection, None, arguments_json)

        prior = termination.check_repeat(call.name, raw_args)
        if prior is not None:
            return Resolution(
                RepeatedCall(
                    name=call.name,
                    prior_tool_call_id=prior.tool_call_id,
                    prior_payload=prior.payload,
                ),
                None,
                arguments_json,
            )

        outcome = dispatch(validated)
        termination.record_call(call.name, raw_args, call.id, outcome.payload())

        final = validated.args if call.name == FINAL_ANSWER_TOOL else None
        if final is None and isinstance(outcome, ToolSucceeded):
            duplicate = termination.duplicate_of(outcome.payload(), call.id)
            if duplicate is not None:
                return Resolution(
                    DuplicateResult(name=call.name, duplicate_of=duplicate),
                    None,
                    arguments_json,
                )

        return Resolution(outcome, final, arguments_json)

    def _trace_tool_call(
        self,
        run_id: str,
        iteration: int,
        call: ToolCall,
        resolution: Resolution,
        duration_ms: float,
    ) -> None:
        spec = self._registry.get(call.name)
        outcome = resolution.outcome
        self._tracer.record_tool_call(
            run_id=run_id,
            iteration=iteration,
            duration_ms=duration_ms,
            tool_call_id=call.id,
            tool_name=call.name,
            arguments_json=resolution.arguments_json,
            outcome_class=type(outcome).__name__,
            is_write=spec is not None and spec.permission is Permission.WRITE,
            # The full traceback, which the model's sanitized message never
            # contains. This is the other half of that split.
            error_detail=outcome.detail if isinstance(outcome, ToolFailed) else None,
        )

    def _elapsed_ms(self, since: float) -> float:
        return (self._clock() - since) * 1000

    def _result(
        self,
        run_id: str,
        started_at: float,
        answer: str | None,
        reason: TerminalReason,
        iterations: int,
        context: ContextBuilder,
        route: FinalAnswerRoute = FinalAnswerRoute.NONE,
        sources: list[str] | None = None,
        insufficient_information: bool = False,
    ) -> RunResult:
        usage = self._model_client.usage
        self._tracer.finish_run(
            run_id,
            terminal_reason=reason.name,
            reason_text=reason.value,
            route=route.name,
            answer=answer,
            sources=json.dumps(sources or []),
            insufficient_information=int(insufficient_information),
            iterations=iterations,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            duration_ms=self._elapsed_ms(started_at),
            ended_at=self._timestamp(),
        )
        return RunResult(
            run_id=run_id,
            answer=answer,
            terminal_reason=reason,
            iterations=iterations,
            usage=usage,
            messages=context.messages(),
            route=route,
            sources=sources or [],
            insufficient_information=insufficient_information,
        )
