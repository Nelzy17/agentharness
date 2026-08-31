"""Observability: what happened, not what the model was thinking.

Architecture principle 8. A trace records observable actions -- the request
made, the arguments validated, the class of outcome, how long it took, what it
cost, and why the run stopped. There is no field for reasoning, and no attempt
anywhere to elicit or reconstruct it. That is a position, not an oversight: a
recorded rationale is a story the model tells about itself after the fact, it
cannot be verified against anything, and a trace that mixes it with observed
facts makes the facts less trustworthy rather than the rationale more so.

The tracer is injected into the loop the same way the model client is, so the
loop cannot tell whether it is writing to SQLite or to memory.
"""

import abc
from dataclasses import dataclass, field
from typing import Any

from agentharness.harness.model_client import TokenUsage
from agentharness.store.runs import RunStore


class Tracer(abc.ABC):
    @abc.abstractmethod
    def start_run(self, run_id: str, goal: str, model: str, started_at: float) -> None: ...

    @abc.abstractmethod
    def record_model_call(
        self,
        run_id: str,
        iteration: int,
        duration_ms: float,
        usage: TokenUsage,
        attempts: int,
    ) -> None: ...

    @abc.abstractmethod
    def record_tool_call(
        self,
        run_id: str,
        iteration: int,
        duration_ms: float,
        tool_call_id: str,
        tool_name: str,
        arguments_json: str | None,
        outcome_class: str,
        is_write: bool,
        error_detail: str | None,
    ) -> None: ...

    @abc.abstractmethod
    def finish_run(self, run_id: str, **summary: Any) -> None: ...

    @abc.abstractmethod
    def fail_run(self, run_id: str, error: str, duration_ms: float) -> None: ...


@dataclass
class InMemoryTracer(Tracer):
    """The default. Keeps everything a SqliteTracer would write, in lists."""

    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def start_run(self, run_id: str, goal: str, model: str, started_at: float) -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "goal": goal,
            "model": model,
            "started_at": started_at,
        }

    def record_model_call(
        self, run_id, iteration, duration_ms, usage: TokenUsage, attempts
    ) -> None:
        self.steps.append(
            {
                "run_id": run_id,
                "sequence": len(self.steps) + 1,
                "iteration": iteration,
                "kind": "model_call",
                "duration_ms": duration_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": usage.cached_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "attempts": attempts,
            }
        )

    def record_tool_call(
        self,
        run_id,
        iteration,
        duration_ms,
        tool_call_id,
        tool_name,
        arguments_json,
        outcome_class,
        is_write,
        error_detail,
    ) -> None:
        self.steps.append(
            {
                "run_id": run_id,
                "sequence": len(self.steps) + 1,
                "iteration": iteration,
                "kind": "tool_call",
                "duration_ms": duration_ms,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments_json": arguments_json,
                "outcome_class": outcome_class,
                "is_write": int(is_write),
                "error_detail": error_detail,
            }
        )

    def finish_run(self, run_id: str, **summary: Any) -> None:
        self.runs[run_id].update(summary)

    def fail_run(self, run_id: str, error: str, duration_ms: float) -> None:
        self.runs[run_id].update(
            {
                "terminal_reason": "HARNESS_ERROR",
                "error": error,
                "duration_ms": duration_ms,
            }
        )

    def steps_for(self, run_id: str) -> list[dict[str, Any]]:
        return [step for step in self.steps if step["run_id"] == run_id]


class SqliteTracer(Tracer):
    def __init__(self, store: RunStore) -> None:
        self._store = store

    def start_run(self, run_id: str, goal: str, model: str, started_at: float) -> None:
        self._store.start_run(run_id, goal, model, started_at)

    def record_model_call(
        self, run_id, iteration, duration_ms, usage: TokenUsage, attempts
    ) -> None:
        self._store.add_step(
            run_id=run_id,
            sequence=self._store.next_sequence(run_id),
            iteration=iteration,
            kind="model_call",
            duration_ms=duration_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            attempts=attempts,
        )

    def record_tool_call(
        self,
        run_id,
        iteration,
        duration_ms,
        tool_call_id,
        tool_name,
        arguments_json,
        outcome_class,
        is_write,
        error_detail,
    ) -> None:
        self._store.add_step(
            run_id=run_id,
            sequence=self._store.next_sequence(run_id),
            iteration=iteration,
            kind="tool_call",
            duration_ms=duration_ms,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            outcome_class=outcome_class,
            is_write=int(is_write),
            error_detail=error_detail,
        )

    def finish_run(self, run_id: str, **summary: Any) -> None:
        self._store.finish_run(run_id, **summary)

    def fail_run(self, run_id: str, error: str, duration_ms: float) -> None:
        self._store.finish_run(
            run_id,
            terminal_reason="HARNESS_ERROR",
            reason_text="the harness could not continue and the run was abandoned",
            error=error,
            duration_ms=duration_ms,
        )
