"""Two endpoints. POST a goal, GET the trace of what happened.

Synchronous by design: runs are bounded to 60 seconds by the termination policy,
so a request thread is an acceptable place to spend one. The limitation and what
would replace it -- a job queue with polling or a webhook -- are noted in the
README rather than built.
"""

import json
import os
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from agentharness.api.schemas import RunDetail, RunRequest, RunResponse, Usage
from agentharness.harness.loop import AgentLoop
from agentharness.harness.model_client import HarnessFatalError, ModelClient
from agentharness.harness.registry import build_registry
from agentharness.harness.tracer import SqliteTracer, Tracer
from agentharness.store.runs import RunStore

DEFAULT_DB_PATH = "./agentharness.db"


def create_app(
    build_loop: Callable[[Tracer], AgentLoop] | None = None,
    store: RunStore | None = None,
) -> FastAPI:
    """Build the app around an injected loop builder and store.

    Tests pass a builder that closes over FakeModelClient; nothing about the
    endpoints changes between that and a real model.
    """
    store = store or RunStore(os.environ.get("AGENTHARNESS_DB_PATH", DEFAULT_DB_PATH))
    build_loop = build_loop or _default_loop_builder
    app = FastAPI(title="AgentHarness", version="0.1.0")

    @app.post("/runs", response_model=RunResponse)
    def create_run(request: RunRequest) -> RunResponse:
        # Generated here rather than inside the run, so a run that fails
        # fatally can still be pointed at.
        run_id = uuid.uuid4().hex
        loop = build_loop(SqliteTracer(store))
        try:
            result = loop.run(request.goal, run_id=run_id)
        except HarnessFatalError as error:
            # The run is already recorded, including the detail. The caller gets
            # an id and a pointer, not our internals.
            raise HTTPException(
                status_code=500,
                detail={
                    "run_id": run_id,
                    "message": (
                        "the run failed and was recorded; retrieve it at "
                        f"/runs/{run_id} to see why"
                    ),
                },
            ) from error

        return RunResponse(
            run_id=result.run_id,
            terminal_reason=result.terminal_reason.name,
            reason_text=result.terminal_reason.value,
            answer=result.answer,
            sources=result.sources,
            insufficient_information=result.insufficient_information,
            route=result.route.name,
            iterations=result.iterations,
            usage=Usage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cached_tokens=result.usage.cached_tokens,
                cache_write_tokens=result.usage.cache_write_tokens,
                total_tokens=result.usage.total_tokens,
            ),
        )

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
        run = dict(run)
        run["sources"] = json.loads(run["sources"] or "[]")
        return RunDetail(run=run, steps=store.get_steps(run_id))

    return app


def _default_loop_builder(tracer: Tracer) -> AgentLoop:
    """The real thing: an OpenAI client against the configured model.

    The model comes from the environment here, where the CLI requires it as an
    argument. That is not an inconsistency: for the CLI the model is the
    variable of an experiment and must be stated per run, while for a server it
    is deployment configuration, fixed for the process and recorded on every
    trace row. Same string, different kind of thing.
    """
    from openai import OpenAI

    model = os.environ.get("AGENTHARNESS_MODEL")
    if not model:
        raise RuntimeError("AGENTHARNESS_MODEL is not set in the environment")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return AgentLoop(ModelClient(OpenAI(), model=model), build_registry(), tracer=tracer)


# No module-level app: building one at import would open a database and read the
# environment as a side effect of importing anything here, including from tests.
# Serve it as a factory instead:
#
#     uvicorn agentharness.api.app:create_app --factory
