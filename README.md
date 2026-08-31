# AgentHarness

A hand-written agent runtime: the harness that sits between an LLM and a set of
tools. No agent framework, at any point. Life Sciences is only a source of
realistic synthetic tools and data.

See `CLAUDE.md` for the rules this repo is built under, `DESIGN.md` for the
architecture, and `DECISIONS.md` for the tradeoffs taken along the way.

## Current state: M0 — domain layer

Five plain Python tool functions over synthetic fixtures. No model client, no
loop, no API yet.

```
src/agentharness/domain/   pydantic types + the single data access path
src/agentharness/tools/    the five tool functions
data/                      checked-in synthetic fixtures
```

## The API

```
uvicorn agentharness.api.app:create_app --factory
```

`POST /runs` takes a goal and blocks until the run terminates. `GET /runs/{id}`
returns the run and every step it took.

On Windows, use PowerShell rather than curl -- curl.exe under PowerShell needs
the JSON body escaped in a way that is easy to get wrong, and getting it wrong
hangs waiting on stdin instead of failing:

```powershell
$body = @{ goal = "Prepare me for tomorrow's meeting with Dr. Evelyn Chen about Nexovar." } | ConvertTo-Json
$run = Invoke-RestMethod -Uri http://localhost:8000/runs -Method Post -Body $body -ContentType application/json
python -m agentharness.cli.trace $run.run_id
```

Synchronous on purpose: the termination policy bounds a run to 60 seconds and
eight iterations, so a request thread is an acceptable place to spend one. The
limitation is real -- a slow model ties up a worker, and a client that
disconnects gets nothing even though the run completed and was recorded. What
would replace it is a job queue: `POST /runs` returns 202 with an id
immediately, the run executes on a worker, and the caller polls `GET /runs/{id}`
or receives a webhook. That is a deployment concern rather than a harness one,
so it is noted rather than built.

Two endpoints and no more. `AGENTHARNESS_MODEL` and `OPENAI_API_KEY` come from
the environment.

## Reading a trace

```
python -m agentharness.cli.trace <run_id>
```

Every model call and tool call, with durations, token counters including cache
reads and writes, the outcome class of each call, and the human-readable reason
the run stopped.

## Autonomous writes

Writes execute without a human confirming them, because autonomous action is
the point of the project. That removes the usual answer to "what stops it
writing nonsense into a physician's record", so four mechanisms carry it
instead:

- **one write tool**, `create_followup`, and no other
- **append-only**: `Followup` is a frozen model, so a stored record cannot be
  modified, and the repository has exactly one removal
- **a cap of two writes per run**, counting attempts rather than successes, so a
  write the domain rejects is not free; exceeding it ends the run with its own
  terminal reason
- **complete attribution**: every record carries the `run_id` and
  `tool_call_id` that produced it, supplied by the dispatcher and absent from
  the schema the model sees, so it cannot be forged

Reversal is `repository.revoke_run(run_id)`, which removes exactly that run's
records and returns their ids. It is a function rather than a CLI on purpose:
domain data is held in memory, so a separate process would load the fixtures,
match nothing and report success while deleting nothing -- and a reversal path
that appears to work and does not is the worst available failure here, given
that reversibility is what replaced human approval.

### At production scale

Follow-ups would be rows rather than an in-memory list, and reversal would be
`DELETE FROM followups WHERE created_by_run_id = ?`. The shape of the answer is
identical -- scoped by run, unable to reach records that carry no attribution --
and only the storage differs. That substitution is a change to
`domain/repository.py` and to nothing else, which is the whole reason every
tool goes through it.

## Running the tests

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```
