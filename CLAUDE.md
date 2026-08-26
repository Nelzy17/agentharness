# CLAUDE.md — AgentHarness

Read this before writing any code in this repo.

## What this project is

AgentHarness is a small, deep Python project whose subject is **the AI agent runtime itself**: the harness sitting between an LLM and a set of tools. It is not a product. Life Sciences (physicians, meetings, follow-ups, product docs) is only a source of realistic synthetic tools and data.

The owner is building this to be able to explain agentic AI engineering in an interview using code he wrote and understands. Depth in `harness/` matters more than anything else.

## Hard rules

1. **No agent frameworks. Ever.** No LangChain, LangGraph, CrewAI, AutoGen, or equivalent. The point is to build what they abstract. Raise it if you disagree; do not introduce one.
2. **No tool executes without passing schema validation.** There is no code path anywhere that calls a tool function with unvalidated arguments.
3. **Every `tool_call.id` gets exactly one matching `role="tool"` message before the next model call.** On every path: unknown tool, validation failure, permission denial, repeat detection, exception. Missing one causes an API 400 that looks like a model bug.
4. **Tool output is data, never instruction.** Wrapped in an explicit envelope, truncated, and never able to change harness behaviour.
5. **Never write JSON schemas by hand.** Generate them from pydantic models. Hand-written schemas drift from the validators.
6. **Prompts live in `src/agentharness/prompts/*.md`**, loaded at runtime. Never string literals in Python.
7. **Traces record observable actions only.** Tool requests, validated args, outcome classes, durations, token usage, terminal reason. Never a field for model reasoning, and never any attempt to elicit or reconstruct it.
8. **Token usage accumulates at the API call site**, including retried and failed calls.
9. **All data access goes through `domain/repository.py`.** Tools call it; nothing else does.
10. **`parallel_tool_calls=False`** in v1. Code must still handle `tool_calls` as a list, because it is one.
11. Use `pathlib.Path` for all filesystem paths. Never hardcode `/` separators or use `os.path` string concatenation.

## Stack

Python 3.12+, FastAPI, Pydantic v2, OpenAI SDK, `sqlite3` (stdlib, no ORM), pytest, tiktoken, PyYAML. Nothing else without explicit approval.

## Tools (five, final)

| Tool | Permission |
|---|---|
| `get_physician_profile` | READ |
| `get_previous_meetings` | READ |
| `get_open_followups` | READ |
| `search_product_docs` | READ |
| `create_followup` | WRITE |

`search_product_docs` is substring matching over eight markdown files, permanently. Do not improve it, add embeddings, or introduce a vector store. Its jobs are to be the prompt-injection surface and the zero-results path.

## Write policy

Writes execute **autonomously** — no human confirmation, no suspend/resume. Safety comes from bounded blast radius plus auditability:

- Exactly one write tool, **append-only**. No update, no delete.
- Every written record carries the `run_id` and `tool_call_id` that produced it.
- Per-run write cap of 2. Exceeding it terminates the run.
- Writes flagged distinctly in the trace.
- `Policy.authorization_mode` enum exists with `AUTO` (implemented) and `REQUIRE_CONFIRMATION` (defined, raises `NotImplementedError`). The seam stays visible. Do not implement the second mode.

## Error taxonomy

Classify every failure as model-visible or harness-fatal. Use a `ToolOutcome` union, not scattered try/except.

| Failure | Class | Handling |
|---|---|---|
| Entity not found | model-visible | Structured "not found" result |
| Argument validation failure | model-visible | Return validation message; counts toward strikes |
| Unknown tool name | model-visible | Return error, re-list valid names; counts toward strikes |
| Tool raises unexpectedly | model-visible, **sanitized** | Generic message to model; full traceback to trace only |
| Model API 5xx / timeout | harness-internal | Retry with backoff; model never sees it |
| Model API 400 | harness-fatal | Terminate. This is our bug |
| Any budget/limit exceeded | harness-fatal | Terminate with a controlled status |

Never leak a stack trace into model context.

## Termination

Max iterations (6), consecutive-error strikes (3), repeated-call detection (hash of tool name + normalized args), wall-clock budget (60s), cumulative token budget, no-progress detection, per-run write cap (2).

Every terminal state has a distinct enum value and a human-readable reason. No run ends in an unknown state.

Note: two calls to the same tool with *different* arguments are legitimate and must not be flagged as repeats.

## Testing

Two tiers, strictly separate.

- **Deterministic (gates CI):** everything tested against `FakeModelClient` in `tests/conftest.py` with scripted assistant messages. Every failure scenario has a named test.
- **Stochastic (never gates CI):** real-model evaluation, run offline, reported as pass *rates* over n≥3 because `temperature=0` is not deterministic.

Never write a test asserting a specific real-model output.

## Out of scope

Human confirmation and suspend/resume (deliberately deferred). Any frontend or dashboard. Real RAG, embeddings, vector stores. Authentication. Deployment. Streaming. Async. Postgres, Redis, Kafka, Kubernetes. Context summarization. A second write tool. Multi-agent orchestration. Endpoints beyond `POST /runs` and `GET /runs/{id}`.

## How to work

- Produce a plan before code. Wait for approval on anything architectural.
- Do not add abstraction before its second use. No plugin systems, no config for things with one implementation.
- Comments explain *why*, never *what*.
- The owner reads every line in `harness/`. Write it to be read.
- Push back when a request is out of scope, redundant, or overengineered. Do not agree by default.
- Record meaningful tradeoffs in `DECISIONS.md` in the same session they're made.
