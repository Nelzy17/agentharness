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

## Running the tests

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```
