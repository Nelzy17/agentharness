# INTERVIEW.md

Not linked from the README, on purpose. This is a preparation file, and a reader
who found it first would see the project through the frame of what it is *for*
rather than what it *is*.

Each question gets a file, a test, and the shortest true answer. Where the
codebase answers badly, that is written down as the answer — the point of the
list is to find those, not to write around them.

---

## The seven areas named

### Agentic AI

`harness/loop.py` (427 lines). The loop is: check termination, assemble context,
call the model, resolve every tool call, emit every tool message, then decide
whether to stop. Three phases per turn, in that order, because ending inside the
resolution loop leaves a `tool_call.id` unanswered and the next request 400s.

Show `test_every_tool_call_is_answered_exactly_once_whatever_the_outcome` —
one assistant message with three calls, one unknown tool, one invalid, one
success, asserting three tool messages with matching ids in order.

The point to make: everyone builds a loop. Almost nobody can say how theirs
stops. Eight conditions, one evaluation point, `TerminalReason` enumerated by a
test so an unreachable state fails the suite.

### Prompt Engineering

`src/agentharness/prompts/system.md`, loaded at runtime, never a string literal.
Tool and field descriptions in `tools/definitions.py` — a deliberate exception,
because a description that drifts from the type it describes is worse than none.

The honest answer is the finding, not the prompt: **prompts express intent,
mechanisms enforce it.** Four data points, two against prompting. `due_date`
worked from a description because the model can check its own draft against a
regex. `sources` did not — two edits changed only the format of the fabrication,
and what fixed it was showing the model a `tool_call_id` it had never seen.
`insufficient_information` is the boundary: the definition settled, the
application did not.

And the retired hypothesis: the envelope paragraph *looked* like it improved
grounding on one run, and a proper test at n=9 found the difference smaller than
the spread and pointing the other way. `eval/envelope.py`.

### Context Engineering

`harness/context.py` is the only file that builds a message. `harness/sanitizer.py`
caps results at 2,000 characters and switches key to `result_partial` so a
fragment cannot pass as a record.

The measured part: **90% of prompt tokens were cache reads** (171,515 of
189,927). The cached prefix is the system prompt and tool definitions, so
truncation must edit the tail — shrinking an early tool result invalidates every
cached byte after it and can cost more than it saves. And cost is quadratic in
the iteration cap: 6 → 8 iterations is 33% more turns for 55% more tokens.

Token counting is calibrated against the API rather than assumed: 1,697 estimated
against 1,651 measured. `test_the_estimate_lands_within_five_percent_of_the_reported_count`
pins it with digests of the prompt and tool definitions, so an edit fails by name.

### AI Harness

The whole repository, but the answer to give is the trust boundary:
`harness/validator.py`. Model output is untrusted input until it passes a schema
*generated from the same pydantic model that enforces it* —
`test_generated_schema_declares_exactly_the_models_fields` is the anti-drift
assertion, and the reason hand-written schemas are banned.

Two kinds of validation, which is the distinction worth drawing: *structural*
(decidable from the value) and *contextual* (decidable only from run state).
Cited `tool_call_id`s are the second kind and cannot live in a schema without
making the schema a liar about what it enforces.

### REST APIs

`api/app.py`, two endpoints, 114 lines. `POST /runs` blocks; `GET /runs/{id}`
returns the run and every step.

**Say plainly that this is the least interesting part of the project.** It is
table stakes and it is deliberately small. The interesting decisions around it
are that a fatally failed run is still recorded and retrievable — that is the run
someone wants to inspect — and that the 500 body carries a run id and a pointer
rather than internals.

### Data Piping

**This is the weakest area and should be conceded early.** JSON fixtures load
into memory once at import in `domain/repository.py`. There is no pipeline, no
streaming, no batch, no schema evolution, no backfill.

What can honestly be claimed: a single data access path, so swapping storage is
a one-file change; and that the eval's re-scoring path
(`eval/rescore.py`) is a genuine data-reprocessing story — 39 runs re-scored from
persisted traces after a scorer bug, with no new API calls, and a test asserting
scoring a live run and a rebuilt one agree on every metric.

If they want ETL experience, this project is not evidence of it.

### Object Querying

**Also weak, and worth saying so.** `domain/repository.py` is list comprehensions
over in-memory lists. `store/runs.py` is hand-written SQL over two tables with no
ORM, which was chosen so nothing hides the query — but two tables and six
statements is not a querying story.

The defensible piece is the resolution semantics: names resolve to exactly one
record, none, or several, and the ambiguous case returns candidates rather than
guessing. `test_hyphenated_last_name_makes_chen_ambiguous` — "Chen" matches both
Evelyn Chen and Daniel Chen-Ruiz, and the harness says so instead of picking one.

---

## The learning list

### The agent execution loop
See Agentic AI above. `harness/loop.py`, `tests/unit/test_loop.py`.

### Tool/function calling end to end
`tools/definitions.py` generates strict-mode schemas from pydantic models;
`harness/registry.py` emits the payload; `harness/dispatcher.py` is the only
place a tool function is invoked. Verified against the API docs rather than
assumed: strict mode requires every property in `required`, forbids `default`,
and expresses optionals as nullable unions — four transformations, all in one
function, `test_generated_schema_conforms_to_strict_mode`.

### Argument validation as a trust boundary
`harness/validator.py`. Also the sharpest security detail in the project: the
validation message never echoes the rejected input, because an argument the model
was steered into producing should not be read back to it.
`test_the_rejected_input_is_not_read_back_to_the_model`.

### Context construction, ordering, budgeting
See Context Engineering. Add that overflow *stops the run* rather than dropping
messages: an agent that quietly forgets what it retrieved is worse than one that
stops, and dropping an early exchange would destroy the cached prefix anyway.

### Termination policy
`harness/policy.py`. Eight conditions, one ordered list, most-diagnostic first
with the iteration cap last as the catch-all.
`test_every_terminal_reason_is_reachable`.

The repeat-detection design is worth the airtime: argument identity answers "is
the model stuck" and terminates; result identity answers "is this worth context"
and deduplicates without terminating. Two questions, two mechanisms.

### Error taxonomy and failure recovery
`harness/outcomes.py`. Model-visible or harness-fatal declared on the type.
Recovery is measurable: `tests/unit/test_failures.py` holds an executable index
mapping every failure scenario to the tests covering it, AST-scanned so a renamed
test fails the index rather than silently uncovering a scenario.

### Read/write permission and authorization of model-proposed actions
`harness/policy.py`, `tests/unit/test_write_policy.py`. Four mechanisms, because
there is no human in the loop: one write tool, append-only enforced by a frozen
model, a cap counting attempts, complete attribution the model cannot forge.

**Concede the deliberate gap:** `REQUIRE_CONFIRMATION` raises
`NotImplementedError`. Be able to say what implementing it costs — run-state
persistence, a status enum, a resume endpoint — because that is the answer that
shows you understand why it was deferred rather than missed.

### Prompt injection through tool results
`data/docs/nexovar-field-notes-dosing-questions.md` carries the payload;
`eval/adversarial.py` measures whether the model acts on it. 5 cases, n=3, **no
write tool requested in any run**.

The two details that make this more than a demo: a run where the payload never
reached the context is reported *inconclusive*, not passed; and exposure is
nondeterministic — the same goal retrieved the document on 2 of 3 runs — which is
the argument for structural defence over detection.

### Trace-based observability
`harness/tracer.py`, `store/schema.sql`, `cli/trace.py`. The position to state:
the trace records observable actions and has no column for reasoning, because a
recorded rationale cannot be checked against anything and mixing it with observed
facts makes the facts less trustworthy. `test_no_column_can_hold_reasoning_or_a_transcript`
asserts the exact column set.

### Agent evaluation with real metrics
`eval/`. 13 cases, n=3, two tiers, rates with variance and applicability.

The finding to lead with: **constraints substitute for capability on everything
they can express.** A 30× more expensive model was level on everything a schema
states, behind on iterations and latency, and ahead only on judgement under
ambiguity.

---

## Questions this codebase answers badly

Worth rehearsing, because being caught without these is worse than conceding them.

**"How do you know the agent's answers are correct?"** I do not. Grounding
measures citation discipline — did it cite something real — not whether the
answer follows from what it cited. `nakamura_unknown` scores task completion 1.00
with outcome `answered`, and an invented profile would score identically. Closing
it needs the domain result status in the trace.

**"What happens at scale / under concurrency?"** Untested. Runs are synchronous,
domain data is a module-level list, and nothing has ever run two concurrently.
`run_id` is threaded rather than held on the loop specifically so concurrent runs
would not corrupt attribution — but that is a design precaution, not a
demonstration.

**"How does it handle a tool that is slow or flaky?"** Retries exist for the
*model* API only. Tool calls have no timeout and no retry; a hanging tool hangs
the run until the wall-clock budget fires, and that path has never been exercised
against a genuinely slow tool.

**"Show me the deployment."** There isn't one. No container, no CI, no
environments. Deliberately out of scope, and RepPilot is the project that
demonstrates deployment.

**"n=3?"** Yes. Enough to catch a broken metric, not enough to separate a 0.90
from a 0.95. Every rate in the README is what happened three times.

**"Have you used a framework in anger?"** No, and this project is not a substitute
for having done so. What it does support is a specific argument about *when* one
earns its place — durable execution, multi-agent handoff, large tool surfaces,
provider portability — and why the tradeoff is only visible after building the
layer by hand.
