# DECISIONS

Tradeoffs recorded in the session they were made. Settled here means settled;
do not re-litigate.

---

## M0 — domain layer

**Data directory resolved from `__file__`, with no override.**
`repository.py` computes `Path(__file__).resolve().parents[3] / "data"` and that
is the whole mechanism. An `AGENTHARNESS_DATA_DIR` env var was considered and
cut: there is exactly one data directory, and configuration for a thing with one
implementation is ceremony. Consequence: the package cannot be installed as a
wheel and still find its fixtures. It is never installed as a wheel.

**Dates are `str` on the domain entities.**
The entities describe fixture records, and the fixtures are already ISO strings
that sort correctly. Parsing them to `datetime.date` would buy validation in the
wrong place — the date a *model* supplies to `create_followup` is an argument,
and arguments are validated by the per-tool pydantic models in M1. Keeping the
entities as `str` also means a domain object never raises on construction, which
is what lets the M0 tools honour "expected conditions are data, not exceptions".

**Tool results carry a `status`, and `"empty"` is one of its values.**
Three of the five tools can legitimately return nothing while having resolved
their subject perfectly well: Alvarez has no meetings, Zelmarin has no docs. An
empty list is ambiguous evidence — it could mean "none exist" or "lookup went
wrong" — so the status says which. `"empty"` is a success, not a failure. The
other three values are the resolution outcomes: `ok | not_found | ambiguous`.
This is a different axis from the `ToolOutcome` union coming in M1, which
classifies *harness* failures; these classify *domain* results.

**Name resolution is token-subset matching, not substring matching.**
A query is normalised (lowercased, honorifics dropped, punctuation stripped) and
split into tokens; a record matches when every query token appears in its name's
token set, with hyphenated parts contributing both the whole and the components.
`"Chen"` therefore matches Evelyn Chen *and* Daniel Chen-Ruiz and is reported as
ambiguous, which is the point of having Chen-Ruiz in the fixtures. Naive
substring matching gets the same answer here by accident; exact last-name
matching gets it wrong by silently resolving to Evelyn.

**A query that normalises to zero tokens matches everyone.**
`"Dr."` reduces to an empty token set, and the empty set is a subset of every
name, so all six physicians match and the result is `ambiguous` listing all six.
This is deliberate rather than incidental: the alternative — special-casing it to
`not_found` — would be a worse answer to a request that named nobody, and the
ambiguity is exactly what the M8 "prepare me for my meeting" case wants. There is
a test asserting the six, so the behaviour cannot drift silently.

**`search_product_docs` ignores query terms shorter than four characters.**
Without it a natural-language query scores "what", "does" and "the" across every
document and ranking degenerates into document length. One `len(term) >= 4`
filter fixes it with no stopword list. Scoring stays per-term substring counting
with a x3 weight on title and product. No stemming, no IDF, no fuzzy matching,
ever — this tool is the injection surface and the zero-results path, not a
retrieval engine.

**The snippet window is +/-200 characters, and the injection test is what pins it.**
The window has to be wide enough that a realistic dosing query surfaces the
injection payload in the text the model actually sees. Asserting only that the
payload exists somewhere in the fixture file would let M7's injection test pass
while testing nothing. `test_injection_payload_is_inside_the_returned_snippet`
is therefore treated as an invariant on the fixture *and* the window: the payload
is positioned near the top of the document body so the assertion holds, and if
either the radius or the fixture moves, that test fails first.

**`tools/_resolution.py` is a small addition to the file list in DESIGN Part 4.**
Four tools need the same two sentences of wording for not-found and ambiguous.
Shared strings only — each tool still branches on all three outcomes itself,
because each returns a different result type. This is deliberately not a shared
"handle resolution failure" helper; that would hide the branch the reader most
needs to see.

**Dependencies are added at the milestone that first imports them.**
`pydantic` and `pytest` now; `openai` and `tiktoken` at M2/M3, `fastapi` at M5,
`pyyaml` at M8. Document front matter is hand-parsed (about eight lines) rather
than pulling PyYAML into the domain layer early.

**The repository is append-only and generates ids from the existing maximum.**
No update, no delete, anywhere in `repository.py`. `append_followup` derives the
next id from the highest existing suffix rather than from list length, so a
fixture edit cannot cause a silent id collision. `reset()` exists for test
isolation, not for production use.

---

## M1 — registry, schemas, and the validation boundary

**Chat Completions, not the Responses API.**
The Responses API manages conversation state server-side. Managing that state is
the thing this project exists to implement by hand: the message array, the
ordering, the one tool message per tool call, the truncation. Choosing the API
that makes the message array my problem is the point of the exercise, not an
inconvenience of it. The practical consequence is the tool definition shape --
Chat Completions nests name, description, parameters and strict under a
"function" key, where Responses flattens them -- and that nesting is written
plainly in `to_tool_definition` rather than abstracted over.

**Strict tool calling, with one transformation function.**
Verified 2026-08-26 against the OpenAI function-calling and structured-outputs
guides, with pydantic 2.13.4. Strict mode requires every property to appear in
`required`, requires `additionalProperties: false` on every object, forbids
`default`, and expresses optional parameters as nullable unions rather than by
omission. Pydantic's `model_json_schema()` differs from that in four ways: it
emits `default`, it lists only fields without defaults in `required`, it adds
`title`, and it cannot express "optional" the way strict mode wants. All four
are handled in `to_openai_schema` and nowhere else. `pattern` and the numeric
bounds are supported keywords, so `due_date`'s regex reaches the model as part
of the contract rather than only as a validator.

Sources: https://developers.openai.com/api/docs/guides/function-calling and
https://developers.openai.com/api/docs/guides/structured-outputs.

**`extra="forbid"` satisfies the trust boundary and strict mode at once.**
The config is there because a model emitting an undeclared field is either
confused or being steered by injected content, and silently dropping it hides
both. It also happens to make pydantic emit `additionalProperties: false`, which
is exactly what strict mode requires. One config line, two jobs: the schema the
model is sent and the validator that enforces it agree because they come from
the same declaration.

**Tool and field descriptions stay in Python, against CLAUDE.md rule 6.**
They are schema metadata. A description that has drifted from the type it
describes is worse than no description, and separating the two files makes drift
the default. System prompts are standalone artifacts with no such coupling and
stay in `prompts/`. The exception is narrow: it covers descriptions attached to
a field or a tool, nothing else.

**`limit` is not exposed to the model.**
The args model and the tool function signature are allowed to differ, and here
they do: `get_previous_meetings` keeps its `limit=5` default in Python and
declares only `physician_name` in its schema. How much history belongs in the
context window is the harness's decision. Offering it to the model would have
forced a choice between a required field it must always send, a nullable field
needing coercion, or dropping strict mode -- three solutions to a problem that
does not need to exist over fixtures holding at most three meetings. A test
asserts the divergence so it reads as deliberate.

**`ValidatedCall` is not a `ToolOutcome`.**
`validate()` returns `ValidatedCall | ToolOutcome`. That way `ToolOutcome` means
exactly one thing -- a resolved call that owes the model a message -- and every
member can render one. The alternative, keeping `ValidatedCall` in the union,
forces a `to_tool_message()` that raises, which is a union member admitting it
does not belong. M2's execution results join the union; a validated call never
does.

**The validation message is a budget, not a log line.**
It becomes a tool message and is the model's only chance to correct itself, so
it is one line per failed field, capped, with the documentation URL and the
input echo suppressed at source via `errors(include_url=False,
include_input=False)`. Not echoing the input is a security choice as much as a
budget one: an argument the model was steered into producing should not be read
back to it. Tests assert under 300 characters and no URL, on the failure path
where iterations are already being spent.

**No `$ref` inliner.**
The args models are flat, so pydantic emits no `$defs`. Rather than write an
inliner for a case that cannot currently arise, `to_openai_schema` raises when
it sees one, and a test hands it a nested model to prove the guard fires. An
unasserted guard is a guard discovered when it fails to fire.

---

## M2 — the execution loop

**The SDK's types stop at `model_client.py`.**
The loop, the context builder and the tests work with `AssistantMessage`,
`ToolCall`, `TokenUsage` and `ModelResponse`, declared in that file and
converted from the SDK's objects there. Two things follow. The fake client is a
peer of the real one rather than a mock of the SDK, so "the loop cannot tell
which client it has" is structurally true. And the message array stays ours to
build, which is the reason for choosing Chat Completions in the first place.

**The SDK client is injected into `ModelClient`.**
Retry, fatal classification and usage accrual are the interesting behaviour in
that file, and none of it is testable if the client is constructed inside. A
stub with a `chat.completions.create` exercises every branch with no network.

**Retryable means timeout, connection error, or 5xx. Everything else is fatal.**
A 400 is a malformed request -- a message array that breaks the tool-call
protocol, or a schema the API rejects -- so retrying sends the same broken
request again. Other 4xx are ours too: a bad key, a bad model name. 429 is
knowingly not retried in v1: rate limiting is a real transient condition, but
handling it properly means honouring `Retry-After` rather than backing off
blindly, and a run that hits a rate limit failing loudly is more honest than one
that stalls. Revisit if it ever fires.

**Usage accrues inside the attempt loop, not around it.**
The accumulator is updated the moment a response object exists and before
anything inspects it. A 5xx carries no usage, so today a retried call costs what
its successful attempt reported -- but the accrual point is what keeps that true
if an arrived response is ever discarded. Counting only what the loop ends up
using is how a cost metric quietly becomes a lie.

**The protocol invariant is enforced in `ContextBuilder`, not just observed.**
The loop resolves every tool call to an outcome before emitting any tool
message, so the outcome list has the tool call list's length by construction and
no branch can skip one. On top of that, `ContextBuilder.messages()` refuses to
hand back an array in which an assistant message's `tool_call.id` has no
matching tool message. The convention is what usually holds; the check is what
turns a future `continue` into a `HarnessFatalError` naming the unanswered call
instead of an opaque 400 a second later. The one place that assembles messages
is the one place that can check them.

**Malformed JSON arguments reuse `InvalidArguments`.**
Arguments arrive as a string, so they can fail before any schema is consulted.
That is the same class of failure as a schema rejection -- the model wrote them
and the model can fix them -- so it produces the existing outcome rather than a
new union member. Parsing happens in the loop beside validation, not in
`ModelClient`: it is trust-boundary work.

**Tool results are wrapped from the start.**
`ToolSucceeded.to_tool_message()` emits `{"tool": ..., "result": ...}`. The
sanitizer in M3 adds truncation and whatever else, but leaving a domain result
unwrapped in the context for a milestone would break CLAUDE.md rule 4, and the
envelope is one line. The envelope is composed by string rather than by
re-parsing the result JSON: `result_json` is already valid JSON, and `tool_name`
comes from the registered spec rather than from the model, so neither needs
escaping.

**A tool exception tells the model the tool name and nothing else.**
M0 made every expected condition a structured result, so an exception is a real
fault. The traceback goes to `logging` and no further: it carries file paths,
fixture contents and internal structure, and putting it in the context window
would hand whatever caused the failure a free channel to speak to the model.
Tests assert the sanitized message contains no path, no exception class, and no
traceback, and that the traceback did reach the log.

**`TerminalReason` lives in `loop.py` for now.**
Two values in M2, `COMPLETED` and `MAX_ITERATIONS`, with the human-readable
reason as the enum value. M4 builds the rest of the termination policy and will
likely move the enum to `policy.py` with it, rather than creating an
almost-empty file today.

**`cli/smoke.py` is a small addition to the file list in DESIGN Part 4.**
The model identifier is a required command-line argument, not configuration: it
is the variable of the experiment, and M8 runs the same code against two tiers
to compare them. A value in `.env` becomes an implicit default nobody remembers
setting, and an unset variable that stops and asks is a config branch to reason
about. `OPENAI_API_KEY` comes from the environment; nothing in the repo parses
`.env`.

**`reasoning_effort="none"` is hardcoded in the create call.**
Chat Completions returns a 400 for a request carrying function tools on this
model family unless reasoning is disabled. It is not a performance tweak and the
comment in `model_client.py` says so, because a reader who mistakes it for one
will "improve" it and get a 400 that looks like a schema problem. The API's own
suggestion was to move to `/v1/responses`, which we declined: that endpoint
manages the message array server-side, and building the message array by hand is
the thing this project exists to do. Hardcoded rather than configurable -- one
implementation, and configuration for one implementation is ceremony.

**Open question for M8: is reasoning-off a confound in the tier comparison?**
The restriction above was observed on Luna. If Sol or Terra accept function
tools with reasoning enabled, then a Luna-vs-Sol comparison run through this
client is measuring reasoning-off against reasoning-on rather than tier
capability, and the tool-selection delta M8 reports would be a different number
than it claims to be. Verify at M8 by sending each tier a tool-carrying request
with reasoning enabled. Then either force `"none"` on every tier so the
comparison is like-for-like, or report the asymmetry explicitly alongside the
numbers. Do not let this go unexamined into the eval report.

**Token accrual counts cache reads and writes separately.**
The usage payload carries `prompt_tokens_details.cached_tokens` alongside
`prompt_tokens`, and cached input bills at roughly a tenth of the standard rate.
Accruing only `prompt_tokens` would have made M8's cost metric overstate the
bill substantially: the smoke run's final call cached up to 2298 of its 2737
prompt tokens. `TokenUsage` therefore carries `cached_tokens` and
`cache_write_tokens` as well, accrued at the same call site and on the same
terms, including retried calls. Both are subsets of `prompt_tokens` rather than
additions to it, so `total_tokens` stays prompt plus completion; applying the
rate weights is M8's job, and counting honestly is the client's.

**Prompt caching constrains context ordering, and that is a reason rather than a
convention.**
What gets cached is the stable prefix of the request: the system prompt and the
tool definitions, which are byte-identical on every iteration of a run. The
cache hit is on a prefix, so it survives only as long as nothing before the
first varying byte changes. Two consequences for M3. Ordering must keep the
system prompt and tool definitions first and byte-stable across iterations --
moving anything variable ahead of them, or regenerating the tool schemas with a
different key order, destroys the hit and multiplies input cost by roughly ten.
And truncation must edit the tail rather than the head: rewriting an earlier
tool result to save tokens invalidates every cached byte after it, which costs
far more than the truncation saves. This is measured behaviour from the smoke
run, not a guess.

**M8 eval case, observed in the smoke run: grounding under a truncated extract.**
`search_product_docs` returned a snippet that cut off mid-sentence in the middle
of a titration schedule. The model stated that the extract was incomplete and
declined to present the fuller schedule rather than completing it from its own
knowledge. That is exactly the positive case the grounding metric is meant to
detect, and it arose naturally rather than being provoked. Add it to
`cases.yaml` as a case in its own right: a query whose best-matching document is
truncated at the snippet boundary, scored on whether the answer flags the
incompleteness instead of filling it in. It also gives the metric a positive
example to calibrate against, which a set built only from refusal cases would
lack.

---

## M3 — context, truncation, and the budget

**Budget overflow stops the run. It does not drop messages.**
The arithmetic is the argument. Six iterations, tool results capped at 2,000
characters, a fixed prefix of about 1,430 tokens: the largest context this
harness can assemble is roughly 4,500 tokens against a 12,000-token budget. A
drop-oldest-exchange strategy would be code that cannot fire on any input the
system accepts -- dead code wearing the costume of sophistication, testable only
by lying to it with a fake budget.

It is also the better policy under the cache constraint. Dropping an early
exchange shifts every byte after it, so the first dropped message destroys the
cached prefix beyond that point and the next call re-bills the remainder at full
rate. A policy whose failure mode is "quietly forget what you retrieved, then
pay ten times more to rediscover it" is worse than stopping. And an agent that
silently forgets what it retrieved is worse than one that stops and says so:
the forgetting is invisible in the output, while the stop is a terminal state
with a reason attached.

So per-result truncation is the mechanism that keeps context bounded, and the
total budget is the backstop that proves the bound holds. Messages are appended
and never revisited, which makes "truncation edits the tail, never the head"
true by construction rather than by discipline.

**The per-result cap is 2,000 characters, chosen against measured sizes.**
The largest result the fixtures produce is 1,537 characters (a filtered document
search); the next largest are 1,369 and 1,287. The cap therefore leaves ordinary
traffic untouched with about 30% headroom and bounds the pathological case. A
cap that fired on healthy data would degrade every run to protect against a case
that has not happened.

**A truncated result changes key rather than being cut in half.**
Cutting serialized JSON at a character offset leaves an envelope that is not
parseable. Instead, an oversized payload is emitted as
`{"tool": ..., "result_partial": "<escaped first 2000 chars>", "omitted_chars":
N, "note": ...}`. The envelope stays valid JSON, the untruncated case is
unchanged, and a different key makes it impossible to mistake a fragment for a
whole record. That signal is load-bearing: the smoke run's final answer flagged
its own incompleteness rather than inventing a titration schedule, and that
behaviour depends on the model knowing it holds a fragment.

**One escaping path for every envelope.**
Composing the trusted cases by string and escaping only the untrusted ones is an
invariant that has to be re-derived correctly by whoever adds the next outcome
member in M4 or M6, and the comment explaining it would read as documentation
rather than as a constraint. Everything goes through `json.dumps`. A test feeds
a hostile tool name (`x", "result": {"status": "ok`) through the unknown-tool
path and asserts the envelope still has exactly two keys.

Byte-identity with M2's string composition turned out not to be achievable
through a single `json.dumps` path, and on inspection it does not matter:
tool messages sit after the cached prefix, so their bytes were never part of
the cache hit. What matters is that the format is stable across iterations,
which it is. The output is now fully compact (`separators=(",", ":")`), which is
slightly cheaper than the old mixed spacing. The property that is asserted is
the one that counts: a result under the cap round-trips byte for byte, unmodified.

**The tool name is bounded at 64 characters in the envelope.**
On the unknown-tool path it is model-supplied text sitting in a structural
field. `json.dumps` handles escaping; the cap stops a model from spending an
entire tool message on a name.

**`o200k_base` is fixed rather than derived from the model name.**
`tiktoken.encoding_for_model` raises on identifiers it does not know, which
would make the budget a configuration branch with an error path, for a count
that is an estimate used to enforce a limit rather than to bill anyone. Tool
definitions are included in the count: they are about 1,200 tokens here, and a
budget that ignored them would be measuring the smaller half of the context.

**`ContextBuilder` owns the tool definitions as well as the messages.**
The stable prefix is one thing to the API -- the bytes that get cached -- so it
is one object's responsibility here. The loop asks the builder for both the
messages and the tools rather than assembling half the prefix itself.

**The token estimate was 18% high, and the overhead was located rather than absorbed.**
The first call of the smoke run estimated 1,538 prompt tokens against 1,300
reported. Decomposing it: our serialized tool JSON counted 1,181, and the two
opening messages counted 357 as JSON against 322 counting content plus a fixed
per-message overhead. So 35 tokens of the 238 were message-counting error and
about 203 were in the tool definitions.

Two different fixes, because they have two different causes. The message error
was ours: encoding `json.dumps(message)` charges for the key names and escapes
every newline into two characters. `_message_tokens` now counts what the API
actually transmits -- role, content, tool call names and arguments, tool_call_id
-- and the error disappears at source. The tool-definition error is not ours to
fix: the API bills its own representation of the schema, not the JSON text we
send, and we cannot see that representation. That component gets a measured
correction factor of 0.85 (the measurement implies about 0.83; the rounder,
higher figure keeps the estimate on the conservative side).

The corrected estimate is 1,328 against 1,300 reported, 2.2% high. Two tests
hold the line: one asserts within 5%, and one asserts the estimate is never
below the reported figure, because a budget that underestimates stops failing
safe. The correction is documented as calibrated to this specific tool set and
requires re-measuring if the tools change. A global fudge factor over the whole
count would have hidden the fact that one component was wrong for a reason we
could fix and the other for a reason we could not.

**Caching held across M3 and improved.**
The second smoke run recorded 8,711 cached reads against 2,650 cache writes,
climbing on every iteration. The envelope sentence added to `system.md` cost
roughly 80 tokens on the fixed prefix and did not break prefix stability, which
is what the M3 cache-preservation test asserts structurally.

**Hypothesis for M8, not a conclusion: the envelope sentence may improve grounding.**
On the same goal, the M3 run's final answer was better grounded than M2's. It
stated that the documentation does not cover renal-specific dosing at all,
warned against presenting an undocumented schedule, and closed with an explicit
insufficiency statement. The only changed variable was the paragraph in
`system.md` naming the envelope and the `result_partial` marker.

This is one run of a stochastic system and proves nothing on its own. It is
recorded as a hypothesis for M8 to test properly: run the grounding cases with
and without that paragraph, n>=3 each, and report the difference in refusal and
insufficiency rates. If it holds, it is evidence that naming the trust boundary
in the prompt does work that a general instruction to "answer only from tools"
does not. If it does not hold, the paragraph still earns its place for M7, and
the honest thing is to say the grounding difference was noise.
