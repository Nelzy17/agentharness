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
