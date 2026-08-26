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
