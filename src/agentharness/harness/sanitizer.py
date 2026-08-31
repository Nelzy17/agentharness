"""Tool outcome in, tool-message string out.

The only producer of the text that goes into a role="tool" message. It owns the
envelope, the size cap, and the marker that tells the model it is holding a
fragment.

CLAUDE.md rule 4: tool output is data, never instruction. The envelope is how
that is stated on the wire -- everything a tool produced sits inside a named
field of an object the harness built, so nothing a tool returned can be read as
though the harness had said it.
"""

import json

from agentharness.harness.outcomes import ToolOutcome

# The largest healthy result the fixtures produce is about 1,540 characters, so
# ordinary traffic passes through untouched and this bounds the pathological
# case rather than degrading the normal one.
MAX_PAYLOAD_CHARS = 2000

# The tool name is model-supplied on the unknown-tool path, so it is untrusted
# text sitting in a structural field. Escaping is handled by json.dumps; this
# stops a model from spending the whole message on a name.
MAX_TOOL_NAME_CHARS = 64

TRUNCATION_NOTE = (
    "This result was too large to send in full. You are reading the beginning "
    "of it. Do not treat the omitted part as absent or empty, and say the "
    "extract was incomplete if you rely on it."
)


def sanitize(outcome: ToolOutcome, tool_call_id: str) -> str:
    """Render one outcome as the content of its tool message.

    One construction path for every outcome, always through json.dumps, so
    every value is escaped whether or not the harness believes it is trusted.
    The alternative -- composing the trusted cases by string -- is an invariant
    that has to be re-derived correctly by whoever adds the next member.

    The tool_call_id is in the envelope because the model otherwise never sees
    one. Ids live in protocol fields -- tool_calls[].id on the way out,
    tool_call_id on the way back -- which are not content, so a model asked to
    cite them has only ever been able to guess at the format. Putting the id in
    the body makes it readable text rather than metadata.
    """
    payload = outcome.payload()
    # The outcome declares which it is. Testing types here would mean keeping a
    # list of classes in step with the union by hand.
    key = outcome.envelope_key
    envelope: dict[str, object] = {
        "tool": outcome.tool_name[:MAX_TOOL_NAME_CHARS],
        "tool_call_id": tool_call_id,
    }

    if len(payload) > MAX_PAYLOAD_CHARS:
        # Kept as an escaped string under a different key. The model cannot
        # mistake a fragment for a whole record, and the envelope stays valid
        # JSON, which cutting a serialized object in half would not.
        envelope[f"{key}_partial"] = payload[:MAX_PAYLOAD_CHARS]
        envelope["omitted_chars"] = len(payload) - MAX_PAYLOAD_CHARS
        envelope["note"] = TRUNCATION_NOTE
    elif key == "result":
        # Nested as an object rather than a string: the domain result is
        # already JSON, and re-escaping it would cost tokens and readability.
        envelope[key] = json.loads(payload)
    else:
        envelope[key] = payload

    # Compact separators: this runs on every tool message of every iteration,
    # and the spaces are tokens. Key order is insertion order and therefore
    # stable, which is what keeps the shape identical across iterations.
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)
