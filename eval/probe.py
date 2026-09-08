"""Does this model accept function tools with reasoning enabled?

    python -m eval.probe --model gpt-5.6-luna

Luna returns a 400 for a tool-carrying request unless reasoning_effort is
"none", which is why ModelClient hardcodes it. If a stronger tier accepts tools
with reasoning on, then a comparison between the two through this harness is
measuring reasoning-off against reasoning-off on one and reasoning-off against
what the other *could* have done -- an asymmetry worth reporting rather than
discovering in the numbers.

The probe talks to the SDK directly. ModelClient hardcodes the setting, so it
cannot ask this question, and adding a switch to production code so a diagnostic
can run would be the wrong trade.
"""

import argparse
import json
import os
import sys

from openai import BadRequestError, OpenAI

from agentharness.harness.registry import build_registry

MESSAGES = [{"role": "user", "content": "Say ok."}]


def probe(client: OpenAI, model: str, tools: list[dict], **extra) -> tuple[bool, str]:
    try:
        client.chat.completions.create(
            model=model, messages=MESSAGES, tools=tools, parallel_tool_calls=False, **extra
        )
    except BadRequestError as error:
        return False, str(error)[:300]
    except Exception as error:  # noqa: BLE001 - the probe reports, it does not handle
        return False, f"{type(error).__name__}: {str(error)[:300]}"
    return True, "accepted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in the environment", file=sys.stderr)
        return 1

    client = OpenAI()
    tools = build_registry().tool_definitions()

    results = {
        "reasoning omitted": probe(client, arguments.model, tools),
        "reasoning_effort=none": probe(
            client, arguments.model, tools, reasoning_effort="none"
        ),
        "reasoning_effort=low": probe(
            client, arguments.model, tools, reasoning_effort="low"
        ),
    }

    print(f"model {arguments.model}")
    for label, (accepted, detail) in results.items():
        print(f"  {label:24} {'accepted' if accepted else 'REJECTED'}  {detail if not accepted else ''}")
    print()
    print(json.dumps({k: v[0] for k, v in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
