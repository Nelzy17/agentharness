"""A single real-model run, printed as the message sequence the harness built.

Not a test. Nothing gates on it. It exists so the thing can be watched working
against a real model, which scripted tests cannot show.

    python -m agentharness.cli.smoke --model <name>

The model is a required argument rather than configuration: it is the variable
of the experiment, and M8 runs the same code against two tiers to compare them.
OPENAI_API_KEY is read from the environment.
"""

import argparse
import logging
import os
import sys

from openai import OpenAI

from agentharness.domain import repository
from agentharness.harness.loop import AgentLoop
from agentharness.harness.model_client import ModelClient
from agentharness.harness.registry import build_registry
from agentharness.harness.tracer import SqliteTracer
from agentharness.store.runs import RunStore

# Named because they are run repeatedly and retyping a goal changes it. The
# write goal states the due date: without one the model asks for it rather than
# inventing it, which is correct behaviour and leaves the write path untested.
GOALS = {
    "brief": "Prepare me for tomorrow's meeting with Dr. Evelyn Chen about Nexovar.",
    "write": (
        "Create a follow-up for Dr. Patel to send the dosing sheet, due 2026-09-15."
    ),
}
PREVIEW = 400


def render(message: dict) -> str:
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        calls = ", ".join(
            f"{call['function']['name']}({call['function']['arguments']})"
            for call in message["tool_calls"]
        )
        return f"calls {calls}"
    if role == "tool":
        return f"[{message['tool_call_id']}] {_clip(message['content'])}"
    return _clip(message.get("content") or "")


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW else f"{text[:PREVIEW]}... [{len(text)} chars]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model identifier to run against")
    parser.add_argument(
        "--goal",
        default="brief",
        help=f"a preset name ({', '.join(GOALS)}) or a goal written out in full",
    )
    parser.add_argument(
        "--db",
        help="persist the trace here, so `python -m agentharness.cli.trace` can render it",
    )
    arguments = parser.parse_args(argv)
    goal = GOALS.get(arguments.goal, arguments.goal)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        return 1

    # ModelClient logs the API's raw usage payload at DEBUG rather than
    # returning it, so that the SDK's types stay inside that file. Turning that
    # one logger on is how this script shows which token fields came back.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("    %(message)s"))
    usage_log = logging.getLogger("agentharness.harness.model_client")
    usage_log.setLevel(logging.DEBUG)
    usage_log.addHandler(handler)

    registry = build_registry()
    client = ModelClient(OpenAI(api_key=api_key), model=arguments.model)
    tracer = SqliteTracer(RunStore(arguments.db)) if arguments.db else None
    result = AgentLoop(client, registry, tracer=tracer).run(goal)

    for index, message in enumerate(result.messages):
        print(f"{index:>2}  {message['role']:<9} {render(message)}")

    tool_calls = sum(len(m.get("tool_calls", [])) for m in result.messages)
    print()
    print(f"model            {arguments.model}")
    print(f"terminated       {result.terminal_reason.name}: {result.terminal_reason.value}")
    print(f"iterations       {result.iterations}")
    print(f"tool calls       {tool_calls}")
    print(f"tokens           {result.usage.total_tokens} "
          f"({result.usage.prompt_tokens} prompt, {result.usage.completion_tokens} completion)")
    print(f"of which cached  {result.usage.cached_tokens} read, "
          f"{result.usage.cache_write_tokens} written")
    print(f"route            {result.route.name}")
    print(f"sources          {', '.join(result.sources) or '(none cited)'}")
    if result.insufficient_information:
        print("                 the answer declares the information insufficient")

    # Written records, with the attribution that makes them reversible. The
    # repository is in this process, so revoke_run would work here too.
    written = [
        followup
        for followup in repository.all_followups()
        if followup.created_by_run_id == result.run_id
    ]
    if written:
        print()
        print("records written by this run:")
        for followup in written:
            print(
                f"  {followup.followup_id}  due {followup.due_date}  "
                f"tool_call_id={followup.created_by_tool_call_id}"
            )
            print(f"      {followup.description}")
        cited = [f for f in written if f.created_by_tool_call_id in result.sources]
        print(f"  cited in sources: {len(cited)} of {len(written)}")
    if arguments.db:
        print()
        print(f"python -m agentharness.cli.trace {result.run_id} --db {arguments.db}")

    print()
    print(result.answer or "(no answer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
