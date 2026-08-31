"""The control tool: how the model says it is finished.

Not a domain tool. It touches no records and goes nowhere near the repository.
It exists so that ending a run is something the model does explicitly, with an
answer whose sources can be checked, rather than something inferred from the
absence of a tool call.
"""

from typing import Literal

from pydantic import BaseModel


class FinalAnswerAck(BaseModel):
    status: Literal["ok"]
    message: str


def submit_final_answer(
    answer: str, sources: list[str], insufficient_information: bool
) -> FinalAnswerAck:
    """Acknowledge the answer. Recording it is the loop's job.

    The arguments are unused here on purpose. The loop reads them from the
    validated call; this function exists so the control tool dispatches like
    every other tool and owes the model exactly one tool message. The
    acknowledgement deliberately does not echo the answer back into the context,
    which would double its cost for no benefit.
    """
    return FinalAnswerAck(status="ok", message="Answer recorded. The run ends here.")
