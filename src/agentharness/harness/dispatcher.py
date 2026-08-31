"""Validated call in, outcome out.

The only place a tool function is invoked. Nothing reaches a tool that did not
come through the validator first.
"""

import logging

from agentharness.harness.outcomes import ToolFailed, ToolOutcome, ToolSucceeded
from agentharness.harness.validator import ValidatedCall

logger = logging.getLogger(__name__)


def dispatch(call: ValidatedCall) -> ToolOutcome:
    try:
        result = call.spec.function(**call.args.model_dump())
    except Exception:
        # A tool raising is a bug. M0 made every expected condition -- missing
        # physician, empty history, no matching documents -- a structured
        # result, so an exception here means something is genuinely broken.
        #
        # The traceback goes to the log and no further. It carries file paths,
        # fixture contents and internal structure, and putting it in the
        # context window would hand whatever caused the failure a free channel
        # to speak to the model.
        logger.exception("tool %s raised", call.spec.name)
        return ToolFailed(tool_name=call.spec.name)
    return ToolSucceeded(tool_name=call.spec.name, result_json=result.model_dump_json())
