"""Validated call in, outcome out.

The only place a tool function is invoked. Nothing reaches a tool that did not
come through the validator first.
"""

import logging
import traceback

from agentharness.domain.models import WriteAttribution
from agentharness.harness.model_client import HarnessFatalError
from agentharness.harness.outcomes import ToolFailed, ToolOutcome, ToolSucceeded
from agentharness.harness.validator import ValidatedCall
from agentharness.tools.definitions import Permission

logger = logging.getLogger(__name__)


def dispatch(call: ValidatedCall, attribution: WriteAttribution | None = None) -> ToolOutcome:
    """Run one validated call, supplying attribution to the tools that write.

    One declaration -- permission=WRITE on the spec -- drives schema generation,
    the policy check, the trace flag and this. A write dispatched without
    attribution is a harness bug rather than a model error, so it raises here
    instead of producing an unattributable record.
    """
    arguments = call.args.model_dump()
    if call.spec.permission is Permission.WRITE:
        if attribution is None:
            raise HarnessFatalError(
                f"{call.spec.name} writes and was dispatched without attribution"
            )
        arguments["attribution"] = attribution

    try:
        result = call.spec.function(**arguments)
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
        return ToolFailed(name=call.spec.name, detail=traceback.format_exc())
    return ToolSucceeded(name=call.spec.name, result_json=result.model_dump_json())
