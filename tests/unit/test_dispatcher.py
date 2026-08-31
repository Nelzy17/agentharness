"""Dispatch: the only place a tool function is called."""

import json
from dataclasses import replace

from agentharness.harness.dispatcher import dispatch
from agentharness.harness.outcomes import ToolFailed, ToolSucceeded
from agentharness.harness.registry import build_registry
from agentharness.harness.validator import Validator

SECRET = "/srv/secret/fixtures.db"


def validated(tool_name, arguments):
    return Validator(build_registry()).validate(tool_name, arguments)


def test_a_successful_call_carries_the_serialized_domain_result():
    outcome = dispatch(validated("get_physician_profile", {"physician_name": "Evelyn Chen"}))

    assert isinstance(outcome, ToolSucceeded)
    assert json.loads(outcome.result_json)["physician"]["physician_id"] == "phy-001"


def test_the_payload_is_the_bare_domain_result():
    """Wrapping moved to the sanitizer in M3; dispatch carries the result only.

    The envelope assertions that used to live here are now in test_sanitizer.py.
    """
    outcome = dispatch(validated("get_physician_profile", {"physician_name": "Evelyn Chen"}))
    assert json.loads(outcome.payload())["status"] == "ok"


def test_a_domain_not_found_is_a_success_not_a_failure():
    """M0 made expected conditions data. Dispatch must not reclassify them."""
    outcome = dispatch(validated("get_physician_profile", {"physician_name": "Nakamura"}))

    assert isinstance(outcome, ToolSucceeded)
    assert json.loads(outcome.result_json)["status"] == "not_found"


def test_a_tool_that_raises_becomes_a_sanitized_failure(caplog):
    call = validated("get_physician_profile", {"physician_name": "Evelyn Chen"})

    def explode(**kwargs):
        raise RuntimeError(f"could not open {SECRET}")

    outcome = dispatch(replace(call, spec=replace(call.spec, function=explode)))

    assert isinstance(outcome, ToolFailed)
    assert outcome.model_visible is True
    message = outcome.payload()
    assert "get_physician_profile" in message
    for leak in (SECRET, "RuntimeError", "Traceback"):
        assert leak not in message


def test_the_traceback_goes_to_the_log_and_only_to_the_log(caplog):
    call = validated("get_physician_profile", {"physician_name": "Evelyn Chen"})

    def explode(**kwargs):
        raise RuntimeError(f"could not open {SECRET}")

    with caplog.at_level("ERROR"):
        outcome = dispatch(replace(call, spec=replace(call.spec, function=explode)))

    assert SECRET in caplog.text
    assert "Traceback" in caplog.text
    assert SECRET not in outcome.payload()


def test_arguments_reach_the_function_by_name():
    outcome = dispatch(
        validated(
            "search_product_docs", {"query": "renal impairment", "product_name": "Nexovar"}
        )
    )
    results = json.loads(outcome.result_json)["results"]

    assert results
    assert all(snippet["product"] == "Nexovar" for snippet in results)
