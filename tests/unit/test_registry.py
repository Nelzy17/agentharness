"""Structural tests over the registry, the specs, and the generated schemas.

These assert the contract rather than any one tool's behaviour: that the
schemas the model is sent match the models that enforce them, and that they are
shaped the way strict tool calling requires.
"""

import json

import pytest
from pydantic import BaseModel, ConfigDict

from agentharness.harness.registry import ToolRegistry, build_registry
from agentharness.tools.definitions import (
    TOOL_SPECS,
    GetPreviousMeetingsArgs,
    Permission,
    ToolSpec,
    to_openai_schema,
)
from agentharness.tools.meetings import get_previous_meetings

EXPECTED_NAMES = [
    "get_physician_profile",
    "get_previous_meetings",
    "get_open_followups",
    "search_product_docs",
    "create_followup",
]


@pytest.fixture
def registry() -> ToolRegistry:
    return build_registry()


def test_exactly_five_tools_are_registered(registry):
    assert registry.names() == EXPECTED_NAMES


def test_exactly_one_tool_writes_and_it_is_create_followup(registry):
    writes = [
        spec.name for spec in registry.specs() if spec.permission is Permission.WRITE
    ]
    assert writes == ["create_followup"]


def test_lookup_of_an_unregistered_name_returns_none(registry):
    assert registry.get("get_physician") is None


def test_duplicate_names_are_rejected_at_construction():
    with pytest.raises(ValueError, match="duplicate tool names"):
        ToolRegistry([TOOL_SPECS[0], TOOL_SPECS[0]])


# --- descriptions are prompt surface, so they are tested like one ------------

@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_every_tool_has_a_description(spec: ToolSpec):
    assert spec.description.strip()


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_every_field_of_every_args_model_has_a_description(spec: ToolSpec):
    missing = [
        name
        for name, field in spec.args_model.model_fields.items()
        if not (field.description or "").strip()
    ]
    assert not missing, f"{spec.args_model.__name__} fields without a description: {missing}"


# --- the schemas the model is sent -------------------------------------------

@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_generated_schema_declares_exactly_the_models_fields(spec: ToolSpec):
    """The anti-drift assertion: this is why hand-written schemas are banned."""
    schema = to_openai_schema(spec.args_model)
    assert set(schema["properties"]) == set(spec.args_model.model_fields)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_generated_schema_survives_a_json_round_trip(spec: ToolSpec):
    schema = to_openai_schema(spec.args_model)
    assert json.loads(json.dumps(schema)) == schema


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_generated_schema_conforms_to_strict_mode(spec: ToolSpec):
    """Assert the API's contract, not the generator's implementation.

    Strict tool calling requires every property in `required` and
    `additionalProperties: false` on every object, forbids `default`, and
    accepts no $ref that this generator would have to inline.
    """
    _assert_strict(to_openai_schema(spec.args_model))


def _assert_strict(node, path="parameters"):
    assert "default" not in node, f"{path} carries a default, which strict mode rejects"
    assert "title" not in node, f"{path} carries a title, which is only token cost"
    assert "$ref" not in node, f"{path} carries a $ref"
    if node.get("type") == "object":
        assert node.get("additionalProperties") is False, f"{path} allows extra properties"
        assert set(node.get("required", [])) == set(node.get("properties", {})), (
            f"{path} does not require every property"
        )
        for name, subschema in node.get("properties", {}).items():
            _assert_strict(subschema, f"{path}.{name}")
    for index, option in enumerate(node.get("anyOf", [])):
        _assert_strict(option, f"{path}.anyOf[{index}]")


def test_tool_definitions_are_nested_under_a_function_key(registry):
    definition = registry.tool_definitions()[0]
    assert definition["type"] == "function"
    assert definition["function"]["strict"] is True
    assert set(definition["function"]) == {
        "name",
        "description",
        "strict",
        "parameters",
    }


def test_the_whole_payload_is_json_serializable(registry):
    payload = registry.tool_definitions()
    assert json.loads(json.dumps(payload)) == payload


def test_optional_field_is_required_and_nullable_rather_than_omitted(registry):
    """Strict mode has no optional properties, only nullable ones."""
    schema = registry.get("search_product_docs").args_model
    parameters = to_openai_schema(schema)
    assert set(parameters["required"]) == {"query", "product_name"}
    assert {"type": "null"} in parameters["properties"]["product_name"]["anyOf"]


def test_nested_models_are_rejected_rather_than_silently_emitting_a_ref():
    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        inner: Inner

    with pytest.raises(ValueError, match=r"\$ref/\$defs"):
        to_openai_schema(Outer)


def test_meetings_schema_omits_the_limit_the_function_still_accepts():
    """The args model and the function signature are allowed to differ.

    How much history goes into the context window is the harness's decision, so
    `limit` is never offered to the model, while the function keeps its default.
    """
    import inspect

    assert set(GetPreviousMeetingsArgs.model_fields) == {"physician_name"}
    assert inspect.signature(get_previous_meetings).parameters["limit"].default == 5
