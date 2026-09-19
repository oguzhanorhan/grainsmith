"""JSON schema-payload pins (React backend).

The React form of "grainsmith studio" is generated from the ``GET /api/schema``
payload produced by :func:`grainsmith.ui.schema_json.schema_to_json`, so it must
never silently drop a schema field; the critical pin below mirrors
``tests/test_ui_formspec.py`` across the JSON boundary.
"""
from __future__ import annotations

import json
from typing import Any

from grainsmith.config.introspect import discover
from grainsmith.config.schema import RunConfig
from grainsmith.ui.schema_json import schema_to_json


def _schema_field_names() -> set[str]:
    names: set[str] = set()
    for model in discover(RunConfig):
        names.update(model.model_fields.keys())
    return names


def _field_names_in_json(payload: dict[str, Any]) -> set[str]:
    """Every field name appearing anywhere in the serialized JSON payload."""
    names: set[str] = set()

    def walk_fields(fields: list[dict[str, Any]]) -> None:
        for f in fields:
            names.add(f["name"])
            if f.get("children") is not None:
                walk_fields(f["children"]["fields"])
            if f.get("element_model") is not None:
                walk_fields(f["element_model"]["fields"])

    for section in payload["sections"]:
        names.add(section["name"])
        walk_fields(section["fields"])
    return names


def test_schema_json_is_json_serializable():
    # Must not raise: the payload is the GET /api/schema body.
    text = json.dumps(schema_to_json())
    assert text
    # Round-trips back to an equal structure.
    assert json.loads(text) == schema_to_json()


def test_schema_json_covers_every_schema_field():
    payload = schema_to_json()
    covered = _field_names_in_json(payload)
    missing = sorted(_schema_field_names() - covered)
    assert not missing, f"schema JSON dropped schema fields: {missing}"


def test_schema_json_top_level_shape():
    payload = schema_to_json()
    assert payload["title"] == "grainsmith studio"
    assert payload["version"]
    assert isinstance(payload["sections"], list)
    assert payload["sections"], "sections must be non-empty"
    for section in payload["sections"]:
        assert set(section) == {"name", "model", "fields"}
        assert isinstance(section["name"], str)
        assert isinstance(section["model"], str)
        assert isinstance(section["fields"], list)


def test_nested_model_section_has_populated_children():
    payload = schema_to_json()
    crystal = next(s for s in payload["sections"] if s["name"] == "crystal")
    # crystal is a single nested model: at least one field recurses via children.
    assert any(
        f.get("children") is not None and f["children"]["fields"]
        for f in crystal["fields"]
    ), "crystal section should carry a populated nested 'children' model"


def test_list_of_models_field_carries_element_model():
    payload = schema_to_json()
    # phases is a list[Model]: its field must describe ONE element via element_model.
    phases = next(s for s in payload["sections"] if s["name"] == "phases")
    field = phases["fields"][0]
    assert field["element_model"] is not None, "phases must carry an element_model"
    assert field["element_model"]["fields"], "phases element_model must have fields"


def test_field_has_expected_keys_and_choice_widget():
    payload = schema_to_json()
    expected_keys = {
        "name", "widget", "required", "optional", "default", "default_repr",
        "description", "constraints_repr", "choices", "sentinel_choices",
        "bounds", "children", "element_model",
    }
    seen_choice = False

    def walk(fields: list[dict[str, Any]]) -> None:
        nonlocal seen_choice
        for f in fields:
            assert set(f) == expected_keys
            if f["widget"] == "choice":
                assert f["choices"], "a 'choice' widget must carry its options"
                seen_choice = True
            if f.get("children") is not None:
                walk(f["children"]["fields"])
            if f.get("element_model") is not None:
                walk(f["element_model"]["fields"])

    for section in payload["sections"]:
        walk(section["fields"])
    assert seen_choice, "expected at least one Literal/choice field in the schema"
