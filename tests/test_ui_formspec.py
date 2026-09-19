"""Form-spec pins.

The guided form is generated from the schema, so it must never silently drop a
field; these pins fail if the form-spec tree loses a RunConfig-reachable field
or mis-classifies a representative widget.
"""
from __future__ import annotations

from grainsmith.config.introspect import discover
from grainsmith.config.schema import RunConfig
from grainsmith.ui.formspec import build_form, find_field, iter_field_names, iter_fields


def _schema_field_names() -> set[str]:
    names: set[str] = set()
    for model in discover(RunConfig):
        names.update(model.model_fields.keys())
    return names


def test_form_covers_every_schema_field():
    covered = iter_field_names(build_form())
    missing = sorted(_schema_field_names() - covered)
    assert not missing, f"form-spec dropped schema fields: {missing}"


def test_literal_field_is_a_choice_widget():
    fields = iter_fields(build_form())
    assert any(
        f.name == "mode" and f.widget == "choice"
        and f.spec.choices and "fixed" in f.spec.choices
        for f in fields
    ), "seed.mode Literal should surface as a 'choice' widget with its options"


def test_bounded_int_keeps_its_bounds():
    fields = iter_fields(build_form())
    assert any(
        f.name == "verbose" and f.widget == "int"
        and f.spec.bounds.get("ge") == 0 and f.spec.bounds.get("le") == 3
        for f in fields
    ), "meta.verbose should be an 'int' widget carrying its ge/le bounds"


def test_nested_model_is_recursed():
    tree = build_form()
    assert "crystal" in {s.name for s in tree}
    # crystal recurses into its sub-models
    assert find_field(tree, "lattice") is not None
    assert find_field(tree, "space_group") is not None
