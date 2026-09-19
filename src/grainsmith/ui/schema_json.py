"""JSON serialization of the schema form-spec tree.

Pure logic — **no FastAPI, no Streamlit, no I/O**.  This module turns the
form-spec tree built by :mod:`grainsmith.ui.formspec` (which is itself derived
from :mod:`grainsmith.config.introspect`) into a plain, JSON-safe ``dict`` so a
non-Python client — the React form of "grainsmith studio" — can rebuild the
guided config form without re-implementing any schema knowledge.

It is the **serialization seam**: the React form is generated from this payload,
so the schema-derivation guarantee (a renamed/added field can never be silently
dropped from the form) established for ``build_form()`` is preserved across the
language boundary.  The drift pin in
``tests/test_ui_schema_json.py`` asserts that every RunConfig-reachable field
name appears somewhere in the serialized payload, mirroring
``tests/test_ui_formspec.py``.

This is the ``GET /api/schema`` body (served by :mod:`grainsmith.ui.server`); the server layer only has to
``json.dumps`` it.  The recursion mirrors ``formspec`` exactly: ``children`` for
a single nested model, ``element_model`` for a ``list[Model]``.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

import grainsmith
from grainsmith.config.schema import RunConfig
from grainsmith.ui.formspec import FormField, FormModel, build_form

# Bound keys we surface (only those present on the field's metadata appear).
_BOUND_KEYS = ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern", "multiple_of")


def _json_safe(value: Any) -> Any:
    """Coerce a field default to something ``json.dumps`` accepts.

    JSON-native scalars/containers pass through; everything else (e.g.
    ``PydanticUndefined``, enums, model instances) degrades to ``None`` so
    serialization never raises.  ``formspec``/``introspect`` already map
    ``PydanticUndefined`` to ``None``, so this is a belt-and-braces guard.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return None


def _bounds_to_json(bounds: dict[str, Any]) -> dict[str, Any] | None:
    """Return only the recognised, JSON-safe bound keys, or ``None`` if empty."""
    out: dict[str, Any] = {}
    for key in _BOUND_KEYS:
        if key in bounds and bounds[key] is not None:
            value = bounds[key]
            # ``pattern`` is a regex string; numeric bounds are JSON scalars.
            out[key] = value if isinstance(value, bool | int | float | str) else str(value)
    return out or None


def _field_to_json(ff: FormField) -> dict[str, Any]:
    """Serialize one :class:`FormField` (recursing into structural extras)."""
    spec = ff.spec
    return {
        "name": spec.name,
        "widget": spec.widget,
        "required": spec.required,
        "optional": spec.optional,
        "default": _json_safe(spec.default),
        "default_repr": spec.default_repr or None,
        "description": spec.description or None,
        "constraints_repr": spec.constraints_repr or None,
        "choices": list(spec.choices) if spec.choices is not None else None,
        "sentinel_choices": (
            list(spec.sentinel_choices)
            if spec.sentinel_choices is not None else None
        ),
        "bounds": _bounds_to_json(spec.bounds),
        "children": _model_to_json(ff.children) if ff.children is not None else None,
        "element_model": (
            _model_to_json(ff.element_model) if ff.element_model is not None else None
        ),
    }


def _model_to_json(form: FormModel) -> dict[str, Any]:
    """Serialize a (possibly nested) :class:`FormModel`."""
    return {
        "model": form.model.__name__,
        "title": form.title,
        "fields": [_field_to_json(ff) for ff in form.fields],
    }


def schema_to_json(model: type[BaseModel] = RunConfig) -> dict[str, Any]:
    """Serialize the *model* form-spec tree to a JSON-safe ``dict``.

    Defaults to :class:`RunConfig`.  The returned mapping is the
    ``GET /api/schema`` payload::

        {
            "version": <grainsmith.__version__>,
            "title": "grainsmith studio",
            "sections": [ {"name", "model", "fields": [...]}, ... ],
        }

    Each ``<Field>`` carries the introspected schema facts plus optional
    ``children`` (single nested model) / ``element_model`` (``list[Model]``)
    sub-models, mirroring :func:`grainsmith.ui.formspec.build_form`.  The result
    is guaranteed JSON-serializable: ``json.dumps(schema_to_json())`` succeeds.
    """
    sections = [
        {
            "name": section.name,
            "model": section.form.model.__name__,
            "fields": [_field_to_json(ff) for ff in section.form.fields],
        }
        for section in build_form(model)
    ]
    return {
        "version": grainsmith.__version__,
        "title": "grainsmith studio",
        "sections": sections,
    }
