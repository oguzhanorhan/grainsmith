"""Recursive *form-spec* tree for :class:`grainsmith.config.schema.RunConfig`.

Pure logic — **no Streamlit, no I/O**.  This module turns the schema facts
exposed by :mod:`grainsmith.config.introspect` into a UI-framework-neutral tree
that a thin presentation layer (:mod:`grainsmith.ui.schema_json`, consumed by
the React builder form) can walk to lay out widgets.  Keeping it separate
makes the form layout unit-testable without a browser and pins it to the
schema so a renamed/added field can never be silently dropped from the form
(``tests/test_ui_formspec.py``).

Shape of the tree
-----------------
The root is a list of :class:`FormSection` (one per top-level ``RunConfig``
field).  Each section / nested model carries a list of :class:`FormField`:

* a *leaf* field (bool/int/float/str/choice/list/dict/any) is a plain
  :class:`FormField` whose ``spec`` is the introspected
  :class:`~grainsmith.config.introspect.FieldSpec`;
* a *single nested model* field (e.g. ``crystal``, ``box.overlap_removal``)
  carries a populated ``children`` list — the recursion;
* a *list-of-models* field (e.g. ``phases``, ``orientation.components``) carries
  an ``element_model`` form (a :class:`FormModel`) describing ONE element so the
  UI can render a repeatable sub-form.

Recursion is cycle-guarded by a ``seen`` set of model classes; the schema is a
DAG today, but the guard keeps the builder safe if that ever changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from grainsmith.config.introspect import (
    FieldSpec,
    field_specs,
    nested_models,
    unwrap_optional,
)
from grainsmith.config.schema import RunConfig


@dataclass(frozen=True)
class FormModel:
    """A model rendered as a (possibly nested) group of fields."""

    model: type[BaseModel]
    title: str
    fields: list[FormField] = field(default_factory=list)


@dataclass(frozen=True)
class FormField:
    """One schema field reduced to what a form renderer needs.

    ``spec`` is the raw introspection fact.  Exactly one of the structural
    extras may be set:

    * ``children`` — a populated :class:`FormModel` when this field is a single
      (optional) nested model (``widget == 'model'``);
    * ``element_model`` — a :class:`FormModel` describing ONE list element when
      this field is a ``list[Model]`` (``widget == 'list'`` over a model).

    For plain scalar/list/dict leaves both are ``None``.
    """

    spec: FieldSpec
    children: FormModel | None = None
    element_model: FormModel | None = None

    @property
    def name(self) -> str:
        """Convenience: the field name (same as ``spec.name``)."""
        return self.spec.name

    @property
    def widget(self) -> str:
        """Convenience: the widget hint (same as ``spec.widget``)."""
        return self.spec.widget


@dataclass(frozen=True)
class FormSection:
    """A top-level ``RunConfig`` field rendered as a labelled section."""

    name: str
    form: FormModel


def _build_model(
    model: type[BaseModel], title: str, seen: frozenset[type[BaseModel]]
) -> FormModel:
    """Recursively build a :class:`FormModel` for one pydantic model."""
    fields: list[FormField] = []
    for spec in field_specs(model):
        children: FormModel | None = None
        element_model: FormModel | None = None

        if spec.nested_model is not None and spec.nested_model not in seen:
            children = _build_model(
                spec.nested_model,
                spec.nested_model.__name__,
                seen | {spec.nested_model},
            )
        elif spec.widget == "list":
            # list[Model] (e.g. phases, components, from_list, wyckoff_sites):
            # describe ONE element so the UI can render a repeatable sub-form.
            elem_models = nested_models(unwrap_optional(spec.annotation))
            if elem_models and elem_models[0] not in seen:
                em = elem_models[0]
                element_model = _build_model(em, em.__name__, seen | {em})

        fields.append(
            FormField(spec=spec, children=children, element_model=element_model)
        )
    return FormModel(model=model, title=title, fields=fields)


def build_form(model: type[BaseModel] = RunConfig) -> list[FormSection]:
    """Build the section list for *model* (defaults to :class:`RunConfig`).

    One :class:`FormSection` per top-level field; nested models and
    list-element models are recursed into.  The result is a pure data tree —
    no widget objects, no Streamlit calls.
    """
    sections: list[FormSection] = []
    for spec in field_specs(model):
        seen: frozenset[type[BaseModel]] = frozenset({model})
        if spec.nested_model is not None:
            form = _build_model(spec.nested_model, spec.name, seen | {spec.nested_model})
        elif spec.widget == "list":
            elem_models = nested_models(unwrap_optional(spec.annotation))
            if elem_models:
                em = elem_models[0]
                form = FormModel(
                    model=em,
                    title=spec.name,
                    fields=[
                        FormField(
                            spec=spec,
                            element_model=_build_model(
                                em, em.__name__, seen | {em}
                            ),
                        )
                    ],
                )
            else:
                form = FormModel(model=model, title=spec.name, fields=[FormField(spec=spec)])
        else:
            # A top-level scalar/list/dict leaf (none exist in RunConfig today,
            # but the builder must not assume that).
            form = FormModel(model=model, title=spec.name, fields=[FormField(spec=spec)])
        sections.append(FormSection(name=spec.name, form=form))
    return sections


def iter_field_names(sections: list[FormSection]) -> set[str]:
    """Every field name appearing anywhere in the form-spec tree.

    Used by the drift pin (``tests/test_ui_formspec.py``) to assert the form
    can never silently drop a schema field.
    """
    names: set[str] = set()

    def walk(form: FormModel) -> None:
        for ff in form.fields:
            names.add(ff.name)
            if ff.children is not None:
                walk(ff.children)
            if ff.element_model is not None:
                walk(ff.element_model)

    for section in sections:
        names.add(section.name)
        walk(section.form)
    return names


def iter_fields(sections: list[FormSection]) -> list[FormField]:
    """Flatten every :class:`FormField` in the tree (depth-first)."""
    out: list[FormField] = []

    def walk(form: FormModel) -> None:
        for ff in form.fields:
            out.append(ff)
            if ff.children is not None:
                walk(ff.children)
            if ff.element_model is not None:
                walk(ff.element_model)

    for section in sections:
        walk(section.form)
    return out


def find_field(sections: list[FormSection], name: str) -> FormField | None:
    """Return the first :class:`FormField` with ``name`` (depth-first), or None."""
    for ff in iter_fields(sections):
        if ff.name == name:
            return ff
    return None
