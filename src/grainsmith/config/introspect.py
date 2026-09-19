"""Pure introspection of the pydantic configuration schema.

This module is the **single, dependency-free** place that walks the
``grainsmith.config.schema`` model tree.  It carries no UI, plotting or
documentation logic — only structured facts about the schema — so it can be
shared by two very different consumers without coupling them:

* ``tools/gen_config_reference.py`` renders ``docs/config_reference.md`` from it
  (a drift-proof test, ``tests/test_docs.py``, pins the output byte-for-byte);
* ``grainsmith.ui`` (optional ``[ui]`` extra) builds form widgets from it.

The two string renderers ``default_repr`` and ``constraints_repr`` match
``tools/gen_config_reference.py``'s exact formatting so the generated
documentation stays byte-identical; the remaining helpers expose the same
facts as *data* (``widget_kind``, ``literal_choices``, ``numeric_bounds``,
``FieldSpec``) for programmatic consumers such as the form builder.
"""
from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------


def is_model(tp: Any) -> bool:
    """True if ``tp`` is a pydantic ``BaseModel`` subclass."""
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_union(tp: Any) -> bool:
    return get_origin(tp) is typing.Union or isinstance(tp, types.UnionType)


def is_optional(tp: Any) -> bool:
    """True if ``None`` is an allowed value of the annotation (``X | None``)."""
    return _is_union(tp) and type(None) in get_args(tp)


def unwrap_optional(tp: Any) -> Any:
    """Return the sole non-``None`` member of ``X | None``; else ``tp`` unchanged."""
    if _is_union(tp):
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


# ---------------------------------------------------------------------------
# Required / default / constraint renderers (verbatim — keep docs byte-stable)
# ---------------------------------------------------------------------------


def is_required(field: Any) -> bool:
    return field.default is PydanticUndefined and field.default_factory is None


def default_repr(field: Any) -> str:
    """Render a field default (handles required, factories, nested models)."""
    if field.default is not PydanticUndefined and field.default is not None:
        return f"`{field.default!r}`"
    if field.default is None and not is_required(field):
        return "`None`"
    if field.default_factory is not None:
        try:
            produced = field.default_factory()
        except Exception:
            return "(factory)"
        if isinstance(produced, BaseModel):
            return "(sub-block defaults)"
        return f"`{produced!r}`"
    return "**required**"


def constraints_repr(field: Any) -> str:
    """Pull annotated-types / pydantic numeric + length + pattern constraints."""
    out: list[str] = []
    for m in field.metadata:
        for attr, sym in (
            ("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<"),
        ):
            val = getattr(m, attr, None)
            if val is not None:
                out.append(f"{sym} {val}")
        min_len = getattr(m, "min_length", None)
        if min_len is not None:
            out.append(f"len ≥ {min_len}")
        max_len = getattr(m, "max_length", None)
        if max_len is not None:
            out.append(f"len ≤ {max_len}")
        pat = getattr(m, "pattern", None)
        if pat is not None:
            out.append(f"pattern `{pat}`")
    return ", ".join(out)


# ---------------------------------------------------------------------------
# Model discovery (verbatim — keep docs ordering byte-stable)
# ---------------------------------------------------------------------------


def nested_models(tp: Any) -> list[type[BaseModel]]:
    """All ``BaseModel`` subclasses mentioned anywhere in a type annotation."""
    if is_model(tp):
        return [tp]
    found: list[type[BaseModel]] = []
    for a in get_args(tp):
        found.extend(nested_models(a))
    return found


def discover(root: type[BaseModel]) -> list[type[BaseModel]]:
    """Depth-first, definition-order list of unique models reachable from root."""
    seen: list[type[BaseModel]] = []

    def visit(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.append(model)
        for field in model.model_fields.values():
            for sub in nested_models(field.annotation):
                visit(sub)

    visit(root)
    return seen


# ---------------------------------------------------------------------------
# Structured facts for programmatic consumers (the form builder)
# ---------------------------------------------------------------------------


def literal_choices(tp: Any) -> tuple[Any, ...] | None:
    """The allowed values of a ``Literal`` (also through an ``X | None`` wrap)."""
    if get_origin(tp) is typing.Literal:
        return get_args(tp)
    if _is_union(tp):
        for a in get_args(tp):
            if get_origin(a) is typing.Literal:
                return get_args(a)
    return None


def numeric_bounds(field: Any) -> dict[str, Any]:
    """Raw constraint *values* (for widget min/max/step), as a flat dict."""
    bounds: dict[str, Any] = {}
    for m in field.metadata:
        for attr in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern"):
            val = getattr(m, attr, None)
            if val is not None:
                bounds[attr] = val
    return bounds


def sentinel_choices(tp: Any) -> tuple[Any, ...] | None:
    """The ``Literal`` sentinel values of a MIXED ``T | Literal[...]`` union
    (e.g. ``float | Literal["auto"]``), or ``None`` for a pure ``Literal`` or a
    non-union type.

    A mixed union widens a numeric/text field with a small set of magic string
    values (``"auto"``).  Those sentinels are surfaced separately from
    :func:`literal_choices` so the form can offer BOTH the base widget (a numeric
    input) and the sentinel toggle — collapsing such a field to a choice-only
    widget hides the numeric alternative entirely.
    """
    inner = unwrap_optional(tp)
    if not _is_union(inner):
        return None
    literals: list[Any] = []
    has_non_literal = False
    for a in get_args(inner):
        if get_origin(a) is typing.Literal:
            literals.extend(get_args(a))
        elif a is not type(None):
            has_non_literal = True
    if literals and has_non_literal:
        return tuple(literals)
    return None


def widget_kind(tp: Any) -> str:
    """A coarse widget hint: one of
    ``bool|int|float|str|choice|model|list|dict|any``.

    A PURE ``Literal`` (optionally ``Optional``-wrapped) maps to ``choice``.  A
    MIXED ``T | Literal[...]`` union takes the widget of its base type ``T`` (its
    sentinels are surfaced via :func:`sentinel_choices`), so a numeric-plus-
    ``"auto"`` field renders as a number input rather than a choice."""
    if sentinel_choices(tp) is None and literal_choices(tp) is not None:
        return "choice"
    inner = unwrap_optional(tp)
    if sentinel_choices(tp) is not None:
        # Reduce the mixed union to its single non-Literal base type.
        bases = [a for a in get_args(inner)
                 if get_origin(a) is not typing.Literal and a is not type(None)]
        if len(bases) == 1:
            inner = bases[0]
    if is_model(inner):
        return "model"
    origin = get_origin(inner)
    if origin in (list, typing.List):  # noqa: UP006
        return "list"
    if origin in (dict, typing.Dict):  # noqa: UP006
        return "dict"
    if inner is bool:  # before int — bool is an int subclass
        return "bool"
    if inner is int:
        return "int"
    if inner is float:
        return "float"
    if inner is str:
        return "str"
    return "any"


@dataclass(frozen=True)
class FieldSpec:
    """A schema field reduced to the facts a form builder needs.

    Neutral with respect to any UI framework: ``widget`` is a hint, not a
    Streamlit call.  ``nested_model`` is set only for a (possibly optional)
    single nested model; for ``list[Model]`` use :func:`nested_models` on
    ``annotation`` to find the element model.
    """

    name: str
    annotation: Any
    widget: str
    required: bool
    optional: bool
    default: Any
    default_repr: str
    constraints_repr: str
    description: str
    choices: tuple[Any, ...] | None
    sentinel_choices: tuple[Any, ...] | None
    nested_model: type[BaseModel] | None
    bounds: dict[str, Any]


def field_specs(model: type[BaseModel]) -> list[FieldSpec]:
    """Structured, definition-order field specs for one model (non-recursive)."""
    specs: list[FieldSpec] = []
    for name, field in model.model_fields.items():
        inner = unwrap_optional(field.annotation)
        nested = inner if is_model(inner) else None
        if nested is None and _is_union(inner):
            # Mixed Literal|Model union (e.g. DopantConfig.sites: preset
            # names | SitesCoordsConfig): reduce to the single model arm
            # so the form recurses into it instead of silently dropping
            # its fields (drift pin tests/test_ui_formspec.py).
            bases = [a for a in get_args(inner)
                     if get_origin(a) is not typing.Literal
                     and a is not type(None) and is_model(a)]
            if len(bases) == 1:
                nested = bases[0]
        specs.append(
            FieldSpec(
                name=name,
                annotation=field.annotation,
                widget=widget_kind(field.annotation),
                required=is_required(field),
                optional=is_optional(field.annotation),
                default=(field.default if field.default is not PydanticUndefined else None),
                default_repr=default_repr(field),
                constraints_repr=constraints_repr(field),
                description=field.description or "",
                choices=(None if sentinel_choices(field.annotation)
                         else literal_choices(field.annotation)),
                sentinel_choices=sentinel_choices(field.annotation),
                nested_model=nested,
                bounds=numeric_bounds(field),
            )
        )
    return specs
