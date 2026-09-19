"""Generate ``docs/config_reference.md`` from the pydantic schema.

The schema (``grainsmith.config.schema``) is the **single source of truth** for
the configuration; this script walks every ``BaseModel`` reachable from
``RunConfig`` and emits, per model, a table of *field · type · default ·
constraints · description*.  A drift-proof test (``tests/test_docs.py``)
regenerates the document in memory and fails if the committed file is stale,
so the reference can never silently drift from the schema.

Usage::

    python -m tools.gen_config_reference            # write docs/config_reference.md
    python -m tools.gen_config_reference --check     # exit 1 if the file is stale
    python -m tools.gen_config_reference --stdout     # print, do not write
"""
from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from grainsmith.config.introspect import (
    constraints_repr,
    default_repr,
    discover,
    is_model,
    nested_models,
)
from grainsmith.config.schema import RunConfig

DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "config_reference.md"

_HEADER = """\
# grainsmith configuration reference

> **GENERATED FILE — do not edit by hand.**
> Produced by `tools/gen_config_reference.py` from `grainsmith.config.schema`
> (the single source of truth). Regenerate after any schema change:
> `python -m tools.gen_config_reference`. A drift-proof test
> (`tests/test_docs.py`) fails if this file is out of sync with the schema.

Every model below enforces `extra='forbid'` — an unknown key is a config error
(gate G1), not a silent no-op. Cross-field rules that span models (e.g. "exactly
one of `crystal` or `phases`") live in `config/resolve.py` and are documented in
`docs/gates.md` (G1) and `docs/manual.md`, not here.

"""


def _type_str(tp: Any) -> str:
    """Readable type string; nested models render as their class name."""
    if tp is type(None):
        return "null"
    origin = get_origin(tp)
    if origin is typing.Literal:
        return "one of: " + " | ".join(repr(a) for a in get_args(tp))
    if origin in (list, typing.List):  # noqa: UP006
        args = get_args(tp)
        return f"list[{_type_str(args[0])}]" if args else "list"
    if origin in (dict, typing.Dict):  # noqa: UP006
        ka, va = get_args(tp)
        return f"dict[{_type_str(ka)}, {_type_str(va)}]"
    # Optional / unions (incl. X | None)
    if origin is typing.Union or str(type(tp)) == "<class 'types.UnionType'>":
        parts = [_type_str(a) for a in get_args(tp)]
        return " | ".join(parts)
    if is_model(tp):
        return f"[{tp.__name__}](#{tp.__name__.lower()})"
    return getattr(tp, "__name__", str(tp))


def _md_escape(text: str) -> str:
    """Single-line, table-safe text."""
    return " ".join(text.split()).replace("|", "\\|")


def _contains_list_of(tp: Any, sub: type[BaseModel]) -> bool:
    """True when *sub* is reached through a list wrapper inside *tp*."""
    if get_origin(tp) in (list, typing.List):  # noqa: UP006
        return any(a is sub or _contains_list_of(a, sub) for a in get_args(tp))
    return any(_contains_list_of(a, sub) for a in get_args(tp))


def _config_paths(root: type[BaseModel]) -> dict[type[BaseModel], list[str]]:
    """Dotted YAML path(s) at which each nested model appears.

    A model reached through a list renders its step as ``name[]`` (one
    entry per list item). Models reachable from several parents (e.g. a
    per-phase crystal block) list every path.
    """
    paths: dict[type[BaseModel], list[str]] = {root: []}

    def visit(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            for sub in nested_models(field.annotation):
                step = f"{name}[]" if _contains_list_of(
                    field.annotation, sub) else name
                p = f"{prefix}.{step}" if prefix else step
                known = paths.setdefault(sub, [])
                if p not in known:
                    known.append(p)
                    visit(sub, p)

    visit(root, "")
    return paths


def _render_model(model: type[BaseModel], is_root: bool,
                  paths: list[str]) -> str:
    title = f"## {model.__name__}" + (" (root)" if is_root else "")
    doc = (model.__doc__ or "").strip().split("\n\n")[0]
    doc = " ".join(doc.split())
    lines = [title, "", f'<a id="{model.__name__.lower()}"></a>', "", doc, ""]
    if paths:
        # the full dotted YAML path(s): every field below nests under this
        # prefix — writing one at the wrong level is a G1 config error.
        lines.append("**Config path:** " +
                     " · ".join(f"`{p}`" for p in paths))
        lines.append("")
    lines.append("| field | type | default | constraints | description |")
    lines.append("|---|---|---|---|---|")
    for name, field in model.model_fields.items():
        type_cell = _type_str(field.annotation).replace("|", "\\|")
        constr_cell = (constraints_repr(field) or "—").replace("|", "\\|")
        lines.append(
            f"| `{name}` | {type_cell} | "
            f"{default_repr(field)} | {constr_cell} | "
            f"{_md_escape(field.description or '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def render() -> str:
    models = discover(RunConfig)
    paths = _config_paths(RunConfig)
    body = "\n".join(
        _render_model(m, is_root=(m is RunConfig), paths=paths.get(m, []))
        for m in models
    )
    return _HEADER + body + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate config_reference.md.")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if docs/config_reference.md is stale.")
    parser.add_argument("--stdout", action="store_true",
                        help="print to stdout instead of writing the file.")
    args = parser.parse_args(argv)

    content = render()
    if args.stdout:
        sys.stdout.write(content)
        return 0
    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        if current != content:
            sys.stderr.write(
                "docs/config_reference.md is STALE — run "
                "`python -m tools.gen_config_reference`.\n")
            return 1
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(content, encoding="utf-8")
    sys.stdout.write(f"wrote {DOC_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
