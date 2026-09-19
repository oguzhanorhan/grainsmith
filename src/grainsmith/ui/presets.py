"""Shipped example configs exposed as named presets.

Pure logic — **no Streamlit**.  The builder page offers the ``*.yaml`` files
under ``examples/``, recursively, as a starting point; this module locates
them and loads them through the engine's own validator (via
:func:`grainsmith.ui.yaml_builder.load_yaml`), so a preset that ever drifted
out of schema would fail loudly here rather than produce a broken form.

The examples directory ships in the source tree at ``<repo>/examples``; it is
located relative to this file (``src/grainsmith/ui`` -> repo root is three
parents up) so it works from an editable install.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from grainsmith.config.schema import RunConfig
from grainsmith.ui.yaml_builder import load_yaml


def examples_dir() -> Path:
    """Absolute path to the shipped ``examples/`` directory."""
    return Path(__file__).resolve().parents[3] / "examples"


def discover_presets(root: Path | None = None) -> dict[str, Path]:
    """Map ``{preset_name: yaml_path}`` for every ``*.yaml`` under *root*,
    recursively.

    *root* defaults to :func:`examples_dir`.  The preset name is the file stem
    (e.g. ``b2_nial_flat``); stems are unique across the whole examples tree,
    so a flat ``{stem: path}`` mapping is unambiguous even though the files
    themselves live in scenario subfolders (``basics/``, ``cif/``, ...).  The
    mapping is sorted by stem for a stable UI list.  Returns an empty dict
    when the directory is absent (e.g. a wheel install without the examples).
    """
    base = root if root is not None else examples_dir()
    if not base.is_dir():
        return {}
    return {p.stem: p for p in sorted(base.rglob("*.yaml"), key=lambda p: p.stem)}


def load_preset(name_or_path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Load a preset by name (file stem) or by path -> ``(config, values)``.

    A ``str`` is treated ONLY as a preset name resolved against
    :func:`discover_presets`; it is never used as a filesystem path.  This is a
    security boundary: ``GET /api/presets/{name}`` forwards the raw URL segment
    here, so a bare non-preset name must not read a same-named file from the
    process working directory.  A :class:`pathlib.Path` is loaded directly (for
    programmatic / CLI callers).  Validation goes through the engine
    (:func:`load_yaml`), so the returned values reproduce a valid run.

    Raises
    ------
    KeyError
        If a *name* (``str``) matches no shipped preset.
    """
    if isinstance(name_or_path, Path):
        return load_yaml(name_or_path)
    presets = discover_presets()
    if name_or_path in presets:
        return load_yaml(presets[name_or_path])
    raise KeyError(
        f"unknown preset {name_or_path!r}; available: {sorted(presets)}"
    )
