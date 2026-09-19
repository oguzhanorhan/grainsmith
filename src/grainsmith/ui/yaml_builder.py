"""Form-values <-> validated YAML, routed through the ENGINE'S own validator.

Pure logic — **no Streamlit**.  This is the module that guarantees the UI never
forks a second config code path: every value the UI collects is turned into a
plain ``dict`` and handed to :func:`grainsmith.config.resolve.resolve_config`,
which performs *exactly* the validation the CLI performs — Pydantic schema
checks (``extra='forbid'`` -> unknown key is an error) **and** the cross-field
rules in ``resolve.py`` (seed/crystal/phases/curved/texture/voxel rules).  A
YAML the UI emits therefore reproduces the run identically (seed included).

Public API
----------
config_from_values(values)  -> RunConfig    (validates, may raise ConfigError)
to_yaml(values)             -> str          (validated config serialised)
config_to_yaml(config)      -> str          (a RunConfig serialised)
load_yaml(text_or_path)     -> (RunConfig, values)   (parse for editing)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from grainsmith.config.resolve import resolve_config
from grainsmith.config.schema import RunConfig
from grainsmith.errors import ConfigError


def _prune(value: Any) -> Any:
    """Drop ``None`` leaves and empty containers so the emitted dict is the
    minimal mapping the user actually set.

    Pydantic happily accepts a sparse mapping (missing keys fall back to their
    defaults); pruning keeps round-tripped YAML readable and avoids forcing
    ``None`` onto fields whose *default* is a non-``None`` factory.  Booleans
    and zeros are preserved (only ``None`` / ``{}`` / ``[]`` are removed).
    """
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if v is not None and v != {} and v != []}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def config_from_values(values: dict[str, Any]) -> RunConfig:
    """Validate raw form ``values`` through the engine and return a RunConfig.

    The dict is pruned of empty leaves, then handed verbatim to
    :func:`grainsmith.config.resolve.resolve_config` — the SAME function
    ``load_config`` calls.  Any failure (schema, unknown key, or a cross-field
    rule) surfaces as the engine's :class:`~grainsmith.errors.ConfigError` with
    its original message, so the UI shows exactly what the CLI would.
    """
    return resolve_config(_prune(values))


def config_to_yaml(config: RunConfig) -> str:
    """Serialise a validated :class:`RunConfig` to canonical YAML.

    Uses ``model_dump(mode='json')`` (the same dump the resolved-config writer
    uses) so enums/Literals become plain scalars and the result reloads
    cleanly.  Keys are sorted for a stable, diff-friendly document.
    """
    raw: dict[str, Any] = config.model_dump(mode="json")
    return yaml.dump(raw, default_flow_style=False, sort_keys=True)


def to_yaml(values: dict[str, Any]) -> str:
    """Validate ``values`` (engine path) then serialise to YAML.

    Raises :class:`~grainsmith.errors.ConfigError` if validation fails — the UI
    must surface that message rather than emit an invalid file.
    """
    return config_to_yaml(config_from_values(values))


def _values_from_config(config: RunConfig) -> dict[str, Any]:
    """A plain-scalar dict mirroring *config* (round-trip seed for editing)."""
    return config.model_dump(mode="json")


def load_yaml(text_or_path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Parse existing YAML into ``(config, values)``.

    A :class:`pathlib.Path` is read from disk.  A ``str`` is ALWAYS parsed as
    YAML *text* — it is never interpreted as a filename.  This is a security
    boundary: the ``POST /api/yaml/to-values`` endpoint forwards raw client text
    here, so treating a string that happens to name a local file as a path to
    read would let any caller disclose file contents (e.g. ``/etc/hostname``).
    Callers that mean a file on disk must pass a :class:`pathlib.Path`
    explicitly (as :mod:`grainsmith.ui.presets` does).  The parsed mapping is
    validated through the engine (so editing starts from a known-good state)
    and the returned ``values`` dict can seed the form widgets.

    For the :class:`~pathlib.Path` form, ``crystal.cif.file`` /
    ``boundaries.voxel_import.file`` fall back to the YAML file's own
    directory exactly like :func:`grainsmith.config.resolve.load_config`
    (its ``base_dir``) — every shipped preset's asset paths are relative
    to the preset's own directory, so this keeps preset editing working
    regardless of the studio server's CWD. A ``str`` (raw text with no
    filesystem location of its own) gets no such fallback — same as
    passing a raw dict to ``resolve_config`` directly.

    Raises
    ------
    ConfigError
        On a YAML parse error, a non-mapping document, or any schema /
        cross-field validation failure.
    """
    base_dir: Path | None = None
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8")
        base_dir = text_or_path.resolve().parent
    else:
        text = text_or_path

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config must be a YAML mapping, got {type(raw).__name__}."
        )

    config = resolve_config(raw, base_dir=base_dir)
    return config, _values_from_config(config)
