"""Cross-field config validation rules (gate G1, §7)."""
import logging

import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError


def _raw(**overrides):
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu",
                               "coords": [0.0, 0.0, 0.0]}],
        },
    }
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            raw[key].update(sub)
        else:
            raw[key] = sub
    return raw


def test_valid_config_passes():
    cfg = resolve_config(_raw())
    assert cfg.grains.number == 4


def test_fixed_seed_requires_value():
    with pytest.raises(ConfigError, match="seed.value"):
        resolve_config(_raw(seed={"mode": "fixed", "value": None}))


def test_vacuum_requires_free_axis():
    with pytest.raises(ConfigError, match="vacuum"):
        resolve_config(_raw(box={"lengths": [40.0, 40.0, 40.0],
                                 "periodic": [True, True, True],
                                 "vacuum": 5.0}))


def test_unknown_key_rejected():
    raw = _raw()
    raw["grains"]["typo_field"] = 1
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_csl_requires_cubic_space_group():
    raw = _raw(analysis={"csl": True})
    raw["crystal"]["space_group"]["number"] = 194  # hexagonal
    with pytest.raises(ConfigError, match="cubic"):
        resolve_config(raw)
    # cubic SG passes the config-time rule
    cfg = resolve_config(_raw(analysis={"csl": True}))
    assert cfg.analysis.csl


def test_occupancy_must_sum_to_one():
    raw = _raw()
    raw["crystal"]["wyckoff_sites"] = [
        {"element": {"Ti": 0.90, "Al": 0.05}, "coords": [0.0, 0.0, 0.0]},
    ]
    with pytest.raises(ConfigError, match="sum to"):
        resolve_config(raw)


def test_occupancy_must_be_positive():
    raw = _raw()
    raw["crystal"]["wyckoff_sites"] = [
        {"element": {"Ti": 1.10, "Al": -0.10}, "coords": [0.0, 0.0, 0.0]},
    ]
    with pytest.raises(ConfigError, match="positive"):
        resolve_config(raw)


def test_occupancy_within_tolerance_passes():
    raw = _raw()
    raw["crystal"]["wyckoff_sites"] = [
        {"element": {"Ti": 0.9000000004, "Al": 0.0999999999},
         "coords": [0.0, 0.0, 0.0]},
    ]
    cfg = resolve_config(raw)
    assert isinstance(cfg.crystal.wyckoff_sites[0].element, dict)


def test_warp_amplitude_guard():
    raw = _raw(boundaries={
        "geometry": "curved",
        "curved": {"method": "warp", "amplitude": 1000.0,
                   "correlation_length": 15.0},
    })
    with pytest.raises(ConfigError, match="amplitude"):
        resolve_config(raw)


# ---------------------------------------------------------------------------
# Fix #24 — aspect_ratio_range validation
# ---------------------------------------------------------------------------

def test_aspect_ratio_range_inverted_rejected():
    """Inverted range [5.0, 2.0] must be rejected."""
    raw = _raw(boundaries={
        "geometry": "curved",
        "curved": {"method": "anisotropic", "aspect_ratio_range": [5.0, 2.0]},
    })
    with pytest.raises(ConfigError, match="aspect_ratio_range"):
        resolve_config(raw)


def test_aspect_ratio_range_negative_rejected():
    """Negative values like [-1, 2] must be rejected."""
    raw = _raw(boundaries={
        "geometry": "curved",
        "curved": {"method": "anisotropic", "aspect_ratio_range": [-1.0, 2.0]},
    })
    with pytest.raises(ConfigError, match="aspect_ratio_range"):
        resolve_config(raw)


def test_aspect_ratio_range_valid_passes():
    """Valid range [1.0, 3.0] must pass validation."""
    raw = _raw(boundaries={
        "geometry": "curved",
        "curved": {"method": "anisotropic", "aspect_ratio_range": [1.0, 3.0]},
    })
    cfg = resolve_config(raw)
    assert cfg.boundaries.curved.aspect_ratio_range == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Fix #11 — spread_deg upper bound of 180°
# ---------------------------------------------------------------------------

def test_fiber_spread_deg_over_180_rejected():
    """spread_deg > 180 in OrientationFiberConfig must be rejected."""
    raw = _raw(orientation={
        "scheme": "fiber",
        "fiber": {
            "crystal_axis": [1.0, 0.0, 0.0],
            "sample_direction": "z",
            "spread_deg": 200.0,
        },
    })
    with pytest.raises(ConfigError, match="spread_deg"):
        resolve_config(raw)


def test_odf_component_spread_deg_over_180_rejected():
    """spread_deg > 180 in OdfComponentConfig must be rejected."""
    raw = _raw(orientation={
        "scheme": "odf_components",
        "components": [
            {
                "euler_bunge_deg": [0.0, 0.0, 0.0],
                "weight": 1.0,
                "spread_deg": 200.0,
            }
        ],
    })
    with pytest.raises(ConfigError, match="spread_deg"):
        resolve_config(raw)


# ---------------------------------------------------------------------------
# Fix #9 (historical) — Rule 8 self_affine spectrum does not emit spurious
# warning. Superseded by Rule 8a: 'warp' + spectrum
# 'self_affine' is no longer merely "ignores correlation_length, so no
# Rule 8 warning" — it is now a hard ConfigError, raised BEFORE Rule 8 is
# ever evaluated (warp is a coordinate diffeomorphism and cannot produce a
# genuinely self-affine boundary regardless of Hurst exponent; docs/physics.md
# §5b). The scenario Fix #9's test exercised (self_affine reaching Rule 8's
# warning check at all) can no longer occur, so the test below pins the
# NEW behavior at the same call site instead of the old warning-suppression
# behavior.
# ---------------------------------------------------------------------------

def _raw_big(**overrides):
    """Like _raw but with a large box (100³) so Rule 7 a_clip is generous."""
    raw = _raw()
    raw["box"]["lengths"] = [100.0, 100.0, 100.0]
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            raw[key].update(sub)
        else:
            raw[key] = sub
    return raw


def test_rule8a_warp_self_affine_raises_before_rule8_is_reached(caplog):
    """'warp' + spectrum 'self_affine' is a hard ConfigError (Rule 8a),
    raised unconditionally regardless of amplitude/correlation_length —
    it never reaches Rule 8's bijectivity-warning check at all (the
    scenario test_rule8_warning_not_emitted_for_self_affine, pre-this-task,
    used to exercise: self_affine reaching Rule 8 and being exempted from
    its warning). amplitude=4.0, correlation_length=5.0 here is the exact
    combination that used to trigger (or, for self_affine, be exempted
    from) Rule 8's warning -- now none of that machinery is reached.
    """
    raw = _raw_big(boundaries={
        "geometry": "curved",
        "curved": {
            "method": "warp",
            "spectrum": "self_affine",
            "amplitude": 4.0,
            "correlation_length": 5.0,
            "l_min": 8.0,
            "l_max": 40.0,
        },
    })
    with caplog.at_level(logging.WARNING, logger="grainsmith.config.resolve"):
        with pytest.raises(ConfigError, match="does not accept spectrum "
                                              "'self_affine'"):
            resolve_config(raw)
    # Rule 8a raises before Rule 8 is ever reached, so no bijectivity
    # warning should have been logged either.
    bijectivity_warnings = [
        r for r in caplog.records
        if "bijectivity" in r.getMessage()
    ]
    assert not bijectivity_warnings, (
        "Rule 8a should raise before Rule 8's bijectivity check runs"
    )


def test_rule8_warning_emitted_for_gaussian(caplog):
    """gaussian spectrum + large amplitude MUST still trigger the bijectivity warning.

    amplitude=4.0, correlation_length=5.0 → 4.0 > 0.3*5.0=1.5, fires Rule 8.
    """
    raw = _raw_big(boundaries={
        "geometry": "curved",
        "curved": {
            "method": "warp",
            "spectrum": "gaussian",
            "amplitude": 4.0,
            "correlation_length": 5.0,
        },
    })
    with caplog.at_level(logging.WARNING, logger="grainsmith.config.resolve"):
        resolve_config(raw)
    bijectivity_warnings = [
        r for r in caplog.records
        if "bijectivity" in r.getMessage()
    ]
    assert bijectivity_warnings, (
        "Rule 8 bijectivity warning MUST fire for gaussian spectrum"
    )


def test_mesh_format_rejects_unsupported_value():
    """output.mesh.format is constrained to the only supported value ('ply');
    an unsupported value must be rejected at config validation, not silently
    accepted then ignored by the pipeline."""
    import pytest
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError
    raw = {
        "meta": {"title": "m", "verbose": 0},
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {"space_group": {"number": 225}, "lattice": {"a": 3.615},
                    "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}]},
        "output": {"directory": "out", "mesh": {"enabled": True, "format": "stl"}},
    }
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_hybrid_literal_field_is_numeric_with_sentinel():
    """A `T | Literal[sentinel]` field (min_seed_distance, voxel_grid) must map to
    the NUMERIC widget with the sentinel offered separately — not collapse to a
    choice-only widget that hides the numeric input from the generated form."""
    from typing import Literal
    from grainsmith.config.introspect import (
        sentinel_choices,
        widget_kind,
    )
    assert widget_kind(float | Literal["auto"]) == "float"
    assert sentinel_choices(float | Literal["auto"]) == ("auto",)
    assert widget_kind(int | Literal["auto"]) == "int"
    # A pure Literal stays a plain choice with no sentinel split.
    assert widget_kind(Literal["a", "b"]) == "choice"
    assert sentinel_choices(Literal["a", "b"]) is None


def test_schema_json_hybrid_fields_expose_sentinel():
    """The serialized schema for the two hybrid RunConfig fields carries a numeric
    widget plus sentinel_choices == ['auto']."""
    from grainsmith.ui.schema_json import schema_to_json
    doc = schema_to_json()
    fields = {}
    for section in doc["sections"]:
        def walk(model_fields):
            for f in model_fields:
                fields[f["name"]] = f
                if f.get("children"):
                    walk(f["children"]["fields"])
        walk(section["fields"])
    msd = fields["min_seed_distance"]
    assert msd["widget"] == "float"
    assert msd["sentinel_choices"] == ["auto"]
    vg = fields["voxel_grid"]
    assert vg["widget"] == "int"
    assert vg["sentinel_choices"] == ["auto"]


def test_flat_geometry_with_curved_block_is_config_error():
    """Rule 6: a curved block under geometry 'flat' must be refused, not
    silently ignored (honest refusal beats silent ignoring)."""
    with pytest.raises(ConfigError, match="curved"):
        resolve_config(_raw(boundaries={
            "geometry": "flat",
            "curved": {"method": "anisotropic"},
        }))


def test_extra_forbidden_error_names_the_valid_fields():
    """The schema-error message for a misplaced key must list the valid
    fields of the model it landed on (e.g. the grains.type slip must
    point the user at size_distribution)."""
    raw = _raw()
    raw["grains"]["type"] = "equal"
    with pytest.raises(ConfigError, match="size_distribution"):
        resolve_config(raw)


def test_extra_forbidden_hint_in_nested_model():
    raw = _raw(boundaries={"geometry": "curved",
                           "curved": {"method": "warp", "typo_knob": 1}})
    with pytest.raises(ConfigError, match="correlation_length"):
        resolve_config(raw)
