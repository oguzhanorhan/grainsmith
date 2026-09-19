"""Tests for the mdf_target canonical-name/alias machinery (STEP 3a).

Covers:
  * canonical_target_type maps both deprecated aliases and leaves
    canonical names (and 'histogram') untouched.
  * build_bins_and_target gives IDENTICAL output for a canonical name and
    its deprecated alias (sigma3_angle_enriched == csl_enriched,
    haar_random == mackenzie).
  * resolve_config emits a DeprecationWarning for the two deprecated
    aliases and none for the canonical names.
  * The mackenzie deprecation message names the point group / space
    group for a non-cubic crystal.
  * sigma3_angle_enriched is rejected for a non-cubic crystal exactly
    like its deprecated alias csl_enriched.
  * resolve_config never rewrites the user's `type` value in place --
    the config_sha256 stability guarantee (config_sha256 is the sha256
    of the dumped RESOLVED config, so silently normalising a deprecated
    alias would change the recorded hash of every archived run that
    used it).
  * The three new MdfTargetConfig fields (odf_drift_max,
    odf_kernel_halfwidth_deg, odf_null_samples) validate their bounds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from grainsmith.config.resolve import resolve_config
from grainsmith.config.schema import MdfTargetConfig
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.errors import ConfigError
from grainsmith.orientation import canonical_target_type
from grainsmith.orientation.mdf import build_bins_and_target


def _sym_cubic() -> np.ndarray:
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    rots, _ = symmetry_ops(hall_from_international(221))
    return proper_rotation_quaternions(A, rots)


def _raw_config(space_group: int = 225, **orientation):
    """Minimal resolve_config-ready raw dict, cubic (Cu, Fm-3m) by
    default -- same shape test_odf_mdf.py's own `_raw_config` uses."""
    cfg = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": space_group},
            "lattice": {"a": 3.615} if space_group == 225 else
                       {"a": 2.95, "c": 4.68},
            "wyckoff_sites": [
                {"element": "Cu" if space_group == 225 else "Ti",
                 "coords": [0.0, 0.0, 0.0] if space_group == 225 else
                           [1 / 3, 2 / 3, 0.25]}],
        },
        "orientation": orientation,
    }
    return cfg


# ---------------------------------------------------------------------------
# canonical_target_type
# ---------------------------------------------------------------------------


def test_canonical_target_type_maps_aliases():
    assert canonical_target_type("mackenzie") == "haar_random"
    assert canonical_target_type("csl_enriched") == "sigma3_angle_enriched"


def test_canonical_target_type_leaves_canonical_names_and_histogram():
    assert canonical_target_type("haar_random") == "haar_random"
    assert canonical_target_type("sigma3_angle_enriched") == "sigma3_angle_enriched"
    assert canonical_target_type("histogram") == "histogram"


# ---------------------------------------------------------------------------
# build_bins_and_target: alias and canonical name give IDENTICAL output
# ---------------------------------------------------------------------------


def test_sigma3_angle_enriched_matches_csl_enriched_output():
    sym = _sym_cubic()
    cfg_new = MdfTargetConfig(type="sigma3_angle_enriched", sigma3_fraction=0.4)
    cfg_old = MdfTargetConfig(type="csl_enriched", sigma3_fraction=0.4)
    edges_new, target_new, ref_new = build_bins_and_target(cfg_new, sym, 30)
    edges_old, target_old, ref_old = build_bins_and_target(cfg_old, sym, 30)
    np.testing.assert_array_equal(edges_new, edges_old)
    np.testing.assert_array_equal(target_new, target_old)
    np.testing.assert_array_equal(ref_new, ref_old)


def test_haar_random_matches_mackenzie_output():
    sym = _sym_cubic()
    cfg_new = MdfTargetConfig(type="haar_random")
    cfg_old = MdfTargetConfig(type="mackenzie")
    edges_new, target_new, ref_new = build_bins_and_target(cfg_new, sym, 30)
    edges_old, target_old, ref_old = build_bins_and_target(cfg_old, sym, 30)
    np.testing.assert_array_equal(edges_new, edges_old)
    np.testing.assert_array_equal(target_new, target_old)
    np.testing.assert_array_equal(ref_new, ref_old)


# ---------------------------------------------------------------------------
# resolve_config: deprecation warnings for the aliases, none for canonical
# ---------------------------------------------------------------------------


def test_csl_enriched_emits_deprecation_warning():
    raw = _raw_config(scheme="random_uniform",
                       mdf_target={"type": "csl_enriched"})
    with pytest.warns(DeprecationWarning, match="sigma3_angle_enriched"):
        resolve_config(raw)


def test_mackenzie_emits_deprecation_warning():
    raw = _raw_config(scheme="random_uniform",
                       mdf_target={"type": "mackenzie"})
    with pytest.warns(DeprecationWarning, match="haar_random"):
        resolve_config(raw)


@pytest.mark.parametrize("target_type", ["haar_random", "sigma3_angle_enriched"])
def test_canonical_names_emit_no_deprecation_warning(target_type):
    """Unlike the deprecated aliases above, the canonical names must not
    trigger the mdf_target deprecation path at all -- checked by
    recording every warning (rather than turning every DeprecationWarning
    process-wide into a hard error, which would also trip on unrelated
    warnings from third-party dependencies and make this test flaky
    across environments) and asserting none of them mention mdf_target."""
    raw = _raw_config(scheme="random_uniform",
                       mdf_target={"type": target_type})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_config(raw)
    mdf_warnings = [w for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and "mdf_target" in str(w.message)]
    assert mdf_warnings == []


def test_mackenzie_deprecation_names_point_group_for_noncubic():
    """SG 194 (hexagonal) is not cubic; the mackenzie deprecation message
    must identify the point group / space group so a user configuring a
    non-cubic crystal understands the reference curve changed under
    them."""
    raw = _raw_config(space_group=194, scheme="random_uniform",
                       mdf_target={"type": "mackenzie"})
    with pytest.warns(DeprecationWarning, match="194"):
        resolve_config(raw)


def test_sigma3_angle_enriched_rejected_for_noncubic():
    """Same cubic-only rejection as the deprecated csl_enriched alias."""
    raw = _raw_config(space_group=194, scheme="random_uniform",
                       mdf_target={"type": "sigma3_angle_enriched"})
    with pytest.raises(ConfigError, match="cubic"):
        resolve_config(raw)


def test_resolve_config_does_not_rewrite_deprecated_type():
    """config_sha256 is the sha256 of the dumped RESOLVED config
    (config/resolve.py's `_config_yaml`); silently rewriting a deprecated
    `type` alias to its canonical name here would change the recorded
    hash of every archived run that used the old name, breaking its
    reproducibility. resolve_config must therefore echo the user's value
    back verbatim -- only warn, never mutate."""
    raw = _raw_config(scheme="random_uniform",
                       mdf_target={"type": "csl_enriched"})
    with pytest.warns(DeprecationWarning):
        resolved = resolve_config(raw)
    assert resolved.orientation.mdf_target is not None
    assert resolved.orientation.mdf_target.type == "csl_enriched"


# ---------------------------------------------------------------------------
# New MdfTargetConfig fields: bounds validation
# ---------------------------------------------------------------------------


def test_odf_drift_max_rejects_negative():
    with pytest.raises(ValidationError):
        MdfTargetConfig(odf_drift_max=-0.01)


def test_odf_drift_max_accepts_none_and_zero():
    assert MdfTargetConfig(odf_drift_max=None).odf_drift_max is None
    assert MdfTargetConfig(odf_drift_max=0.0).odf_drift_max == 0.0


@pytest.mark.parametrize("bad_halfwidth", [0.0, 180.0])
def test_odf_kernel_halfwidth_deg_rejects_boundary_values(bad_halfwidth):
    with pytest.raises(ValidationError):
        MdfTargetConfig(odf_kernel_halfwidth_deg=bad_halfwidth)


def test_odf_null_samples_rejects_zero():
    with pytest.raises(ValidationError):
        MdfTargetConfig(odf_null_samples=0)
