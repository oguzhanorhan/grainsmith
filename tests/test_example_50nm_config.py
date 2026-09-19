"""Schema/resolve-only (gate G1) validation of
``examples/self_affine_gb/pdau_perturbed_50nm_H05.yaml``.

This example is deliberately NOT run end-to-end by the test suite (or by
anything else in this repo) -- its own honestly-stated cost estimate is
~5.3e7 atoms and several GB of per-color field memory during
tessellation, well beyond what a CI/test run should pay for. What IS
checked here is exactly the fast, cheap gate G1 the file's own header
comment promises: the YAML parses, the schema/cross-field rules
(``config.resolve.load_config``) accept it, and the specific
`amplitude_convention: reference_wavelength` + wide (~5.5 octave) band
this file exists to demonstrate resolve to the values its own comments
claim -- WITHOUT constructing a ``PerturbedDistanceTessellation`` (no
field synthesis, no tessellation, no fill).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grainsmith.config.resolve import load_config
from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY
from grainsmith.seeding import wigner_seitz_radius
from grainsmith.tessellation.warp import reference_shell_kappa


@pytest.fixture(scope="module")
def cfg(examples_dir: Path):
    return load_config(examples_dir / "self_affine_gb" / "pdau_perturbed_50nm_H05.yaml")


def test_50nm_example_loads_and_resolves(cfg):
    """G1: the file parses and passes every schema + cross-field rule --
    in particular Rule 29 (amplitude_convention/reference_wavelength) and
    Rule 20/7b (self_affine band + seed-containment guard), all evaluated
    against this file's own wide (~5.5-octave) band and
    reference_wavelength-convention amplitude."""
    curved = cfg.boundaries.curved
    assert cfg.boundaries.geometry == "curved"
    assert curved.method == "perturbed_distance"
    assert curved.spectrum == "self_affine"
    assert curved.amplitude_convention == "reference_wavelength"
    assert curved.reference_wavelength is None  # -> defaults to l_max
    assert curved.hurst == pytest.approx(0.5)


def test_50nm_example_grain_diameter_matches_target(cfg):
    """The resolved (box, grains.number) geometry actually gives a ~500 A
    (50 nm) equivalent grain diameter, as the file's header comments and
    filename both claim -- computed via the SAME
    grainsmith.seeding.wigner_seitz_radius the pipeline itself calls for
    min_seed_distance: auto, not re-derived independently."""
    L = float(cfg.box.lengths[0])
    n = cfg.grains.number
    r_ws = wigner_seitz_radius(L ** 3, n)
    d_eq_nm = 2.0 * r_ws / 10.0
    assert d_eq_nm == pytest.approx(50.0, abs=1.0), (
        f"resolved equivalent grain diameter {d_eq_nm:.2f} nm is not "
        "within 1 nm of the ~50 nm target this example's filename and "
        "header comments claim"
    )
    # min_seed_distance: auto must resolve to this same r_ws.
    assert cfg.grains.min_seed_distance == "auto"


def test_50nm_example_band_clears_octave_threshold(cfg):
    """[l_min, l_max] spans ~5.5 octaves -- the regime docs/physics.md
    Sec 5b(g) identifies as necessary for gate G20's d_b_estimated to
    resolve `hurst` at a conventionally LARGE effect size (eta^2 > 80%;
    a merely statistically-significant response is reached much earlier,
    ~3.2 octaves per the same section's table) -- the entire reason this
    example exists, as opposed to just reusing pdau_perturbed_self_affine.
    yaml's own ~1.7-octave band (eta^2 ~= 6%, indistinguishable from
    noise)."""
    curved = cfg.boundaries.curved
    octaves = np.log2(curved.l_max / curved.l_min)
    assert octaves >= 5.0, (
        f"{octaves:.2f} octaves does not reach the ~5.5-octave, "
        "large-effect-size regime this example is specifically sized for"
    )
    # Sanity: matches the file's own commented-out expectation (5.50).
    assert octaves == pytest.approx(5.50, abs=0.05)


def test_50nm_example_amplitude_under_reference_wavelength_ceiling(cfg):
    """The file's own `amplitude` (A0, reference_wavelength convention)
    sits safely under BOTH (a) the closed-form reference-octave ceiling
    Rule 29(d) checks at resolve time and (b) the raw sup-norm
    seed-containment ceiling a_max -- confirms the YAML's own amplitude
    comment block's arithmetic against the actual installed code, not
    just against resolve_config's silence (which only proves it did not
    exceed the ceiling, not that the file's STATED derivation was
    correct)."""
    curved = cfg.boundaries.curved
    L = float(cfg.box.lengths[0])
    n = cfg.grains.number
    r_ws = wigner_seitz_radius(L ** 3, n)
    a_max = PERTURBED_DISTANCE_SAFETY * r_ws / (2.0 * ETA_CLIP)
    kappa = reference_shell_kappa(curved.hurst, curved.l_min, curved.l_max,
                                  curved.reference_wavelength)
    a0_max = a_max * kappa

    assert curved.amplitude < a_max
    assert curved.amplitude < a0_max
    # Matches this file's own comment ("~90% of A0_max"), within a loose
    # tolerance (the file rounds its stated constants to a few sig figs).
    assert curved.amplitude / a0_max == pytest.approx(0.90, abs=0.02)


def test_50nm_example_l_min_is_two_nearest_neighbor_distances(cfg):
    """l_min = 2*d_nn for this file's own Vegard PdAu lattice parameter --
    the atomic-resolution floor the header comment claims, not an
    arbitrary round number."""
    curved = cfg.boundaries.curved
    a_lattice = cfg.crystal.lattice.a
    d_nn = a_lattice / np.sqrt(2.0)  # FCC nearest-neighbor distance
    assert curved.l_min == pytest.approx(2.0 * d_nn, rel=1e-3)


def test_50nm_example_not_exercised_by_tessellation_construction():
    """Documents (rather than merely asserting by omission) that this
    test module never IMPORTS ``PerturbedDistanceTessellation`` or
    ``grainsmith.pipeline`` -- the example's own G1-only validation
    contract (schema/resolve, no field synthesis, no tessellation, no
    fill). Checked against this module's actual import statements (not
    a whole-file substring scan, which would also flag this docstring's
    own prose) so the guard survives future comments that merely
    mention these names."""
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
            imported_names.update(
                f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = {"grainsmith.pipeline",
                 "grainsmith.tessellation.perturbed.PerturbedDistanceTessellation"}
    hit = forbidden & imported_names
    assert not hit, f"this G1-only test module must not import {hit}"
