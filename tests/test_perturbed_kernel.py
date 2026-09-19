"""Mandatory validation for the numba-fused perturbed_distance kernels
(``tessellation/perturbed.py``'s "Numba acceleration" section) — K1
``owns()``, K2 ``margin()``, K3 ``grain_of()``/``_candidates_home()``.
Covers:

* Randomized equality batteries (owns/margin/grain_of, over every
  geometry x spectrum x n_grains combination this file parametrizes,
  including a full K3 battery — not a smoke test — and the R5
  ``connectivity_check=True`` / nonzero-``reassigned_fraction`` combination).
* Engineered tie cases: R1 (periodic-boundary / grid-cell-boundary
  battery, both ``_interp_eta`` directly and ``margin()``'s min-image
  ``np.rint`` folding), R2 (high-entropy unclipped-precision field
  canary), R3 (the four-case same-color mirrored/symmetric tie
  construction, checked on both ``owns()`` and ``grain_of()``), R4
  (synthetic candidate lists bypassing ``query_ball_point`` — K1's
  home-absent case and K3's empty-row sentinel).
* A fast, always-run small-scale ``owns()`` fill-candidate-grid
  replay (mirrors ``tests/test_owns_kernel.py``'s
  ``test_fig2_curved_real_system_mask_equality`` pattern at fixture
  scale) plus the opt-in slow full replay of the shipped
  ``examples/advanced/adv_pdau_perturbed.yaml`` system, gated by
  ``GRAINSMITH_RUN_PERTURBED_FIG2_VALIDATION=1`` /
  ``@pytest.mark.perturbed_fig2_validation``.

The byte-identical ``GRAINSMITH_NO_NUMBA``/``--jobs``/``gb_curvature.csv``
checks that would naturally extend ``tests/test_end_to_end.py`` and
``tests/test_curvature.py`` are instead pinned by new, self-contained
tests at the bottom of THIS file, using the pipeline's own ``run()``
entry point on a small ``perturbed_distance`` config (the exact
amplitude/hurst/l_min/l_max values are test_curvature.py's own proven
``G21_WARN_BOUNDARIES_OVERRIDE``, duplicated here as data rather than
imported, to keep this file self-contained) — functionally equivalent
coverage, different file. Also not touched, for the same reason:
``tests/conftest.py`` (the ``perturbed_fig2_validation`` marker used
below is therefore unregistered — pytest will emit
``PytestUnknownMarkWarning`` for it, not a failure, since this repo does
not set ``--strict-markers``).
"""
from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

import grainsmith.tessellation.perturbed as perturbed_mod
from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY
from grainsmith.seeding import seed_grains, wigner_seitz_radius
from grainsmith.tessellation.perturbed import PerturbedDistanceTessellation

REPO_ROOT = Path(__file__).resolve().parent.parent
PDAU_1M_YAML = REPO_ROOT / "examples" / "advanced" / "adv_pdau_perturbed.yaml"


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


@contextlib.contextmanager
def _forced_backend(use_numba: bool):
    """Force ``perturbed_mod._USE_NUMBA`` for the duration of the block,
    then restore whatever it was (module-level dispatch flag, monkeypatch-
    able by design — see the module docstring's "Numba acceleration"
    section). Used instead of an outer ``owns_backend``-style pytest
    fixture parametrization so each tessellation is built ONCE per test
    and both paths are run directly on the same query batch: monkeypatch
    ``perturbed._USE_NUMBA`` to run both paths on the SAME random X."""
    saved = perturbed_mod._USE_NUMBA
    perturbed_mod._USE_NUMBA = use_numba
    try:
        yield
    finally:
        perturbed_mod._USE_NUMBA = saved


# ---------------------------------------------------------------------------
# Randomized equality batteries
# ---------------------------------------------------------------------------

_GEOMETRIES = [
    ("periodic", np.array([60.0, 60.0, 60.0]), [True, True, True]),
    ("slab_z", np.array([60.0, 60.0, 45.0]), [True, True, False]),
    ("two_free", np.array([55.0, 45.0, 45.0]), [True, False, False]),
]

_N_LEVELS = [("small", 6), ("moderate", 20)]  # small forces color reuse


def _build_tess(
    Lg: np.ndarray, periodic: list[bool], spectrum: str, n: int, *,
    amplitude_frac: float = 0.7, hurst: float = 0.7, grid_size: int = 16,
    geom_seed: int = 4, field_seed: int = 104,
    connectivity_check: bool = True, l_min: float = 8.0, l_max: float = 30.0,
    correlation_length: float = 15.0,
) -> tuple[PerturbedDistanceTessellation, np.ndarray]:
    """Deterministic small ``PerturbedDistanceTessellation`` fixture,
    generalized over geometry/spectrum/grain-count (test_perturbed.py's
    own ``_make``/``_geometry`` convention, parametrized further for this
    file's battery). ``l_min``/``l_max``/``grid_size`` defaults are
    test_perturbed.py's own proven-safe values (Nyquist floor
    ``l_min >= 2*h_field`` holds for every ``_GEOMETRIES`` box at
    ``grid_size=16``)."""
    seeds = seed_grains(n, Lg, periodic, _rng(geom_seed))
    msd = wigner_seitz_radius(float(np.prod(Lg)), n)
    a_max = PERTURBED_DISTANCE_SAFETY * msd / (2.0 * ETA_CLIP)
    tess = PerturbedDistanceTessellation(
        seeds, Lg, periodic, amplitude=amplitude_frac * a_max,
        min_seed_distance=msd, rng=_rng(field_seed), grid_size=grid_size,
        connectivity_check=connectivity_check, spectrum=spectrum,
        hurst=hurst, l_min=l_min, l_max=l_max,
        correlation_length=correlation_length)
    return tess, seeds


@pytest.mark.parametrize("n_label,n_grains", _N_LEVELS, ids=[lvl for lvl, _ in _N_LEVELS])
@pytest.mark.parametrize("spectrum", ["gaussian", "self_affine"])
@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_kernel_random_battery_owns_margin_grain_of(geom, spectrum, n_label, n_grains):
    """For every (geometry, spectrum, n_grains) combination,
    owns()/margin()/grain_of() must agree between the numba kernel and
    the NumPy reference — margin() to 0 ULP, not ``np.isclose`` (the
    determinism contract requires bit-identity) — on random in-box AND
    out-of-box query points
    (``owns()``'s own documented contract, exercised by ``fill_grain``'s
    unwrapped lattice points)."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    label, Lg, periodic = geom
    tess, seeds = _build_tess(Lg, periodic, spectrum, n_grains)
    rng = _rng(2024)
    X_in = rng.uniform(0.0, 1.0, size=(250, 3)) * Lg
    X_out = rng.uniform(-0.2, 1.2, size=(60, 3)) * Lg
    X = np.concatenate([X_in, X_out], axis=0)

    with _forced_backend(False):
        grain_of_ref = tess.grain_of(X)
        owns_ref = {i: tess.owns(X, i) for i in range(tess.n_grains)}
        margin_ref = {i: tess.margin(X, i) for i in range(tess.n_grains)}
    with _forced_backend(True):
        grain_of_got = tess.grain_of(X)
        owns_got = {i: tess.owns(X, i) for i in range(tess.n_grains)}
        margin_got = {i: tess.margin(X, i) for i in range(tess.n_grains)}

    np.testing.assert_array_equal(
        grain_of_got, grain_of_ref,
        err_msg=f"grain_of mismatch {label}/{spectrum}/{n_label}")
    for i in range(tess.n_grains):
        np.testing.assert_array_equal(
            owns_got[i], owns_ref[i],
            err_msg=f"owns mismatch {label}/{spectrum}/{n_label} grain {i}")
        np.testing.assert_array_equal(
            margin_got[i], margin_ref[i],
            err_msg=f"margin mismatch (0 ULP) {label}/{spectrum}/{n_label} grain {i}")


def test_r5_connectivity_check_nonzero_reassignment_battery():
    """R5: at least one battery
    run must have ``connectivity_check=True`` on a configuration whose G5
    repair touches a NONZERO fraction of voxels, so the analytic-then-
    override two-stage structure (``_apply_override``/
    ``_apply_override_owns``, applied AFTER the kernel result)
    is exercised on the repaired branch, not only the (measured
    far more common) untouched branch. Reuses the exact geometry/
    amplitude/hurst/grid_size combination
    tests/test_perturbed.py::test_repair_override_consistent_with_voxel_labels
    already proved triggers repair."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    Lg = np.array([60.0, 60.0, 60.0])
    periodic = [True, True, True]
    tess, seeds = _build_tess(Lg, periodic, "self_affine", 6,
                              amplitude_frac=1.0, hurst=0.5, grid_size=20)
    assert tess.reassigned_fraction > 0.0, (
        "fixture no longer triggers G5 repair -- tune amplitude/hurst/"
        "grid_size until reassigned_fraction > 0 (the required nonzero-"
        "reassigned_fraction connectivity-check case)")

    rng = _rng(4242)
    X = np.concatenate([
        rng.uniform(0.0, 1.0, size=(300, 3)) * Lg,
        rng.uniform(-0.2, 1.2, size=(80, 3)) * Lg,
    ])
    with _forced_backend(False):
        grain_of_ref = tess.grain_of(X)
        owns_ref = {i: tess.owns(X, i) for i in range(tess.n_grains)}
    with _forced_backend(True):
        grain_of_got = tess.grain_of(X)
        owns_got = {i: tess.owns(X, i) for i in range(tess.n_grains)}

    np.testing.assert_array_equal(grain_of_got, grain_of_ref)
    for i in range(tess.n_grains):
        np.testing.assert_array_equal(owns_got[i], owns_ref[i])


# ---------------------------------------------------------------------------
# Engineered tie cases
# ---------------------------------------------------------------------------


def test_r1_interp_eta_periodic_and_grid_cell_boundary_battery():
    """R1: numba's float64 ``%``/``floor`` at periodic-wrap and grid-cell
    boundary values must reproduce ``_interpolate_scalar_field`` bit-for-
    bit — ``_interp_eta`` checked directly, bypassing candidate search
    entirely, at these exact edge values: ``0``, ``L``, ``-1e-12``,
    ``L+1e-12``, and several ``k*L/N`` grid-cell boundaries."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    Lg = np.array([60.0, 60.0, 48.0])
    periodic = [True, True, False]
    tess, _ = _build_tess(Lg, periodic, "self_affine", 6)
    field = tess.fields[0]
    Nx, Ny, Nz = tess._grid_shape
    grid_shape = tess._grid_shape

    def boundary_vals(L_ax, N_ax):
        vals = [0.0, L_ax, -1e-12, L_ax + 1e-12]
        for k in range(0, N_ax + 1, max(1, N_ax // 4)):
            vals.append(k * L_ax / N_ax)
        return np.array(vals, dtype=np.float64)

    bx = boundary_vals(Lg[0], Nx)
    by = boundary_vals(Lg[1], Ny)
    bz = boundary_vals(Lg[2], Nz)

    rng = _rng(555)
    n = 400
    X = np.stack([
        rng.choice(bx, size=n), rng.choice(by, size=n), rng.choice(bz, size=n),
    ], axis=1)

    ref = perturbed_mod._interpolate_scalar_field(field, X, Lg, periodic, grid_shape)
    got = np.array([
        perturbed_mod._interp_eta(
            float(X[k, 0]), float(X[k, 1]), float(X[k, 2]), field,
            Nx, Ny, Nz, Lg[0], Lg[1], Lg[2],
            periodic[0], periodic[1], periodic[2])
        for k in range(n)
    ])
    np.testing.assert_array_equal(got, ref)


def test_r1_margin_min_image_half_l_np_rint_tie():
    """R1's ``margin()``-specific half: the min-image ``np.rint`` fold at
    ``X = seeds[i] + L/2`` exactly (the true round-half-to-even tie
    point on a periodic axis) must agree between backends — locks in
    ``np.rint`` (not the builtin ``round()``) behaving
    like ``np.round`` there."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    Lg = np.array([60.0, 60.0, 45.0])
    periodic = [True, True, False]
    tess, seeds = _build_tess(Lg, periodic, "self_affine", 6)
    i = 0
    for ax in range(3):
        if not periodic[ax]:
            continue
        X = seeds[i].copy()[None, :]
        X[0, ax] += Lg[ax] / 2.0
        with _forced_backend(False):
            ref = tess.margin(X, i)
        with _forced_backend(True):
            got = tess.margin(X, i)
        assert got[0] == ref[0], f"axis {ax} half-L tie mismatch"


def test_r2_float32_widening_high_entropy_field():
    """R2: the float32-field -> float64-math widening must happen AT THE
    READ, not via an accumulator that stays float32.
    A deliberately high-entropy, unclipped-precision synthetic field
    (many significant bits, not the smooth low-frequency content a real
    synthesized field would carry) is the sharpest canary for this —
    ``_interp_eta`` vs ``_interpolate_scalar_field`` directly, exact
    equality, no tessellation needed."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = _rng(321)
    Nx, Ny, Nz = 12, 14, 10
    field = rng.uniform(-ETA_CLIP, ETA_CLIP, size=(Nx, Ny, Nz)).astype(np.float32)
    Lg = np.array([40.0, 44.0, 36.0])
    periodic = [True, True, False]
    grid_shape = (Nx, Ny, Nz)
    X = rng.uniform(-0.1, 1.1, size=(3000, 3)) * Lg

    ref = perturbed_mod._interpolate_scalar_field(field, X, Lg, periodic, grid_shape)
    got = np.array([
        perturbed_mod._interp_eta(
            float(X[k, 0]), float(X[k, 1]), float(X[k, 2]), field,
            Nx, Ny, Nz, Lg[0], Lg[1], Lg[2],
            periodic[0], periodic[1], periodic[2])
        for k in range(len(X))
    ])
    np.testing.assert_array_equal(got, ref)


def _r3_fixture():
    """Four hand-built seeds in a periodic 40 Å box, engineered so grain
    0's HOME replica and its own periodic image along z tie exactly at
    two query points (mechanism 1: same grain -> same
    color -> bit-identical eta; symmetric coordinates -> bit-identical
    draw), one with the tied image's replica index BELOW home_idx
    (z-shift -1) and one ABOVE (z-shift +1). Used for the mech1_below/
    mech1_above cases only -- grains 2/3 are irrelevant to those two
    query points ([10,20,0] and [10,20,40], far from grains 2/3's seeds
    at y=10/y=30) and are kept here only as harmless extra candidates
    matching the original four-grain layout.

    Built with ``connectivity_check=False`` (no G5 override
    array to mask the analytic tie result under test).
    """
    Lg = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = np.array([
        [10.0, 20.0, 20.0],
        [30.0, 20.0, 20.0],
        [20.0, 10.0, 20.0],
        [20.0, 30.0, 20.0],
    ])
    tess = PerturbedDistanceTessellation(
        seeds, Lg, periodic, amplitude=0.7 * PERTURBED_DISTANCE_SAFETY
        * 14.0 / (2.0 * ETA_CLIP),
        min_seed_distance=14.0, rng=_rng(707), grid_size=12,
        connectivity_check=False, spectrum="self_affine", hurst=0.6,
        l_min=8.0, l_max=15.0)
    # Force grains 0 and 1 onto the same color (the sanctioned
    # direct-override mechanism) -- independent of whatever the greedy
    # graph coloring happened to assign them.
    tess._color_of_grain[1] = tess._color_of_grain[0]
    return tess, seeds


def _r3_fixture_mech2():
    """Two hand-built seeds (mechanism 2 only), engineered so grains 0
    and 1 -- forced onto the SAME color by direct ``_color_of_grain``
    override, the sanctioned mechanism -- tie exactly at the
    box's x-midpoint AND that tie is the ROW MINIMUM: querying
    ``owns(X, 1)``/``grain_of`` puts the tied competitor (grain 0)
    BELOW home_idx, querying ``owns(X, 0)`` puts it ABOVE.

    Fix-round note: the original four-grain ``_r3_fixture`` used this
    same query point with all four seeds present, but grain 2 (seed
    [20,10,20], equidistant from the box center at [20,20,20]) STRICTLY
    WON that row via a lower field-perturbed value, so the intended
    grain-0/grain-1 tie never reached the row minimum -- the mech2 R3
    cases were backend-vs-backend equal for a reason unrelated to the
    tie-break rule under test (grain_of() always returned gid 2).
    Dropping grains 2 and 3 entirely (rather than moving them outside
    the candidate search radius) removes every other candidate that
    could beat the tie, so the mech2 cases now actually exercise K3's
    tie-break rule.

    Built with ``connectivity_check=False`` (no G5 override
    array to mask the analytic tie result under test).
    """
    Lg = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = np.array([
        [10.0, 20.0, 20.0],
        [30.0, 20.0, 20.0],
    ])
    tess = PerturbedDistanceTessellation(
        seeds, Lg, periodic, amplitude=0.7 * PERTURBED_DISTANCE_SAFETY
        * 14.0 / (2.0 * ETA_CLIP),
        min_seed_distance=14.0, rng=_rng(707), grid_size=12,
        connectivity_check=False, spectrum="self_affine", hurst=0.6,
        l_min=8.0, l_max=15.0)
    # Force grains 0 and 1 onto the same color (the sanctioned
    # direct-override mechanism) -- independent of whatever the greedy
    # graph coloring happened to assign them.
    tess._color_of_grain[1] = tess._color_of_grain[0]
    return tess, seeds


@pytest.mark.parametrize("case", ["mech1_below", "mech1_above",
                                  "mech2_below", "mech2_above"])
def test_r3_engineered_tie_four_cases(case):
    """R3: the four {mirrored-position, symmetric-layout} x {competitor
    replica index below home_idx, above home_idx} engineered ties,
    asserting ``owns()`` AND ``grain_of()``
    agree between the numba kernel and NumPy at the EXACT tie point —
    backend-vs-backend equality, not a hand-predicted winner -- PLUS,
    for the mech2 cases, a
    direct pin of the winning gid (fix-round addition, see
    ``_r3_fixture_mech2``'s docstring: the mech2 tie must actually be
    the row minimum for this test to mean anything, and pinning the
    winner is how that is confirmed rather than assumed)."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    expected_winner = None

    if case == "mech1_below":
        tess, seeds = _r3_fixture()
        # grain 0's home vs its OWN z-shift-(-1) periodic image: midpoint
        # z = (20 + (20-40))/2 = 0 -- exact tie, image replica index <
        # home_idx (block (0,0,-1) sorts before identity (0,0,0)).
        X = np.array([[10.0, 20.0, 0.0]])
        home_i = 0
    elif case == "mech1_above":
        tess, seeds = _r3_fixture()
        # z-shift (+1) image: midpoint z = (20 + (20+40))/2 = 40 -- exact
        # tie, image replica index > home_idx.
        X = np.array([[10.0, 20.0, 40.0]])
        home_i = 0
    elif case == "mech2_below":
        tess, seeds = _r3_fixture_mech2()
        # grains 0 (idx0) / 1 (idx1) forced same color, symmetric about
        # x=20 -- querying grain 1 puts the tied competitor (grain 0,
        # idx0 < idx1) BELOW home_idx. Lower gid (0) wins the tie.
        X = np.array([[20.0, 20.0, 20.0]])
        home_i = 1
        expected_winner = 0
    else:  # mech2_above
        tess, seeds = _r3_fixture_mech2()
        # Querying grain 0 puts competitor grain 1 (idx1 > idx0) ABOVE
        # home_idx. Lower gid (0) still wins the tie.
        X = np.array([[20.0, 20.0, 20.0]])
        home_i = 0
        expected_winner = 0

    with _forced_backend(False):
        owns_ref = bool(tess.owns(X, home_i)[0])
        grain_of_ref = int(tess.grain_of(X)[0])
    with _forced_backend(True):
        owns_got = bool(tess.owns(X, home_i)[0])
        grain_of_got = int(tess.grain_of(X)[0])

    assert owns_got == owns_ref, f"{case}: owns() tie-break diverges"
    assert grain_of_got == grain_of_ref, f"{case}: grain_of() tie-break diverges"

    if expected_winner is not None:
        assert grain_of_ref == expected_winner, (
            f"{case}: fixture no longer produces a row-minimum tie -- "
            f"expected grain_of()=={expected_winner} (lower gid wins "
            f"K3's first-minimum-wins tie rule), got "
            f"{grain_of_ref} instead (another candidate is winning the "
            "row outright, so this case is not testing the tie-break "
            "rule)")


def test_r4_k1_home_absent_from_candidate_list_returns_false():
    """R4 (K1): a synthetic candidate list that deliberately EXCLUDES
    ``home_idx`` (bypassing ``query_ball_point`` entirely) must make the
    kernel return False for that row regardless of the other candidates'
    values (``seen_home`` correctness, not an optimization)."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    Lg, periodic = _GEOMETRIES[0][1], _GEOMETRIES[0][2]
    tess, seeds = _build_tess(Lg, periodic, "self_affine", 6,
                              connectivity_check=False)
    i = 0
    home_idx = tess._identity_block * tess._n + i
    other_idx = 0 if home_idx != 0 else 1
    X = np.array([[seeds[i, 0] + 0.1, seeds[i, 1], seeds[i, 2]]])
    offsets = np.array([0, 1], dtype=np.int64)
    col_flat = np.array([other_idx], dtype=np.int64)

    Nx, Ny, Nz = tess._grid_shape
    per_x, per_y, per_z = periodic
    Lx, Ly, Lz = Lg
    got = perturbed_mod._owns_kernel_perturbed(
        X, offsets, col_flat, tess._rep_seeds, tess._color_of_grain,
        tess._fields, Nx, Ny, Nz, Lx, Ly, Lz, per_x, per_y, per_z,
        tess._amplitude, tess._n, home_idx)
    assert bool(got[0]) is False


def test_r4_k3_grain_of_empty_candidate_row_sentinel():
    """R4 (K3): a synthetic batch with one EMPTY candidate row (never
    occurs via real ``query_ball_point`` output) must return the ``-1``
    sentinel for that row, while every non-empty row in the SAME batch
    still matches the NumPy reference exactly."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    Lg, periodic = _GEOMETRIES[0][1], _GEOMETRIES[0][2]
    tess, seeds = _build_tess(Lg, periodic, "self_affine", 6,
                              connectivity_check=False)
    rng = _rng(9001)
    X = rng.uniform(0.0, 1.0, size=(3, 3)) * Lg

    A = tess._amplitude
    search_pad = 2.0 * A * ETA_CLIP + 1e-9
    d1, _ = tess._home_tree_query(X)
    radius = d1 + search_pad
    cand = tess._tree.query_ball_point(X, r=radius)
    offsets, col_flat = tess._offsets_and_col(cand)

    # Doctor row 1 to be empty: splice its slice out of col_flat and
    # collapse the offsets accordingly; row 0's and row 2's own slices
    # (the actual candidate content) are otherwise untouched.
    o0, o1, o2, o3 = (int(offsets[0]), int(offsets[1]),
                      int(offsets[2]), int(offsets[3]))
    doctored_col = np.concatenate([col_flat[o0:o1], col_flat[o2:o3]])
    row2_len = o3 - o2
    doctored_offsets = np.array([0, o1, o1, o1 + row2_len], dtype=np.int64)

    Nx, Ny, Nz = tess._grid_shape
    per_x, per_y, per_z = periodic
    Lx, Ly, Lz = Lg
    got = perturbed_mod._grain_of_kernel_perturbed(
        X, doctored_offsets, doctored_col, tess._rep_seeds,
        tess._color_of_grain, tess._fields, Nx, Ny, Nz, Lx, Ly, Lz,
        per_x, per_y, per_z, A, tess._n)

    assert int(got[1]) == -1
    with _forced_backend(False):
        ref0 = int(tess.grain_of(X[0:1])[0])
        ref2 = int(tess.grain_of(X[2:3])[0])
    assert int(got[0]) == ref0
    assert int(got[2]) == ref2


# ---------------------------------------------------------------------------
# Fill-candidate-grid replay
# ---------------------------------------------------------------------------


class _CompareOwnsBothPaths:
    """Drop-in for ``tess`` inside ``fill_grain``: on every ``owns()``
    call (one per fill.py chunk), compute BOTH the numba and NumPy mask
    on the SAME candidate chunk, tally any mismatch, and return the
    numba mask (production behavior). Same pattern as
    ``tests/test_owns_kernel.py``'s ``_CompareBothPaths``."""

    def __init__(self, inner):
        self._inner = inner
        self.total_candidates = 0
        self.total_mismatches = 0
        self.mismatch_examples: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cell_vertices_rel(self, i):
        cvr = getattr(self._inner, "cell_vertices_rel", None)
        return cvr(i) if cvr is not None else None

    def owns(self, X, i):
        perturbed_mod._USE_NUMBA = False
        numpy_mask = self._inner.owns(X, i)
        perturbed_mod._USE_NUMBA = True
        numba_mask = self._inner.owns(X, i)
        diff = numpy_mask != numba_mask
        n_diff = int(diff.sum())
        self.total_candidates += len(X)
        if n_diff:
            self.total_mismatches += n_diff
            idx = np.where(diff)[0][:3]
            self.mismatch_examples.append(
                f"grain {i}: {n_diff} mismatches, e.g. X[{idx}]={X[idx]}")
        return numba_mask


def _fcc_cu_crystal():
    from grainsmith.crystal.cell import cell_matrix
    from grainsmith.crystal.spacegroup import (
        WyckoffSite, expand_wyckoff, hall_from_international, symmetry_ops,
    )
    A = cell_matrix(3.615, 3.615, 3.615, 90.0, 90.0, 90.0)
    hall = hall_from_international(225)
    rots, trans = symmetry_ops(hall)
    sites = [WyckoffSite("Cu", [0.0, 0.0, 0.0])]
    basis = expand_wyckoff(sites, rots, trans)
    return A, basis


@pytest.mark.parametrize("spectrum", ["gaussian", "self_affine"])
@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_kernel_owns_fill_candidate_grid_replay_fast(geom, spectrum):
    """Fast, always-run replay: a small purpose-built
    perturbed_distance tessellation (this file's own ``_GEOMETRIES``
    scale — the same order of magnitude as test_perturbed.py's own
    fixtures), replayed through ``fill_grain``'s OWN candidate-grid
    construction (not a synthetic battery) — every ``owns()`` call
    fill.py makes is intercepted and checked numba-vs-NumPy on every
    real candidate, no sampling. Covers the same spectrum/periodicity
    crossings the randomized battery above does; the expensive
    ~948k-atom full replay is the opt-in marker below."""
    from grainsmith.atoms.fill import fill_grain

    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    label, Lg, periodic = geom
    tess, seeds = _build_tess(Lg, periodic, spectrum, 5, amplitude_frac=0.6,
                              geom_seed=13, field_seed=205)
    A_mat, basis = _fcc_cu_crystal()
    q_id = np.array([1.0, 0.0, 0.0, 0.0])

    saved = perturbed_mod._USE_NUMBA
    comparator = _CompareOwnsBothPaths(tess)
    try:
        for gi in range(tess.n_grains):
            fill_grain(gi, comparator, basis.frac, basis.species,
                      basis.occupancy, A_mat, q_id, Lg, periodic,
                      _rng(300 + gi), store_margin=False, a_clip=0.0)
    finally:
        perturbed_mod._USE_NUMBA = saved

    assert comparator.total_candidates > 0
    assert comparator.total_mismatches == 0, (
        f"{comparator.total_mismatches}/{comparator.total_candidates} "
        f"owns() candidates mismatch on the small {label}/{spectrum} fill "
        "replay:\n" + "\n".join(comparator.mismatch_examples))


def _perturbed_fig2_validation_enabled() -> bool:
    return os.environ.get("GRAINSMITH_RUN_PERTURBED_FIG2_VALIDATION", "").strip() in (
        "1", "true", "yes", "on")


@pytest.mark.perturbed_fig2_validation
@pytest.mark.skipif(not _perturbed_fig2_validation_enabled(),
                    reason="expensive (~948k atoms, ~100:1+ candidate:kept "
                          "ratio); opt in with "
                          "GRAINSMITH_RUN_PERTURBED_FIG2_VALIDATION=1")
def test_perturbed_1m_real_system_owns_mask_equality():
    """Full opt-in replay: build the EXACT shipped
    ``examples/advanced/adv_pdau_perturbed.yaml`` tessellation, replay
    ``fill_grain``'s own candidate-grid construction for every grain, and
    assert the numba mask equals the NumPy mask on EVERY single candidate
    (no sampling). A mismatch here is a stop-ship signal for the kernel,
    same standard as tests/test_owns_kernel.py's fig2_validation marker."""
    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    assert PDAU_1M_YAML.is_file(), f"missing {PDAU_1M_YAML}"

    from grainsmith.atoms.fill import fill_grain
    from grainsmith.config.resolve import load_config
    from grainsmith.pipeline import (
        _build_crystal,
        _resolve_min_seed_distance,
        _stage_orientation,
        _stage_seeding,
        _stage_tessellation,
    )
    from grainsmith.rng import make_rng

    config = load_config(PDAU_1M_YAML)
    assert config.boundaries.geometry == "curved"
    assert config.boundaries.curved.method == "perturbed_distance"

    seed_val = int(config.seed.value)
    rng_bundle = make_rng(seed_val)
    msd = _resolve_min_seed_distance(config)
    seeds = _stage_seeding(config, msd, rng_bundle.seeding)
    tess, _ = _stage_tessellation(config, seeds, msd, rng_bundle.fields,
                                  rng_sizes=rng_bundle.sizes)
    assert isinstance(tess, PerturbedDistanceTessellation)
    crystal = _build_crystal(config.crystal, config.output.lammps.masses)
    quats = _stage_orientation(config, tess.n_grains, crystal.A,
                               rng_bundle.orientation)
    occ_streams = rng_bundle.occupancy_streams(tess.n_grains)

    L = np.asarray(config.box.lengths, dtype=np.float64)
    periodic = config.box.periodic

    comparator = _CompareOwnsBothPaths(tess)
    saved = perturbed_mod._USE_NUMBA
    t0 = time.perf_counter()
    try:
        for gi in range(tess.n_grains):
            tg0 = time.perf_counter()
            fill_grain(gi, comparator, crystal.basis.frac,
                      crystal.basis.species, crystal.basis.occupancy,
                      crystal.A, quats[gi], L, periodic, occ_streams[gi],
                      store_margin=False, a_clip=0.0)
            print(f"[perturbed_fig2_validation] grain {gi}/"
                 f"{tess.n_grains - 1} done in "
                 f"{time.perf_counter() - tg0:.1f} s; running totals: "
                 f"{comparator.total_candidates:,} candidates, "
                 f"{comparator.total_mismatches} mismatches so far.",
                 flush=True)
    finally:
        perturbed_mod._USE_NUMBA = saved
    elapsed = time.perf_counter() - t0

    print(f"[perturbed_fig2_validation] {comparator.total_candidates:,} "
         f"candidates checked across {tess.n_grains} grains in "
         f"{elapsed:.1f} s; {comparator.total_mismatches} mismatches.")
    assert comparator.total_mismatches == 0, (
        f"{comparator.total_mismatches} / {comparator.total_candidates} "
        "candidates mismatch between numba and numpy on the shipped "
        f"adv_pdau_perturbed system:\n"
        + "\n".join(comparator.mismatch_examples))


# ---------------------------------------------------------------------------
# e2e byte-identity checks (see module docstring) kept local to this file
# rather than extending tests/test_end_to_end.py / tests/test_curvature.py.
# ---------------------------------------------------------------------------

_TS = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

# Same amplitude/hurst/l_min/l_max as tests/test_curvature.py's own
# G21_WARN_BOUNDARIES_OVERRIDE (proven to build/run cleanly at this
# grains/box scale); duplicated here as plain data to keep this file
# self-contained.
_E2E_BOUNDARIES_OVERRIDE = {
    "geometry": "curved",
    "curved": {"method": "perturbed_distance", "amplitude": 1.0,
              "spectrum": "self_affine", "hurst": 0.9,
              "l_min": 5.0, "l_max": 13.0},
}


def _e2e_config(outdir):
    from grainsmith.config.resolve import resolve_config
    raw = {
        "meta": {"title": "perturbed_kernel_e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": 20260823},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": _E2E_BOUNDARIES_OVERRIDE,
        "analysis": {"gb_curvature": True},
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    return resolve_config(raw)


def test_e2e_numba_vs_numpy_byte_identical(tmp_path):
    """A full ``perturbed_distance`` pipeline run with
    the numba kernels active must reproduce the exact same atoms
    (positions/species/grain labels) and the exact same
    ``gb_curvature.csv`` bytes (no provenance header, so no timestamp/
    config-hash normalization needed — unlike ``polycrystal.data``,
    whose provenance line embeds a config-sha that differs across
    different ``output.directory`` values, so atoms are compared
    directly rather than via a raw file-byte diff, mirroring
    tests/test_end_to_end.py::test_jobs_parallel_pipeline_bit_identical's
    own reasoning) as ``GRAINSMITH_NO_NUMBA=1`` would (here: the
    module flag forced False directly, same escape hatch, in-process)."""
    from grainsmith.pipeline import run

    if not perturbed_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = perturbed_mod._USE_NUMBA
    try:
        perturbed_mod._USE_NUMBA = True
        res_a = run(_e2e_config(tmp_path / "a"))
        perturbed_mod._USE_NUMBA = False
        res_b = run(_e2e_config(tmp_path / "b"))
    finally:
        perturbed_mod._USE_NUMBA = saved

    assert res_a.gates.all_passed() and res_b.gates.all_passed()
    np.testing.assert_array_equal(res_a.atoms.pos, res_b.atoms.pos)
    assert np.array_equal(res_a.atoms.species, res_b.atoms.species)
    assert np.array_equal(res_a.atoms.grain, res_b.atoms.grain)
    assert ((tmp_path / "a" / "gb_curvature.csv").read_bytes()
            == (tmp_path / "b" / "gb_curvature.csv").read_bytes())


def test_e2e_jobs_1_vs_2_byte_identical(tmp_path):
    """``--jobs``: the same
    perturbed_distance/gb_curvature config must produce bit-identical
    atoms and gb_curvature.csv bytes at jobs=1 vs jobs=2 (§13: --jobs is
    an execution detail, not physics) — under whichever kernel path is
    currently active (default: numba, if installed)."""
    from grainsmith.pipeline import run

    res_a = run(_e2e_config(tmp_path / "j1"), jobs=1)
    res_b = run(_e2e_config(tmp_path / "j2"), jobs=2)
    assert res_a.gates.all_passed() and res_b.gates.all_passed()
    np.testing.assert_array_equal(res_a.atoms.pos, res_b.atoms.pos)
    assert np.array_equal(res_a.atoms.species, res_b.atoms.species)
    assert np.array_equal(res_a.atoms.grain, res_b.atoms.grain)
    assert ((tmp_path / "j1" / "gb_curvature.csv").read_bytes()
            == (tmp_path / "j2" / "gb_curvature.csv").read_bytes())
