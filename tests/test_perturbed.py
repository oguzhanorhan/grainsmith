"""Tests for tessellation/perturbed.py (§6.6 extension M6:
``perturbed_distance`` level-set self-affine grain-boundary backend).

Companion to test_warp.py / test_weighted.py / test_self_affine.py — same
fixture conventions (fixed PCG64 seeds via ``_rng``, small boxes/grids for
speed). Pins the Phase-1 contracts documented in the design note
(design_note_perturbed.md, agent workspace) and the module docstring of
``tessellation/perturbed.py``:

* determinism (same seed -> bit-identical grain_of)
* A=0 reduces to FlatTessellation exactly
* G5 repair-and-report leaves every grain a single periodic component
* D_b responds to the Hurst exponent in the documented direction
  (lower H -> rougher boundary -> higher box-counting D_b)
* the two ConfigError guard paths (pre-flight amplitude ceiling, §4.2;
  post-hoc exact seed-ownership, §4.3a) and the shared Rule 20 band
  checks (l_min < l_max, l_min >= 2*h_field)
* RNG consumption is a function of (seeds, box, l_min, l_max, grid) only
  -- NOT of hurst or amplitude (design note §10)

All builds here use small boxes/grids (<=32 per axis) specifically so this
file adds a few seconds, not minutes, to the suite total.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from grainsmith.config.resolve import resolve_config
from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY
from grainsmith.errors import ConfigError, TessellationError
from grainsmith.seeding import seed_grains, wigner_seitz_radius
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.perturbed import (
    PerturbedDistanceTessellation,
    _find_fragment_mask,
    box_count_dimension,
)

PER = [True, True, True]


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _geometry(n: int = 6, L: float = 60.0, seed: int = 4):
    """Deterministic small periodic box + seeds + the method's own
    seed-containment ceiling A_max for that geometry."""
    box = np.array([L, L, L])
    seeds = seed_grains(n, box, PER, _rng(seed))
    msd = wigner_seitz_radius(float(L**3), n)
    a_max = PERTURBED_DISTANCE_SAFETY * msd / (2.0 * ETA_CLIP)
    return seeds, box, msd, a_max


def _make(
    n=6, L=60.0, geom_seed=4, amplitude_frac=0.6, field_seed=104,
    grid_size=16, spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
    connectivity_check=True,
):
    """Build a PerturbedDistanceTessellation on a small deterministic
    geometry; amplitude is specified as a fraction of that geometry's own
    seed-containment ceiling A_max (so callers never accidentally trip
    guard 1 by hardcoding an absolute amplitude against the wrong msd)."""
    seeds, box, msd, a_max = _geometry(n=n, L=L, seed=geom_seed)
    tess = PerturbedDistanceTessellation(
        seeds, box, PER,
        amplitude=amplitude_frac * a_max,
        min_seed_distance=msd,
        rng=_rng(field_seed),
        grid_size=grid_size,
        connectivity_check=connectivity_check,
        spectrum=spectrum, hurst=hurst, l_min=l_min, l_max=l_max,
    )
    return tess, seeds, box, msd, a_max


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_bit_identical_grain_of():
    """Same seed (geometry + field) -> bit-identical grain_of, fields, and
    graph coloring on a fresh construction (independent object instances,
    not a cached result)."""
    t1, *_ = _make()
    t2, *_ = _make()
    pts = _rng(999).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    np.testing.assert_array_equal(t1.grain_of(pts), t2.grain_of(pts))
    np.testing.assert_array_equal(t1._fields, t2._fields)
    np.testing.assert_array_equal(t1._color_of_grain, t2._color_of_grain)
    assert t1.reassigned_fraction == t2.reassigned_fraction


def test_determinism_owns_and_margin():
    """owns()/margin() are pure functions of the same deterministic state
    -- not just grain_of()."""
    t1, *_ = _make()
    t2, *_ = _make()
    pts = _rng(1000).uniform(0.0, 1.0, size=(200, 3)) * 60.0
    for i in range(t1.n_grains):
        np.testing.assert_array_equal(t1.owns(pts, i), t2.owns(pts, i))
        np.testing.assert_allclose(t1.margin(pts, i), t2.margin(pts, i))


# ---------------------------------------------------------------------------
# A=0 -> flat Voronoi (bijective limit, no diffeomorphism needed here since
# the assignment rule itself degenerates: perturbation term vanishes)
# ---------------------------------------------------------------------------


def test_amplitude_zero_equals_flat_grain_of():
    """A=0 collapses the level-set rule to plain Voronoi exactly (the
    perturbation term A*eta vanishes identically) -- grain_of matches
    FlatTessellation bit-for-bit on random query points."""
    seeds, box, msd, _ = _geometry()
    tess = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.0, min_seed_distance=msd,
        rng=_rng(104), grid_size=16, connectivity_check=True,
        spectrum="gaussian", correlation_length=15.0,
    )
    flat = FlatTessellation(seeds, box, PER)
    pts = _rng(55).uniform(0.0, 1.0, size=(400, 3)) * 60.0
    np.testing.assert_array_equal(tess.grain_of(pts), flat.grain_of(pts))


def test_amplitude_zero_equals_flat_owns():
    """Same A=0 reduction for owns() (the compact-cell fill contract)."""
    seeds, box, msd, _ = _geometry()
    tess = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.0, min_seed_distance=msd,
        rng=_rng(104), grid_size=16, connectivity_check=True,
        spectrum="gaussian", correlation_length=15.0,
    )
    flat = FlatTessellation(seeds, box, PER)
    pts = _rng(56).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    for i in range(tess.n_grains):
        np.testing.assert_array_equal(tess.owns(pts, i), flat.owns(pts, i))


# ---------------------------------------------------------------------------
# Membership-API contracts shared with weighted.py / warp.py (owns tiling,
# margin sign, bounding_radius coverage) -- the fill/overlap stages depend
# on these regardless of which curved backend produced the tessellation.
# ---------------------------------------------------------------------------


def test_owns_tiles_torus_exactly():
    """Exact-once compact-cell tiling over grains x {-1,0,1}^3 periodic
    lifts, at a non-trivial amplitude (§6.8 invariant)."""
    tess, seeds, box, *_ = _make(amplitude_frac=0.6, hurst=0.6, grid_size=20)
    pts = _rng(77).uniform(0.0, 1.0, size=(150, 3)) * 60.0
    owned = np.zeros(len(pts), dtype=int)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * box
                for i in range(tess.n_grains):
                    owned += tess.owns(lift, i).astype(int)
    assert np.all(owned == 1), f"tiling violated: min={owned.min()}, max={owned.max()}"


def test_margin_signed_and_grain_specific():
    """margin(x, i) >= 0 iff grain_of(x) == i (weighted.py's convention,
    reused verbatim here per the module docstring)."""
    tess, *_ = _make(amplitude_frac=0.6, hurst=0.6, grid_size=20)
    pts = _rng(77).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    gids = tess.grain_of(pts)
    for i in range(tess.n_grains):
        m = tess.margin(pts, i)
        assert np.all(np.isfinite(m))
        inside = gids == i
        assert np.all(m[inside] >= -1e-9)
        assert np.all(m[~inside] <= 1e-9)


def test_bounding_radius_covers_owned_points():
    """Every point owns() assigns to grain i lies within bounding_radius(i)
    of seed i, over all periodic lifts (§6.8 enumeration-cover guarantee
    the atom fill relies on)."""
    tess, seeds, box, *_ = _make(amplitude_frac=0.6, hurst=0.6, grid_size=20)
    pts = _rng(78).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * box
                for i in range(tess.n_grains):
                    mask = tess.owns(lift, i)
                    if not np.any(mask):
                        continue
                    d = np.linalg.norm(lift[mask] - seeds[i], axis=1)
                    assert float(np.max(d)) <= tess.bounding_radius(i) + 1e-9


# ---------------------------------------------------------------------------
# G5: periodic-connectivity repair-and-report
# ---------------------------------------------------------------------------


def test_repair_leaves_every_grain_a_single_periodic_component():
    """After construction (repair runs at construction time, §6 of the
    design note), every grain's voxel label is a SINGLE periodically-
    connected component -- checked directly via the same fragment-mask
    helper the repair loop itself uses as its convergence test, at the
    guard-ceiling amplitude where repair is most likely to be exercised."""
    tess, *_ = _make(amplitude_frac=1.0, hurst=0.5, grid_size=20)
    frag_mask = _find_fragment_mask(tess.voxel_grid.labels, tess.n_grains, PER)
    assert not np.any(frag_mask), (
        f"{int(frag_mask.sum())} voxel(s) remain fragmented post-construction"
    )


def test_repair_reassigned_fraction_reported_and_bounded():
    """reassigned_fraction is a finite, non-negative, plausibly-small QA
    metric (not a pass/fail gate) -- and 0.0 exactly when nothing needed
    repair (A=0, flat-Voronoi facets have no periodic-connectivity risk)."""
    seeds, box, msd, _ = _geometry(n=5, L=50.0, seed=11)
    flat_like = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.0, min_seed_distance=msd,
        rng=_rng(111), grid_size=16, connectivity_check=True,
        spectrum="gaussian", correlation_length=15.0,
    )
    assert flat_like.reassigned_fraction == 0.0

    tess, *_ = _make(amplitude_frac=1.0, hurst=0.5, grid_size=20)
    frac = tess.reassigned_fraction
    assert np.isfinite(frac) and 0.0 <= frac < 0.15  # task's plausibility ceiling


def test_repair_override_consistent_with_voxel_labels():
    """Where repair changed a voxel's label, grain_of() at that voxel's
    CENTER must return the repaired label (the override the fill stage
    actually reads, not a cosmetic side channel -- design note §6)."""
    tess, seeds, box, *_ = _make(amplitude_frac=1.0, hurst=0.5, grid_size=20)
    if tess._override_labels is None:
        pytest.skip("no repair occurred at this amplitude/seed (nothing to check)")
    vg = tess.voxel_grid
    touched_idx = np.argwhere(tess._touched_mask)
    assert len(touched_idx) > 0
    # Sample a handful of touched voxel centers and confirm grain_of agrees
    # with the stored (repaired) label there.
    sample = touched_idx[:: max(1, len(touched_idx) // 20)][:20]
    centers = (sample + 0.5) * vg.h_vec
    got = tess.grain_of(centers)
    expected = vg.labels[sample[:, 0], sample[:, 1], sample[:, 2]]
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# D_b(H) direction: rougher (lower H) -> higher box-counting dimension.
#
# Fixed small box/grid, averaged over 3 field seeds at ONE grain geometry
# to damp single-realization box-counting noise (per-grain/section D_b
# has visible seed-to-seed scatter on a handful of grains -- design note
# §7/§11 open risk 4) while staying well under a few seconds per build.
# ---------------------------------------------------------------------------

_DBH_L = 150.0
_DBH_N = 24
_DBH_GRID = 32
_DBH_H_FIELD = _DBH_L / _DBH_GRID          # 4.6875 Å
_DBH_L_MIN = 2.3 * _DBH_H_FIELD            # 10.78 Å (> 2*h_field Nyquist floor)
_DBH_L_MAX = 4.0 * _DBH_L_MIN              # 43.1 Å (< min(L)/2 = 75 Å cap)
_DBH_FIELD_SEEDS = (31, 32, 33)


def _dbh_ensemble(hurst: float) -> list[float]:
    seeds, box, msd, a_max = _geometry(n=_DBH_N, L=_DBH_L, seed=4)
    out = []
    for fs in _DBH_FIELD_SEEDS:
        tess = PerturbedDistanceTessellation(
            seeds, box, PER, amplitude=a_max, min_seed_distance=msd,
            rng=_rng(fs), grid_size=_DBH_GRID, connectivity_check=True,
            spectrum="self_affine", hurst=hurst,
            l_min=_DBH_L_MIN, l_max=_DBH_L_MAX,
        )
        db = tess.d_b_estimate(k_largest=_DBH_N, n_sections=3)
        assert db is not None
        out.append(db)
    return out


def test_db_monotonic_in_hurst_direction():
    """D_b(H=0.3) > D_b(H=0.95) on average (rougher, lower-H field ->
    higher box-counting dimension; design note §1/§7, back-to-back with
    the 2D proof-of-concept H=0.9 -> D_b~=1.11, H=0.5 -> D_b~=1.20 trend).
    Compared as an ensemble MEAN over 3 field seeds at fixed geometry
    (single-realization D_b is noisy enough that a pairwise same-seed
    comparison occasionally reverses -- see design note open risk 4);
    the mean-direction test is the one that is actually robust."""
    db_lo = _dbh_ensemble(0.3)
    db_hi = _dbh_ensemble(0.95)
    assert np.mean(db_lo) > np.mean(db_hi), (
        f"D_b(H=0.3) mean={np.mean(db_lo):.4f} {db_lo} "
        f"vs D_b(H=0.95) mean={np.mean(db_hi):.4f} {db_hi}"
    )


def test_db_estimate_flat_limit_is_one():
    """A=0 (flat-Voronoi facets, geometrically straight in cross-section)
    box-counts to D_b ~= 1.05 at this test's grid resolution (grid_size=24,
    measured directly: 1.0497), not the geometric 1.000 -- the discretized
    3D voxel-grid pipeline (rasterize onto a finite grid, THEN section,
    THEN box-count the section's pixel boundary) carries its own
    resolution-dependent pixelation bias even for dead-straight facets,
    layered on top of (and empirically larger than, at this resolution,
    than) the documented smooth-curve estimator bias the design note (§7)
    discusses for genuinely curved boundaries. The assertion below is
    deliberately loose (0.15, not ~0.01) to bound this pixelation
    contribution without pinning its exact magnitude, which is grid-size
    dependent and not the property this test exists to check -- the point
    is that A=0 stays CLOSE to 1.0 and does not drift toward the D_b~1.1-1.2
    range that non-zero amplitude produces (see test_db_monotonic_in_
    hurst_direction), not that it hits 1.000 exactly."""
    seeds, box, msd, _ = _geometry(n=8, L=80.0, seed=3)
    tess = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.0, min_seed_distance=msd,
        rng=_rng(9), grid_size=24, connectivity_check=True,
        spectrum="self_affine", hurst=0.8, l_min=8.0, l_max=30.0,
    )
    db = tess.d_b_estimate()
    assert db is not None
    assert abs(db - 1.0) < 0.15


def test_hurst_estimate_property():
    """G13's Hurst back-estimate, owned by perturbed_distance since
    Phase 3 (moved here from WarpTessellation.hurst_estimate, removed --
    see tests/test_self_affine.py's module docstring): fitted from the
    radially averaged 3D PSD of the synthesized per-color fields, same
    contract WarpTessellation.hurst_estimate used to pin for its
    displacement field (tessellation/warp.estimate_hurst is shared
    verbatim)."""
    seeds, box, msd, _ = _geometry(n=8, L=80.0, seed=11)
    tess = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.4, min_seed_distance=msd,
        rng=_rng(4), grid_size=32, connectivity_check=False,
        spectrum="self_affine", hurst=0.8, l_min=10.0, l_max=40.0,
    )
    assert abs(tess.hurst_estimate() - 0.8) < 0.15
    # gaussian field has no Hurst exponent -- honest refusal, same
    # TessellationError contract as before.
    g = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.4, min_seed_distance=msd,
        rng=_rng(4), grid_size=32, connectivity_check=False,
        spectrum="gaussian", correlation_length=15.0,
    )
    with pytest.raises(TessellationError, match="self_affine"):
        g.hurst_estimate()


def test_box_count_dimension_none_on_degenerate_masks():
    """box_count_dimension refuses (returns None, not a fabricated slope)
    on masks with no usable boundary or too small for the eps range --
    the 'n/a, never fabricated' contract G20 relies on."""
    tiny = np.zeros((10, 10), dtype=bool)
    tiny[5, 5] = True
    assert box_count_dimension(tiny) is None
    all_one = np.ones((50, 50), dtype=bool)
    assert box_count_dimension(all_one) is None  # no perimeter at all


# ---------------------------------------------------------------------------
# Config error paths
# ---------------------------------------------------------------------------


def test_amplitude_exceeding_a_max_raises_at_construction():
    """Guard 1 (pre-flight, hard): amplitude beyond the seed-containment
    ceiling A_max = C*min_seed_distance/(2*ETA_CLIP) raises ConfigError
    naming the seed-containment guard, mirroring warp's analogous test."""
    seeds, box, msd, a_max = _geometry()
    with pytest.raises(ConfigError, match="seed-containment guard"):
        PerturbedDistanceTessellation(
            seeds, box, PER, amplitude=1.5 * a_max, min_seed_distance=msd,
            rng=_rng(1), grid_size=16, connectivity_check=False,
            spectrum="gaussian", correlation_length=15.0,
        )


def test_amplitude_exceeding_a_max_raises_at_resolve():
    """The identical guard is duplicated at resolve-time (Rule 7b,
    config/resolve.py) as a cheap pre-flight check before paying for
    field synthesis -- verified via the public resolve_config entry
    point, not just the tessellation constructor."""
    raw = _perturbed_raw_config(amplitude=1000.0)
    with pytest.raises(ConfigError, match="perturbed_distance"):
        resolve_config(raw)


def test_seed_ownership_posthoc_guard_trips_when_preflight_passes():
    """Guard 2 (post-hoc, exact): a config can pass guard 1 (amplitude
    under the ceiling computed from a CLAIMED min_seed_distance) yet still
    capture a seed if the seeds' TRUE spacing is much smaller than claimed
    -- the exact grain_of(seed_i) == i check (design note §4.3a) is what
    actually catches this, not the arithmetic ceiling. Reproduces
    deterministically at the fixed field seed used here (verified over 30
    field seeds during design; this one trips every time)."""
    box = np.array([100.0, 100.0, 100.0])
    seeds = np.array([
        [50.0, 50.0, 50.0],
        [50.05, 50.0, 50.0],   # TRUE nearest spacing: 0.05 Å
        [10.0, 10.0, 10.0],
        [90.0, 90.0, 90.0],
    ])
    claimed_msd = 40.0  # wildly overstates the true 0.05 Å spacing above
    a_max_claimed = PERTURBED_DISTANCE_SAFETY * claimed_msd / (2.0 * ETA_CLIP)
    amplitude = 0.9 * a_max_claimed
    assert amplitude <= a_max_claimed  # guard 1 (arithmetic ceiling) PASSES

    with pytest.raises(ConfigError, match="not self-owned"):
        PerturbedDistanceTessellation(
            seeds, box, PER, amplitude=amplitude, min_seed_distance=claimed_msd,
            rng=_rng(9000), grid_size=24, connectivity_check=True,
            spectrum="gaussian", correlation_length=15.0,
        )


def test_l_min_below_grid_nyquist_raises():
    """l_min < 2*h_field (grid cannot resolve the requested shortest
    wavelength) is a hard ConfigError, checked against the ACTUAL grid at
    construction time (mirrors test_self_affine.py's warp analogue,
    test_l_min_grid_resolvability)."""
    seeds, box, msd, a_max = _geometry(n=6, L=80.0, seed=2)
    with pytest.raises(ConfigError, match="2·h_field"):
        PerturbedDistanceTessellation(
            seeds, box, PER, amplitude=a_max, min_seed_distance=msd,
            rng=_rng(1), grid_size=16,  # h_field = 80/16 = 5 -> 2h=10
            connectivity_check=False,
            spectrum="self_affine", hurst=0.8, l_min=8.0, l_max=40.0,
        )


def test_rule20_l_min_ge_l_max_raises_at_resolve():
    """l_min >= l_max is rejected at resolve-time for perturbed_distance,
    sharing Rule 20's textual check with warp (config/resolve.py)."""
    raw = _perturbed_raw_config(l_min=30.0, l_max=8.0)
    with pytest.raises(ConfigError, match="l_min < l_max"):
        resolve_config(raw)


def test_rule20_l_max_exceeds_box_cap_raises_at_resolve():
    """l_max > min(box.lengths)/2 is rejected at resolve-time (the same
    box-representability cap warp's self_affine spectrum enforces)."""
    raw = _perturbed_raw_config(l_max=50.0)  # box is 80 Å -> cap is 40 Å
    with pytest.raises(ConfigError, match=r"min\(box.lengths\)/2"):
        resolve_config(raw)


def test_gaussian_spectrum_accepted_for_perturbed_distance():
    """Unlike warp's self_affine-only Rule 20 exclusivity, 'gaussian' is
    perturbed_distance's own DEFAULT spectrum and resolves without
    tripping any self_affine-specific rule (l_min/l_max/hurst are simply
    inert for this spectrum, same convention as warp)."""
    raw = _perturbed_raw_config(spectrum="gaussian", amplitude=0.5,
                                correlation_length=15.0)
    cfg = resolve_config(raw)
    assert cfg.boundaries.curved.method == "perturbed_distance"
    assert cfg.boundaries.curved.spectrum == "gaussian"


def test_self_affine_spectrum_method_restriction_names_perturbed_distance_only():
    """Rule 20's method-restriction error (spectrum='self_affine' with a
    method that doesn't consume a spectral field at all) now names
    perturbed_distance ONLY -- this test's own former name/docstring
    ("...still_names_both_methods") pinned an intermediate state where
    self_affine was legal on either warp or perturbed_distance; THIS
    task's Rule 8a removes warp+self_affine unconditionally (a coordinate
    diffeomorphism cannot produce a genuinely self-affine boundary regardless
    of Hurst exponent -- docs/physics.md §5b), so Rule 20 -- reached only
    once Rule 8a has already ruled out method=='warp' -- has exactly one
    self_affine-capable method left to name. See
    test_self_affine.py::test_rule20_perturbed_distance_only, updated
    alongside this file for the same reason."""
    raw = _perturbed_raw_config(method="additive_weights", weight_sigma=1.0)
    with pytest.raises(ConfigError, match="perturbed_distance method only"):
        resolve_config(raw)


def _perturbed_raw_config(**curved_overrides):
    """Minimal valid raw config dict with boundaries.curved.method =
    perturbed_distance, for exercising config/resolve.py Rule 7b/20
    without going through the full pipeline. Mirrors test_self_affine.py's
    ``_raw`` helper (same box/grains/crystal skeleton)."""
    blk = {
        "method": "perturbed_distance", "amplitude": 1.0,
        "spectrum": "self_affine", "hurst": 0.8, "l_min": 8.0, "l_max": 30.0,
    }
    blk.update(curved_overrides)
    return {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [80.0, 80.0, 80.0]},
        "grains": {"number": 8},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0, 0, 0]}],
        },
        "boundaries": {"geometry": "curved", "curved": blk},
    }


# ---------------------------------------------------------------------------
# RNG discipline: consumption is a function of (seeds, box, l_min, l_max,
# grid) only -- NOT of hurst or amplitude (design note §10).
# ---------------------------------------------------------------------------


def test_rng_consumption_independent_of_hurst():
    """Same field seed, different hurst -> the SAME post-construction RNG
    state (i.e. identical number and values of draws consumed), because
    only the spectral envelope applied to already-drawn white noise
    changes with H, never the draw sequence itself."""
    seeds, box, msd, a_max = _geometry()
    rng_a = _rng(999)
    PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=a_max, min_seed_distance=msd,
        rng=rng_a, grid_size=16, connectivity_check=False,
        spectrum="self_affine", hurst=0.3, l_min=8.0, l_max=30.0,
    )
    rng_b = _rng(999)
    PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=a_max, min_seed_distance=msd,
        rng=rng_b, grid_size=16, connectivity_check=False,
        spectrum="self_affine", hurst=0.95, l_min=8.0, l_max=30.0,
    )
    # Draw one more sample from each stream: if construction consumed a
    # different number/sequence of random values for a different hurst,
    # these post-construction draws would diverge.
    np.testing.assert_array_equal(
        rng_a.standard_normal(8), rng_b.standard_normal(8),
    )


def test_rng_consumption_independent_of_amplitude():
    """Same field seed, different amplitude -> identical post-construction
    RNG state (amplitude is applied at evaluation time, never baked into
    the stored field draws) -- and the stored (pre-scaling) fields
    themselves are bit-identical across amplitudes."""
    seeds, box, msd, a_max = _geometry()
    rng_a = _rng(999)
    t_a = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=a_max, min_seed_distance=msd,
        rng=rng_a, grid_size=16, connectivity_check=False,
        spectrum="self_affine", hurst=0.3, l_min=8.0, l_max=30.0,
    )
    rng_b = _rng(999)
    t_b = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.1 * a_max, min_seed_distance=msd,
        rng=rng_b, grid_size=16, connectivity_check=False,
        spectrum="self_affine", hurst=0.3, l_min=8.0, l_max=30.0,
    )
    np.testing.assert_array_equal(t_a._fields, t_b._fields)
    np.testing.assert_array_equal(
        rng_a.standard_normal(8), rng_b.standard_normal(8),
    )


def test_rng_consumption_independent_of_hurst_end_to_end_pipeline():
    """Same guarantee, exercised through the public pipeline entry point:
    two runs differing ONLY in boundaries.curved.hurst must derive
    identical seeding/orientation/etc. streams (those are earlier RNGBundle
    children, unaffected regardless) AND, per this method's specific
    contract, must consume the SAME number of field-stream draws --
    checked indirectly via n_colors and the synthesized (pre-clip) field
    shapes matching, since the pipeline does not expose the raw stream."""
    raw_a = _perturbed_raw_config(hurst=0.3)
    raw_b = _perturbed_raw_config(hurst=0.95)
    cfg_a = resolve_config(raw_a)
    cfg_b = resolve_config(raw_b)
    from grainsmith.pipeline import _stage_tessellation
    from grainsmith.rng import make_rng
    from grainsmith.seeding import seed_grains as _seed_grains

    rng_bundle_a = make_rng(cfg_a.seed.value)
    rng_bundle_b = make_rng(cfg_b.seed.value)
    seeds_a = _seed_grains(cfg_a.grains.number,
                           np.asarray(cfg_a.box.lengths), PER,
                           rng_bundle_a.seeding)
    seeds_b = _seed_grains(cfg_b.grains.number,
                           np.asarray(cfg_b.box.lengths), PER,
                           rng_bundle_b.seeding)
    np.testing.assert_array_equal(seeds_a, seeds_b)  # seeding stream unaffected

    from grainsmith.seeding import wigner_seitz_radius as _wsr
    msd = _wsr(float(np.prod(cfg_a.box.lengths)), cfg_a.grains.number)
    tess_a, _ = _stage_tessellation(cfg_a, seeds_a, msd, rng_bundle_a.fields)
    tess_b, _ = _stage_tessellation(cfg_b, seeds_b, msd, rng_bundle_b.fields)
    assert tess_a.n_colors == tess_b.n_colors
    assert tess_a._fields.shape == tess_b._fields.shape


# ---------------------------------------------------------------------------
# amplitude_convention / reference_wavelength --
# construction-level contracts complementing test_self_affine.py's
# resolve.py-level Rule 29 tests.
# ---------------------------------------------------------------------------


def test_reference_wavelength_convention_matches_equivalent_total_rms_construction():
    """A PerturbedDistanceTessellation built with amplitude_convention:
    'reference_wavelength' produces EXACTLY the same grain_of() as one
    built with amplitude_convention: 'total_rms' and the pre-converted
    equivalent amplitude (amplitude_reference / kappa) -- the convention
    is purely a re-labeling of the SAME field-synthesis/argmin pipeline,
    never a different code path."""
    from grainsmith.tessellation.warp import reference_shell_kappa

    seeds, box, msd, a_max = _geometry()
    kappa = reference_shell_kappa(0.7, 8.0, 30.0)
    a0_max = a_max * kappa
    amplitude_ref = 0.9 * a0_max

    tess_ref = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=amplitude_ref, min_seed_distance=msd,
        rng=_rng(104), grid_size=16, connectivity_check=True,
        spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
        amplitude_convention="reference_wavelength",
    )
    assert tess_ref.amplitude_convention == "reference_wavelength"
    assert tess_ref.amplitude_reference == pytest.approx(amplitude_ref)
    assert tess_ref.kappa == pytest.approx(kappa)
    assert tess_ref.amplitude_total_rms == pytest.approx(amplitude_ref / kappa)

    tess_equiv = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=tess_ref.amplitude_total_rms,
        min_seed_distance=msd,
        rng=_rng(104), grid_size=16, connectivity_check=True,
        spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
        amplitude_convention="total_rms",
    )
    assert tess_equiv.amplitude_convention == "total_rms"
    assert tess_equiv.amplitude_reference is None
    assert tess_equiv.kappa == pytest.approx(1.0)

    pts = _rng(999).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    np.testing.assert_array_equal(tess_ref.grain_of(pts), tess_equiv.grain_of(pts))
    np.testing.assert_array_equal(tess_ref._fields, tess_equiv._fields)


def test_total_rms_convention_default_bit_identical_to_pre_task_construction():
    """DEFAULT BEHAVIOR UNCHANGED: constructing WITHOUT passing
    amplitude_convention at all (positional/keyword call exactly as
    every other call site in this file does) is bit-identical to
    passing amplitude_convention='total_rms' explicitly -- kappa stays
    1.0, amplitude_reference stays None, and grain_of/fields match."""
    seeds, box, msd, a_max = _geometry()
    common = dict(
        seeds=seeds, box_lengths=box, periodic=PER,
        amplitude=0.5 * a_max, min_seed_distance=msd,
        grid_size=16, connectivity_check=True,
        spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
    )
    tess_implicit = PerturbedDistanceTessellation(rng=_rng(104), **common)
    tess_explicit = PerturbedDistanceTessellation(
        rng=_rng(104), amplitude_convention="total_rms", **common)

    assert tess_implicit.amplitude_convention == "total_rms"
    assert tess_implicit.kappa == 1.0
    assert tess_implicit.amplitude_reference is None
    assert tess_implicit.amplitude_total_rms == common["amplitude"]

    pts = _rng(999).uniform(0.0, 1.0, size=(300, 3)) * 60.0
    np.testing.assert_array_equal(tess_implicit.grain_of(pts), tess_explicit.grain_of(pts))
    np.testing.assert_array_equal(tess_implicit._fields, tess_explicit._fields)


def test_reference_wavelength_guard_binds_total_rms_equivalent_at_construction():
    """The seed-containment guard (Guard 1, ConfigError, pre-flight) is
    re-checked at CONSTRUCTION time against the REALIZED total-RMS-
    equivalent amplitude under amplitude_convention: reference_wavelength
    -- an amplitude whose face value alone would pass a naive `> a_max`
    check must still raise once its total-RMS equivalent (amplitude /
    kappa) exceeds a_max. Complements
    test_self_affine.py::test_rule29d_guard_rechecked_on_realized_total_rms_equivalent
    (that test exercises the SAME contract one layer up, at
    config/resolve.py's pre-flight Rule 29(d) check)."""
    from grainsmith.tessellation.warp import reference_shell_kappa

    seeds, box, msd, a_max = _geometry()
    kappa = reference_shell_kappa(0.7, 8.0, 30.0)
    a0_max = a_max * kappa
    over_ceiling = 1.05 * a0_max
    assert over_ceiling < a_max  # would WRONGLY pass a plain total_rms-style check

    with pytest.raises(ConfigError, match="seed-containment guard"):
        PerturbedDistanceTessellation(
            seeds, box, PER, amplitude=over_ceiling, min_seed_distance=msd,
            rng=_rng(104), grid_size=16, connectivity_check=True,
            spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
            amplitude_convention="reference_wavelength",
        )
    # the identical numeric amplitude is fine under total_rms (below a_max)
    PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=over_ceiling, min_seed_distance=msd,
        rng=_rng(104), grid_size=16, connectivity_check=True,
        spectrum="self_affine", hurst=0.7, l_min=8.0, l_max=30.0,
        amplitude_convention="total_rms",
    )


def test_reference_wavelength_octave_increase_raises_local_roughness_sign():
    """Sign test (small, fast synthetic field; complements the S1-style
    numeric pin in test_warp.py): fixing the SAME white-noise draw and
    widening l_max at fixed l_min, the reference_wavelength convention
    with a FIXED ABSOLUTE reference_wavelength (anchored near l_min, not
    moving with l_max) accumulates local-roughness monotonically as
    octaves are added, correcting the total_rms convention's inverted
    (monotonically DECREASING) trend at fixed `amplitude` -- see
    docs/physics.md Sec 5b(f) for the same measurement on a larger grid.

    A local-roughness PROXY (mean nearest-neighbor finite-difference RMS
    of the raw synthesized field, not d_b_estimate -- this needs to be
    fast and does not need an actual PerturbedDistanceTessellation
    construction) is used since it responds directly and cheaply to the
    field's own amplitude at l_min scale, unlike d_b_estimate whose
    box-counting fit is dominated by decade-budget noise at these tiny
    grid sizes (see G20's own docstring)."""
    from grainsmith.tessellation.warp import (
        amplitude_to_total_rms,
        reference_shell_kappa,
        synthesize_grf,
    )

    grid_shape = (48, 48, 48)
    L = np.array([120.0, 120.0, 120.0])
    hurst = 0.7
    l_min = 8.0
    l_max_values = [16.0, 32.0, 48.0]  # 1, 2, 3 octaves
    amplitude_a0 = 1.0
    reference_wavelength = 16.0  # FIXED absolute anchor near l_min

    def local_roughness(field):
        d0 = np.diff(field, axis=0)
        d1 = np.diff(field, axis=1)
        d2 = np.diff(field, axis=2)
        return float(np.sqrt(np.mean(d0**2) + np.mean(d1**2) + np.mean(d2**2)))

    roughness_total_rms = []
    roughness_reference = []
    for l_max in l_max_values:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(2026)))
        raw = synthesize_grf(grid_shape, correlation_length=15.0,
                             box_lengths=L, rng=rng, spectrum="self_affine",
                             hurst=hurst, l_min=l_min, l_max=l_max,
                             n_components=1)[0]
        kappa = reference_shell_kappa(hurst, l_min, l_max, reference_wavelength)
        amplitude_total = amplitude_to_total_rms(amplitude_a0, kappa)
        roughness_total_rms.append(local_roughness(amplitude_a0 * raw))
        roughness_reference.append(local_roughness(amplitude_total * raw))

    # total_rms convention (fixed face-value amplitude): monotonically
    # DECREASING local roughness as octaves are added -- the documented
    # inversion the reference_wavelength reparametrization exists to correct.
    assert all(roughness_total_rms[i] > roughness_total_rms[i + 1]
              for i in range(len(roughness_total_rms) - 1)), roughness_total_rms
    # reference_wavelength convention (fixed reference-octave amplitude,
    # FIXED ABSOLUTE anchor): monotonically NON-DECREASING -- corrected
    # direction.
    assert all(roughness_reference[i] <= roughness_reference[i + 1] + 1e-9
              for i in range(len(roughness_reference) - 1)), roughness_reference


# ---------------------------------------------------------------------------
# G20 band-limited fit window
# ---------------------------------------------------------------------------


def test_box_count_dimension_eps_max_px_overrides_eps_ratio():
    """box_count_dimension's new eps_max_px parameter, when given,
    overrides eps_ratio entirely; eps_max_px=None (the default) preserves
    the original fixed-decade behavior exactly (no call site regression)."""
    rng = np.random.default_rng(7)
    mask = rng.random((60, 60)) > 0.5
    # default behavior (eps_max_px=None) unchanged vs. explicit eps_ratio
    db_default = box_count_dimension(mask, eps_min_px=3, eps_ratio=10)
    db_explicit_none = box_count_dimension(mask, eps_min_px=3, eps_ratio=10,
                                           eps_max_px=None)
    assert db_default == db_explicit_none
    # an explicit eps_max_px produces a DIFFERENT (narrower-band) fit
    db_narrow = box_count_dimension(mask, eps_min_px=3, eps_max_px=9)
    assert db_narrow is not None
    # narrower band is a different (not necessarily equal) fit -- just
    # confirm it actually took the override path and returned a finite
    # number, not silently falling back to the eps_ratio default (which
    # would give eps_max_px=30 here, a much wider band).
    assert db_narrow != db_default or True  # documents intent; see below
    db_ratio3 = box_count_dimension(mask, eps_min_px=3, eps_ratio=3)  # eps_max=9
    assert db_narrow == pytest.approx(db_ratio3)


def test_box_count_dimension_degenerate_band_returns_none():
    """eps_min_px >= eps_max_px (a degenerate/inverted band, e.g. from a
    synthesis band that collapses to <1 px at a given section's pixel
    scale) returns None rather than raising or fabricating a fit."""
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    assert box_count_dimension(mask, eps_min_px=5, eps_max_px=5) is None
    assert box_count_dimension(mask, eps_min_px=8, eps_max_px=5) is None


def test_d_b_estimate_uses_band_limited_window_for_self_affine():
    """PerturbedDistanceTessellation.d_b_estimate (gate G20) restricts its
    box-counting fit to eps in [l_min, l_max] (converted to the section's
    pixel scale) for the self_affine spectrum -- both the flat-limit
    (A=0) and Hurst-monotonicity properties survive this change
    (re-verified here on the SAME small geometries test_db_estimate_
    flat_limit_is_one / test_db_monotonic_in_hurst_direction use, at
    their existing tolerances -- this is a regression check on the
    banded window, not a new physics claim)."""
    seeds, box, msd, _ = _geometry(n=8, L=80.0, seed=3)
    tess = PerturbedDistanceTessellation(
        seeds, box, PER, amplitude=0.0, min_seed_distance=msd,
        rng=_rng(9), grid_size=24, connectivity_check=True,
        spectrum="self_affine", hurst=0.8, l_min=8.0, l_max=30.0,
    )
    db = tess.d_b_estimate()
    assert db is not None
    assert abs(db - 1.0) < 0.15  # same flat-limit tolerance as before

    db_lo = _dbh_ensemble(0.3)
    db_hi = _dbh_ensemble(0.95)
    assert np.mean(db_lo) > np.mean(db_hi), (
        f"D_b(H=0.3) mean={np.mean(db_lo):.4f} vs D_b(H=0.95) mean={np.mean(db_hi):.4f}"
    )


# ---------------------------------------------------------------------------
# _d1_excluding k-escalation (2, 4, n_blocks+1) vs. the old single
# k=n_blocks+1 query -- byte-identity gate, mirrors test_weighted.py's
# reference-equality pattern / test_owns_kernel.py's numba-vs-numpy
# oracle style, applied here to two implementations of the same query.
# ---------------------------------------------------------------------------


def _reference_d1_excluding_k28(obj, X, i):
    """Frozen reference implementation: the previous single
    k=min(n_blocks+1, n_replicas) cKDTree query, no escalation. Reads
    only ``obj._rep_seeds`` / ``obj._n`` / ``obj._tree`` -- works
    identically on a real PerturbedDistanceTessellation and on the
    minimal stand-in built by ``_kernel_stub`` below."""
    n_blocks = len(obj._rep_seeds) // obj._n
    k = min(n_blocks + 1, len(obj._rep_seeds))
    dists, idxs = obj._tree.query(X, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    grain_ids_k = idxs % obj._n
    not_i = grain_ids_k != i
    ok = np.any(not_i, axis=1)
    if not np.all(ok):
        raise TessellationError(
            f"perturbed_distance margin(): no competing grain found "
            f"for grain {i} within its own replica count nearest "
            "neighbors — degenerate system (e.g. n_grains=1, which "
            "has no grain boundary to measure margin against)."
        )
    first_col = np.argmax(not_i, axis=1)
    return dists[np.arange(len(X)), first_col]


class _KernelStub:
    """Exposes only the three attributes ``_d1_excluding`` reads
    (``_rep_seeds``, ``_n``, ``_tree``) so degenerate replica layouts
    (co-located seeds, exact ties, a forced-escalation neighbourhood) can
    be probed directly, without going through
    PerturbedDistanceTessellation's constructor guards (guard 2 in
    particular rejects co-located / non-self-owned seeds outright, which
    is exactly the geometry some of these cases need)."""

    def __init__(self, rep_seeds, n_grains):
        self._rep_seeds = np.asarray(rep_seeds, dtype=np.float64)
        self._n = n_grains
        self._tree = cKDTree(self._rep_seeds)

    _d1_excluding = PerturbedDistanceTessellation._d1_excluding


def _assert_d1_excluding_matches(obj, X, i):
    out = obj._d1_excluding(np.asarray(X, dtype=np.float64), i)
    ref = _reference_d1_excluding_k28(obj, np.asarray(X, dtype=np.float64), i)
    assert out.dtype == ref.dtype == np.float64
    assert out.shape == ref.shape
    assert out.tobytes() == ref.tobytes()
    return out


def test_d1_excluding_k_escalation_bitwise():
    """out.tobytes() from the (2, 4, n_blocks+1) escalation must equal
    the frozen k=n_blocks+1-only reference, bit for bit, across
    randomized perturbed systems and the degenerate cases enumerated
    below."""
    # --- randomized perturbed systems: several geometries, several
    # grains each, query points both inside and outside the box (mirrors
    # test_weighted.py's [-0.2, 1.2]*L battery).
    for geom_seed in (1, 2, 3):
        tess, seeds, box, msd, a_max = _make(
            n=10, L=50.0, geom_seed=geom_seed, amplitude_frac=0.5,
            field_seed=200 + geom_seed, grid_size=16,
        )
        rng = _rng(3000 + geom_seed)
        X = rng.uniform(-0.2, 1.2, size=(500, 3)) * 50.0
        for i in (0, 3, 9):
            _assert_d1_excluding_matches(tess, X, i)

    # --- degenerate: n_grains=2, two grains with EXACTLY co-located
    # seeds (guard 2 forbids this via the full constructor, hence the
    # stub) -- forces an immediate i-vs-non-i tie at distance 0.
    rep_colocated = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    stub = _KernelStub(rep_colocated, n_grains=2)
    X = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    _assert_d1_excluding_matches(stub, X, 0)
    _assert_d1_excluding_matches(stub, X, 1)

    # --- degenerate: points exactly on the bisector of two single-
    # replica grains (i-replica / non-i-replica tie at a known distance).
    rep_bisector = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    stub = _KernelStub(rep_bisector, n_grains=2)
    X = np.array([[5.0, 0.0, 0.0], [5.0, 3.0, -2.0], [5.0, -7.0, 1.0]])
    _assert_d1_excluding_matches(stub, X, 0)
    _assert_d1_excluding_matches(stub, X, 1)

    # --- degenerate: a box-corner point equidistant from all 8 replicas
    # of a periodic cube split 4-vs-4 between two grains -- an 8-way tie
    # spanning both i and non-i replicas.
    a = 3.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    rep_corner = signs * a  # 8 points, all at distance a*sqrt(3) from origin
    stub = _KernelStub(rep_corner, n_grains=2)  # grain id = row % 2 -> 4-vs-4 split
    X = np.array([[0.0, 0.0, 0.0]])
    _assert_d1_excluding_matches(stub, X, 0)
    _assert_d1_excluding_matches(stub, X, 1)

    # --- degenerate: force escalation past BOTH k=2 and k=4 -- grain i
    # owns the 27 nearest replicas to X (n_blocks=27, mirroring the real
    # curvature_heavy geometry's n_blocks+1=28), so k=2 and k=4 columns
    # are entirely grain i and only the final k=28 query resolves it.
    rng = _rng(42)
    own_near = np.array([0.2, 0.2, 0.2]) + rng.uniform(-0.05, 0.05, size=(27, 3))
    other_far_a = np.array([5.0, 5.0, 5.0]) + rng.uniform(-0.1, 0.1, size=(27, 3))
    other_far_b = np.array([-5.0, -5.0, -5.0]) + rng.uniform(-0.1, 0.1, size=(27, 3))
    # index = block*n_grains + gid (real replica layout's own convention,
    # weighted.py-style) so idx % 3 gives the intended grain id per block.
    rep_escalate = np.stack([own_near, other_far_a, other_far_b], axis=1).reshape(-1, 3)
    stub = _KernelStub(rep_escalate, n_grains=3)  # grain id = idx % 3
    X = np.array([[0.2, 0.2, 0.2]])
    out = _assert_d1_excluding_matches(stub, X, 0)
    # sanity: escalation really was necessary (the nearest 4 columns
    # around X are indeed all grain-0's own replicas) and the resolved
    # distance is one of the far replicas', not a near one.
    d_near_only, idx_near_only = stub._tree.query(X, k=4)
    assert np.all((idx_near_only % 3) == 0)
    assert out[0] > 1.0

    # --- degenerate: small n_blocks (e.g. a box periodic on only one axis
    # has n_blocks=3, pigeonhole bound n_blocks+1=4) makes the k=4 step and
    # the k=n_blocks+1 step clip to the SAME k_eff=4 -- pins the
    # pigeonhole-bound clip (no step queries past n_blocks+1) together with
    # the step-dedup (that repeated k_eff=4 collapses into one query)
    # against the frozen k=n_blocks+1-only reference.
    rng = _rng(11)
    own_near = np.array([0.2, 0.2, 0.2]) + rng.uniform(-0.05, 0.05, size=(3, 3))
    other_far_a = np.array([5.0, 5.0, 5.0]) + rng.uniform(-0.1, 0.1, size=(3, 3))
    other_far_b = np.array([-5.0, -5.0, -5.0]) + rng.uniform(-0.1, 0.1, size=(3, 3))
    rep_small_blocks = np.stack(
        [own_near, other_far_a, other_far_b], axis=1).reshape(-1, 3)
    stub = _KernelStub(rep_small_blocks, n_grains=3)  # n_blocks = 9 // 3 = 3
    X = np.array([[0.2, 0.2, 0.2]])
    out = _assert_d1_excluding_matches(stub, X, 0)
    # sanity: k=2 alone is unresolved (both columns are grain 0's own
    # replicas), so escalation to k=4 (== n_blocks+1) was really exercised.
    d_near_only, idx_near_only = stub._tree.query(X, k=2)
    assert np.all((idx_near_only % 3) == 0)
    assert out[0] > 1.0

    # --- degenerate: n_grains=1 (no competing grain exists at all) must
    # still raise TessellationError, with escalation exhausting every
    # replica before giving up (same contract as the frozen reference
    # query).
    rep_single = rng.uniform(0.0, 10.0, size=(9, 3))
    stub = _KernelStub(rep_single, n_grains=1)
    X = rng.uniform(0.0, 10.0, size=(5, 3))
    with pytest.raises(TessellationError, match="no competing grain found"):
        stub._d1_excluding(X, 0)
    with pytest.raises(TessellationError, match="no competing grain found"):
        _reference_d1_excluding_k28(stub, X, 0)


def test_bound_radius_uses_a_max_not_amplitude():
    """``_bound_radius``'s slack term MUST be ``2*a_max*ETA_CLIP``, never
    ``2*self._amplitude*ETA_CLIP``.

    Substituting the run's own amplitude looks like a free tightening --
    a_max is a config-legal CEILING independent of the requested
    amplitude, and ``fill_grain`` CUBES this radius to size its lattice
    grid, so the apparent waste is large. It is wrong: the smaller radius
    under-covers and silently drops owned atoms from the fill (measured on
    examples/self_affine_gb/pdau_perturbed_self_affine.yaml: 535,789 ->
    535,769 fill atoms, 20 lost).

    That regression is nearly invisible downstream -- the missing atoms
    also remove overlap partners, so overlap removal deletes 23 fewer and
    the FINAL counts differ by only 3 (508,065 vs 508,068). This test
    therefore pins the radius formula itself rather than an atom count.

    See the comment block at the assignment in tessellation/perturbed.py
    for why 2*A*ETA_CLIP bounds the distance PERTURBATION (the correct
    search_pad in owns()/margin()) but not the ownership REACH.
    """
    # amplitude_frac well under 1 so the two candidate formulas differ.
    tess, seeds, box, _msd, a_max = _make(amplitude_frac=0.3)
    assert tess._amplitude < a_max, (
        "fixture must sit strictly below the ceiling or the two formulas "
        "coincide and the test proves nothing")

    flat = FlatTessellation(seeds, box, PER)
    flat_max = max(flat.bounding_radius(i) for i in range(tess.n_grains))
    expected = flat_max + 2.0 * a_max * ETA_CLIP
    too_small = flat_max + 2.0 * tess._amplitude * ETA_CLIP

    assert tess.bounding_radius(0) == pytest.approx(expected, rel=1e-12)
    assert tess.bounding_radius(0) > too_small
    # Uniform across grains: every grain gets the same (global-max) radius.
    assert len({tess.bounding_radius(i) for i in range(tess.n_grains)}) == 1
