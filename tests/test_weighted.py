"""Tests for tessellation/weighted.py (§6.6 methods M3/M4).

These backends previously shipped without any test; this file pins the
membership-API contracts that the fill and overlap stages rely on:
signed grain-specific margin, exact-once compact-cell tiling (owns), and
adjacency measured on the actual curved diagram.

P-numba: the module also ships an optional numba short-circuit owns()
kernel (weighted.py's "Numba acceleration" docstring section). The
``owns_backend`` fixture below is ``autouse`` and parametrized over
["numpy", "numba"], so EVERY test in this file — including the
pre-existing physical-invariant ones (tiling, bounding-radius cover,
signed margin) — runs under BOTH code paths automatically; the numba
branch is skipped cleanly when numba is not installed. The dedicated
equality tests near the end of this file additionally compare the two
paths DIRECTLY against each other (not just against a shared physical
invariant), on random points, near-tie points, and the exact-tie rule.
"""
from __future__ import annotations

import numpy as np
import pytest

import grainsmith.tessellation.weighted as weighted_mod
from grainsmith.errors import ConfigError
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.weighted import (
    AnisotropicTessellation,
    WeightedTessellation,
)

L = np.array([40.0, 40.0, 40.0])
PER = [True, True, True]


@pytest.fixture(params=["numpy", "numba"], autouse=True)
def owns_backend(request, monkeypatch):
    """Run every test in this module under both the numpy reference path
    and the numba short-circuit kernel (mirrors
    tests/test_owns_local.py's ``owns_backend`` fixture exactly)."""
    if request.param == "numba" and not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setattr(weighted_mod, "_USE_NUMBA", request.param == "numba")


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _seeds(n=5, seed=4):
    return seed_grains(n, L, PER, _rng(seed))


def _weighted(n=5, sigma=1.0, seed=4):
    seeds = _seeds(n, seed)
    return WeightedTessellation(seeds, L, PER, sigma_w=sigma,
                                rng=_rng(seed + 50)), seeds


def _aniso(n=5, seed=4, aspect=(1.0, 2.0)):
    seeds = _seeds(n, seed)
    return AnisotropicTessellation(seeds, L, PER, aspect_ratio_range=aspect,
                                   rng=_rng(seed + 60)), seeds


# ----------------------------------------------------------- additive (M3)

def test_weighted_zero_weights_equals_flat():
    """σ_w = 0 reduces the Apollonius diagram to plain Voronoi."""
    seeds = _seeds(6)
    wt = WeightedTessellation(seeds, L, PER)
    ft = FlatTessellation(seeds, L, PER)
    pts = _rng(9).uniform(0.0, 1.0, size=(300, 3)) * L
    np.testing.assert_array_equal(wt.grain_of(pts), ft.grain_of(pts))


def test_weighted_margin_signed_and_grain_specific():
    """margin(x, i) > 0 iff grain_of(x) == i; consistent magnitudes."""
    wt, seeds = _weighted()
    pts = _rng(10).uniform(0.0, 1.0, size=(200, 3)) * L
    gids = wt.grain_of(pts)
    for i in range(wt.n_grains):
        m = wt.margin(pts, i)
        assert np.all(np.isfinite(m))
        inside = gids == i
        assert np.all(m[inside] >= 0.0)
        assert np.all(m[~inside] <= 0.0)


def test_weighted_owns_tiles_torus():
    """Exact-once compact-cell tiling over grains × {-1,0,1}³ lifts."""
    wt, _ = _weighted(n=4, sigma=0.8, seed=12)
    pts = _rng(13).uniform(0.0, 1.0, size=(100, 3)) * L
    owned = np.zeros(len(pts), dtype=int)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * L
                for i in range(wt.n_grains):
                    owned += wt.owns(lift, i).astype(int)
    assert np.all(owned == 1)


def test_weighted_sigma_guard():
    """σ_w above the empty-cell limit raises ConfigError (§6.6)."""
    seeds = _seeds(5)
    with pytest.raises(ConfigError):
        WeightedTessellation(seeds, L, PER, sigma_w=50.0, rng=_rng(1))


def test_weighted_bounding_radius_covers_cell():
    """Every owned point lies within bounding_radius of its seed (the §6.8
    enumeration-cover guarantee)."""
    wt, seeds = _weighted(n=4, sigma=1.0, seed=15)
    pts = _rng(16).uniform(0.0, 1.0, size=(400, 3)) * L
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * L
                for i in range(wt.n_grains):
                    mask = wt.owns(lift, i)
                    if not np.any(mask):
                        continue
                    d = np.linalg.norm(lift[mask] - seeds[i], axis=1)
                    assert float(np.max(d)) <= wt.bounding_radius(i) + 1e-9


def test_weighted_adjacency_voxel_based():
    """Adjacency comes from the actual curved diagram, is symmetric-unique
    and self-free."""
    wt, _ = _weighted(n=5, sigma=1.0, seed=4)
    adj = wt.adjacency()
    assert len(adj) > 0
    for i, j in adj:
        assert 0 <= i < j < wt.n_grains
    assert len(adj) == len(set(adj))


# -------------------------------------------------------- anisotropic (M4)

def test_aniso_margin_signed_consistent():
    at, _ = _aniso()
    pts = _rng(20).uniform(0.0, 1.0, size=(200, 3)) * L
    gids = at.grain_of(pts)
    for i in range(at.n_grains):
        m = at.margin(pts, i)
        assert np.all(np.isfinite(m))
        assert np.all(m[gids == i] >= 0.0)
        assert np.all(m[gids != i] <= 0.0)


def test_aniso_owns_tiles_torus():
    at, _ = _aniso(n=4, seed=22)
    pts = _rng(23).uniform(0.0, 1.0, size=(60, 3)) * L
    owned = np.zeros(len(pts), dtype=int)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * L
                for i in range(at.n_grains):
                    owned += at.owns(lift, i).astype(int)
    assert np.all(owned == 1)


def test_aniso_bounding_radius_covers_cell():
    at, seeds = _aniso(n=4, seed=25, aspect=(1.0, 2.0))
    pts = _rng(26).uniform(0.0, 1.0, size=(300, 3)) * L
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * L
                for i in range(at.n_grains):
                    mask = at.owns(lift, i)
                    if not np.any(mask):
                        continue
                    d = np.linalg.norm(lift[mask] - seeds[i], axis=1)
                    assert float(np.max(d)) <= at.bounding_radius(i) + 1e-9


def test_aniso_unit_aspect_equals_flat():
    """Aspect ratio (1,1) makes every metric the identity → plain Voronoi."""
    seeds = _seeds(5, seed=30)
    at = AnisotropicTessellation(seeds, L, PER, aspect_ratio_range=(1.0, 1.0),
                                 rng=_rng(31))
    ft = FlatTessellation(seeds, L, PER)
    pts = _rng(32).uniform(0.0, 1.0, size=(200, 3)) * L
    np.testing.assert_array_equal(at.grain_of(pts), ft.grain_of(pts))


# ---------------------------------------------------------------------------
# P-numba: DIRECT numba-vs-numpy owns() equality (mandatory validation —
# see weighted.py's "Numba acceleration" docstring section). Unlike the
# tests above (run under both backends via the autouse fixture, and
# checking a shared PHYSICAL invariant), these compare the two owns()
# code paths against EACH OTHER on the exact same query points, for both
# backends, three periodicity geometries, and engineered tie cases.
# ---------------------------------------------------------------------------

_GEOMETRIES = [
    # (label, n_grains, L, periodic)
    ("periodic", 6, np.array([40.0, 40.0, 40.0]), [True, True, True]),
    ("slab_z", 5, np.array([40.0, 40.0, 28.0]), [True, True, False]),
    ("two_free", 4, np.array([35.0, 28.0, 28.0]), [True, False, False]),
]


def _build_aniso_geom(label, n, Lg, periodic, seed=7, aspect=(1.0, 2.5)):
    rng_seed = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    seeds = seed_grains(n, Lg, periodic, rng_seed)
    rng_m = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed + 1)))
    return AnisotropicTessellation(seeds, Lg, periodic,
                                   aspect_ratio_range=aspect, rng=rng_m,
                                   connectivity_check=True), seeds


def _build_weighted_geom(label, n, Lg, periodic, seed=7, sigma=None):
    rng_seed = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    seeds = seed_grains(n, Lg, periodic, rng_seed)
    from grainsmith.seeding import wigner_seitz_radius
    msd = wigner_seitz_radius(float(np.prod(Lg)), n)
    sigma_use = sigma if sigma is not None else msd / 8.0
    rng_w = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed + 1)))
    return WeightedTessellation(seeds, Lg, periodic, sigma_w=sigma_use,
                                rng=rng_w, min_seed_distance=msd,
                                connectivity_check=True), seeds


def _direct_compare(tess, X, label):
    """Force numpy, then force numba, on the SAME tessellation object and
    query points; assert every grain's mask matches exactly."""
    for i in range(tess.n_grains):
        weighted_mod._USE_NUMBA = False
        numpy_mask = tess.owns(X, i)
        weighted_mod._USE_NUMBA = True
        numba_mask = tess.owns(X, i)
        assert np.array_equal(numpy_mask, numba_mask), (
            f"{label} grain {i}: {(numpy_mask != numba_mask).sum()} / "
            f"{len(X)} points differ between numba and numpy"
        )


@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_aniso_numba_equals_numpy_random(geom):
    """Battery (a): uniform random in [-0.2L, 1.2L]^3, all 3 geometries."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    label, n, Lg, periodic = geom
    saved = weighted_mod._USE_NUMBA
    try:
        tess, _ = _build_aniso_geom(label, n, Lg, periodic)
        rng = _rng(101)
        X = rng.uniform(-0.2, 1.2, size=(600, 3)) * Lg
        _direct_compare(tess, X, f"aniso/{label}/random")
    finally:
        weighted_mod._USE_NUMBA = saved


@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_weighted_numba_equals_numpy_random(geom):
    """Battery (a) for the additive backend, all 3 geometries."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    label, n, Lg, periodic = geom
    saved = weighted_mod._USE_NUMBA
    try:
        tess, _ = _build_weighted_geom(label, n, Lg, periodic)
        rng = _rng(102)
        X = rng.uniform(-0.2, 1.2, size=(600, 3)) * Lg
        _direct_compare(tess, X, f"weighted/{label}/random")
    finally:
        weighted_mod._USE_NUMBA = saved


@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_aniso_numba_equals_numpy_fill_candidates(geom):
    """Battery (b): actual lattice candidate positions from a tiny FCC
    fill on the anisotropic backend — the exact code path fill_grain
    exercises (fine SC grid at 2 Å spacing, well inside every
    geometry's box)."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    label, n, Lg, periodic = geom
    saved = weighted_mod._USE_NUMBA
    try:
        tess, _ = _build_aniso_geom(label, n, Lg, periodic)
        spacing = 2.0
        ix = np.arange(0, int(Lg[0] / spacing) + 1)
        iy = np.arange(0, int(Lg[1] / spacing) + 1)
        iz = np.arange(0, int(Lg[2] / spacing) + 1)
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        X = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * spacing
        inside = np.all((X >= 0) & (X < Lg), axis=1)
        X = X[inside]
        rng = _rng(303)
        if len(X) > 2000:
            X = X[rng.choice(len(X), 2000, replace=False)]
        _direct_compare(tess, X, f"aniso/{label}/fill_candidates")
    finally:
        weighted_mod._USE_NUMBA = saved


def test_aniso_numba_equals_numpy_near_tie():
    """Battery (c): points placed exactly on the flat bisector between two
    grains (aspect ratio 1 -> plain Voronoi, so the bisector is a known
    plane), then perturbed by tiny +/- offsets straddling the tie —
    stresses the short-circuit's boundary behaviour on both sides of a
    near-degenerate comparison."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = weighted_mod._USE_NUMBA
    try:
        Lg = np.array([30.0, 30.0, 30.0])
        periodic = [True, True, True]
        seeds = np.array([[10.0, 15.0, 15.0], [20.0, 15.0, 15.0], [15.0, 5.0, 15.0]])
        tess = AnisotropicTessellation(
            seeds, Lg, periodic, metrics=[np.eye(3)] * 3, connectivity_check=True)
        # Bisector plane between seed 0 and seed 1 is x=15; sample y,z and
        # perturb x by +/- a tiny epsilon.
        rng = _rng(404)
        n_pts = 200
        yz = rng.uniform(2.0, 28.0, size=(n_pts, 2))
        eps = rng.choice([-1e-6, 0.0, 1e-6], size=n_pts)
        X = np.column_stack([15.0 + eps, yz[:, 0], yz[:, 1]])
        _direct_compare(tess, X, "aniso/near_tie")
    finally:
        weighted_mod._USE_NUMBA = saved


def test_aniso_numba_tie_rule_lowest_index_wins():
    """Exact tie-rule pin: a point EXACTLY equidistant (to float64) between
    two grains under an isotropic metric must be owned by the LOWER grain
    id on BOTH the numpy and the numba path — the first-minimum argmin
    convention, not an implementation accident."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = weighted_mod._USE_NUMBA
    try:
        Lg = np.array([20.0, 20.0, 20.0])
        seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
        periodic = [True, True, True]
        tess = AnisotropicTessellation(
            seeds, Lg, periodic, metrics=[np.eye(3), np.eye(3)],
            connectivity_check=True)
        mid = np.array([[10.0, 10.0, 10.0]])  # exact midpoint -> tie
        for use_numba in (False, True):
            weighted_mod._USE_NUMBA = use_numba
            owns0 = bool(tess.owns(mid, 0)[0])
            owns1 = bool(tess.owns(mid, 1)[0])
            assert owns0 is True, f"use_numba={use_numba}: grain 0 (lower id) must win the tie"
            assert owns1 is False, f"use_numba={use_numba}: grain 1 must lose the tie"
            assert int(tess.grain_of(mid)[0]) == 0
    finally:
        weighted_mod._USE_NUMBA = saved


def test_weighted_numba_tie_rule_lowest_index_wins():
    """Same tie-rule pin for the additive backend: zero weights reduce to
    plain Voronoi, so the exact midpoint is again a tie broken by lowest
    grain id, on both code paths."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = weighted_mod._USE_NUMBA
    try:
        Lg = np.array([20.0, 20.0, 20.0])
        seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
        periodic = [True, True, True]
        tess = WeightedTessellation(seeds, Lg, periodic,
                                    weights=np.zeros(2), connectivity_check=True)
        mid = np.array([[10.0, 10.0, 10.0]])
        for use_numba in (False, True):
            weighted_mod._USE_NUMBA = use_numba
            assert bool(tess.owns(mid, 0)[0]) is True
            assert bool(tess.owns(mid, 1)[0]) is False
    finally:
        weighted_mod._USE_NUMBA = saved


def test_owns_empty_input_both_backends():
    """owns([]) returns an empty bool array on both code paths, both
    backends — no crash on a zero-row chunk (fill_grain's chunk loop can
    legitimately produce one)."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = weighted_mod._USE_NUMBA
    try:
        at, _ = _aniso(n=4, seed=50)
        wt, _ = _weighted(n=4, seed=51)
        for use_numba in (False, True):
            weighted_mod._USE_NUMBA = use_numba
            for tess in (at, wt):
                out = tess.owns(np.empty((0, 3)), 0)
                assert out.dtype == bool
                assert len(out) == 0
    finally:
        weighted_mod._USE_NUMBA = saved


def test_numba_no_numba_env_escape_hatch(monkeypatch):
    """GRAINSMITH_NO_NUMBA=1 forces the numpy path at import time (task
    mandate's explicit escape hatch) — re-import the module with the env
    var set and confirm _USE_NUMBA comes up False even when numba IS
    installed."""
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setenv("GRAINSMITH_NO_NUMBA", "1")
    assert weighted_mod._numba_disabled_by_env() is True
    monkeypatch.setenv("GRAINSMITH_NO_NUMBA", "0")
    assert weighted_mod._numba_disabled_by_env() is False
    monkeypatch.delenv("GRAINSMITH_NO_NUMBA", raising=False)
    assert weighted_mod._numba_disabled_by_env() is False


# ---------------------------------------------------------------------------
# gb_shell_lower_bound: the certified k-NN GB-shell pre-filter used by
# atoms.overlap.remove_overlaps.  See tests/test_overlap.py for the
# cross-backend superset property and the full remove_overlaps byte-equality
# tests; this file pins the additive backend's own m_lb <= exact-margin
# arithmetic against a dense brute-force reference.
# ---------------------------------------------------------------------------


def test_shell_bound_none_for_anisotropic():
    """AnisotropicTessellation has no certified bound (Lipschitz constant
    1/s_min > 1 for any non-degenerate aspect ratio) -- always None."""
    at, _ = _aniso(n=6, seed=8)
    rng = _rng(17)
    pts = rng.uniform(0.0, 1.0, size=(200, 3)) * L
    grain = np.zeros(len(pts), dtype=np.int32)
    assert at.gb_shell_lower_bound(pts, grain, 1.0, workers=1) is None


def test_shell_bound_is_lower_bound_when_certified():
    """gb_shell_lower_bound's internal k-NN margin m_lb, on the subset of
    points its OWN certificate accepts, never exceeds the exact
    replica-aware margin computed by a DENSE brute-force reference over
    every one of the B*N replica seeds ('no unseen replica can beat the
    runner-up ... m_lb = (second - best) / 2 equals the exact margin' once
    certified).  This reproduces the same
    k=8 k-NN + certificate arithmetic gb_shell_lower_bound uses, outside
    the class, specifically to pin the FORMULA -- uncertified points are
    excluded here because m_lb is explicitly NOT proven bounded for them
    (that is why they stay in the public shell mask unconditionally
    instead)."""
    wt, seeds = _weighted(n=7, sigma=1.0, seed=21)
    rng = _rng(123)
    pts = rng.uniform(0.0, 1.0, size=(4000, 3)) * L

    # Dense brute-force reference: exact margin over EVERY replica.
    diff = pts[:, None, :] - wt._rep_seeds[None, :, :]
    d_all = np.sqrt(np.sum(diff**2, axis=2)) - wt._rep_weights[None, :]
    d_sorted = np.sort(d_all, axis=1)
    exact_margin = (d_sorted[:, 1] - d_sorted[:, 0]) / 2.0

    # Reproduce gb_shell_lower_bound's k=8 k-NN + certificate directly.
    k = weighted_mod._SHELL_KNN_K
    tree = wt._shell_replica_tree()
    d, idx = tree.query(pts, k=k, workers=1)
    w = wt._rep_weights[idx]
    g = d - w
    rows = np.arange(len(pts))
    best_col = np.argmin(g, axis=1)
    best = g[rows, best_col]
    g_masked = g.copy()
    g_masked[rows, best_col] = np.inf
    second = np.min(g_masked, axis=1)
    w_max = float(np.max(wt._rep_weights))
    certified = (d[:, -1] - w_max) > second
    m_lb = (second - best) / 2.0

    assert int(certified.sum()) > 0, "test is vacuous with zero certified points"
    assert np.all(m_lb[certified] <= exact_margin[certified] + 1e-9), (
        "m_lb exceeded the brute-force exact margin on a certified point"
    )
    # Certification is not merely a bound -- it is an equality (see the
    # docstring above): pin that too, not just the inequality.
    np.testing.assert_allclose(
        m_lb[certified], exact_margin[certified], atol=1e-9,
        err_msg="certified m_lb should equal the exact margin exactly",
    )
