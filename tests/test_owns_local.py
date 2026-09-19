"""Mandatory equality tests for the local-set owns() optimisation.

For every grain i, every backend (Flat / Power with weights / Power without
weights), and every geometry (fully periodic, slab-z, two-free-axes) the new
owns(X, i) must return a bit-identical boolean mask to the reference
_owns_global(X, i).

Three query-point batteries are exercised per scenario:

  (a) uniform random in [−0.2L, 1.2L]^3   — bulk interior + outside-box lifts
  (b) actual lattice candidate positions from a small real fill pipeline
  (c) points placed on cell faces then perturbed ±tol/2 to stress tie-breaks

A brute-force ground-truth check is also performed on a small fully-periodic
config: for each point the generalized distance to ALL replicas is computed
and the result is compared to argmin == home under the lowest-index tie-break.
"""
from __future__ import annotations

import numpy as np
import pytest

import grainsmith.tessellation._local_owns as _local_owns
from grainsmith.constants import OWNS_TIE_TOL
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.power import PowerTessellation


# ---------------------------------------------------------------------------
# P3.4: run the ENTIRE equality battery under both the NumPy path and the
# optional Numba kernel.  Asserting owns() == _owns_global (a NumPy reference)
# under the Numba backend proves the kernel is bit-identical to NumPy on every
# scenario.  The Numba variant is skipped cleanly when numba is not installed.
# ---------------------------------------------------------------------------

@pytest.fixture(params=["numpy", "numba"], autouse=True)
def owns_backend(request, monkeypatch):
    if request.param == "numba" and not _local_owns._HAVE_NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setattr(_local_owns, "_USE_NUMBA", request.param == "numba")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _make_flat(n: int, L: np.ndarray, periodic: list[bool], seed: int = 42
               ) -> FlatTessellation:
    rng = _rng(seed)
    seeds = seed_grains(n, L, periodic, rng)
    return FlatTessellation(seeds, L, periodic)


def _make_power(n: int, L: np.ndarray, periodic: list[bool], seed: int = 42,
                with_weights: bool = True) -> PowerTessellation:
    rng = _rng(seed)
    seeds = seed_grains(n, L, periodic, rng)
    if with_weights:
        rng2 = _rng(seed + 1000)
        # Moderate weights: < 5% of L² so cells stay non-degenerate.
        weights = rng2.uniform(0.0, 0.05 * float(np.max(L)) ** 2, size=n)
    else:
        weights = np.zeros(n)
    return PowerTessellation(seeds, L, periodic, weights=weights)


def _random_points(L: np.ndarray, n_pts: int, rng, lo: float = -0.2,
                   hi: float = 1.2) -> np.ndarray:
    """Uniform random in [lo·L, hi·L]^3."""
    return rng.uniform(lo, hi, size=(n_pts, 3)) * L


def _face_perturbed_points(tess: FlatTessellation, L: np.ndarray,
                           n_per_face: int = 10, rng=None) -> np.ndarray:
    """Sample points on cell-face planes then perturb by ±tol/2."""
    if rng is None:
        rng = _rng(999)
    tol = OWNS_TIE_TOL * float(np.max(L))
    pts: list[np.ndarray] = []
    for cell in tess.cells:
        for face in cell.faces:
            verts = face.vertices            # already absolute in flat cells
            # Sample random convex combinations of the vertices.
            w = rng.dirichlet(np.ones(len(verts)), size=n_per_face)
            on_face = w @ verts              # (n_per_face, 3)
            # Perturb by ±tol/2 along the face normal.
            signs = rng.choice([-1.0, 1.0], size=n_per_face)
            perturbed = on_face + (signs * tol / 2.0)[:, None] * face.unit_normal
            pts.append(perturbed)
    if not pts:
        return np.empty((0, 3))
    return np.vstack(pts)


# ---------------------------------------------------------------------------
# Core equality assertion
# ---------------------------------------------------------------------------

def _assert_owns_equal(tess, L: np.ndarray, X: np.ndarray,
                       label: str) -> None:
    """For every grain i: new owns == reference _owns_global."""
    n = tess.n_grains
    for i in range(n):
        new_mask = tess.owns(X, i)
        ref_mask = tess._owns_global(X, i)
        assert np.array_equal(new_mask, ref_mask), (
            f"{label} grain {i}: "
            f"{(new_mask != ref_mask).sum()} / {len(X)} points differ"
        )


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

# (label, n_grains, L, periodic, with_weights)
_SCENARIOS = [
    # Flat backend — 3 geometries
    ("flat_periodic",   6, [50., 50., 50.], [True,  True,  True],  None),
    ("flat_slab_z",     5, [50., 50., 30.], [True,  True,  False], None),
    ("flat_2free",      4, [40., 40., 30.], [True,  False, False], None),
    # Power backend with weights — 3 geometries
    ("power_periodic",  6, [40., 40., 40.], [True,  True,  True],  True),
    ("power_slab_z",    5, [40., 40., 25.], [True,  True,  False], True),
    ("power_2free",     4, [35., 35., 25.], [True,  False, False], True),
    # Power backend zero weights (identical to Voronoi) — full periodic only
    ("power_zero_w",    6, [40., 40., 40.], [True,  True,  True],  False),
]


def _build_tess(scenario):
    label, n, Lv, periodic, with_weights = scenario
    L = np.array(Lv, dtype=np.float64)
    if with_weights is None:
        return _make_flat(n, L, periodic, seed=42), L
    else:
        return _make_power(n, L, periodic, seed=42,
                           with_weights=with_weights), L


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_owns_local_equals_global_random(scenario):
    """Battery (a): uniform random in [−0.2L, 1.2L]^3."""
    tess, L = _build_tess(scenario)
    rng = _rng(101)
    X = _random_points(L, 500, rng)
    _assert_owns_equal(tess, L, X, scenario[0] + "/random")


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_owns_local_equals_global_face_perturbed(scenario):
    """Battery (c): points on cell faces perturbed by ±tol/2."""
    tess, L = _build_tess(scenario)
    rng = _rng(202)
    # FlatTessellation and PowerTessellation both have .cells
    X = _face_perturbed_points(tess, L, n_per_face=8, rng=rng)
    if len(X) == 0:
        pytest.skip("no face points generated")
    _assert_owns_equal(tess, L, X, scenario[0] + "/face_perturbed")


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_owns_local_equals_global_fill_candidates(scenario):
    """Battery (b): actual lattice candidate positions from a tiny fill."""
    label, n, Lv, periodic, with_weights = scenario
    L = np.array(Lv, dtype=np.float64)
    tess, _ = _build_tess(scenario)

    # Generate a small FCC-like candidate set inside the box.
    # We avoid importing the full pipeline; instead we build a simple
    # SC grid at inter-atomic spacing ≈ 2 Å, which exercises integer-lattice
    # candidate points that may land exactly on Voronoi bisectors.
    spacing = 2.0
    ix = np.arange(0, int(L[0] / spacing) + 1)
    iy = np.arange(0, int(L[1] / spacing) + 1)
    iz = np.arange(0, int(L[2] / spacing) + 1)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
    X = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * spacing
    # Keep only points inside the box.
    inside = np.all((X >= 0) & (X < L), axis=1)
    X = X[inside]
    if len(X) == 0:
        pytest.skip("no candidate points")
    # Subsample for speed (keep at most 2000).
    rng = _rng(303)
    if len(X) > 2000:
        X = X[rng.choice(len(X), 2000, replace=False)]
    _assert_owns_equal(tess, L, X, label + "/fill_candidates")


# ---------------------------------------------------------------------------
# Empty-input edge case
# ---------------------------------------------------------------------------

def test_owns_empty_input():
    """owns([]) returns empty bool array without error."""
    L = np.array([30., 30., 30.])
    tess = _make_flat(4, L, [True, True, True])
    result = tess.owns(np.empty((0, 3)), 0)
    assert result.dtype == bool
    assert len(result) == 0


def test_numba_mask_equals_numpy_directly():
    """Direct Numba-vs-NumPy mask equality (P3.4), independent of the
    _owns_global reference: toggle the backend and compare owns() outputs."""
    if not _local_owns._HAVE_NUMBA:
        pytest.skip("numba not installed")
    saved = _local_owns._USE_NUMBA
    try:
        for tess, L in (_build_tess(s) for s in (_SCENARIOS[0], _SCENARIOS[3])):
            rng = _rng(123)
            X = _random_points(L, 800, rng)
            for i in range(tess.n_grains):
                _local_owns._USE_NUMBA = False
                numpy_mask = tess.owns(X, i)
                _local_owns._USE_NUMBA = True
                numba_mask = tess.owns(X, i)
                assert np.array_equal(numba_mask, numpy_mask), (
                    f"numba != numpy for grain {i}: "
                    f"{(numba_mask != numpy_mask).sum()} / {len(X)} differ"
                )
    finally:
        _local_owns._USE_NUMBA = saved


# ---------------------------------------------------------------------------
# Brute-force ground-truth check (fully periodic, small config)
# ---------------------------------------------------------------------------

def test_flat_owns_brute_force():
    """Brute-force: for each point compute Euclidean distance to ALL replicas
    and verify owns(X, i) == (argmin_replica == home_i), with the lowest-
    global-replica-index tie-break.  Tests FlatTessellation."""
    n = 5
    L = np.array([30., 30., 30.])
    tess = _make_flat(n, L, [True, True, True], seed=7)

    rng = _rng(55)
    X = _random_points(L, 200, rng, lo=0.0, hi=1.0)
    tol = OWNS_TIE_TOL * float(np.max(L))
    P = tess._all_pts                        # (B*N, 3)

    for i in range(n):
        home_idx = tess._identity_block * tess._n + i
        # Brute-force: D[r, k] = distance from X[r] to P[k]
        D = np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2)  # (R, B*N)
        d1 = D.min(axis=1)
        # Mask: within tolerance
        within = D <= d1[:, None] + tol
        # Lowest replica index among tied-nearest
        all_ids = np.arange(len(P), dtype=np.int64)
        masked = np.where(within, all_ids[None, :], len(P) + 1)
        min_id = masked.min(axis=1)
        expected = (min_id == home_idx)

        got = tess.owns(X, i)
        assert np.array_equal(got, expected), (
            f"brute-force mismatch for grain {i}: "
            f"{(got != expected).sum()} / {len(X)} points differ"
        )


def test_power_owns_brute_force():
    """Brute-force: for each point compute 4D lifted distance to ALL replicas
    and verify owns(X, i) == (argmin_replica == home_i).  Tests PowerTessellation."""
    n = 5
    L = np.array([30., 30., 30.])
    tess = _make_power(n, L, [True, True, True], seed=7, with_weights=True)

    rng = _rng(66)
    X = _random_points(L, 200, rng, lo=0.0, hi=1.0)
    tol = OWNS_TIE_TOL * float(np.max(L))
    Q = np.column_stack([X, np.zeros(len(X))])   # (R, 4)
    P4 = tess._lifted                             # (B*N, 4)

    for i in range(n):
        home_idx = tess._identity_block * tess._n + i
        D = np.linalg.norm(Q[:, None, :] - P4[None, :, :], axis=2)  # (R, B*N)
        d1 = D.min(axis=1)
        within = D <= d1[:, None] + tol
        all_ids = np.arange(len(P4), dtype=np.int64)
        masked = np.where(within, all_ids[None, :], len(P4) + 1)
        min_id = masked.min(axis=1)
        expected = (min_id == home_idx)

        got = tess.owns(X, i)
        assert np.array_equal(got, expected), (
            f"power brute-force mismatch for grain {i}: "
            f"{(got != expected).sum()} / {len(X)} points differ"
        )


# ---------------------------------------------------------------------------
# Tiling invariant (sanity — complements the existing parametrized tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_owns_local_tiles_torus_exactly_once(scenario):
    """Each torus point is owned exactly once over all grains and all
    {-1,0,1}^3 periodic lifts — the §6.8 fill-correctness invariant."""
    label, n, Lv, periodic, with_weights = scenario
    L = np.array(Lv, dtype=np.float64)
    tess, _ = _build_tess(scenario)

    rng = _rng(404)
    base = rng.uniform(0.0, 1.0, size=(150, 3)) * L
    lifts = np.array(
        [[sx, sy, sz]
         for sx in (-1, 0, 1)
         for sy in (-1, 0, 1)
         for sz in (-1, 0, 1)], dtype=np.float64) * L

    counts = np.zeros(len(base), dtype=int)
    for lift in lifts:
        X = base + lift
        for g in range(n):
            counts += tess.owns(X, g).astype(int)
    assert np.all(counts == 1), (
        f"{label}: tiling violated min={counts.min()} max={counts.max()}"
    )


# ---------------------------------------------------------------------------
# Free-axis wall ownership (slab geometry): the wall-MIRROR replica must never
# win a tie against the grain's own home replica.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["flat", "power"])
def test_owns_free_axis_walls_not_lost_to_mirror(backend):
    """On a free (non-periodic) axis a commensurate atom plane sits on BOTH the
    x=0 and x=L walls.  A query point on either wall is exactly equidistant to
    the grain's home replica and to its own wall-MIRROR replica; the lowest
    global-replica-index tie-break must resolve to the real home, not to the
    fictitious mirror (which no real grain claims).  Otherwise the entire x=0
    atom layer is silently dropped from a slab fill.  Both walls must behave
    symmetrically, and the fast owns() must stay bit-identical to _owns_global.
    """
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[10.0, 10.0, 10.0]])
    periodic = [False, True, True]
    if backend == "flat":
        tess = FlatTessellation(seeds, L, periodic)
    else:
        tess = PowerTessellation(seeds, L, periodic, weights=np.zeros(1))

    low = np.array([[0.0, 10.0, 10.0]])
    high = np.array([[20.0, 10.0, 10.0]])
    assert bool(tess.owns(low, 0)[0]) is True
    assert bool(tess.owns(high, 0)[0]) is True
    # The optimised path and the O(N log N) reference must agree.
    for X in (low, high):
        assert np.array_equal(tess.owns(X, 0), tess._owns_global(X, 0))
