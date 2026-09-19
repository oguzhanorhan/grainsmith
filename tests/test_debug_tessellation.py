"""Scientific-debug property tests for tessellation/seeding/rng (§6.4-6.8).

Regression-pins the defects found and fixed in this debug pass, plus a
partition-of-unity ("§6.8 contract") sweep across every tessellation
backend: every torus point (including its {-1,0,1}^p periodic lifts) must
be owned by EXACTLY ONE grain under owns(). This is the load-bearing
invariant the atom-fill stage depends on (fill uses owns() to generate
each lattice point exactly once with no wrap-and-deduplicate step).
"""
from __future__ import annotations

from itertools import product as iproduct

import numpy as np
import pytest

from grainsmith.rng import STAGE_NAMES, make_rng
from grainsmith.seeding import (
    lloyd_relax,
    seed_grains,
    wigner_seitz_radius,
)
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.power import PowerTessellation
from grainsmith.tessellation.sdot import _jacobian, _volumes, fit_power_weights
from grainsmith.tessellation.single import SingleCrystalTessellation
from grainsmith.tessellation.voxel_import import VoxelTessellation
from grainsmith.tessellation.warp import WarpTessellation, synthesize_grf
from grainsmith.tessellation.weighted import (
    AnisotropicTessellation,
    WeightedTessellation,
)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _owns_partition_ok(tess, X, L, periodic) -> tuple[int, np.ndarray]:
    """Sum owns(X + lift, i) over every grain i and every periodic lift;
    return (n_bad, counts). A correct backend gives counts == 1 everywhere.
    """
    n = tess.n_grains
    shifts_per_axis = [([-1, 0, 1] if periodic[ax] else [0]) for ax in range(3)]
    counts = np.zeros(len(X), dtype=int)
    for shift in iproduct(*shifts_per_axis):
        Xs = X + np.asarray(shift, dtype=np.float64) * L
        for i in range(n):
            counts += tess.owns(Xs, i).astype(int)
    bad = np.where(counts != 1)[0]
    return len(bad), counts


# ---------------------------------------------------------------------------
# §6.8 ownership partition-of-unity, every backend, periodic edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic", [
    [True, True, True], [True, True, False], [True, False, True],
])
def test_flat_owns_partition_of_unity(periodic):
    n, L = 10, np.array([40.0, 40.0, 40.0])
    seeds = seed_grains(n, L, periodic, _rng(1))
    tess = FlatTessellation(seeds, L, periodic)
    X = _rng(2).uniform(0.0, L, size=(3000, 3))
    n_bad, counts = _owns_partition_ok(tess, X, L, periodic)
    assert n_bad == 0, f"{n_bad} torus points not owned exactly once"


def test_flat_owns_partition_box_edges_and_corners():
    """Exact box-boundary points (min-image tie points) are owned once."""
    n, L = 10, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(1))
    tess = FlatTessellation(seeds, L, periodic)
    corners = np.array(list(iproduct([0.0, L[0]], [0.0, L[1]], [0.0, L[2]])))
    edges = np.array([[L[0] / 2, 0.0, 0.0], [0.0, L[1] / 2, 0.0],
                      [0.0, 0.0, L[2] / 2]])
    X = np.vstack([corners, edges])
    n_bad, counts = _owns_partition_ok(tess, X, L, periodic)
    assert n_bad == 0, f"counts={counts}"


def test_flat_owns_single_grain():
    """n=1 (trivial diagram, still must tile the torus exactly)."""
    L = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(1, L, periodic, _rng(11))
    tess = FlatTessellation(seeds, L, periodic)
    X = _rng(12).uniform(0.0, L, size=(500, 3))
    n_bad, _ = _owns_partition_ok(tess, X, L, periodic)
    assert n_bad == 0


def test_power_owns_partition_of_unity_weighted():
    n, L = 10, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(5))
    w = _rng(6).normal(0.0, 2.0, size=n)
    w -= w.mean()
    tess = PowerTessellation(seeds, L, periodic, weights=w)
    X = _rng(7).uniform(0.0, L, size=(2000, 3))
    n_bad, counts = _owns_partition_ok(tess, X, L, periodic)
    assert n_bad == 0, f"counts distribution: {np.unique(counts)}"


def test_power_owns_partition_mixed_periodicity():
    n, L = 10, np.array([40.0, 40.0, 40.0])
    periodic = [True, False, True]
    seeds = seed_grains(n, L, periodic, _rng(8))
    w = _rng(9).normal(0.0, 1.5, size=n)
    w -= w.mean()
    tess = PowerTessellation(seeds, L, periodic, weights=w)
    X = _rng(10).uniform(0.0, L, size=(1500, 3))
    n_bad, _ = _owns_partition_ok(tess, X, L, periodic)
    assert n_bad == 0


def test_weighted_and_anisotropic_owns_partition_of_unity():
    n, L = 8, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(13))

    tw = WeightedTessellation(seeds, L, periodic, sigma_w=1.0,
                              rng=_rng(14), min_seed_distance=6.0)
    X = _rng(15).uniform(0.0, L, size=(1500, 3))
    n_bad, _ = _owns_partition_ok(tw, X, L, periodic)
    assert n_bad == 0

    ta = AnisotropicTessellation(seeds, L, periodic, rng=_rng(16))
    n_bad, _ = _owns_partition_ok(ta, X, L, periodic)
    assert n_bad == 0


def test_single_crystal_owns_partition_orthogonal_and_triclinic():
    L = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    ts = SingleCrystalTessellation(L, periodic)
    X = _rng(17).uniform(0.0, L, size=(1500, 3))
    n_bad, _ = _owns_partition_ok(ts, X, L, periodic)
    assert n_bad == 0

    # triclinic box.cells lattice-multiple box: shift by H columns,
    # not diag(L), for the periodic lifts.
    H = np.array([[40.0, 5.0, 3.0], [0.0, 38.0, 2.0], [0.0, 0.0, 42.0]])
    ts_tri = SingleCrystalTessellation(L, periodic, cell_matrix=H)
    frac = _rng(18).uniform(0.0, 1.0, size=(1500, 3))
    Xtri = frac @ H.T
    n = ts_tri.n_grains
    shifts = list(iproduct([-1, 0, 1], repeat=3))
    counts = np.zeros(len(Xtri), dtype=int)
    for shift in shifts:
        Xs = Xtri + H @ np.asarray(shift, dtype=np.float64)
        for i in range(n):
            counts += ts_tri.owns(Xs, i).astype(int)
    assert np.all(counts == 1)


def test_single_crystal_free_axis_keeps_both_wall_layers():
    """§4b.1: a free axis is NOT periodic, so frac=0 and frac=1 are two
    distinct wall atom layers — owns() must keep BOTH, not dedup one."""
    L = np.array([40.0, 40.0, 40.0])
    ts = SingleCrystalTessellation(L, [True, True, False])
    lo = np.array([[5.0, 5.0, 0.0]])
    hi = np.array([[5.0, 5.0, 40.0]])
    assert bool(ts.owns(lo, 0)[0])
    assert bool(ts.owns(hi, 0)[0])


def test_voxel_import_owns_partition_of_unity():
    n, L = 8, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    shape = (20, 20, 20)
    r = _rng(19)
    seeds = r.uniform(0.0, L, size=(n, 3))
    xs = (np.arange(shape[0]) + 0.5) * (L[0] / shape[0])
    ys = (np.arange(shape[1]) + 0.5) * (L[1] / shape[1])
    zs = (np.arange(shape[2]) + 0.5) * (L[2] / shape[2])
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    d2 = np.zeros((len(pts), n))
    for i, s in enumerate(seeds):
        dr = pts - s
        dr -= L * np.round(dr / L)
        d2[:, i] = np.sum(dr**2, axis=1)
    labels = np.argmin(d2, axis=1).reshape(shape).astype(np.int32)
    tv = VoxelTessellation(labels, L, periodic)
    X = _rng(20).uniform(0.0, L, size=(1500, 3))
    n_bad, _ = _owns_partition_ok(tv, X, L, periodic)
    assert n_bad == 0


def test_voxel_import_owns_partition_wrap_spanning_grain():
    """A grain occupying two disjoint slabs at x<2 and x>=14 of a 16-wide
    grid (i.e. it straddles the periodic image boundary) must still tile
    the torus exactly once under owns()."""
    shape = (16, 16, 16)
    L = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    labels = np.ones(shape, dtype=np.int32)
    xs = np.arange(shape[0])
    labels[(xs < 2) | (xs >= 14), :, :] = 0
    tv = VoxelTessellation(labels, L, periodic)
    X = _rng(21).uniform(0.0, L, size=(2000, 3))
    n_bad, _ = _owns_partition_ok(tv, X, L, periodic)
    assert n_bad == 0


def test_warp_owns_partition_of_unity():
    n, L = 8, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(22))
    base = FlatTessellation(seeds, L, periodic)
    bundle = make_rng(123)
    warp = WarpTessellation(base, L, periodic, amplitude=1.0,
                            correlation_length=8.0, min_seed_distance=6.0,
                            rng=bundle.fields, connectivity_check=True)
    X = _rng(23).uniform(0.0, L, size=(1500, 3))
    n_bad, _ = _owns_partition_ok(warp, X, L, periodic)
    assert n_bad == 0


# ---------------------------------------------------------------------------
# fast owns() vs the O(N log N) reference (_owns_global) — every backend
# ---------------------------------------------------------------------------

def test_flat_fast_owns_matches_global_reference():
    n, L = 10, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(1))
    tess = FlatTessellation(seeds, L, periodic)
    X = _rng(2).uniform(0.0, L, size=(2000, 3))
    for i in range(n):
        assert np.array_equal(tess.owns(X, i), tess._owns_global(X, i))


def test_power_fast_owns_matches_global_reference():
    n, L = 8, np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(5))
    w = _rng(6).normal(0.0, 1.0, size=n)
    w -= w.mean()
    tess = PowerTessellation(seeds, L, periodic, weights=w)
    X = _rng(7).uniform(0.0, L, size=(2000, 3))
    for i in range(n):
        assert np.array_equal(tess.owns(X, i), tess._owns_global(X, i))


# ---------------------------------------------------------------------------
# SDOT Jacobian: analytic vs finite-difference (paper Claim: rel err ~1e-9)
# ---------------------------------------------------------------------------

def test_sdot_jacobian_matches_finite_difference():
    n, L = 8, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(30))
    w0 = _rng(31).normal(0.0, 0.5, size=n)
    w0 -= w0.mean()

    tess0 = PowerTessellation(seeds, L, periodic, weights=w0)
    J = _jacobian(tess0, n)
    assert abs(J.sum(axis=1)).max() < 1e-10, "Laplacian rows must sum to 0"

    eps = 1e-4
    J_fd = np.zeros((n, n))
    for j in range(n):
        wp, wm = w0.copy(), w0.copy()
        wp[j] += eps
        wm[j] -= eps
        Vp = _volumes(PowerTessellation(seeds, L, periodic, weights=wp))
        Vm = _volumes(PowerTessellation(seeds, L, periodic, weights=wm))
        J_fd[:, j] = (Vp - Vm) / (2 * eps)

    rel = np.abs(J - J_fd) / (np.abs(J_fd) + 1e-12)
    assert rel.max() < 1e-6, f"max rel err {rel.max():.2e} (paper claims ~1e-9)"


def test_sdot_fit_converges_on_lognormal_targets():
    from grainsmith.tessellation.sdot import sample_target_volumes

    n, L = 10, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(40))
    targets = sample_target_volumes("lognormal", n, float(L[0] ** 3),
                                    _rng(41), sigma_log=0.35)
    res = fit_power_weights(seeds, L, periodic, targets, vol_tol=1e-3,
                            max_iter=30)
    assert res.converged
    assert res.max_rel_error <= 1e-3


# ---------------------------------------------------------------------------
# GRF gaussian spectrum: zero-mean / unit-RMS regression (fix in this pass)
# ---------------------------------------------------------------------------

def test_grf_gaussian_is_zero_mean_and_unit_rms():
    """Defect fix: the k=0 (DC) mode was left IN the gaussian spectrum,
    so a random per-realization mean offset (a rigid translation that
    warps nothing) inflated std(u) — the calibration denominator — well
    above the fluctuating part's own RMS. The field must be exactly
    zero-mean (matching the self_affine convention) and exactly unit-RMS
    for every realization, not just in expectation."""
    L = np.array([50.0, 50.0, 50.0])
    for seed in range(8):
        field = synthesize_grf((32, 32, 32), 15.0, L, _rng(100 + seed),
                               spectrum="gaussian")
        for alpha in range(3):
            u = field[alpha]
            assert abs(float(np.mean(u))) < 1e-9, (
                f"seed={seed} alpha={alpha}: mean={np.mean(u):.3e} (DC mode "
                "leaking into the field — see warp.py synthesize_grf fix)")
            rms = float(np.sqrt(np.mean(u**2)))
            assert abs(rms - 1.0) < 1e-9, f"seed={seed} alpha={alpha}: rms={rms}"


def test_grf_gaussian_rms_not_inflated_by_dc_mode():
    """Before the fix, requesting amplitude=A delivered a fluctuating part
    with RMS well below A (up to ~2x low) for realistic (ell, L, grid)
    ratios, because part of the RMS budget was consumed by the harmless
    (non-warping) DC offset. After the fix std(u) == RMS(u) exactly, so
    amplitude-scaled fields deliver their full requested fluctuation."""
    L = np.array([50.0, 50.0, 50.0])
    ell, grid = 15.0, 32
    devs = []
    for seed in range(30):
        field = synthesize_grf((grid, grid, grid), ell, L, _rng(200 + seed),
                               spectrum="gaussian")
        for alpha in range(3):
            devs.append(float(np.sqrt(np.mean(field[alpha] ** 2))))
    devs = np.asarray(devs)
    assert np.max(np.abs(devs - 1.0)) < 1e-8, (
        f"RMS should be exactly 1.0 for every realization, got spread "
        f"{devs.min():.4f}..{devs.max():.4f}")


def test_grf_self_affine_unaffected_by_dc_fix():
    """self_affine already excludes k=0 by construction (band-limited);
    the gaussian-path DC fix must not change its numeric output."""
    L = np.array([80.0, 80.0, 80.0])
    f1 = synthesize_grf((32, 32, 32), 15.0, L, _rng(9), spectrum="self_affine",
                        hurst=0.8, l_min=8.0, l_max=30.0)
    f2 = synthesize_grf((32, 32, 32), 15.0, L, _rng(9), spectrum="self_affine",
                        hurst=0.8, l_min=8.0, l_max=30.0)
    np.testing.assert_array_equal(f1, f2)
    for alpha in range(3):
        assert abs(float(np.mean(f1[alpha]))) < 1e-12


# ---------------------------------------------------------------------------
# seeding.py: RSA min-distance, determinism, spatial-hash == brute-force
# ---------------------------------------------------------------------------

def test_seed_grains_deterministic_same_seed():
    L = np.array([60.0, 60.0, 60.0])
    s1 = seed_grains(20, L, [True, True, True], _rng(100))
    s2 = seed_grains(20, L, [True, True, True], _rng(100))
    assert np.array_equal(s1, s2)


def test_seed_grains_deterministic_with_lloyd():
    L = np.array([60.0, 60.0, 60.0])
    s1 = seed_grains(20, L, [True, True, True], _rng(101), lloyd_iterations=3)
    s2 = seed_grains(20, L, [True, True, True], _rng(101), lloyd_iterations=3)
    assert np.array_equal(s1, s2)


def _min_pairwise_pbc_distance(seeds, L, periodic):
    n = len(seeds)
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dr = seeds[i] - seeds[j]
            for ax in range(3):
                if periodic[ax]:
                    dr[ax] -= L[ax] * round(dr[ax] / L[ax])
            dmin = min(dmin, float(np.linalg.norm(dr)))
    return dmin


def test_rsa_respects_min_seed_distance_pbc():
    n, L = 30, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(200))
    r_ws = wigner_seitz_radius(float(np.prod(L)), n)
    dmin = _min_pairwise_pbc_distance(seeds, L, periodic)
    assert dmin >= r_ws - 1e-9


def test_rsa_respects_min_seed_distance_mixed_periodicity():
    n, L = 15, np.array([50.0, 50.0, 50.0])
    periodic = [True, False, True]
    seeds = seed_grains(n, L, periodic, _rng(201))
    r_ws = wigner_seitz_radius(float(np.prod(L)), n)
    dmin = _min_pairwise_pbc_distance(seeds, L, periodic)
    assert dmin >= r_ws - 1e-9
    assert seeds[:, 1].min() >= 0.0 and seeds[:, 1].max() <= L[1]


def _seed_grains_bruteforce(n, L, periodic, rng, min_seed_distance=None):
    L = np.asarray(L, dtype=np.float64)
    if min_seed_distance is None:
        min_seed_distance = wigner_seitz_radius(float(np.prod(L)), n)
    r2min = min_seed_distance**2
    sinv = np.where(periodic, 1.0 / L, 0.0)
    from grainsmith.constants import RNG_MAX_ATTEMPTS_FACTOR
    max_attempts = RNG_MAX_ATTEMPTS_FACTOR * n
    seeds = np.empty((n, 3))
    placed = 0
    for _ in range(max_attempts):
        if placed == n:
            break
        p = rng.uniform(0.0, 1.0, size=3) * L
        ok = True
        for k in range(placed):
            dr = p - seeds[k]
            dr = dr - np.round(dr * sinv) * L
            if np.dot(dr, dr) < r2min:
                ok = False
                break
        if ok:
            seeds[placed] = p
            placed += 1
    return seeds[:placed]


def test_rsa_spatial_hash_matches_bruteforce_reference():
    """The B4 spatial-hash grid must make the IDENTICAL accept/reject
    decision as the brute-force O(n^2) loop for every candidate, given
    the same RNG stream (seeding.py module docstring proof)."""
    n, L = 25, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    fast = seed_grains(n, L, periodic, _rng(500))
    brute = _seed_grains_bruteforce(n, L, periodic, _rng(500))
    assert np.array_equal(fast, brute)


def test_lloyd_relax_nearest_seed_matches_bruteforce_pbc():
    n, L = 12, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(600))
    from grainsmith.seeding import _nearest_seed_min_image

    X = _rng(601).uniform(0.0, L, size=(500, 3))
    labels = _nearest_seed_min_image(X, seeds, periodic, L)

    def brute_nearest(X, seeds, L):
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            dr = x[None, :] - seeds
            dr -= L * np.round(dr / L)
            out[i] = int(np.argmin(np.sum(dr**2, axis=1)))
        return out

    assert np.array_equal(labels, brute_nearest(X, seeds, L))


def test_lloyd_relax_stays_in_box_and_is_deterministic():
    n, L = 12, np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(n, L, periodic, _rng(700))
    r1 = lloyd_relax(seeds, L, periodic, iterations=4)
    r2 = lloyd_relax(seeds, L, periodic, iterations=4)
    assert np.array_equal(r1, r2)
    assert np.all(r1 >= 0.0) and np.all(r1 <= L)


# ---------------------------------------------------------------------------
# rng.py: stream independence + statelessness (no hidden global RNG state)
# ---------------------------------------------------------------------------

def test_rng_bundle_reproducible_from_seed():
    b1 = make_rng(7)
    b2 = make_rng(7)
    assert b1.seeding.random() == b2.seeding.random()


def test_rng_stage_streams_all_distinct():
    b = make_rng(42)
    vals = {name: getattr(b, name).random(5) for name in STAGE_NAMES}
    for a, c in [(a, c) for i, a in enumerate(STAGE_NAMES)
                 for c in STAGE_NAMES[i + 1:]]:
        assert not np.array_equal(vals[a], vals[c]), f"{a} == {c}"


def test_rng_occupancy_streams_reproducible():
    b1 = make_rng(7)
    s1 = b1.occupancy_streams(5)
    b2 = make_rng(7)
    s2 = b2.occupancy_streams(5)
    assert all(a.random() == b.random() for a, b in zip(s1, s2, strict=False))


def test_rng_occupancy_streams_stateless_call_count_independent():
    """occupancy_streams(k) must depend only on (seed, grain id), not on
    how many times or with what k it has previously been called — the
    per-grain SeedSequence is built from scratch each call (spawn_key
    extension), not via SeedSequence.spawn() (which counts children and
    would make the result depend on call history)."""
    b1 = make_rng(7)
    first3 = b1.occupancy_streams(3)
    b2 = make_rng(7)
    five = b2.occupancy_streams(5)
    assert all(a.random() == b.random() for a, b in zip(first3, five[:3], strict=False))


def test_rng_occupancy_and_doping_streams_distinct():
    b = make_rng(42)
    occ = b.occupancy_streams(3)
    dop = b.doping_streams(3)
    for i in range(3):
        assert occ[i].random() != dop[i].random()
