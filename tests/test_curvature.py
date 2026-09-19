"""GB curvature analysis (analysis/curvature.py) — kernel, sampling,
pipeline e2e."""
import csv as _csv
import multiprocessing

import numpy as np
import pytest

from grainsmith.analysis.curvature import (
    _STENCIL,
    CurvatureResult,
    PairCurvature,
    _hk_from_stencil,
    _pair_curvature,
    analyze_curvature,
    attach_curvature,
    global_curvature_rows,
    per_grain_gauss_bonnet,
)
from grainsmith.config.resolve import resolve_config   # NOTE: .resolve —
# grainsmith/config/__init__.py is empty; this matches test_end_to_end.py:9
from grainsmith.constants import (
    CURV_G16_DROP_TOL,
    CURV_G21_GAUSS_BONNET_TOL,
    MEMORY_HARD_LIMIT_BYTES,
)
from grainsmith.pipeline import run

SEED = 20260611


def _config(outdir, **overrides):
    # copied verbatim from tests/test_end_to_end.py::_config (lines 15-42)
    raw = {
        "meta": {"title": "e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu",
                               "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            for k2, v2 in sub.items():
                if isinstance(v2, dict) and isinstance(raw[key].get(k2),
                                                       dict):
                    raw[key][k2].update(v2)
                else:
                    raw[key][k2] = v2
        else:
            raw[key] = sub
    return resolve_config(raw)


def _phi_from_margins(pts, h_vec, margin_i, margin_j):
    """(M, 19) stencil values of φ = margin_i − margin_j around pts."""
    offsets = _STENCIL * np.asarray(h_vec, dtype=np.float64)[None, :]
    X = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return (margin_i(X) - margin_j(X)).reshape(len(pts), 19)


def _sphere_margins(R):
    """Grain i = ball of radius R at the origin; j = its complement."""
    def m_i(X):
        return R - np.linalg.norm(X, axis=1)

    def m_j(X):
        return np.linalg.norm(X, axis=1) - R
    return m_i, m_j


def _surface_points_sphere(R, n=64):
    """Deterministic points on the sphere r = R (golden-spiral)."""
    k = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * k / n
    r_xy = np.sqrt(1.0 - z**2)
    ang = np.pi * (1.0 + np.sqrt(5.0)) * k
    return R * np.stack([r_xy * np.cos(ang), r_xy * np.sin(ang), z], axis=1)


def test_kernel_sphere_H_and_K():
    R = 20.0
    h = np.array([R / 50.0] * 3)
    pts = _surface_points_sphere(R)
    m_i, m_j = _sphere_margins(R)
    phi = _phi_from_margins(pts, h, m_i, m_j)
    H, K, ok = _hk_from_stencil(phi, h)
    assert ok.all()
    # Sign contract: grain i is the ball => H = +1/R (grain i convex).
    np.testing.assert_allclose(H, 1.0 / R, rtol=1e-2)
    np.testing.assert_allclose(K, 1.0 / R**2, rtol=2e-2)


def test_kernel_cylinder():
    R = 15.0
    h = np.array([R / 50.0] * 3)
    ang = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    pts = np.stack([R * np.cos(ang), R * np.sin(ang),
                    np.linspace(-5, 5, 32)], axis=1)

    def m_i(X):
        return R - np.linalg.norm(X[:, :2], axis=1)

    def m_j(X):
        return np.linalg.norm(X[:, :2], axis=1) - R
    phi = _phi_from_margins(pts, h, m_i, m_j)
    H, K, ok = _hk_from_stencil(phi, h)
    assert ok.all()
    np.testing.assert_allclose(H, 1.0 / (2.0 * R), rtol=1e-2)
    np.testing.assert_allclose(K, 0.0, atol=1e-6)


def test_kernel_plane_is_zero():
    h = np.array([0.5, 0.5, 0.5])
    pts = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 0.5]])

    def m_i(X):
        return 4.0 - X[:, 0]

    def m_j(X):
        return X[:, 0] - 4.0
    phi = _phi_from_margins(pts, h, m_i, m_j)
    H, K, ok = _hk_from_stencil(phi, h)
    assert ok.all()
    np.testing.assert_allclose(H, 0.0, atol=1e-12)
    np.testing.assert_allclose(K, 0.0, atol=1e-12)


def test_kernel_degenerate_dropped():
    h = np.array([0.5, 0.5, 0.5])
    phi = np.zeros((3, 19))          # |grad| = 0 < CURV_GRAD_MIN
    H, K, ok = _hk_from_stencil(phi, h)
    assert not ok.any()
    np.testing.assert_array_equal(H, 0.0)
    np.testing.assert_array_equal(K, 0.0)


def test_kernel_anisotropic_h_vec():
    R = 20.0
    h = np.array([R / 50.0, R / 40.0, R / 60.0])   # per-axis steps
    pts = _surface_points_sphere(R, n=32)
    m_i, m_j = _sphere_margins(R)
    phi = _phi_from_margins(pts, h, m_i, m_j)
    H, K, ok = _hk_from_stencil(phi, h)
    assert ok.all()
    np.testing.assert_allclose(H, 1.0 / R, rtol=1e-2)


class _SphereTess:
    """Two-'grain' stub: grain 0 = ball of radius R centered in an
    L-box, grain 1 = complement.  Implements the Tessellation methods
    curvature and build_voxel_grid actually touch."""

    # build_voxel_grid reads tess.memory_limit_bytes (D1/D2) -- this stub
    # doesn't subclass Tessellation, so it needs the attribute explicitly.
    memory_limit_bytes = MEMORY_HARD_LIMIT_BYTES

    def __init__(self, R, L):
        self.R = R
        self.center = np.full(3, L / 2.0)
        self.n_grains = 2

    def _r(self, X):
        return np.linalg.norm(np.asarray(X, dtype=np.float64)
                              - self.center, axis=1)

    def margin(self, X, i):
        r = self._r(X)
        return (self.R - r) if i == 0 else (r - self.R)

    def grain_of(self, X):
        return np.where(self._r(X) < self.R, 0, 1).astype(np.int32)

    def adjacency(self):
        return [(0, 1)]


def test_pair_curvature_sphere_stub():
    R, L = 20.0, 60.0
    tess = _SphereTess(R, L)
    pts = _surface_points_sphere(R) + L / 2.0
    h_vec = np.array([R / 50.0] * 3)
    H, K, ok = _pair_curvature(tess, 0, 1, pts, h_vec)
    assert ok.all()
    np.testing.assert_allclose(H, 1.0 / R, rtol=1e-2)
    np.testing.assert_allclose(K, 1.0 / R**2, rtol=2e-2)


class _ShellTess:
    """Three-'grain' concentric-shell stub: grain 0 = ball r<R1, grain 1 =
    the shell R1<r<R2, grain 2 = outside r>R2 -- TWO boundary pairs,
    unlike _SphereTess's one. _curvature_voxel falls back to serial for
    <=1 pair regardless of jobs, so this is the minimal fixture that
    actually exercises the jobs>1 ProcessPoolExecutor dispatch."""

    # build_voxel_grid reads tess.memory_limit_bytes (D1/D2) -- this stub
    # doesn't subclass Tessellation, so it needs the attribute explicitly.
    memory_limit_bytes = MEMORY_HARD_LIMIT_BYTES

    def __init__(self, R1, R2, L):
        self.R1, self.R2 = R1, R2
        self.center = np.full(3, L / 2.0)
        self.n_grains = 3

    def _r(self, X):
        return np.linalg.norm(np.asarray(X, dtype=np.float64)
                              - self.center, axis=1)

    def margin(self, X, i):
        r = self._r(X)
        if i == 0:
            return self.R1 - r
        if i == 1:
            return np.minimum(r - self.R1, self.R2 - r)
        return r - self.R2

    def grain_of(self, X):
        r = self._r(X)
        return np.where(r < self.R1, 0,
                        np.where(r < self.R2, 1, 2)).astype(np.int32)

    def adjacency(self):
        return [(0, 1), (1, 2)]


def _assert_pairs_byte_identical(pairs1, pairs2):
    """p1/p2 arrays equal both by value AND by raw bytes.

    assert_array_equal alone would pass on e.g. -0.0 == 0.0 (IEEE 754:
    the two compare equal) or on NaN == NaN (numpy's testing helpers
    special-case NaN as equal for exactly this kind of comparison), so
    neither proves the "bit-identical for every jobs value" contract
    (module docstring / analyze_curvature's jobs paragraph). .tobytes()
    compares the raw representation instead, so -0.0 vs 0.0 or any NaN
    payload/sign difference would fail it even though it passes
    assert_array_equal.
    """
    assert set(pairs1) == set(pairs2)
    for key in pairs1:
        p1, p2 = pairs1[key], pairs2[key]
        np.testing.assert_array_equal(p1.H, p2.H)
        np.testing.assert_array_equal(p1.K, p2.K)
        np.testing.assert_array_equal(p1.areas, p2.areas)
        np.testing.assert_array_equal(p1.points, p2.points)
        assert p1.H.tobytes() == p2.H.tobytes()
        assert p1.K.tobytes() == p2.K.tobytes()
        assert p1.areas.tobytes() == p2.areas.tobytes()
        assert p1.points.tobytes() == p2.points.tobytes()


def test_analyze_curvature_jobs_parallel_matches_serial():
    """jobs=2 must reproduce the jobs=1 result bit-for-bit: per-pair
    purity + ordered reassembly (module docstring / analyze_curvature's
    jobs paragraph) means H, K, areas, points, n_raw and n_dropped are
    all identical regardless of jobs."""
    from grainsmith.tessellation.voxel import build_voxel_grid
    R1, R2, L = 12.0, 20.0, 60.0
    tess = _ShellTess(R1, R2, L)
    box = np.array([L, L, L])
    vg = build_voxel_grid(tess, box, grid_size=48)
    res1 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=1)
    res2 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=2)
    assert set(res1.pairs) == set(res2.pairs) == {(0, 1), (1, 2)}
    assert res1.n_raw == res2.n_raw
    assert res1.n_dropped == res2.n_dropped
    _assert_pairs_byte_identical(res1.pairs, res2.pairs)


@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason=(
        "needs the 'fork' start method: this test patches "
        "grainsmith.analysis.curvature.CURV_GRAD_MIN, a module global that "
        "the jobs=2 workers read inside _hk_from_stencil. Fork inherits the "
        "patched module; spawn (Windows, macOS) re-imports it and reads the "
        "real value, so serial and parallel would legitimately disagree. "
        "Propagating the patch into a spawned worker would require a "
        "production test hook, which is not acceptable — jobs-invariance "
        "itself stays covered on spawn platforms by "
        "test_analyze_curvature_jobs_parallel_matches_serial (unpatched) and "
        "..._mixed_empty_pair (patches a parent-side instance attribute)."
    ),
)
def test_analyze_curvature_jobs_parallel_matches_serial_degenerate_mask(
        monkeypatch):
    """The test above never exercises a non-trivial degenerate mask
    (n_dropped == 0 for both pairs in every case it builds). Force a
    REAL partial drop instead: measured directly, this exact
    _ShellTess/grid_size=48 fixture shows |grad phi| sits in a narrow
    band per pair (pair (0,1) in [1.9925, 1.9999], pair (1,2) in
    [1.99737, 1.99998]), so CURV_GRAD_MIN=1.998 splits BOTH pairs
    into a genuine ok/dropped mix (measured: pair (0,1) 240 kept /
    1464 dropped; pair (1,2) 3288 kept / 1584 dropped) rather than an
    all-or-nothing mask.

    CURV_GRAD_MIN is imported into curvature.py's own namespace at
    import time (``from grainsmith.constants import CURV_GRAD_MIN``),
    and _hk_from_stencil reads that bare module-global name directly —
    so the patch must land on grainsmith.analysis.curvature.
    CURV_GRAD_MIN, NOT grainsmith.constants.CURV_GRAD_MIN (the latter
    would leave curvature.py's already-bound reference untouched).
    This test REQUIRES the 'fork' start method and is skipped otherwise
    (see the skipif above). Under fork the jobs=2 workers inherit the
    patched module attribute from the parent, matching jobs=1's in-process
    call; under spawn they re-import curvature.py and read the real
    CURV_GRAD_MIN, so the two paths would differ for a reason that has
    nothing to do with the jobs-invariance this test exists to check.
    """
    import grainsmith.analysis.curvature as curvature_mod
    from grainsmith.tessellation.voxel import build_voxel_grid

    monkeypatch.setattr(curvature_mod, "CURV_GRAD_MIN", 1.998)
    R1, R2, L = 12.0, 20.0, 60.0
    tess = _ShellTess(R1, R2, L)
    box = np.array([L, L, L])
    vg = build_voxel_grid(tess, box, grid_size=48)
    res1 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=1)
    res2 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=2)

    assert set(res1.pairs) == set(res2.pairs) == {(0, 1), (1, 2)}
    assert res1.n_raw == res2.n_raw
    assert res1.n_dropped == res2.n_dropped
    assert res1.n_dropped > 0 and res2.n_dropped > 0
    # a genuine MIX, not every sample dropped nor every sample kept.
    for key in res1.pairs:
        assert 0 < res1.pairs[key].H.size < res1.n_raw
    _assert_pairs_byte_identical(res1.pairs, res2.pairs)


def test_analyze_curvature_jobs_parallel_matches_serial_mixed_empty_pair(
        monkeypatch):
    """Item 6: empty-pts pairs are skipped when building parallel worker
    tasks (nothing for a worker to do), which means the positional
    reassembly against the OTHER, non-empty pairs must still line up.
    Force pair (1, 2)'s sample set to genuinely empty (real voxel-face
    extraction on this fixture never produces an empty pair on its own,
    so this is crafted directly, per the item's own suggestion) while
    pair (0, 1) keeps its normal non-empty samples, and confirm jobs=1
    vs jobs=2 still agree bit-for-bit -- including n_raw/n_dropped and
    the empty pair's own (zero-length) arrays."""
    from grainsmith.tessellation.voxel import build_voxel_grid
    R1, R2, L = 12.0, 20.0, 60.0
    tess = _ShellTess(R1, R2, L)
    box = np.array([L, L, L])
    vg = build_voxel_grid(tess, box, grid_size=48)
    real_faces = vg.gb_face_samples([True, True, True])
    assert set(real_faces) == {(0, 1), (1, 2)}   # sanity: both non-empty
    forced_faces = dict(real_faces)
    forced_faces[(1, 2)] = (np.empty((0, 3)), np.empty(0))
    monkeypatch.setattr(vg, "gb_face_samples",
                        lambda periodic=None: forced_faces)

    res1 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=1)
    res2 = analyze_curvature(tess, box, [True, True, True], vg=vg, jobs=2)

    assert set(res1.pairs) == set(res2.pairs) == {(0, 1), (1, 2)}
    assert res1.pairs[(1, 2)].H.size == 0
    assert res2.pairs[(1, 2)].H.size == 0
    assert res1.pairs[(0, 1)].H.size > 0
    assert res2.pairs[(0, 1)].H.size > 0
    assert res1.n_raw == res2.n_raw == len(real_faces[(0, 1)][0])
    assert res1.n_dropped == res2.n_dropped
    _assert_pairs_byte_identical(res1.pairs, res2.pairs)


def test_gb_face_samples_public_accessor():
    from grainsmith.tessellation.voxel import build_voxel_grid
    R, L = 20.0, 60.0
    tess = _SphereTess(R, L)
    vg = build_voxel_grid(tess, np.array([L, L, L]), grid_size=48)
    samples = vg.gb_face_samples([True, True, True])
    assert set(samples) == {(0, 1)}
    pts, areas = samples[(0, 1)]
    assert pts.shape[1] == 3 and len(pts) == len(areas)
    # Voxel-face area estimator: total within the staircase bound
    # [A_true, sqrt(3)*A_true] of the sphere area.
    a_true = 4.0 * np.pi * R**2
    assert a_true * 0.9 <= float(areas.sum()) <= a_true * np.sqrt(3) * 1.1
    # End-to-end on real sample points: area-weighted <|H|> ≈ 1/R.
    H, K, ok = _pair_curvature(tess, 0, 1, pts, vg.h_vec)
    w = areas[ok]
    h_abs = float(np.sum(w * np.abs(H[ok])) / np.sum(w))
    np.testing.assert_allclose(h_abs, 1.0 / R, rtol=0.05)


def test_analyze_curvature_voxel_path_sphere():
    from grainsmith.tessellation.voxel import build_voxel_grid
    R, L = 20.0, 60.0
    tess = _SphereTess(R, L)
    box = np.array([L, L, L])
    vg = build_voxel_grid(tess, box, grid_size=48)
    res = analyze_curvature(tess, box, [True, True, True], vg=vg)
    assert set(res.pairs) == {(0, 1)}
    pc = res.pairs[(0, 1)]
    assert res.n_raw == len(pc.H) + res.n_dropped
    assert res.dropped_fraction < 0.05
    # all retained points wrapped to [0, L)
    assert (pc.points >= 0.0).all() and (pc.points < L).all()
    w = pc.areas
    h_mean = float(np.sum(w * pc.H) / np.sum(w))
    np.testing.assert_allclose(h_mean, 1.0 / R, rtol=0.05)


def test_attach_curvature_stats_and_empty_pair():
    from grainsmith.analysis.boundaries import BoundaryReport

    def _report(i, j):
        return BoundaryReport(
            grain_i=i, grain_j=j, seed_distance_A=1.0,
            misorientation_deg=10.0, axis_u=1, axis_v=0, axis_w=0,
            axis_dev_deg=0.0, area_A2=1.0, mean_normal_x=1.0,
            mean_normal_y=0.0, mean_normal_z=0.0, normal_spread_deg=0.0,
            plane_i_hkl="(100)", plane_i_dev_deg=0.0, plane_j_hkl="(100)",
            plane_j_dev_deg=0.0, character="twist",
            character_angle_deg=0.0, csl_sigma="", n_overlap_deleted=0)

    r01, r02 = _report(0, 1), _report(0, 2)
    res = CurvatureResult(
        pairs={(0, 1): PairCurvature(
            points=np.zeros((2, 3)),
            areas=np.array([1.0, 3.0]),
            H=np.array([0.1, -0.1]),
            K=np.array([0.01, 0.01]))},
        n_raw=2, n_dropped=0)
    attach_curvature([r01, r02], res)
    # area-weighted: H_mean = (1*0.1 + 3*(-0.1))/4 = -0.05
    assert abs(r01.H_mean_invA - (-0.05)) < 1e-12
    assert abs(r01.H_abs_mean_invA - 0.1) < 1e-12
    # std: sqrt((1*(0.15)^2 + 3*(0.05)^2)/4) = sqrt(0.0075)
    assert abs(r01.H_std_invA - np.sqrt(0.0075)) < 1e-12
    assert abs(r01.K_mean_invA2 - 0.01) < 1e-12
    assert r01.curv_n_samples == 2
    # pair with no samples -> NaN stats, 0 count
    assert np.isnan(r02.H_mean_invA) and r02.curv_n_samples == 0


def test_analyze_curvature_empty_adjacency():
    """Single-crystal runs: no pairs, no crash, empty result (vg=None)."""

    class _NoPairs:
        def adjacency(self):
            return []

    res = analyze_curvature(_NoPairs(), np.array([40.0] * 3),
                            [True, True, True], vg=None)
    assert res.pairs == {} and res.n_raw == 0 and res.n_dropped == 0
    assert res.dropped_fraction == 0.0


def test_global_rows_shape():
    res = CurvatureResult(
        pairs={(0, 1): PairCurvature(
            points=np.zeros((1, 3)), areas=np.array([2.0]),
            H=np.array([0.05]), K=np.array([0.0025]))},
        n_raw=2, n_dropped=1)
    rows = global_curvature_rows(res)
    keys = [k for (_, k, _) in rows]
    assert keys == ["H_mean_invA", "H_abs_mean_invA", "K_mean_invA2",
                    "n_samples", "dropped_fraction"]
    assert all(section == "curvature" for (section, _, _) in rows)
    assert dict((k, v) for (_, k, v) in rows)["dropped_fraction"] == 0.5


# ---------------------------------------------------------------------------
# per_grain_gauss_bonnet (G21 input) — synthetic aggregation + analytic
# closed-surface anchor.
# ---------------------------------------------------------------------------


def test_per_grain_gauss_bonnet_flat_is_zero():
    """Flat geometry's exact H=K=0 samples (module docstring: 'flat runs
    never reach [the level-set] kernel ... writes literal zeros') must
    integrate to exactly 0 for every grain — the G21 negative control."""
    res = CurvatureResult(
        pairs={(0, 1): PairCurvature(
            points=np.zeros((3, 3)), areas=np.array([10.0, 20.0, 5.0]),
            H=np.zeros(3), K=np.zeros(3))},
        n_raw=3, n_dropped=0)
    totals = per_grain_gauss_bonnet(res, n_grains=2)
    assert totals.shape == (2,)
    np.testing.assert_array_equal(totals, [0.0, 0.0])


def test_per_grain_gauss_bonnet_closed_sphere_hits_4pi():
    """Analytic anchor: sample K = 1/R² over an area that sums to the
    FULL closed-sphere area 4πR² (the ENTIRE surface treated as
    face-interior, i.e. the pathological case with no excluded
    edges/vertices at all) must integrate to exactly 4π on BOTH sides of
    the pair — K is invariant under the (i, j) flip (module docstring),
    so grain 0 and grain 1 (its complement) each see the full budget.
    This is the theoretical ceiling gate_g21_gauss_bonnet's 4π tolerance
    is pinned to (constants.CURV_G21_GAUSS_BONNET_TOL docstring)."""
    R = 20.0
    area_total = 4.0 * np.pi * R**2
    n = 5
    pc = PairCurvature(
        points=np.zeros((n, 3)),
        areas=np.full(n, area_total / n),
        H=np.full(n, 1.0 / R),
        K=np.full(n, 1.0 / R**2))
    res = CurvatureResult(pairs={(0, 1): pc}, n_raw=n, n_dropped=0)
    totals = per_grain_gauss_bonnet(res, n_grains=2)
    np.testing.assert_allclose(totals, [4.0 * np.pi, 4.0 * np.pi], rtol=1e-12)
    assert abs(totals[0] - CURV_G21_GAUSS_BONNET_TOL) < 1e-9


def test_per_grain_gauss_bonnet_multi_pair_accumulates_per_grain():
    """Three grains, two shared boundaries: grain 1 (in both pairs) sums
    BOTH contributions; grain 2 (no retained samples) reports 0."""
    res = CurvatureResult(
        pairs={
            (0, 1): PairCurvature(
                points=np.zeros((2, 3)), areas=np.array([1.0, 3.0]),
                H=np.zeros(2), K=np.array([0.1, 0.2])),
            (1, 3): PairCurvature(
                points=np.zeros((1, 3)), areas=np.array([2.0]),
                H=np.zeros(1), K=np.array([-0.5])),
            (2, 3): PairCurvature(       # empty pair (all samples dropped)
                points=np.empty((0, 3)), areas=np.empty(0),
                H=np.empty(0), K=np.empty(0)),
        },
        n_raw=4, n_dropped=1)
    totals = per_grain_gauss_bonnet(res, n_grains=4)
    assert totals.shape == (4,)
    # grain 0: 1*0.1 + 3*0.2 = 0.7
    assert abs(totals[0] - 0.7) < 1e-12
    # grain 1: (1*0.1 + 3*0.2) + (2*-0.5) = 0.7 - 1.0 = -0.3
    assert abs(totals[1] - (-0.3)) < 1e-12
    # grain 2: only the empty pair -> 0
    assert totals[2] == 0.0
    # grain 3: 2*-0.5 = -1.0
    assert abs(totals[3] - (-1.0)) < 1e-12


def test_per_grain_gauss_bonnet_empty_adjacency():
    """Single-crystal / no-boundary runs: n_grains fixes the output
    length even with zero pairs — no crash, all zeros."""
    res = CurvatureResult(pairs={}, n_raw=0, n_dropped=0)
    totals = per_grain_gauss_bonnet(res, n_grains=1)
    np.testing.assert_array_equal(totals, [0.0])


# ---------------------------------------------------------------------------
# Pipeline e2e (config field, wiring, files) — Task 7
# ---------------------------------------------------------------------------


def test_e2e_flat_curvature_exact_zeros(tmp_path):
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True})
    res = run(cfg)
    assert res.gates.all_passed()
    assert "G16" in {r.gate for r in res.gates.results()}
    # G21 (Gauss-Bonnet face-interior residual): flat geometry's exact
    # H=K=0 samples integrate to exactly 0 per grain -> reports 'ok',
    # never a WARN (the flat-Voronoi negative control for this gate).
    g21 = next(r for r in res.gates.results() if r.gate == "G21")
    assert g21.passed and g21.measured == 0.0
    assert "WARN" not in g21.message

    with open(tmp_path / "out" / "boundaries.csv", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows, "expected boundary rows"
    for r in rows:
        assert r["H_mean_invA"] == "0.0"
        assert r["H_std_invA"] == "0.0"
        assert r["H_abs_mean_invA"] == "0.0"
        assert r["K_mean_invA2"] == "0.0"
        assert r["K_std_invA2"] == "0.0"
        assert int(r["curv_n_samples"]) >= 1

    with open(tmp_path / "out" / "gb_curvature.csv", newline="") as fh:
        srows = list(_csv.DictReader(fh))
    assert srows
    for r in srows:
        assert r["H_invA"] == "0.0" and r["K_invA2"] == "0.0"
        for key in ("x", "y", "z"):
            assert 0.0 <= float(r[key]) < 40.0
    # gb_curvature.csv listed in MANIFEST.txt
    manifest = (tmp_path / "out" / "MANIFEST.txt").read_text()
    assert "gb_curvature.csv" in manifest

    import json
    micro = json.loads(
        (tmp_path / "out" / "microstructure.json").read_text())
    b0 = micro["boundaries"][0]
    assert "H_mean_invA" in b0 and b0["H_mean_invA"] == 0.0


def test_e2e_flat_default_off_unchanged(tmp_path):
    cfg = _config(tmp_path / "out")
    res = run(cfg)
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G16" not in gate_ids
    assert "G21" not in gate_ids   # gb_curvature off -> G21 never runs
    with open(tmp_path / "out" / "boundaries.csv", newline="") as fh:
        header = fh.readline().strip().split(",")
    assert "H_mean_invA" not in header
    assert not (tmp_path / "out" / "gb_curvature.csv").exists()


# Boundaries override copied verbatim from tests/test_end_to_end.py's
# curved variant (test_curved_warp_all_gates_and_outputs, lines ~124-130).
# amplitude retuned 2.0 -> 0.6 (warp.py DC-mode-exclusion fix, tessellation
# debug wave: excluding the k=0 mode from the gaussian spectrum removes a
# rigid-translation contribution that used to inflate std(u) and dilute the
# genuine-waviness fraction of a given amplitude budget, so the SAME
# amplitude now delivers a larger max‖∇u‖; retuned so G6 keeps passing on
# this box/grain/correlation_length combination).
CURVED_BOUNDARIES_OVERRIDE = {
    "geometry": "curved",
    "curved": {"method": "warp", "amplitude": 0.6,
              "correlation_length": 10.0},
}


def test_e2e_curved_curvature(tmp_path):
    # boundaries override copied verbatim from
    # tests/test_end_to_end.py's curved variant (lines ~124-130).
    cfg = _config(tmp_path / "out",
                  analysis={"gb_curvature": True},
                  boundaries=CURVED_BOUNDARIES_OVERRIDE)
    res = run(cfg)
    assert res.gates.all_passed()
    # G16 is warn-only (passed=True always), so assert the MEASURED value:
    g16 = next(r for r in res.gates.results() if r.gate == "G16")
    assert g16.measured <= CURV_G16_DROP_TOL
    # G21 (Gauss-Bonnet face-interior residual): also warn-only, and
    # ALWAYS passed=True regardless of the measured value (see
    # gate_g21_gauss_bonnet docstring: the residual is a coarse,
    # non-resolution-convergent signal, not a tight zero-residual test
    # — calibration showed even mild curved configs routinely land on
    # either side of the 4π line depending on seed/geometry). This
    # SPECIFIC fixed-seed config happens to measure under the tolerance
    # (measured directly, reproducible byte-for-byte at this seed),
    # which exercises the PASS/'ok' message branch; the companion WARN
    # branch is exercised separately by
    # test_e2e_perturbed_distance_g21_warn's more aggressive config.
    g21 = next(r for r in res.gates.results() if r.gate == "G21")
    assert g21.passed  # always True, warn-only, independent of measured
    assert g21.measured < CURV_G21_GAUSS_BONNET_TOL
    assert "WARN" not in g21.message
    with open(tmp_path / "out" / "gb_curvature.csv", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows
    hs = np.array([float(r["H_invA"]) for r in rows])
    assert np.isfinite(hs).all()
    assert np.abs(hs).max() > 0.0          # genuinely curved
    with open(tmp_path / "out" / "boundaries.csv", newline="") as fh:
        brows = list(_csv.DictReader(fh))
    for r in brows:
        assert int(r["curv_n_samples"]) >= 0


def test_e2e_curved_rerun_byte_identical_across_jobs(tmp_path):
    """Spec: bit-identical outputs across --jobs values ON THE CURVED
    kernel path (a flat run writes literal zeros and proves nothing)."""
    cfg_a = _config(tmp_path / "a", analysis={"gb_curvature": True},
                    boundaries=CURVED_BOUNDARIES_OVERRIDE)
    cfg_b = _config(tmp_path / "b", analysis={"gb_curvature": True},
                    boundaries=CURVED_BOUNDARIES_OVERRIDE)
    run(cfg_a, jobs=1)
    run(cfg_b, jobs=2)
    for name in ("gb_curvature.csv", "boundaries.csv"):
        assert ((tmp_path / "a" / name).read_bytes()
                == (tmp_path / "b" / name).read_bytes())


# G21 end-to-end WARN trigger: a small, fast perturbed_distance run whose
# amplitude sits close to the seed-containment ceiling (A_max =
# PERTURBED_DISTANCE_SAFETY * min_seed_distance / (2*ETA_CLIP), see
# constants.py) reliably lands the worst-grain face-interior residual
# above the 4π tolerance (measured directly at this exact config:
# worst per-grain |∮K dA| ≈ 30.4 vs 4π ≈ 12.57 — NOTE per the gate's own
# calibration, this is not unique to aggressive configs: even mild
# curved runs land on either side of 4π depending on seed/geometry, see
# gate_g21_gauss_bonnet/CURV_G21_GAUSS_BONNET_TOL docstrings; this
# config was chosen simply because it reproducibly WARNs, to exercise
# that message branch deterministically). 4 grains / 40 Å box /
# grid_size="auto" keeps this under ~3 s so it is safe to run in the
# default test suite.
G21_WARN_BOUNDARIES_OVERRIDE = {
    "geometry": "curved",
    "curved": {"method": "perturbed_distance", "amplitude": 1.0,
              "spectrum": "self_affine", "hurst": 0.9,
              "l_min": 5.0, "l_max": 13.0},
}


def test_e2e_perturbed_distance_g21_warn(tmp_path):
    """A deliberately aggressive perturbed_distance run trips G21's WARN
    branch end-to-end (never G16 hard-failing, never the run itself
    failing — G21 is warn-only and diagnoses the ESTIMATOR): the pipeline
    completes, all gates still report passed=True, and G21's message +
    measured value carry the violation."""
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True},
                  grains={"number": 4},
                  boundaries=G21_WARN_BOUNDARIES_OVERRIDE)
    res = run(cfg)
    assert res.gates.all_passed()          # warn-only: never hard-fails
    gate_ids = {r.gate for r in res.gates.results()}
    assert {"G16", "G19", "G20", "G21"} <= gate_ids

    g21 = next(r for r in res.gates.results() if r.gate == "G21")
    assert g21.passed
    assert g21.measured > CURV_G21_GAUSS_BONNET_TOL
    assert "WARN" in g21.message
    assert "grain" in g21.message          # names the worst-offending grain

    # Reported in summary.csv exactly like every other gate (existing
    # 'gates' section convention, PASS|measured=...|message format; the
    # value is RFC 4180-quoted because the message itself contains
    # commas, same as every other multi-clause gate message).
    summary = (tmp_path / "out" / "summary.csv").read_text(encoding="utf-8")
    assert 'gates,G21,"PASS | measured=' in summary
    assert "WARN" in summary.split("gates,G21,")[1].splitlines()[0]

    # And in microstructure.json's 'gates' list (same GateResult fields
    # every other gate uses — io/publication.py).
    import json
    micro = json.loads(
        (tmp_path / "out" / "microstructure.json").read_text())
    g21_json = next(g for g in micro["gates"] if g["gate"] == "G21")
    assert g21_json["passed"] is True
    assert abs(g21_json["measured"] - g21.measured) < 1e-9


def test_e2e_perturbed_distance_g21_reproducible(tmp_path):
    """Same seed -> bit-identical G21 measured value (reproducibility is
    physics, not polish — project mandate): re-running the exact WARN
    config from test_e2e_perturbed_distance_g21_warn twice must agree."""
    cfg_a = _config(tmp_path / "a", analysis={"gb_curvature": True},
                    grains={"number": 4},
                    boundaries=G21_WARN_BOUNDARIES_OVERRIDE)
    cfg_b = _config(tmp_path / "b", analysis={"gb_curvature": True},
                    grains={"number": 4},
                    boundaries=G21_WARN_BOUNDARIES_OVERRIDE)
    res_a = run(cfg_a)
    res_b = run(cfg_b)
    g21_a = next(r for r in res_a.gates.results() if r.gate == "G21")
    g21_b = next(r for r in res_b.gates.results() if r.gate == "G21")
    assert g21_a.measured == g21_b.measured
    assert g21_a.message == g21_b.message


def test_e2e_single_crystal_curvature(tmp_path):
    """grains.number == 1 → SingleCrystalTessellation (not a
    FlatTessellation, vg=None): empty adjacency must yield a header-only
    gb_curvature.csv and G16 = 0/0, not a crash."""
    cfg = _config(tmp_path / "out", grains={"number": 1},
                  analysis={"gb_curvature": True})
    res = run(cfg)
    g16 = next(r for r in res.gates.results() if r.gate == "G16")
    assert g16.measured == 0.0
    # G21: empty adjacency (no boundary at all) must also not crash and
    # must report the same trivial 'ok' as G16 does for 0/0.
    g21 = next(r for r in res.gates.results() if r.gate == "G21")
    assert g21.passed and g21.measured == 0.0
    assert "WARN" not in g21.message
    lines = (tmp_path / "out" / "gb_curvature.csv").read_text().splitlines()
    assert len(lines) == 1                 # header only


# Minimal two-phase config, copied from ALPHA/BETA in tests/test_phases.py
# (== examples/alloys_multiphase/ti_alpha_beta.yaml): no multiphase e2e exists in
# tests/test_end_to_end.py, so this sources the phases/crystal blocks from
# there instead.  'crystal' is set to None to satisfy
# resolve Rule 21 ('crystal' and 'phases' are mutually exclusive).
MULTIPHASE_OVERRIDES = {
    "crystal": None,
    "grains": {"number": 6},
    "orientation": {"scheme": "random_uniform"},
    "phases": [
        {"name": "alpha", "fraction": 0.6, "crystal": {
            "space_group": {"number": 194},
            "lattice": {"a": 2.951, "c": 4.684},
            "wyckoff_sites": [{"element": "Ti",
                               "coords": [1.0 / 3.0, 2.0 / 3.0, 0.25],
                               "letter": "c"}],
        }},
        {"name": "beta", "fraction": 0.4, "crystal": {
            "space_group": {"number": 229},
            "lattice": {"a": 3.32},
            "wyckoff_sites": [{"element": "Ti", "coords": [0.0, 0.0, 0.0]}],
        }},
    ],
}


def test_e2e_multiphase_curvature_columns(tmp_path):
    """Exercise the multiphase + curvature column-selection branch."""
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True},
                  **MULTIPHASE_OVERRIDES)
    run(cfg)
    from grainsmith.io.reports import BOUNDARIES_COLUMNS_PHASES_CURVATURE
    with open(tmp_path / "out" / "boundaries.csv", newline="") as fh:
        header = fh.readline().strip().split(",")
    assert header == BOUNDARIES_COLUMNS_PHASES_CURVATURE


def test_resolve_rejects_voxel_import_curvature(tmp_path):
    import pytest as _pytest

    from grainsmith.errors import ConfigError

    raw_over = {"analysis": {"gb_curvature": True},
                "boundaries": {"geometry": "voxel_import"}}
    with _pytest.raises(ConfigError, match="voxel_import"):
        _config(tmp_path / "out", **raw_over)


# ---------------------------------------------------------------------------
# METHODS.md paragraph + curvature_hist.plt gnuplot emitter — Task 8
# ---------------------------------------------------------------------------


def test_e2e_curvature_methods_and_gnuplot(tmp_path):
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True})
    run(cfg)
    methods = (tmp_path / "out" / "METHODS.md").read_text()
    assert "curvature" in methods.lower()
    assert "1/(2h)" in methods                  # resolution limit stated
    assert "GateResult(" not in methods         # no dataclass repr leak
    assert "None" not in methods.split("G16")[1][:40]
    plt = tmp_path / "out" / "curvature_hist.plt"
    assert plt.exists()
    text = plt.read_text()
    assert text.startswith("# ")                    # provenance line
    assert "gb_curvature.csv" in text
    assert "grain = -1" in text                     # per-grain filter knob
    manifest = (tmp_path / "out" / "MANIFEST.txt").read_text()
    assert "curvature_hist.plt" in manifest


def test_e2e_curvature_methods_reports_g21(tmp_path):
    """G21 (Gauss-Bonnet face-interior residual) is registered by the
    pipeline alongside G16 whenever analysis.gb_curvature is set
    (pipeline.py), but METHODS.md used to never mention it at all —
    regression coverage for that drift fix."""
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True})
    res = run(cfg)
    assert any(r.gate == "G21" for r in res.gates.results())
    methods = (tmp_path / "out" / "METHODS.md").read_text()
    assert "gate G21" in methods
    assert "Gauss-Bonnet" in methods
    assert "GateResult(" not in methods
    assert "None" not in methods.split("gate G21")[1][:60]


def test_e2e_default_off_no_gnuplot_hist(tmp_path):
    cfg = _config(tmp_path / "out")
    run(cfg)
    assert not (tmp_path / "out" / "curvature_hist.plt").exists()


# ---------------------------------------------------------------------------
# Curved-boundary backend smoke tests (WP1 follow-up): the pre-existing
# curved e2e coverage above (test_e2e_curved_curvature et al.) only
# exercises boundaries.curved.method 'warp'.  These close the gap for the
# other two curved backends, additive_weights and anisotropic — both go
# through the same voxel gb_curvature path but can produce planar/
# near-planar boundaries, so only finiteness of H is asserted (not
# nonzero, unlike the warp case).
#
# additive_weights.weight_sigma=1.0: the standard _config box (40^3, 4
# grains) auto-resolves min_seed_distance = wigner_seitz_radius(64000, 4)
# ≈ 15.63 Å, so the Rule 9 empty-cell guard limit (min_seed_distance/6 ≈
# 2.61 Å) comfortably allows weight_sigma=1.0.
# ---------------------------------------------------------------------------

ADDITIVE_WEIGHTS_BOUNDARIES_OVERRIDE = {
    "geometry": "curved",
    "curved": {"method": "additive_weights", "weight_sigma": 1.0},
}

ANISOTROPIC_BOUNDARIES_OVERRIDE = {
    "geometry": "curved",
    "curved": {"method": "anisotropic", "aspect_ratio_range": [1.0, 2.5]},
}


@pytest.mark.parametrize("boundaries_override", [
    ADDITIVE_WEIGHTS_BOUNDARIES_OVERRIDE,
    ANISOTROPIC_BOUNDARIES_OVERRIDE,
], ids=["additive_weights", "anisotropic"])
def test_e2e_curved_backend_smoke(tmp_path, boundaries_override):
    cfg = _config(tmp_path / "out", analysis={"gb_curvature": True},
                  boundaries=boundaries_override)
    res = run(cfg)
    assert res.gates.all_passed()
    g16 = next(r for r in res.gates.results() if r.gate == "G16")
    assert g16.measured <= CURV_G16_DROP_TOL
    with open(tmp_path / "out" / "gb_curvature.csv", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows, "expected gb_curvature.csv rows"
    hs = np.array([float(r["H_invA"]) for r in rows])
    assert np.isfinite(hs).all()


# ---------------------------------------------------------------------------
# Optional mesh cross-check (requires scikit-image) — Task 9
# ---------------------------------------------------------------------------


def _mesh_mean_abs_H(verts, faces):
    """Area-weighted <|H|> of a triangle mesh via the cotangent formula.

    H_i = |Δ x_i| / 2 with the discrete Laplace–Beltrami operator
    (uniform cotangent weights, barycentric vertex areas) — standard
    discrete-differential-geometry estimator, accurate enough for a
    15 % consistency check on a smooth closed-ish surface.
    """
    nv = len(verts)
    lap = np.zeros((nv, 3))
    w_sum = np.zeros(nv)
    area_v = np.zeros(nv)
    tri = verts[faces]                      # (F, 3, 3)
    for corner in range(3):
        a = tri[:, corner]
        b = tri[:, (corner + 1) % 3]
        c = tri[:, (corner + 2) % 3]
        # cotangent at vertex a, opposite edge (b, c)
        u, v = b - a, c - a
        cross = np.cross(u, v)
        cot = (np.einsum("ij,ij->i", u, v)
               / np.maximum(np.linalg.norm(cross, axis=1), 1e-30))
        ib, ic = faces[:, (corner + 1) % 3], faces[:, (corner + 2) % 3]
        for (p, q) in ((ib, ic), (ic, ib)):
            np.add.at(lap, p, 0.5 * cot[:, None] * (verts[q] - verts[p]))
            np.add.at(w_sum, p, 0.5 * cot)
        tri_area = 0.5 * np.linalg.norm(cross, axis=1)
        np.add.at(area_v, faces[:, corner], tri_area / 3.0)
    interior = (area_v > 1e-12) & (w_sum != 0.0)
    h = 0.5 * np.linalg.norm(lap[interior], axis=1) / area_v[interior]
    return float(np.sum(area_v[interior] * h)
                 / np.sum(area_v[interior]))


def test_mesh_cross_check_sphere():
    # importorskip INSIDE the test: a module-level skip would silence
    # the whole test_curvature.py module when scikit-image is absent.
    pytest.importorskip("skimage", reason="mesh extra not installed")
    from grainsmith.tessellation.voxel import build_voxel_grid

    R, L = 20.0, 60.0
    tess = _SphereTess(R, L)
    box = np.array([L, L, L])
    vg = build_voxel_grid(tess, box, grid_size=64)

    res = analyze_curvature(tess, box, [True, True, True], vg=vg)
    pc = res.pairs[(0, 1)]
    h_ls = float(np.sum(pc.areas * np.abs(pc.H)) / np.sum(pc.areas))

    verts, faces = vg.gb_mesh((0, 1))
    assert len(faces) > 0
    h_mesh = _mesh_mean_abs_H(verts, np.asarray(faces, dtype=np.int64))

    assert abs(h_ls - h_mesh) / h_mesh < 0.15          # spec tolerance
    assert abs(h_ls - 1.0 / R) / (1.0 / R) < 0.05      # analytic anchor
