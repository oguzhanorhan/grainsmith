"""Tests for analysis/ (P6, §6.10) and the P6-scope gates (§9).

Controlled fixtures with analytically known answers:
- a slab bicrystal with a SINGLE flat boundary plane (z = 15, normal ẑ) so
  area, mean normal, spread, plane indices and the tilt/twist character are
  exact;
- Σ5 36.87°⟨100⟩ and Σ3 60°⟨111⟩ misorientations for CSL-Brandon matching.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.analysis import analyze_boundaries, analyze_grains
from grainsmith.atoms.fill import AtomBlock
from grainsmith.atoms.overlap import OverlapLedger, resolve_cutoff
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.errors import ConfigError
from grainsmith.orientation.quaternion import axis_angle_to_quat
from grainsmith.qa import (
    gate_g3_voxel,
    gate_g7_min_distance,
    gate_g8_atom_count,
    gate_g9_composition,
    nominal_composition,
)
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation

Q_ID = np.array([1.0, 0.0, 0.0, 0.0])
SIGMA5_DEG = np.degrees(np.arccos(4.0 / 5.0))  # 36.8699°


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


@pytest.fixture(scope="module")
def cubic():
    """Cubic cell matrix + the 24 proper rotations of m-3m."""
    A = cell_matrix(3.615, 3.615, 3.615, 90.0, 90.0, 90.0)
    rots, _ = symmetry_ops(hall_from_international(221))
    sym = proper_rotation_quaternions(A, rots)
    assert len(sym) == 24
    return A, sym


def _slab_bicrystal():
    """Two grains stacked along the free z axis: exactly ONE boundary
    plane at z = 15 with outward (0→1) normal +ẑ and area 400 Å²."""
    L = np.array([20.0, 20.0, 30.0])
    periodic = [True, True, False]
    seeds = np.array([[10.0, 10.0, 7.5], [10.0, 10.0, 22.5]])
    return FlatTessellation(seeds, L, periodic), L, periodic


# ------------------------------------------------------------- boundaries

def test_sigma5_tilt_boundary(cubic):
    """Σ5 rotation about [100] across a z-normal plane: a pure TILT
    boundary with exact area/normal/plane indices and CSL Σ5."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, axis_angle_to_quat([1.0, 0.0, 0.0], SIGMA5_DEG)])
    reps = analyze_boundaries(tess, quats, sym, A, L, per, csl=True)
    assert len(reps) == 1
    r = reps[0]

    assert (r.grain_i, r.grain_j) == (0, 1)
    assert abs(r.seed_distance_A - 15.0) < 1e-12
    assert abs(r.misorientation_deg - SIGMA5_DEG) < 0.05
    assert sorted(abs(v) for v in (r.axis_u, r.axis_v, r.axis_w)) == [0, 0, 1]
    assert r.axis_dev_deg < 0.01

    assert abs(r.area_A2 - 400.0) < 1e-9
    assert abs(r.mean_normal_z - 1.0) < 1e-12
    assert r.normal_spread_deg < 1e-9

    # Rotation axis x̂ lies IN the z-normal boundary plane → ψ = 90° → tilt
    assert r.character == "tilt"
    assert abs(r.character_angle_deg - 90.0) < 0.5

    # Boundary plane in both crystal frames: (001) for the unrotated grain,
    # R_jᵀ ẑ = (0, sin θ, cos θ) ∝ (0, 3, 4) for the Σ5-rotated grain.
    assert r.plane_i_hkl == "(0 0 1)" and r.plane_i_dev_deg < 0.01
    assert r.plane_j_hkl == "(0 3 4)" and r.plane_j_dev_deg < 0.01

    assert r.csl_sigma == "5"


def test_twist_boundary(cubic):
    """Rotation about the boundary normal (ẑ) → ψ ≈ 0 → twist; 7°⟨001⟩
    falls outside every Brandon window of the table (nearest: Σ25a at
    16.26°⟨100⟩ with a 3° window).  Note 15°⟨001⟩ would legitimately match
    Σ25a (deviation 1.26° < 3°)."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, axis_angle_to_quat([0.0, 0.0, 1.0], 7.0)])
    reps = analyze_boundaries(tess, quats, sym, A, L, per, csl=True)
    r = reps[0]
    assert abs(r.misorientation_deg - 7.0) < 0.05
    assert r.character == "twist"
    assert r.character_angle_deg < 0.5
    assert r.csl_sigma == ""


@pytest.mark.parametrize("other", [
    Q_ID, -Q_ID, axis_angle_to_quat([0.0, 0.0, 1.0], 90.0),
])
def test_zero_disorientation_has_no_axis_or_boundary_character(cubic, other):
    matrix, symmetry = cubic
    tess, lengths, periodic = _slab_bicrystal()
    report = analyze_boundaries(
        tess, np.array([Q_ID, other]), symmetry, matrix,
        lengths, periodic, csl=True,
    )[0]
    assert report.misorientation_deg == pytest.approx(0.0, abs=1e-10)
    assert (report.axis_u, report.axis_v, report.axis_w) == (0, 0, 0)
    assert report.axis_crystal is None
    assert np.isnan(report.axis_dev_deg)
    assert report.character == "undefined"
    assert np.isnan(report.character_angle_deg)
    assert report.plane_i_hkl == "(0 0 1)"
    assert report.plane_j_hkl == "(0 0 1)"
    assert report.area_A2 == pytest.approx(400.0)
    assert report.csl_sigma == ""


def test_csl_brandon_window_edge(cubic):
    """15°⟨001⟩ lies INSIDE the Σ25a Brandon window (|15−16.26| = 1.26° <
    15/√25 = 3°) and must be labelled — pins the class-deviation algebra
    (S_a inside the product) that a naive relative-rotation reduction
    misses."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, axis_angle_to_quat([0.0, 0.0, 1.0], 15.0)])
    reps = analyze_boundaries(tess, quats, sym, A, L, per, csl=True)
    assert reps[0].csl_sigma == "25a"


def test_csl_sigma3_mixed(cubic):
    """Σ3 60°⟨111⟩ across a z-normal plane: ψ = arccos(1/√3) ≈ 54.7° →
    mixed; CSL Σ3."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, axis_angle_to_quat([1.0, 1.0, 1.0], 60.0)])
    reps = analyze_boundaries(tess, quats, sym, A, L, per, csl=True)
    r = reps[0]
    assert abs(r.misorientation_deg - 60.0) < 0.05
    assert r.csl_sigma == "3"
    assert r.character == "mixed"
    assert abs(r.character_angle_deg - 54.7356) < 0.5


def test_undefined_mean_normal_antiparallel_sheets(cubic):
    """A fully periodic bicrystal meets through two antiparallel sheets:
    the resultant vanishes → no mean plane, character 'undefined', but the
    total area is still reported."""
    A, sym = cubic
    L = np.array([20.0, 20.0, 20.0])
    per = [True, True, True]
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, per)
    quats = np.array([Q_ID, axis_angle_to_quat([1.0, 0.0, 0.0], SIGMA5_DEG)])
    reps = analyze_boundaries(tess, quats, sym, A, L, per)
    r = reps[0]
    assert abs(r.area_A2 - 800.0) < 1e-9
    assert r.character == "undefined"
    assert np.isnan(r.character_angle_deg)
    assert r.plane_i_hkl == "(0 0 0)"


def test_csl_requires_cubic(cubic):
    """csl=True with a non-cubic point group (≠ 24 proper rotations) is a
    v1 ConfigError (§6.10)."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, Q_ID])
    with pytest.raises(ConfigError):
        analyze_boundaries(tess, quats, sym[:12], A, L, per, csl=True)


# Canonical cubic CSL representatives (Randle & Engler 2000), cross-checked
# against Aimsgb (Cheng et al. 2018) Table A.2.  Each CSL_TABLE entry must be a
# valid representative of THIS misorientation class (class-deviation ~ 0).
# Pins the Σ19a axis fix: 26.53° is ⟨110⟩ (not ⟨100⟩ — a 20.15° mismatch).
_CSL_REFERENCE = {
    "3": (60.00, [1, 1, 1]), "5": (36.87, [1, 0, 0]), "7": (38.21, [1, 1, 1]),
    "9": (38.94, [1, 1, 0]), "11": (50.48, [1, 1, 0]), "13a": (22.62, [1, 0, 0]),
    "13b": (27.80, [1, 1, 1]), "15": (48.19, [2, 1, 0]), "17a": (28.07, [1, 0, 0]),
    "17b": (61.93, [2, 2, 1]), "19a": (26.53, [1, 1, 0]), "19b": (46.83, [1, 1, 1]),
    "21a": (21.79, [1, 1, 1]), "21b": (44.41, [2, 1, 1]), "23": (40.45, [3, 1, 1]),
    "25a": (16.26, [1, 0, 0]), "25b": (51.68, [3, 3, 1]), "27a": (31.59, [1, 1, 0]),
    "27b": (35.43, [2, 1, 0]), "29a": (43.60, [1, 0, 0]),
}


def test_csl_table_entries_are_valid_class_representatives(cubic):
    """Every CSL_TABLE (Σ, θ, axis) entry must be a genuine representative of
    its misorientation class — class-deviation to the standard reference ~ 0.

    This pins the C6 review fix (Σ19a = 26.53°⟨110⟩, not ⟨100⟩): a wrong axis
    is a ~20° class mismatch the Brandon matcher cannot recover from."""
    from grainsmith.analysis.boundaries import _misorientation_class_deviation
    from grainsmith.constants import CSL_TABLE

    _, sym = cubic
    assert set(CSL_TABLE) == set(_CSL_REFERENCE), "CSL_TABLE / reference drift"
    for label, (theta, axis) in CSL_TABLE.items():
        rtheta, raxis = _CSL_REFERENCE[label]
        q_stored = axis_angle_to_quat(np.asarray(axis, float), theta)
        q_ref = axis_angle_to_quat(np.asarray(raxis, float), rtheta)
        dev = _misorientation_class_deviation(q_stored, q_ref, sym)
        assert dev < 0.05, f"Σ{label} not a valid class representative (dev {dev:.3f}°)"


def test_sigma19_axis_is_110(cubic):
    """Σ19a is 26.53°⟨110⟩: a real Σ19 boundary is labelled, and the (wrong)
    26.53°⟨100⟩ misorientation is NOT mistaken for Σ19 (regression for C6)."""
    A, sym = cubic
    tess, L, per = _slab_bicrystal()

    q19 = axis_angle_to_quat([1.0, 1.0, 0.0], 26.53)   # true Σ19a
    reps = analyze_boundaries(tess, np.array([Q_ID, q19]), sym, A, L, per, csl=True)
    assert reps[0].csl_sigma == "19a"

    q_wrong = axis_angle_to_quat([1.0, 0.0, 0.0], 26.53)  # 26.53°⟨100⟩ is not Σ19
    reps = analyze_boundaries(tess, np.array([Q_ID, q_wrong]), sym, A, L, per, csl=True)
    assert reps[0].csl_sigma != "19a"


def test_boundaries_ledger_column(cubic):
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, axis_angle_to_quat([0, 0, 1], 10.0)])
    ledger = OverlapLedger()
    for _ in range(3):
        ledger.record("Cu", 0, 1)
    reps = analyze_boundaries(tess, quats, sym, A, L, per, ledger=ledger)
    assert reps[0].n_overlap_deleted == 3


def _assert_boundary_reports_equal(r1, r2):
    """Field-by-field equality of two BoundaryReport rows, NaN-aware
    (misorientation/character/plane fields are NaN for undefined/
    interphase pairs, where plain == would spuriously fail)."""
    import dataclasses
    import math
    for f in dataclasses.fields(r1):
        v1 = getattr(r1, f.name)
        v2 = getattr(r2, f.name)
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            assert (v1 is None) == (v2 is None), f.name
            if v1 is not None:
                np.testing.assert_array_equal(v1, v2, err_msg=f.name)
        elif isinstance(v1, float) and math.isnan(v1):
            assert isinstance(v2, float) and math.isnan(v2), f.name
        else:
            assert v1 == v2, f"{f.name}: {v1!r} != {v2!r}"


def test_boundaries_jobs_parallel_matches_serial(cubic):
    """jobs=2 must reproduce jobs=1 bit-for-bit on the voxel backend: the
    per-pair kernel (_pair_frames + tessellation.voxel._gb_pair_normals +
    _pair_report_fields) is a pure function of its inputs and results are
    reassembled in tess.adjacency() order (analyze_boundaries' jobs
    paragraph), so every BoundaryReport field — geometry, misorientation,
    boundary-plane indices, tilt/twist character, CSL Σ — is identical
    regardless of jobs (mirrors analyze_curvature's own jobs=2-vs-1 test)."""
    from grainsmith.orientation.samplers import random_uniform
    from grainsmith.tessellation.weighted import WeightedTessellation

    A, sym = cubic
    L = np.array([40.0, 40.0, 40.0])
    per = [True, True, True]
    seeds = seed_grains(6, L, per, _rng(11))
    tess = WeightedTessellation(seeds, L, per, sigma_w=0.8, rng=_rng(12))
    quats = random_uniform(6, _rng(13))
    assert len(tess.adjacency()) >= 2

    reps1 = analyze_boundaries(tess, quats, sym, A, L, per, csl=True, jobs=1)
    reps2 = analyze_boundaries(tess, quats, sym, A, L, per, csl=True, jobs=2)

    assert [(r.grain_i, r.grain_j) for r in reps1] == \
           [(r.grain_i, r.grain_j) for r in reps2] == tess.adjacency()
    for r1, r2 in zip(reps1, reps2, strict=True):
        _assert_boundary_reports_equal(r1, r2)


def test_boundaries_curved_voxel_path(cubic):
    """Curved backend (additive weights → voxel geometry): every adjacent
    pair gets an area > 0, a unit-or-degenerate mean normal and a ψ in
    [0°, 90°]."""
    from grainsmith.orientation.samplers import random_uniform
    from grainsmith.tessellation.weighted import WeightedTessellation

    A, sym = cubic
    L = np.array([40.0, 40.0, 40.0])
    per = [True, True, True]
    seeds = seed_grains(4, L, per, _rng(8))
    tess = WeightedTessellation(seeds, L, per, sigma_w=0.8, rng=_rng(9))
    quats = random_uniform(4, _rng(10))
    reps = analyze_boundaries(tess, quats, sym, A, L, per)
    assert {(r.grain_i, r.grain_j) for r in reps} == set(tess.adjacency())
    for r in reps:
        assert r.area_A2 > 0.0
        n = np.array([r.mean_normal_x, r.mean_normal_y, r.mean_normal_z])
        if r.character == "undefined":
            continue
        assert abs(np.linalg.norm(n) - 1.0) < 1e-9
        assert 0.0 <= r.character_angle_deg <= 90.0
        assert 0.0 <= r.normal_spread_deg <= 90.0


# ------------------------------------------------------------------ grains

def test_analyze_grains_slab(cubic):
    A, sym = cubic
    tess, L, per = _slab_bicrystal()
    quats = np.array([Q_ID, Q_ID])
    atoms = AtomBlock(
        pos=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 20.0], [2.0, 2.0, 25.0]]),
        species=np.array(["Cu", "Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 1, 1], dtype=np.int32),
    )
    reps = analyze_grains(tess, quats, A, L, atoms=atoms)
    assert len(reps) == 2
    for r, vol, n_at in zip(reps, (6000.0, 6000.0), (1, 2), strict=True):
        assert abs(r.volume_A3 - vol) < 1e-6
        assert abs(r.volume_fraction - 0.5) < 1e-12
        assert r.n_neighbors == 1
        assert r.n_atoms == n_at
        # Identity orientation descriptors
        assert r.z_plane_hkl == "(0 0 1)" and r.z_plane_dev_deg < 1e-9
        assert r.x_dir_uvw == "[1 0 0]" and r.x_dir_dev_deg < 1e-9
        assert abs(r.angle_deg) < 1e-9


# ------------------------------------------------------------------- gates

def test_gate_g8_thresholds():
    ok = gate_g8_atom_count(n_final=1000, n_deleted=0, rho_atom=1.0,
                            box_volume=1000.0)
    assert ok.passed and "ok" in ok.message

    warn = gate_g8_atom_count(n_final=970, n_deleted=0, rho_atom=1.0,
                              box_volume=1000.0)
    assert warn.passed and "WARN" in warn.message

    fail = gate_g8_atom_count(n_final=900, n_deleted=0, rho_atom=1.0,
                              box_volume=1000.0)
    assert not fail.passed


def test_gate_g9_warn_only():
    res = gate_g9_composition({"Ni": 480, "Al": 520},
                              {"Ni": 0.5, "Al": 0.5})
    assert res.passed  # never hard-fails
    assert abs(res.measured - 0.02) < 1e-12
    assert "WARN" in res.message

    res_ok = gate_g9_composition({"Ni": 500, "Al": 500},
                                 {"Ni": 0.5, "Al": 0.5})
    assert "WARN" not in res_ok.message


def test_gate_g7_fresh_query():
    L = np.array([20.0, 20.0, 20.0])
    close = AtomBlock(
        pos=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]]),
        species=np.array(["Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 1], dtype=np.int32),
    )
    assert not gate_g7_min_distance(close, 2.0, [True] * 3, L).passed

    far = AtomBlock(
        pos=np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]]),
        species=np.array(["Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 1], dtype=np.int32),
    )
    assert gate_g7_min_distance(far, 2.0, [True] * 3, L).passed


def test_gate_g3_voxel_exact():
    L = np.array([20.0, 20.0, 20.0])
    vols = np.array([4000.0, 4000.0])
    res = gate_g3_voxel(vols, L, np.array([1.0, 1.0, 1.0]))
    assert res.passed and res.measured < 1e-12


def test_nominal_composition():
    assert nominal_composition(["Ni", "Al"], [None, None]) == {
        "Ni": 0.5, "Al": 0.5}
    frac = nominal_composition(["Ti", "Al"],
                               [{"Ti": 0.9, "Al": 0.1}, None])
    assert abs(frac["Ti"] - 0.45) < 1e-12
    assert abs(frac["Al"] - 0.55) < 1e-12


def test_gate_g16_warn_only():
    from grainsmith.qa import gate_g16_curvature

    res = gate_g16_curvature(n_dropped=10, n_raw=100)   # 10 % > 5 % tol
    assert res.gate == "G16"
    assert res.passed                                   # never hard-fails
    assert abs(res.measured - 0.10) < 1e-12
    assert "WARN" in res.message

    res_ok = gate_g16_curvature(n_dropped=1, n_raw=100)
    assert res_ok.passed and "WARN" not in res_ok.message

    res_empty = gate_g16_curvature(n_dropped=0, n_raw=0)  # flat / no GB
    assert res_empty.passed and res_empty.measured == 0.0
    assert "WARN" not in res_empty.message


def test_gate_g21_warn_only():
    """G21 (per-grain Gauss-Bonnet face-interior residual): synthetic
    per-grain totals drive PASS ('ok') below 4π and WARN above it —
    always warn-only (passed=True unconditionally, per the spec: this
    diagnoses the gb_curvature ESTIMATOR, never the generated model)."""
    from grainsmith.qa import gate_g21_gauss_bonnet

    tiny = np.array([0.01, -0.02, 0.0, 0.5])   # all |.| << 4pi
    res_ok = gate_g21_gauss_bonnet(tiny)
    assert res_ok.gate == "G21"
    assert res_ok.passed                                # never hard-fails
    assert abs(res_ok.measured - 0.5) < 1e-12           # worst = max |.|
    assert "WARN" not in res_ok.message
    assert "0 grain(s)" in res_ok.message

    # One grain's face-interior integral exceeds the 4π = 12.566... bound
    # (the reported bug: measured up to ~54.1, several times 4π).
    over = np.array([0.01, 54.1, 0.0])
    res_warn = gate_g21_gauss_bonnet(over)
    assert res_warn.passed                              # STILL never fails
    assert abs(res_warn.measured - 54.1) < 1e-12
    assert "WARN" in res_warn.message
    assert "1 grain(s)" in res_warn.message
    assert "grain 1" in res_warn.message                # worst-grain id
    assert "54.1" in res_warn.message
    assert "4π=12.5664" in res_warn.message or "4\u03c0=12.5664" in res_warn.message

    # Exactly at the 4π boundary is NOT a violation (strict >).
    from grainsmith.constants import CURV_G21_GAUSS_BONNET_TOL
    at_bound = np.array([CURV_G21_GAUSS_BONNET_TOL])
    res_bound = gate_g21_gauss_bonnet(at_bound)
    assert "WARN" not in res_bound.message

    # Empty (no grains with retained samples, e.g. single-crystal runs)
    # never crashes and reports ok/0.0.
    res_empty = gate_g21_gauss_bonnet(np.array([]))
    assert res_empty.passed and res_empty.measured == 0.0
    assert "WARN" not in res_empty.message


# ----------------------------------------------------------- cutoff parser

def test_resolve_cutoff_forms():
    assert resolve_cutoff(1.5) == 1.5
    assert resolve_cutoff("1.5") == 1.5
    assert abs(resolve_cutoff("0.85*d_nn", d_nn=2.0) - 1.7) < 1e-12
    assert abs(resolve_cutoff(" 0.85 * d_nn ", d_nn=2.0) - 1.7) < 1e-12


def test_resolve_cutoff_rejects():
    with pytest.raises(ConfigError):
        resolve_cutoff("0.85*dnn")
    with pytest.raises(ConfigError):
        resolve_cutoff("0.85*d_nn", d_nn=np.inf)  # the 1-atom-basis trap
    with pytest.raises(ConfigError):
        resolve_cutoff(-1.0)
    with pytest.raises(ConfigError):
        resolve_cutoff("abc")


def test_explicit_voxel_grid_shared_by_boundaries_and_grains():
    """A curved backend with an explicit analysis.voxel_grid must measure grain
    volumes AND boundary geometry on the SAME grid.  Before the fix, grains.csv/
    statistics.csv used the explicit grid while boundaries.csv silently fell back
    to the tessellation's 'auto' grid via get_voxel_grid — desyncing the CSVs."""
    from types import SimpleNamespace

    from grainsmith.analysis.grains import get_voxel_grid
    from grainsmith.pipeline import _analysis_voxel_grid
    from grainsmith.seeding import seed_grains
    from grainsmith.tessellation.weighted import WeightedTessellation

    L = np.array([40.0, 40.0, 40.0])
    periodic = [True, True, True]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(3)))
    seeds = seed_grains(6, L, periodic, rng)
    tess = WeightedTessellation(seeds, L, periodic, sigma_w=1.0,
                                rng=np.random.default_rng(7))

    auto_shape = tuple(tess.voxel_grid().shape)   # primes tess._voxel at 'auto'
    explicit = min(auto_shape) + 12               # a resolution distinct from auto
    config = SimpleNamespace(
        analysis=SimpleNamespace(voxel_grid=explicit),
        box=SimpleNamespace(lengths=L),
    )

    grains_grid = _analysis_voxel_grid(config, tess)     # grains/statistics use this
    boundaries_grid = get_voxel_grid(tess, L)            # analyze_boundaries uses this
    assert tuple(boundaries_grid.shape) == tuple(grains_grid.shape)
    assert tuple(grains_grid.shape) != auto_shape        # the explicit grid took effect
