"""Tests for triclinic (box.cells) single-crystal lattice-multiple
boxes: cell-matrix/tilt math, config validation (Rule 28), the general
(triclinic) PBC minimum-image metric, LAMMPS/extxyz triclinic writers, and
end-to-end builds from CIF crystals with independently re-verified atom
counts and minimum distances.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.crystal.cell import cell_matrix, reduce_triclinic_tilts
from grainsmith.errors import ConfigError
from grainsmith.tessellation.single import SingleCrystalTessellation


# ---------------------------------------------------------------------------
# Cell-matrix / tilt-factor math
# ---------------------------------------------------------------------------


def test_tilt_cubic_ti2ni_zero_tilts():
    """A cubic cell (a = b = c, all angles 90° — e.g. Ti2Ni SG 227,
    a ≈ 11.2437 Å per examples/assets/Ti2Ni.cif; the exact value is irrelevant
    to this tilt-math test) degenerates to a diagonal box matrix for ANY
    integer multiple — zero tilt factors exactly."""
    A = cell_matrix(11.32, 11.32, 11.32, 90.0, 90.0, 90.0)
    H = A @ np.diag([2.0, 3.0, 4.0])
    Hr = reduce_triclinic_tilts(H)
    np.testing.assert_allclose(Hr[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(Hr[0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(Hr[1, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(Hr), np.diag(H))


def test_tilt_monoclinic_tini_xz_negative():
    """TiNi-like monoclinic (beta=105.2296 deg): the raw H = A @ diag(n)
    has xz = c*cos(beta)*n3 < 0 (beta > 90 deg -> cos(beta) < 0); after
    reduction the sign is preserved (reduction only subtracts INTEGER
    multiples of a/b, it does not flip the sign of a tilt already within
    bounds for a large-enough n)."""
    beta = 105.2296
    A = cell_matrix(4.58, 2.85, 4.32, 90.0, beta, 90.0)
    assert A[0, 2] < 0.0  # c_x = c*cos(beta), cos(105.2296 deg) < 0
    n1, n2, n3 = 6, 5, 4
    H = A @ np.diag([float(n1), float(n2), float(n3)])
    Hr = reduce_triclinic_tilts(H)
    # xz sign must still be negative after reduction (unchanged bound check).
    assert Hr[0, 2] < 0.0
    # Volume/diagonal preserved exactly (integer basis change only).
    np.testing.assert_allclose(np.diag(Hr), np.diag(H))
    assert np.linalg.det(Hr) == pytest.approx(np.linalg.det(H))
    # Reduced tilt within the LAMMPS bound.
    assert abs(Hr[0, 2]) <= Hr[0, 0] / 2.0 + 1e-9
    assert abs(Hr[0, 1]) <= Hr[0, 0] / 2.0 + 1e-9
    assert abs(Hr[1, 2]) <= Hr[1, 1] / 2.0 + 1e-9


def test_tilt_hexagonal_tini3_xy_half_edge():
    """Hexagonal gamma=120 deg: b_x = b*cos(120) = -b/2 exactly, so for n1
    == n2 the raw xy tilt already sits at exactly -ax/2 (the LAMMPS
    boundary case) — reduction must leave it there (a no-op), not push it
    to +ax/2 (the equivalent value at the opposite bound)."""
    A = cell_matrix(5.1, 5.1, 8.3, 90.0, 90.0, 120.0)
    assert A[0, 1] == pytest.approx(-5.1 / 2.0)
    n1 = n2 = 3
    H = A @ np.diag([float(n1), float(n2), 2.0])
    Hr = reduce_triclinic_tilts(H)
    assert Hr[0, 1] == pytest.approx(-H[0, 0] / 2.0)
    assert abs(Hr[0, 1]) <= Hr[0, 0] / 2.0 + 1e-9


def test_reduce_triclinic_tilts_preserves_lattice():
    """The integer GL(3,Z) reduction must represent the SAME sublattice:
    every reduced box vector is an integer combination of the raw ones,
    and vice versa (both directions -> same lattice, not merely 'similar
    volume')."""
    A = cell_matrix(4.58, 2.85, 4.32, 90.0, 105.2296, 90.0)
    H = A @ np.diag([7.0, 6.0, 5.0])
    Hr = reduce_triclinic_tilts(H)
    M = np.linalg.inv(H) @ Hr  # Hr = H @ M
    Minv = np.linalg.inv(M)
    np.testing.assert_allclose(M, np.round(M), atol=1e-8)
    np.testing.assert_allclose(Minv, np.round(Minv), atol=1e-8)
    assert abs(round(np.linalg.det(M))) == 1  # unimodular: pure re-basis


# ---------------------------------------------------------------------------
# Triclinic minimum-image metric vs brute-force 27-image reference
# ---------------------------------------------------------------------------


def _brute_force_pairs(pos, cutoff, H):
    """O(N^2 * 27) reference: literal 27-image enumeration, no KDTree."""
    n = len(pos)
    Hinv = np.linalg.inv(H)
    frac = (Hinv @ pos.T).T
    shifts = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
              for k in (-1, 0, 1)]
    pairs = set()
    for a in range(n):
        for b in range(a + 1, n):
            dmin = np.inf
            for s in shifts:
                df = frac[a] - frac[b] + np.array(s, dtype=np.float64)
                dc = H @ df
                dmin = min(dmin, float(np.linalg.norm(dc)))
            if dmin < cutoff:
                pairs.add((a, b))
    return pairs


def test_triclinic_min_image_matches_brute_force():
    from grainsmith.atoms.overlap import (
        _pbc_pairs_general,
        _wrap_positions_general,
    )

    A = cell_matrix(5.1, 5.1, 8.3, 90.0, 90.0, 120.0)
    H = A @ np.diag([3.0, 3.0, 2.0])
    Hinv = np.linalg.inv(H)
    rng = np.random.Generator(np.random.PCG64(11))
    frac = rng.uniform(0.0, 1.0, size=(40, 3))
    pos = (H @ frac.T).T
    cutoff = 2.0

    wrapped = _wrap_positions_general(pos, [True, True, True], H, Hinv)
    got = set(_pbc_pairs_general(wrapped, cutoff, [True, True, True], H, Hinv))
    ref = _brute_force_pairs(pos, cutoff, H)
    assert got == ref
    assert len(ref) > 0  # sanity: cutoff picks up SOME pairs in this sample


def test_triclinic_min_image_reduces_to_orthogonal():
    """For a diagonal H the general path must reproduce the same pairs as
    the existing orthogonal KDTree(boxsize=L) path exactly."""
    from grainsmith.atoms.overlap import (
        _pbc_pairs,
        _pbc_pairs_general,
        _wrap_positions,
        _wrap_positions_general,
    )

    L = np.array([20.0, 20.0, 20.0])
    H = np.diag(L)
    rng = np.random.Generator(np.random.PCG64(3))
    pos = rng.uniform(0.0, 20.0, size=(60, 3))
    cutoff = 3.0
    periodic = [True, True, True]

    w1 = _wrap_positions(pos, periodic, L)
    ref = set(_pbc_pairs(w1, cutoff, periodic, L))

    w2 = _wrap_positions_general(pos, periodic, H, np.linalg.inv(H))
    got = set(_pbc_pairs_general(w2, cutoff, periodic, H, np.linalg.inv(H)))
    assert got == ref


# ---------------------------------------------------------------------------
# Bounding radius (covering sphere) correctness
# ---------------------------------------------------------------------------


def test_bounding_radius_covers_all_corners_tilted():
    """The covering radius must be the MAX over all 8 (+-a+-b+-c)/2 sign
    combinations, not just the (+,+,+) corner — for a tilted cell these
    differ."""
    A = cell_matrix(5.1, 5.1, 8.3, 90.0, 90.0, 120.0)
    H = A @ np.diag([2.0, 2.0, 2.0])
    tess = SingleCrystalTessellation(np.diag(H), [True, True, True],
                                     cell_matrix=H)
    r = tess.bounding_radius(0)
    signs = np.array([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0)
                      for k in (-1.0, 1.0)])
    corners = 0.5 * (signs @ H.T)
    expected = float(np.max(np.linalg.norm(corners, axis=1)))
    assert r == pytest.approx(expected)
    # And it must NOT equal the naive (a+b+c)/2 norm for this tilted cell.
    naive = 0.5 * float(np.linalg.norm(H @ np.ones(3)))
    assert r > naive + 1e-6


def test_bounding_radius_reduces_to_orthogonal_diagonal():
    L = np.array([10.0, 20.0, 30.0])
    tess = SingleCrystalTessellation(L, [True, True, True])
    assert tess.bounding_radius(0) == pytest.approx(
        0.5 * float(np.linalg.norm(L)))


# ---------------------------------------------------------------------------
# Config validation (Rule 28)
# ---------------------------------------------------------------------------


def _raw(**over):
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"cells": [2, 2, 2], "periodic": [True, True, True]},
        "grains": {"number": 1},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "fixed",
                        "fixed": {"euler_bunge_deg": [0.0, 0.0, 0.0]}},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    return raw


def test_rule28_both_lengths_and_cells_rejected():
    raw = _raw(box={"lengths": [10.0, 10.0, 10.0]})
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve_config(raw)


def test_rule28_neither_lengths_nor_cells_rejected():
    raw = _raw()
    del raw["box"]["cells"]
    with pytest.raises(ConfigError, match="Exactly one of"):
        resolve_config(raw)


def test_rule28_n_grains_gt_1_rejected():
    raw = _raw(grains={"number": 4})
    with pytest.raises(ConfigError, match="grains.number: 1"):
        resolve_config(raw)


def test_rule28_non_identity_orientation_rejected():
    raw = _raw(orientation={"scheme": "random_uniform"})
    with pytest.raises(ConfigError, match="IDENTITY"):
        resolve_config(raw)


def test_rule28_rotated_fixed_orientation_rejected():
    raw = _raw(orientation={"scheme": "fixed",
                           "fixed": {"axis_angle": {"axis": [0, 0, 1],
                                                    "angle_deg": 15.0}}})
    with pytest.raises(ConfigError, match="IDENTITY"):
        resolve_config(raw)


def test_rule28_vacuum_rejected():
    raw = _raw(box={"vacuum": 5.0, "periodic": [True, True, False]})
    with pytest.raises(ConfigError, match="box.periodic"):
        resolve_config(raw)


def test_rule28_free_axis_rejected():
    raw = _raw(box={"periodic": [True, True, False]})
    with pytest.raises(ConfigError, match="box.periodic"):
        resolve_config(raw)


def test_rule28_phases_rejected():
    raw = _raw()
    raw["phases"] = [{"name": "a", "fraction": 1.0,
                      "crystal": raw.pop("crystal")}]
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_rule28_valid_resolves_h_matrix():
    raw = _raw()
    cfg = resolve_config(raw)
    assert cfg.box.resolved_h is not None
    H = np.asarray(cfg.box.resolved_h)
    A = cell_matrix(3.615, 3.615, 3.615, 90.0, 90.0, 90.0)
    expected = A @ np.diag([2.0, 2.0, 2.0])
    np.testing.assert_allclose(H, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end builds from CIF crystals: exact atom count + independent
# min-distance re-verification (ASE mic) + G14 == 0
# ---------------------------------------------------------------------------


def _run_cif_triclinic(tmp_path, examples_dir, cif_name, cells, extra=None):
    from grainsmith.pipeline import run

    raw = {
        "seed": {"mode": "fixed", "value": 271828},
        "box": {"cells": list(cells), "periodic": [True, True, True]},
        "grains": {"number": 1},
        "crystal": {"cif": {"file": str(examples_dir / "assets" / cif_name)}},
        "orientation": {"scheme": "fixed",
                        "fixed": {"euler_bunge_deg": [0.0, 0.0, 0.0]}},
        "boundaries": {"overlap_removal": {"enabled": True,
                                           "cutoff": "0.85*d_nn"}},
        "output": {"directory": str(tmp_path / "out")},
    }
    if extra:
        raw.update(extra)
    return run(resolve_config(raw))


def _independent_min_distance(res) -> float:
    """Re-derive the minimum PBC distance via ASE (mic=True), completely
    independent of grainsmith's own overlap/gate code paths."""
    import ase

    H = np.asarray(res.config.box.resolved_h, dtype=np.float64)
    at = ase.Atoms(symbols=list(res.atoms.species),
                   positions=res.atoms.pos, cell=H.T, pbc=True)
    d = at.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


@pytest.mark.parametrize("cif_name,z,cells", [
    ("TiNi.cif", 4, (3, 3, 3)),
    ("TiNi3.cif", 16, (2, 2, 2)),
    ("Ti2Ni.cif", 96, (1, 1, 1)),
])
def test_e2e_triclinic_exact_atom_count_and_g14(tmp_path, examples_dir,
                                                cif_name, z, cells):
    res = _run_cif_triclinic(tmp_path, examples_dir, cif_name, cells)
    n1, n2, n3 = cells
    assert len(res.atoms) == n1 * n2 * n3 * z
    assert res.n_deleted == 0
    assert res.gates.all_passed()
    g14 = [r for r in res.gates.results() if r.gate == "G14"][0]
    assert g14.measured < 1e-8
    assert "WARN" not in g14.message


@pytest.mark.parametrize("cif_name,cells", [
    ("TiNi.cif", (3, 3, 3)),
    ("TiNi3.cif", (2, 2, 2)),
])
def test_e2e_triclinic_independent_min_distance(tmp_path, examples_dir,
                                                cif_name, cells):
    """Independent re-verification (ASE mic distances) of the hard overlap
    gate: minimum PBC distance must equal the ideal-crystal d_nn (cutoff
    was 0.85*d_nn, so no atoms should have been close enough to delete)."""
    res = _run_cif_triclinic(tmp_path, examples_dir, cif_name, cells)
    dmin = _independent_min_distance(res)
    assert dmin == pytest.approx(res.d_nn, rel=1e-6)
    assert dmin >= 0.85 * res.d_nn - 1e-9


def test_e2e_triclinic_lammps_roundtrip(tmp_path, examples_dir):
    """write_lammps -> parse_lammps: tilt factors and atom positions
    survive the round-trip, and every atom lies within the tilted box."""
    from grainsmith.io.lammps import parse_lammps

    res = _run_cif_triclinic(tmp_path, examples_dir, "TiNi.cif", (3, 3, 3))
    data_files = list(res.outdir.glob("*.data"))
    assert len(data_files) == 1
    parsed = parse_lammps(data_files[0])
    H = np.asarray(res.config.box.resolved_h, dtype=np.float64)
    assert parsed["tilt"] is not None
    np.testing.assert_allclose(
        parsed["tilt"], [H[0, 1], H[0, 2], H[1, 2]], atol=1e-9)
    np.testing.assert_allclose(
        parsed["bounds"], [[0.0, H[0, 0]], [0.0, H[1, 1]], [0.0, H[2, 2]]],
        atol=1e-9)
    assert parsed["natoms"] == len(res.atoms)
    # Containment: every parsed atom's fractional coordinate in [0, 1).
    Hinv = np.linalg.inv(H)
    frac = (Hinv @ parsed["pos"].T).T
    assert np.all(frac >= -1e-9) and np.all(frac < 1.0 + 1e-9)


def test_e2e_triclinic_extxyz_full_cell(tmp_path, examples_dir):
    """extxyz Lattice carries the FULL 3x3 (transposed, row-major) box
    matrix, not just the diagonal."""
    res = _run_cif_triclinic(tmp_path, examples_dir, "TiNi3.cif", (2, 2, 2))
    xyz_files = list(res.outdir.glob("*.extxyz"))
    assert len(xyz_files) == 1
    comment = xyz_files[0].read_text(encoding="utf-8").splitlines()[1]
    assert "Lattice=" in comment
    lat_str = comment.split('Lattice="')[1].split('"')[0]
    vals = [float(v) for v in lat_str.split()]
    assert len(vals) == 9
    Ht = np.asarray(res.config.box.resolved_h, dtype=np.float64).T
    np.testing.assert_allclose(np.array(vals).reshape(3, 3), Ht, atol=1e-9)


def test_e2e_triclinic_deterministic(tmp_path, examples_dir):
    res1 = _run_cif_triclinic(tmp_path / "a", examples_dir, "TiNi.cif",
                              (2, 2, 2))
    res2 = _run_cif_triclinic(tmp_path / "b", examples_dir, "TiNi.cif",
                              (2, 2, 2))
    np.testing.assert_array_equal(res1.atoms.pos, res2.atoms.pos)
    np.testing.assert_array_equal(res1.atoms.species, res2.atoms.species)
