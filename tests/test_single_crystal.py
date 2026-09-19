"""Tests for the single-crystal backend + gate G14."""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError
from grainsmith.tessellation.single import (
    SingleCrystalTessellation,
    commensurability_misfit,
)

L = np.array([36.15, 36.15, 36.15])
PER = [True, True, True]
A_CU = np.diag([3.615, 3.615, 3.615])


# ---------------------------------------------------------------------------
# Backend unit tests
# ---------------------------------------------------------------------------


def test_trivial_contract():
    tess = SingleCrystalTessellation(L, PER)
    assert tess.n_grains == 1
    np.testing.assert_allclose(tess.seeds, [L / 2.0])
    assert tess.adjacency() == []
    assert tess.bounding_radius(0) == pytest.approx(
        0.5 * float(np.linalg.norm(L)))
    assert tess.total_volume() == pytest.approx(float(np.prod(L)))
    X = np.array([[1.0, 2.0, 3.0], [30.0, 30.0, 30.0]])
    np.testing.assert_array_equal(tess.grain_of(X), [0, 0])


def test_owns_tiling_invariant():
    """Every torus point has exactly one owned lift among the 27 images —
    the §6.8 fill contract."""
    rng = np.random.Generator(np.random.PCG64(5))
    pts = rng.uniform(0.0, 1.0, size=(200, 3)) * L
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=np.float64) * L
    tess = SingleCrystalTessellation(L, PER)
    counts = np.zeros(len(pts), dtype=int)
    for s in shifts:
        counts += tess.owns(pts + s, 0).astype(int)
    np.testing.assert_array_equal(counts, np.ones(len(pts), dtype=int))


def test_owns_window_edge_plane():
    """A commensurate atom plane sits EXACTLY on the 0 ≡ L window edge;
    the tol-shifted window must own exactly one of its two float lifts
    (the regression behind SINGLE_WINDOW_TOL — a raw [0, L) test drops
    the whole plane)."""
    tess = SingleCrystalTessellation(L, PER)
    # The two lifts as the fill computes them: T·a + L/2 for T = ∓5.
    lo = np.array([[-5 * 3.615 + 36.15 / 2.0, 1.0, 1.0]])
    hi = np.array([[+5 * 3.615 + 36.15 / 2.0, 1.0, 1.0]])
    owned = int(tess.owns(lo, 0)[0]) + int(tess.owns(hi, 0)[0])
    assert owned == 1


def test_owns_free_axis_high_wall_owned():
    """On a FREE (non-periodic) axis, x=0 and x=L are DISTINCT positions, so a
    commensurate atom plane lands on BOTH walls and each must be owned exactly
    once.  The periodic torus-dedup window [-tol, L-tol) is correct only for a
    periodic axis; applied to a free axis it silently drops the high (x=L) wall
    atom layer."""
    Lf = np.array([10.0, 10.0, 10.0])
    tess = SingleCrystalTessellation(Lf, [False, True, True])
    low = bool(tess.owns(np.array([[0.0, 5.0, 5.0]]), 0)[0])
    high = bool(tess.owns(np.array([[10.0, 5.0, 5.0]]), 0)[0])
    assert (low, high) == (True, True)


def test_margin_distance_to_faces():
    tess = SingleCrystalTessellation(L, PER)
    X = np.array([[1.0, 17.0, 30.0]])
    assert tess.margin(X, 0)[0] == pytest.approx(1.0)
    # Outside the window: negative.
    assert tess.margin(np.array([[-2.0, 17.0, 17.0]]), 0)[0] == pytest.approx(-2.0)
    # No periodic axis → no boundary: half box diagonal.
    free = SingleCrystalTessellation(L, [False, False, False])
    assert free.margin(X, 0)[0] == pytest.approx(
        0.5 * float(np.linalg.norm(L)))


# ---------------------------------------------------------------------------
# Commensurability misfit (gate G14 input)
# ---------------------------------------------------------------------------


def test_misfit_commensurate_identity():
    """Box = 10·a per axis, identity orientation → strain at float64
    roundoff scale, far below COMMENSURATE_TOL."""
    misfits = commensurability_misfit(A_CU, L, PER)
    assert sorted(misfits) == [0, 1, 2]
    assert max(s for _, s in misfits.values()) < 1e-12


def test_misfit_rotated_incommensurate():
    """A 15° rotation about z has no lattice period along x/y of this box
    (cos 15° irrational direction) — large misfit there, axis z intact."""
    th = np.radians(15.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    misfits = commensurability_misfit(R @ A_CU, L, PER)
    assert misfits[0][1] > 1e-3 and misfits[1][1] > 1e-3
    assert misfits[2][1] < 1e-12


def test_misfit_skips_free_axes():
    misfits = commensurability_misfit(A_CU, L, [True, False, False])
    assert sorted(misfits) == [0]


# ---------------------------------------------------------------------------
# Resolve rules (Rule 24)
# ---------------------------------------------------------------------------


def _raw(**over):
    raw = {
        "seed": {"mode": "fixed", "value": 7},
        "box": {"lengths": [36.15, 36.15, 36.15]},
        "grains": {"number": 1},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "fixed",
                        "fixed": {"euler_bunge_deg": [0.0, 0.0, 0.0]}},
    }
    raw.update(over)
    return raw


def test_rule24_curved_rejected():
    raw = _raw(boundaries={"geometry": "curved", "curved": {"method": "warp"}})
    with pytest.raises(ConfigError, match="single crystal"):
        resolve_config(raw)


def test_rule24_size_distribution_rejected():
    raw = _raw()
    raw["grains"]["size_distribution"] = {"type": "equal"}
    with pytest.raises(ConfigError, match="meaningless for a single"):
        resolve_config(raw)


def test_rule24_mdf_target_rejected():
    raw = _raw()
    raw["orientation"]["mdf_target"] = {"type": "mackenzie"}
    with pytest.raises(ConfigError, match=">= 2 grains"):
        resolve_config(raw)


# ---------------------------------------------------------------------------
# End-to-end (G14 + exact counts)
# ---------------------------------------------------------------------------


def _run(tmp_path, **over):
    from grainsmith.pipeline import run
    raw = _raw(output={"directory": str(tmp_path / "out")}, **over)
    return run(resolve_config(raw))


def test_e2e_commensurate_perfect_crystal(tmp_path):
    """10×10×10 FCC cells, identity orientation: EXACTLY 4·10³ atoms,
    zero deleted, G14 commensurate, all gates pass."""
    res = _run(tmp_path)
    assert len(res.atoms) == 4000
    assert res.n_deleted == 0
    assert res.gates.all_passed()
    g14 = [r for r in res.gates.results() if r.gate == "G14"][0]
    assert "ok" in g14.message and "WARN" not in g14.message
    g3 = [r for r in res.gates.results() if r.gate == "G3"][0]
    assert g3.measured == 0.0
    assert res.boundary_reports == []
    # No vertices.csv (not a FlatTessellation) and no seeding timing row.
    assert not (res.outdir / "vertices.csv").exists()
    assert "seeding" not in res.timings
    # grains.csv volume == box volume exactly.
    assert res.grain_reports[0].volume_A3 == pytest.approx(36.15**3)
    assert res.grain_reports[0].volume_fraction == 1.0


def test_e2e_rotated_self_boundary(tmp_path):
    """Incommensurate orientation: G14 WARNs, the box-face self-boundary
    is cleaned by overlap removal, G7 still proves the cutoff."""
    res = _run(tmp_path, orientation={
        "scheme": "fixed",
        "fixed": {"axis_angle": {"axis": [0, 0, 1], "angle_deg": 15.0}}})
    g14 = [r for r in res.gates.results() if r.gate == "G14"][0]
    assert "WARN" in g14.message and "INCOMMENSURATE" in g14.message
    assert res.n_deleted > 0
    assert res.gates.all_passed()   # G14 is warn-only


def test_e2e_slab_single_crystal(tmp_path):
    """Free z axis: no z self-image; commensurate in x/y only."""
    raw = _raw(output={"directory": str(tmp_path / "out")})
    raw["box"] = {"lengths": [36.15, 36.15, 36.15],
                  "periodic": [True, True, False], "vacuum": 10.0}
    from grainsmith.pipeline import run
    res = run(resolve_config(raw))
    assert res.gates.all_passed()
    g14 = [r for r in res.gates.results() if r.gate == "G14"][0]
    assert "WARN" not in g14.message


def test_e2e_deterministic(tmp_path):
    res1 = _run(tmp_path / "a")
    res2 = _run(tmp_path / "b")
    np.testing.assert_array_equal(res1.atoms.pos, res2.atoms.pos)
    np.testing.assert_array_equal(res1.atoms.species, res2.atoms.species)
