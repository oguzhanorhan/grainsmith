"""Tests for atoms/fill.py — §10 test_fill.py.

Pins the §6.8 correctness model: home cells tile the torus exactly once,
so a single grain spanning a commensurate periodic box reproduces the ideal
crystal count EXACTLY (G8 ± 0), and a rotated (incommensurate) lattice still
reproduces the ideal density with no duplicated/interpenetrating sites.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import KDTree

from grainsmith.atoms.fill import AtomBlock, _compute_d_nn, fill_grain
from grainsmith.constants import MEMORY_HARD_LIMIT_BYTES
from grainsmith.errors import TessellationError
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.spacegroup import (
    WyckoffSite,
    expand_wyckoff,
    hall_from_international,
    symmetry_ops,
)
from grainsmith.orientation.quaternion import axis_angle_to_quat
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


A_FCC = 3.615  # Å, Cu
FRAC_FCC = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
])


def _fcc_crystal():
    A = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
    hall = hall_from_international(225)
    rots, trans = symmetry_ops(hall)
    sites = [WyckoffSite("Cu", [0.0, 0.0, 0.0])]
    basis = expand_wyckoff(sites, rots, trans)
    return A, basis


def _bcc_crystal():
    A = cell_matrix(2.88, 2.88, 2.88, 90.0, 90.0, 90.0)
    hall = hall_from_international(221)
    rots, trans = symmetry_ops(hall)
    sites = [
        WyckoffSite("Ni", [0.0, 0.0, 0.0]),
        WyckoffSite("Al", [0.5, 0.5, 0.5]),
    ]
    basis = expand_wyckoff(sites, rots, trans)
    return A, basis


def _single_grain_box(n_cells: int):
    L = np.array([n_cells * A_FCC] * 3)
    seeds = np.array([[0.47 * L[0], 0.52 * L[1], 0.49 * L[2]]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    return tess, L


# ------------------------------------------------------------------ G8 pins

def test_fill_single_grain_exact_count_identity():
    """Single grain spanning a commensurate periodic box, identity
    orientation: ideal crystal count ± 0 (§10, gate G8 pin)."""
    A, basis = _fcc_crystal()
    n_cells = 4
    tess, L = _single_grain_box(n_cells)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    block = fill_grain(0, tess, basis.frac, basis.species, basis.occupancy,
                       A, q_id, L, [True, True, True], _rng(0))
    expected = 4 * n_cells**3
    assert len(block) == expected, f"{len(block)} atoms != {expected} (exact)"

    # Wrapped positions are unique on the torus (no double counting)
    wrapped = block.pos % L
    uniq = np.unique(np.round(wrapped, 4), axis=0)
    assert len(uniq) == len(block), "duplicate torus sites after wrapping"


def test_compute_d_nn_skewed_cell_matches_brute_force():
    """A skewed (non-reduced) triclinic cell's true nearest-neighbour distance
    can require lattice-translation coefficients outside {-1,0,1}; the fixed ±1
    shell overestimated d_nn several-fold (here ~3.62 vs ~0.89 Å), inflating the
    default overlap cutoff 0.85·d_nn and spuriously deleting boundary atoms."""
    A = cell_matrix(8.337, 7.864, 3.619, 60.05, 115.67, 55.73)
    frac = np.array([[0.0, 0.0, 0.0]])
    d = _compute_d_nn(frac, A)
    best = np.inf
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                if i == j == k == 0:
                    continue
                v = A @ np.array([i, j, k], dtype=float)
                best = min(best, float(np.linalg.norm(v)))
    assert d == pytest.approx(best, rel=1e-9), f"d_nn {d} != brute force {best}"


def test_fill_commensurate_slab_keeps_both_free_walls():
    """Regression: a commensurate simple-cubic slab (free x axis, L = n·a) must
    retain BOTH the x=0 and x=L atom layers.  The owns() free-axis wall defect
    silently dropped one full wall layer (80 → 64 atoms for this cell), a
    scientifically wrong surface for any slab/thin-film run."""
    from grainsmith.tessellation.single import SingleCrystalTessellation
    a = 2.5
    L = np.array([10.0, 10.0, 10.0])       # 4 cells per axis
    A = cell_matrix(a, a, a, 90.0, 90.0, 90.0)
    frac = np.array([[0.0, 0.0, 0.0]])
    periodic = [False, True, True]         # x is the free (slab) axis
    tess = SingleCrystalTessellation(L, periodic)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    block = fill_grain(0, tess, frac, ["Cu"], [None], A, q_id, L,
                       periodic, _rng(0))
    xs = np.unique(np.round(block.pos[:, 0], 4))
    assert np.isclose(xs.min(), 0.0), f"low wall x=0 missing: {xs}"
    assert np.isclose(xs.max(), 10.0), f"high wall x=L missing: {xs}"
    assert len(xs) == 5, f"expected 5 x-layers (both walls), got {xs}"
    assert len(block) == 5 * 4 * 4, f"{len(block)} atoms != 80"


def test_fill_rotated_density_invariant():
    """45° rotation (incommensurate lattice): density within 2% of ideal and
    no interpenetrating sites — the former wrap+dedup scheme failed this by
    +656% (AUDIT C1 regression)."""
    A, basis = _fcc_crystal()
    n_cells = 6
    tess, L = _single_grain_box(n_cells)
    q_rot = axis_angle_to_quat([0.0, 0.0, 1.0], 45.0)
    block = fill_grain(0, tess, basis.frac, basis.species, basis.occupancy,
                       A, q_rot, L, [True, True, True], _rng(1))
    expected = 4 * n_cells**3
    rel = abs(len(block) - expected) / expected
    assert rel <= 0.02, f"density error {rel:.3%} ({len(block)} vs {expected})"

    # An incommensurate single grain in a periodic box necessarily forms a
    # self-image boundary at the cell faces, where close pairs are physical
    # (the overlap stage's job).  Interpenetrating duplicate lattices — the
    # old failure mode — produce close pairs THROUGHOUT the cell interior,
    # so we assert no close pair away from the boundary band.
    d_nn = _compute_d_nn(basis.frac, A)
    wrapped = block.pos % L
    wrapped[wrapped >= L] = 0.0
    tree = KDTree(wrapped, boxsize=L)
    close = tree.query_pairs(0.8 * d_nn)
    margins = tess.margin(block.pos, 0)
    interior_close = [
        (a, b) for a, b in close
        if margins[a] > d_nn and margins[b] > d_nn
    ]
    assert not interior_close, (
        f"{len(interior_close)} close pair(s) in the cell interior — "
        "interpenetrating lattices (AUDIT C1 failure mode)"
    )


def test_fill_two_grains_total_density():
    """Two grains: total atom count within 5% of ideal density (G8 band);
    GB surface fluctuations are the only allowed deviation."""
    A, basis = _fcc_crystal()
    n_cells = 4
    L = np.array([n_cells * A_FCC] * 3)
    rng = _rng(2)
    seeds = seed_grains(2, L, [True, True, True], rng)
    tess = FlatTessellation(seeds, L, [True, True, True])
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    blocks = [
        fill_grain(g, tess, basis.frac, basis.species, basis.occupancy,
                   A, q_id, L, [True, True, True], rng)
        for g in range(2)
    ]
    total = AtomBlock.concatenate(blocks)
    expected = 4 * n_cells**3
    rel = abs(len(total) - expected) / expected
    assert rel <= 0.05, f"two-grain density error {rel:.3%}"


# ------------------------------------------------------------------ basics

def test_fill_species_correct():
    """Species array matches basis symbols."""
    A, basis = _bcc_crystal()
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[10.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    block = fill_grain(0, tess, basis.frac, basis.species, basis.occupancy,
                       A, q_id, L, [True, True, True], _rng(1))
    assert set(np.unique(block.species)).issubset({"Ni", "Al"})


def test_fill_grain_id():
    """All atoms carry the grain_id passed to fill_grain."""
    A, basis = _fcc_crystal()
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[10.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    block = fill_grain(0, tess, basis.frac, basis.species, basis.occupancy,
                       A, q_id, L, [True, True, True], _rng(2))
    assert np.all(block.grain == 0)


def test_atom_block_concatenate():
    b1 = AtomBlock(
        pos=np.zeros((3, 3), dtype=np.float64),
        species=np.array(["Cu", "Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 0, 0], dtype=np.int32),
    )
    b2 = AtomBlock(
        pos=np.ones((2, 3), dtype=np.float64),
        species=np.array(["Ni", "Al"], dtype="U2"),
        grain=np.array([1, 1], dtype=np.int32),
    )
    combined = AtomBlock.concatenate([b1, b2])
    assert len(combined) == 5
    assert combined.pos.shape == (5, 3)


# ------------------------------------------------------- occupancy and d_nn

def test_occupancy_stoichiometry():
    """Solid-solution occupancy sampling reproduces the nominal fraction
    within 4σ of the binomial expectation (fixed seed, §10)."""
    A = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
    n_cells = 6
    tess, L = _single_grain_box(n_cells)
    occ = {"Cu": 0.7, "Ni": 0.3}
    species = ["Cu"] * 4
    occupancy = [occ] * 4
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    block = fill_grain(0, tess, FRAC_FCC, species, occupancy,
                       A, q_id, L, [True, True, True], _rng(7))
    n_tot = len(block)
    assert n_tot == 4 * n_cells**3
    n_ni = int(np.sum(block.species == "Ni"))
    mean = 0.3 * n_tot
    sigma = np.sqrt(n_tot * 0.3 * 0.7)
    assert abs(n_ni - mean) < 4.0 * sigma, (
        f"Ni count {n_ni} vs binomial {mean:.1f} ± {sigma:.1f}"
    )


def test_d_nn_single_atom_basis():
    """d_nn for a one-atom basis is the nearest periodic-image distance —
    the unconditional self-masking returned inf here (AUDIT regression)."""
    a = 3.0
    A = cell_matrix(a, a, a, 90.0, 90.0, 90.0)
    d = _compute_d_nn(np.array([[0.0, 0.0, 0.0]]), A)
    assert np.isfinite(d)
    assert abs(d - a) < 1e-12


def test_d_nn_fcc_and_b2():
    A = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
    d = _compute_d_nn(FRAC_FCC, A)
    assert abs(d - A_FCC / np.sqrt(2)) < 1e-12

    a = 2.88
    A2 = cell_matrix(a, a, a, 90.0, 90.0, 90.0)
    frac_b2 = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    d2 = _compute_d_nn(frac_b2, A2)
    assert abs(d2 - a * np.sqrt(3) / 2) < 1e-12


def test_fill_grains_parallel_matches_serial():
    """§13: jobs=2 must reproduce jobs=1 bit-identically — per-grain rng
    streams are stateless re-derivations (same master seed → same atoms),
    including stochastic occupancy sampling."""
    from grainsmith.atoms.fill import fill_grains
    from grainsmith.rng import make_rng

    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    frac = np.array([[0.0, 0.0, 0.0]])
    species = ["Cu"]
    occupancy = [{"Cu": 0.5, "Ni": 0.5}]   # exercises the rng streams
    quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 30.0),
    ])

    bundle = make_rng(7)
    common = (tess, frac, species, occupancy, A, quats, L,
              [True, True, True])
    blocks_s = fill_grains(*common, rngs=bundle.occupancy_streams(2),
                           jobs=1)
    blocks_p = fill_grains(*common, rngs=bundle.occupancy_streams(2),
                           jobs=2)

    assert len(blocks_s) == len(blocks_p) == 2
    for bs, bp in zip(blocks_s, blocks_p, strict=True):
        assert len(bs) > 0
        np.testing.assert_array_equal(bs.pos, bp.pos)
        assert np.array_equal(bs.species, bp.species)
        assert np.array_equal(bs.grain, bp.grain)
    # the occupancy draw really is stochastic (both species appear)
    all_species = np.concatenate([b.species for b in blocks_s])
    assert {"Cu", "Ni"} <= set(all_species.tolist())


def test_occupancy_streams_stateless():
    """RNGBundle.occupancy_streams: repeated calls and different n give the
    same stream for the same grain id (stateless spawn-key extension)."""
    from grainsmith.rng import make_rng

    b = make_rng(123)
    a1 = [g.random(4) for g in b.occupancy_streams(3)]
    a2 = [g.random(4) for g in b.occupancy_streams(3)]
    a3 = [g.random(4) for g in b.occupancy_streams(5)[:3]]
    for x, y, z in zip(a1, a2, a3, strict=True):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, z)
    # different grains get different streams
    assert not np.array_equal(a1[0], a1[1])


# ---------------------------------------------------------------------------
# Fix #5 — int32 overflow: large n_max triggers memory guard cleanly
# ---------------------------------------------------------------------------


def test_fill_memory_guard_large_n_max():
    """Fix #5: a huge bounding radius must trigger TessellationError via the
    memory guard rather than silently overflowing int32 to a negative value
    (which previously bypassed the guard and crashed in np.meshgrid).

    We use a mock tessellation whose bounding_radius returns a value large
    enough that n_max ~700 per axis (> int32 overflow threshold ~740 per axis
    when products wrap to negative).  The guard must fire before any meshgrid
    allocation."""

    # Build a minimal real tess for seeds/periodic — we only mock bounding_radius
    L = np.array([10.0, 10.0, 10.0])
    seeds = np.array([[5.0, 5.0, 5.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])

    # Wrap tess with a stub bounding_radius returning a huge value.
    # n_max per axis ≈ ceil(huge_r / a) — with a=3 Å and r=3000 Å,
    # n_max ≈ 1000 per axis → grid ≈ 2001^3 ≈ 8e9 > int32 max.
    # Only bounding_radius/owns/margin are called by fill_grain; seeds is
    # accessed as an attribute; periodic is a separate fill_grain argument.
    class _BigRadiusTess:
        seeds = tess.seeds
        # fill_grain now reads tess.memory_limit_bytes (D1/D2, not the
        # constants.py module constant) -- this duck-typed stub doesn't
        # subclass Tessellation, so it needs the attribute explicitly.
        memory_limit_bytes = MEMORY_HARD_LIMIT_BYTES

        def bounding_radius(self, grain_id):
            return 3000.0  # Å — forces n_max ~1000 per axis

        def owns(self, X, grain_id):
            return tess.owns(X, grain_id)

        def margin(self, X, grain_id):
            return tess.margin(X, grain_id)

    A = np.eye(3) * 3.0  # 3 Å cubic cell
    frac = np.array([[0.0, 0.0, 0.0]])
    species = ["Cu"]
    occupancy = [None]
    q_id = np.array([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(TessellationError, match="lattice grid"):
        fill_grain(
            0, _BigRadiusTess(), frac, species, occupancy,
            A, q_id, L, [True, True, True], _rng(0),
        )


# ---------------------------------------------------------------------------
# Fix B1 — _attach_margins searchsorted matches per-grain boolean scan
# ---------------------------------------------------------------------------


def test_attach_margins_searchsorted_matches_reference():
    """Fix B1: the O(n_atoms) searchsorted implementation of _attach_margins
    must give margins identical to the old O(n_grains*n_atoms) per-grain
    boolean-scan reference.  The atom array is sorted by grain id as the
    pipeline guarantees via _sort_by_grain before calling _attach_margins."""
    from grainsmith.pipeline import _attach_margins

    # Two-grain tessellation with a GB at x=10
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])

    # Hand-build a small sorted atom block (grain 0 first, then grain 1)
    pos0 = np.array([
        [3.0, 10.0, 10.0],
        [7.0, 10.0, 10.0],
        [9.0, 10.0, 10.0],
    ])
    pos1 = np.array([
        [11.0, 10.0, 10.0],
        [15.0, 10.0, 10.0],
        [19.0, 10.0, 10.0],
    ])
    pos_all = np.vstack([pos0, pos1])
    grain_all = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    atoms = AtomBlock(
        pos=pos_all,
        species=np.array(["Cu"] * 6, dtype="U2"),
        grain=grain_all,
        gb_margin=None,
    )

    # Reference: old per-grain boolean-scan implementation
    ref_margins = np.empty(len(atoms), dtype=np.float64)
    for i in range(2):
        mask = atoms.grain == i
        ref_margins[mask] = tess.margin(atoms.pos[mask], i)

    # New searchsorted implementation (mutates atoms in place — copy first)
    atoms_copy = AtomBlock(
        pos=atoms.pos.copy(),
        species=atoms.species.copy(),
        grain=atoms.grain.copy(),
        gb_margin=None,
    )
    result = _attach_margins(atoms_copy, tess, n_grains=2)

    np.testing.assert_allclose(
        result.gb_margin, ref_margins, atol=1e-12,
        err_msg="_attach_margins searchsorted diverges from per-grain scan"
    )


# ---------------------------------------------------------------------------
# §13: the fill pool must bound its AGGREGATE memory, not just one allocation
# ---------------------------------------------------------------------------


def _two_grain_fill_case():
    """Shared 2-grain fixture for the pool-sizing tests below."""
    from grainsmith.rng import make_rng

    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 30.0),
    ])
    common = (tess, np.array([[0.0, 0.0, 0.0]]), ["Cu"],
              [{"Cu": 0.5, "Ni": 0.5}], A, quats, L, [True, True, True])
    return tess, A, quats, common, make_rng(7)


def test_estimate_fill_grid_bytes_matches_the_in_worker_guard():
    """The driver-side price must be the SAME number fill_grain charges
    itself, or the pre-flight would clamp against a drifting estimate."""
    from grainsmith.atoms.fill import (
        FILL_GRID_BYTES_PER_POINT,
        estimate_fill_grid_bytes,
        lattice_grid_extent,
    )
    from grainsmith.orientation.quaternion import quat_to_matrix

    tess, A, quats, _, _ = _two_grain_fill_case()
    est = estimate_fill_grid_bytes(tess, np.array([[0.0, 0.0, 0.0]]),
                                   A, quats, a_clip=0.0)
    assert len(est) == 2
    Ainv = np.linalg.inv(A)
    for i, got in enumerate(est):
        _, _, n_grid = lattice_grid_extent(
            i, tess, quat_to_matrix(quats[i]), Ainv,
            np.linalg.norm(Ainv, axis=1), r_basis_max=0.0, a_clip=0.0)
        assert got == n_grid * FILL_GRID_BYTES_PER_POINT


def test_fill_pool_clamps_workers_to_the_memory_budget(caplog):
    """With a budget that fits ONE grid, --jobs 2 must clamp to 1 worker
    and say so — the failure mode this replaces is a BrokenProcessPool
    from an OOM-killed worker (the guard is per-allocation, so it never
    fired on the aggregate)."""
    import logging

    from grainsmith.atoms.fill import (
        _resolve_fill_workers,
        estimate_fill_grid_bytes,
    )

    tess, A, quats, _, _ = _two_grain_fill_case()
    est = estimate_fill_grid_bytes(tess, np.array([[0.0, 0.0, 0.0]]),
                                   A, quats, a_clip=0.0)
    tess.memory_limit_bytes = float(max(est)) * 1.5   # room for 1, not 2

    with caplog.at_level(logging.WARNING, logger="grainsmith.atoms.fill"):
        workers, peak = _resolve_fill_workers(est, jobs=2, n=2, tess=tess)

    assert workers == 1
    assert peak == max(est)
    assert "clamping to --jobs 1" in caplog.text
    # Never clamps below 1, even when a single grid exceeds the budget.
    tess.memory_limit_bytes = 1.0
    assert _resolve_fill_workers(est, jobs=2, n=2, tess=tess)[0] == 1


def test_fill_pool_clamp_does_not_change_the_atoms():
    """Clamping is output-SAFE: fill_grains promises bit-identical blocks
    for every worker count, so a memory-clamped run differs only in speed."""
    from grainsmith.atoms.fill import estimate_fill_grid_bytes, fill_grains

    tess, A, quats, common, bundle = _two_grain_fill_case()
    est = estimate_fill_grid_bytes(tess, np.array([[0.0, 0.0, 0.0]]),
                                   A, quats, a_clip=0.0)

    tess.memory_limit_bytes = float(MEMORY_HARD_LIMIT_BYTES)
    free = fill_grains(*common, rngs=bundle.occupancy_streams(2), jobs=2)
    tess.memory_limit_bytes = float(max(est)) * 1.5   # forces 1 worker
    clamped = fill_grains(*common, rngs=bundle.occupancy_streams(2), jobs=2)

    for a, b in zip(free, clamped, strict=True):
        assert len(a) > 0
        np.testing.assert_array_equal(a.pos, b.pos)
        np.testing.assert_array_equal(a.species, b.species)


def test_broken_pool_is_reported_as_a_memory_diagnosis():
    """A worker killed by the OOM-killer must not surface as a bare
    BrokenProcessPool: the driver's own RSS report stays green (it samples
    only the driver), so the traceback alone points nowhere."""
    import concurrent.futures
    from concurrent.futures.process import BrokenProcessPool

    from grainsmith.atoms import fill as fill_mod

    def _boom(*_args, **_kwargs):
        raise BrokenProcessPool("worker vanished")

    # _run_fill_pool imports ProcessPoolExecutor inside the function, so the
    # patch has to land on the module it imports FROM, not on fill.
    saved = concurrent.futures.ProcessPoolExecutor
    concurrent.futures.ProcessPoolExecutor = _boom
    try:
        with pytest.raises(TessellationError) as exc:
            fill_mod._run_fill_pool(
                fill_mod._fill_one, {}, [None, None], 2, workers=2,
                peak=11_384_479_632)
    finally:
        concurrent.futures.ProcessPoolExecutor = saved

    message = str(exc.value)
    assert "OOM-killer" in message
    assert "grains.number" in message      # the highest-leverage remedy
    assert "--jobs 1" in message
    assert "2 workers" in message
    # The measured per-grain peak and the aggregate it implies, so the
    # reader can see WHY it died without re-deriving anything.
    assert "11.4 GB" in message and "22.8 GB" in message
