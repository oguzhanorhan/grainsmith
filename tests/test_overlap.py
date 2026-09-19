"""Tests for atoms/overlap.py — §10 test_overlap.py.

Covers the full §10 row: constructed two-grain close pair removed per
policy (including the GEOMETRIC correctness of delete_shallower), intra-
grain pair raises, ledger bookkeeping (species + grain pair), PBC pair
across the boundary, midpoint_merge semantics, and the §6.9 post-condition.

P4 additions: deterministic canonical pair ordering + parallel
neighbour discovery (``jobs``/``workers``).  See the block below
``test_midpoint_merge_cascade_does_not_raise_intra_grain`` for the
determinism, route-equivalence, and boundary-inclusivity coverage.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import KDTree

from grainsmith.atoms.fill import AtomBlock, fill_grain
from grainsmith.atoms.overlap import (
    _min_image_vec_general,
    _pairs_from_tree,
    _pbc_pairs,
    _pbc_pairs_general,
    _wrap_positions,
    _wrap_positions_general,
    remove_overlaps,
)
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.spacegroup import (
    WyckoffSite,
    expand_wyckoff,
    hall_from_international,
    symmetry_ops,
)
from grainsmith.errors import ConfigError, FillError
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.power import PowerTessellation
from grainsmith.tessellation.single import SingleCrystalTessellation
from grainsmith.tessellation.weighted import (
    AnisotropicTessellation,
    WeightedTessellation,
)

L20 = np.array([20.0, 20.0, 20.0])
PER = [True, True, True]


def _two_grain_tess():
    """Two grains with GB planes at x = 10 and x = 0 (≡ 20)."""
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    return FlatTessellation(seeds, L20, PER)


def _block(pos, species, grain):
    return AtomBlock(
        pos=np.asarray(pos, dtype=np.float64),
        species=np.asarray(species, dtype="U2"),
        grain=np.asarray(grain, dtype=np.int32),
    )


def _skewed_image_case():
    matrix = np.array([[100.0, 0.0, 21.0],
                       [0.0, 100.0, 31.0],
                       [0.0, 0.0, 1.0]])
    fractional = np.array([[0.0, 0.0, 0.0], [0.735, 0.085, 0.5]])
    return matrix, (matrix @ fractional.T).T


@pytest.mark.parametrize("workers", [1, 2])
def test_triclinic_pairs_beyond_fixed_image_shell(workers):
    matrix, positions = _skewed_image_case()
    assert _pbc_pairs_general(
        positions, 4.0, PER, matrix, np.linalg.inv(matrix), workers=workers,
    ) == [(0, 1)]


def test_triclinic_minimum_image_is_not_fractional_rounding():
    matrix, positions = _skewed_image_case()
    displacement = _min_image_vec_general(
        positions[1] - positions[0], PER, matrix, np.linalg.inv(matrix))
    np.testing.assert_allclose(displacement, [0.0, 0.0, -3.5], atol=1e-12)


def test_g7_detects_close_pair_in_highly_skewed_cell():
    from grainsmith.qa import gate_g7_min_distance

    matrix, positions = _skewed_image_case()
    atoms = _block(positions, ["Cu", "Cu"], [0, 0])
    result = gate_g7_min_distance(atoms, 4.0, PER, np.diag(matrix), matrix)
    assert not result.passed
    assert result.measured == 1


def test_delete_shallower_removes_geometrically_shallower():
    """The atom CLOSER to the boundary loses — margins are real signed
    distances, not NaN (AUDIT C3 regression: NaN margins silently degraded
    this policy to a grain-id tie-break)."""
    tess = _two_grain_tess()
    # Atom A: grain 0 at x=8.0 → margin 2.0 ; atom B: grain 1 at x=10.4 → 0.4
    atoms = _block([[8.0, 10.0, 10.0], [10.4, 10.0, 10.0]],
                   ["Ni", "Al"], [0, 1])
    result, ledger = remove_overlaps(atoms, tess, cutoff=3.0,
                                     policy="delete_shallower",
                                     periodic=PER, box_lengths=L20)
    assert len(result) == 1
    assert result.species[0] == "Ni" and int(result.grain[0]) == 0, (
        "delete_shallower must remove the shallower atom (grain 1, margin 0.4)"
    )
    assert ledger.total_deleted == 1
    assert ledger.deletions_by_species == {"Al": 1}


def test_margin_tie_breaks_to_higher_grain_id():
    """Exact margin tie → the higher grain id loses (§6.9)."""
    tess = _two_grain_tess()
    atoms = _block([[9.5, 10.0, 10.0], [10.5, 10.0, 10.0]],
                   ["Ni", "Al"], [0, 1])
    result, _ = remove_overlaps(atoms, tess, cutoff=2.0,
                                policy="delete_shallower",
                                periodic=PER, box_lengths=L20)
    assert len(result) == 1
    assert int(result.grain[0]) == 0


def test_pbc_pair_detected_and_resolved():
    """Pair straddling the periodic boundary (x ≈ 0 ≡ 20) is detected; the
    tie resolves against the higher grain id."""
    tess = _two_grain_tess()
    atoms = _block(
        [[0.5, 10.0, 10.0], [19.5, 10.0, 10.0], [5.0, 3.0, 3.0]],
        ["Ni", "Al", "Ni"], [0, 1, 0])
    result, ledger = remove_overlaps(atoms, tess, cutoff=2.0,
                                     policy="delete_shallower",
                                     periodic=PER, box_lengths=L20)
    assert ledger.total_deleted == 1
    assert np.all(result.grain != 1), "the grain-1 atom of the PBC pair loses"


def test_intra_grain_raises(monkeypatch):
    """Intra-grain overlap means a crystal-build bug → FillError.

    Placed deep in grain 0's interior (margin 7.0 for x=3.0 against the
    x=10 boundary, well outside the GB-shell pre-filter's cutoff=2.0 --
    the shell filter is forced off here so this test pins the
    unconditional baseline behaviour independent of that heuristic. A
    duplicate-lattice-point bug that sits far from every grain boundary
    AND is searched through the (default, active) shell pre-filter is a
    separate, documented, deliberately-accepted gap -- see
    test_overlap_shell_duplicate_bug_deep_interior_known_gap below."""
    monkeypatch.setattr(FlatTessellation, "gb_shell_lower_bound",
                        lambda self, *a, **k: None)
    tess = _two_grain_tess()
    atoms = _block([[3.0, 10.0, 10.0], [3.05, 10.0, 10.0]],
                   ["Ni", "Ni"], [0, 0])
    with pytest.raises(FillError):
        remove_overlaps(atoms, tess, cutoff=2.0, policy="delete_shallower",
                        periodic=PER, box_lengths=L20)


def test_keep_lower_id_and_ledger():
    """keep_lower_id deletes the higher-grain atom; ledger records species
    and grain pair."""
    tess = _two_grain_tess()
    atoms = _block([[9.5, 10.0, 10.0], [10.5, 10.0, 10.0]],
                   ["Ni", "Al"], [0, 1])
    result, ledger = remove_overlaps(atoms, tess, cutoff=2.0,
                                     policy="keep_lower_id",
                                     periodic=PER, box_lengths=L20)
    assert len(result) == 1
    assert int(result.grain[0]) == 0
    assert ledger.total_deleted == 1
    assert ledger.deletions_by_pair == {(0, 1): 1}
    assert ledger.deletions_by_species == {"Al": 1}


def test_midpoint_merge_refuses_multispecies():
    """§6.9: midpoint_merge is elemental-only."""
    tess = _two_grain_tess()
    atoms = _block([[9.5, 10.0, 10.0], [10.5, 10.0, 10.0]],
                   ["Ni", "Al"], [0, 1])
    with pytest.raises(ConfigError):
        remove_overlaps(atoms, tess, cutoff=2.0, policy="midpoint_merge",
                        periodic=PER, box_lengths=L20)


def test_midpoint_merge_position_and_grain():
    """Merged atom sits at the midpoint and belongs to the LOWER grain id
    (AUDIT regression: it used to keep the higher grain id)."""
    tess = _two_grain_tess()
    atoms = _block([[9.5, 10.0, 10.0], [10.5, 10.0, 10.0]],
                   ["Cu", "Cu"], [0, 1])
    result, ledger = remove_overlaps(atoms, tess, cutoff=2.0,
                                     policy="midpoint_merge",
                                     periodic=PER, box_lengths=L20)
    assert len(result) == 1
    assert int(result.grain[0]) == 0
    np.testing.assert_allclose(result.pos[0], [10.0, 10.0, 10.0], atol=1e-12)


def test_midpoint_merge_min_image_midpoint():
    """A PBC-straddling pair merges on the boundary (x ≈ 0), not at the
    naive arithmetic midpoint in the box center (AUDIT regression)."""
    tess = _two_grain_tess()
    atoms = _block([[0.5, 10.0, 10.0], [19.5, 10.0, 10.0]],
                   ["Cu", "Cu"], [0, 1])
    result, _ = remove_overlaps(atoms, tess, cutoff=2.0,
                                policy="midpoint_merge",
                                periodic=PER, box_lengths=L20)
    assert len(result) == 1
    x = float(result.pos[0, 0]) % 20.0
    assert min(x, 20.0 - x) < 1e-9, f"midpoint x={x} not on the boundary"
    assert int(result.grain[0]) == 0


def test_post_condition_no_residual_pairs():
    """After removal, a fresh PBC neighbor query finds no pair < cutoff."""
    tess = _two_grain_tess()
    atoms = _block(
        [[9.3, 10.0, 10.0], [10.4, 10.0, 10.0], [9.6, 11.0, 10.0],
         [10.7, 11.0, 10.0], [3.0, 3.0, 3.0]],
        ["Ni", "Al", "Ni", "Al", "Ni"],
        [0, 1, 0, 1, 0])
    cutoff = 1.5
    result, _ = remove_overlaps(atoms, tess, cutoff=cutoff,
                                policy="delete_shallower",
                                periodic=PER, box_lengths=L20)
    wrapped = result.pos % L20
    wrapped[wrapped >= L20] = 0.0
    tree = KDTree(wrapped, boxsize=L20)
    assert not tree.query_pairs(cutoff)


def test_margins_finite_in_periodic_two_grain():
    """The configuration that used to produce NaN margins (periodic
    self-image faces) now yields finite signed values."""
    tess = _two_grain_tess()
    pts = np.array([[9.5, 10.0, 10.0], [5.0, 10.0, 10.0], [10.5, 10.0, 10.0]])
    m0 = tess.margin(pts, 0)
    assert np.all(np.isfinite(m0))
    assert m0[0] > 0 and m0[1] > 0 and m0[2] < 0


def test_midpoint_merge_cascade_does_not_raise_intra_grain():
    """Regression: a midpoint-merged atom sits OFF the lattice and may end
    up < cutoff from its own grain's neighbors on a later pass — that is a
    legitimate cascade (merge again), not a crystal-build error.  Only
    PRISTINE same-grain pairs below cutoff at the unwrapped distance raise.
    """
    import numpy as np

    from grainsmith.atoms.fill import AtomBlock

    L = np.array([20.0, 20.0, 20.0])
    # d=(0,0,0) g0 and b=(1.05,0,0) g0 are a valid lattice pair (> cutoff);
    # c=(0.75,0,0) g1 overlaps both → merging walks a survivor into the
    # 0..1.05 gap, creating same-grain sub-cutoff pairs with moved atoms.
    atoms = AtomBlock(
        pos=np.array([[0.0, 0.0, 0.0],
                      [1.05, 0.0, 0.0],
                      [0.75, 0.0, 0.0]]),
        species=np.array(["Cu", "Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 0, 1], dtype=np.int32),
    )
    tess = _two_grain_tess()  # margins unused by midpoint_merge
    result, ledger = remove_overlaps(
        atoms, tess, cutoff=1.0, policy="midpoint_merge",
        periodic=[True, True, True], box_lengths=L,
    )
    # cascade converges; the post-condition (fresh query) already proved
    # no remaining sub-cutoff pair
    assert ledger.total_deleted == 3 - len(result)
    assert len(result) >= 1
    assert set(result.grain.tolist()) <= {0, 1}


def test_midpoint_merge_shell_filter_does_not_drop_moved_atom_pair():
    """Regression, same cascade shape as
    test_midpoint_merge_cascade_does_not_raise_intra_grain but centered on
    grain 0's seed in y/z (so the only nearby GB is the real x=0 boundary
    with grain 1, not a y/z self-image face): the first-pass merge
    relocates d to x=0.375 (moved, grain 0); b at x=1.05 survives UNMOVED
    with margin 1.05 -- just OUTSIDE the shell (cutoff=1.0) -- while
    sitting only 0.675 A from the moved survivor, well under cutoff.  The
    shell's exactness proof requires the pair's two endpoints to be owned
    by DIFFERENT replicas; both are grain 0 here, so nothing bounds b's
    margin by the moved survivor's position.  Forcing only the MOVED atom
    into the shell (rather than bypassing the shell entirely on any moved
    query) leaves b excluded, the (b, d) pair goes undetected, and the
    cascade stops one merge generation early -- asserts byte-for-byte
    equality between the shell filter's on and forced-off behaviour, which
    that narrower fix violates.

    remove_overlaps only applies the shell filter at ``workers > 1`` (the
    jobs=1 serial route skips it -- see remove_overlaps' docstring), so the
    "on" run below passes ``_force_shell=True`` to exercise pass 0's
    shell-active code path even at the default jobs=1 used here; this is
    the same test-only knob ``test_overlap_shell_bitwise`` uses for the
    same reason."""
    L = np.array([20.0, 20.0, 20.0])
    atoms = _block(
        [[0.0, 10.0, 10.0], [1.05, 10.0, 10.0], [0.75, 10.0, 10.0]],
        ["Cu", "Cu", "Cu"], [0, 0, 1])
    tess = _two_grain_tess()

    def _run(force_off):
        orig = FlatTessellation.gb_shell_lower_bound
        if force_off:
            FlatTessellation.gb_shell_lower_bound = (
                lambda self, *a, **k: None)
        try:
            return remove_overlaps(
                atoms, tess, cutoff=1.0, policy="midpoint_merge",
                periodic=[True, True, True], box_lengths=L,
                _force_shell=not force_off,
            )
        finally:
            FlatTessellation.gb_shell_lower_bound = orig

    result_on, ledger_on = _run(force_off=False)
    result_off, ledger_off = _run(force_off=True)

    assert result_on.pos.tobytes() == result_off.pos.tobytes(), (
        "shell filter changed the final positions"
    )
    assert result_on.grain.tobytes() == result_off.grain.tobytes()
    assert ledger_on.total_deleted == ledger_off.total_deleted
    # The unfiltered baseline performs a SECOND merge generation (b's own
    # margin excludes it from the shell, but it is genuinely < cutoff from
    # the moved survivor) -- pin that, so this test is not vacuous.
    assert ledger_off.total_deleted == 2


# ---------------------------------------------------------------------------
# P4: deterministic canonical pair ordering + parallel discovery.
#
# _discover_pairs/_pairs_from_tree grew a `workers` route: workers == 1 uses
# KDTree.query_pairs (the pre-P4 code path); workers != 1 uses a vectorized
# two-pass ball-query discovery instead, purely for DISCOVERY SPEED.  Both
# routes are required to return the identical pair SET, and remove_overlaps'
# pair PROCESSING is required to be a pure function of that set (independent
# of which route produced it, and of the set's construction order) — this
# section proves both properties directly, plus the query_pairs-equivalent
# inclusive (d <= cutoff) boundary semantics of the parallel route.
# ---------------------------------------------------------------------------


def _synthetic_two_grain_overlap_block():
    """A 6x6 grid of engineered atom pairs straddling the x=10 GB plane of
    :func:`_two_grain_tess`, spaced >= 3.0 apart in y/z (no incidental
    intra-grain overlap).  The per-pair (grain-0, grain-1) half-gaps from
    the boundary cycle through exact margin TIES (including one pair sitting
    at EXACTLY cutoff=1.0 separation) and clearly asymmetric gaps.  The
    pairs are DISJOINT (no chained conflicts), so per-pair resolution is
    order-independent by construction: what this fixture proves is that the
    tie-break rule and the boundary-inclusivity semantics are identical
    across discovery routes and worker counts -- NOT that it would catch an
    order-sensitive deletion loop (that requires chained conflicts, which
    the pipeline-level byte-identity test and the deferred same-grain
    verdict logic cover).
    """
    ys = np.arange(2.0, 18.0, 3.0)
    zs = np.arange(2.0, 18.0, 3.0)
    offset_schedule = [
        (0.30, 0.30),   # tie, well inside cutoff
        (0.49, 0.49),   # tie, near cutoff (2d = 0.98 < 1.0)
        (0.50, 0.50),   # tie, EXACTLY at cutoff (2d = 1.00 == 1.0)
        (0.20, 0.45),   # asymmetric: grain 0 shallower
        (0.45, 0.20),   # asymmetric: grain 1 shallower
        (0.05, 0.05),   # tie, very close pair
    ]
    pos, species, grain = [], [], []
    k = 0
    for y in ys:
        for z in zs:
            d0, d1 = offset_schedule[k % len(offset_schedule)]
            pos.append([10.0 - d0, y, z])
            species.append("Ni")
            grain.append(0)
            pos.append([10.0 + d1, y, z])
            species.append("Al")
            grain.append(1)
            k += 1
    return AtomBlock(
        pos=np.asarray(pos, dtype=np.float64),
        species=np.asarray(species, dtype="U2"),
        grain=np.asarray(grain, dtype=np.int32),
    )


def _survivor_signature(result):
    """Order-independent fingerprint of a remove_overlaps result: sorted
    (species, grain, rounded position) rows, so two runs that keep the same
    physical atoms compare equal regardless of internal array order."""
    return sorted(zip(
        result.species.tolist(), result.grain.tolist(),
        [tuple(np.round(p, 9)) for p in result.pos.tolist()],
        strict=True,
    ))


@pytest.mark.parametrize("policy", ["delete_shallower", "keep_lower_id"])
def test_determinism_jobs1_vs_jobs8_synthetic_ties(policy):
    """Same input -> identical deleted-atom set for jobs=1 vs jobs=8, on a
    block engineered with margin ties (including an exact-cutoff tie) that
    the engineered ties are resolved by the deterministic grain-id rule
    identically on both routes (the fixture's disjoint pairs make per-pair
    resolution order-free by construction; see the fixture docstring)."""
    tess = _two_grain_tess()
    atoms_a = _synthetic_two_grain_overlap_block()
    atoms_b = _synthetic_two_grain_overlap_block()

    result1, ledger1 = remove_overlaps(atoms_a, tess, cutoff=1.0, policy=policy,
                                       periodic=PER, box_lengths=L20, jobs=1)
    result8, ledger8 = remove_overlaps(atoms_b, tess, cutoff=1.0, policy=policy,
                                       periodic=PER, box_lengths=L20, jobs=8)

    assert ledger1.total_deleted == ledger8.total_deleted
    assert ledger1.deletions_by_species == ledger8.deletions_by_species
    assert ledger1.deletions_by_pair == ledger8.deletions_by_pair
    assert _survivor_signature(result1) == _survivor_signature(result8), (
        "jobs=1 and jobs=8 must delete exactly the same atoms (P4 "
        "determinism), including at margin-tie and exact-cutoff pairs"
    )


def test_determinism_many_worker_counts_agree():
    """Sweep several worker counts (not just 1 vs 8): jobs maps directly to
    scipy's `workers` (see remove_overlaps), so this guards the pair-set's
    worker-count invariance across the actual range of thread counts a real
    --jobs sweep uses, not just a single hardcoded thread count."""
    tess = _two_grain_tess()
    cutoff = 1.0
    signatures = []
    for jobs in (1, 2, 4, 8):
        atoms = _synthetic_two_grain_overlap_block()
        result, _ = remove_overlaps(atoms, tess, cutoff=cutoff,
                                    policy="delete_shallower",
                                    periodic=PER, box_lengths=L20, jobs=jobs)
        signatures.append(_survivor_signature(result))
    assert all(sig == signatures[0] for sig in signatures[1:])


@pytest.mark.parametrize("periodic", [
    [True, True, True],    # all-periodic -> KDTree(boxsize=L) route
    [True, True, False],   # mixed-periodicity -> ghost-padding route
    [True, False, False],
    [False, False, False],
])
def test_pbc_pairs_route_equivalence_random_clouds(periodic):
    """_pbc_pairs(workers=1) and _pbc_pairs(workers=-1) return the IDENTICAL
    pair set on random point clouds, for every periodicity combination
    (all-periodic boxsize-tree path and the mixed/none ghost-padding path)."""
    rng = np.random.default_rng(20260815)
    L = np.array([15.0, 12.0, 9.0])
    cutoff = 1.2
    for trial in range(6):
        pos = rng.uniform(0, 1, size=(300, 3)) * L
        wrapped = _wrap_positions(pos, periodic, L)
        pairs_serial = set(map(tuple, _pbc_pairs(
            wrapped, cutoff, periodic, L, workers=1)))
        pairs_parallel = set(map(tuple, _pbc_pairs(
            wrapped, cutoff, periodic, L, workers=-1)))
        assert pairs_serial == pairs_parallel, (
            f"trial {trial}, periodic={periodic}: route mismatch, "
            f"sym diff = {pairs_serial ^ pairs_parallel}"
        )
        assert len(pairs_serial) > 0, "test is vacuous without any pairs"


def test_pbc_pairs_route_equivalence_mixed_periodicity_ghost_forced():
    """Mixed-periodicity case where a third of the points are deliberately
    placed within `cutoff` of a periodic face (forcing the ghost-padding
    branch to actually add ghosts, not just exercise the fast interior
    path)."""
    rng = np.random.default_rng(99)
    L = np.array([10.0, 10.0, 10.0])
    periodic = [True, False, True]
    cutoff = 0.8
    n = 200
    pos = rng.uniform(0, 1, size=(n, 3)) * L
    near_face = rng.choice(n, size=n // 3, replace=False)
    pos[near_face, 0] = rng.uniform(0, 0.3, size=len(near_face))
    pos[near_face, 2] = rng.uniform(9.7, 10.0, size=len(near_face))
    wrapped = _wrap_positions(pos, periodic, L)
    pairs_serial = set(map(tuple, _pbc_pairs(
        wrapped, cutoff, periodic, L, workers=1)))
    pairs_parallel = set(map(tuple, _pbc_pairs(
        wrapped, cutoff, periodic, L, workers=-1)))
    assert pairs_serial == pairs_parallel
    assert len(pairs_serial) > 0


@pytest.mark.parametrize("periodic", [
    [True, True, True],
    [True, True, False],
])
def test_pbc_pairs_general_route_equivalence_triclinic(periodic):
    """_pbc_pairs_general (triclinic 27-image path) gives identical
    pair sets for workers=1 vs workers=-1, on a skewed (non-orthogonal)
    box matrix."""
    rng = np.random.default_rng(7)
    H = np.array([
        [12.0, 2.0, 1.0],
        [0.0, 10.0, 1.5],
        [0.0, 0.0, 9.0],
    ])
    Hinv = np.linalg.inv(H)
    cutoff = 1.1
    for trial in range(4):
        frac = rng.uniform(0, 1, size=(250, 3))
        pos = (H @ frac.T).T
        wrapped = _wrap_positions_general(pos, periodic, H, Hinv)
        pairs_serial = set(_pbc_pairs_general(
            wrapped, cutoff, periodic, H, Hinv, workers=1))
        pairs_parallel = set(_pbc_pairs_general(
            wrapped, cutoff, periodic, H, Hinv, workers=-1))
        assert pairs_serial == pairs_parallel, (
            f"trial {trial}, periodic={periodic}: triclinic route mismatch, "
            f"sym diff = {pairs_serial ^ pairs_parallel}"
        )
        assert len(pairs_serial) > 0


def test_boundary_inclusivity_exact_cutoff_distance():
    """A pair at EXACTLY cutoff distance is treated identically by both
    discovery routes: query_pairs' documented semantics are inclusive
    (d <= cutoff), and the k-NN route's distance_upper_bound is inflated by
    a relative epsilon specifically so it matches that inclusive boundary —
    the two routes must never disagree at d == cutoff."""
    cutoff = 2.0
    # A third, distant point keeps the tree non-trivial without
    # contaminating the boundary pair's neighbourhood.
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    assert float(np.linalg.norm(pts[1] - pts[0])) == cutoff  # construction check
    tree = KDTree(pts)
    pairs_serial = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=1).tolist()))
    pairs_parallel = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=-1).tolist()))
    assert (0, 1) in pairs_serial, "query_pairs must include the exact-boundary pair"
    assert (0, 1) in pairs_parallel, "k-NN route must include the exact-boundary pair"
    assert pairs_serial == pairs_parallel


def test_boundary_inclusivity_just_outside_cutoff_excluded_by_both():
    """A pair one ULP beyond cutoff is excluded by both routes (neither
    route is more permissive than the other at the boundary)."""
    cutoff = 2.0
    just_over = np.nextafter(cutoff, np.inf)
    pts = np.array([[0.0, 0.0, 0.0], [just_over, 0.0, 0.0], [50.0, 50.0, 50.0]])
    tree = KDTree(pts)
    pairs_serial = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=1).tolist()))
    pairs_parallel = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=-1).tolist()))
    assert (0, 1) not in pairs_serial
    assert (0, 1) not in pairs_parallel
    assert pairs_serial == pairs_parallel == set()


def test_boundary_inclusivity_sweep_random_near_cutoff_pairs():
    """Bulk sweep: many pairs placed within a tight band around cutoff (both
    just inside and just outside), well separated in space so each pair's
    verdict is independent -- the two routes must agree on EVERY one."""
    rng = np.random.default_rng(31415)
    cutoff = 1.5
    n_pairs = 400
    spacing = 50.0
    offsets = rng.uniform(-1e-6, 1e-6, size=n_pairs)
    dirs = rng.normal(size=(n_pairs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts_a = np.stack([np.arange(n_pairs) * spacing,
                       np.zeros(n_pairs), np.zeros(n_pairs)], axis=1)
    pts_b = pts_a + dirs * (cutoff + offsets)[:, None]
    all_pts = np.empty((2 * n_pairs, 3))
    all_pts[0::2] = pts_a
    all_pts[1::2] = pts_b

    tree = KDTree(all_pts)
    pairs_serial = set(map(tuple, _pairs_from_tree(tree, all_pts, cutoff, workers=1).tolist()))
    pairs_parallel = set(map(tuple, _pairs_from_tree(tree, all_pts, cutoff, workers=-1).tolist()))
    assert pairs_serial == pairs_parallel, (
        f"sym diff at near-cutoff boundary sweep: {pairs_serial ^ pairs_parallel}"
    )


def test_parallel_route_matches_query_pairs_on_dense_cluster():
    """A dense cluster where many points have > 8 neighbours within cutoff
    (large per-point neighbour lists in the ball-query route) still agrees
    exactly with the serial query_pairs route."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 3.0, size=(500, 3))
    cutoff = 1.0
    tree = KDTree(pts)
    pairs_serial = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=1).tolist()))
    pairs_parallel = set(map(tuple, _pairs_from_tree(tree, pts, cutoff, workers=-1).tolist()))
    assert pairs_serial == pairs_parallel
    # sanity: cutoff is large relative to density, so this is genuinely a
    # "many neighbours per point" case (long ball lists), not a no-op.
    counts = np.bincount(np.concatenate([
        np.asarray(list(pairs_serial))[:, 0], np.asarray(list(pairs_serial))[:, 1],
    ]).astype(np.intp), minlength=len(pts))
    assert counts.max() > 8


def test_parallel_route_all_pairs_within_cutoff_complete():
    """When cutoff captures EVERY pair (each point's ball is the whole
    set), the two-pass ball route still finds all n*(n-1)/2 pairs,
    matching query_pairs exactly."""
    rng = np.random.default_rng(1)
    n = 50
    pts = rng.uniform(0, 0.01, size=(n, 3))  # tiny cluster
    cutoff = 1.0  # cutoff >> cluster extent -> every pair is within cutoff
    tree = KDTree(pts)
    pairs_serial = _pairs_from_tree(tree, pts, cutoff, workers=1)
    pairs_parallel = _pairs_from_tree(tree, pts, cutoff, workers=-1)
    expected = n * (n - 1) // 2
    assert len(pairs_serial) == expected
    assert len(pairs_parallel) == expected
    assert set(map(tuple, pairs_serial.tolist())) == set(map(tuple, pairs_parallel.tolist()))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_pairs_from_tree_small_n_edge_cases(n):
    """n < 2 short-circuits to an empty result without building a
    degenerate query; n in {2, 3} still agrees between routes."""
    pts = np.random.default_rng(n).uniform(0, 5, size=(n, 3)) if n else np.empty((0, 3))
    cutoff = 10.0
    if n < 2:
        pairs = _pairs_from_tree(None, pts, cutoff, workers=1)  # tree unused for n<2
        assert pairs.shape == (0, 2)
        return
    tree = KDTree(pts)
    pairs_serial = _pairs_from_tree(tree, pts, cutoff, workers=1)
    pairs_parallel = _pairs_from_tree(tree, pts, cutoff, workers=-1)
    expected = n * (n - 1) // 2
    assert len(pairs_serial) == expected == len(pairs_parallel)


def test_parallel_chunking_is_row_count_independent():
    """The row-chunked ball-query counting pass (bounded transient memory
    for large point counts -- see _KNN_CHUNK_ROWS) returns a BIT-IDENTICAL
    result regardless of the chunk size, including pathologically small
    chunks that force many chunk boundaries mid-cluster."""
    import grainsmith.atoms.overlap as overlap_mod

    rng = np.random.default_rng(2026)
    n = 733
    pts = rng.uniform(0, 50, size=(n, 3))
    cutoff = 1.3
    tree = KDTree(pts)

    baseline = _pairs_from_tree(tree, pts, cutoff, workers=-1)
    baseline_set = set(map(tuple, baseline.tolist()))

    orig_chunk = overlap_mod._KNN_CHUNK_ROWS
    try:
        for chunk_size in (1, 7, 50, 500, n, 10_000):
            overlap_mod._KNN_CHUNK_ROWS = chunk_size
            result = _pairs_from_tree(tree, pts, cutoff, workers=-1)
            assert set(map(tuple, result.tolist())) == baseline_set, (
                f"chunk_size={chunk_size} disagrees with the unchunked baseline"
            )
            assert np.array_equal(result, baseline), (
                f"chunk_size={chunk_size} preserves the pair SET but not the "
                "canonical ORDER"
            )
    finally:
        overlap_mod._KNN_CHUNK_ROWS = orig_chunk


# ---------------------------------------------------------------------------
# remove_overlaps is a provably single-pass algorithm under
# delete_shallower/keep_lower_id -- pass >= 1 filters pass 0's pair list
# against the `deleted` mask (O(n_pairs), no tree) instead of rebuilding a
# KDTree over all survivors.  These tests prove the shortcut
# (remove_overlaps' default) is bit-identical to always re-querying
# (`_force_full_requery=True`), and that it actually skips the tree
# rebuild (call-count instrumentation).
# ---------------------------------------------------------------------------


def _random_overlap_system(rng, force_duplicate_bug=False, force_self_image=False):
    """A small randomized periodic multi-grain system with several
    engineered close pairs (inter-grain), optionally augmented with a
    same-grain SELF-IMAGE pair (close only through the periodic wrap --
    the unwrapped separation is ~L, so this is a legitimate GB, not a
    build error) and/or a duplicated-LATTICE-POINT pair (two atoms of one
    grain close at the UNWRAPPED distance, which §6.9 requires to raise
    FillError).  Single species throughout so the same system exercises
    midpoint_merge too (elemental-only)."""
    L_side = float(rng.uniform(18.0, 30.0))
    L = np.array([L_side, L_side, L_side])
    n_grains = int(rng.integers(2, 5))
    seeds = rng.uniform(2.0, L_side - 2.0, size=(n_grains, 3))
    tess = FlatTessellation(seeds, L, PER)
    cutoff = float(rng.uniform(0.6, 1.4))

    pos: list[np.ndarray] = []
    grain: list[int] = []
    for _ in range(int(rng.integers(2, 7))):
        g0, g1 = rng.choice(n_grains, size=2, replace=False)
        base = rng.uniform(3.0, L_side - 3.0, size=3)
        d = float(rng.uniform(0.05, cutoff * 0.9))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        pos.append(base - 0.5 * d * axis)
        grain.append(int(g0))
        pos.append(base + 0.5 * d * axis)
        grain.append(int(g1))

    if force_self_image:
        g = int(rng.integers(0, n_grains))
        y, z = rng.uniform(3.0, L_side - 3.0, size=2)
        d = float(rng.uniform(0.05, cutoff * 0.9))
        pos.append(np.array([0.3 * d, y, z]))
        grain.append(g)
        pos.append(np.array([L_side - 0.3 * d, y, z]))
        grain.append(g)

    if force_duplicate_bug:
        g = int(rng.integers(0, n_grains))
        base = rng.uniform(4.0, L_side - 4.0, size=3)
        d = float(rng.uniform(0.01, cutoff * 0.5))
        pos.append(base)
        grain.append(g)
        pos.append(base + np.array([d, 0.0, 0.0]))
        grain.append(g)

    atoms = _block(pos, ["Cu"] * len(pos), grain)
    return atoms, tess, L, cutoff


@pytest.mark.parametrize(
    "policy", ["delete_shallower", "keep_lower_id", "midpoint_merge"]
)
def test_overlap_single_pass_equivalence(policy):
    """The single-pass deleted-mask filter (default) is bit-identical to
    always re-querying (``_force_full_requery=True``) across ~50
    randomized systems, including self-image periodic GB geometries and a
    forced duplicated-lattice-point FillError case.  For midpoint_merge
    both configurations run the identical always-re-query code path (see
    remove_overlaps), so this doubles as a no-op regression guard there.
    """
    rng = np.random.default_rng(2026_08_23)
    n_trials = 50
    for i in range(n_trials):
        force_self_image = i % 5 == 0
        force_duplicate_bug = i % 7 == 0
        atoms, tess, L, cutoff = _random_overlap_system(
            rng, force_duplicate_bug=force_duplicate_bug,
            force_self_image=force_self_image)

        raised_fast = raised_forced = None
        try:
            result_fast, ledger_fast = remove_overlaps(
                atoms, tess, cutoff=cutoff, policy=policy,
                periodic=PER, box_lengths=L, _force_full_requery=False)
        except FillError as exc:
            raised_fast = exc
        try:
            result_forced, ledger_forced = remove_overlaps(
                atoms, tess, cutoff=cutoff, policy=policy,
                periodic=PER, box_lengths=L, _force_full_requery=True)
        except FillError as exc:
            raised_forced = exc

        if raised_fast is not None or raised_forced is not None:
            assert raised_fast is not None and raised_forced is not None, (
                f"trial {i} ({policy}): fast path and forced re-query "
                f"DISAGREED on whether to raise FillError "
                f"(fast={raised_fast!r}, forced={raised_forced!r})"
            )
            assert str(raised_fast) == str(raised_forced), f"trial {i} ({policy})"
            continue

        assert result_fast.pos.tobytes() == result_forced.pos.tobytes(), (
            f"trial {i} ({policy}): positions differ"
        )
        assert result_fast.species.tobytes() == result_forced.species.tobytes(), (
            f"trial {i} ({policy}): species differ"
        )
        assert result_fast.grain.tobytes() == result_forced.grain.tobytes(), (
            f"trial {i} ({policy}): grain assignment differs"
        )
        assert (ledger_fast.deletions_by_species
                == ledger_forced.deletions_by_species), f"trial {i} ({policy})"
        assert (ledger_fast.deletions_by_pair
                == ledger_forced.deletions_by_pair), f"trial {i} ({policy})"
        assert ledger_fast.total_deleted == ledger_forced.total_deleted, (
            f"trial {i} ({policy})"
        )


def test_overlap_second_pass_is_empty(monkeypatch):
    """delete_shallower and keep_lower_id trigger exactly ONE pair-
    discovery call per remove_overlaps -- ``_pairs_from_tree`` is the
    common choke point for every periodicity route (``_pbc_pairs``'
    all-periodic and ghost-padding branches, and ``_pbc_pairs_general``),
    so instrumenting it there covers all three.  midpoint_merge, which can
    create new pairs by moving survivors, must keep re-querying: >= 1 call
    always, and the merge-cascade fixture forces >= 2."""
    import grainsmith.atoms.overlap as overlap_mod

    calls = {"n": 0}
    orig = overlap_mod._pairs_from_tree

    def _counting_wrapper(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(overlap_mod, "_pairs_from_tree", _counting_wrapper)

    tess = _two_grain_tess()
    for policy in ("delete_shallower", "keep_lower_id"):
        calls["n"] = 0
        atoms = _synthetic_two_grain_overlap_block()
        remove_overlaps(atoms, tess, cutoff=1.0, policy=policy,
                        periodic=PER, box_lengths=L20)
        assert calls["n"] == 1, (
            f"{policy}: expected exactly 1 pair-discovery call (single-pass "
            f"proof), got {calls['n']}"
        )

    calls["n"] = 0
    atoms = AtomBlock(
        pos=np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        species=np.array(["Cu", "Cu", "Cu"], dtype="U2"),
        grain=np.array([0, 0, 1], dtype=np.int32),
    )
    remove_overlaps(atoms, tess, cutoff=1.0, policy="midpoint_merge",
                    periodic=PER, box_lengths=L20)
    assert calls["n"] >= 2, (
        "midpoint_merge's merge-cascade fixture is expected to force "
        f"multiple real re-queries, got {calls['n']}"
    )


# ---------------------------------------------------------------------------
# GB-shell pre-filter: Tessellation.gb_shell_lower_bound + remove_overlaps'
# _pairs_with_shell integration.
#
# The exactness precondition ("same-replica atoms of a perfect lattice are
# >= d_nn > cutoff apart") requires a REAL lattice fill, not a synthetic
# random point cloud -- two independent uniform-random points can be
# arbitrarily close deep inside one grain, which has nothing to do with a
# grain boundary and would make the superset property meaningless to test.
# The helpers below build real FCC-Cu fills (mirrors tests/test_fill.py's
# fixtures) so the shell tests exercise the actual precondition.
# ---------------------------------------------------------------------------

_A_FCC = 3.615  # Å, Cu (matches tests/test_fill.py)
_Q_ID = np.array([1.0, 0.0, 0.0, 0.0])
_D_NN_FCC = _A_FCC / np.sqrt(2.0)  # ≈ 2.556 Å nearest-neighbour distance


def _fcc_basis():
    A = cell_matrix(_A_FCC, _A_FCC, _A_FCC, 90.0, 90.0, 90.0)
    hall = hall_from_international(225)
    rots, trans = symmetry_ops(hall)
    sites = [WyckoffSite("Cu", [0.0, 0.0, 0.0])]
    basis = expand_wyckoff(sites, rots, trans)
    return A, basis


def _shell_rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _fill_all_grains(tess, n_grains, L, periodic, seed_base=100):
    """Real FCC-Cu lattice fill of every grain of *tess* (identity
    orientation, single species) -- guarantees the physical precondition
    the GB-shell bound needs, unlike a synthetic random point cloud (see
    the section docstring above)."""
    A, basis = _fcc_basis()
    blocks = [
        fill_grain(g, tess, basis.frac, basis.species, basis.occupancy,
                  A, _Q_ID, L, periodic, _shell_rng(seed_base + g))
        for g in range(n_grains)
    ]
    return AtomBlock.concatenate(blocks)


def _multi_grain_flat_case(n_grains=6, mult=4.37, seed=1):
    """A periodic multi-grain box, real FCC fill, cutoff well below d_nn so
    every close pair found is a genuine inter-grain or self-image GB
    contact (never an accidental same-lattice coincidence)."""
    Lval = mult * _A_FCC * 3.0
    L = np.array([Lval, Lval, Lval])
    seeds = seed_grains(n_grains, L, PER, _shell_rng(seed))
    return L, seeds


def _self_image_flat_case():
    """A single grain in a box whose edge is an INCOMMENSURATE multiple of
    the lattice constant (4.15*a): the grain meets its own periodic image
    with a genuine sub-cutoff misfit (measured non-empty at cutoff=1.0 Å,
    d_nn≈2.556 Å), forcing periodic SELF-IMAGE grain-boundary pairs --
    the case ``weighted.margin()`` is blind to and that
    ``FlatTessellation.gb_shell_lower_bound`` must still get right via the
    cell's self-image faces."""
    Lval = 4.15 * _A_FCC
    L = np.array([Lval, Lval, Lval])
    seeds = np.array([[0.5 * Lval, 0.5 * Lval, 0.5 * Lval]])
    return L, seeds


def _inject_duplicate_bug(atoms, tess, L, cutoff, periodic=PER,
                          near_boundary=True):
    """Append a same-grain pair at a genuine duplicate-lattice-point
    separation (well under *cutoff*, at the geometrically correct owning
    grain).

    ``near_boundary=True`` (default) duplicates a REAL lattice atom close
    to a genuine inter-grain boundary (smallest margin among grain *gi*'s
    atoms) that has no OTHER real atom within *cutoff* -- i.e. it is not
    already part of a genuine inter-grain conflict.  That isolation
    matters at a cutoff approaching d_nn: without it, the injected twin's
    same-grain pair can be silently skipped (``remove_overlaps``'s
    ``if deleted[oi] or deleted[oj]: continue``) because its source atom
    gets deleted first via an UNRELATED real conflict, masking the
    injected bug instead of raising it -- a construction artifact, not a
    property of the shell filter, so both the near-boundary property
    (small margin, inside the GB shell -- must raise FillError identically
    with and without the filter, see test_overlap_shell_bitwise) and the
    isolation property (the twin pair is the only sub-cutoff pair either
    atom is in) are required together.

    ``near_boundary=False`` places it deep in a grain's interior instead --
    the deliberately-accepted gap the shell pre-filter does NOT close (see
    test_overlap_shell_duplicate_bug_deep_interior_known_gap)."""
    if near_boundary:
        gi, _ = tess.adjacency()[0]
        wrapped = _wrap_positions(atoms.pos, periodic, L)
        # A safety margin above *cutoff* (rather than cutoff itself): the
        # twin sits 0.05 A off the duplicated source atom, so a source
        # whose true nearest-real-neighbour distance clears cutoff by
        # less than that offset would still leave the TWIN entangled.
        real_pairs = _pbc_pairs(wrapped, cutoff + 0.1, periodic, L,
                                workers=1)
        entangled = set(
            np.asarray(real_pairs, dtype=np.intp).ravel().tolist())
        candidates = np.array(
            [c for c in np.where(atoms.grain == gi)[0] if c not in entangled])
        margins = tess.margin(atoms.pos[candidates], gi)
        src = int(candidates[np.argmin(margins)])
        loc = atoms.pos[src].copy()
        g = int(atoms.grain[src])
    else:
        loc = np.array([0.31, 0.53, 0.44]) * L
        g = int(tess.grain_of(loc[None, :])[0])
    pos = np.vstack([atoms.pos, loc, loc + np.array([0.05, 0.0, 0.0])])
    species = np.concatenate([atoms.species, ["Cu", "Cu"]])
    grain = np.concatenate([atoms.grain, [g, g]])
    return AtomBlock(
        pos=pos, species=species.astype(atoms.species.dtype),
        grain=grain.astype(np.int32),
    )


def _tess_for_backend(backend, seeds, L, periodic, seed):
    """Construct the tessellation backend under test over the SAME seeds/
    box/periodicity: flat and power share FlatTessellation's cell-based
    margin/adjacency (power with nonzero weights), weighted uses the
    additive k-NN shell bound."""
    if backend == "power":
        weights = _shell_rng(seed).normal(0.0, 3.0, size=len(seeds))
        return PowerTessellation(seeds, L, periodic, weights=weights)
    if backend == "weighted":
        return WeightedTessellation(seeds, L, periodic, sigma_w=1.0,
                                    rng=_shell_rng(seed))
    return FlatTessellation(seeds, L, periodic)  # "flat", "flat_free_axis"


def _shell_case_geometry(backend, case):
    """(L, seeds, periodic, n_grains) for one (backend, case) combination
    of the test_overlap_shell_bitwise grid."""
    periodic = [True, True, False] if backend == "flat_free_axis" else PER
    if case == "self_image":
        Lval = 4.15 * _A_FCC
        L = np.array([Lval, Lval, Lval])
        seeds = np.array([[0.5 * Lval, 0.5 * Lval, 0.5 * Lval]])
        return L, seeds, periodic, 1
    if backend == "flat_free_axis":
        Lval = 4.0 * _A_FCC
        L = np.array([Lval, Lval, Lval])
        seeds = _shell_rng(4).uniform(2.0, Lval - 2.0, size=(5, 3))
        return L, seeds, periodic, 5
    L, seeds = _multi_grain_flat_case(n_grains=8, mult=3.1, seed=1)
    return L, seeds, periodic, len(seeds)


def _build_shell_case(backend, case, cutoff):
    """(tess, atoms, periodic, L) for one (backend, case) combination of
    the test_overlap_shell_bitwise grid, on a real FCC-Cu fill."""
    L, seeds, periodic, n_grains = _shell_case_geometry(backend, case)
    tess = _tess_for_backend(backend, seeds, L, periodic, seed=50)
    atoms = _fill_all_grains(tess, n_grains, L, periodic,
                             seed_base=7 if case == "self_image" else 100)
    if case == "duplicate_bug":
        atoms = _inject_duplicate_bug(atoms, tess, L, cutoff, periodic)
    return tess, atoms, periodic, L


@pytest.mark.parametrize("cutoff", [1.0, 0.85 * _D_NN_FCC],
                         ids=["small", "production_0.85dnn"])
@pytest.mark.parametrize("case", [
    "flat_multi", "power_multi", "weighted_additive_multi",
    "flat_self_image", "flat_free_axis",
])
def test_shell_mask_is_superset_of_pair_endpoints(case, cutoff):
    """gb_shell_lower_bound's mask is a SUPERSET of every true pair
    endpoint (the exactness proof this whole pre-filter rests on), on
    real FCC-Cu fills: a periodic multi-grain flat system, the same
    system under the power (Laguerre) and additive-weighted backends, a
    box small enough to force periodic SELF-IMAGE grain boundaries, and a
    box with one free (non-periodic) axis.  Runs at a small cutoff and at
    the production default (0.85*d_nn) -- power's exact face-plane margin
    path (module docstring, ``_exact_margin_shell_mask``) is sanctioned
    for use at the production cutoff specifically, so ``power_multi``
    covers that scope explicitly at both cutoffs, not only the small
    one."""
    assert cutoff < _D_NN_FCC, "cutoff must stay below d_nn (see module note)"

    if case == "flat_multi":
        L, seeds = _multi_grain_flat_case()
        tess = FlatTessellation(seeds, L, PER)
        atoms = _fill_all_grains(tess, len(seeds), L, PER)
    elif case == "power_multi":
        L, seeds = _multi_grain_flat_case(seed=2)
        weights = _shell_rng(50).normal(0.0, 3.0, size=len(seeds))
        tess = PowerTessellation(seeds, L, PER, weights=weights)
        atoms = _fill_all_grains(tess, len(seeds), L, PER)
    elif case == "weighted_additive_multi":
        L, seeds = _multi_grain_flat_case(seed=3)
        tess = WeightedTessellation(seeds, L, PER, sigma_w=1.0,
                                    rng=_shell_rng(51))
        atoms = _fill_all_grains(tess, len(seeds), L, PER)
    elif case == "flat_self_image":
        L, seeds = _self_image_flat_case()
        tess = FlatTessellation(seeds, L, PER)
        atoms = _fill_all_grains(tess, len(seeds), L, PER, seed_base=7)
    else:  # flat_free_axis
        periodic_free = [True, True, False]
        Lval = 4.0 * _A_FCC
        L = np.array([Lval, Lval, Lval])
        seeds = _shell_rng(4).uniform(2.0, Lval - 2.0, size=(5, 3))
        tess = FlatTessellation(seeds, L, periodic_free)
        atoms = _fill_all_grains(tess, 5, L, periodic_free, seed_base=200)

    periodic = [True, True, False] if case == "flat_free_axis" else PER
    shell = tess.gb_shell_lower_bound(atoms.pos, atoms.grain, cutoff, workers=1)
    assert shell is not None

    wrapped = _wrap_positions(atoms.pos, periodic, L)
    full_pairs = _pbc_pairs(wrapped, cutoff, periodic, L, workers=1)
    assert len(full_pairs) > 0, f"{case}: test is vacuous without any pairs"
    endpoints = np.unique(np.asarray(full_pairs, dtype=np.intp).ravel())
    assert shell[endpoints].all(), (
        f"{case}: shell mask dropped {int((~shell[endpoints]).sum())} of "
        f"{len(endpoints)} true pair endpoints"
    )
    print(f"[gb_shell] {case}: n={len(atoms)}, pairs={len(full_pairs)}, "
          f"shell fraction={float(shell.mean()):.4f}")


def test_overlap_shell_empty_mask_no_crash():
    """MUST-FIX regression: an EMPTY shell (mask all False, not merely a
    partial one) must not crash ``_pairs_with_shell``.

    A single grain in a fully free (non-periodic) box has no grain-
    boundary faces at all -- every face is a WALL_NEIGHBOR_ID box-wall
    face, excluded by ``margin()`` -- so every atom's margin is +inf and
    ``gb_shell_lower_bound``'s exact-margin fallback returns an ALL-FALSE
    mask.  Before the fix, ``shell_idx = np.flatnonzero(shell)`` was then
    empty, and ``_pairs(pos_act_w[shell_idx])`` on the resulting (0, 3)
    array reached ``_pbc_pairs``'s mixed/non-fully-periodic ghost-padding
    branch, where ``np.array([])`` on an empty Python ghost list produces
    shape (0,) instead of (0, 3) -- scipy's KDTree then raised
    ``ValueError: data must be of shape (n, m)``.  Reproduced end-to-end
    via ``remove_overlaps`` on a real FCC-Cu single-grain fill (matches
    the MUST-FIX #1 report: a 1-grain fully-free box at cutoff=1.0).

    The shell filter is only reached at ``workers > 1`` in production
    (see ``remove_overlaps``' docstring), so ``_force_shell=True`` is
    needed here to exercise the vulnerable code path at the default
    jobs=1 used in this test."""
    periodic = [False, False, False]
    L = np.array([4.0 * _A_FCC, 4.0 * _A_FCC, 4.0 * _A_FCC])
    seeds = np.array([[0.5 * L[0], 0.5 * L[1], 0.5 * L[2]]])
    tess = FlatTessellation(seeds, L, periodic)
    atoms = _fill_all_grains(tess, 1, L, periodic)

    shell = tess.gb_shell_lower_bound(atoms.pos, atoms.grain, 1.0, workers=1)
    assert shell is not None and not shell.any(), (
        "test setup: expected an ALL-FALSE shell mask (single grain, no "
        "GB faces, every margin +inf) -- if this fails, the fixture no "
        "longer reproduces the empty-shell edge case this test targets"
    )

    # Must not raise: a single grain's own FCC lattice has no pair closer
    # than cutoff=1.0 (well below d_nn), so nothing is removed.
    result, ledger = remove_overlaps(
        atoms, tess, cutoff=1.0, policy="delete_shallower",
        periodic=periodic, box_lengths=L, _force_shell=True)
    assert len(result) == len(atoms)
    assert ledger.total_deleted == 0


class _MarginOnlyTess:
    """Minimal duck-typed tess stub implementing ONLY the required
    ``margin()`` contract -- no ``gb_shell_lower_bound`` attribute at all
    (mirrors tests/test_debug_atomsqa.py's ``TestMidpointMergeChain
    ._FakeTess`` pattern, reused here for shell-path coverage that class
    doesn't exercise -- see below)."""

    def margin(self, pos, i):
        return np.zeros(len(pos))


def test_overlap_margin_only_stub_survives_jobs2_shell_lookup():
    """``_pairs_with_shell``'s ``getattr(tess, "gb_shell_lower_bound",
    None)`` lookup (atoms/overlap.py) -- not a direct attribute access --
    is what keeps a margin-only duck-typed tess stub working once the
    shell code path is actually reached. ``Tessellation``'s base
    implementation already returns None for ``gb_shell_lower_bound``, but
    a lightweight test double like ``_MarginOnlyTess`` above doesn't even
    subclass ``Tessellation`` -- it duck-types only ``margin()`` -- so the
    attribute is missing entirely, not merely None-valued. A direct
    ``tess.gb_shell_lower_bound(...)`` call would raise AttributeError.

    The shell path is only reached at ``workers > 1``, i.e. ``jobs > 1``
    (``workers <= 1`` returns early unless ``_force_shell`` -- see
    remove_overlaps' docstring); tests/test_debug_atomsqa.py's existing
    ``_FakeTess`` coverage all runs at the default jobs=1, so it never
    exercises this getattr lookup at all. This test forces jobs=2 (pass 0
    runs on unmoved positions for every policy, so it reaches the lookup
    unconditionally there) and checks two things: the call completes
    without raising, and -- since the shell filter is proven output-
    invariant (_pairs_with_shell's docstring) -- jobs=2 produces the exact
    same bytes as jobs=1 on the same input."""
    L = np.array([100.0, 100.0, 100.0])
    periodic = [False, False, False]
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    species = np.array(["Fe", "Fe", "Fe"])
    grain = np.array([0, 1, 2], dtype=np.int32)
    tess = _MarginOnlyTess()
    assert not hasattr(tess, "gb_shell_lower_bound"), (
        "test setup: _MarginOnlyTess must NOT define gb_shell_lower_bound "
        "-- a direct (non-getattr) lookup on it would raise AttributeError, "
        "which is exactly the regression this test guards against"
    )

    def _run(jobs):
        atoms = AtomBlock(pos=pos.copy(), species=species.copy(),
                          grain=grain.copy(), gb_margin=None)
        return remove_overlaps(
            atoms, tess, cutoff=0.7, policy="midpoint_merge",
            periodic=periodic, box_lengths=L, jobs=jobs)

    result1, ledger1 = _run(1)
    # Must not raise AttributeError: jobs=2 -> workers=2 -> pass 0's
    # _pairs_with_shell reaches the gb_shell_lower_bound getattr lookup.
    result2, ledger2 = _run(2)

    assert result2.pos.tobytes() == result1.pos.tobytes()
    assert result2.species.tobytes() == result1.species.tobytes()
    assert result2.grain.tobytes() == result1.grain.tobytes()
    assert ledger2.total_deleted == ledger1.total_deleted == 1


@pytest.mark.parametrize("cutoff_factor", [0.60, 0.85, 0.95])
@pytest.mark.parametrize(
    "backend", ["flat", "flat_free_axis", "power", "weighted"])
@pytest.mark.parametrize("no_numba", [False, True])
@pytest.mark.parametrize(
    "policy", ["delete_shallower", "keep_lower_id", "midpoint_merge"])
@pytest.mark.parametrize("case", ["normal", "self_image", "duplicate_bug"])
def test_overlap_shell_bitwise(policy, case, no_numba, backend, cutoff_factor,
                               monkeypatch):
    """Full remove_overlaps byte-equality with the shell filter enabled vs.
    forced off (the tessellation's own gb_shell_lower_bound monkeypatched
    to always return None), across all three policies, both
    GRAINSMITH_NO_NUMBA settings, every backend that provides a certified
    shell bound (flat, flat with a free axis, power, additive-weighted),
    and cutoffs from well below to just below d_nn -- including the
    self-image case and the duplicated-lattice-point case, which must
    raise FillError identically either way.

    The cutoff and backend axes matter for ``midpoint_merge``: at
    cutoff=0.60*d_nn a merge cascade rarely reaches a second generation,
    but at 0.85 and 0.95*d_nn (bracketing the production default) it
    does, which is where a moved survivor can newly pair with an unmoved
    same-grain lattice neighbour sharing its owner replica -- the case
    the shell bound's exactness proof does not cover (see
    ``atoms.overlap._pairs_with_shell``'s docstring).

    remove_overlaps only applies the shell filter at ``workers > 1``
    (jobs=1, used implicitly below, takes the unchanged full-N serial
    route -- see remove_overlaps' docstring); the "on" run therefore
    passes ``_force_shell=True`` so this test still exercises and proves
    the filtered route bit-for-bit at workers=1, not only at jobs > 1."""
    import grainsmith.tessellation._local_owns as local_owns_mod
    import grainsmith.tessellation.weighted as weighted_mod

    if local_owns_mod._HAVE_NUMBA:
        monkeypatch.setattr(local_owns_mod, "_USE_NUMBA", not no_numba)
    if weighted_mod._HAVE_NUMBA:
        monkeypatch.setattr(weighted_mod, "_USE_NUMBA", not no_numba)

    cutoff = cutoff_factor * _D_NN_FCC
    tess, atoms, periodic, L = _build_shell_case(backend, case, cutoff)
    tess_cls = type(tess)

    def _run(force_off):
        orig = tess_cls.gb_shell_lower_bound
        if force_off:
            tess_cls.gb_shell_lower_bound = lambda self, *a, **k: None
        try:
            return remove_overlaps(atoms, tess, cutoff, policy, periodic, L,
                                   _force_shell=not force_off)
        finally:
            tess_cls.gb_shell_lower_bound = orig

    raised_on = raised_off = None
    try:
        result_on, ledger_on = _run(False)
    except FillError as exc:
        raised_on = exc
    try:
        result_off, ledger_off = _run(True)
    except FillError as exc:
        raised_off = exc

    tag = f"{backend}/{case}/{policy}/cutoff={cutoff_factor}*d_nn"
    if raised_on is not None or raised_off is not None:
        assert raised_on is not None and raised_off is not None, (
            f"{tag}: shell filter changed whether FillError was raised "
            f"(on={raised_on!r}, off={raised_off!r})"
        )
        assert str(raised_on) == str(raised_off)
        assert case == "duplicate_bug", (
            f"{tag}: unexpected FillError {raised_on!r}"
        )
        return

    assert case != "duplicate_bug", f"{tag}: expected FillError, none raised"
    assert result_on.pos.tobytes() == result_off.pos.tobytes(), (
        f"{tag}: positions differ"
    )
    assert result_on.species.tobytes() == result_off.species.tobytes(), (
        f"{tag}: species differ"
    )
    assert result_on.grain.tobytes() == result_off.grain.tobytes(), (
        f"{tag}: grain assignment differs"
    )
    assert ledger_on.deletions_by_species == ledger_off.deletions_by_species
    assert ledger_on.deletions_by_pair == ledger_off.deletions_by_pair
    assert ledger_on.total_deleted == ledger_off.total_deleted


def test_overlap_shell_duplicate_bug_deep_interior_known_gap():
    """Documents, as a passing (not xfail) test, the ONE deliberately-
    accepted gap of the GB-shell pre-filter: the shell's exactness
    proof bounds distance to a DIFFERENT replica's boundary only, so a
    duplicate-lattice-point crystal-build bug that ALSO sits far from
    every grain boundary (large margin) can ESCAPE the shell-filtered
    search entirely -- remove_overlaps returns successfully instead of
    raising FillError, unlike the unfiltered baseline. A full-population
    same-grain-only supplementary search to close this gap costs as much
    as the unfiltered baseline itself (measured 0.5-0.6 s at 1.37M atoms;
    ``KDTree.query_pairs`` has no ``workers`` knob, so the cost does not
    shrink with --jobs either), which would erase this pre-filter's entire
    saving. ``qa.gate_g7_min_distance`` -- a SEPARATE, always full-N,
    never shell-filtered fresh query the pipeline runs unconditionally
    after overlap removal -- remains the independent catcher of exactly
    this case at the pipeline level (never filtered: see
    gate_g7_min_distance's own implementation in qa.py, untouched by this
    change). remove_overlaps only applies the shell filter at ``workers >
    1`` (jobs=1, used here, otherwise takes the unchanged full-N route --
    see remove_overlaps' docstring), so the ACTIVE run below passes
    ``_force_shell=True`` to reach this gap deliberately, the same way a
    real ``--jobs > 1`` run would reach it in production."""
    cutoff = 1.0
    L, seeds = _multi_grain_flat_case()
    tess = FlatTessellation(seeds, L, PER)
    atoms = _fill_all_grains(tess, len(seeds), L, PER)
    atoms = _inject_duplicate_bug(atoms, tess, L, cutoff, near_boundary=False)

    # Unfiltered (shell forced off): the pre-existing, unconditional
    # baseline behaviour -- still raises FillError.
    orig = FlatTessellation.gb_shell_lower_bound
    FlatTessellation.gb_shell_lower_bound = lambda self, *a, **k: None
    try:
        with pytest.raises(FillError):
            remove_overlaps(atoms, tess, cutoff, "delete_shallower", PER, L)
    finally:
        FlatTessellation.gb_shell_lower_bound = orig

    # Shell filter ACTIVE (forced on via _force_shell -- see docstring):
    # completes WITHOUT raising -- the documented gap, pinned here rather
    # than left as a silent behaviour change. The two INJECTED duplicate
    # atoms specifically must both survive untouched (the system's own
    # genuine GB pairs are still legitimately deleted either way, so a
    # plain atom-count comparison would not isolate the injected pair).
    injected = atoms.pos[-2:]
    result, _ = remove_overlaps(atoms, tess, cutoff, "delete_shallower",
                                PER, L, _force_shell=True)
    for p in injected:
        assert np.any(np.all(np.isclose(result.pos, p), axis=1)), (
            "expected the deep-interior duplicate pair to survive "
            "undetected under the shell filter (the documented, accepted "
            "gap) -- if this now fails, either the gap was closed (update "
            "this test to match) or something else changed"
        )
    # The independent, always full-N pipeline gate still catches it.
    from grainsmith.qa import gate_g7_min_distance
    g7 = gate_g7_min_distance(result, cutoff, PER, L)
    assert not g7.passed, (
        "gate_g7_min_distance must independently catch the surviving "
        "duplicate pair even though remove_overlaps' own check missed it"
    )


def _tiny_perturbed_tess():
    from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY
    from grainsmith.tessellation.perturbed import PerturbedDistanceTessellation
    from grainsmith.seeding import wigner_seitz_radius

    L = np.array([60.0, 60.0, 60.0])
    seeds = seed_grains(6, L, PER, _shell_rng(4))
    msd = wigner_seitz_radius(float(60.0**3), 6)
    a_max = PERTURBED_DISTANCE_SAFETY * msd / (2.0 * ETA_CLIP)
    tess = PerturbedDistanceTessellation(
        seeds, L, PER, amplitude=0.6 * a_max, min_seed_distance=msd,
        rng=_shell_rng(104), grid_size=16, spectrum="self_affine",
        hurst=0.7, l_min=8.0, l_max=30.0,
    )
    return tess


def _tiny_warp_tess():
    from grainsmith.seeding import wigner_seitz_radius
    from grainsmith.tessellation.warp import WarpTessellation

    L = np.array([50.0, 50.0, 50.0])
    n = 5
    seeds = seed_grains(n, L, PER, _shell_rng(1))
    base = FlatTessellation(seeds, L, PER)
    r_ws = wigner_seitz_radius(float(50.0**3), n)
    tess = WarpTessellation(
        base=base, box_lengths=L, periodic=PER, amplitude=1.0,
        correlation_length=15.0, min_seed_distance=r_ws,
        rng=_shell_rng(107), grid_size=16,
    )
    return tess


def _tiny_voxel_tess():
    from grainsmith.tessellation.voxel_import import VoxelTessellation

    labels = np.zeros((8, 8, 8), dtype=np.int32)
    labels[4:, :, :] = 1
    L = np.array([20.0, 20.0, 20.0])
    tess = VoxelTessellation(labels, L, PER)
    return tess


def _tiny_aniso_tess():
    L, seeds = _multi_grain_flat_case(seed=9)
    tess = AnisotropicTessellation(seeds, L, PER, rng=_shell_rng(52))
    return tess


def _tiny_single_tess():
    L = np.array([20.0, 20.0, 20.0])
    return SingleCrystalTessellation(L, PER)


@pytest.mark.parametrize("builder", [
    _tiny_aniso_tess, _tiny_warp_tess, _tiny_perturbed_tess,
    _tiny_voxel_tess, _tiny_single_tess,
])
def test_gb_shell_lower_bound_none_for_unsupported_backends(builder):
    """Anisotropic, Warp, Perturbed, Voxel(import), and Single all return
    None (never a wrong/unsound mask) -- see each backend's
    gb_shell_lower_bound docstring for the specific reason (Lipschitz
    constant != 1, a documented proxy margin, or an unproven precondition
    on the moved/free-axis case)."""
    tess = builder()
    rng = _shell_rng(999)
    pts = rng.uniform(0.0, 1.0, size=(50, 3)) * np.array([20.0, 20.0, 20.0])
    grain = np.zeros(50, dtype=np.int32)
    assert tess.gb_shell_lower_bound(pts, grain, 1.0, workers=1) is None
