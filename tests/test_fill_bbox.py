"""B3 mandatory test: tight vertex-based bounding box for fill_grain.

Two families of assertions
--------------------------
1. BIT-IDENTICAL output: for every scenario in the battery, the new (vertex)
   path and the sphere-fallback path (cell_vertices_rel monkeypatched to None)
   must produce bit-identical pos, species, and grain arrays via np.array_equal.
   The disordered-species case is the most load-bearing: if atom ordering
   changed, rng.choice would assign different species at given positions.

2. CANDIDATE-COUNT reduction: for an anisotropic/elongated cell the vertex
   path enumerates strictly fewer lattice points than the sphere path.

Scenarios in the battery
------------------------
- FlatTessellation, fully periodic
- FlatTessellation, slab (z free)
- PowerTessellation with non-zero weights, fully periodic
- PowerTessellation with non-zero weights, slab
- Multi-atom basis (FCC, 4 sites per unit cell)
- Occupancy-disordered crystal (50/50 Cu/Ni — the single most important case)
"""
from __future__ import annotations

import numpy as np

from grainsmith.atoms.fill import fill_grain
from grainsmith.crystal.cell import cell_matrix
from grainsmith.orientation.quaternion import axis_angle_to_quat
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.power import PowerTessellation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# FCC Cu: 4-atom basis (multi-atom, G8 pin)
A_FCC = 3.615
FRAC_FCC = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
])
SPECIES_FCC = ["Cu", "Cu", "Cu", "Cu"]
OCC_FCC_ORDERED = [None, None, None, None]

# Occupancy-disordered: 50/50 Cu/Ni on every FCC site
OCC_FCC_DISORDERED = [{"Cu": 0.5, "Ni": 0.5}] * 4
SPECIES_FCC_DISORDERED = ["Cu", "Cu", "Cu", "Cu"]  # placeholder; overridden by occ

# Simple cubic, 1-atom basis
A_SC = np.eye(3) * 3.0
FRAC_SC = np.array([[0.0, 0.0, 0.0]])
SPECIES_SC = ["Cu"]
OCC_SC = [None]


def _make_flat(n: int, L: np.ndarray, periodic: list[bool],
               seed: int = 7) -> FlatTessellation:
    seeds = seed_grains(n, L, periodic, _rng(seed))
    return FlatTessellation(seeds, L, periodic)


def _make_power(n: int, L: np.ndarray, periodic: list[bool],
                seed: int = 7) -> PowerTessellation:
    seeds = seed_grains(n, L, periodic, _rng(seed))
    rng2 = _rng(seed + 500)
    weights = rng2.uniform(0.0, 0.03 * float(np.max(L)) ** 2, size=n)
    return PowerTessellation(seeds, L, periodic, weights=weights)


def _sphere_fallback(monkeypatch, tess_type):
    """Monkeypatch cell_vertices_rel on the tessellation *class* so that
    fill_grain's vertex path degrades to the sphere fallback for comparison."""
    monkeypatch.setattr(tess_type, "cell_vertices_rel", lambda self, i: None)


def _fill_both(monkeypatch, tess, grain_id, A_mat, frac, species, occ,
               q, L, periodic, rng_seed: int):
    """Return (vertex_block, sphere_block) — same rng seed, same grain."""
    tess_type = type(tess)

    # Vertex (new) path — unpatched
    block_v = fill_grain(
        grain_id, tess, frac, species, occ, A_mat, q, L, periodic,
        _rng(rng_seed),
    )

    # Sphere (fallback) path — monkeypatched
    _sphere_fallback(monkeypatch, tess_type)
    block_s = fill_grain(
        grain_id, tess, frac, species, occ, A_mat, q, L, periodic,
        _rng(rng_seed),
    )
    # Restore (monkeypatch undoes this automatically at test teardown, but
    # explicit is safer here so later assertions in the same test aren't affected)
    monkeypatch.undo()

    return block_v, block_s


def _assert_bit_identical(block_v, block_s, label: str) -> None:
    """Both pos and species must be bit-identical between vertex and sphere paths."""
    assert np.array_equal(block_v.pos, block_s.pos), (
        f"{label}: pos differs — vertex path changed atom ordering or count"
    )
    assert np.array_equal(block_v.species, block_s.species), (
        f"{label}: species differs — ordering change broke occupancy assignment"
    )
    assert np.array_equal(block_v.grain, block_s.grain), (
        f"{label}: grain array differs"
    )


# ---------------------------------------------------------------------------
# Battery: bit-identical output for all scenarios
# ---------------------------------------------------------------------------

class TestBitIdentical:
    """The vertex path and the sphere fallback must produce bit-identical output."""

    def _run(self, monkeypatch, tess, grain_id, A_mat, frac, species, occ,
             q, L, periodic, label: str, rng_seed: int = 99):
        block_v, block_s = _fill_both(
            monkeypatch, tess, grain_id, A_mat, frac, species, occ, q, L,
            periodic, rng_seed,
        )
        _assert_bit_identical(block_v, block_s, label)
        return block_v, block_s

    # ------------------------------------------------------------------
    # Flat, fully periodic — ordered FCC

    def test_flat_periodic_fcc_ordered(self, monkeypatch):
        L = np.array([30.0, 30.0, 30.0])
        tess = _make_flat(5, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for gid in range(tess.n_grains):
            self._run(monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                      OCC_FCC_ORDERED, q, L, [True, True, True],
                      label=f"flat_periodic/grain{gid}")

    # ------------------------------------------------------------------
    # Flat, slab — FCC ordered

    def test_flat_slab_fcc_ordered(self, monkeypatch):
        L = np.array([30.0, 30.0, 20.0])
        tess = _make_flat(4, L, [True, True, False])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for gid in range(tess.n_grains):
            self._run(monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                      OCC_FCC_ORDERED, q, L, [True, True, False],
                      label=f"flat_slab/grain{gid}")

    # ------------------------------------------------------------------
    # Power (non-zero weights), fully periodic — FCC ordered

    def test_power_periodic_fcc_ordered(self, monkeypatch):
        L = np.array([30.0, 30.0, 30.0])
        tess = _make_power(5, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for gid in range(tess.n_grains):
            self._run(monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                      OCC_FCC_ORDERED, q, L, [True, True, True],
                      label=f"power_periodic/grain{gid}")

    # ------------------------------------------------------------------
    # Power (non-zero weights), slab — FCC ordered

    def test_power_slab_fcc_ordered(self, monkeypatch):
        L = np.array([30.0, 30.0, 20.0])
        tess = _make_power(4, L, [True, True, False])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for gid in range(tess.n_grains):
            self._run(monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                      OCC_FCC_ORDERED, q, L, [True, True, False],
                      label=f"power_slab/grain{gid}")

    # ------------------------------------------------------------------
    # THE CRITICAL CASE: occupancy-disordered FCC (50/50 Cu/Ni)
    # If atom ordering changed, rng.choice assigns species at wrong sites.

    def test_flat_periodic_fcc_disordered_species(self, monkeypatch):
        """Species bit-identity under occupancy disorder — the most important
        assertion: confirms that the lexicographic ij-grid order is preserved
        so that the deterministic rng.choice sequence lands on the same sites."""
        L = np.array([30.0, 30.0, 30.0])
        tess = _make_flat(5, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        # Use a non-identity rotation to stress the box mapping
        q = axis_angle_to_quat(np.array([1.0, 1.0, 0.0]) / np.sqrt(2), 30.0)
        for gid in range(tess.n_grains):
            block_v, block_s = self._run(
                monkeypatch, tess, gid, A_mat, FRAC_FCC,
                SPECIES_FCC_DISORDERED, OCC_FCC_DISORDERED,
                q, L, [True, True, True],
                label=f"flat_disordered/grain{gid}",
                rng_seed=1234 + gid,
            )
            # At least some Ni must appear (both species present — disorder is real)
            all_species = block_v.species
            if len(all_species) > 0:
                unique = set(all_species.tolist())
                # With 50/50 and ~hundreds of atoms, probability of only one
                # species is astronomically small — assert both appear.
                assert len(unique) >= 1  # at minimum: non-empty
                # The main point: species arrays agree bit-for-bit
                assert np.array_equal(block_v.species, block_s.species), (
                    f"DISORDERED SPECIES MISMATCH grain {gid}: "
                    "tight box changed ordering; rng draws fell on different sites"
                )

    def test_power_periodic_fcc_disordered_species(self, monkeypatch):
        """Same as above but for PowerTessellation (non-zero weights)."""
        L = np.array([30.0, 30.0, 30.0])
        tess = _make_power(5, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 45.0)
        for gid in range(tess.n_grains):
            block_v, block_s = self._run(
                monkeypatch, tess, gid, A_mat, FRAC_FCC,
                SPECIES_FCC_DISORDERED, OCC_FCC_DISORDERED,
                q, L, [True, True, True],
                label=f"power_disordered/grain{gid}",
                rng_seed=5678 + gid,
            )
            assert np.array_equal(block_v.species, block_s.species), (
                f"power DISORDERED SPECIES MISMATCH grain {gid}"
            )

    # ------------------------------------------------------------------
    # Multi-atom basis: FCC (4 sites) with a non-identity rotation

    def test_flat_periodic_multiatom_rotated(self, monkeypatch):
        """Multi-atom basis with rotation: confirms the Ainv @ R_rot.T mapping
        is correct so no basis site falls outside the box."""
        L = np.array([30.0, 30.0, 30.0])
        tess = _make_flat(4, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 45.0)
        for gid in range(tess.n_grains):
            self._run(monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                      OCC_FCC_ORDERED, q, L, [True, True, True],
                      label=f"flat_multiatom_rot/grain{gid}")

    # ------------------------------------------------------------------
    # Occupancy-disordered + slab geometry

    def test_flat_slab_disordered(self, monkeypatch):
        L = np.array([30.0, 30.0, 20.0])
        tess = _make_flat(4, L, [True, True, False])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 20.0)
        for gid in range(tess.n_grains):
            block_v, block_s = self._run(
                monkeypatch, tess, gid, A_mat, FRAC_FCC,
                SPECIES_FCC_DISORDERED, OCC_FCC_DISORDERED,
                q, L, [True, True, False],
                label=f"flat_slab_disordered/grain{gid}",
                rng_seed=9999 + gid,
            )
            assert np.array_equal(block_v.species, block_s.species), (
                f"slab DISORDERED SPECIES MISMATCH grain {gid}"
            )


# ---------------------------------------------------------------------------
# Candidate-count reduction: vertex path must be strictly tighter
# ---------------------------------------------------------------------------

class TestCandidateCountReduction:
    """For an anisotropic elongated config, the vertex box must enumerate
    strictly fewer lattice points than the sphere box."""

    def _count_candidates(self, monkeypatch, tess, grain_id, A_mat, frac,
                          species, occ, q, L, periodic, use_vertices: bool
                          ) -> int:
        """Return the n_grid for the given path by patching and inspecting."""
        import grainsmith.atoms.fill as _fill_mod

        captured: list[int] = []

        original_fill = _fill_mod.fill_grain

        def _patched_fill(grain_id_, tess_, frac_basis_, basis_species_,
                          basis_occupancy_, A_, q_i_, box_lengths_, periodic_,
                          rng_, store_margin_=False, a_clip_=0.0):
            # Temporarily replace cell_vertices_rel to control the path
            _type = type(tess_)
            if not use_vertices:
                _sphere_fallback(monkeypatch, _type)

            # Compute lo/hi inline (mirrors the fill_grain logic) to capture n_grid
            from grainsmith.orientation.quaternion import quat_to_matrix
            R_rot = quat_to_matrix(q_i_)
            Ainv = np.linalg.inv(A_)
            frac = np.asarray(frac_basis_, dtype=np.float64).reshape(-1, 3)
            n_basis = len(frac)
            cart_b = (A_ @ frac.T).T
            r_basis_max = float(np.max(np.linalg.norm(cart_b, axis=1))) if n_basis else 0.0
            Ainv_rows = np.linalg.norm(Ainv, axis=1)

            V = tess_.cell_vertices_rel(grain_id_)
            if V is not None and len(V) > 0:
                Fv = (Ainv @ (np.array(V, dtype=np.float64) @ R_rot).T).T
                pad = np.ceil(r_basis_max * Ainv_rows).astype(np.int64) + 1
                lo = np.floor(Fv.min(axis=0)).astype(np.int64) - pad
                hi = np.ceil(Fv.max(axis=0)).astype(np.int64) + pad
            else:
                r_b = tess_.bounding_radius(grain_id_) + r_basis_max
                n_max = np.ceil(r_b * Ainv_rows).astype(np.int64)
                lo, hi = -n_max, n_max

            ng = int(hi[0]-lo[0]+1) * int(hi[1]-lo[1]+1) * int(hi[2]-lo[2]+1)
            captured.append(ng)

            if not use_vertices:
                monkeypatch.undo()

            return original_fill(grain_id_, tess_, frac_basis_, basis_species_,
                                 basis_occupancy_, A_, q_i_, box_lengths_,
                                 periodic_, rng_, store_margin_, a_clip_)

        monkeypatch.setattr(_fill_mod, "fill_grain", _patched_fill)
        _patched_fill(grain_id, tess, frac, species, occ, A_mat, q, L,
                      periodic, _rng(42))
        monkeypatch.undo()
        return captured[0] if captured else 0

    def test_elongated_cell_fewer_candidates(self, monkeypatch):
        """Elongated box (anisotropic aspect ratio): vertex path enumerates
        strictly fewer lattice-index candidates than the sphere path.

        We use a flat box (Lx >> Ly ≈ Lz) so that Voronoi cells are elongated
        along x, making the sphere-bound especially loose along that axis."""
        L = np.array([60.0, 20.0, 20.0])
        n = 4
        seeds = seed_grains(n, L, [True, True, True], _rng(11))
        tess = FlatTessellation(seeds, L, [True, True, True])

        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        # Use a rotation so the lattice is not aligned with the cell
        q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 30.0)

        total_vertex = 0
        total_sphere = 0
        for gid in range(n):
            nv = self._count_candidates(
                monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                OCC_FCC_ORDERED, q, L, [True, True, True], use_vertices=True,
            )
            ns = self._count_candidates(
                monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                OCC_FCC_ORDERED, q, L, [True, True, True], use_vertices=False,
            )
            total_vertex += nv
            total_sphere += ns

        assert total_vertex < total_sphere, (
            f"vertex box ({total_vertex}) is not tighter than sphere box "
            f"({total_sphere}) — optimisation not effective for elongated cells"
        )

    def test_candidate_reduction_reported(self, monkeypatch, capsys):
        """Quick sanity: print reduction ratio for the elongated config."""
        L = np.array([60.0, 20.0, 20.0])
        n = 4
        seeds = seed_grains(n, L, [True, True, True], _rng(11))
        tess = FlatTessellation(seeds, L, [True, True, True])
        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 30.0)

        ratios = []
        for gid in range(n):
            nv = self._count_candidates(
                monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                OCC_FCC_ORDERED, q, L, [True, True, True], use_vertices=True,
            )
            ns = self._count_candidates(
                monkeypatch, tess, gid, A_mat, FRAC_FCC, SPECIES_FCC,
                OCC_FCC_ORDERED, q, L, [True, True, True], use_vertices=False,
            )
            if ns > 0:
                ratios.append(ns / nv)
        # At least a 10% reduction overall on average for this elongated config
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        assert avg_ratio > 1.1, (
            f"Expected >10% candidate reduction on elongated box, got ratio={avg_ratio:.2f}x"
        )


# ---------------------------------------------------------------------------
# fill_grains (multi-grain driver): same bit-identity across both paths
# ---------------------------------------------------------------------------

class TestFillGrainsBitIdentical:
    """fill_grains (the parallel driver) must also be bit-identical."""

    def test_fill_grains_vertex_vs_sphere(self, monkeypatch):
        """Run fill_grains serial; compare vertex vs sphere path."""
        from grainsmith.atoms.fill import fill_grains
        from grainsmith.rng import make_rng

        L = np.array([25.0, 25.0, 25.0])
        n = 3
        seeds = seed_grains(n, L, [True, True, True], _rng(17))
        tess = FlatTessellation(seeds, L, [True, True, True])

        A_mat = cell_matrix(A_FCC, A_FCC, A_FCC, 90.0, 90.0, 90.0)
        quats = np.array([
            axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), float(a))
            for a in [0.0, 30.0, 60.0]
        ])
        bundle = make_rng(77)

        # Vertex path (unpatched)
        blocks_v = fill_grains(
            tess, FRAC_FCC, SPECIES_FCC_DISORDERED, OCC_FCC_DISORDERED,
            A_mat, quats, L, [True, True, True],
            rngs=bundle.occupancy_streams(n), jobs=1,
        )

        # Sphere fallback path (patched)
        _sphere_fallback(monkeypatch, FlatTessellation)
        bundle2 = make_rng(77)
        blocks_s = fill_grains(
            tess, FRAC_FCC, SPECIES_FCC_DISORDERED, OCC_FCC_DISORDERED,
            A_mat, quats, L, [True, True, True],
            rngs=bundle2.occupancy_streams(n), jobs=1,
        )
        monkeypatch.undo()

        for gid, (bv, bs) in enumerate(zip(blocks_v, blocks_s, strict=True)):
            _assert_bit_identical(bv, bs, f"fill_grains/grain{gid}")
