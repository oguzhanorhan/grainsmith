"""Scientific debug wave — atoms/qa/pipeline track.

Targeted regression + differential coverage for the generalized
triclinic overlap path, the fill-stage tiling invariant, deterministic
phase assignment, and one message-hygiene fix in the doping QA gate.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from grainsmith.atoms.doping import _min_distance_ok, _solve_probabilities
from grainsmith.atoms.fill import AtomBlock, _compute_d_nn, fill_grain, fill_grains
from grainsmith.atoms.overlap import (
    _pbc_pairs,
    _pbc_pairs_general,
    _wrap_positions,
    _wrap_positions_general,
    remove_overlaps,
    resolve_cutoff,
)
from grainsmith.errors import ConfigError
from grainsmith.orientation.quaternion import quat_to_matrix
from grainsmith.orientation.samplers import random_uniform
from grainsmith.phases import assign_phases
from grainsmith.qa import gate_g18_doping_geometry
from grainsmith.tessellation.flat import FlatTessellation


def _brute_force_pairs(pos, cutoff, periodic, L, shell=1):
    """Reference PBC neighbor pairs via explicit integer-shift enumeration
    (independent of any ghost-padding shortcut in the module under test)."""
    n = len(pos)
    shift_ranges = [range(-shell, shell + 1) if p else [0] for p in periodic]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            best = np.inf
            for s in product(*shift_ranges):
                shift = np.array(s, dtype=float) * L
                d = np.linalg.norm(pos[i] - (pos[j] + shift))
                best = min(best, d)
            if best < cutoff:
                pairs.add((i, j))
    return pairs


class TestOrthogonalVsGeneralOverlapPath:
    """The general (27-image) triclinic path must reproduce the fast
    orthogonal KDTree(boxsize=L) path EXACTLY on an orthogonal cell —
    the equivalence the triclinic generalization must preserve."""

    @pytest.mark.parametrize("periodic", [
        [True, True, True], [True, True, False],
        [True, False, False], [False, False, False],
    ])
    def test_pbc_pairs_matches_general_path(self, periodic):
        rng = np.random.default_rng(42)
        L = np.array([10.0, 10.0, 10.0])
        H = np.diag(L)
        Hinv = np.linalg.inv(H)
        cutoff = 1.0
        mismatches = 0
        for trial in range(200):
            pos = rng.random((8, 3)) * L
            if trial % 2 == 0:
                # bias toward box faces to stress the ghost-padding logic
                pos = np.where(rng.random(pos.shape) < 0.5, pos * 0.05, pos)
                pos = np.where(rng.random(pos.shape) < 0.5, L - pos * 0.05, pos)
            wrapped = _wrap_positions(pos.copy(), periodic, L)
            wrapped_g = _wrap_positions_general(pos.copy(), periodic, H, Hinv)
            p1 = set(_pbc_pairs(wrapped, cutoff, periodic, L))
            p2 = set(_pbc_pairs_general(wrapped_g, cutoff, periodic, H, Hinv))
            if p1 != p2:
                mismatches += 1
        assert mismatches == 0

    @pytest.mark.parametrize("periodic", [
        [True, True, False], [True, False, True], [False, True, True],
    ])
    def test_mixed_periodicity_ghost_pad_is_exact(self, periodic):
        """Per-axis (non-cumulative) ghost padding is mathematically
        complete for an ORTHOGONAL box: any pair whose true minimum-image
        distance requires a simultaneous two-axis shift necessarily has
        BOTH atoms within `cutoff` of a boundary on the shifted axes (the
        separable minimum-image argument), so cross-combining each atom's
        own single-axis ghosts already covers the diagonal case. Verified
        here against an independent brute-force multi-shell reference —
        this is the guarantee the doping.py 'KNOWN v1 LIMIT' comment
        narrows to mixed-periodicity corner cases that do NOT arise for a
        single close pair (self-conflict pruning among >2 candidates).
        """
        rng = np.random.default_rng(2024)
        L = np.array([7.0, 13.0, 20.0])
        mismatches = 0
        for _ in range(300):
            n_pts = rng.integers(4, 25)
            pos = rng.random((n_pts, 3)) * L
            near = rng.random(n_pts) < 0.7
            for ax in range(3):
                if periodic[ax]:
                    lo = rng.random(n_pts) < 0.5
                    sel_lo, sel_hi = near & lo, near & ~lo
                    pos[sel_lo, ax] = rng.uniform(0, 0.4, size=np.sum(sel_lo))
                    pos[sel_hi, ax] = rng.uniform(
                        L[ax] - 0.4, L[ax], size=np.sum(sel_hi))
            cutoff = rng.uniform(0.15, 0.8)
            wrapped = _wrap_positions(pos.copy(), periodic, L)
            fast_pairs = set(_pbc_pairs(wrapped, cutoff, periodic, L))
            true_pairs = _brute_force_pairs(pos, cutoff, periodic, L, shell=2)
            if fast_pairs != true_pairs:
                mismatches += 1
        assert mismatches == 0

    def test_general_path_on_skewed_triclinic_cell(self):
        """The 27-image general path must match a wide brute-force shell
        on a strongly skewed (non-Niggli-reduced) triclinic cell."""
        a = np.array([10.0, 0.0, 0.0])
        b = np.array([8.0, 3.0, 0.0])   # highly skewed: tilt >> length
        c = np.array([0.0, 0.0, 10.0])
        H = np.column_stack([a, b, c])
        Hinv = np.linalg.inv(H)
        periodic = [True, True, True]
        cutoff = 1.0

        def brute_min_dist(p0, p1, H, periodic, shell=3):
            rngs = [range(-shell, shell + 1) if p else [0] for p in periodic]
            best = np.inf
            for s in product(*rngs):
                shift = H @ np.array(s, dtype=float)
                best = min(best, float(np.linalg.norm((p0 + shift) - p1)))
            return best

        rng = np.random.default_rng(7)
        mismatches = 0
        for _ in range(300):
            frac = rng.random((2, 3))
            cart = (H @ frac.T).T
            wrapped = _wrap_positions_general(cart.copy(), periodic, H, Hinv)
            pairs = _pbc_pairs_general(wrapped, cutoff, periodic, H, Hinv)
            found = len(pairs) > 0
            true_found = brute_min_dist(cart[0], cart[1], H, periodic) < cutoff
            if found != true_found:
                mismatches += 1
        assert mismatches == 0


class TestComputeDnn:
    """d_nn (fill._compute_d_nn) must match a much wider brute-force
    neighbour shell, including for strongly skewed triclinic cells where
    a fixed +-1 shell over the RAW (unreduced) cell would miss the true
    nearest neighbour."""

    def test_basis_must_share_the_reduced_search_cell(self):
        from grainsmith.atoms.fill import _compute_d_nn

        matrix = np.array([[100.0, 0.0, 21.0],
                           [0.0, 100.0, 31.0],
                           [0.0, 0.0, 1.0]])
        basis = np.array([[0.0, 0.0, 0.0], [0.735, 0.085, 0.5]])
        assert _compute_d_nn(basis, matrix) == pytest.approx(3.5, abs=1e-12)
        translated = basis + np.array([[0, 0, 0], [2, -1, 7]])
        assert _compute_d_nn(translated, matrix) == pytest.approx(3.5, abs=1e-11)

    def test_matches_wide_brute_force_shell_random_skew(self):
        rng = np.random.default_rng(123)

        def brute_dnn(frac_basis, A, shell=6):
            cart = (A @ frac_basis.T).T
            n = len(cart)
            d_min = np.inf
            for idx in np.ndindex(2 * shell + 1, 2 * shell + 1, 2 * shell + 1):
                sv = np.array(idx, dtype=np.float64) - shell
                img = cart + (A @ sv)
                for k in range(n):
                    dists = np.linalg.norm(img - cart[k], axis=1)
                    if np.allclose(sv, 0):
                        dists[k] = np.inf
                    d_min = min(d_min, float(np.min(dists)))
            return d_min

        checked = 0
        for _ in range(20):
            A = rng.uniform(-1, 1, (3, 3)) * rng.uniform(2, 15)
            if abs(np.linalg.det(A)) < 1.0:
                continue
            n_basis = int(rng.integers(1, 4))
            frac_basis = rng.random((n_basis, 3))
            d_fn = _compute_d_nn(frac_basis, A)
            d_brute = brute_dnn(frac_basis, A, shell=6)
            assert np.isclose(d_fn, d_brute, rtol=1e-6, atol=1e-8), (
                f"d_nn={d_fn} vs brute={d_brute} for A=\n{A}")
            checked += 1
        assert checked >= 10  # most random draws should pass the det filter


class TestFillWindowEquivalence:
    """The tight vertex-based enumeration window (cell_vertices_rel) and
    the sphere-fallback window must select the IDENTICAL atom set — the
    vertex path is a tightening, not a different membership rule."""

    def test_vertex_and_sphere_windows_agree(self):
        rng = np.random.default_rng(321)
        periodic = [True, True, True]

        class NoVertexTess:
            """Wraps a Tessellation but hides cell_vertices_rel to force
            the sphere-fallback path in fill_grain."""
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "cell_vertices_rel":
                    raise AttributeError
                return getattr(self._inner, name)

        for trial in range(5):
            L = np.array([15.0, 15.0, 15.0])
            seeds = rng.random((5, 3)) * L
            tess = FlatTessellation(seeds, L, periodic)
            nv_tess = NoVertexTess(tess)
            A = np.array([[3.0, 0.5, 0.0], [0.0, 2.8, 0.3], [0.0, 0.0, 3.2]])
            n_basis = int(rng.integers(1, 3))
            frac_basis = rng.random((n_basis, 3))
            basis_species = ["Fe"] * n_basis
            basis_occ = [None] * n_basis
            quats = random_uniform(tess.n_grains, rng)
            for g in range(tess.n_grains):
                b1 = fill_grain(g, tess, frac_basis, basis_species, basis_occ,
                                A, quats[g], L, periodic,
                                np.random.default_rng(5))
                b2 = fill_grain(g, nv_tess, frac_basis, basis_species,
                                basis_occ, A, quats[g], L, periodic,
                                np.random.default_rng(5))
                assert len(b1) == len(b2), (
                    f"trial {trial} grain {g}: vertex={len(b1)} "
                    f"sphere={len(b2)}")
                if len(b1) > 0:
                    idx1 = np.lexsort(b1.pos.T)
                    idx2 = np.lexsort(b2.pos.T)
                    assert np.allclose(b1.pos[idx1], b2.pos[idx2], atol=1e-9)


class TestFillTilingExactness:
    """fill_grains must reproduce the exact torus-tiling count: every
    lattice point owned by exactly one grain, no duplicates, matching an
    independent brute-force large-window enumeration through owns()."""

    def test_total_atom_count_matches_brute_force_enumeration(self):
        rng = np.random.default_rng(4242)
        L = np.array([12.0, 12.0, 12.0])
        periodic = [True, True, True]
        seeds = rng.random((4, 3)) * L
        tess = FlatTessellation(seeds, L, periodic)
        A = np.eye(3) * 2.5
        frac_basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        basis_species = ["Fe", "Fe"]
        basis_occ = [None, None]
        quats = random_uniform(tess.n_grains, rng)

        rngs = [np.random.default_rng(i) for i in range(tess.n_grains)]
        blocks = fill_grains(tess, frac_basis, basis_species, basis_occ, A,
                             quats, L, periodic, rngs)
        n_fill = sum(len(b) for b in blocks)

        # Independent brute-force reference: enumerate well beyond the
        # expected radius and rely on owns() to partition (no dedup step).
        Ainv = np.linalg.inv(A)
        all_positions = []
        for g in range(tess.n_grains):
            R_rot = quat_to_matrix(quats[g])
            c_i = tess.seeds[g]
            r_b = tess.bounding_radius(g)
            Ainv_rows = np.linalg.norm(Ainv, axis=1)
            n_max = np.ceil((r_b + 5) * Ainv_rows).astype(int) + 5
            ranges = [np.arange(-n_max[k], n_max[k] + 1) for k in range(3)]
            iu, iv, iw = np.meshgrid(*ranges, indexing="ij")
            T = np.stack([iu.ravel(), iv.ravel(), iw.ravel()],
                        axis=1).astype(np.float64)
            F = T[:, None, :] + frac_basis[None, :, :]
            X_cryst = (A @ F.reshape(-1, 3).T).T
            X_lab = (R_rot @ X_cryst.T).T + c_i
            keep = tess.owns(X_lab, g)
            all_positions.append(X_lab[keep])
        brute_pos = np.concatenate(all_positions, axis=0)

        assert n_fill == len(brute_pos)
        uniq = np.unique(np.round(brute_pos, 6), axis=0)
        assert len(uniq) == len(brute_pos), "owns() must partition with no duplicates"

    def test_jobs_invariance_bit_identical(self):
        """fill_grains output must be identical (position-for-position)
        for jobs=1 vs jobs=3 — each grain draws from its own stream."""
        rng = np.random.default_rng(99)
        L = np.array([12.0, 12.0, 12.0])
        periodic = [True, True, True]
        seeds = rng.random((5, 3)) * L
        tess = FlatTessellation(seeds, L, periodic)
        A = np.eye(3) * 2.2
        frac_basis = np.array([[0.0, 0.0, 0.0]])
        basis_species = ["Fe"]
        basis_occ = [None]
        quats = random_uniform(tess.n_grains, rng)

        rngs1 = [np.random.default_rng(i) for i in range(tess.n_grains)]
        rngs3 = [np.random.default_rng(i) for i in range(tess.n_grains)]
        b1 = fill_grains(tess, frac_basis, basis_species, basis_occ, A,
                         quats, L, periodic, rngs1, jobs=1)
        b3 = fill_grains(tess, frac_basis, basis_species, basis_occ, A,
                         quats, L, periodic, rngs3, jobs=3)
        assert len(b1) == len(b3)
        for g in range(tess.n_grains):
            assert np.array_equal(b1[g].pos, b3[g].pos)
            assert np.array_equal(b1[g].species, b3[g].species)


class TestAssignPhasesGuarantee:
    """assign_phases must give every phase at least one grain even under
    extreme fraction skew (resolve Rule 23 only guarantees n_grains >=
    n_phases, not that the greedy step alone would reach every phase)."""

    def test_every_phase_gets_at_least_one_grain(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            n_grains = int(rng.integers(3, 8))
            n_phases = int(rng.integers(2, n_grains + 1))
            volumes = rng.uniform(1, 100, n_grains)
            raw = rng.uniform(0.01, 1, n_phases)
            fractions = raw / raw.sum()
            phase_of = assign_phases(volumes, fractions)
            assert set(phase_of.tolist()) == set(range(n_phases))

    def test_too_few_grains_raises(self):
        with pytest.raises(ValueError, match="cannot cover"):
            assign_phases(np.array([1.0, 2.0]), np.array([0.3, 0.3, 0.4]))


class TestSolveProbabilitiesEdgeCases:
    """_solve_probabilities (doping shell/bulk mass balance) edge cases:
    zero target, zero candidate population, and the infeasibility hint
    selection across bad_bulk / bad_shell / both branches."""

    def test_zero_target_zero_population_no_raise(self):
        assert _solve_probabilities(0, 0, 0, 1.0, "t") == (0.0, 0.0)

    def test_positive_target_zero_population_raises(self):
        with pytest.raises(ConfigError, match="no candidate sites"):
            _solve_probabilities(5, 0, 0, 1.0, "t")

    def test_infeasible_bulk_only(self):
        with pytest.raises(ConfigError, match="p_bulk"):
            _solve_probabilities(100, 0, 10, 1.0, "t")

    def test_infeasible_shell_only(self):
        with pytest.raises(ConfigError, match="p_shell"):
            _solve_probabilities(100, 10, 0, 1.0, "t")

    def test_infeasible_both_reports_lower_concentration(self):
        with pytest.raises(ConfigError, match="lower the concentration"):
            _solve_probabilities(1000, 10, 10, 2.0, "t")


class TestMinDistanceOkAndGhostPad:
    """doping._min_distance_ok (used for interstitial candidate rejection
    against existing atoms) must match a brute-force minimum-image
    distance check under mixed periodicity."""

    def test_matches_brute_force_min_image(self):
        rng = np.random.default_rng(2)
        L = np.array([10.0, 10.0, 10.0])
        periodic = [True, True, False]
        mismatches = 0
        for _ in range(300):
            n_cand = int(rng.integers(1, 10))
            n_exist = int(rng.integers(1, 10))
            cand = rng.random((n_cand, 3)) * L
            exist = rng.random((n_exist, 3)) * L
            r = rng.uniform(0.2, 1.5)
            ok = _min_distance_ok(cand, exist, r, periodic, L)
            for i in range(n_cand):
                best = np.inf
                for j in range(n_exist):
                    dr = cand[i] - exist[j]
                    for ax in range(3):
                        if periodic[ax]:
                            dr[ax] -= round(dr[ax] / L[ax]) * L[ax]
                    best = min(best, float(np.linalg.norm(dr)))
                if ok[i] != (best >= r):
                    mismatches += 1
        assert mismatches == 0


class TestMidpointMergeChain:
    """midpoint_merge on a 3-atom mutual-overlap chain: the pass-wise
    memoization and moved/grain reassignment must converge to a
    physically sane result (survivors placed at min-image midpoints,
    total_deleted counted correctly)."""

    class _FakeTess:
        def __init__(self, seeds):
            self.seeds = seeds

        def margin(self, pos, i):
            return np.zeros(len(pos))

    def test_three_atom_chain_converges(self):
        L = np.array([100.0, 100.0, 100.0])
        periodic = [False, False, False]
        pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        species = np.array(["Fe", "Fe", "Fe"])
        grain = np.array([0, 1, 2], dtype=np.int32)
        atoms = AtomBlock(pos=pos, species=species, grain=grain, gb_margin=None)
        tess = self._FakeTess(seeds=np.zeros((3, 3)))

        result, ledger = remove_overlaps(
            atoms, tess, cutoff=0.7, policy="midpoint_merge",
            periodic=periodic, box_lengths=L)
        assert len(result) == 2
        assert ledger.total_deleted == 1
        # remaining atoms must be pairwise farther than cutoff
        dists = np.linalg.norm(result.pos[0] - result.pos[1])
        assert dists >= 0.7


class TestResolveCutoffEdgeCases:
    def test_expression_without_d_nn_raises(self):
        with pytest.raises(ConfigError, match="finite positive"):
            resolve_cutoff("0.85*d_nn", d_nn=None)

    def test_negative_absolute_raises(self):
        with pytest.raises(ConfigError, match="positive and finite"):
            resolve_cutoff(-1.0)

    def test_garbage_string_raises(self):
        with pytest.raises(ConfigError, match="Invalid overlap cutoff"):
            resolve_cutoff("abc")

    def test_valid_expression_and_absolute(self):
        assert resolve_cutoff("1.5*d_nn", d_nn=2.0) == pytest.approx(3.0)
        assert resolve_cutoff(2.5) == pytest.approx(2.5)


class TestGateG18MessageHygiene:
    """Regression test for the message-hygiene fix: G18 must not append
    imperative remediation advice ('Lower min_distance...') when the gate
    PASSES (0 violations) — every other gate in qa.py keeps PASS messages
    purely descriptive."""

    def test_pass_message_has_no_remediation_advice(self):
        r = gate_g18_doping_geometry(0, "C: 0 violation(s)")
        assert r.passed is True
        assert "Lower min_distance" not in r.message
        assert "0 dopant pair(s)" in r.message

    def test_fail_message_keeps_remediation_advice(self):
        r = gate_g18_doping_geometry(3, "C: 3 violation(s)")
        assert r.passed is False
        assert "Lower min_distance" in r.message
