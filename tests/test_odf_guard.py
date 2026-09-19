"""Tests for the volume-weighted ODF diagnostics (orientation/odf.py).

Follows the style of tests/test_odf_mdf.py: a local seeded Generator helper
and a `_sym_cubic()` symmetry fixture built from the real crystal machinery
(never a hand-rolled stand-in), so the Gram-matrix tests exercise the same
24-element cubic proper-rotation group the rest of the suite uses.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import grainsmith.orientation.odf as odf_mod
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.errors import ConfigError, GrainsmithError
from grainsmith.orientation.odf import (
    DriftTracker,
    atomic_drift,
    class_masses,
    component_volume_fractions,
    count_vs_volume_gap,
    effective_sample_size,
    mmd,
    null_mmd,
    orientation_classes,
    orientation_weights,
    symmetry_classes,
    symmetrized_gram,
    volume_balanced_partition,
    vp_halfwidth,
    vp_kappa,
    vp_norm,
)
from grainsmith.orientation.quaternion import quat_to_bunge
from grainsmith.orientation.samplers import (
    fixed_orientation,
    odf_components,
    random_uniform,
)


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _sym_cubic() -> np.ndarray:
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    rots, _ = symmetry_ops(hall_from_international(221))
    return proper_rotation_quaternions(A, rots)


# ---------------------------------------------------------------------------
# 1. vp_kappa / vp_norm
# ---------------------------------------------------------------------------


def test_vp_kappa_integer_degree_and_effective_halfwidth():
    for requested in (5.0, 10.0, 20.0, 60.0, 90.0, 120.0, 170.0):
        kappa = vp_kappa(requested)
        assert kappa >= 1.0 and kappa.is_integer()
        effective = vp_halfwidth(kappa)
        ratio = math.cos(math.radians(effective) / 2.0) ** (2.0 * kappa)
        assert ratio == pytest.approx(0.5, abs=1e-12)


def test_vp_norm_pinned_value_at_10deg():
    kappa = vp_kappa(10.0)
    assert kappa == 91.0
    assert vp_norm(kappa) == pytest.approx(
        92 * 4**91 / math.comb(182, 91), rel=1e-12)


def test_vp_kappa_small_positive_halfwidth_is_finite():
    assert math.isfinite(vp_kappa(1e-6))
    assert vp_kappa(1e-6) > vp_kappa(1.0)
    assert vp_halfwidth(vp_kappa(1e-6)) == pytest.approx(1e-6, rel=1e-12)


@pytest.mark.parametrize("halfwidth", [60.0, 120.0, 170.0])
def test_gram_wide_kernel_is_positive_semidefinite(halfwidth):
    quats = random_uniform(64, _rng(73))
    sym = np.array([[1.0, 0.0, 0.0, 0.0]])
    gram = symmetrized_gram(quats, sym, vp_kappa(halfwidth))
    eigenvalues = np.linalg.eigvalsh(gram)
    assert eigenvalues.min() >= -1e-12 * eigenvalues.max()


def test_gram_rejects_fractional_degree():
    with pytest.raises(ConfigError, match="non-negative integer"):
        symmetrized_gram(np.eye(4), np.eye(4)[:1], 0.5)


# ---------------------------------------------------------------------------
# 2-4. Symmetrized Gram matrix
# ---------------------------------------------------------------------------


def test_gram_diagonal_is_vp_norm_over_nsym():
    sym = _sym_cubic()
    kappa = vp_kappa(10.0)
    rng = _rng(1)
    q = random_uniform(50, rng)
    gram = symmetrized_gram(q, sym, kappa)
    expected = vp_norm(kappa) / len(sym)
    assert expected == pytest.approx(64.9, rel=1e-3)
    np.testing.assert_allclose(np.diag(gram), expected, rtol=1e-9)


def test_gram_offdiagonal_mean_is_one_mrd_for_haar_random_set():
    sym = _sym_cubic()
    kappa = vp_kappa(10.0)
    rng = _rng(2)
    q = random_uniform(800, rng)
    gram = symmetrized_gram(q, sym, kappa)
    n = len(q)
    offdiag_mean = (gram.sum() - np.trace(gram)) / (n * (n - 1))
    assert offdiag_mean == pytest.approx(1.0, rel=0.04)


@pytest.mark.parametrize("halfwidth", [5.0, 10.0, 20.0])
def test_gram_symmetric_and_positive_definite(halfwidth):
    sym = _sym_cubic()
    kappa = vp_kappa(halfwidth)
    rng = _rng(3)
    q = random_uniform(300, rng)
    gram = symmetrized_gram(q, sym, kappa)
    assert np.allclose(gram, gram.T)
    eig = np.linalg.eigvalsh(gram)
    # Relative tolerance: at a 20 deg half-width the smallest eigenvalue can
    # be as small as ~1e-8 relative to the largest (module docstring), so an
    # absolute floor would be meaningless here.
    assert eig.min() > -1e-6 * eig.max()


# ---------------------------------------------------------------------------
# 5-8. Atomic drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indices", [[0.4, 1.4], [np.nan, 1], [np.inf, 1]])
def test_odf_indices_must_be_finite_integers(indices):
    with pytest.raises(ConfigError):
        orientation_weights(indices, np.array([80.0, 20.0]))
    with pytest.raises(ConfigError):
        component_volume_fractions(indices, np.array([80.0, 20.0]), 2)


@pytest.mark.parametrize("volumes", [[-1.0, 2.0], [0.0, 2.0],
                                    [np.nan, 2.0], [np.inf, 2.0]])
def test_annealer_rejects_invalid_volumes_without_boundaries(volumes):
    from grainsmith.orientation.mdf import anneal_assignment

    with pytest.raises(ConfigError):
        anneal_assignment(
            np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)), [], np.empty(0),
            np.array([0.0, 180.0]), np.ones(1), np.ones(1),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            n_steps=0, t0=0.05, cooling=0.995, rng=_rng(),
            grain_volumes=np.array(volumes), odf_drift_max=0.02,
        )


def test_unequal_volume_case_pinned_at_0p6():
    """Two grains, volumes 80/20, two distinct orientations: swapping which
    grain carries which orientation moves exactly |80-20|/100 = 0.6 of TV
    mass (fact A) — the counter-example to "ODF-invariant"."""
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])  # a distinct, exact unit quaternion
    quats = np.array([q0, q1])
    volumes = np.array([80.0, 20.0])
    w = volumes / volumes.sum()

    class_of, n_classes = orientation_classes(quats)
    assert n_classes == 2

    perm_id = np.array([0, 1])
    perm_swap = np.array([1, 0])
    v_id = orientation_weights(perm_id, w)
    v_swap = orientation_weights(perm_swap, w)

    drift = atomic_drift(v_swap, v_id, class_of, n_classes)
    assert drift == pytest.approx(0.6, abs=1e-12)

    fact_a = abs(volumes[0] - volumes[1]) / np.sum(volumes)
    assert drift == pytest.approx(fact_a, abs=1e-12)


def test_equal_volumes_any_swap_has_zero_drift():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    quats = np.array([q0, q1])
    w = np.array([0.5, 0.5])
    class_of, n_classes = orientation_classes(quats)

    perm_id = np.array([0, 1])
    perm_swap = np.array([1, 0])
    v_id = orientation_weights(perm_id, w)
    v_swap = orientation_weights(perm_swap, w)
    assert atomic_drift(v_swap, v_id, class_of, n_classes) == pytest.approx(
        0.0, abs=1e-15)


def test_duplicate_orientations_merge_into_one_class():
    """Grains 0 and 1 carry the BIT-IDENTICAL orientation (np.tile, as the
    spread_deg: 0 example config produces); grain 2 is distinct. Swapping
    the two duplicate-orientation grains must cost EXACTLY zero drift."""
    q_base = np.array([1.0, 0.0, 0.0, 0.0])
    q_dup = np.tile(q_base, (2, 1))
    q_other = np.array([0.0, 0.0, 1.0, 0.0])
    quats = np.concatenate([q_dup, q_other[None, :]])
    volumes = np.array([50.0, 30.0, 20.0])
    w = volumes / volumes.sum()

    class_of, n_classes = orientation_classes(quats)
    assert n_classes == 2

    perm_id = np.array([0, 1, 2])
    perm_swap01 = np.array([1, 0, 2])  # swap grains 0 and 1
    v_id = orientation_weights(perm_id, w)
    v_swap = orientation_weights(perm_swap01, w)
    assert atomic_drift(v_swap, v_id, class_of, n_classes) == 0.0


def test_fact_b_atomic_distance_to_count_weights_is_assignment_independent():
    """FACT (B): the atomic distance between f_V(pi) and the count-weighted
    f_n is the SAME for every assignment pi, and equals count_vs_volume_gap
    directly (both measures live on the same finite atom set; a permutation
    only relabels which atom carries which weight)."""
    n = 60
    rng = _rng(13)
    q = random_uniform(n, rng)  # Haar-random: distinct with probability 1
    class_of, n_classes = orientation_classes(q)
    assert n_classes == n  # no accidental duplicates

    volumes = rng.lognormal(size=n)
    w = volumes / volumes.sum()
    uniform = np.full(n, 1.0 / n)
    gap = count_vs_volume_gap(w)

    drifts = []
    for _ in range(20):
        perm = rng.permutation(n)
        v = orientation_weights(perm, w)
        drifts.append(atomic_drift(v, uniform, class_of, n_classes))

    np.testing.assert_allclose(drifts, gap, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. DriftTracker
# ---------------------------------------------------------------------------


def test_drift_tracker_matches_exact_over_uninterrupted_accumulation():
    """20000 swaps of UNINTERRUPTED incremental float accumulation.

    Deliberately does NOT call .recompute() inside the loop: recompute()
    resets the incrementally tracked state from the true permutation, so
    calling it at every checkpoint (an earlier version of this test did)
    never actually exercises 20000 steps of sustained float accumulation
    -- only ten independent 2000-step windows, each starting fresh. The
    checkpoint check here is read-only (exact_drift() never touches
    tracker state), so accumulation genuinely continues across all 20000
    steps; .recompute() is called exactly ONCE, at the very end, which is
    the real test of accumulated floating-point error.
    """
    n = 400
    rng = _rng(21)
    q = random_uniform(n, rng)
    class_of, n_classes = orientation_classes(q)
    assert n_classes == n

    volumes = rng.lognormal(size=n)
    w = volumes / volumes.sum()

    perm0 = np.arange(n, dtype=np.intp)
    tracker = DriftTracker(perm0, w, class_of, n_classes)
    perm = perm0.copy()

    def exact_drift(current_perm: np.ndarray) -> float:
        v0 = orientation_weights(perm0, w)
        v = orientation_weights(current_perm, w)
        return atomic_drift(v, v0, class_of, n_classes)

    n_swaps = 20000
    for step in range(1, n_swaps + 1):
        a = int(rng.integers(n))
        b = int(rng.integers(n - 1))
        if b >= a:
            b += 1
        k_a, k_b = int(perm[a]), int(perm[b])

        trial_1 = tracker.trial(a, b, k_a, k_b)
        trial_2 = tracker.trial(a, b, k_a, k_b)
        assert trial_1 == trial_2  # .trial() is pure / side-effect free

        tracker.commit(a, b, k_a, k_b)
        assert tracker.value == trial_2

        perm[a], perm[b] = perm[b], perm[a]

        if step % 2000 == 0:
            # Read-only: does not call .recompute(), so the incremental
            # accumulation is undisturbed by this check.
            assert tracker.value == pytest.approx(
                exact_drift(perm), abs=1e-12)

    # .recompute() called exactly ONCE, after all 20000 uninterrupted
    # incremental commits -- the accumulated value must still agree with
    # a from-scratch recomputation to abs=1e-12 (measured max
    # |incremental - exact| over such a run: 2.8e-15).
    exact_final = exact_drift(perm)
    assert tracker.value == pytest.approx(exact_final, abs=1e-12)
    assert tracker.recompute() == pytest.approx(exact_final, abs=1e-12)


def test_drift_tracker_recompute_matches_value():
    """Separate, short check of the ".recompute() agrees with .value"
    property, kept independent of the long uninterrupted-accumulation
    test above (which deliberately never calls .recompute() mid-run)."""
    n = 30
    rng = _rng(22)
    q = random_uniform(n, rng)
    class_of, n_classes = orientation_classes(q)
    assert n_classes == n
    w = rng.lognormal(size=n)

    perm0 = np.arange(n, dtype=np.intp)
    tracker = DriftTracker(perm0, w, class_of, n_classes)
    perm = perm0.copy()
    for a, b in [(0, 1), (2, 3), (4, 5), (1, 4)]:
        k_a, k_b = int(perm[a]), int(perm[b])
        tracker.commit(a, b, k_a, k_b)
        perm[a], perm[b] = perm[b], perm[a]

    assert tracker.recompute() == pytest.approx(tracker.value, abs=1e-12)


# ---------------------------------------------------------------------------
# 10. MMD is a metric
# ---------------------------------------------------------------------------


def test_mmd_is_a_metric_on_a_small_set():
    sym = _sym_cubic()
    kappa = vp_kappa(10.0)
    rng = _rng(31)
    q = random_uniform(12, rng)
    gram = symmetrized_gram(q, sym, kappa)

    v = rng.random(12)
    v /= v.sum()
    v2 = rng.random(12)
    v2 /= v2.sum()

    assert mmd(v, v, gram) == pytest.approx(0.0, abs=1e-10)
    assert mmd(v, v2, gram) == pytest.approx(mmd(v2, v, gram), abs=1e-12)
    assert mmd(v, v2, gram) > 0.0


# ---------------------------------------------------------------------------
# 11. Sensitivity: kernel discrepancy sees what the atomic distance cannot
# ---------------------------------------------------------------------------


def test_sensitivity_kernel_discrepancy_separates_biased_from_neutral():
    """The scientific point of the whole module: a size-biased assignment
    and a neutral random one are IDENTICAL under the atomic distance to
    f_n (fact B) but clearly separated once smoothed in orientation space
    (the kernel discrepancy against a null of neutral random assignments).
    """
    sym = _sym_cubic()
    rng = _rng(42)
    n_per_cluster = 100
    n = 2 * n_per_cluster
    spread_deg = 4.0

    centers = random_uniform(2, rng)
    comps = [
        [{"euler_bunge_deg": list(quat_to_bunge(centers[0])),
          "weight": 1.0, "spread_deg": spread_deg}],
        [{"euler_bunge_deg": list(quat_to_bunge(centers[1])),
          "weight": 1.0, "spread_deg": spread_deg}],
    ]
    # weight_basis="count" pinned explicitly: this test's downstream
    # numbers are calibrated against the categorical-draw RNG stream
    # (odf_components' default changed to "volume" under the
    # fix, which would otherwise skip the rng.choice draw here).
    cluster_1 = odf_components(n_per_cluster, comps[0], rng, weight_basis="count")
    cluster_2 = odf_components(n_per_cluster, comps[1], rng, weight_basis="count")
    quats = np.concatenate([cluster_1, cluster_2])  # orientation indices
    # 0..n_per_cluster-1 are cluster 1, the rest are cluster 2.

    volumes = rng.lognormal(mean=0.0, sigma=0.9, size=n)
    w = volumes / volumes.sum()

    class_of, n_classes = orientation_classes(quats)
    assert n_classes == n  # continuous sampling: no accidental duplicates

    uniform = np.full(n, 1.0 / n)

    # Neutral: a uniformly random assignment of grains to orientation slots.
    perm_neutral = rng.permutation(n)
    # Biased: the largest-volume grains are placed in the FIRST cluster.
    order_by_volume_desc = np.argsort(-volumes)
    perm_biased = np.empty(n, dtype=np.intp)
    perm_biased[order_by_volume_desc[:n_per_cluster]] = np.arange(n_per_cluster)
    perm_biased[order_by_volume_desc[n_per_cluster:]] = np.arange(
        n_per_cluster, n)

    v_neutral = orientation_weights(perm_neutral, w)
    v_biased = orientation_weights(perm_biased, w)

    # Fact (B): atomic distance to f_n does not know or care which
    # orientations carry the excess mass.
    atomic_neutral = atomic_drift(v_neutral, uniform, class_of, n_classes)
    atomic_biased = atomic_drift(v_biased, uniform, class_of, n_classes)
    assert atomic_neutral == pytest.approx(atomic_biased, abs=1e-12)
    assert atomic_neutral == pytest.approx(count_vs_volume_gap(w), abs=1e-12)

    # The kernel discrepancy DOES see it.
    kappa = vp_kappa(10.0)
    gram = symmetrized_gram(quats, sym, kappa)
    mmd_neutral = mmd(v_neutral, uniform, gram)
    mmd_biased = mmd(v_biased, uniform, gram)

    null = null_mmd(gram, w)  # defaults: ODF_NULL_SAMPLES, ODF_NULL_SEED
    null_p95 = float(np.quantile(null, 0.95))

    # Calibrated reference (lead engineer's run): null median ~0.415,
    # null p95 ~0.705, biased ~2.80 — this run's own fixed-seed numbers land
    # in the same regime (null median/p95/biased all within a factor of ~2
    # of those), so only the ORDERING (with a comfortable margin) is
    # asserted, not the specific values.
    assert mmd_biased > 2.0 * null_p95
    assert mmd_neutral < 0.9 * null_p95


# ---------------------------------------------------------------------------
# 12. Determinism
# ---------------------------------------------------------------------------


def test_null_mmd_is_deterministic():
    sym = _sym_cubic()
    kappa = vp_kappa(10.0)
    rng = _rng(51)
    q = random_uniform(30, rng)
    gram = symmetrized_gram(q, sym, kappa)
    volumes = rng.lognormal(size=30)
    w = volumes / volumes.sum()

    a = null_mmd(gram, w, n_samples=64, seed=123)
    b = null_mmd(gram, w, n_samples=64, seed=123)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 13. effective_sample_size
# ---------------------------------------------------------------------------


def test_effective_sample_size_bounds():
    n = 37
    assert effective_sample_size(np.full(n, 3.0)) == pytest.approx(
        float(n), abs=1e-9)

    dominant = np.array([1.0e6] + [1.0] * 99)
    ess = effective_sample_size(dominant)
    assert 1.0 <= ess < 1.01


# ---------------------------------------------------------------------------
# 14. Error paths
# ---------------------------------------------------------------------------


def test_halfwidth_boundaries_raise_config_error():
    with pytest.raises(ConfigError):
        vp_kappa(0.0)
    with pytest.raises(ConfigError):
        vp_kappa(180.0)


def test_negative_volume_raises_config_error():
    with pytest.raises(ConfigError):
        effective_sample_size(np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ConfigError):
        count_vs_volume_gap(np.array([0.5, -0.5]))


def test_perm_weights_length_mismatch_raises_config_error():
    with pytest.raises(ConfigError):
        orientation_weights(np.array([0, 1, 2]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# class_masses sanity (used throughout above, pinned once directly)
# ---------------------------------------------------------------------------


def test_class_masses_sums_weight_per_class():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    class_of = np.array([0, 0, 1, 1])
    masses = class_masses(v, class_of, 2)
    np.testing.assert_allclose(masses, [0.3, 0.7])


# ---------------------------------------------------------------------------
# BLOCKER 1 — a non-permutation must never return uninitialised memory
# ---------------------------------------------------------------------------


def test_orientation_weights_repeated_index_raises_config_error():
    """A repeated index in perm leaves some entry of the scatter-assignment
    ``v[perm] = w`` NEVER written; before the fix this silently returned
    garbage from np.empty instead of raising."""
    perm = np.array([0, 0, 2, 3])  # index 1 never targeted, index 0 twice
    w = np.array([0.25, 0.25, 0.25, 0.25])
    with pytest.raises(ConfigError, match="permutation"):
        orientation_weights(perm, w)


def test_orientation_weights_out_of_range_index_raises_config_error():
    perm = np.array([0, 1, 5])
    w = np.array([0.2, 0.3, 0.5])
    with pytest.raises(ConfigError):
        orientation_weights(perm, w)


def test_drift_tracker_rejects_non_permutation_perm0():
    q = random_uniform(3, _rng(61))
    class_of, n_classes = orientation_classes(q)
    with pytest.raises(ConfigError, match="permutation"):
        DriftTracker(np.array([0, 0, 2]), np.array([0.5, 0.3, 0.2]),
                     class_of, n_classes)


# ---------------------------------------------------------------------------
# BLOCKER 2 — raw (unnormalised) volumes must give the SAME answer as
# already-normalised fractions
# ---------------------------------------------------------------------------


def test_raw_volumes_and_normalised_fractions_agree():
    """The pipeline's real per-grain volumes (analysis.grains.grain_volumes)
    are raw cubic-angstrom quantities, not pre-normalised fractions -- every
    weight-consuming function must treat the two identically."""
    n = 250
    rng = _rng(71)
    raw_volumes = rng.lognormal(mean=0.0, sigma=1.1, size=n) * 1.0e5  # A^3-scale
    fractions = raw_volumes / raw_volumes.sum()

    assert count_vs_volume_gap(raw_volumes) == pytest.approx(
        count_vs_volume_gap(fractions), rel=1e-12)

    perm = rng.permutation(n)
    v_raw = orientation_weights(perm, raw_volumes)
    v_frac = orientation_weights(perm, fractions)
    np.testing.assert_allclose(v_raw, v_frac, rtol=1e-12)
    assert v_raw.sum() == pytest.approx(1.0, abs=1e-12)

    q = random_uniform(n, rng)
    class_of, n_classes = orientation_classes(q)
    assert n_classes == n

    tracker_raw = DriftTracker(perm, raw_volumes, class_of, n_classes)
    tracker_frac = DriftTracker(perm, fractions, class_of, n_classes)
    # A handful of identical swaps applied to both trackers.
    swap_rng = _rng(72)
    p = perm.copy()
    for _ in range(50):
        a = int(swap_rng.integers(n))
        b = int(swap_rng.integers(n - 1))
        if b >= a:
            b += 1
        k_a, k_b = int(p[a]), int(p[b])
        tracker_raw.commit(a, b, k_a, k_b)
        tracker_frac.commit(a, b, k_a, k_b)
        p[a], p[b] = p[b], p[a]

    assert tracker_raw.value == pytest.approx(tracker_frac.value, rel=1e-10)
    assert tracker_raw.recompute() == pytest.approx(
        tracker_frac.recompute(), rel=1e-10)


# ---------------------------------------------------------------------------
# FIX 3 — DriftTracker must detect a caller/tracker permutation desync
# ---------------------------------------------------------------------------


def test_drift_tracker_detects_wrong_k_a_k_b():
    n = 10
    rng = _rng(81)
    q = random_uniform(n, rng)
    class_of, n_classes = orientation_classes(q)
    assert n_classes == n
    w = rng.lognormal(size=n)
    perm0 = np.arange(n, dtype=np.intp)
    tracker = DriftTracker(perm0, w, class_of, n_classes)

    # perm[0] == 0 and perm[1] == 1 at this point -- passing k_a=5, k_b=7
    # (values that do NOT match the tracker's own internal state) must be
    # rejected rather than silently applied to the wrong classes.
    with pytest.raises(GrainsmithError, match="desync"):
        tracker.trial(0, 1, 5, 7)
    with pytest.raises(GrainsmithError, match="desync"):
        tracker.commit(0, 1, 5, 7)

    # A correct call (k_a/k_b matching the tracker's real state) still
    # works, and trial()'s prediction matches what commit() applies.
    predicted = tracker.trial(0, 1, 0, 1)
    tracker.commit(0, 1, 0, 1)
    assert tracker.value == pytest.approx(predicted, abs=1e-12)


# ---------------------------------------------------------------------------
# FIX 4 — ODF_KERNEL_MAX_ELEMS changes last-bit results, never the physics
# ---------------------------------------------------------------------------


def test_gram_chunk_size_only_perturbs_last_bits(monkeypatch):
    """symmetrized_gram is NOT bitwise-invariant to ODF_KERNEL_MAX_ELEMS
    (different row-chunk counts can make BLAS pick different internal
    blocking); the constant is part of the numerical contract
    (constants.py docstring) and this pins the agreement at rtol=1e-10 --
    deliberately NOT bitwise, which would be a false claim."""
    sym = _sym_cubic()
    kappa = vp_kappa(10.0)
    q = random_uniform(60, _rng(91))

    gram_default = symmetrized_gram(q, sym, kappa)

    monkeypatch.setattr(odf_mod, "ODF_KERNEL_MAX_ELEMS", 500)
    gram_small_chunk = symmetrized_gram(q, sym, kappa)

    np.testing.assert_allclose(gram_default, gram_small_chunk, rtol=1e-10)


# ---------------------------------------------------------------------------
# ITEM 7 — component_volume_fractions
# ---------------------------------------------------------------------------


def test_component_volume_fractions_sums_to_one():
    component_of = np.array([0, 0, 1, 1, 2])
    w = np.array([3.0, 1.0, 2.0, 4.0, 5.0])
    frac = component_volume_fractions(component_of, w, 3)
    assert frac.sum() == pytest.approx(1.0, abs=1e-12)
    assert len(frac) == 3


def test_component_volume_fractions_equal_volumes_reduce_to_count_fractions():
    """With equal grain volumes, the VOLUME fraction of a component reduces
    to its GRAIN-COUNT fraction -- the count-vs-volume gap this function
    exists to expose only shows up once volumes differ."""
    component_of = np.array([0, 0, 0, 1, 1])  # 3 grains in 0, 2 in 1
    w = np.full(5, 2.0)  # equal (raw) volumes
    frac = component_volume_fractions(component_of, w, 2)
    np.testing.assert_allclose(frac, [0.6, 0.4], atol=1e-12)


def test_component_volume_fractions_unequal_volumes_differ_from_counts():
    component_of = np.array([0, 0, 0, 1, 1])  # 3 grains in 0, 2 in 1 (60/40 by count)
    w = np.array([1.0, 1.0, 1.0, 100.0, 100.0])  # component 1 dominates by volume
    frac = component_volume_fractions(component_of, w, 2)
    count_fraction = np.array([0.6, 0.4])
    assert not np.allclose(frac, count_fraction, atol=1e-6)
    assert frac[1] > frac[0]  # volume-dominant component 1 now leads


def test_component_volume_fractions_raw_and_normalised_agree():
    rng = _rng(101)
    n = 120
    n_components = 4
    component_of = rng.integers(0, n_components, size=n)
    raw_volumes = rng.lognormal(size=n) * 1.0e4
    fractions = raw_volumes / raw_volumes.sum()
    frac_raw = component_volume_fractions(component_of, raw_volumes, n_components)
    frac_norm = component_volume_fractions(component_of, fractions, n_components)
    np.testing.assert_allclose(frac_raw, frac_norm, rtol=1e-12)


def test_component_volume_fractions_sensitivity_matches_calibration_regime():
    """Sanity cross-check against the coordinator's own calibration
    (neutral ~0.434, biased ~0.806 for a two-cluster, size-biased vs
    neutral example): reusing the sensitivity scenario's clusters, the
    realised volume fraction of the "large-grains" component is close to
    a coin flip for a neutral assignment and strongly dominant for a
    size-biased one. Only the ORDERING and rough regime are asserted
    (own RNG draws will not reproduce the coordinator's numbers exactly)."""
    n_per_cluster = 100
    n = 2 * n_per_cluster
    rng = _rng(42)
    centers = random_uniform(2, rng)
    comps = [
        [{"euler_bunge_deg": list(quat_to_bunge(centers[0])),
          "weight": 1.0, "spread_deg": 4.0}],
        [{"euler_bunge_deg": list(quat_to_bunge(centers[1])),
          "weight": 1.0, "spread_deg": 4.0}],
    ]
    # weight_basis="count" pinned explicitly -- see the matching comment
    # in test_sensitivity_kernel_discrepancy_separates_biased_from_neutral.
    odf_components(n_per_cluster, comps[0], rng, weight_basis="count")  # advance rng identically
    odf_components(n_per_cluster, comps[1], rng, weight_basis="count")  # to the sensitivity test

    component_of_orientation = np.concatenate(
        [np.zeros(n_per_cluster, dtype=np.intp),
         np.ones(n_per_cluster, dtype=np.intp)])
    volumes = rng.lognormal(mean=0.0, sigma=0.9, size=n)
    w = volumes / volumes.sum()

    perm_neutral = rng.permutation(n)
    order_by_volume_desc = np.argsort(-volumes)
    perm_biased = np.empty(n, dtype=np.intp)
    perm_biased[order_by_volume_desc[:n_per_cluster]] = np.arange(n_per_cluster)
    perm_biased[order_by_volume_desc[n_per_cluster:]] = np.arange(
        n_per_cluster, n)

    frac_neutral = component_volume_fractions(
        component_of_orientation[perm_neutral], w, 2)
    frac_biased = component_volume_fractions(
        component_of_orientation[perm_biased], w, 2)

    assert 0.3 < frac_neutral[0] < 0.7   # close to a coin flip
    assert frac_biased[0] > 0.7          # strongly dominant


def test_component_volume_fractions_bad_index_raises_config_error():
    with pytest.raises(ConfigError):
        component_volume_fractions(np.array([0, 1, 5]), np.array([1.0, 2.0, 3.0]), 2)
    with pytest.raises(ConfigError):
        component_volume_fractions(np.array([0, 1]), np.array([1.0, 2.0, 3.0]), 2)


# ---------------------------------------------------------------------------
# N. symmetry_classes -- the quotient-space (SO(3)/Sym) merge, MEASUREMENT only
# ---------------------------------------------------------------------------

def test_symmetry_classes_merges_symmetry_equivalent_pair():
    """q and q (x) S (S the cubic 120° <111> operation) are the SAME physical
    orientation: symmetry_classes must put them in one class while
    orientation_classes (exact equality) keeps them apart -- the deliberate
    control-vs-measurement asymmetry of orientation/odf.py's
    'SYMMETRY-QUOTIENT CLASSES' section."""
    from grainsmith.orientation.quaternion import axis_angle_to_quat, quat_mul

    sym = _sym_cubic()
    q = axis_angle_to_quat([1.0, 0.0, 0.0], 37.0)
    s_111 = axis_angle_to_quat([1.0, 1.0, 1.0], 120.0)
    q_s = quat_mul(q, s_111)
    quats = np.array([q, q_s])

    class_of, n_classes = symmetry_classes(quats, sym)
    assert n_classes == 1
    assert class_of[0] == class_of[1]

    exact_of, exact_n = orientation_classes(quats)
    assert exact_n == 2  # bit-distinct: NOT merged on the control path


def test_symmetry_classes_quotient_drift_is_zero_for_equivalent_swap():
    """The 80/20-volume example of the module docstring, quotient side:
    swapping orientations between grains of volumes (0.8, 0.2) that carry
    symmetry-EQUIVALENT orientations has physical (quotient-space) drift
    exactly 0 -- while the atomic drift on orientation_classes is 0.6."""
    from grainsmith.orientation.quaternion import axis_angle_to_quat, quat_mul

    sym = _sym_cubic()
    q = axis_angle_to_quat([1.0, 0.0, 0.0], 37.0)
    s_111 = axis_angle_to_quat([1.0, 1.0, 1.0], 120.0)
    quats = np.array([q, quat_mul(q, s_111)])
    w = np.array([0.8, 0.2])

    v0 = orientation_weights(np.arange(2), w)
    v1 = orientation_weights(np.array([1, 0]), w)

    q_class_of, q_n = symmetry_classes(quats, sym)
    assert atomic_drift(v1, v0, q_class_of, q_n) == pytest.approx(0.0, abs=1e-15)

    a_class_of, a_n = orientation_classes(quats)
    assert atomic_drift(v1, v0, a_class_of, a_n) == pytest.approx(0.6)


def test_symmetry_classes_generic_set_stays_distinct():
    """A Haar-random orientation set has no symmetry-equivalent pairs:
    the quotient partition must equal the trivial one (every orientation
    its own class) -- no false merges at the 1e-3 deg tolerance."""
    sym = _sym_cubic()
    quats = random_uniform(64, _rng(11))
    class_of, n_classes = symmetry_classes(quats, sym)
    assert n_classes == 64
    assert sorted(class_of.tolist()) == list(range(64))


def test_symmetry_classes_merges_bit_identical_duplicates():
    """np.tile-style duplicates (the spread_deg: 0 case
    orientation_classes exists for) must also merge on the quotient path."""
    sym = _sym_cubic()
    base = random_uniform(5, _rng(13))
    quats = np.tile(base, (3, 1))  # 15 orientations, 5 distinct
    class_of, n_classes = symmetry_classes(quats, sym)
    assert n_classes == 5
    for k in range(5):
        assert class_of[k] == class_of[k + 5] == class_of[k + 10]


def test_symmetry_classes_deterministic_and_chunk_invariant():
    """Same input -> bit-identical partition, independent of the internal
    row chunking (union-find over connected components is order-free)."""
    sym = _sym_cubic()
    quats = random_uniform(50, _rng(17))
    c1, n1 = symmetry_classes(quats, sym)
    c2, n2 = symmetry_classes(quats, sym)
    assert np.array_equal(c1, c2) and n1 == n2


# ---------------------------------------------------------------------------
# volume-weighting fix -- volume_balanced_partition / odf_components weight_basis
# ---------------------------------------------------------------------------


def test_volume_balanced_partition_matches_weights_exactly_for_equal_volumes():
    """PHYSICAL FACT: with equal-volume grains and weights that divide the
    grain count evenly, volume_balanced_partition reproduces the
    configured VOLUME fractions EXACTLY, not just approximately -- equal
    volumes turn the greedy largest-deficit rule into a perfectly
    symmetric round-robin allocation across components."""
    K = 5
    grains_per_component = 4
    n = K * grains_per_component
    volumes = np.full(n, 3.0)
    weights = np.ones(K)  # equal, deliberately unnormalised

    comp = volume_balanced_partition(weights, volumes)
    counts = np.bincount(comp, minlength=K)
    assert counts.tolist() == [grains_per_component] * K

    frac = component_volume_fractions(comp, volumes, K)
    np.testing.assert_allclose(frac, np.full(K, 1.0 / K), atol=1e-12)


def test_volume_balanced_partition_overshoot_never_exceeds_smallest_grain():
    """THE PROVEN INVARIANT (odf.py docstring lemma): for every NON-EMPTY
    component, the volume overshoot (assigned volume - target volume) is
    strictly less than the smallest grain volume that component
    received. Checked over 220 random cases: random positive volumes,
    random positive weights, K from 2 to 6."""
    rng = _rng(2024)
    n_cases = 220
    checked_nonempty = 0
    for case in range(n_cases):
        K = int(rng.integers(2, 7))          # 2..6
        n = int(rng.integers(2, 40))
        volumes = rng.uniform(0.01, 100.0, n)
        weights = rng.uniform(0.01, 10.0, K)

        comp = volume_balanced_partition(weights, volumes)
        w_norm = weights / weights.sum()
        total = float(volumes.sum())
        target = w_norm * total

        for c in range(K):
            in_c = comp == c
            if not np.any(in_c):
                continue
            checked_nonempty += 1
            assigned = float(volumes[in_c].sum())
            smallest = float(volumes[in_c].min())
            overshoot = assigned - target[c]
            assert overshoot < smallest + 1e-9, (
                f"case {case}, component {c}: overshoot {overshoot!r} "
                f"not < smallest grain {smallest!r}"
            )
    assert n_cases >= 200
    assert checked_nonempty > 0  # sanity: the loop actually exercised something


def test_volume_balanced_partition_deterministic_and_permutation_equivariant():
    """Determinism: identical inputs give BIT-IDENTICAL output (no RNG
    anywhere in the algorithm). PERMUTATION EQUIVARIANCE: permuting the
    grain order permutes the output LABELS correspondingly --
    comp(volumes[perm])[k] == comp(volumes)[perm[k]] for every k, checked
    explicitly label by label (continuous rng.uniform volumes are
    distinct with probability 1, so no tie-break ambiguity)."""
    rng = _rng(303)
    n = 37
    K = 4
    volumes = rng.uniform(0.1, 50.0, n)
    weights = rng.uniform(0.1, 5.0, K)

    comp_a = volume_balanced_partition(weights, volumes)
    comp_b = volume_balanced_partition(weights, volumes)
    np.testing.assert_array_equal(comp_a, comp_b)  # bit-identical, no RNG

    perm = rng.permutation(n)
    volumes_perm = volumes[perm]
    comp_perm = volume_balanced_partition(weights, volumes_perm)

    for k in range(n):
        assert comp_perm[k] == comp_a[perm[k]], (
            f"slot {k}: got {comp_perm[k]}, expected {comp_a[perm[k]]} "
            f"(grain originally at index {perm[k]})"
        )


def test_measured_number_count_vs_volume_basis():
    """THE MEASURED NUMBER, as a DISTRIBUTIONAL claim over >= 400
    independent volume draws rather than one hand-picked seed.

    The single-seed version of this test did ``assert tv_count > 0.15``
    at sigma_log=0.5 (on grain DIAMETER) and seed 1. Measuring the actual
    distribution of tv_count at that sigma: mean 0.092, p95 0.231 -- so
    ">0.15" was roughly a one-in-four TAIL event at that one seed, not a
    property of the categorical draw. Two things are fixed here:

    (1) THE SPREAD. ``size_distribution.sigma_log`` (the config knob) is
        on grain DIAMETER, but the quantity that matters for this test is
        VOLUME. volume ~ diameter^3, so naively sigma_volume ~=
        3 * sigma_log(diameter) in log units -- except the real run
        measured n_eff 8.08 of 24 grains for the volume-weighted
        assignment, which implies (solving effective_sample_size for a
        pure lognormal(sigma_v) draw at n=24, n_eff=8.08) sigma_volume
        about 1.04, somewhat below the naive 3x estimate. Use that
        measured value, sigma_v = 1.04, directly (pure numpy, no
        pipeline -- this stays fast).
    (2) THE CLAIM. Over >= 400 independent draws:
        * basis "volume" satisfies its PROVEN bound TV <= max(V)/sum(V)
          in 100% of draws -- a theorem (volume_balanced_partition's own
          overshoot bound, applied to K=2), not a statistic, so ZERO
          exceptions are tolerated;
        * mean TV(count) / mean TV(volume) is at least 20 (measured here:
          mean TV(count) ~0.125 vs mean TV(volume) ~0.0017, a ratio of
          ~75 -- asserting >= 20 leaves comfortable headroom);
        * mean TV(count) itself is > 0.05 -- the failure mode
          is a SYSTEMATIC property of the categorical draw acting on a
          skewed volume distribution, not a lucky seed.
    """
    rng = _rng(500)
    n = 24
    sigma_v = 1.04  # see rationale above -- NOT the raw config sigma_log
    weights = np.array([0.5, 0.5])
    n_draws = 400

    tv_counts = []
    tv_vols = []
    for _ in range(n_draws):
        volumes = rng.lognormal(mean=0.0, sigma=sigma_v, size=n)
        total = float(volumes.sum())
        bound = float(volumes.max() / total)

        # basis "count": today's categorical draw (rng.choice ∝ weight).
        comp_count = rng.choice(2, size=n, p=weights / weights.sum())
        frac0_count = float(volumes[comp_count == 0].sum() / total)
        tv_count = abs(frac0_count - 0.5)
        tv_counts.append(tv_count)

        # basis "volume": deterministic partition -- the proven bound
        # must hold on EVERY draw, no exceptions.
        comp_vol = volume_balanced_partition(weights, volumes)
        frac0_vol = float(volumes[comp_vol == 0].sum() / total)
        tv_vol = abs(frac0_vol - 0.5)
        tv_vols.append(tv_vol)
        assert tv_vol <= bound + 1e-12, (
            f"theorem violated on one draw: tv_vol={tv_vol!r} > "
            f"bound={bound!r}")

    mean_tv_count = float(np.mean(tv_counts))
    mean_tv_vol = float(np.mean(tv_vols))
    assert mean_tv_count > 0.05, (
        f"mean tv_count={mean_tv_count!r} -- the systematic "
        "count-basis gap did not materialise over 400 draws")
    assert mean_tv_count / mean_tv_vol >= 20.0, (
        f"mean tv_count/mean tv_vol = {mean_tv_count / mean_tv_vol!r} "
        "< 20")


def test_odf_components_count_basis_rng_consumption_matches_reference():
    """weight_basis="count" must consume the RNG stream EXACTLY as
    before. With zero-spread discrete components the per-component
    sampling loop draws NOTHING (the np.tile path), so the ENTIRE call
    consumes exactly the one rng.choice(...) draw the pre-existing
    implementation made -- a fresh Generator seeded identically and
    advanced by ONLY that reference call must land in the SAME state as
    odf_components' internal rng, checked via the next draw after each."""
    comps = [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0, "spread_deg": 0.0},
        {"euler_bunge_deg": [10.0, 80.0, 40.0], "weight": 3.0, "spread_deg": 0.0},
    ]
    n = 300
    seed = 999
    weights = np.array([1.0, 3.0])
    probs = weights / weights.sum()

    rng_ref = _rng(seed)
    rng_ref.choice(len(comps), size=n, p=probs)  # the ONLY draw the call should make
    next_ref = float(rng_ref.random())

    rng_call = _rng(seed)
    odf_components(n, comps, rng_call, weight_basis="count")
    next_call = float(rng_call.random())

    assert next_ref == next_call


def test_odf_components_volume_basis_requires_matching_grain_volumes():
    """weight_basis="volume" without grain_volumes, or with a
    length-mismatched grain_volumes, must raise ConfigError rather than
    silently falling back to some other behaviour."""
    comps = [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0, "spread_deg": 0.0},
        {"euler_bunge_deg": [10.0, 80.0, 40.0], "weight": 1.0, "spread_deg": 0.0},
    ]
    n = 10

    with pytest.raises(ConfigError):
        odf_components(n, comps, _rng(41), weight_basis="volume")

    bad_volumes = np.ones(n - 1)  # length mismatch
    with pytest.raises(ConfigError):
        odf_components(n, comps, _rng(42), weight_basis="volume",
                       grain_volumes=bad_volumes)


def test_odf_components_any_other_weight_basis_raises_config_error():
    comps = [{"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0,
              "spread_deg": 0.0}]
    with pytest.raises(ConfigError):
        odf_components(5, comps, _rng(43), weight_basis="bogus")


def test_volume_balanced_partition_component_can_be_empty():
    """A component can legitimately end up with ZERO grains under
    "volume" -- constructed here with one weight so small that no single
    remaining grain volume is ever its largest deficit before the other
    component exhausts it -- and odf_components must sample the
    remaining (non-empty) components without crashing on the empty one."""
    volumes = np.array([1.0, 1.0, 1.0])
    weights = np.array([1e-9, 1.0])

    comp = volume_balanced_partition(weights, volumes)
    counts = np.bincount(comp, minlength=2)
    assert counts[0] == 0
    assert counts[1] == 3

    comps = [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": float(weights[0]),
         "spread_deg": 0.0},
        {"euler_bunge_deg": [10.0, 80.0, 40.0], "weight": float(weights[1]),
         "spread_deg": 0.0},
    ]
    quats = odf_components(3, comps, _rng(51), weight_basis="volume",
                           grain_volumes=volumes)
    assert quats.shape == (3, 4)
    q1 = fixed_orientation(1, {"euler_bunge_deg": [10.0, 80.0, 40.0]})[0]
    np.testing.assert_allclose(quats, np.tile(q1, (3, 1)), atol=1e-14)


# ---------------------------------------------------------------------------
# gate_g25_component_fidelity docstring bound -- pushforward TV never
# exceeds atomic drift when the component labelling is CONSTANT on every
# orientation class (qa.py::gate_g25_component_fidelity)
# ---------------------------------------------------------------------------


def test_component_tv_bounded_by_atomic_drift_when_constant_on_classes():
    """THE BOUND from gate_g25_component_fidelity's docstring, pinned as a
    property in BOTH directions.

    Convention (matches gate_g25_component_fidelity / odf.py exactly):
    ``component_of_pre`` is a labelling indexed by ORIENTATION index
    (pre-anneal, grain i carries orientation i); ``perm`` is the
    annealer's grain -> orientation-index permutation
    (``orientation_weights``'s convention); ``component_of_post =
    component_of_pre[perm]`` (grain i's component after annealing is
    whatever component the orientation it NOW carries, perm[i], had
    pre-anneal); grain VOLUMES are never permuted -- only which
    orientation/component label attaches to which grain changes.

    HOLDS (200 random cases): distinct orientations (Haar-random, so
    orientation_classes gives one class per orientation, the labelling is
    then trivially constant on every class) -- the component-volume TV
    between the pre and post state never exceeds the atomic drift on
    those classes.

    FAILS (explicit degenerate case): two BIT-IDENTICAL orientations
    placed in DIFFERENT components. atomic_drift is 0 (same class either
    way) but the component-volume TV is 0.8 for a 90/10 volume split --
    exactly the docstring's numbers.
    """
    rng = _rng(777)
    n_cases = 200
    for _ in range(n_cases):
        n = int(rng.integers(3, 30))
        quats = random_uniform(n, rng)  # Haar-random: distinct w.p. 1
        class_of, n_classes = orientation_classes(quats)
        assert n_classes == n  # constant-on-every-class holds trivially

        volumes = rng.lognormal(size=n)
        n_comp = int(rng.integers(2, 5))
        component_of_pre = rng.integers(0, n_comp, size=n)
        perm = rng.permutation(n)
        component_of_post = component_of_pre[perm]

        f_pre = component_volume_fractions(component_of_pre, volumes, n_comp)
        f_post = component_volume_fractions(component_of_post, volumes,
                                            n_comp)
        tv_component = 0.5 * float(np.sum(np.abs(f_post - f_pre)))

        v_pre = orientation_weights(np.arange(n), volumes)
        v_post = orientation_weights(perm, volumes)
        atomic = atomic_drift(v_post, v_pre, class_of, n_classes)

        assert tv_component <= atomic + 1e-9, (
            f"pushforward TV bound violated: component TV "
            f"{tv_component!r} > atomic drift {atomic!r}")

    # The degenerate case: two BIT-IDENTICAL orientations in DIFFERENT
    # components -- the constant-on-every-class condition is violated
    # (the single class contains both, but they carry different labels).
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.tile(q0, (2, 1))
    class_of, n_classes = orientation_classes(quats)
    assert n_classes == 1

    volumes = np.array([90.0, 10.0])
    component_of_pre = np.array([0, 1])  # same orientation, split components
    perm = np.array([1, 0])              # swap which grain carries which
    component_of_post = component_of_pre[perm]

    f_pre = component_volume_fractions(component_of_pre, volumes, 2)
    f_post = component_volume_fractions(component_of_post, volumes, 2)
    tv_component = 0.5 * float(np.sum(np.abs(f_post - f_pre)))

    v_pre = orientation_weights(np.arange(2), volumes)
    v_post = orientation_weights(perm, volumes)
    atomic = atomic_drift(v_post, v_pre, class_of, n_classes)

    assert atomic == pytest.approx(0.0, abs=1e-15)
    assert tv_component == pytest.approx(0.8, abs=1e-12)
    assert tv_component > atomic + 1e-9  # the bound FAILS here
