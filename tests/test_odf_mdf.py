"""Tests for ODF components + MDF targeting."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.errors import ConfigError
from grainsmith.orientation import odf
from grainsmith.orientation.mdf import (
    anneal_assignment,
    build_bins_and_target,
    chi2_distance,
    histogram_masses,
    reference_angles,
    theta_max_deg,
)
from grainsmith.orientation.misorientation import (
    disorientation,
    disorientation_angles,
)
from grainsmith.orientation.quaternion import (
    axis_angle_to_quat,
    quat_mul,
)
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


_SYM_IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# disorientation_angles (batch, one-sided orbit)
# ---------------------------------------------------------------------------


def test_batch_angles_match_full_disorientation():
    """One-sided-orbit batch angles == full two-sided disorientation()."""
    sym = _sym_cubic()
    rng = _rng(11)
    qa = random_uniform(64, rng)
    qb = random_uniform(64, rng)
    batch = disorientation_angles(qa, qb, sym)
    for k in range(64):
        full = disorientation(qa[k], qb[k], sym).angle_deg
        assert batch[k] == pytest.approx(full, abs=1e-9)


def test_batch_angles_sigma3():
    sym = _sym_cubic()
    q_twin = axis_angle_to_quat(np.array([1.0, 1.0, 1.0]), 60.0)
    ang = disorientation_angles(
        np.array([[1.0, 0.0, 0.0, 0.0]]), q_twin[None, :], sym)
    assert ang[0] == pytest.approx(60.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ODF component sampler
# ---------------------------------------------------------------------------


def test_single_component_zero_spread_reduces_to_fixed():
    """s → 0 limit: one Euler component == orientation.fixed (pin)."""
    euler = [35.0, 45.0, 10.0]
    # weight_basis pinned to "count": this test is about orientation
    # SAMPLING (the zero-spread limit), not the component partition --
    # a single component makes count vs. volume moot anyway, but pin it
    # explicitly rather than rely on the default.
    got = odf_components(
        5, [{"euler_bunge_deg": euler, "weight": 1.0, "spread_deg": 0.0}],
        _rng(), weight_basis="count")
    want = fixed_orientation(5, {"euler_bunge_deg": euler})
    np.testing.assert_allclose(got, want, atol=1e-14)


def test_component_weights_respected():
    """Categorical assignment fractions match the weights (χ²-level) on
    basis "count"; a volume-basis companion below checks what "weight"
    means under basis "volume" instead -- the realised VOLUME fraction,
    not the grain-count fraction, and shows the two genuinely differ."""
    n = 4000
    comps = [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0,
         "spread_deg": 0.0},
        {"euler_bunge_deg": [10.0, 80.0, 40.0], "weight": 3.0,
         "spread_deg": 0.0},
    ]
    q0 = fixed_orientation(1, {"euler_bunge_deg": [0.0, 0.0, 0.0]})[0]

    # weight_basis pinned to "count": this half of the test is about the
    # categorical draw's GRAIN-COUNT fractions, not the volume partition.
    quats = odf_components(n, comps, _rng(3), weight_basis="count")
    is_comp0 = np.all(np.abs(quats - q0) < 1e-12, axis=1)
    frac0 = np.mean(is_comp0)
    # 3-sigma band of a binomial fraction at p=0.25, n=4000: ±0.021
    assert frac0 == pytest.approx(0.25, abs=0.025)

    # --- volume-basis companion -------------------------------------
    # "weight" means VOLUME fraction under basis "volume". Deterministic
    # (non-random, so this is a property of the construction rather than
    # a lucky draw) unequal volumes: 600 "big" grains of volume 10 among
    # 3400 "small" grains of volume 1, heavily right-skewed so the two
    # bases genuinely differ and the count-basis gap below is large and
    # reproducible rather than marginal.
    volumes = np.where(np.arange(n) < 600, 10.0, 1.0)
    total = float(volumes.sum())
    bound = float(volumes.max() / total)  # volume_balanced_partition's
    # own proven overshoot bound (test_odf_guard.py), applied to K=2.

    quats_vol = odf_components(n, comps, _rng(3), weight_basis="volume",
                               grain_volumes=volumes)
    is_comp0_vol = np.all(np.abs(quats_vol - q0) < 1e-12, axis=1)
    frac0_vol = float(volumes[is_comp0_vol].sum() / total)
    tv_vol = abs(frac0_vol - 0.25)
    assert tv_vol <= bound + 1e-12

    # The SAME volumes, realised through the basis "count" assignment
    # above, do NOT satisfy that bound: the categorical draw ignores
    # volume entirely, so whichever of the 600 big grains it happens to
    # place lands the realised volume fraction far outside the tight
    # volume-basis bound (measured margin ~7x at this seed; the skew is
    # constructed so this holds by a wide margin, not by luck).
    frac0_count_vol = float(volumes[is_comp0].sum() / total)
    tv_count_vol = abs(frac0_count_vol - 0.25)
    assert tv_count_vol > bound


def test_spread_angles_follow_haar_corrected_law():
    """Perturbation angles ~ exp(−θ²/2σ²)·sin²(θ/2) (KS test, n=2e4)."""
    n = 20000
    sigma_deg = 10.0
    base = fixed_orientation(1, {"euler_bunge_deg": [0.0, 0.0, 0.0]})[0]
    # weight_basis pinned to "count": this test is about the SPREAD law
    # (single component, so the partition mechanism is irrelevant here).
    quats = odf_components(
        n, [{"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0,
             "spread_deg": sigma_deg}], _rng(5), weight_basis="count")
    # rotation angle to the base orientation (no symmetry reduction)
    cosw = np.abs(quats @ base)
    theta = 2.0 * np.arccos(np.clip(cosw, 0.0, 1.0))

    s = np.radians(sigma_deg)
    grid = np.linspace(0.0, np.pi, 20001)
    pdf = np.exp(-grid**2 / (2.0 * s**2)) * np.sin(grid / 2.0) ** 2
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    f_theory = np.interp(theta, grid, cdf)
    f_emp = (np.argsort(np.argsort(theta)) + 1.0) / n
    ks = float(np.max(np.abs(f_emp - f_theory)))
    # KS critical value at alpha=0.01 is 1.63/sqrt(n) ≈ 0.0115
    assert ks < 0.0115, f"KS={ks:.4f} against the Haar-corrected law"


def test_random_fraction_component():
    """{random: true} mixes a Haar-uniform fraction with the components."""
    n = 1000
    comps = [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 0.5,
         "spread_deg": 0.0},
        {"random": True, "weight": 0.5},
    ]
    # weight_basis pinned to "count": this test is about the SAMPLING mix
    # (fixed component vs. Haar-uniform random component), not the
    # partition mechanism -- and "volume" would additionally require a
    # grain_volumes array this test has no reason to construct.
    quats = odf_components(n, comps, _rng(9), weight_basis="count")
    q0 = fixed_orientation(1, {"euler_bunge_deg": [0.0, 0.0, 0.0]})[0]
    exact = np.all(np.abs(quats - q0) < 1e-12, axis=1)
    assert 0.4 < np.mean(exact) < 0.6          # textured half
    rest = quats[~exact]
    # the random half is dispersed: no two equal quaternions expected
    assert len(np.unique(np.round(rest, 6), axis=0)) == len(rest)


def test_fiber_component():
    """Fiber components reuse the v1 fiber machinery inside the mix."""
    comps = [{"fiber": {"crystal_axis": [0.0, 0.0, 1.0],
                        "sample_direction": "z", "spread_deg": 0.0},
              "weight": 1.0, "spread_deg": 0.0}]
    # weight_basis pinned to "count": this test is about the FIBER
    # sampling path (a single component), not the partition mechanism.
    quats = odf_components(50, comps, _rng(13), weight_basis="count")
    # <001> ∥ z with zero spread: the rotated z-axis equals z for every grain
    from grainsmith.orientation.quaternion import quat_to_matrix
    for q in quats:
        z = quat_to_matrix(q) @ np.array([0.0, 0.0, 1.0])
        assert z[2] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Random-pair reference (Mackenzie)
# ---------------------------------------------------------------------------


def test_reference_identity_group_analytic_mean():
    """No symmetry: p(θ) = (2/π)sin²(θ/2) ⇒ mean = π/2 + 2/π (exact pin)."""
    ang = np.radians(reference_angles(_SYM_IDENTITY))
    want = np.pi / 2.0 + 2.0 / np.pi
    assert float(np.mean(ang)) == pytest.approx(want, abs=5e-3)
    assert theta_max_deg(np.degrees(ang)) == pytest.approx(180.0, abs=0.5)


def test_reference_cubic_mackenzie_shape():
    """Cubic random-pair reference reproduces the Mackenzie law features:
    θ_max ≈ 62.8°, mode near 45°."""
    ref = reference_angles(_sym_cubic())
    t_max = theta_max_deg(ref)
    assert 62.5 <= t_max <= 63.0
    edges = np.linspace(0.0, t_max, 41)
    masses = histogram_masses(ref, np.ones_like(ref), edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[int(np.argmax(masses))])
    assert 40.0 < mode < 50.0
    assert masses.sum() == pytest.approx(1.0, abs=1e-12)
    # determinism: fixed internal seed ⇒ identical curve on a second call
    np.testing.assert_array_equal(ref, reference_angles(_sym_cubic()))


def test_reference_cubic_matches_mackenzie_closed_form():
    """SCIENCE_AUDIT #7: the cubic random-pair reference matches the
    CLOSED-FORM Mackenzie disorientation-angle law (not just self-consistency).
    Pinned load-bearing numbers: the analytic region masses 0–45° ≈ 0.598 and
    45–60° ≈ 0.394, and the Mackenzie median ≈ 42.3°."""
    ref = reference_angles(_sym_cubic())
    assert float(np.mean(ref <= 45.0)) == pytest.approx(0.598, abs=0.012)
    assert float(np.mean((ref > 45.0) & (ref <= 60.0))) == \
        pytest.approx(0.394, abs=0.012)
    assert float(np.median(ref)) == pytest.approx(42.3, abs=0.6)


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------


def _ring_pairs(n: int, k: int = 3) -> list[tuple[int, int]]:
    """Deterministic k-nearest ring adjacency (each grain ~2k neighbors)."""
    pairs = set()
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


class _MdfCfg:
    """Duck-typed stand-in for MdfTargetConfig in unit tests."""
    def __init__(self, **kw):
        self.type = kw.get("type", "mackenzie")
        self.sigma3_fraction = kw.get("sigma3_fraction", 0.3)
        self.bin_edges = kw.get("bin_edges")
        self.densities = kw.get("densities")


def test_mackenzie_negative_control():
    """random_uniform orientations on random adjacency are ALREADY
    Mackenzie-distributed: χ² ≪ G12 default threshold with NO annealing."""
    sym = _sym_cubic()
    n = 300
    quats = random_uniform(n, _rng(17))
    pairs = _ring_pairs(n, k=4)
    areas = np.ones(len(pairs))
    edges, target, ref = build_bins_and_target(_MdfCfg(), sym, 40)
    res = anneal_assignment(quats, pairs, areas, edges, target, ref, sym,
                            n_steps=0, t0=0.05, cooling=0.995, rng=_rng(1))
    assert res.chi2_initial == res.chi2_final
    assert res.chi2_initial < 0.5, (
        f"negative control χ²={res.chi2_initial:.3f} must pass G12 default")
    np.testing.assert_array_equal(res.permutation, np.arange(n))


def test_annealing_reduces_chi2_and_is_deterministic():
    sym = _sym_cubic()
    n = 60
    quats = random_uniform(n, _rng(23))
    pairs = _ring_pairs(n, k=3)
    areas = np.ones(len(pairs))
    cfg = _MdfCfg(type="csl_enriched", sigma3_fraction=0.5)
    edges, target, ref = build_bins_and_target(cfg, sym, 30)

    res1 = anneal_assignment(quats, pairs, areas, edges, target, ref, sym,
                             n_steps=3000, t0=0.05, cooling=0.995,
                             rng=_rng(2))
    res2 = anneal_assignment(quats, pairs, areas, edges, target, ref, sym,
                             n_steps=3000, t0=0.05, cooling=0.995,
                             rng=_rng(2))
    assert res1.chi2_final <= res1.chi2_initial
    assert res1.chi2_final == res2.chi2_final
    np.testing.assert_array_equal(res1.permutation, res2.permutation)
    # the permutation really is a permutation (ODF invariance)
    assert sorted(res1.permutation.tolist()) == list(range(n))

    # the annealer's claimed energy matches a fresh re-measure of the
    # returned assignment (G11-style honesty cross-check)
    p = res1.permutation
    ang = disorientation_angles(
        quats[p[[i for i, _ in pairs]]], quats[p[[j for _, j in pairs]]],
        sym)
    masses = histogram_masses(ang, areas, edges)
    assert chi2_distance(masses, target) == pytest.approx(
        res1.chi2_final, abs=1e-9)


def test_csl_enriched_raises_sigma3_window_fraction():
    """Direction pin: with a twin-containing orientation
    set, csl_enriched annealing raises the area fraction inside the Σ3
    Brandon angular window."""
    sym = _sym_cubic()
    n = 80
    rng = _rng(29)
    base = random_uniform(n // 2, rng)
    q_twin = axis_angle_to_quat(np.array([1.0, 1.0, 1.0]), 60.0)
    twins = np.array([quat_mul(q, q_twin) for q in base])
    quats = np.concatenate([base, twins])  # halves are exact Σ3 partners
    pairs = _ring_pairs(n, k=3)
    areas = np.ones(len(pairs))
    cfg = _MdfCfg(type="csl_enriched", sigma3_fraction=0.6)
    edges, target, ref = build_bins_and_target(cfg, sym, 40)

    def window_fraction(perm: np.ndarray) -> float:
        ang = disorientation_angles(
            quats[perm[[i for i, _ in pairs]]],
            quats[perm[[j for _, j in pairs]]], sym)
        brandon3 = 15.0 / np.sqrt(3.0)
        return float(np.mean(np.abs(ang - 60.0) <= brandon3))

    before = window_fraction(np.arange(n))
    res = anneal_assignment(quats, pairs, areas, edges, target, ref, sym,
                            n_steps=8000, t0=0.05, cooling=0.999,
                            rng=_rng(4))
    after = window_fraction(res.permutation)
    assert after > before, f"Σ3 window fraction {before:.3f} → {after:.3f}"
    assert res.chi2_final < res.chi2_initial


def test_histogram_target():
    """User histogram target: bins and normalization are honored."""
    sym = _sym_cubic()
    cfg = _MdfCfg(type="histogram",
                  bin_edges=[0.0, 20.0, 40.0, 63.0],
                  densities=[0.0, 1.0, 1.0])
    edges, target, ref = build_bins_and_target(cfg, sym, 40)
    np.testing.assert_array_equal(edges, [0.0, 20.0, 40.0, 63.0])
    assert target[0] == 0.0
    assert target.sum() == pytest.approx(1.0)
    # densities 1:1 over widths 20 and 23 → masses 20/43, 23/43
    assert target[1] == pytest.approx(20.0 / 43.0)
    assert ref.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Volume-weighted ODF trust region (odf_drift_max / grain_volumes)
# ---------------------------------------------------------------------------

def _drift_setup():
    """Shared (sym, n, quats, pairs, pair_areas, edges, target, ref) base
    for the trust-region tests below -- EXACTLY the regression-anchor
    setup (test 1), reused so every test in this section anneals the same
    orientation set / adjacency / target."""
    sym = _sym_cubic()
    n = 60
    quats = random_uniform(n, _rng(23))
    pairs = _ring_pairs(n, k=3)
    pair_areas = _rng(101).lognormal(0.0, 0.5, len(pairs))  # non-uniform
    cfg = _MdfCfg(type="csl_enriched", sigma3_fraction=0.5)
    edges, target, ref = build_bins_and_target(cfg, sym, 30)
    return sym, n, quats, pairs, pair_areas, edges, target, ref


def test_trust_region_default_is_bit_identical_regression_anchor():
    """TEST 1 -- REGRESSION ANCHOR. Values captured from the code BEFORE
    the odf_drift_max/grain_volumes change: pins that leaving the two new
    keyword arguments at their defaults reproduces today's behaviour
    bit-for-bit, including RNG stream consumption."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    rng = _rng(2)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=rng)
    next_draw = float(rng.random())

    digest = hashlib.sha256(
        res.permutation.astype(np.int64).tobytes()).hexdigest()
    assert digest == (
        "e3354eb88874a3c1a3ed65f83371e1468afe35fcc1389f26392b4251e9cdd2c3")
    assert res.permutation[:20].tolist() == [
        0, 30, 17, 34, 29, 27, 21, 43, 40, 7, 28, 41, 8, 36, 16, 56, 15, 47,
        5, 33]
    assert res.chi2_initial == 0.3620907677285693
    assert res.chi2_final == 0.0587966533975942
    assert res.n_accepted == 377
    assert next_draw == 0.4482527776129177
    assert res.odf_drift_final is None


def test_trust_region_measure_only_does_not_perturb_anything():
    """TEST 2 -- measure-only mode (grain_volumes given, cap None) tracks
    drift but never vetoes: permutation and RNG consumption must match
    the regression anchor exactly."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    baseline = anneal_assignment(quats, pairs, pair_areas, edges, target,
                                 ref, sym, n_steps=3000, t0=0.05,
                                 cooling=0.995, rng=_rng(2))
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    rng = _rng(2)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=rng,
                            grain_volumes=grain_volumes, odf_drift_max=None)
    next_draw = float(rng.random())

    np.testing.assert_array_equal(res.permutation, baseline.permutation)
    assert next_draw == 0.4482527776129177
    assert res.n_drift_vetoed == 0
    assert res.odf_drift_final is not None
    assert math.isfinite(res.odf_drift_final)
    assert res.odf_drift_final > 0.0


def test_trust_region_binding_cap_does_not_change_rng_stream():
    """TEST 3 -- THE KEY CLAIM: a binding cap changes WHICH moves are
    accepted but never how many random numbers are drawn."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)

    unconstrained = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym,
        n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
        grain_volumes=grain_volumes, odf_drift_max=None)
    cap = 0.25 * unconstrained.odf_drift_final

    rng = _rng(2)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=rng,
                            grain_volumes=grain_volumes, odf_drift_max=cap)
    next_draw = float(rng.random())

    assert next_draw == 0.4482527776129177
    assert res.n_drift_vetoed > 0
    assert not np.array_equal(res.permutation, unconstrained.permutation)


def test_trust_region_cap_actually_binds():
    """TEST 4 -- the cap actually binds, verified independently of the
    reported field (recomputed straight from odf.atomic_drift)."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    unconstrained = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym,
        n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
        grain_volumes=grain_volumes, odf_drift_max=None)
    cap = 0.25 * unconstrained.odf_drift_final

    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
                            grain_volumes=grain_volumes, odf_drift_max=cap)
    assert res.odf_drift_final <= cap + 1e-12

    class_of, n_classes = odf.orientation_classes(quats)
    v_best = odf.orientation_weights(res.permutation, grain_volumes)
    v_ref = odf.orientation_weights(np.arange(n), grain_volumes)
    independent = odf.atomic_drift(v_best, v_ref, class_of, n_classes)
    assert independent <= cap + 1e-12


def test_trust_region_equal_volumes_cap_zero_is_free():
    """TEST 5 -- cap 0 costs nothing when every grain has the SAME volume:
    every swap is drift-neutral, so the walk is untouched."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    baseline = anneal_assignment(quats, pairs, pair_areas, edges, target,
                                 ref, sym, n_steps=3000, t0=0.05,
                                 cooling=0.995, rng=_rng(2))
    grain_volumes = np.full(n, 7.0)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
                            grain_volumes=grain_volumes, odf_drift_max=0.0)
    assert res.n_drift_vetoed == 0
    assert res.odf_drift_final == 0.0
    np.testing.assert_array_equal(res.permutation, baseline.permutation)


def test_trust_region_cap_zero_unequal_volumes_freezes_assignment():
    """TEST 6 -- cap 0 with unequal volumes and all-distinct orientations:
    EVERY transposition of distinct-volume grains costs strictly positive
    drift, so nothing is ever accepted and the identity survives."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
                            grain_volumes=grain_volumes, odf_drift_max=0.0)
    np.testing.assert_array_equal(res.permutation, np.arange(n))
    assert res.n_accepted == 0
    assert res.n_drift_vetoed > 0
    assert res.odf_drift_final == 0.0


def test_trust_region_cap_zero_freezes_the_histogram():
    """TEST 7 -- cap 0 FREEZES the misorientation histogram completely,
    two different ways:

    * distinct orientations (TEST 6): every swap changes which class
      carries which volume, so every trial is vetoed outright -- chi2
      never gets a chance to move.
    * duplicate orientations (this test, mirroring spread_deg: 0 ->
      np.tile bit-identical quaternions): a swap between two grains that
      carry the SAME quaternion is not merely drift-free, it is a literal
      NO-OP for the misorientation histogram too -- the disorientation
      angle of every affected pair is a pure function of the (unchanged)
      quaternion values, so `chi2` cannot move either, even though such
      swaps ARE accepted (they cost nothing, so nothing here should be
      read as "progress" -- n_accepted is not a progress measure).

    eps = 0 is therefore a formal limit, not a useful operating point --
    the annealer is genuinely free to move at cap 0 only when every grain
    additionally has the SAME volume (TEST 5), where every swap is
    drift-free by construction.  The useful regime is eps > 0, exercised
    by test_trust_region_chi2_vs_drift_tradeoff below.

    Floating-point footnote: chi2_final/odf_drift_final are checked to
    float noise (abs=1e-9), not bit-exact equality.  Every accepted
    (here: cost-free) swap still runs the subtract-then-add round trip on
    the histogram bins, and `fl(fl(h - a) + a) == h` is NOT a floating-
    point identity in general -- subtracting a term much smaller than the
    running bin total `h` loses low-order bits that re-adding it does not
    recover.  Verified empirically across eight different orientation
    seeds: the residual is always at the ~1e-14 RELATIVE level (chi2 is
    O(1)), several orders of magnitude under this tolerance but never
    exactly zero -- and it is large enough that `best_perm` can end up
    very slightly (float-noise-level) better than chi2_initial, so it is
    NOT asserted to equal np.arange(n) here the way TEST 6's is.
    """
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    base2 = random_uniform(2, _rng(31))
    quats2 = np.tile(base2, (n // 2, 1))
    assert odf.orientation_classes(quats2)[1] == 2
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)

    res = anneal_assignment(quats2, pairs, pair_areas, edges, target, ref,
                            sym, n_steps=3000, t0=0.05, cooling=0.995,
                            rng=_rng(2), grain_volumes=grain_volumes,
                            odf_drift_max=0.0)
    assert res.chi2_final == pytest.approx(res.chi2_initial, abs=1e-9)
    assert res.odf_drift_final == pytest.approx(0.0, abs=1e-9)


def test_trust_region_chi2_vs_drift_tradeoff():
    """The scientifically interesting result: odf_drift_max trades off
    against how far the annealer can pull the misorientation histogram
    toward its target.  Sweeping the cap over the TEST 1/_drift_setup
    configuration gives (observed reference values, for sanity-checking a
    first run -- only the ORDERING below is asserted):

        cap 0.0   -> drift 0.000000, chi2 0.36209   (= chi2_initial, frozen)
        cap 0.01  -> drift 0.009519, chi2 0.21427
        cap 0.05  -> drift 0.049730, chi2 0.14260
        cap 0.2   -> drift 0.197464, chi2 0.08707
        cap None  -> drift 0.466748, chi2 0.05880

    A cap of 0.01 -- one percent of the volume-weighted ODF's total-
    variation mass -- already recovers about half (~49%: (0.36209 -
    0.21427) / (0.36209 - 0.05880)) of the total achievable chi2
    reduction.  Bounding the ODF drift does not cripple the histogram
    shaping: that is the whole point of offering a trust region instead
    of simply admitting the drift after the fact.
    """
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)

    caps = [0.0, 0.01, 0.05, 0.2, None]
    drifts = []
    chi2s = []
    for cap in caps:
        res = anneal_assignment(
            quats, pairs, pair_areas, edges, target, ref, sym,
            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
            grain_volumes=grain_volumes, odf_drift_max=cap)
        drifts.append(res.odf_drift_final)
        chi2s.append(res.chi2_final)

    for prev, nxt in zip(drifts, drifts[1:], strict=False):
        assert nxt >= prev - 1e-12          # non-decreasing drift
    for prev, nxt in zip(chi2s, chi2s[1:], strict=False):
        assert nxt <= prev + 1e-12          # non-increasing chi2


def test_trust_region_drift_final_describes_returned_best_perm():
    """TEST 8 -- odf_drift_final describes the RETURNED best_perm, not the
    walk's last state (tracker.value): recomputed independently from
    res.permutation with odf.orientation_weights + odf.atomic_drift."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    res = anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
                            grain_volumes=grain_volumes, odf_drift_max=None)

    class_of, n_classes = odf.orientation_classes(quats)
    v_best = odf.orientation_weights(res.permutation, grain_volumes)
    v_ref = odf.orientation_weights(np.arange(n), grain_volumes)
    independent = odf.atomic_drift(v_best, v_ref, class_of, n_classes)
    assert res.odf_drift_final == pytest.approx(independent, abs=1e-12)


def test_trust_region_error_paths():
    """TEST 9 -- each of these is a ConfigError, and each must be raised
    regardless of the (otherwise valid) adjacency/annealing arguments."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    kwargs = dict(n_steps=10, t0=0.05, cooling=0.995)

    with pytest.raises(ConfigError, match="grain_volumes"):
        anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                          rng=_rng(1), grain_volumes=None,
                          odf_drift_max=0.1, **kwargs)

    with pytest.raises(ConfigError, match="odf_drift_max"):
        anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                          rng=_rng(1), grain_volumes=grain_volumes,
                          odf_drift_max=-0.1, **kwargs)

    with pytest.raises(ConfigError, match="grain_volumes"):
        anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                          rng=_rng(1), grain_volumes=grain_volumes[:-1],
                          odf_drift_max=0.1, **kwargs)

    bad_volumes = grain_volumes.copy()
    bad_volumes[0] = 0.0
    with pytest.raises(ConfigError):
        anneal_assignment(quats, pairs, pair_areas, edges, target, ref, sym,
                          rng=_rng(1), grain_volumes=bad_volumes,
                          odf_drift_max=0.1, **kwargs)


def test_trust_region_drift_is_monotone_in_cap():
    """TEST 10 -- monotonicity sanity: a tighter cap can only ever report
    equal-or-smaller final drift (ties allowed, e.g. if a tight cap
    already stops all cross-class moves)."""
    sym, n, quats, pairs, pair_areas, edges, target, ref = _drift_setup()
    grain_volumes = _rng(55).lognormal(0.0, 0.8, n)
    unconstrained = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym,
        n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
        grain_volumes=grain_volumes, odf_drift_max=None)
    full_drift = unconstrained.odf_drift_final

    drifts = []
    for cap in (0.25 * full_drift, 0.5 * full_drift, None):
        res = anneal_assignment(
            quats, pairs, pair_areas, edges, target, ref, sym,
            n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(2),
            grain_volumes=grain_volumes, odf_drift_max=cap)
        drifts.append(res.odf_drift_final)

    for prev, nxt in zip(drifts, drifts[1:], strict=False):
        assert nxt >= prev - 1e-12


# ---------------------------------------------------------------------------
# Config validation (resolve Rules 13–15)
# ---------------------------------------------------------------------------


def _raw_config(**orientation):
    return {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [
                {"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": orientation,
    }


def test_resolve_components_required():
    with pytest.raises(ConfigError, match="components"):
        resolve_config(_raw_config(scheme="odf_components"))


def test_resolve_component_exactly_one_kind():
    with pytest.raises(ConfigError, match="exactly one"):
        resolve_config(_raw_config(
            scheme="odf_components",
            components=[{"euler_bunge_deg": [0, 0, 0], "random": True,
                         "weight": 1.0}]))


def test_resolve_single_random_component():
    with pytest.raises(ConfigError, match="at most one"):
        resolve_config(_raw_config(
            scheme="odf_components",
            components=[{"random": True, "weight": 0.5},
                        {"random": True, "weight": 0.5}]))


def test_resolve_spread_on_random_forbidden():
    with pytest.raises(ConfigError, match="spread_deg"):
        resolve_config(_raw_config(
            scheme="odf_components",
            components=[{"random": True, "weight": 1.0,
                         "spread_deg": 5.0}]))


def test_resolve_histogram_needs_table():
    with pytest.raises(ConfigError, match="bin_edges and densities"):
        resolve_config(_raw_config(
            scheme="random_uniform",
            mdf_target={"type": "histogram"}))


def test_resolve_histogram_edge_count():
    with pytest.raises(ConfigError, match="len"):
        resolve_config(_raw_config(
            scheme="random_uniform",
            mdf_target={"type": "histogram",
                        "bin_edges": [0.0, 30.0, 63.0],
                        "densities": [1.0, 1.0, 1.0]}))


def test_resolve_csl_enriched_needs_cubic():
    raw = _raw_config(scheme="random_uniform",
                      mdf_target={"type": "csl_enriched"})
    raw["crystal"] = {
        "space_group": {"number": 194},
        "lattice": {"a": 2.95, "c": 4.68},
        "wyckoff_sites": [
            {"element": "Ti", "coords": [1 / 3, 2 / 3, 0.25]}],
    }
    with pytest.raises(ConfigError, match="cubic"):
        resolve_config(raw)


# ---------------------------------------------------------------------------
# End-to-end (pipeline wiring, outputs, G12)
# ---------------------------------------------------------------------------


def _e2e_config(outdir, annealing_steps: int = 4000):
    return resolve_config({
        "meta": {"title": "odf+mdf e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": 4242},
        "box": {"lengths": [36.0, 36.0, 36.0]},
        "grains": {"number": 10},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [
                {"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {
            "scheme": "odf_components",
            "components": [
                {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 0.5,
                 "spread_deg": 3.0},
                # Σ3 twin partner of the first component (60° ⟨111⟩)
                {"euler_bunge_deg": [45.0, 70.528779365509308, 45.0],
                 "weight": 0.5, "spread_deg": 3.0},
            ],
            "mdf_target": {
                "type": "csl_enriched",
                "sigma3_fraction": 0.5,
                "annealing_steps": annealing_steps,
                "chi2_max": 1.9,   # warn threshold; e2e pins the row exists
            },
        },
        "analysis": {"csl": True, "mdf_bins": 30},
        "output": {"directory": str(outdir)},
    })


def test_e2e_odf_mdf_outputs_and_g12(tmp_path):
    import csv as _csv

    from grainsmith.io import MDF_COLUMNS
    from grainsmith.pipeline import run

    res = run(_e2e_config(tmp_path / "out"))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G12" in gate_ids and "G11" not in gate_ids

    out = res.outdir
    with (out / "mdf.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.reader(fh))
    assert rows[0] == MDF_COLUMNS
    assert len(rows) == 1 + 30                      # mdf_bins
    # target column filled when targeting is active
    assert all(r[4] != "" for r in rows[1:])

    mtex = (out / "odf_mtex.txt").read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in mtex if not ln.startswith("%")]
    assert len(data_lines) == 10                    # one row per grain
    assert any("loadOrientation_generic" in ln for ln in mtex)
    weights = np.array([float(ln.split()[3]) for ln in data_lines])
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    # summary rows: texture + mdf sections present
    summary = (out / "summary.csv").read_text(encoding="utf-8")
    assert "texture,component_sum_w2" in summary
    assert "mdf,chi2_final_annealer" in summary
    assert "G12" in summary


def test_e2e_annealing_raises_sigma3_area_fraction(tmp_path):
    """Σ3 AREA fraction (true CSL classes from boundaries analysis) rises
    with annealing vs the unannealed assignment — the R2 direction pin."""
    from grainsmith.pipeline import run

    def sigma3_fraction(res) -> float:
        a3 = sum(r.area_A2 for r in res.boundary_reports
                 if r.csl_sigma == "3")
        a_tot = sum(r.area_A2 for r in res.boundary_reports)
        return a3 / a_tot

    res0 = run(_e2e_config(tmp_path / "out0", annealing_steps=0))
    res1 = run(_e2e_config(tmp_path / "out1", annealing_steps=6000))
    assert sigma3_fraction(res1) > sigma3_fraction(res0)


def test_e2e_deterministic_rerun(tmp_path, monkeypatch):
    """Bit-identical mdf.csv and odf_mtex.txt on a re-run (§10).

    The SAME config (incl. the relative output.directory — it is hashed
    into the provenance line) run from two working directories.
    """
    import re

    from grainsmith.pipeline import run

    root1 = tmp_path / "run1"
    root2 = tmp_path / "run2"
    root1.mkdir()
    root2.mkdir()
    monkeypatch.chdir(root1)
    run(_e2e_config("./out"))
    monkeypatch.chdir(root2)
    run(_e2e_config("./out"))
    ts = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    for name in ("mdf.csv", "odf_mtex.txt", "boundaries.csv"):
        b1 = ts.sub(b"<TS>", (root1 / "out" / name).read_bytes())
        b2 = ts.sub(b"<TS>", (root2 / "out" / name).read_bytes())
        assert b1 == b2, f"{name} differs between identical runs"


# ---------------------------------------------------------------------------
# write_odf_mtex: atom_fraction column (G26 companion)
# ---------------------------------------------------------------------------


def _mtex_provenance():
    from grainsmith.io.common import Provenance

    return Provenance(
        version="9.9.9",
        timestamp_iso="2026-01-01T00:00:00Z",
        seed=42,
        config_sha256="a" * 64,
        provenance_sha256="b" * 64,
        title="odf_mtex unit test",
    )


def _mtex_quats():
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 30.0),
        axis_angle_to_quat(np.array([1.0, 1.0, 1.0]), 60.0),
    ])


def test_write_odf_mtex_atom_counts_none_matches_legacy_four_column(
        tmp_path):
    """``atom_counts=None`` must reproduce EXACTLY the file this function
    wrote before ``atom_fraction`` existed -- same 4 columns, same header
    lines, byte for byte. The expected content is built explicitly here
    (not by re-invoking ``write_odf_mtex`` and not from a stored snapshot)."""
    from grainsmith.io.common import fmt
    from grainsmith.io.texture import write_odf_mtex
    from grainsmith.orientation.quaternion import quat_to_bunge

    quats = _mtex_quats()
    volumes = np.array([100.0, 300.0, 600.0])
    weights = volumes / volumes.sum()
    prov = _mtex_provenance()

    path = tmp_path / "odf_mtex.txt"
    write_odf_mtex(path, quats, weights, prov)

    expected_lines = [
        f"% {prov.line()}",
        "% per-grain Bunge Euler angles (degrees) + "
        "volume-fraction weights",
        "% MTEX import:",
        "%   ori = loadOrientation_generic('odf_mtex.txt', "
        "'CS', cs, 'ColumnNames', "
        "{'phi1' 'Phi' 'phi2' 'weight'}, 'Bunge', 'degree');",
        "% phi1 Phi phi2 weight",
    ]
    for q, wgt in zip(quats, weights, strict=True):
        phi1, Phi, phi2 = quat_to_bunge(q)
        expected_lines.append(
            f"{fmt(phi1)} {fmt(Phi)} {fmt(phi2)} {fmt(wgt)}")
    expected = "\n".join(expected_lines) + "\n"

    assert path.read_text(encoding="utf-8") == expected


def test_write_odf_mtex_atom_counts_none_equals_omitted_default(tmp_path):
    """Passing ``atom_counts=None`` explicitly and omitting the argument
    entirely must write identical files (the parameter default is None)."""
    from grainsmith.io.texture import write_odf_mtex

    quats = _mtex_quats()
    volumes = np.array([100.0, 300.0, 600.0])
    weights = volumes / volumes.sum()
    prov = _mtex_provenance()

    p1 = tmp_path / "explicit_none.txt"
    p2 = tmp_path / "omitted.txt"
    write_odf_mtex(p1, quats, weights, prov, atom_counts=None)
    write_odf_mtex(p2, quats, weights, prov)
    assert p1.read_bytes() == p2.read_bytes()


def test_write_odf_mtex_with_atom_counts_adds_fifth_column(tmp_path):
    """With non-degenerate atom counts: 5 columns, atom_fraction sums to 1,
    and the two weight columns differ (atom counts not proportional to
    volumes)."""
    from grainsmith.io.texture import write_odf_mtex

    quats = _mtex_quats()
    volumes = np.array([100.0, 300.0, 600.0])
    weights = volumes / volumes.sum()
    atom_counts = np.array([500.0, 500.0, 9000.0])  # NOT proportional
    prov = _mtex_provenance()

    path = tmp_path / "odf_mtex.txt"
    write_odf_mtex(path, quats, weights, prov, atom_counts=atom_counts)

    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("%")]
    assert len(data_lines) == 3
    rows = [ln.split() for ln in data_lines]
    assert all(len(r) == 5 for r in rows)

    w_col = np.array([float(r[3]) for r in rows])
    af_col = np.array([float(r[4]) for r in rows])
    assert af_col.sum() == pytest.approx(1.0, abs=1e-12)
    assert w_col.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.array_equal(af_col, atom_counts / atom_counts.sum())
    # the two weightings actually differ, grain by grain
    assert not np.allclose(w_col, af_col)

    # header explains which column is which and why they differ
    header = "\n".join(ln for ln in lines if ln.startswith("%"))
    assert "weight" in header and "atom_fraction" in header
    assert "G26" in header
    assert "surface" in header.lower() or "volume" in header.lower()


def test_write_odf_mtex_all_zero_atom_counts_uses_four_column_path(
        tmp_path):
    """All-zero atom counts (a run whose atom block was omitted) must take
    the 4-column path, not raise or divide by zero."""
    from grainsmith.io.texture import write_odf_mtex

    quats = _mtex_quats()
    volumes = np.array([100.0, 300.0, 600.0])
    weights = volumes / volumes.sum()
    prov = _mtex_provenance()

    path_zero = tmp_path / "zero.txt"
    path_none = tmp_path / "none.txt"
    write_odf_mtex(path_zero, quats, weights, prov,
                   atom_counts=np.zeros(3))
    write_odf_mtex(path_none, quats, weights, prov, atom_counts=None)

    assert path_zero.read_bytes() == path_none.read_bytes()
    lines = path_zero.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("%")]
    assert all(len(ln.split()) == 4 for ln in data_lines)


def _parse_mtex_header_column_names(lines: list[str]) -> list[str]:
    """Extract the ``ColumnNames`` list from the MTEX import one-liner."""
    for ln in lines:
        if "ColumnNames" in ln:
            inside = ln.split("{", 1)[1].split("}", 1)[0]
            return inside.replace("'", "").split()
    raise AssertionError("no ColumnNames line found in header")


def _parse_mtex_legend_columns(lines: list[str]) -> list[str]:
    """Extract the ``% phi1 Phi phi2 weight [atom_fraction]`` legend row --
    the last ``%`` line, immediately preceding the data rows."""
    comment_lines = [ln for ln in lines if ln.startswith("%")]
    legend = comment_lines[-1]
    return legend.lstrip("%").split()


@pytest.mark.parametrize("with_atoms", [False, True])
def test_write_odf_mtex_header_matches_written_columns(tmp_path, with_atoms):
    """The legend line and the MTEX ColumnNames list must agree with the
    ACTUAL number of columns written, in both 4- and 5-column modes --
    parsed from the file, never hardcoded."""
    from grainsmith.io.texture import write_odf_mtex

    quats = _mtex_quats()
    volumes = np.array([100.0, 300.0, 600.0])
    weights = volumes / volumes.sum()
    prov = _mtex_provenance()
    atom_counts = np.array([500.0, 500.0, 9000.0]) if with_atoms else None

    path = tmp_path / "odf_mtex.txt"
    write_odf_mtex(path, quats, weights, prov, atom_counts=atom_counts)
    lines = path.read_text(encoding="utf-8").splitlines()

    data_lines = [ln for ln in lines if not ln.startswith("%")]
    n_written = len(data_lines[0].split())

    column_names = _parse_mtex_header_column_names(lines)
    legend_columns = _parse_mtex_legend_columns(lines)

    assert len(column_names) == n_written
    assert len(legend_columns) == n_written
    assert column_names == legend_columns
    expected_cols = (["phi1", "Phi", "phi2", "weight", "atom_fraction"]
                     if with_atoms else ["phi1", "Phi", "phi2", "weight"])
    assert legend_columns == expected_cols
