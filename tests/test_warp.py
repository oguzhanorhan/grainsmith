"""Tests for tessellation/warp.py — §10 test_warp.py.

The previous version of this file passed with weakened assertions (loose
RMS band, no continuity check, blanket ``pytest.raises(Exception)``); these
tests pin the actual §6.6 contracts.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.constants import WARP_GRAD_MAX
from grainsmith.errors import ConfigError, TessellationError
from grainsmith.seeding import seed_grains, wigner_seitz_radius
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.warp import (
    WarpTessellation,
    amplitude_to_reference_wavelength,
    amplitude_to_total_rms,
    reference_shell_kappa,
    synthesize_grf,
)


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _base_tess(n=5, L=50.0, seed=1):
    box = np.array([L, L, L])
    rng = _rng(seed)
    seeds = seed_grains(n, box, [True, True, True], rng)
    return FlatTessellation(seeds, box, [True, True, True]), box


def _make_warp(n=5, L=50.0, amplitude=1.0, corr=15.0, seed=7, grid_size=32):
    base, box = _base_tess(n=n, L=L, seed=seed)
    r_ws = wigner_seitz_radius(float(L**3), n)
    warp = WarpTessellation(
        base=base,
        box_lengths=box,
        periodic=[True, True, True],
        amplitude=amplitude,
        correlation_length=corr,
        min_seed_distance=r_ws,
        rng=_rng(seed + 100),
        grid_size=grid_size,
    )
    return warp, base, box


# ------------------------------------------------------------- GRF field

def test_grf_unit_rms_exact():
    """synthesize_grf returns exactly unit-RMS components (the calibration
    is an explicit normalization, not an approximation)."""
    field = synthesize_grf((16, 16, 16), 8.0, np.array([30.0, 30.0, 30.0]),
                           _rng(5))
    assert field.shape == (3, 16, 16, 16)
    for alpha in range(3):
        assert abs(float(np.std(field[alpha])) - 1.0) < 1e-12


def test_warp_field_periodicity_continuity():
    """True wrap-around continuity: the interpolated displacement at x = 0
    and x = L (same torus point) is identical on every periodic axis."""
    warp, _, box = _make_warp(amplitude=1.0, corr=15.0)
    rng = _rng(17)
    pts = rng.uniform(0.0, 1.0, size=(50, 3)) * box
    for ax in range(3):
        lo = pts.copy()
        lo[:, ax] = 0.0
        hi = pts.copy()
        hi[:, ax] = box[ax]
        u_lo = warp._interpolate_u(lo)
        u_hi = warp._interpolate_u(hi)
        np.testing.assert_allclose(u_lo, u_hi, atol=1e-12,
                                   err_msg=f"seam discontinuity on axis {ax}")


def test_warp_rms_calibration_tight():
    """Delivered per-component RMS equals the requested amplitude exactly
    when the tail clip is inactive (amplitude well below A_clip)."""
    amplitude = 0.5
    warp, _, _ = _make_warp(amplitude=amplitude, corr=15.0)
    for alpha in range(3):
        rms = float(np.std(warp.field[alpha]))
        assert abs(rms - amplitude) < 1e-9, f"component {alpha}: rms={rms}"


# ------------------------------------------------------------- G6 guards

def test_amplitude_exceeding_aclip_raises():
    """Requested amplitude > min_seed_distance/4 is a hard ConfigError
    (seed-containment guard) — not a silent clip."""
    base, box = _base_tess(n=5, L=50.0, seed=2)
    r_ws = wigner_seitz_radius(50.0**3, 5)
    with pytest.raises(ConfigError):
        WarpTessellation(
            base=base, box_lengths=box, periodic=[True, True, True],
            amplitude=50.0, correlation_length=5.0,
            min_seed_distance=r_ws, rng=_rng(99), grid_size=16,
        )


def test_bijectivity_guard_trips():
    """Admissible amplitude but aggressive amplitude/ℓ ratio trips the
    max‖∇u‖ < 0.5 bijectivity guard with TessellationError specifically."""
    base, box = _base_tess(n=5, L=50.0, seed=2)
    r_ws = wigner_seitz_radius(50.0**3, 5)
    amplitude = 0.9 * r_ws / 4.0   # below A_clip → passes guard 1
    with pytest.raises(TessellationError):
        WarpTessellation(
            base=base, box_lengths=box, periodic=[True, True, True],
            amplitude=amplitude, correlation_length=3.0,
            min_seed_distance=r_ws, rng=_rng(99), grid_size=32,
        )


def test_warp_grad_max_below_limit():
    """Moderate parameters satisfy the bijectivity bound.

    amplitude retuned 2.0 -> 1.0 (warp.py DC-mode-exclusion fix: excluding
    k=0 from the gaussian spectrum removes a rigid-translation contribution
    that used to inflate the RMS calibration denominator, so the same
    amplitude now delivers a larger max‖∇u‖ for a fixed correlation_length).
    """
    warp, _, _ = _make_warp(amplitude=1.0, corr=15.0, seed=11)
    assert warp.grad_max < WARP_GRAD_MAX


# ------------------------------------------------------------- membership

def test_warp_adjacency_equals_base():
    """Bijective warp inherits the base adjacency exactly (§6.6 topology).

    amplitude retuned 2.0 -> 1.0 (warp.py DC-mode-exclusion fix, see
    test_warp_grad_max_below_limit above) to keep the warp bijective.
    """
    warp, base, _ = _make_warp(amplitude=1.0, corr=15.0, seed=3)
    assert set(base.adjacency()) == set(warp.adjacency())


def test_warp_grain_of_returns_valid_ids():
    warp, _, box = _make_warp(n=6)
    pts = _rng(0).uniform(0.0, 1.0, size=(100, 3)) * box
    gids = warp.grain_of(pts)
    assert np.all(gids >= 0) and np.all(gids < 6)


def test_warp_owns_tiles_torus():
    """Each torus point is owned exactly once over all grains and all
    {-1,0,1}³ periodic lifts under the warp (bijective warp preserves the
    exact-tiling invariant of §6.8).

    amplitude retuned 1.5 -> 1.0 (warp.py DC-mode-exclusion fix, see
    test_warp_grad_max_below_limit above) to keep the warp bijective.
    """
    warp, _, box = _make_warp(n=5, amplitude=1.0, corr=15.0, seed=9)
    pts = _rng(21).uniform(0.0, 1.0, size=(100, 3)) * box
    owned = np.zeros(len(pts), dtype=int)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * box
                for i in range(warp.n_grains):
                    owned += warp.owns(lift, i).astype(int)
    assert np.all(owned == 1), (
        f"warped torus tiling violated: min={owned.min()}, max={owned.max()}"
    )


def test_warp_g5_connectivity_wired():
    """G5 runs at construction: the voxel grid exists and reports every
    grain singly connected; connectivity_check=False opts out."""
    warp, _, _ = _make_warp(amplitude=1.0, corr=15.0, seed=5)
    vg = warp.voxel_grid
    assert vg is not None
    results = vg.check_connectivity([True, True, True])
    assert all(results.values()) and len(results) == warp.n_grains

    base, box = _base_tess(n=5, L=50.0, seed=5)
    r_ws = wigner_seitz_radius(50.0**3, 5)
    warp_nc = WarpTessellation(
        base=base, box_lengths=box, periodic=[True, True, True],
        amplitude=1.0, correlation_length=15.0, min_seed_distance=r_ws,
        rng=_rng(105), grid_size=32, connectivity_check=False,
    )
    assert warp_nc.voxel_grid is None


def test_warp_noncubic_grid_shape():
    """Grid resolution is set by the SHORTEST edge and scales per axis —
    a 2:1:1 box gets a 2:1:1 grid (AUDIT C5 regression: the old code used
    L[0] as the reference and under-resolved non-cubic boxes 4×)."""
    box = np.array([80.0, 40.0, 40.0])
    rng = _rng(31)
    seeds = seed_grains(4, box, [True, True, True], rng)
    base = FlatTessellation(seeds, box, [True, True, True])
    r_ws = wigner_seitz_radius(float(np.prod(box)), 4)
    warp = WarpTessellation(
        base=base, box_lengths=box, periodic=[True, True, True],
        amplitude=0.5, correlation_length=10.0,
        min_seed_distance=r_ws, rng=_rng(32), grid_size=16,
    )
    assert warp._grid_shape == (32, 16, 16)


def test_grad_max_free_axis_uses_one_sided_stencil():
    """The G6 bijectivity guard must measure the FREE-axis boundary gradient
    with a one-sided stencil — matching _interpolate_u, which clamps there — not
    a periodic wrap that smears the free-surface slope across the wall.  The
    all-roll stencil underestimates a genuine free-surface gradient spike and can
    silently pass a non-bijective warp (a slab/thin-film + warp is a valid,
    resolve.py-permitted combination)."""
    N = (4, 4, 6)
    L = np.array([40.0, 40.0, 60.0])
    periodic = [True, True, False]              # z is the free (slab) axis
    field = np.zeros((3,) + N, dtype=np.float64)
    field[2, :, :, 1] = 5.0                     # steep step at the z=0 wall only

    w = WarpTessellation.__new__(WarpTessellation)
    w._field = field
    w._grid_shape = N
    w._L = L
    w._periodic = periodic
    got = w._compute_grad_max()

    hz = L[2] / N[2]
    expected_free = float(np.max(np.abs(np.gradient(field[2], hz, axis=2))))
    roll_z = (np.roll(field[2], -1, axis=2)
              - np.roll(field[2], 1, axis=2)) / (2 * hz)
    all_roll = float(np.max(np.abs(roll_z)))
    assert expected_free > all_roll             # the discrepancy the fix closes
    assert got == pytest.approx(expected_free)


# ---------------------------------------------------------------------------
# reference_shell_kappa / amplitude convention conversion
# ---------------------------------------------------------------------------


def test_reference_shell_kappa_matches_validated_s1_campaign_point():
    """Pins kappa at the independent feasibility prototype's own S1
    campaign point (H=0.7, l_min=12, l_max=40 Å, on the 192**3 grid /
    L=246 A box the prototype used) to its previously-validated numeric
    value kappa=0.8753 (from the feasibility report) -- an exact discrete
    computation, not the cheaper closed-form continuum approximation
    (see reference_shell_kappa's own docstring for why both exist)."""
    kappa = reference_shell_kappa(
        hurst=0.7, l_min=12.0, l_max=40.0,
        grid_shape=(192, 192, 192),
        box_lengths=np.array([246.0, 246.0, 246.0]),
    )
    assert kappa == pytest.approx(0.8753, abs=5e-4)

    # The prototype's own worked example: amplitude_total_rms=3.978584 ->
    # A0 (reference_wavelength convention) = 3.4825.
    a0 = amplitude_to_reference_wavelength(3.978584, kappa)
    assert a0 == pytest.approx(3.4825, abs=5e-4)


def test_amplitude_conversion_round_trip_both_directions():
    """amplitude_to_reference_wavelength / amplitude_to_total_rms are
    exact inverses of one another for any kappa in (0, 1] -- the mapping
    this method's seed-containment guard re-check (config/resolve.py
    Rule 29(d), PerturbedDistanceTessellation.__init__) depends on being
    reversible without drift."""
    kappa = reference_shell_kappa(hurst=0.6, l_min=10.0, l_max=55.0)
    for amplitude_total in (0.0, 0.5, 3.978584, 21.4):
        a0 = amplitude_to_reference_wavelength(amplitude_total, kappa)
        back = amplitude_to_total_rms(a0, kappa)
        assert back == pytest.approx(amplitude_total, rel=1e-12)
    for amplitude_ref in (0.0, 1.0, 14.35):
        total = amplitude_to_total_rms(amplitude_ref, kappa)
        back = amplitude_to_reference_wavelength(total, kappa)
        assert back == pytest.approx(amplitude_ref, rel=1e-12)


def test_reference_shell_kappa_is_one_when_reference_spans_whole_band():
    """kappa = 1 exactly when the reference octave IS the whole band
    (l_max/l_min = 2, i.e. exactly one octave) -- the whole-band and
    reference-octave shells coincide, so sigma_reference/sigma_total = 1
    regardless of Hurst."""
    for hurst in (0.3, 0.6, 0.95):
        kappa = reference_shell_kappa(hurst, l_min=10.0, l_max=20.0)
        assert kappa == pytest.approx(1.0, abs=1e-12)


def test_reference_shell_kappa_le_one_and_decreases_with_wider_band():
    """kappa <= 1 always (a single reference octave's shell variance
    cannot exceed the whole band's), and, for a fixed reference anchor,
    kappa shrinks monotonically as the band widens around it -- more
    octaves means the reference shell is a smaller slice of the total
    band variance."""
    hurst = 0.7
    l_min = 8.0
    kappas = [reference_shell_kappa(hurst, l_min, l_max, reference_wavelength=l_min * 2)
              for l_max in (16.0, 32.0, 64.0, 128.0)]
    assert all(k <= 1.0 + 1e-12 for k in kappas)
    # strict=False is deliberate: this is the adjacent-pairs idiom, so the two
    # operands differ in length by exactly one by construction.
    assert all(k1 >= k2 for k1, k2 in zip(kappas, kappas[1:], strict=False)), kappas


def test_reference_shell_kappa_closed_form_vs_discrete_agree_approximately():
    """The cheap closed-form (grid-independent) continuum kappa used for
    resolve.py's pre-flight guard check agrees with the exact discrete
    (actual-FFT-grid) kappa PerturbedDistanceTessellation's constructor
    could in principle compute, to within a few percent -- documented in
    reference_shell_kappa's own docstring as an intentional
    "sufficient, not tight" pre-flight approximation, not a promise of
    bit-exact agreement."""
    kappa_closed = reference_shell_kappa(0.7, 12.0, 40.0)
    kappa_discrete = reference_shell_kappa(
        0.7, 12.0, 40.0,
        grid_shape=(192, 192, 192), box_lengths=np.array([246.0, 246.0, 246.0]),
    )
    assert kappa_closed == pytest.approx(kappa_discrete, rel=0.03)
