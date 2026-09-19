"""Tests for the self-affine spectral synthesis math, and for the
removal of the 'warp' + spectrum 'self_affine' combination.

History / scope of this file
-----------------------------
``warp`` used to accept BOTH spectra and own gate G13
(Hurst back-estimation). That combination was REMOVED (not
deprecated): 'warp' is a coordinate diffeomorphism
(grain_of(x) = base.grain_of(x + u(x)), bijective under the G6 guard),
and a bijective map cannot change the box-counting dimension of
the flat-Voronoi surface it displaces — so giving its field a self-affine
PSD only recolors the waviness of a boundary whose box-counting dimension
it cannot change (measured directly: field correlation 0.995/0.975
narrow/wide-band between H=0.5 and H=0.9, cross-section grain-assignment
difference < 0.1 %; see ``docs/physics.md`` §5b for the full argument).
``perturbed_distance`` (``tessellation/perturbed.py``) is now the ONLY method that accepts
spectrum: self_affine and the only owner of gate G13 (paired with its
own G20).

What lives where now
---------------------
- The spectrum-SYNTHESIS MATH (``synthesize_grf``/``estimate_hurst``,
  free functions in ``tessellation/warp.py`` shared verbatim by
  ``perturbed.py`` with ``n_components=1``) is backend-agnostic and its
  tests are UNCHANGED below — they never constructed a
  ``WarpTessellation`` in the first place.
- Tests that used to construct ``WarpTessellation(..., spectrum=
  "self_affine")`` to exercise G6/Nyquist/seam-continuity/hurst_estimate
  physics are GONE from this file: those mechanisms either don't exist
  for perturbed_distance (G6 bijectivity) or are already independently
  tested there (grid-Nyquist: ``test_perturbed.py::
  test_l_min_below_grid_nyquist_raises``; D_b-vs-Hurst response:
  ``test_perturbed.py::test_db_monotonic_in_hurst_direction``;
  hurst_estimate property: ``test_perturbed.py::
  test_hurst_estimate_property``).
- What replaces them here is the contract: warp+self_affine raises
  ConfigError, at both the config-resolve level and the constructor
  (API) level — see the section below.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.warp import (
    WarpTessellation,
    estimate_hurst,
    synthesize_grf,
)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


L80 = np.array([80.0, 80.0, 80.0])
PER = [True, True, True]


# ---------------------------------------------------------------------------
# Spectrum synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hurst", [0.5, 0.8])
def test_psd_slope_matches_target(hurst):
    """Radially averaged PSD of the synthesized field follows
    k^(−(3+2H)): the back-estimated Hurst matches within fit tolerance
    (fixed seed)."""
    field = synthesize_grf((80, 80, 80), 15.0, L80, _rng(),
                           spectrum="self_affine", hurst=hurst,
                           l_min=8.0, l_max=40.0)
    h_est = estimate_hurst(field, L80, 8.0, 40.0)
    assert abs(h_est - hurst) < 0.1, f"H={hurst}: estimated {h_est:.3f}"


def test_gaussian_default_bit_identical():
    """spectrum default unchanged ⇒ v1 gaussian fields bit-identical
    (regression pin: same rng consumption, same code path)."""
    f1 = synthesize_grf((32, 32, 32), 15.0, L80, _rng(7))
    f2 = synthesize_grf((32, 32, 32), 15.0, L80, _rng(7),
                        spectrum="gaussian")
    np.testing.assert_array_equal(f1, f2)


def test_band_limits_enforced_in_kspace():
    """No spectral power outside [2π/l_max, 2π/l_min]; zero mean."""
    field = synthesize_grf((64, 64, 64), 15.0, L80, _rng(3),
                           spectrum="self_affine", hurst=0.8,
                           l_min=10.0, l_max=40.0)
    from grainsmith.tessellation.warp import _k_magnitude
    k = _k_magnitude((64, 64, 64), L80)
    out_of_band = (k < 2 * np.pi / 40.0) | (k > 2 * np.pi / 10.0)
    for alpha in range(3):
        spec = np.abs(np.fft.fftn(field[alpha]))
        # FFT round-trip noise only outside the band
        assert float(np.max(spec[out_of_band])) < 1e-9 * float(np.max(spec))
        assert abs(float(np.mean(field[alpha]))) < 1e-12


def test_empty_band_raises():
    # 4³ grid on the 80 Å box: |k| ∈ 0.0785·{1, √2, √3, 2, …} — the band
    # (0.0824, 0.102) rad/Å falls in the gap between 1 and √2 modes.
    with pytest.raises(ConfigError, match="no Fourier mode"):
        synthesize_grf((4, 4, 4), 15.0, L80, _rng(),
                       spectrum="self_affine", hurst=0.8,
                       l_min=61.6, l_max=76.2)


# ---------------------------------------------------------------------------
# WarpTessellation + self_affine: REMOVED. Both the API level
# (WarpTessellation.__init__) and the config-resolve level (Rule 8a) reject
# the combination unconditionally -- checked here at BOTH levels, since
# tests/any direct caller can construct a WarpTessellation without going
# through resolve_config (the API guard exists precisely for that reason).
# The physics tests that used to live in this section assumed the
# combination was legal; they are gone from here because the combination no
# longer exists. Their fate, checked against tests/test_perturbed.py's
# actual current content rather than assumed:
#   - grid-Nyquist (l_min too small for the field grid): a real analogue
#     exists, test_l_min_below_grid_nyquist_raises.
#   - hurst_estimate() honest refusal on a non-self_affine spectrum: a real
#     analogue exists, test_hurst_estimate_property.
#   - the OLD G6-trips-on-l_min test (bijectivity guard tripping because a
#     narrow self_affine band forces a large local gradient) has NO
#     analogue: perturbed_distance has no G6/bijectivity concept at all (no
#     diffeomorphism to be non-bijective) -- its amplitude ceiling is the
#     unrelated seed-containment guard, already covered by
#     test_amplitude_exceeding_a_max_raises_at_{construction,resolve}.
#   - seam/periodicity continuity across the box boundary has NO analogue in
#     test_perturbed.py either; if that physics matters for
#     perturbed_distance specifically it remains an open test-coverage gap.
# ---------------------------------------------------------------------------


def _base(n: int = 8, seed: int = 11) -> FlatTessellation:
    seeds = _rng(seed).random((n, 3)) * L80
    return FlatTessellation(seeds, L80, PER)


def test_warp_self_affine_raises_at_construction_api_level():
    """WarpTessellation.__init__ rejects spectrum='self_affine'
    unconditionally -- regardless of amplitude/l_min/l_max, and even
    with connectivity_check=False (i.e. this is NOT a G6/Nyquist guard
    that happens to trip; it is checked before those guards even run)."""
    with pytest.raises(ConfigError, match="only accepts spectrum='gaussian'"):
        WarpTessellation(
            _base(), L80, PER, amplitude=0.4, correlation_length=15.0,
            min_seed_distance=20.0, rng=_rng(2), connectivity_check=False,
            spectrum="self_affine", hurst=0.8, l_min=10.0, l_max=40.0,
        )


def test_warp_self_affine_raises_at_construction_regardless_of_guard_state():
    """Same rejection even when every OTHER guard would pass fine (small
    amplitude, generous l_min/l_max, matching the gaussian-legal defaults
    for amplitude/correlation_length) -- self_affine is refused on its own
    terms, not as a side effect of tripping G6 or the Nyquist check."""
    with pytest.raises(ConfigError, match="only accepts spectrum='gaussian'"):
        WarpTessellation(
            _base(), L80, PER, amplitude=1.0, correlation_length=15.0,
            min_seed_distance=20.0, rng=_rng(1), connectivity_check=False,
            grid_size=64,
            spectrum="self_affine", hurst=0.5, l_min=10.0, l_max=30.0,
        )


def test_warp_gaussian_still_constructs_fine():
    """Negative control: the guard is spectrum-specific, not a blanket
    regression -- the default (gaussian) spectrum is completely
    unaffected and still constructs normally."""
    tess = WarpTessellation(
        _base(), L80, PER, amplitude=0.4, correlation_length=15.0,
        min_seed_distance=20.0, rng=_rng(4), connectivity_check=False,
    )
    assert tess.spectrum == "gaussian"


# ---------------------------------------------------------------------------
# Config rules (resolve Rule 8a + Rule 20)
# ---------------------------------------------------------------------------


def _raw(method="perturbed_distance", **curved):
    """Default method is 'perturbed_distance' (not 'warp' as before this
    task) -- it is now the only method for which self_affine resolves
    successfully, so it is the natural default for the l_min/l_max band
    tests (Rule 20) below. Tests that specifically need method='warp' to
    probe Rule 8a's rejection pass it explicitly."""
    blk = {"method": method, "amplitude": 1.0, "spectrum": "self_affine",
           "hurst": 0.8, "l_min": 8.0, "l_max": 30.0}
    blk.update(curved)
    return {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [80.0, 80.0, 80.0]},
        "grains": {"number": 8},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0, 0, 0]}],
        },
        "boundaries": {"geometry": "curved", "curved": blk},
    }


def test_rule8a_warp_self_affine_raises_at_resolve():
    """Rule 8a (resolve-level companion to the API-level
    guard in WarpTessellation.__init__ tested above): method='warp' +
    spectrum='self_affine' is a hard ConfigError at resolve_config time,
    named specifically (not merely swept into Rule 20's generic
    method-restriction message below) -- checked BEFORE Rule 20 is even
    reached (Rule 8a lives earlier in resolve.py, inside the
    ``if curved.method == "warp":`` block)."""
    with pytest.raises(ConfigError,
                       match="'warp' does not accept spectrum 'self_affine'"):
        resolve_config(_raw(method="warp", amplitude=1.0))


def test_rule20_band_order():
    with pytest.raises(ConfigError, match="l_min < l_max"):
        resolve_config(_raw(l_min=30.0, l_max=8.0))


def test_rule20_l_max_box_cap():
    with pytest.raises(ConfigError, match="min\\(box.lengths\\)/2"):
        resolve_config(_raw(l_max=50.0))


def test_rule20_perturbed_distance_only():
    """Rule 20 (restricted to perturbed_distance only -- see the module
    docstring): once Rule 8a has already ruled out method=='warp'
    upstream, Rule 20's OWN method-restriction check (reached for the
    remaining curved methods) has exactly one self_affine-capable method
    left: perturbed_distance. additive_weights has no spectral field at
    all and must still be rejected -- same restriction as before, with
    an updated error-message expectation (see
    test_perturbed.py::test_self_affine_spectrum_method_restriction_names_perturbed_distance_only)."""
    with pytest.raises(ConfigError, match="perturbed_distance method only"):
        resolve_config(_raw(method="additive_weights", weight_sigma=1.0))


# ---------------------------------------------------------------------------
# End-to-end: self-affine spectrum + G13 (now perturbed_distance-owned)
# ---------------------------------------------------------------------------


def _e2e_config(outdir):
    # amplitude is deliberately well under perturbed_distance's own
    # seed-containment ceiling (A_max = PERTURBED_DISTANCE_SAFETY *
    # min_seed_distance / (2*ETA_CLIP)) for this box/grain count -- see
    # test_perturbed.py's _geometry()/_make() helpers for the general
    # pattern; this config was validated end-to-end (measured directly)
    # to pass every gate before being pinned here.
    raw = _raw(method="perturbed_distance", amplitude=0.35,
               l_min=10.0, l_max=25.0, hurst=0.8)
    raw["meta"] = {"verbose": 0}
    raw["box"] = {"lengths": [60.0, 60.0, 60.0]}
    raw["grains"] = {"number": 8}
    raw["output"] = {"directory": str(outdir)}
    return resolve_config(raw)


def test_e2e_self_affine_with_g13(tmp_path):
    """The self_affine end-to-end story moved from 'warp' to
    'perturbed_distance' in Phase 3 -- G13 (Hurst back-estimate) now
    always appears ALONGSIDE G19 (repair-and-report severity) and G20
    (box-counting D_b), never alone, because they share one
    PerturbedDistanceTessellation construction."""
    from grainsmith.pipeline import run

    res = run(_e2e_config(tmp_path / "out"))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert {"G13", "G19", "G20"} <= gate_ids
    assert "G11" not in gate_ids and "G12" not in gate_ids
    assert "G6" in gate_ids  # recorded as n/a (no diffeomorphism to check)
    g6 = [r for r in res.gates.results() if r.gate == "G6"][0]
    assert "n/a" in g6.message

    g13 = [r for r in res.gates.results() if r.gate == "G13"][0]
    assert abs(float(g13.measured) - 0.8) < 0.15
    assert "ok" in g13.message      # within HURST_G13_TOL ⇒ no WARN

    summary = (res.outdir / "summary.csv").read_text(encoding="utf-8")
    assert "tessellation,method,perturbed_distance" in summary
    assert "tessellation,spectrum,self_affine" in summary
    assert "tessellation,hurst,0.8" in summary
    assert "tessellation,hurst_estimated," in summary
    assert "tessellation,d_b_estimated," in summary
    assert "tessellation,l_min_A,10" in summary


def test_e2e_gaussian_warp_has_no_g13(tmp_path):
    """Negative control: the default gaussian spectrum on 'warp' runs
    the v1 gate set exactly (no G13/G19/G20 row) -- warp can never
    populate any of those three any more."""
    from grainsmith.pipeline import run

    raw = _raw(method="warp")
    raw["boundaries"]["curved"] = {"method": "warp", "amplitude": 2.0,
                                   "correlation_length": 20.0}
    raw["meta"] = {"verbose": 0}
    raw["box"] = {"lengths": [60.0, 60.0, 60.0]}
    raw["grains"] = {"number": 8}
    raw["output"] = {"directory": str(tmp_path / "out")}
    res = run(resolve_config(raw))
    assert res.gates.all_passed()
    # G23 (report-only) fires for every single-phase run that reaches the
    # MDF-histogram block, independent of orientation.mdf_target. G26
    # (report-only) fires whenever per-grain atom counts exist.
    assert {r.gate for r in res.gates.results()} == \
        {f"G{k}" for k in range(1, 11)} | {"G23", "G26"}


def test_e2e_gaussian_perturbed_distance_has_no_g13(tmp_path):
    """Negative control, perturbed_distance side: its own DEFAULT
    spectrum (gaussian) runs without G13/G20 either -- those two are
    self_affine-gated, not perturbed_distance-gated (G19 still appears;
    it is unconditional repair-and-report, independent of spectrum)."""
    from grainsmith.pipeline import run

    raw = _raw(method="perturbed_distance", spectrum="gaussian",
               amplitude=0.3, correlation_length=15.0)
    raw["meta"] = {"verbose": 0}
    raw["box"] = {"lengths": [60.0, 60.0, 60.0]}
    raw["grains"] = {"number": 8}
    raw["output"] = {"directory": str(tmp_path / "out")}
    res = run(resolve_config(raw))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G19" in gate_ids
    assert "G13" not in gate_ids and "G20" not in gate_ids


# ---------------------------------------------------------------------------
# amplitude_convention / reference_wavelength cross-field rule (Rule 29)
# ---------------------------------------------------------------------------


def test_rule29a_reference_wavelength_forbidden_under_total_rms():
    """Rule 29(a): reference_wavelength set while amplitude_convention is
    still 'total_rms' (the default) is a ConfigError, not a silently-
    ignored field -- this project's extra='forbid' convention applied to
    a cross-field combination rather than a single unknown key."""
    with pytest.raises(ConfigError, match="reference_wavelength is set"):
        resolve_config(_raw(amplitude_convention="total_rms",
                            reference_wavelength=20.0))


def test_rule29b_reference_wavelength_convention_requires_self_affine():
    """Rule 29(b): amplitude_convention: reference_wavelength has no
    meaning for the gaussian spectrum (no octave/band structure to
    anchor a reference octave against)."""
    with pytest.raises(ConfigError, match="requires spectrum"):
        resolve_config(_raw(spectrum="gaussian", correlation_length=15.0,
                            amplitude_convention="reference_wavelength"))


def test_rule29c_reference_wavelength_must_lie_in_band():
    """Rule 29(c): an explicit reference_wavelength outside [l_min,
    l_max] is a ConfigError (both below l_min and above l_max)."""
    with pytest.raises(ConfigError, match=r"must lie within \[l_min, l_max\]"):
        resolve_config(_raw(amplitude_convention="reference_wavelength",
                            reference_wavelength=5.0, l_min=8.0, l_max=30.0))
    with pytest.raises(ConfigError, match=r"must lie within \[l_min, l_max\]"):
        resolve_config(_raw(amplitude_convention="reference_wavelength",
                            reference_wavelength=45.0, l_min=8.0, l_max=30.0))


def test_rule29d_guard_rechecked_on_realized_total_rms_equivalent():
    """Rule 29(d): the seed-containment guard is re-checked against the
    REALIZED total-RMS-equivalent amplitude (amplitude / kappa), not the
    raw reference_wavelength-convention amplitude value -- a config whose
    `amplitude` alone would pass Rule 7b's plain `> a_max` check must
    still be rejected once its total-RMS equivalent exceeds a_max.

    Geometry here is this file's own _raw() helper (N=8, L=80): a_max =
    PERTURBED_DISTANCE_SAFETY * r_ws / (2*ETA_CLIP) = 2.0678 Å (verified
    directly). kappa(hurst=0.8, l_min=8, l_max=30, ref=l_max) = 0.8730
    (verified directly), so A0_max = a_max*kappa = 1.8052 Å. An amplitude
    of 1.9 Å is BELOW a_max=2.0678 (would pass a naive plain check) but
    ABOVE A0_max=1.8052 -- must raise."""
    from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY
    from grainsmith.seeding import wigner_seitz_radius

    r_ws = wigner_seitz_radius(80.0 ** 3, 8)
    a_max = PERTURBED_DISTANCE_SAFETY * r_ws / (2.0 * ETA_CLIP)
    assert a_max == pytest.approx(2.0678, abs=1e-3)

    with pytest.raises(ConfigError, match="seed-containment guard"):
        resolve_config(_raw(amplitude_convention="reference_wavelength",
                            amplitude=1.9, l_min=8.0, l_max=30.0, hurst=0.8))

    # ... but the SAME 1.9 A is entirely fine under total_rms (well under
    # a_max=2.0678) -- confirms the failure above is convention-specific,
    # not just an ordinary Rule 7b violation.
    cfg = resolve_config(_raw(amplitude_convention="total_rms", amplitude=1.9,
                              l_min=8.0, l_max=30.0, hurst=0.8))
    assert cfg.boundaries.curved.amplitude == 1.9


def test_rule29d_guard_still_passes_safely_under_ceiling():
    """A reference_wavelength-convention amplitude safely under its
    (stricter) A0_max ceiling resolves without error."""
    cfg = resolve_config(_raw(amplitude_convention="reference_wavelength",
                              amplitude=1.0, l_min=8.0, l_max=30.0, hurst=0.8))
    assert cfg.boundaries.curved.amplitude_convention == "reference_wavelength"


def test_default_bit_identical_total_rms_convention_unaffected():
    """DEFAULT BEHAVIOR UNCHANGED: a config that never mentions
    amplitude_convention resolves with amplitude_convention == 'total_rms'
    and reference_wavelength == None, and its resolved amplitude/config
    YAML serialization is byte-identical to an explicit
    amplitude_convention: total_rms config -- the opt-in key changes
    nothing for a config that does not use it."""
    from grainsmith.config.resolve import _config_yaml

    raw_implicit = _raw()  # never mentions amplitude_convention at all
    raw_explicit = _raw(amplitude_convention="total_rms")
    cfg_implicit = resolve_config(raw_implicit)
    cfg_explicit = resolve_config(raw_explicit)
    assert cfg_implicit.boundaries.curved.amplitude_convention == "total_rms"
    assert cfg_implicit.boundaries.curved.reference_wavelength is None
    assert _config_yaml(cfg_implicit) == _config_yaml(cfg_explicit)
