"""Periodic GRF domain-warp tessellation backend (§6.6, method M5).

Domain warp: grain_of(x) = base.grain_of(x + u(x))
where u(x) is a periodic Gaussian random field (GRF). ``WarpTessellation``
supports exactly one amplitude spectrum:

- ``gaussian`` (the only spectrum this backend accepts):
      C(r) = exp(-r²/(2ℓ²))  ⇔  √Φ(k) ∝ exp(-k²ℓ²/4), k = 0 excluded
      (zero-mean field; see the note in ``synthesize_grf`` on how excluding
      the DC mode affects the field's numeric values without changing rng
      CONSUMPTION).

amplitude = A_rms per component (Å), calibrated by unit-RMS normalization.

Guards (G6):
1. Seed containment: ‖u‖ ≤ A_clip = min_seed_distance / 4
2. Bijectivity: max‖∇u‖ < 0.5 (Banach fixed-point injectivity bound)
3. Config-time: warn if A_rms > 0.3 × ℓ
4. Connectivity (G5): each grain = 1 connected voxel component

Why no ``self_affine`` spectrum here
-------------------------------------
``spectrum: self_affine`` (a band-limited power law
√Φ(k) ∝ k^(−(3+2H)/2) for k ∈ [2π/l_max, 2π/l_min]) is not supported on
this warp field — ``WarpTessellation.__init__`` and ``config/resolve.py``
both raise ``ConfigError`` on it — because
``grain_of(x) = base.grain_of(x + u(x))`` is a coordinate DIFFEOMORPHISM
under the G6 bijectivity guard, and a bijective map cannot change the
box-counting dimension of the flat-Voronoi surface it displaces
(see ``docs/physics.md`` §5b for the full Lipschitz argument) — giving
``u`` a self-affine PSD only recolors the waviness of a boundary whose
box-counting dimension it cannot change. Measured directly: synthesizing
the SAME white noise at H = 0.5 vs H = 0.9 on a 460 Å / 10-grain warp geometry
gives a field spatial correlation of 0.995 on a narrow (~0.4 decade)
band and 0.975 on a full-decade band, and the resulting cross-section
grain assignment (an ACTUAL WarpTessellation object, not a hand-rolled
stand-in) differs on 0.0475 % of sampled points on the narrow band and
0.1125 % on the full-decade band — the Hurst exponent's effect on the
actual boundary is far below realization noise either way. For a grain
boundary whose box-counting dimension genuinely responds to a roughness
spectrum, use
``boundaries.curved.method: perturbed_distance`` (``tessellation/
perturbed.py``), which perturbs the assignment rule itself rather than
the coordinate frame and so is not a diffeomorphism. Its ``spectrum:
self_affine`` field is synthesized by the SAME ``synthesize_grf``
function below (with ``n_components=1`` — one independent scalar field
per grain-color instead of one shared 3-component displacement field),
so the spectral-synthesis machinery is not duplicated; only the
*consumer* (a coordinate warp vs. a level-set perturbation) differs.
"""
from __future__ import annotations

import logging

import numpy as np

from grainsmith.constants import VOXEL_GRID_MAX, WARP_GRAD_MAX
from grainsmith.errors import ConfigError, TessellationError
from grainsmith.tessellation.base import Tessellation

log = logging.getLogger(__name__)


def _k_magnitude(
    grid_shape: tuple[int, int, int],
    box_lengths: np.ndarray,
) -> np.ndarray:
    """|k| on the FFT grid of a periodic box (rad/Å), shape grid_shape."""
    Nx, Ny, Nz = grid_shape
    L = np.asarray(box_lengths, dtype=np.float64)
    kx = np.fft.fftfreq(Nx) * Nx / L[0] * (2 * np.pi)
    ky = np.fft.fftfreq(Ny) * Ny / L[1] * (2 * np.pi)
    kz = np.fft.fftfreq(Nz) * Nz / L[2] * (2 * np.pi)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    return np.sqrt(KX**2 + KY**2 + KZ**2)


def synthesize_grf(
    grid_shape: tuple[int, int, int],
    correlation_length: float,
    box_lengths: np.ndarray,
    rng: np.random.Generator,
    spectrum: str = "gaussian",
    hurst: float = 0.8,
    l_min: float = 8.0,
    l_max: float = 60.0,
    n_components: int = 3,
) -> np.ndarray:
    """Synthesize an ``n_components``-component periodic GRF field on a
    voxel grid.

    Returns (n_components, Nx, Ny, Nz) float64 field with unit RMS per
    component (before amplitude scaling).  Field is exactly periodic via
    FFT synthesis; rng consumption is identical for both spectra (one
    white field per component).

    ``n_components`` defaults to 3 (the vector displacement field
    ``WarpTessellation`` needs) and is BIT-IDENTICAL to every pre-existing
    call site at that default (verified: same spectral envelope, same
    per-component ``rng.standard_normal`` draw sequence). Passing
    ``n_components=1`` draws exactly the same first-component randomness
    as the 3-component call for an identical seed — useful for a single
    SCALAR field (e.g. one per-grain perturbation field in
    ``tessellation/perturbed.py``) without duplicating this synthesis
    routine.

    spectrum 'gaussian'    : √Φ(k) = exp(−k²ℓ²/4) with ℓ = correlation_length,
        with the k = 0 (DC) mode EXCLUDED (forced to zero) so the field is
        zero-mean by construction — same convention as 'self_affine' below.
        A random per-realization mean offset is a rigid translation of the
        whole periodic field: it does not warp any boundary (∇u is
        unaffected) and contributes no genuine "waviness", but at k=0,
        exp(-0) = 1 is the LARGEST spectral weight of any mode, so leaving
        it in place inflates std(u) — the calibration denominator — well
        above the fluctuating part's RMS for realistic (ℓ, grid) ratios
        (up to ~2× when ℓ ~ box size). The exclusion is applied to the
        spectral envelope H, not to the random draw itself, so RNG
        consumption — one ``standard_normal`` draw per component — is the
        same whether or not the DC mode is excluded, even though the
        resulting field values differ.
    spectrum 'self_affine' : √Φ(k) ∝ k^(−(3+2H)/2) band-limited to
        k ∈ [2π/l_max, 2π/l_min], zero outside — the k = 0
        mode is outside the band, so the field is zero-mean by
        construction.
    """
    k_mag = _k_magnitude(grid_shape, box_lengths)
    if spectrum == "gaussian":
        H = np.exp(-(k_mag**2) * (correlation_length**2) / 4.0)
        H[0, 0, 0] = 0.0  # exclude the DC mode: zero-mean field (see docstring)
    elif spectrum == "self_affine":
        k_lo = 2.0 * np.pi / l_max
        k_hi = 2.0 * np.pi / l_min
        band = (k_mag >= k_lo) & (k_mag <= k_hi)
        if not np.any(band):
            raise ConfigError(
                f"self_affine band [2π/{l_max:g}, 2π/{l_min:g}] Å⁻¹ "
                f"contains no Fourier mode of the {grid_shape} grid — "
                "widen [l_min, l_max] or refine the grid.")
        H = np.zeros_like(k_mag)
        H[band] = k_mag[band] ** (-(3.0 + 2.0 * hurst) / 2.0)
    else:
        raise ConfigError(f"Unknown warp spectrum {spectrum!r} "
                          "(gaussian | self_affine).")

    Nx, Ny, Nz = grid_shape
    field = np.zeros((n_components, Nx, Ny, Nz), dtype=np.float64)
    for alpha in range(n_components):
        W = rng.standard_normal((Nx, Ny, Nz))
        Wk = np.fft.fftn(W)
        Uk = Wk * H
        u = np.fft.ifftn(Uk).real
        std = float(np.std(u))
        if std > 0:
            u = u / std  # unit RMS
        field[alpha] = u
    return field


def estimate_hurst(
    field: np.ndarray,
    box_lengths: np.ndarray,
    l_min: float,
    l_max: float,
    n_bins: int = 16,
    *,
    memory_limit_bytes: float | None = None,
    memory_limit_source: str = "config",
) -> float:
    """Back-estimate the Hurst exponent of a synthesized field (gate G13).

    Documented choice: G13 fits the radially averaged **3D**
    power spectrum of the field ITSELF — the synthesized target law is
    P(k) ∝ k^(−(3+2H)) inside the band, so a linear fit of
    log P̄ vs log k over geometric shell bins within
    [2π/l_max, 2π/l_min] yields Ĥ = −(slope + 3)/2.  NOTE:
    this is the 3D-field estimator, not the 2D planar-cut surface PSD; the
    physical self-affine surface relation C(q) ∝ q^(−(2+2H)) (the planar
    restriction, D = 3 − H) is the *derived* consequence of the same 3D law,
    not what is fitted here — the 3D fit is far more robust on narrow bands.

    ``field`` is (n_components, Nx, Ny, Nz): the PSD is averaged over
    however many leading components it has — 3 for warp's vector
    displacement field u(x) (its historical use), or ``n_colors`` for
    perturbed_distance's per-color scalar η fields (its only current use
    — see ``PerturbedDistanceTessellation.hurst_estimate``). Every
    component/color here was synthesized independently from the SAME
    spectral envelope (spectrum/hurst/l_min/l_max), so averaging their PSDs
    is a legitimate multi-realization ensemble estimate of that one target
    law in either case — more colors (perturbed_distance typically
    allocates more than 3) is a LARGER ensemble, not a different estimator.
    Global LINEAR rescaling (amplitude, warp's A_clip) provably does not
    move a log-log slope — a multiplicative constant only shifts the
    intercept. perturbed_distance's pointwise ``np.clip(..., -ETA_CLIP,
    ETA_CLIP)`` truncation is NOT linear and has no such guarantee in
    principle, but measured directly (matching ETA_CLIP's own docstring:
    the clip removes <0.3% of samples per component): clipping shifts
    Ĥ by ≤0.00073 at H ∈ {0.5, 0.7, 0.9} on a
    20-component field — a factor of ~200 (≈2.3 orders of magnitude) below
    HURST_G13_TOL = 0.15 — so its effect on G13 in practice is negligible,
    not zero by construction.
    Finite-band/finite-size bias is why G13 only warns (HURST_G13_TOL).
    Float64 conversion is per component, never a copy of all colors.
    The memory guard covers estimated diagnostic workspace before the
    frequency grid/FFT allocation; existing field storage is not duplicated.
    """
    from grainsmith.constants import MEMORY_HARD_LIMIT_BYTES
    from grainsmith.memory import build_memory_guard_message

    field = np.asarray(field)
    n_components = field.shape[0]
    estimated_bytes = int(np.prod(field.shape[1:])) * 64
    limit = MEMORY_HARD_LIMIT_BYTES if memory_limit_bytes is None else memory_limit_bytes
    if estimated_bytes > limit:
        raise TessellationError(build_memory_guard_message(
            "Hurst diagnostic workspace", estimated_bytes, limit, memory_limit_source,
            extra_advice="Use a smaller field grid or reduce the problem size.",
        ))
    log.info("Hurst diagnostic: %d colors on grid %s, %.3g GB estimated workspace; "
             "one color converted at a time.",
             n_components, field.shape[1:], estimated_bytes / 1e9)
    k_mag = _k_magnitude(field.shape[1:], box_lengths)
    power = np.zeros_like(k_mag)
    for alpha in range(n_components):
        power += np.abs(np.fft.fftn(np.asarray(field[alpha], dtype=np.float64))) ** 2
    k_lo = 2.0 * np.pi / l_max
    k_hi = 2.0 * np.pi / l_min
    band = (k_mag >= k_lo) & (k_mag <= k_hi) & (power > 0.0)
    edges = np.geomspace(k_lo, k_hi, n_bins + 1)
    log_k: list[float] = []
    log_p: list[float] = []
    for b in range(n_bins):
        sel = band & (k_mag >= edges[b]) & (k_mag < edges[b + 1])
        if not np.any(sel):
            continue
        log_k.append(np.log(float(np.mean(k_mag[sel]))))
        log_p.append(np.log(float(np.mean(power[sel]))))
    if len(log_k) < 3:
        raise TessellationError(
            "Hurst estimation needs ≥ 3 populated PSD shells inside the "
            "self_affine band — the band is too narrow for this grid.")
    slope = float(np.polyfit(log_k, log_p, 1)[0])
    return -(slope + 3.0) / 2.0


def _shell_integral_closed_form(k_lo: float, k_hi: float, hurst: float) -> float:
    """∫[k_lo, k_hi] k^(-(3+2H)) · k² dk = (k_lo^(-2H) − k_hi^(-2H)) / (2H).

    The k² factor is the 3D radial shell measure (surface area of a
    sphere of radius k grows as k²); the integrand k^(-(3+2H))·k² =
    k^(-(1+2H)) integrates in closed form for any H > 0 (schema enforces
    ``hurst > 0``, so 2H is never zero here).
    """
    return (k_lo ** (-2.0 * hurst) - k_hi ** (-2.0 * hurst)) / (2.0 * hurst)


def reference_shell_kappa(
    hurst: float,
    l_min: float,
    l_max: float,
    reference_wavelength: float | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    box_lengths: np.ndarray | None = None,
) -> float:
    """κ = σ(reference octave) / σ(whole synthesis band) for the
    self_affine spectrum P(k) ∝ k^(-(3+2H)) — the exact, invertible
    scale factor between the two ``perturbed_distance`` amplitude
    conventions (``boundaries.curved.amplitude_convention``).

    Background
    ----------
    ``synthesize_grf``'s self_affine branch normalizes to UNIT RMS over
    the WHOLE band [2π/l_max, 2π/l_min] before ``amplitude`` rescales it
    (the ``total_rms`` convention — grainsmith's original and still the
    default). Because the band-integrated variance of a k^(-(3+2H))
    spectrum is dominated by its LOW-k (long-wavelength) end, widening
    the band (more octaves, e.g. by lowering l_min or raising l_max)
    shrinks the fraction of that fixed total-RMS budget any short
    wavelength gets — so at fixed ``amplitude`` adding a coarse octave
    can REDUCE small-scale roughness (see docs/physics.md §5b; measured
    directly as a −16.1% swing in boundary area S_V from adding one
    octave, the wrong sign).

    The ``reference_wavelength`` convention instead fixes the RMS power
    contributed by ONE reference octave, [reference_wavelength/2,
    reference_wavelength] (clipped to the synthesis band), letting every
    OTHER resolved octave's contribution accumulate independently on top
    — so ``amplitude`` (renamed ``amplitude`` still, just reinterpreted)
    keeps a stable physical meaning (Å-RMS at a named wavelength)
    regardless of how many octaves happen to be in band.

    CAVEAT on the default anchor (``reference_wavelength=None`` → l_max):
    this anchor MOVES with the band, and under this default the
    small-scale-roughness-suppression trend above is NOT fully corrected
    — local roughness still decreases (just less steeply than under
    ``total_rms``) as octaves are added at fixed `amplitude`. Only an
    explicit **fixed absolute**
    ``reference_wavelength`` (anchored near l_min, not moving with
    l_max) inverts the trend to the physically expected monotonic
    increase. The moving default is intended for the common case where
    l_max itself represents a physical scale (e.g. the grain radius)
    that should stay the roughness reference as other settings change —
    it is not, by itself, a fix for a controlled band-width scan. See
    docs/physics.md §5b(f) for the measurement.

    The two conventions are related by one scalar:

        κ = σ_reference_octave / σ_total_band  ∈ (0, 1]  (κ = 1 exactly
            when the reference octave IS the whole band, e.g.
            l_max/l_min = 2)

        amplitude_reference_wavelength = amplitude_total_rms · κ
        amplitude_total_rms            = amplitude_reference_wavelength / κ

    (``amplitude_to_reference_wavelength`` / ``amplitude_to_total_rms``
    below wrap exactly these two lines — use them rather than
    multiplying/dividing by κ inline, since the two directions are easy
    to invert by mistake.) Because κ ≤ 1, a seed-containment ceiling
    a_max expressed on the total_rms-equivalent amplitude maps to a
    STRICTER reference_wavelength ceiling a_max·κ ≤ a_max, never a
    looser one — the guard cannot be weakened by a convention switch.

    Parameters
    ----------
    hurst, l_min, l_max : the self_affine band, as in the schema.
    reference_wavelength : anchor wavelength in Å (default: l_max).
        Clamped so the reference shell never extends outside
        [2π/l_max, 2π/l_min] in k-space (i.e. outside [l_min, l_max] in
        wavelength) even if ``reference_wavelength`` sits near an edge.
    grid_shape, box_lengths : when BOTH given, κ is computed as an EXACT
        sum over an actual discrete FFT grid ``synthesize_grf`` would draw
        the field on — matches that grid's realized discretization
        exactly, including any finite-grid deviation from the continuum
        law. Available for verification/analysis use (e.g. reproducing
        a specific realized field's own κ after the fact, as this
        project's own test suite does); NOT what
        ``PerturbedDistanceTessellation.__init__`` calls — see below.
        When omitted (the default), κ is the closed-form CONTINUUM
        integral of the target power law — a cheap, grid-independent
        estimate usable BEFORE any grid exists.
        ``config/resolve.py``'s Rule 29(d) pre-flight guard and
        ``PerturbedDistanceTessellation.__init__``'s own guard 1 BOTH
        deliberately call this closed-form branch (grid_shape/
        box_lengths omitted) — guard 1 runs BEFORE the constructor
        resolves its own grid (the grid resolution and field synthesis
        happen later in ``__init__``), so the grid-exact discrete branch
        is not actually available to it yet; calling the closed-form
        branch here also keeps guard 1's number bit-identical to what
        Rule 29(d) already checked at resolve time, rather than
        re-deriving a second, slightly different κ. This is the same
        "sufficient, not tight, pre-flight" spirit as this method's
        other resolve-time checks (see
        ``constants.PERTURBED_DISTANCE_SAFETY``'s docstring) — not a
        promise of bit-exact agreement between the closed-form κ and
        the eventual realized field's OWN discrete spectrum.
    """
    ref = l_max if reference_wavelength is None else float(reference_wavelength)
    k_lo_full, k_hi_full = 2.0 * np.pi / l_max, 2.0 * np.pi / l_min
    k_lo_ref = max(2.0 * np.pi / ref, k_lo_full)
    k_hi_ref = min(2.0 * np.pi / (ref / 2.0), k_hi_full)

    if grid_shape is not None and box_lengths is not None:
        k_mag = _k_magnitude(grid_shape, np.asarray(box_lengths, dtype=np.float64))
        band = (k_mag >= k_lo_full) & (k_mag <= k_hi_full)
        exponent = -(3.0 + 2.0 * hurst)
        var_total = float(np.sum(k_mag[band] ** exponent))
        ref_shell = band & (k_mag >= k_lo_ref) & (k_mag <= k_hi_ref)
        p_ref = float(np.sum(k_mag[ref_shell] ** exponent))
    else:
        var_total = _shell_integral_closed_form(k_lo_full, k_hi_full, hurst)
        p_ref = _shell_integral_closed_form(k_lo_ref, k_hi_ref, hurst)

    return float(np.sqrt(p_ref / var_total))


def amplitude_to_reference_wavelength(amplitude_total_rms: float, kappa: float) -> float:
    """Convert a ``total_rms``-convention amplitude to its
    ``reference_wavelength``-convention equivalent: A0 = A_total · κ
    (see ``reference_shell_kappa``)."""
    return float(amplitude_total_rms) * float(kappa)


def amplitude_to_total_rms(amplitude_reference: float, kappa: float) -> float:
    """Convert a ``reference_wavelength``-convention amplitude to its
    ``total_rms``-convention equivalent: A_total = A0 / κ
    (see ``reference_shell_kappa``)."""
    return float(amplitude_reference) / float(kappa)


class WarpTessellation(Tessellation):
    """Domain-warped Voronoi tessellation (§6.6, method M5).

    Wraps any base Tessellation by evaluating at x + u(x).
    Requires the base to be a FlatTessellation (or weighted/anisotropic).
    """

    def __init__(
        self,
        base: Tessellation,
        box_lengths: np.ndarray,
        periodic: list[bool],
        amplitude: float,
        correlation_length: float,
        min_seed_distance: float,
        rng: np.random.Generator,
        grid_size: int | str = "auto",
        connectivity_check: bool = True,
        spectrum: str = "gaussian",
        hurst: float = 0.8,
        l_min: float = 8.0,
        l_max: float = 60.0,
        *,
        memory_limit_bytes: float | None = None,
        memory_limit_source: str = "config",
    ) -> None:
        # §13: stamp the per-run memory budget FIRST, before the
        # connectivity_check below builds a voxel grid at construction time
        # (see tessellation/base.py's Tessellation.memory_limit_bytes
        # docstring). Only overwrite the class default when the caller
        # actually passed one, so direct construction (tests) keeps the
        # 16 GB class default.
        if memory_limit_bytes is not None:
            self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_source = memory_limit_source

        # API-level guard (mirrors config/resolve.py Rule 8a): `warp` only
        # ever accepts the gaussian spectrum. self_affine is REMOVED here,
        # not deprecated — see the module docstring's "Why no self_affine
        # spectrum here" section for the diffeomorphism argument and the
        # measured field-correlation/cross-section numbers. This check is
        # enforced at the constructor too (not just config/resolve.py)
        # because tests and any direct API caller can construct a
        # WarpTessellation without going through resolve_config.
        if spectrum != "gaussian":
            raise ConfigError(
                f"WarpTessellation only accepts spectrum='gaussian' "
                f"(got {spectrum!r}). For a self-affine grain boundary, use "
                "boundaries.curved.method: perturbed_distance (spectrum: "
                "self_affine is supported there); for warp's own "
                "roughness, tune spectrum: gaussian + correlation_length."
            )
        self._base = base
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        self._amplitude = amplitude
        self._correlation_length = correlation_length
        self._spectrum = spectrum
        self._hurst = hurst
        self._l_min = l_min
        self._l_max = l_max

        # The length scale governing the gradient (guard heuristics and the
        # auto grid): ℓ for gaussian, l_min for self_affine (the gradient of
        # a band-limited power law is dominated by its shortest wavelength).
        ell = correlation_length if spectrum == "gaussian" else l_min

        # Config-time heuristic (§6.6 guard 3)
        if amplitude > 0.3 * ell:
            scale_name = "ℓ" if spectrum == "gaussian" else "l_min"
            log.warning(
                f"WarpTessellation: amplitude={amplitude:.3g} > "
                f"0.3*{scale_name}={0.3 * ell:.3g}. Bijectivity guard may "
                f"trip. Consider lower amplitude or larger {scale_name}."
            )

        # A_clip (guard 1).  The REQUESTED RMS amplitude must respect the
        # seed-containment bound; the field clip below only trims Gaussian
        # tails.  Silently delivering a smaller amplitude than requested
        # would be a hidden misconfiguration (§7 hard rule, mirrored in
        # config/resolve.py).
        a_clip = min_seed_distance / 4.0
        if amplitude > a_clip:
            raise ConfigError(
                f"warp amplitude={amplitude:.4g} Å exceeds A_clip = "
                f"min_seed_distance/4 = {a_clip:.4g} Å (seed-containment "
                "guard, §6.6 G6). Reduce amplitude or increase "
                "min_seed_distance."
            )

        # Grid size: `grid_size` counts voxels along the SHORTEST box edge;
        # other axes scale by L_axis / L_min so the voxel spacing h ≈ ℓ/4
        # is isotropic (using L[0] as the reference silently under-resolves
        # non-cubic boxes).
        L_min = float(np.min(self._L))
        if grid_size == "auto":
            n_short = max(16, int(np.ceil(L_min / (ell / 4.0))))
        else:
            n_short = int(grid_size)
        n_short = min(n_short, VOXEL_GRID_MAX)
        h_t = L_min / n_short
        nx, ny, nz = (int(np.ceil(self._L[ax] / h_t)) for ax in range(3))
        if max(nx, ny, nz) > VOXEL_GRID_MAX:
            log.warning(
                f"WarpTessellation: grid {(nx, ny, nz)} clamped to ≤ "
                f"{VOXEL_GRID_MAX} per axis; warp field resolution is "
                "coarser than the ℓ/4 (or l_min/4) target."
            )
            nx, ny, nz = (min(g, VOXEL_GRID_MAX) for g in (nx, ny, nz))
        self._grid_shape: tuple[int, int, int] = (nx, ny, nz)

        # self_affine resolvability: the shortest band wavelength must be
        # representable on the ACTUAL grid.
        if spectrum == "self_affine":
            h_field = float(max(self._L[ax] / self._grid_shape[ax]
                                for ax in range(3)))
            if l_min < 2.0 * h_field:
                raise ConfigError(
                    f"self_affine l_min={l_min:g} Å is below 2·h_field="
                    f"{2.0 * h_field:g} Å (grid {self._grid_shape} cannot "
                    "resolve it; Nyquist). Increase l_min or refine the "
                    "grid (analysis.voxel_grid / a box that avoids the "
                    f"{VOXEL_GRID_MAX}-per-axis clamp).")

        # Synthesize GRF field
        raw_field = synthesize_grf(
            self._grid_shape, correlation_length, self._L, rng,
            spectrum=spectrum, hurst=hurst, l_min=l_min, l_max=l_max)
        # Scale to amplitude A_rms
        self._field = raw_field * amplitude  # (3, Nx, Ny, Nz)

        # Guard 1: clip to A_clip
        u_norm = np.sqrt(np.sum(self._field**2, axis=0))
        max_norm = float(np.max(u_norm))
        if max_norm > a_clip:
            scale = a_clip / max_norm
            self._field *= scale
            log.info(f"WarpTessellation: field clipped from {max_norm:.4g} to {a_clip:.4g} Å")
        # Max displacement (vector L2 norm) — used by bounding_radius
        self._u_max = float(np.max(np.sqrt(np.sum(self._field**2, axis=0))))

        # Guard 2: bijectivity check via finite differences
        grad_max = self._compute_grad_max()
        if grad_max >= WARP_GRAD_MAX:
            advice = ("Reduce amplitude or increase correlation_length."
                      if spectrum == "gaussian" else
                      f"Reduce amplitude or increase l_min={l_min:g} Å "
                      "(the gradient of a self-affine field is dominated "
                      "by its shortest wavelength).")
            raise TessellationError(
                f"Warp bijectivity guard failed: max‖∇u‖ = {grad_max:.4g} "
                f">= {WARP_GRAD_MAX} (G6). {advice}"
            )
        self._grad_max = grad_max
        log.info(f"WarpTessellation: max‖∇u‖ = {grad_max:.4f} (limit {WARP_GRAD_MAX})")

        # Guard 4 / gate G5: every grain must be exactly one periodically-
        # connected voxel component of the WARPED membership (§6.6).
        self._voxel = None
        if connectivity_check:
            from grainsmith.tessellation.voxel import build_voxel_grid
            self._voxel = build_voxel_grid(
                self, self._L, "auto",
                r_ws=min_seed_distance,
                correlation_length=ell,
            )
            self._voxel.check_connectivity(self._periodic)
            log.info("WarpTessellation: G5 OK — all grains singly connected.")

    def _compute_grad_max(self) -> float:
        """Compute max‖∇u‖₂ via finite differences on the grid.

        The per-axis stencil matches _interpolate_u's boundary handling so the
        guard measures the gradient of the field actually evaluated at query
        time: a PERIODIC axis uses the wrap-around (np.roll) central difference,
        while a FREE axis uses one-sided differences at the first/last layer
        (np.gradient), because the interpolant clamps there.  Rolling a free
        axis instead measures a spurious wrap-around jump across the wall and
        can miss a genuine gradient spike at the free surface, silently passing
        a non-bijective warp (G6).
        """
        N = self._grid_shape
        L = self._L
        h = [L[ax] / N[ax] for ax in range(3)]
        grad2 = np.zeros(N, dtype=np.float64)
        for alpha in range(3):
            u = self._field[alpha]
            for ax in range(3):
                if self._periodic[ax]:
                    d = (np.roll(u, -1, axis=ax)
                         - np.roll(u, 1, axis=ax)) / (2 * h[ax])
                else:
                    d = np.gradient(u, h[ax], axis=ax)
                grad2 += d**2
        return float(np.max(np.sqrt(grad2)))

    def _interpolate_u(self, X: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of u at lab positions X (any frame).

        Hand-rolled so the boundary handling is exact per axis:
        - periodic axis: true periodic wrap (sample N ≡ sample 0; scipy's
          map_coordinates mode='wrap' is NOT this — it overlaps the first
          and last samples, shortening the period to (N−1)h);
        - free axis: clamp to the boundary sample (no wrap across a wall).

        Grid registration: sample j sits at the voxel center (j + 0.5)·h.
        The field is periodic, so u(x) = u(x mod L) — unwrapped inputs are
        handled by the index wrap itself.
        """
        n_pts = len(X)
        i0 = np.empty((3, n_pts), dtype=np.int64)
        i1 = np.empty((3, n_pts), dtype=np.int64)
        frac = np.empty((3, n_pts), dtype=np.float64)
        for ax in range(3):
            N = self._grid_shape[ax]
            t = X[:, ax] / self._L[ax] * N - 0.5
            if self._periodic[ax]:
                t = t % N
                lo = np.floor(t).astype(np.int64)
                frac[ax] = t - lo
                i0[ax] = lo % N
                i1[ax] = (lo + 1) % N
            else:
                t = np.clip(t, 0.0, N - 1.0)
                lo = np.minimum(np.floor(t).astype(np.int64), N - 2)
                frac[ax] = t - lo
                i0[ax] = lo
                i1[ax] = lo + 1
        fx, fy, fz = frac
        gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz

        u_interp = np.empty((n_pts, 3), dtype=np.float64)
        for alpha in range(3):
            F = self._field[alpha]
            u_interp[:, alpha] = (
                F[i0[0], i0[1], i0[2]] * gx * gy * gz
                + F[i1[0], i0[1], i0[2]] * fx * gy * gz
                + F[i0[0], i1[1], i0[2]] * gx * fy * gz
                + F[i0[0], i0[1], i1[2]] * gx * gy * fz
                + F[i1[0], i1[1], i0[2]] * fx * fy * gz
                + F[i1[0], i0[1], i1[2]] * fx * gy * fz
                + F[i0[0], i1[1], i1[2]] * gx * fy * fz
                + F[i1[0], i1[1], i1[2]] * fx * fy * fz
            )
        return u_interp

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        """Membership via domain warp: grain_of(x) = base.grain_of(x + u(x)).

        u is periodic and the base handles wrapping/min-image internally,
        so no explicit wrapping is needed here.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        u = self._interpolate_u(X)
        return self._base.grain_of(X + u)

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact home-cell ownership of the warped cell (§6.8).

        The warped home cell of grain i is the preimage of the base home
        cell under x → x + u(x); with ‖u‖ ≤ A_clip its preimage lies within
        bounding_radius(i) = base radius + max‖u‖ of the seed.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        u = self._interpolate_u(X)
        return self._base.owns(X + u, i)

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed boundary distance evaluated in the warped (base) frame.

        This is the base-cell margin at the warped point — a consistent
        monotone proxy for the true warped-boundary distance (exact up to
        the local metric distortion ‖∇u‖ < 0.5)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        u = self._interpolate_u(X)
        return self._base.margin(X + u, i)

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> None:
        """No certified GB-shell bound: ``margin()`` here is a documented
        PROXY (its own docstring above), the base-cell margin evaluated at
        the warped point ``x + u(x)``, exact only up to the local metric
        distortion ``‖∇u‖ < 0.5`` -- not the exact Euclidean distance to
        the warped-cell boundary, and not proven 1-Lipschitz in the
        UNWARPED coordinate ``x`` that the overlap cutoff is measured in
        (the exact Lipschitz constant needed, ``1 + grad_max``, is not
        derived/certified here). Always returns ``None`` (unchanged
        full-N overlap behaviour)."""
        return None

    def adjacency(self) -> list[tuple[int, int]]:
        """Warp inherits topology from base (bijective warp preserves adjacency)."""
        return self._base.adjacency()

    def bounding_radius(self, i: int) -> float:
        return self._base.bounding_radius(i) + self._u_max

    @property
    def seeds(self) -> np.ndarray:
        return self._base.seeds

    @property
    def n_grains(self) -> int:
        return self._base.n_grains

    @property
    def grad_max(self) -> float:
        return self._grad_max

    @property
    def voxel_grid(self):
        """VoxelGrid built for the G5 check (None if connectivity_check=False);
        reusable by the analysis stage."""
        return self._voxel

    @property
    def field(self) -> np.ndarray:
        return self._field

    @property
    def spectrum(self) -> str:
        return self._spectrum

    # NOTE: WarpTessellation has no hurst_estimate() method. Since the
    # constructor guard above rejects spectrum: self_affine unconditionally,
    # self._spectrum is always "gaussian", for which Hurst back-estimation
    # is not meaningful -- so no such method is exposed here. G13
    # (Hurst back-estimation) instead lives on
    # ``PerturbedDistanceTessellation.hurst_estimate`` (tessellation/
    # perturbed.py), the only self_affine consumer.
