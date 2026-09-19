"""QA gate registry and gate evaluators.

Gates G1-G10 are the core set; G11-G16 cover volume targeting, MDF
targeting, Hurst back-estimation, single-crystal commensurability, phase
volume fractions, and GB curvature sanity; G17-G18 the doping stage
(docs/physics.md §9), G19-G20 the perturbed_distance tessellation backend
(tessellation/perturbed.py), and G21 alongside G16 as a second, independent
read of the same gb_curvature analysis (analysis/curvature.py). G22-G24
close the volume-weighted-ODF / Sigma3 / post-warp-volume gaps: G22 re-measures the assignment annealer's own drift
claim and its kernel-smoothed bias signature (orientation/odf.py), G23
reports the angle-window-vs-true-CSL Sigma3 gap that the annealing
objective cannot close, and G24 measures the volume displacement the
domain warp introduces AFTER G11's own (unwarped-base) check. G25 puts
each texture component's configured weight next to its realised
grain-count and volume fractions. G26 closes the LAST gap in that group:
G22/odf_drift_max control the TESSELLATION-volume-weighted ODF, but the
artefact shipped to an MD user is ATOM-count weighted, and G26 measures
the (atoms/grain)^(-1/3) discretisation floor between the two.
All results land in summary.csv regardless of which gates a given run
exercises.

Gate locations
--------------
G1  config schema + cross-field rules     → config/resolve.py (raises)
G2  spglib round-trip                     → crystal/verify.py (raises)
G3  volume sum                            → flat: enforced at construction
                                            (flat.py); voxel: gate_g3_voxel
G4  Euler V−E+F=2 per cell                → enforced at construction (flat.py)
G5  curved connectivity + empty cells     → enforced at construction
                                            (warp.py / weighted.py / voxel.py);
                                            perturbed_distance instead REPAIRS
                                            fragments and reports severity via
                                            G19 (tessellation/perturbed.py)
G6  warp guards (A_clip, ‖∇u‖)            → enforced at construction (warp.py)
G7  post-overlap min distance             → gate_g7_min_distance (also
                                            self-checked inside remove_overlaps)
G8  atom count vs theoretical             → gate_g8_atom_count
G9  composition drift                     → gate_g9_composition (warn-only)
G10 LAMMPS re-parse round-trip            → io/lammps.py
G11 re-measured grain volume vs SDOT target → gate_g11_volume_targets (hard)
G12 MDF-target chi2 distance              → gate_g12_mdf_target (warn-only)
G13 back-estimated Hurst exponent         → gate_g13_hurst (warn-only;
                                            perturbed_distance only — warp
                                            does not accept
                                            spectrum: self_affine, see
                                            tessellation/warp.py)
G14 single-crystal box/lattice commensurability → gate_g14_commensurate
                                            (warn-only)
G15 re-measured phase volume fractions    → gate_g15_phase_fractions
                                            (warn-only)
G16 GB-curvature degenerate-sample fraction → gate_g16_curvature (warn-only)
G17 dopant composition + enrichment       → gate_g17_doping_composition
                                            (warn-only)
G18 post-placement dopant min-distance    → gate_g18_doping_geometry (hard)
G19 perturbed_distance reassigned-voxel fraction → gate_g19_reassigned_fraction
                                            (warn-only; repair-and-report, not
                                            hard-fail — see G5 note above)
G20 perturbed_distance box-counting roughness index → gate_g20_db_estimate
                                            (warn-only diagnostic; NOT a
                                            converged, large-effect-size
                                            fractal-dimension certificate
                                            below ~5.5 synthesis-band
                                            octaves — see its docstring)
G21 per-grain Gauss-Bonnet face-interior residual → gate_g21_gauss_bonnet
                                            (warn-only; gb_curvature only,
                                            diagnoses the ESTIMATOR, not
                                            the model — see docstring below)
G22 volume-weighted ODF fidelity          → gate_g22_odf_fidelity
                                            (warn-only; re-measures the
                                            assignment annealer's own drift
                                            claim + a kernel-smoothed bias
                                            signature vs. a null — see
                                            orientation/odf.py)
G23 angle-window vs CSL Sigma3 consistency → gate_g23_sigma3_consistency
                                            (report-only, never trips;
                                            single-phase MDF-histogram
                                            runs only)
G24 final (post-warp) per-grain volume fidelity → gate_g24_final_volume_targets
                                            (warn-only; only when the warp
                                            geometry differs from the SDOT
                                            base G11 already checked — see
                                            docstring below)
G25 texture-component fidelity            → gate_g25_component_fidelity
                                            (report-only, never trips;
                                            odf_components scheme only —
                                            configured weight vs realised
                                            grain-COUNT vs realised VOLUME
                                            fraction per component)
G26 atomistic ODF-weighting discretisation → gate_g26_odf_weighting_floor
    floor                                  (report-only, never trips;
                                            tessellation-volume weight vs
                                            atom-count weight, the gap
                                            odf_drift_max does NOT cover —
                                            see docstring below)

Construction-enforced gates raise immediately; the evaluator functions here
return GateResult so the pipeline can record every outcome into summary.csv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GateResult:
    """Result of a single QA gate evaluation.

    Attributes
    ----------
    gate : str
        Gate identifier (e.g. ``"G1"``, ``"G3"``).
    passed : bool
        Whether the gate passed.
    measured : Any
        The measured value (for threshold gates, e.g. relative volume error).
    message : str
        Human-readable description of the outcome; always set for failures.
    """

    gate: str
    passed: bool
    measured: Any = None
    message: str = ""


class QAGates:
    """Registry that accumulates QA gate results for one pipeline run.

    All gate results (pass and fail) are stored so they can be written to
    ``summary.csv`` at the end of the run.  Calling ``require()`` on a
    failing gate raises :exc:`~grainsmith.errors.QAGateError` immediately,
    aborting the pipeline with an actionable error message.
    """

    def __init__(self) -> None:
        self._results: list[GateResult] = []

    def record(self, r: GateResult) -> None:
        """Append *r* to the result list without raising on failure.

        Use ``record`` for non-fatal gates (e.g. warnings).
        Use ``require`` for mandatory gates that must abort on failure.

        Parameters
        ----------
        r : GateResult
            Gate result to record.
        """
        self._results.append(r)

    def all_passed(self) -> bool:
        """Return ``True`` if and only if every recorded gate passed."""
        return all(r.passed for r in self._results)

    def results(self) -> list[GateResult]:
        """Return a shallow copy of all recorded gate results."""
        return list(self._results)

    def require(self, r: GateResult) -> None:
        """Record *r* and raise :exc:`~grainsmith.errors.QAGateError` if it failed.

        This is the primary mechanism for hard-failing the pipeline when a
        mandatory gate trips.  The gate identifier, measured value, and
        message are all embedded in the exception so the user can diagnose
        the issue without reading source code.

        Parameters
        ----------
        r : GateResult
            Gate result to record and check.

        Raises
        ------
        grainsmith.errors.QAGateError
            If ``r.passed`` is ``False``.
        """
        self.record(r)
        if not r.passed:
            from grainsmith.errors import QAGateError

            raise QAGateError(
                f"Gate {r.gate} failed: {r.message} (measured={r.measured})"
            )


# ---------------------------------------------------------------------------
# Gate evaluators (§9)
# ---------------------------------------------------------------------------

def gate_g3_voxel(volumes: np.ndarray, box_lengths: np.ndarray,
                  voxel_size: np.ndarray) -> GateResult:
    """G3, curved geometry: Σ grain voxel volumes == box volume within one
    voxel layer of the box surface (hard)."""
    L = np.asarray(box_lengths, dtype=np.float64)
    box_vol = float(np.prod(L))
    total = float(np.sum(volumes))
    h = float(np.max(voxel_size))
    surface_layer = 2.0 * h * (L[0] * L[1] + L[1] * L[2] + L[0] * L[2])
    err = abs(total - box_vol)
    return GateResult(
        gate="G3",
        passed=err <= surface_layer,
        measured=err / box_vol,
        message=(f"voxel volume sum {total:.6g} vs box {box_vol:.6g} Å³ "
                 f"(tolerance: 1 voxel layer = {surface_layer:.4g} Å³)"),
    )


def gate_g7_min_distance(atoms, cutoff: float, periodic: list[bool],
                         box_lengths: np.ndarray,
                         cell_matrix: np.ndarray | None = None,
                         jobs: int = 1) -> GateResult:
    """G7: a fresh PBC neighbor query on the final structure proves the
    minimum inter-atomic distance ≥ cutoff (hard).

    *cell_matrix* (columns = box vectors), when given, routes the query
    through the general triclinic (reciprocal-bounded periodic-image) path
    (``box.cells``) instead of the orthogonal ``KDTree(boxsize=L)``
    shortcut — a plain per-axis min-image is WRONG in a tilted cell.
    """
    from grainsmith.atoms.overlap import (
        _pbc_pairs,
        _pbc_pairs_general,
        _wrap_positions,
        _wrap_positions_general,
    )

    L = np.asarray(box_lengths, dtype=np.float64)
    if len(atoms) < 2:
        return GateResult("G7", True, 0, "fewer than 2 atoms")
    # workers == jobs: the pair SET is worker-count invariant by
    # construction (see overlap._discover_pairs / remove_overlaps), so this
    # is a pure measurement-scaling knob. jobs == 1 keeps the serial
    # query_pairs route; jobs > 1 uses exactly that many scipy threads for
    # the ball-query route, scaling with --jobs. jobs <= 1 (including
    # non-positive values) maps to workers=1 -- scipy's query_ball_point
    # rejects workers=0 outright, and workers=-1 means "all cores" -- so
    # only jobs > 1 is passed through verbatim.
    workers = jobs if jobs > 1 else 1
    if cell_matrix is not None:
        H = np.asarray(cell_matrix, dtype=np.float64)
        Hinv = np.linalg.inv(H)
        wrapped = _wrap_positions_general(atoms.pos, periodic, H, Hinv)
        n_pairs = len(_pbc_pairs_general(wrapped, cutoff, periodic, H, Hinv,
                                         workers=workers))
    else:
        wrapped = _wrap_positions(atoms.pos, periodic, L)
        n_pairs = len(_pbc_pairs(wrapped, cutoff, periodic, L,
                                 workers=workers))
    return GateResult(
        gate="G7",
        passed=n_pairs == 0,
        measured=n_pairs,
        message=(f"{n_pairs} atom pair(s) closer than cutoff={cutoff:.4g} Å "
                 "after overlap removal"),
    )


def gate_g8_atom_count(n_final: int, n_deleted: int, rho_atom: float,
                       box_volume: float) -> GateResult:
    """G8: |N_final − N_est| with N_est = ρ_atom·V_box − deleted;
    warn > 2 % (message), fail > 5 % (hard).  ρ_atom = atoms/Å³ of the
    ideal crystal (Z_cell / det A)."""
    n_est = rho_atom * box_volume - n_deleted
    if n_est <= 0:
        return GateResult("G8", False, n_est,
                          "theoretical atom count is non-positive")
    rel = abs(n_final - n_est) / n_est
    level = "FAIL" if rel > 0.05 else ("WARN" if rel > 0.02 else "ok")
    return GateResult(
        gate="G8",
        passed=rel <= 0.05,
        measured=rel,
        message=(f"{level}: N_final={n_final} vs N_est={n_est:.0f} "
                 f"(rel. dev. {rel:.3%}; warn > 2%, fail > 5%)"),
    )


def gate_g9_composition(final_counts: dict[str, int],
                        nominal_fractions: dict[str, float]) -> GateResult:
    """G9: composition drift after deletions vs nominal stoichiometry.
    Warn-only per §9 ('warn > 1 % per species, report always'): the gate
    never hard-fails, the message carries the per-species drift."""
    n_total = sum(final_counts.values())
    drifts: dict[str, float] = {}
    species = set(final_counts) | set(nominal_fractions)
    for sp in sorted(species):
        x_final = final_counts.get(sp, 0) / n_total if n_total else 0.0
        x_nom = nominal_fractions.get(sp, 0.0)
        drifts[sp] = abs(x_final - x_nom)
    max_drift = max(drifts.values()) if drifts else 0.0
    detail = ", ".join(f"{sp}: {d:.3%}" for sp, d in drifts.items())
    level = "WARN" if max_drift > 0.01 else "ok"
    return GateResult(
        gate="G9",
        passed=True,
        measured=max_drift,
        message=f"{level}: composition drift per species — {detail}",
    )


def gate_g11_volume_targets(volumes: np.ndarray, targets: np.ndarray,
                            vol_tol: float,
                            warp_active: bool = False) -> GateResult:
    """G11: RE-MEASURED per-grain volume error vs the SDOT
    targets <= vol_tol (hard).  Measured on the final tessellation cells,
    not on the optimizer's own claim.

    When *warp_active* is True the volumes are those of the *unwarped* power
    base: the volume target is enforced on the base, and the
    subsequent domain warp displaces grain boundaries, so the realized
    per-grain volumes deviate from the targets (bounded by the G6
    ``‖∇u‖<0.5`` + amplitude-clip guard).  The gate decision is unchanged;
    only the message records this so a user does not read the base-cell PASS
    as a guarantee on the realized warped cells."""
    volumes = np.asarray(volumes, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if np.any(targets <= 0.0):
        # A non-positive target volume is a degenerate spec, not a measurable
        # relative error; report it explicitly rather than letting 0/0 = NaN
        # poison the whole gate (max/mean → NaN).
        n_bad = int(np.sum(targets <= 0.0))
        return GateResult(
            gate="G11", passed=False, measured=float("nan"),
            message=(f"FAIL: {n_bad} target grain volume(s) are non-positive — "
                     "cannot evaluate relative volume error (degenerate target)."),
        )
    rel = np.abs(volumes - targets) / targets
    max_rel = float(np.max(rel))
    base_note = (" — measured on the UNWARPED power base; the warp displaces "
                 "boundaries, so realized cell volumes deviate from targets "
                 "(bounded by the G6 grad guard)") if warp_active else ""
    return GateResult(
        gate="G11",
        passed=max_rel <= vol_tol,
        measured=max_rel,
        message=(f"max relative grain-volume error {max_rel:.3e} vs "
                 f"target distribution (tol {vol_tol:g}; "
                 f"mean {float(np.mean(rel)):.3e}){base_note}"),
    )


def gate_g12_mdf_target(chi2: float, chi2_max: float) -> GateResult:
    """G12 (warn-only): symmetric χ² distance between the
    RE-MEASURED area-weighted neighbor-MDF histogram of the final
    structure and the configured target.

    Annealing is a heuristic, so the gate never hard-fails (honesty over
    hard failure) — the message carries WARN when the
    distance exceeds the threshold, mirroring the G9 pattern.
    """
    level = "WARN" if chi2 > chi2_max else "ok"
    return GateResult(
        gate="G12",
        passed=True,
        measured=chi2,
        message=(f"{level}: re-measured MDF-target chi2 distance "
                 f"{chi2:.4f} (threshold {chi2_max:g}; symmetric chi2 "
                 "over area-weighted bin masses)"),
    )


def gate_g13_hurst(h_est: float, h_target: float) -> GateResult:
    """G13 (warn-only): back-estimated Hurst exponent of the
    generated self-affine field vs the configured one.

    Owned by ``perturbed_distance`` (its per-color η
    fields) — ``warp`` does not accept spectrum: self_affine at all (a
    coordinate diffeomorphism cannot produce a genuinely self-affine
    boundary regardless of Hurst exponent; docs/physics.md §5b), so it
    never triggers this gate.

    Ĥ is fitted from the radially averaged PSD over the configured band
    (tessellation/warp.estimate_hurst, applied to whichever field the
    caller passes); the finite-band/finite-size bias is documented at
    HURST_G13_TOL — the gate never hard-fails."""
    from grainsmith.constants import HURST_G13_TOL

    dev = abs(h_est - h_target)
    level = "WARN" if dev > HURST_G13_TOL else "ok"
    return GateResult(
        gate="G13",
        passed=True,
        measured=h_est,
        message=(f"{level}: back-estimated Hurst H_hat={h_est:.3f} vs "
                 f"target H={h_target:g} (|dev|={dev:.3f}, tol "
                 f"{HURST_G13_TOL:g}; band-limited PSD fit)"),
    )


def gate_g14_commensurate(
        misfits: dict[int, tuple[float, float]]) -> GateResult:
    """G14 (warn-only): box/lattice commensurability of a single
    crystal (tessellation.single.commensurability_misfit).

    A WARN means the rotated lattice has no period matching the box on at
    least one periodic axis — the box faces are then genuine self-boundary
    defects (cleaned by overlap removal like any GB).  Sometimes that is
    exactly what is wanted, so the gate never hard-fails."""
    from grainsmith.constants import COMMENSURATE_TOL

    if not misfits:
        return GateResult(
            "G14", True, measured=0.0,
            message="ok: no periodic axis — a free-standing single "
                    "crystal has no self-images to match")
    max_strain = max(strain for _, strain in misfits.values())
    detail = ", ".join(
        f"axis {'xyz'[ax]}: {delta:.3g} A (eps={strain:.3g})"
        for ax, (delta, strain) in sorted(misfits.items()))
    if max_strain <= COMMENSURATE_TOL:
        return GateResult(
            "G14", True, measured=max_strain,
            message=f"ok: lattice commensurate with the box — {detail} "
                    f"(tol {COMMENSURATE_TOL:g})")
    return GateResult(
        "G14", True, measured=max_strain,
        message=(f"WARN: lattice INCOMMENSURATE with the box — {detail} "
                 f"(tol {COMMENSURATE_TOL:g}). The periodic box faces are "
                 "self-boundary defects; use a commensurate box (integer "
                 "lattice multiples) and/or an identity orientation for a "
                 "perfect crystal."))


def gate_g15_phase_fractions(
        achieved: np.ndarray, targets: np.ndarray, names: list[str],
        granularity: float) -> GateResult:
    """G15 (warn-only): RE-MEASURED phase volume fractions vs
    the configured targets.

    *granularity* is the largest single grain volume over the box volume —
    moving one grain changes a fraction by at most that much, so deviation
    within it is unavoidable discretization while deviation beyond it
    means the assignment could be improved.  The V_max/V_box bound holds for
    the greedy LPT assignment; with ≥ 3 phases at extreme fractions and few
    grains the every-phase-≥1-grain guard in phases.assign_phases can
    force-feed a phase against the greedy choice and push max_dev above the
    bound (≲ 2× observed) — still only a WARN.  Never
    hard-fails: fraction resolution is a property of the grain count, not an
    error."""
    achieved = np.asarray(achieved, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    dev = np.abs(achieved - targets)
    max_dev = float(np.max(dev))
    detail = ", ".join(
        f"{nm}: {a:.4f} vs {t:.4f}"
        for nm, a, t in zip(names, achieved, targets, strict=True))
    level = "WARN" if max_dev > granularity else "ok"
    advice = ("; increase grains.number for finer fraction resolution"
              if level == "WARN" else "")
    return GateResult(
        gate="G15",
        passed=True,
        measured=max_dev,
        message=(f"{level}: achieved phase volume fractions — {detail} "
                 f"(max dev {max_dev:.4f}, granularity bound "
                 f"V_max/V_box = {granularity:.4f}{advice})"),
    )


def gate_g16_curvature(n_dropped: int, n_raw: int) -> GateResult:
    """G16: GB-curvature sanity (warn-only).

    Fraction of boundary samples dropped as degenerate
    (|∇φ| < CURV_GRAD_MIN) during level-set curvature evaluation.
    Flat runs compute nothing and report 0/0 → trivially ok.
    """
    from grainsmith.constants import CURV_G16_DROP_TOL

    frac = n_dropped / n_raw if n_raw else 0.0
    level = "WARN" if frac > CURV_G16_DROP_TOL else "ok"
    return GateResult(
        gate="G16",
        passed=True,
        measured=frac,
        message=(f"{level}: degenerate curvature samples "
                 f"{n_dropped}/{n_raw} ({frac:.3%})"),
    )


def gate_g17_doping_composition(per_dopant: list[dict]) -> GateResult:
    """G17: dopant composition + enrichment vs targets (warn-only).

    Same tolerance rule as G9 for the concentration (warn above 1
    percentage point of drift); achieved enrichment must lie within
    ±15 % relative of the target when segregation is active.
    """
    worst = 0.0
    parts = []
    warn = False
    for r in per_dopant:
        drift = abs(r["achieved"] - r["nominal"])
        worst = max(worst, drift)
        seg = ""
        if r.get("enrichment_target") is not None:
            e_t = r["enrichment_target"]
            e_a = r["enrichment_achieved"]
            if e_a is None or not np.isfinite(e_a):
                warn = True
                seg = ", E=n/a"
            else:
                rel = abs(e_a - e_t) / e_t
                if rel > 0.15:
                    warn = True
                seg = f", E={e_a:.2f}/{e_t:.2f}"
        if drift > 0.01:
            warn = True
        parts.append(f"{r['element']}: {r['achieved']:.4f}/"
                     f"{r['nominal']:.4f}{seg}")
    level = "WARN" if warn else "ok"
    return GateResult(
        gate="G17",
        passed=True,
        measured=worst,
        message=(f"{level}: dopant achieved/nominal fraction "
                 f"(and enrichment) — {'; '.join(parts)}"),
    )


def gate_g18_doping_geometry(n_violations: int, detail: str) -> GateResult:
    """G18: post-placement dopant min-distance re-verification (hard).

    An independent fresh neighbor query (G7-style) must find zero
    dopant–atom pairs closer than that dopant's min_distance.
    """
    advice = (" Lower min_distance, choose a different site preset, "
              "or lower the concentration.") if n_violations else ""
    return GateResult(
        gate="G18",
        passed=n_violations == 0,
        measured=n_violations,
        message=(f"{n_violations} dopant pair(s) closer than "
                 f"min_distance after placement — {detail}.{advice}"),
    )


def gate_g19_reassigned_fraction(reassigned_fraction: float) -> GateResult:
    """G19 (warn-only): perturbed_distance G5 repair-and-report severity.

    Unlike warp/weighted's G5 (hard TessellationError on macroscopic
    fragmentation, enforced at construction), perturbed_distance repairs
    disconnected fragments by majority-vote reassignment
    (tessellation/perturbed.py repair_connectivity) and never hard-fails
    on this gate — this is a QA signal on how MUCH repair was needed, not
    a pass/fail test of whether the structure is usable (it always is:
    repair either converges or the constructor itself raises
    TessellationError on non-convergence/isolated pockets, which never
    reaches summary.csv reporting).
    """
    from grainsmith.constants import REASSIGNED_FRACTION_WARN_TOL

    level = ("WARN" if reassigned_fraction > REASSIGNED_FRACTION_WARN_TOL
             else "ok")
    return GateResult(
        gate="G19",
        passed=True,
        measured=reassigned_fraction,
        message=(f"{level}: G5 repair reassigned "
                 f"{reassigned_fraction:.4%} of voxels (warn > "
                 f"{REASSIGNED_FRACTION_WARN_TOL:.0%}) — perturbed_distance "
                 "repair-and-report, not a hard connectivity gate."),
    )


def gate_g20_db_estimate(d_b: float | None) -> GateResult:
    """G20 (warn-only): perturbed_distance box-counting ROUGHNESS INDEX
    (tessellation/perturbed.py d_b_estimate / box_count_dimension), the
    G13 analogue for this method.

    Framing: this is deliberately called a
    "box-counting roughness index", not a "fractal-dimension estimate"
    — at MD-typical grain sizes, box-counting's ability to separate the
    Hurst exponent's effect on boundary roughness from finite-band/
    finite-size noise is a GRADIENT with octave count, not a sharp
    cliff: measured, the effect is indistinguishable from noise at the
    ~1.7-octave band ``examples/self_affine_gb/pdau_perturbed_self_affine.yaml``
    uses (η² ≈ 6%, p ≈ 0.14), while a synthetic-field Monte Carlo shows
    the effect becomes statistically SIGNIFICANT (p ≪ 0.01) by
    ~3.2 octaves but only reaches a conventionally LARGE effect size
    (η² > 80%) at ~5.5 octaves (≈50 nm grain diameter at an atomic-scale
    l_min) — significance and effect-size magnitude cross their
    respective thresholds at different octave counts, and both numbers
    matter for different purposes (see docs/physics.md §5b(g) for the
    full octave/η² table). Below that ~5.5-octave regime, ``d_b`` is
    therefore honest evidence of LOCAL cross-section roughness at the
    resolved scales — exactly what its box-counting FIT measures — not
    a converged, LARGE-effect-size fractal-dimension measurement
    directly comparable to Braun et al. 2020's D_b = 1.174 ± 0.004
    (measured over grains ≳45 μm, itself a consequence of their own
    protocol's minimum-grain-size criterion d_min = ε_max/0.4 with
    ε_min ≥ 3s). A comparison at that same effect-size regime needs a
    grain (and hence a synthesis band) at the upper end of the
    MD-accessible range — see ``examples/`` for a worked ≥50 nm/
    5.5-octave config and docs/physics.md §5b for the full argument.

    Never hard-fails regardless of grain size: a box-counting fit from a
    finite, noisy, single-realization boundary on a handful of sampled
    grains/sections is an ESTIMATE with documented bias, not a
    certificate — same philosophy as gate_g13_hurst. ``d_b``
    is None when no sampled (grain, section) combination yielded a
    usable estimate (reported as n/a, never fabricated).
    """
    if d_b is None:
        return GateResult(
            gate="G20", passed=True, measured=float("nan"),
            message=("n/a: no (grain, section) combination yielded a "
                     "usable box-counting fit (sampled grains too small "
                     "relative to the section resolution)."),
        )
    return GateResult(
        gate="G20", passed=True, measured=d_b,
        message=(f"box-counting roughness index D_b={d_b:.3f} (2D "
                 "cross-section perimeter fit over the synthesis band "
                 "[l_min, l_max], Braun et al. 2020 protocol; flat-"
                 "Voronoi limit is D_b=1.000). At MD-typical grain "
                 "sizes (below ~5.5 octaves of synthesis band) treat "
                 "this as a local-roughness diagnostic, not a converged, "
                 "large-effect-size fractal-dimension measurement; see "
                 "docs/physics.md §5b."),
    )


def gate_g21_gauss_bonnet(per_grain_totals) -> GateResult:
    """G21 (warn-only): per-grain Gauss-Bonnet face-interior residual —
    a coarse fidelity signal for the gb_curvature ESTIMATOR, not a
    judgment on the generated model.

    Only evaluated when ``analysis.gb_curvature`` is on (pipeline.py,
    alongside G16); *per_grain_totals* is
    ``analysis.curvature.per_grain_gauss_bonnet(curvature_res,
    tess.n_grains)`` — Σ_faces K·dA per grain over the FACE-INTERIOR
    samples ``analyze_curvature`` retains.

    G16 vs G21, and what a WARN here actually means
    -------------------------------------------------
    G16 asks "did we keep enough samples" (the degenerate-sample DROP
    fraction). G21 asks the complementary question: "how much of the
    excluded edge/vertex curvature budget is leaking into the kept,
    face-interior set". The Gauss-Bonnet theorem fixes
    ``∮_S K dA = 4π·(1 − g)`` exactly for any closed orientable surface
    S of genus g; a periodic-boundary grain is such a surface. The
    DESIGN INTENT is that a Voronoi-like grain's curvature is DOMINATED
    by its edges/vertices (angle-defect Dirac mass — e.g. a cube's
    eight corners alone already sum to 4π), which ``analyze_curvature``
    deliberately EXCLUDES (its stencils are undefined on a
    non-differentiable edge/vertex, §6.7 of that module), leaving a
    small face-interior remainder.

    IMPORTANT — that intent is NOT what this gate's own calibration
    sweep measures (full numbers in ``constants.CURV_G21_GAUSS_BONNET_
    TOL``): near-FLAT baselines (amplitude 0.01, six seeds) already
    cluster at 0.977-1.041x this tolerance — AT one topological unit,
    not "small" relative to it — and ordinary curved runs routinely
    reach 5-14x. The residual does NOT shrink toward 0 as
    ``analysis.voxel_grid`` is refined over the range tested, and a
    2-grain bicrystal with no triple junctions at all still measures
    8-13x, so this is not merely a many-grain or many-triple-junction
    artefact. A WARN here — the measured value exceeding one
    topological unit (4π) — is therefore ROUTINE on ordinary curved
    runs with the current estimator, including near-flat ones, not
    proof that something is newly wrong. Treat it as a standing, coarse
    caveat on the quantitative reliability of gb_curvature.csv's H/K
    columns for that run, worth a second look if it is unusually large
    or grows unexpectedly between otherwise-similar configurations —
    not as an actionable defect report by itself.

    Never hard-fails and never touches the atomistic output: G7 (min
    interatomic distance), G3/G4 (volume/topology) already certify the
    generated structure independently of this diagnostic. ``passed`` is
    always ``True``; the message and ``measured`` (the WORST — maximum
    |Σ K dA| — grain) carry the severity signal, mirroring
    gate_g16_curvature/gate_g19_reassigned_fraction's pattern for a
    tool-fidelity (not model-validity) concern.

    Parameters
    ----------
    per_grain_totals : array-like, (n_grains,) float64
        Σ_faces K·dA per grain (``analysis.curvature.
        per_grain_gauss_bonnet``); empty is treated as "no grains", never
        as an error (single-crystal / empty-adjacency runs).
    """
    from grainsmith.constants import CURV_G21_GAUSS_BONNET_TOL

    totals = np.asarray(per_grain_totals, dtype=np.float64)
    if totals.size == 0:
        return GateResult(
            gate="G21", passed=True, measured=0.0,
            message="ok: no grains with retained curvature samples "
                    "(empty adjacency).",
        )
    abs_totals = np.abs(totals)
    worst_grain = int(np.argmax(abs_totals))
    worst_value = float(abs_totals[worst_grain])
    n_violations = int(np.sum(abs_totals > CURV_G21_GAUSS_BONNET_TOL))
    level = "WARN" if n_violations > 0 else "ok"
    advice = (" — face-interior sampling leaked more than one topological "
              "unit of curvature on this run (routine on curved runs with "
              "the current estimator, see gate docstring/docs/gates.md "
              "G21 for calibration context); the atomistic output itself "
              "is unaffected (see G3/G4/G7)."
              if n_violations > 0 else "")
    return GateResult(
        gate="G21",
        passed=True,
        measured=worst_value,
        message=(f"{level}: {n_violations} grain(s) with face-interior "
                 f"|Σ K·dA| > 4π={CURV_G21_GAUSS_BONNET_TOL:.4f} "
                 f"(worst: grain {worst_grain}, |Σ K·dA|="
                 f"{worst_value:.4f}){advice}"),
    )


_G22_GRAM_MEMORY_FRACTION: float = 0.25
"""qa.py-internal policy for gate_g22_odf_fidelity's pre-flight memory
check -- NOT a constants.py entry, because it is not a physics tolerance:
it is the share of the run's resolved §13 per-allocation budget this one
WARN-only diagnostic is allowed to claim before it skips its own kernel
block rather than risk the diagnostic itself triggering an OOM (or
tripping ``orientation.odf.symmetrized_gram``'s own GrainsmithError guard,
which must never propagate out of a gate that only ever warns). A
"modest fraction" rather than the full budget: the Gram build is optional
evidence on top of the always-computed atomic drift, not the run's
primary allocation, so it should not be allowed to compete with (or
starve) whatever the tessellation/fill/analysis stages still need from
the same budget."""


def gate_g22_odf_fidelity(
    volumes: np.ndarray,
    pre_anneal_permutation: np.ndarray,
    quats: np.ndarray,
    sym_quats: np.ndarray,
    odf_drift_final_annealer: float | None,
    odf_drift_max: float | None,
    kernel_halfwidth_deg: float,
    null_samples: int,
    memory_limit_bytes: float,
    memory_limit_source: str,
) -> GateResult:
    """G22 (warn-only, NEVER fails): volume-weighted ODF fidelity of the
    FINAL structure, re-measured independently of the annealer's own
    bookkeeping (mirrors the G11/G12/G15 "never trust the optimizer's own
    claim" discipline). Evaluated in the ANALYSIS stage, once the final
    per-grain `volumes` array exists -- not inside the annealer, which
    only ever sees the PRE-fill tessellation.

    KEY FACT that makes this cheap (see ``pipeline._stage_mdf``, which
    returns ``quats[res.permutation]``): downstream of that reassignment,
    ``quats[i]`` is the orientation held by grain ``i`` -- the orientation
    INDEX equals the GRAIN index. So, with ``w = volumes / volumes.sum()``:

    * the volume-weighted measure of the FINAL assignment is just ``w``;
    * the count-weighted reference f_n is the uniform vector ``1/N``;
    * the volume-weighted measure of the PRE-annealing assignment is
      ``w[pre_anneal_permutation]`` (``pre_anneal_permutation`` is
      ``MDFResult.permutation``, the map the annealer itself applied).

    Proof sketch for that last line: writing ``perm`` for
    ``pre_anneal_permutation`` and ``quats_old``/``quats_new`` for the
    orientation array before/after ``_stage_mdf``'s reassignment,
    ``quats_new[i] = quats_old[perm[i]]``. The PRE-annealing assignment put
    weight ``w_i`` on ``quats_old[i]``; in the NEW (post-permutation) index
    space that same physical orientation ``quats_old[i]`` sits at whichever
    slot ``j`` satisfies ``perm[j] == i``, i.e. ``j = perm^-1(i)``. So the
    mass landing at NEW slot ``j`` is ``w[perm[j]]`` -- exactly
    ``w[pre_anneal_permutation]``, computed once, with no explicit inverse
    permutation ever needed.

    This function then computes, using ``grainsmith.orientation.odf``:

    * ``class_of, n_classes = orientation_classes(quats)`` -- exact-
      duplicate-merged orientation classes of the FINAL orientation set;
    * ``drift = atomic_drift(w, v_initial, class_of, n_classes)`` -- an
      INDEPENDENT re-measure of the same atomic drift the annealer itself
      tracked and reported as ``MDFResult.odf_drift_final``, cross-checked
      below;
    * ``n_eff = effective_sample_size(volumes)`` and
      ``gap = count_vs_volume_gap(volumes)`` -- always reported, cheap,
      assignment-independent context for how large a volume-vs-count gap
      this grain-size distribution can even produce (fact B in
      orientation/odf.py's module docstring);
    * and, only when it fits the memory guard below, the kernel-smoothed
      discrepancy ``mmd_vs_count``/``mmd_vs_initial`` against a symmetrized
      Gram matrix at ``kernel_halfwidth_deg``, plus ``null_mmd``'s null
      distribution for ``mmd_vs_count`` (median, 2.5th/97.5th percentiles,
      and -- informational only, see VALUE VS. QUANTILE below -- the
      empirical quantile ``mmd_vs_count`` itself falls at);
    * and, inside the same memory-guarded block, the QUOTIENT-space drift:
      ``atomic_drift`` evaluated on symmetry-merged classes
      (:func:`~grainsmith.orientation.odf.symmetry_classes`).  The physical
      ODF of a single-phase material lives on SO(3)/Sym, and quotienting is
      a pushforward (TV never grows), so this value is guaranteed <= the
      labelled-set ``drift`` above -- the sharper physical number, reported
      next to the controlled bound (orientation/odf.py, "SYMMETRY-QUOTIENT
      CLASSES").

    INTERPRETATION -- read this before reading the message. The atomic
    ``drift`` is the CONTROL quantity: bounding it during annealing is
    what LICENSES the claim that the volume-weighted ODF is preserved to
    within an explicit epsilon (Markov-contraction argument,
    orientation/odf.py). It is NOT, by itself, evidence of bias. Measured
    directly: on a 60-grain lognormal test case, an UNCONSTRAINED anneal's
    drift sits at the 89th percentile of what random (non-adversarial)
    permutations of the SAME grain volumes produce against the SAME
    orientation set (null median 0.427, p95 0.483) -- i.e. comfortably
    inside the bulk of "this is just what the count-vs-volume gap looks
    like for these grain sizes", not an outlier. Bias detection -- "did
    the annealer specifically concentrate volume onto a subset of
    orientations" -- is what ``mmd_vs_count`` versus its OWN null
    (``null_mmd``, sampled at the SAME uniform reference) is for; the two
    questions are independent and this gate's message must never conflate
    a large ``drift`` with a bias finding, or a small one with the absence
    of one.

    THE NULL TEST IS TWO-SIDED, AND THIS IS NOT OPTIONAL -- a real defect
    caught in review, worked example below. ``mmd_vs_count`` measures a
    DISTANCE (in kernel space) from the volume-weighted measure to the
    count-weighted reference f_n. An energy-independent (non-adversarial)
    assignment of these same grain volumes produces a CHARACTERISTIC
    distance to f_n -- landing in EITHER tail of that null, not only the
    upper one, is evidence that the assignment correlated orientation with
    grain size. Landing ANOMALOUSLY CLOSE to f_n is just as informative as
    landing anomalously far: it means the annealer's own search happened
    to settle on (or near) an unusually volume-balanced assignment, which
    is itself a property of that specific assignment, not a generic
    feature of "no bias occurred". A MID-RANGE quantile -- not "small" --
    is the genuine "nothing statistically resolvable" outcome.

    Worked example (a 60-grain lognormal orientation/adjacency/volume
    setup, cubic point group, reproduced exactly in
    tests/test_odf_gates.py's two-sided regression test): the SAME
    orientation set and grain volumes, annealed with no cap versus a tight
    ``odf_drift_max=0.02`` cap --

        no cap    : drift 0.4024, mmd_vs_count 0.6329 -- null median
                0.6943, [p2.5, p97.5] = [0.6389, 0.7555] -> WARN
                    (below p2.5; empirical quantile 1.6%)
        cap 0.02  : drift 0.0197, mmd_vs_count 0.6907 -- null median
                0.6916, [p2.5, p97.5] = [0.6423, 0.7683] -> ok
                    (inside the interval; empirical quantile 47.7%)

    The UNCAPPED run has the LARGER atomic drift by a factor of 20, yet its
    kernel-smoothed distance to f_n lands BELOW the central 95% interval's
    lower bound -- a ONE-SIDED test checking only
    ``mmd_vs_count > null_p97_5`` would have called this "ok" and flagged
    nothing, precisely backwards: this is the run whose search happened to
    settle unusually close to the count-weighted reference, a genuinely
    atypical outcome worth surfacing. The CAPPED run, despite its far
    smaller drift, sits comfortably inside the interval -- the boring,
    expected "nothing resolvable" result. Checking only the upper tail
    would have missed the first case and, in general, would silently
    misread "anomalously close to uniform" as "no bias" on any run whose
    search happens to land there.

    VALUE VS. QUANTILE -- a real defect caught in review, and why the
    verdict is decided on the VALUE against ``[null_p2_5, null_p97_5]``
    rather than on the empirical quantile ``null_quantile = mean(null <=
    mmd_vs_count)``, even though the two framings agree almost everywhere.
    A run was observed reporting ``mmd_vs_count=0.3439`` against a printed
    interval of ``[0.3494, ...]`` while the SAME message asserted
    "inside the interval" -- 0.3439 is plainly below 0.3494. The two
    framings disagree exactly at the tail boundary because they have
    different resolution: the empirical quantile is quantized to
    ``1/len(null)`` (about 0.4% per sample at the default
    ``odf_null_samples=256``), so a value sitting a hair below ``p2_5``
    can still read as an empirical quantile of, say, 2.7% -- comfortably
    "inside 2.5%-97.5%" under the quantile framing while being, in raw
    value terms, below the very percentile that framing itself reports.
    Deciding the verdict on the SAME value-vs-interval comparison the
    message prints removes the possibility of that contradiction by
    construction; ``null_quantile`` is still reported, but as
    INFORMATIONAL CONTEXT only, never as the decision variable. A run
    whose value sits within roughly one sample's worth of either bound
    should be read as BORDERLINE rather than as a clean verdict either
    way -- widening ``odf_null_samples`` sharpens that boundary if it
    matters for a specific run.

    ``mmd_vs_initial`` is DESCRIPTIVE ONLY (no null -- see
    :func:`~grainsmith.orientation.odf.null_mmd`'s own docstring for why a
    null built for ``mmd_vs_count`` must never be reused for this), but it
    is the more DIRECTLY interpretable of the two kernel numbers: "how far
    did the anneal move the kernel-smoothed volume-weighted ODF from where
    it started". In the worked example above it behaves exactly as
    expected -- 1.0359 uncapped versus 0.1522 capped -- tracking the
    atomic drift's own 0.4024-vs-0.0197 ratio far more directly than
    ``mmd_vs_count`` does (which, being a distance to a FIXED external
    reference rather than to the run's own starting point, is not
    monotonic in drift at all, as the worked example itself demonstrates).

    A SECOND floor worth reading `drift`/`mmd_vs_count` against: ``n_eff``
    can collapse well below the grain count on its own, independent of
    annealing. Measured directly on the reference end-to-end scenario (24
    grains, ``size_distribution: {type: lognormal, sigma_log: 0.5}``): a
    seed landing at ``n_eff=4.25`` (of 24) is unremarkable at that sigma --
    a handful of grains already dominate the total volume before the
    annealer ever runs (fact B, orientation/odf.py). When ``n_eff`` is
    that small, the volume-weighted ODF is effectively a measure over a
    handful of atoms; read ``drift`` against that floor, not against zero
    -- a "large-looking" drift on a small ``n_eff`` run may simply reflect
    how few grains are carrying most of the mass, not an unusually
    aggressive reassignment.

    VERDICT (``passed`` is always ``True`` -- this is a diagnostic, never a
    hard gate):

    * WARN when the kernel block ran and ``mmd_vs_count`` itself (the
      VALUE, not the empirical quantile -- see VALUE VS. QUANTILE above)
      falls outside the central 95% interval -- i.e. ``mmd_vs_count <
      null_p2_5 or mmd_vs_count > null_p97_5`` (TWO-SIDED, see above) --
      the final assignment carries a statistically resolvable
      orientation-size correlation, in EITHER direction, beyond what a
      neutral assignment of these same volumes would produce.
    * WARN when a cap was configured (``odf_drift_max`` not None) and
      ``drift > odf_drift_max + 1e-9``. The ``1e-9`` slack is REQUIRED, not
      cosmetic: ``atomic_drift`` sums per-class volume masses in a
      different summation order than the annealer's own incremental
      ``DriftTracker`` bookkeeping, so an exactly-satisfied cap can read a
      few ULP high on independent re-measurement (measured directly) --
      an exact ``>`` comparison would occasionally WARN on a run that in
      fact satisfied its cap.
    * otherwise "ok".

    ``measured`` is ``mmd_vs_count`` when the kernel block ran, else
    ``drift``.

    MEMORY (§13): ``orientation.odf.symmetrized_gram`` allocates an N×N
    float64 matrix (8*N^2 bytes) and RAISES ``GrainsmithError`` above the
    resolved §13 budget -- correct for a primary allocation, but a WARN-
    only diagnostic must never abort a run that would otherwise complete
    cleanly. So this gate estimates ``8*N*N`` bytes itself, BEFORE ever
    calling ``symmetrized_gram``, and skips the whole kernel block --
    reporting the exact atomic quantities above regardless, plus the skip
    reason (N and the estimated size) -- whenever that estimate exceeds
    ``_G22_GRAM_MEMORY_FRACTION`` of the run's resolved
    ``memory_limit_bytes`` (threaded through exactly the way the rest of
    ``pipeline.py`` already does: ``tess.memory_limit_bytes`` /
    ``tess.memory_limit_source``, set by ``pipeline._apply_runtime_limits``
    -- not a new mechanism). Every other exception this function might hit
    is also caught and turned into an "ok"-level diagnostic-failure message
    rather than propagated: nothing in this gate may abort a run.

    Also cross-checks ``odf_drift_final_annealer`` (``MDFResult.
    odf_drift_final``, the annealer's OWN claim) against the re-measured
    ``drift`` and notes a disagreement above 1e-9 in the message -- the
    same "do not trust the optimizer's own claim" discipline G12 already
    applies to the annealer's χ² claim.

    Parameters
    ----------
    volumes : (N,) float64
        Final (post-fill-stage) per-grain volumes; grain index == the
        orientation index of ``quats`` (see KEY FACT above).
    pre_anneal_permutation : (N,) int
        ``MDFResult.permutation`` -- the map ``_stage_mdf`` applied to
        reach the FINAL ``quats`` from the sampler's own pre-annealing
        orientation array.
    quats : (N, 4) float64
        FINAL grain orientations (post-``_stage_mdf`` reassignment).
    sym_quats : (Nsym, 4) float64
        Crystal point-group symmetry operators (right-sided orbit).
    odf_drift_final_annealer : float | None
        ``MDFResult.odf_drift_final`` -- the annealer's own claim, or
        ``None`` when it was never tracked (should not occur once
        ``pipeline._stage_mdf`` always threads ``grain_volumes`` through,
        but tolerated defensively).
    odf_drift_max : float | None
        ``MdfTargetConfig.odf_drift_max`` -- the configured cap, or
        ``None`` when the drift was only measured, never constrained.
    kernel_halfwidth_deg : float
        ``MdfTargetConfig.odf_kernel_halfwidth_deg``.
    null_samples : int
        ``MdfTargetConfig.odf_null_samples``.
    memory_limit_bytes, memory_limit_source : float, str
        The run's resolved §13 budget (``tess.memory_limit_bytes`` /
        ``tess.memory_limit_source``).
    """
    try:
        from grainsmith.orientation.odf import (
            atomic_drift,
            count_vs_volume_gap,
            effective_sample_size,
            mmd,
            null_mmd,
            orientation_classes,
            symmetrized_gram,
            symmetry_classes,
            vp_halfwidth,
            vp_kappa,
        )

        v = np.asarray(volumes, dtype=np.float64).reshape(-1)
        n = len(v)
        w = v / float(np.sum(v))
        perm0 = np.asarray(pre_anneal_permutation, dtype=np.intp).reshape(-1)
        class_of, n_classes = orientation_classes(quats)
        v_initial = w[perm0]
        drift = atomic_drift(w, v_initial, class_of, n_classes)
        n_eff = effective_sample_size(v)
        gap = count_vs_volume_gap(v)
    except Exception as exc:  # pragma: no cover - diagnostic must never abort
        return GateResult(
            gate="G22", passed=True, measured=float("nan"),
            message=("ok: G22 could not be evaluated "
                     f"({type(exc).__name__}: {exc})."),
        )

    cap_note = ("no cap" if odf_drift_max is None
                else f"cap {float(odf_drift_max):g}")
    parts = [
        f"n_eff={n_eff:.2f} (of {n} grains)",
        f"count-vs-volume gap={gap:.4f}",
        f"drift={drift:.6f} ({cap_note})",
    ]

    disagreement_note = ""
    if odf_drift_final_annealer is not None:
        try:
            disagreement = abs(drift - float(odf_drift_final_annealer))
        except (TypeError, ValueError):
            disagreement = float("nan")
        if math.isfinite(disagreement) and disagreement > 1e-9:
            disagreement_note = (
                "; NOTE: annealer's own odf_drift_final="
                f"{float(odf_drift_final_annealer):.6f} disagrees with "
                f"this re-measurement by {disagreement:.3e} (> 1e-9)"
            )

    warn = odf_drift_max is not None and drift > float(odf_drift_max) + 1e-9
    measured = drift

    est_bytes = 8.0 * n * n
    budget = _G22_GRAM_MEMORY_FRACTION * float(memory_limit_bytes)
    if est_bytes > budget:
        parts.append(
            "kernel discrepancy NOT evaluated: the symmetrized Gram matrix "
            f"for N={n} orientations is estimated at {est_bytes / 1e9:.3f} "
            f"GB, above {_G22_GRAM_MEMORY_FRACTION:.0%} of the resolved "
            f"{memory_limit_source} memory budget "
            f"({float(memory_limit_bytes) / 1e9:.3f} GB) reserved for "
            "this diagnostic"
        )
    else:
        try:
            from grainsmith.constants import ODF_QUOTIENT_TOL_DEG

            kappa = vp_kappa(float(kernel_halfwidth_deg))
            gram = symmetrized_gram(
                quats, sym_quats, kappa,
                memory_limit_bytes=memory_limit_bytes,
                memory_limit_source=memory_limit_source,
            )
            uniform = np.full(n, 1.0 / n, dtype=np.float64)
            mmd_vs_count = mmd(w, uniform, gram)
            mmd_vs_initial = mmd(w, v_initial, gram)
            null = null_mmd(gram, v, n_samples=int(null_samples))
            null_median = float(np.median(null))
            null_p2_5 = float(np.percentile(null, 2.5))
            null_p97_5 = float(np.percentile(null, 97.5))
            # Informational context only -- see the "VALUE, not empirical
            # quantile" note below for why this must NEVER be the decision
            # variable.
            null_quantile = float(np.mean(null <= mmd_vs_count))
            # TWO-SIDED: either tail is evidence of an assignment-induced
            # orientation-size correlation -- landing anomalously CLOSE to
            # the count-weighted reference is just as informative as
            # landing anomalously far (see docstring's worked example: a
            # one-sided upper-tail-only check reads the LOW-tail case as
            # "ok" when it is in fact the more unusual assignment).
            #
            # VALUE, not empirical quantile, decides the verdict: deciding
            # on `null_quantile < 0.025 or > 0.975` while REPORTING
            # value-vs-[p2.5, p97.5] can disagree with the printed interval
            # right at the boundary -- with `len(null)` samples the
            # empirical quantile only resolves to 1/len(null) (~0.4% at the
            # default 256), so a value a hair below p2.5 can still show an
            # empirical quantile above 2.5% (or vice versa), producing a
            # message that names a value BELOW the stated lower bound while
            # asserting "inside" it (a real, observed contradiction this
            # fixes). Deciding directly on the value against the SAME
            # interval the message prints makes the sentence and the
            # verdict structurally unable to disagree.
            kernel_warn = mmd_vs_count < null_p2_5 or mmd_vs_count > null_p97_5
            parts.append(
                f"kernel (halfwidth={vp_halfwidth(kappa):.6g} deg, "
                f"requested_halfwidth={float(kernel_halfwidth_deg):g} deg, "
                f"kappa={kappa:g}, "
                f"{len(null)} null samples): mmd_vs_count="
                f"{mmd_vs_count:.4f} vs. central 95% interval "
                f"[{null_p2_5:.4f}, {null_p97_5:.4f}] (median "
                f"{null_median:.4f}, empirical quantile {null_quantile:.1%}"
                f" -- informational) -- "
                f"{'OUTSIDE' if kernel_warn else 'inside'} the interval, "
                f"mmd_vs_initial={mmd_vs_initial:.4f} (descriptive only, "
                "no null -- the more directly interpretable "
                "how-far-did-the-anneal-move-it number)"
            )
            q_class_of, q_n_classes = symmetry_classes(quats, sym_quats)
            drift_quotient = atomic_drift(
                w, v_initial, q_class_of, q_n_classes)
            parts.append(
                f"quotient-space drift (symmetry-merged classes, tol "
                f"{ODF_QUOTIENT_TOL_DEG:g} deg) = {drift_quotient:.6f} -- "
                "the physical ODF lives on SO(3)/Sym and merging can only "
                "shrink TV, so this is the sharper physical number and the "
                "atomic drift above is its guaranteed upper bound "
                "(orientation/odf.py, 'SYMMETRY-QUOTIENT CLASSES')"
            )
            if kernel_warn:
                warn = True
            measured = mmd_vs_count
        except Exception as exc:  # pragma: no cover - diagnostic must never abort
            parts.append(
                "kernel discrepancy NOT evaluated: unexpected error while "
                f"building it ({type(exc).__name__}: {exc})"
            )

    level = "WARN" if warn else "ok"
    return GateResult(
        gate="G22", passed=True, measured=measured,
        message=f"{level}: " + "; ".join(parts) + disagreement_note,
    )


def gate_g23_sigma3_consistency(
    misorientation_deg: np.ndarray,
    area_a2: np.ndarray,
    csl_sigma: list[str] | None,
) -> GateResult:
    """G23 (report-only, NEVER trips -- ``passed`` is always ``True`` and
    the message never carries a WARN): puts two Sigma3 boundary-area
    fractions that are NOT interchangeable side by side, for any single-
    phase run that reaches the MDF-histogram block -- independent of
    whether ``orientation.mdf_target`` is even configured, since the gap
    below is informative either way.

    * ``sigma3_angle_window_area_fraction`` -- area-weighted fraction of
      boundaries with ``|misorientation_deg - 60| <= 15/sqrt(3)`` (the
      Sigma3 Brandon ANGULAR window), computed directly from the per-
      boundary ``misorientation_deg``/``area_A2`` arrays already in scope
      in the pipeline's MDF-histogram block -- NOT from the binned
      histogram.  A boundary with a non-finite (NaN) ``misorientation_deg``
      (interphase boundaries, which carry no single-phase misorientation)
      compares False against the window bound, so it contributes its area
      to the total-area DENOMINATOR but never to the in-window numerator --
      a conservative dilution (the record counts as not-in-window, never as
      in-window, and its angle is never silently coerced to a number).
      This matches the validation campaign's ``brandon``
      computation exactly (local tooling, not published with this
      repository): same window test, same ``nansum(area)`` total
      denominator.  (The CSL side has no analogous subtlety: ``csl_sigma``
      labels are strings -- an unclassified or interphase boundary carries
      ``""``, never a NaN -- and both fractions share the same total-area
      denominator.)
    * ``sigma3_csl_area_fraction`` -- area-weighted fraction of boundaries
      whose ``csl_sigma`` is exactly ``"3"`` (the TRUE Sigma3 class: angle
      AND <111> axis, via the Brandon deviation from the exact CSL
      rotation, ``analysis/boundaries.py``). Only evaluated when
      ``analysis.csl`` is True -- ``csl_sigma`` is ``None`` (this fraction
      is NOT evaluated) precisely when ``analysis.csl`` is False; in that
      case this is reported as NOT EVALUATED, never as ``0.0``, since
      ``0.0`` would be a false claim that no Sigma3 boundaries exist (CSL
      classification was simply never attempted).
    * the ratio window/csl, reported only when the CSL value is available,
      finite, and non-zero (undefined/uninformative otherwise).

    WHY THE GAP IS EXPECTED, NOT A DEFECT: when ``mdf_target`` is
    configured, the annealing objective (``orientation/mdf.py``) is the
    ONE-DIMENSIONAL, area-weighted marginal of the disorientation ANGLE
    only. Placing a target in the Sigma3 Brandon angular window therefore
    enforces only the angular condition that is NECESSARY for Sigma3 --
    the <111> axis condition CSL classification also requires never
    enters the energy at all, so the annealer has no mechanism to close
    that gap even at full convergence. Recorded campaign values for scale
    (MDF validation campaign, case M2 -- local tooling, not published
    with this repository):
    the angle-window area fraction runs 0.24-0.44 while the TRUE CSL
    Sigma3 area fraction stays 0.02-0.06 -- roughly an order of magnitude
    apart, by construction, not by a bug in either measurement.

    ``measured`` is the PAIR ``(angle_window_fraction, csl_fraction |
    None)`` -- not a single scalar like most other gates -- so
    ``pipeline._summary_rows`` can write the machine-readable
    ``sigma3,angle_window_area_fraction`` / ``sigma3,csl_area_fraction``
    summary.csv rows straight off this field, with an explicit empty cell
    for the CSL side when it was not evaluated, rather than string-
    splitting ``message``.
    """
    ang = np.asarray(misorientation_deg, dtype=np.float64).reshape(-1)
    area = np.asarray(area_a2, dtype=np.float64).reshape(-1)
    a_total = float(np.nansum(area))
    from grainsmith.constants import CSL_BRANDON_FACTOR

    brandon3 = CSL_BRANDON_FACTOR / math.sqrt(3.0)
    in_window = np.abs(ang - 60.0) <= brandon3
    window_frac = (float(np.nansum(area[in_window]) / a_total)
                   if a_total > 0.0 else float("nan"))

    ratio: float | None = None
    if csl_sigma is None:
        csl_frac: float | None = None
        csl_part = "CSL sigma3 area fraction: NOT EVALUATED (analysis.csl is False)"
    else:
        is_sigma3 = np.asarray([s == "3" for s in csl_sigma])
        csl_frac = (float(np.nansum(area[is_sigma3]) / a_total)
                    if a_total > 0.0 else float("nan"))
        csl_part = f"CSL sigma3 area fraction {csl_frac:.4f}"
        if (math.isfinite(csl_frac) and csl_frac != 0.0
                and math.isfinite(window_frac)):
            ratio = window_frac / csl_frac

    ratio_part = (f"; ratio window/csl={ratio:.3f}" if ratio is not None
                  else "")
    return GateResult(
        gate="G23",
        passed=True,
        # (angle_window_fraction, csl_fraction | None) -- a pair, not a
        # single scalar like most other gates: `_summary_rows` reads both
        # numbers straight off this field so the CSL-not-evaluated case can
        # write an explicit empty cell rather than string-splitting the
        # message (see pipeline.py's summary-row assembly, module `E` of
        # the STEP 3b spec).
        measured=(window_frac, csl_frac),
        message=(
            f"angle-window Sigma3 area fraction {window_frac:.4f} "
            f"(|misorientation-60 deg| <= 15/sqrt(3) deg) vs {csl_part}"
            f"{ratio_part} -- EXPECTED gap, not a defect: the annealing "
            "objective (when mdf_target is configured) is the angle "
            "marginal only, the <111> axis condition true CSL "
            "classification also requires is absent from that objective "
            "(campaign case M2: angle-window 0.24-0.44 vs true CSL "
            "Sigma3 0.02-0.06)."
        ),
    )


def gate_g24_final_volume_targets(
    volumes: np.ndarray, target_volumes: np.ndarray,
) -> GateResult:
    """G24 (warn-only): RE-MEASURED FINAL (post-warp) per-grain volume
    error vs. the ``grains.size_distribution`` targets -- closes a real
    hole left by G11: ``gate_g11_volume_targets`` is evaluated on the
    UNWARPED power base (its own docstring already admits the realized,
    warped cells deviate from that), but nothing ever measured that
    deviation until now.

    Run this exactly when the geometry changed AFTER the volume fit --
    i.e. when ``sdot_res is not None and sdot_res.tess is not tess``
    (``pipeline.py``): for flat + ``size_distribution`` those are the SAME
    object (the power diagram IS the final tessellation, ``pipeline.py``
    already comments this at the G11 call site), so this gate correctly
    never fires there and can never double-report G11 on identical
    volumes. Only warp + ``size_distribution`` reaches this gate.

    Compares the FINAL per-grain ``volumes`` (measured in the analysis
    stage, after fill/overlap -- the same array ``grains.csv`` reports)
    against ``sdot_res.target_volumes``; ``measured`` is the max relative
    error, the message also reports the mean. See
    ``constants.WARP_VOLUME_G24_TOL`` for the full calibration this
    threshold is based on -- summarized: the realized post-warp error
    (2-4% at the calibrated amplitude/grain-count combinations) is two-
    plus orders of magnitude above both the unwarped-base G11 error
    (~1e-4-1e-5) and the voxel-discretisation floor alone (~0.2-0.4%), so
    0.10 flags genuinely large warp-driven displacement, not measurement
    noise; ``size_distribution.vol_tol`` (default 1e-3) is deliberately
    NOT reused here -- it sits below the noise floor and would false-fail
    on every warped run.

    The message ALWAYS states plainly that this compares against the
    UNWARPED-base targets (the same targets G11 checked the base against),
    so a reader cannot mistake this for a second G11 reading.
    """
    from grainsmith.constants import WARP_VOLUME_G24_TOL

    volumes = np.asarray(volumes, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_volumes, dtype=np.float64).reshape(-1)
    if np.any(targets <= 0.0):
        n_bad = int(np.sum(targets <= 0.0))
        return GateResult(
            gate="G24", passed=True, measured=float("nan"),
            message=(f"WARN: {n_bad} target grain volume(s) are "
                     "non-positive -- cannot evaluate relative volume "
                     "error (degenerate target)."),
        )
    rel = np.abs(volumes - targets) / targets
    max_rel = float(np.max(rel))
    mean_rel = float(np.mean(rel))
    level = "WARN" if max_rel > WARP_VOLUME_G24_TOL else "ok"
    return GateResult(
        gate="G24",
        passed=True,
        measured=max_rel,
        message=(f"{level}: post-warp per-grain volume error vs the "
                 "UNWARPED-base size_distribution targets (NOT the same "
                 f"comparison as G11) -- max relative error {max_rel:.4f}, "
                 f"mean {mean_rel:.4f} (tol {WARP_VOLUME_G24_TOL:g})"),
    )


def gate_g25_component_fidelity(
    component_weights: np.ndarray,
    component_of_pre: np.ndarray,
    component_of_final: np.ndarray,
    volumes: np.ndarray,
    basis: str,
) -> GateResult:
    """G25 (report-only, NEVER trips -- ``passed`` is always ``True`` and
    the message never carries a WARN): for the ``odf_components`` scheme,
    puts each texture component's CONFIGURED weight next to the fractions
    the sampler and any later assignment annealing actually realise, so
    the count-vs-volume gap behind the volume-weighted-ODF criticism is
    visible directly rather than only inferable.

    ``component_of_pre`` is the grain -> component label BEFORE any MDF
    annealing (``odf_components``'s own return, ``weight_basis`` "count"
    or "volume" -- see ``orientation.samplers.odf_components`` and
    ``orientation.odf.volume_balanced_partition``); ``component_of_final``
    is the SAME labelling permuted by the annealer, i.e.
    ``component_of_final[i] == component_of_pre[perm][i]`` where ``perm``
    is ``MDFResult.permutation`` (identity when no ``mdf_target`` is
    configured). ``basis`` is ``config.orientation.component_weight_basis``,
    recorded verbatim so a reader does not have to cross-reference
    config.yaml to know which realisation mechanism produced the numbers
    below.

    Five numbers per component ``k``:

    * ``w_cfg``          -- the configured ``weight``, normalised;
    * ``f_count``         -- the realised GRAIN-COUNT fraction, reported
      ONCE (not pre/post): a permutation can never change a ``bincount``,
      so ``component_of_pre`` and ``component_of_final`` give the exact
      same count fraction by construction -- see KEY FACT below;
    * ``f_vol_pre``       -- the VOLUME fraction realised by
      ``component_of_pre`` (:func:`~grainsmith.orientation.odf.
      component_volume_fractions`) -- what the sampler itself produced,
      before any annealing;
    * ``f_vol_post``      -- the VOLUME fraction realised by
      ``component_of_final`` -- the physically meaningful, FINAL share of
      the volume-weighted ODF;
    * ``mean_volume_A3``  -- mean volume of the grains ``component_of_pre``
      places in component ``k`` (``nan`` if the component received zero
      grains). Measured on the PRE-anneal labelling specifically because
      the size-component correlation this reports is a property of the
      SAMPLER assignment itself (``volume_balanced_partition``'s honest-
      cost paragraph: "the largest grain deterministically lands in the
      largest-deficit component"), so it is kept unmixed with whatever
      redistribution the annealer performs afterward.

    KEY FACT (why ``f_count`` is not split pre/post): as ``i`` ranges over
    ``0..n-1``, ``perm[i]`` ranges over the same set (a permutation is a
    bijection), so ``{component_of_final[i]}_i`` and
    ``{component_of_pre[j]}_j`` are the SAME multiset of component labels
    -- their bincounts, hence ``f_count``, are identical.

    ``measured`` (``None`` if the gate could not be evaluated) is::

        {"basis": basis,
         "per_component": [(w_cfg, f_count, f_vol_pre, f_vol_post,
                            mean_volume_A3), ...],
         "tv_cfg_vs_pre":  0.5*sum(abs(f_vol_pre  - w_cfg)),
         "tv_pre_vs_post": 0.5*sum(abs(f_vol_post - f_vol_pre)),
         "tv_cfg_vs_post": 0.5*sum(abs(f_vol_post - w_cfg))}

    so ``pipeline._summary_rows`` can write machine-readable rows straight
    off this field.

    THE BOUND, AND EXACTLY WHEN IT HOLDS: the component labelling maps
    orientation INDICES to components. Whenever that labelling is
    CONSTANT on every orientation class -- which is AUTOMATIC in the
    generic case where all sampled orientations are distinct, since
    every class is then a singleton -- the component-volume measure is a
    pushforward of the class-volume measure, and total variation never
    grows under a pushforward. Therefore ``tv_pre_vs_post`` <= the atomic
    drift G22 reports, and hence <= ``odf_drift_max`` whenever a cap is
    configured: bounded generically, not merely measured. Measured
    confirmation on the reference scenario with ``odf_drift_max`` 0.02:
    ``tv_pre_vs_post`` 0.011141 <= atomic drift 0.019966 <= cap 0.02.

    The bound FAILS only in the degenerate case where that condition is
    violated: two zero-spread ``odf_components`` entries with IDENTICAL
    ``euler_bunge_deg`` are accepted by ``resolve_config``, so two
    bit-identical orientations can land in DIFFERENT components; swapping
    those two grains gives ``atomic_drift == 0.0`` (G22 and
    ``odf_drift_max`` both operate on orientation CLASSES -- two grains
    sharing an orientation are the same class regardless of which
    component either was labelled -- so the swap is invisible to them)
    while the component-volume TV can be as large as 0.8 for a
    two-component, unequal-volume pair. ``odf_components`` does not
    guarantee the component labelling is CONSTANT on every orientation
    class (two grains sharing a class always share a component), so
    this gate still MEASURES ``tv_pre_vs_post`` directly (above) rather
    than relying on the bound alone. Never warns: no universal threshold
    exists -- basis "count" has a gap set by the grain-size distribution
    by design, basis "volume" is exact at the sampler and only drifts
    through whatever annealing ran afterward -- the intended use is
    transparency (the numbers land in summary.csv and in this message),
    not thresholding.
    """
    try:
        from grainsmith.errors import ConfigError
        from grainsmith.orientation.odf import component_volume_fractions

        w_cfg = np.asarray(component_weights, dtype=np.float64).reshape(-1)
        w_cfg = w_cfg / float(np.sum(w_cfg))
        n_comp = len(w_cfg)
        comp_pre = np.asarray(component_of_pre, dtype=np.intp).reshape(-1)
        comp_post = np.asarray(component_of_final, dtype=np.intp).reshape(-1)
        v = np.asarray(volumes, dtype=np.float64).reshape(-1)
        n = len(v)
        if len(comp_pre) != n or len(comp_post) != n:
            raise ConfigError(
                "component_of_pre/component_of_final and volumes must "
                f"have the same length; got {len(comp_pre)}, "
                f"{len(comp_post)}, {n}.")
        counts = np.bincount(comp_pre, minlength=n_comp)
        f_count = counts.astype(np.float64) / n
        f_vol_pre = component_volume_fractions(comp_pre, v, n_comp)
        f_vol_post = component_volume_fractions(comp_post, v, n_comp)
        vol_sums = np.bincount(comp_pre, weights=v, minlength=n_comp)
        mean_vol = np.full(n_comp, np.nan, dtype=np.float64)
        nonzero = counts > 0
        mean_vol[nonzero] = vol_sums[nonzero] / counts[nonzero]
    except Exception as exc:  # a diagnostic must never abort a run
        return GateResult(
            gate="G25", passed=True, measured=None,
            message=("ok: G25 could not be evaluated "
                     f"({type(exc).__name__}: {exc})."),
        )

    tv_cfg_vs_pre = float(0.5 * np.sum(np.abs(f_vol_pre - w_cfg)))
    tv_pre_vs_post = float(0.5 * np.sum(np.abs(f_vol_post - f_vol_pre)))
    tv_cfg_vs_post = float(0.5 * np.sum(np.abs(f_vol_post - w_cfg)))

    per = "; ".join(
        f"component {k}: configured {w_cfg[k]:.4f}, count {f_count[k]:.4f} "
        f"({int(counts[k])}/{n}), volume pre {f_vol_pre[k]:.4f}, volume "
        f"post {f_vol_post[k]:.4f}, mean volume {mean_vol[k]:.6g} A^3"
        for k in range(n_comp)
    )
    per_component = tuple(
        (float(w_cfg[k]), float(f_count[k]), float(f_vol_pre[k]),
         float(f_vol_post[k]), float(mean_vol[k]))
        for k in range(n_comp)
    )
    measured = {
        "basis": basis,
        "per_component": per_component,
        "tv_cfg_vs_pre": tv_cfg_vs_pre,
        "tv_pre_vs_post": tv_pre_vs_post,
        "tv_cfg_vs_post": tv_cfg_vs_post,
    }
    return GateResult(
        gate="G25", passed=True, measured=measured,
        message=(
            f"ok: texture-component fidelity (basis={basis}) -- {per}. "
            f"TV(cfg,pre)={tv_cfg_vs_pre:.4f}, "
            f"TV(pre,post)={tv_pre_vs_post:.4f}, "
            f"TV(cfg,post)={tv_cfg_vs_post:.4f}."
        ),
    )


def gate_g26_odf_weighting_floor(
    volumes: np.ndarray,
    n_atoms: np.ndarray,
    phase_of: np.ndarray | None = None,
    phase_names: list[str] | None = None,
) -> GateResult:
    r"""G26 (report-only, NEVER trips -- ``passed`` is always ``True`` and
    the message never carries a WARN, and no exception may ever escape:
    every code path below is wrapped so a broken input degrades to an
    "ok: could not be evaluated" message rather than aborting the run):
    measures the ATOMISTIC DISCRETISATION FLOOR of the volume-weighted
    ODF.

    grainsmith controls and exports the volume-weighted ODF using
    TESSELLATION grain volumes: ``odf_mtex.txt`` writes
    ``volumes / sum(volumes)`` as its weight column (``pipeline.py``),
    gate **G22** re-measures drift on those same volumes, and
    ``mdf_target.odf_drift_max`` caps it. But the artefact actually shipped
    to an MD user is ATOMISTIC, and the ODF a downstream simulation
    experiences is weighted by the number of ATOMS in each grain, not by
    its polyhedral/voxel volume. This gate puts those two weightings side
    by side and reports the gap -- it is what makes the ``odf_drift_max``
    guarantee honest: that guarantee is about the tessellation-volume
    weighting only, and this gate measures how much of it survives into
    the atomistic export.

    THIS IS A DISCRETISATION FLOOR, NOT AN ERROR. An atomistic structure
    with *n* atoms in a grain cannot resolve a volume-weighted ODF finer
    than this floor -- no choice of ``odf_drift_max`` changes that, since
    the annealer and G22 both operate on the CONTINUOUS tessellation-volume
    weighting, while the export is always discrete atom counts.

    MEASURED (reference scenario: 24 grains, box 45 A, ``size_distribution``
    lognormal ``sigma_log`` 0.5, two texture components at weight 0.5, seed
    4242, cubic Cu; tessellation-volume weights -- what ``odf_mtex.txt``
    exports and what ``odf_drift_max`` controls -- vs. the ATOM-COUNT
    weights actually present in the exported structure, same config, box
    size varied)::

        atoms/grain    TV(volume vs atom-count)    max per-grain |dw|
              275               0.027418                 0.009107
             1092               0.018910                 0.006912
             3279               0.013133                 0.004724
             9191               0.008633                 0.003126

    Fitted exponent -0.33, i.e. the floor decays as
    ``(atoms per grain)^(-1/3)`` -- a SURFACE effect (every atom within one
    interatomic spacing of a grain boundary is only probabilistically
    assigned to one side or the other, and that shell's atom count is a
    fixed fraction of the grain's SURFACE, which scales as ``1/L`` of its
    volume) -- NOT Poisson counting noise, which would give the steeper
    ``(atoms per grain)^(-1/2)``. At 275 atoms per grain the floor (0.027)
    EXCEEDS a typical ``odf_drift_max`` of 0.02: a drift cap tighter than
    this floor buys nothing that survives into the exported structure --
    that is the reason this gate exists.

    Per (sub)population evaluated, this reports:

    * ``tv``    -- ``0.5 * sum(abs(w_vol - w_atom))``, the total-variation
      distance between ``w_vol = volumes / sum(volumes)`` (what
      ``odf_mtex.txt`` exports) and ``w_atom = n_atoms / sum(n_atoms)``
      (what the exported atomistic structure actually realises);
    * ``max_dw`` -- ``max(abs(w_vol - w_atom))``, the single largest
      per-grain weight discrepancy;
    * mean atoms per grain -- context for how far into the
      ``(atoms/grain)^(-1/3)`` floor curve this run sits.

    NOT EVALUATED (never a false ``0.0``) whenever the atom counts driving
    ``w_atom`` are absent or all zero -- ``analyze_grains`` sets
    ``n_atoms`` to 0 per grain when the atom block is omitted, and a
    zero-sum weight vector cannot be normalised; reporting ``0.0`` in that
    case would be a false claim that the two weightings coincide exactly,
    when in truth the atom-count side was simply never measured. The
    message says so explicitly.

    MULTIPHASE. Different phases have different atom number densities (a
    consequence of differing crystal structures and lattice parameters),
    so a GLOBAL atom-count weighting mixed across phases would compare
    unlike things -- the gap would mostly read off the density RATIO
    between phases, not the within-phase discretisation floor this gate
    exists to quantify, and the number would be meaningless. When
    ``phase_of`` is given (paired with ``phase_names`` -- the same
    grain -> phase-name pairing ``pipeline.py`` already uses to write one
    ``odf_mtex_<name>.txt`` per phase), the statistic is computed and
    reported SEPARATELY within each phase's own grain subset, never
    globally; a phase whose subset happens to carry no atoms is reported
    NOT EVALUATED for THAT phase only, other phases unaffected.

    ``measured`` is ``None`` when nothing could be evaluated; for a
    single-phase run (``phase_of`` is ``None``) it is the 3-tuple
    ``(tv, max_dw, mean_atoms_per_grain)``; for a multiphase run it is a
    tuple of that same shape, one entry per phase in ``phase_names``
    order, with a phase-local ``None`` wherever that one phase's atom
    counts were absent -- so ``pipeline._summary_rows`` can read
    ``weighting_tv_volume_vs_atoms`` / ``weighting_max_dw`` /
    ``atoms_per_grain_mean`` straight off this one field (each suffixed
    ``_<phase name>`` in the multiphase case), the same "read the gate's
    own ``measured``, never string-split ``message``" pattern G23/G25
    already use.
    """
    try:
        volumes_arr = np.asarray(volumes, dtype=np.float64).reshape(-1)
        atoms_arr = np.asarray(n_atoms, dtype=np.float64).reshape(-1)
        if len(volumes_arr) != len(atoms_arr):
            raise ValueError(
                "volumes and n_atoms must have the same length; got "
                f"{len(volumes_arr)} and {len(atoms_arr)}.")

        def _stat(v: np.ndarray, a: np.ndarray) -> tuple | None:
            """None (NOT EVALUATED) or (tv, max_dw, mean_atoms_per_grain)."""
            a_total = float(np.sum(a))
            if not (a_total > 0.0):
                return None
            v_total = float(np.sum(v))
            if not (v_total > 0.0):
                return None
            w_vol = v / v_total
            w_atom = a / a_total
            dw = np.abs(w_vol - w_atom)
            return (float(0.5 * np.sum(dw)), float(np.max(dw)),
                    float(np.mean(a)))

        if phase_of is None:
            stat = _stat(volumes_arr, atoms_arr)
            if stat is None:
                return GateResult(
                    gate="G26", passed=True, measured=None,
                    message=(
                        "ok: G26 NOT EVALUATED -- atom counts are absent "
                        "or all zero (no atom block for this run), so "
                        "the atom-count-weighted ODF cannot be formed; "
                        "this reports NOT EVALUATED rather than a "
                        "false zero-gap reading."
                    ),
                )
            tv, max_dw, mean_atoms = stat
            return GateResult(
                gate="G26", passed=True, measured=stat,
                message=(
                    "ok: atomistic ODF-weighting discretisation floor -- "
                    f"TV(tessellation-volume weight, atom-count weight) = "
                    f"{tv:.6f}, max per-grain weight difference "
                    f"{max_dw:.6f}, mean {mean_atoms:.1f} atoms/grain. "
                    "DISCRETISATION FLOOR, not an error: decays as "
                    "(atoms/grain)^(-1/3) (a boundary-shell surface "
                    "effect, not Poisson counting noise's steeper "
                    "-1/2); at small grain sizes (e.g. 275 atoms/grain: "
                    "floor 0.027) it EXCEEDS a typical odf_drift_max of "
                    "0.02, so a tighter drift cap buys nothing that "
                    "survives into the exported structure."
                ),
            )

        # --- multiphase: one statistic per phase, never mixed globally ---
        if phase_names is None:
            raise ValueError(
                "phase_names is required when phase_of is given.")
        phase_of_arr = np.asarray(phase_of, dtype=np.intp).reshape(-1)
        if len(phase_of_arr) != len(volumes_arr):
            raise ValueError(
                "phase_of and volumes must have the same length; got "
                f"{len(phase_of_arr)} and {len(volumes_arr)}.")
        per_phase: list[tuple | None] = []
        parts: list[str] = []
        for p, name in enumerate(phase_names):
            mask = phase_of_arr == p
            stat = _stat(volumes_arr[mask], atoms_arr[mask])
            per_phase.append(stat)
            if stat is None:
                parts.append(f"phase {name}: NOT EVALUATED")
            else:
                tv, max_dw, mean_atoms = stat
                parts.append(
                    f"phase {name}: TV={tv:.6f}, max_dw={max_dw:.6f}, "
                    f"mean {mean_atoms:.1f} atoms/grain")
        return GateResult(
            gate="G26", passed=True, measured=tuple(per_phase),
            message=(
                "ok: atomistic ODF-weighting discretisation floor, per "
                "phase (never mixed globally -- phases have different "
                "atom number densities): " + "; ".join(parts) + ". "
                "DISCRETISATION FLOOR, not an error -- see "
                "gate_g26_odf_weighting_floor's docstring for the "
                "(atoms/grain)^(-1/3) scaling and why a tighter "
                "odf_drift_max cannot buy resolution below it."
            ),
        )
    except Exception as exc:  # a diagnostic must never abort a run
        return GateResult(
            gate="G26", passed=True, measured=None,
            message=("ok: G26 could not be evaluated "
                     f"({type(exc).__name__}: {exc})."),
        )


def nominal_composition(basis_species: list[str],
                        basis_occupancy: list[dict | None]) -> dict[str, float]:
    """Nominal per-species atom fractions of the ideal crystal basis,
    occupancy dicts included (input to gate_g9_composition)."""
    fractions: dict[str, float] = {}
    n_sites = len(basis_species)
    for sp, occ in zip(basis_species, basis_occupancy, strict=True):
        if occ is None:
            fractions[sp] = fractions.get(sp, 0.0) + 1.0 / n_sites
        else:
            for sym, x in occ.items():
                fractions[sym] = fractions.get(sym, 0.0) + float(x) / n_sites
    return fractions


def nominal_composition_phases(
        nominals: list[dict[str, float]],
        atom_weights: np.ndarray) -> dict[str, float]:
    """Atom-count-weighted mixture of per-phase nominal compositions:
    weight_p ∝ ρ_p · V_p^assigned — the expected SHARE of
    atoms each phase contributes, so gate G9 compares the final global
    composition against what the phase partition itself predicts."""
    w = np.asarray(atom_weights, dtype=np.float64)
    w = w / float(np.sum(w))
    fractions: dict[str, float] = {}
    for nom, wp in zip(nominals, w, strict=True):
        for sp, x in nom.items():
            fractions[sp] = fractions.get(sp, 0.0) + float(wp) * float(x)
    return fractions
