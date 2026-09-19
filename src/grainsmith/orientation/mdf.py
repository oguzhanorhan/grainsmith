"""Area-weighted disorientation-angle shaping by assignment annealing.

The orientation SET {q_k} sampled by the configured scheme is fixed; a
simulated annealing over the assignment permutation π (which grain gets
which orientation) pulls the area-weighted neighbor misorientation-ANGLE
histogram toward a target.  Because only the assignment changes, the
NUMBER-weighted discrete orientation distribution f_n = (1/N) Σ_k δ_{q_k}
is exactly invariant by construction.  The VOLUME-weighted ODF -- which is
how an ODF is actually defined, by VOLUME fraction of material, not grain
COUNT -- is NOT invariant in general: a transposition of grains a and b
(of unequal volume V_a, V_b) moves |V_a - V_b| / Σ_i V_i of f_V's
total-variation mass between the two orientations involved.

ATTRIBUTION -- this module implements Miodownik et al. 1999's "conserved
texture" algorithm (their own term): pick two domains at random, swap
their orientations, accept by Metropolis on H = Σ_k (S_k^model -
S_k^desired)^2 over binned disorientation-ANGLE counts (they in turn
credit the swap move to Kiewel, Bunge & Fritsche 1996). Their own
Discussion already names the gap corrected above: because domain volumes
are not part of their texture description, swapping orientations between
domains of unequal size means "the texture is not strictly being
conserved", and truly conserving it would require replacing the discrete
orientation list with an ODF -- their conclusions name boundary area,
volume and CSL type as the next steps. This module already area-weights
the histogram (their first item); ``odf_drift_max`` below addresses the
second (Miodownik et al. 1999). Their notion of "conserved" is therefore
the NUMBER-weighted one, exactly as this module's f_n above.

Saylor et al. 2004 is NOT this algorithm's ancestor and is cited here for
a different reason: they optimise the FULL misorientation distribution
(all three parameters, not only the angle marginal targeted here) and
fold the VOLUME-weighted orientation distribution f(g) directly into
their error function -- prior art for conserving f(g) by putting it in
the objective, and the source of this codebase's f(g)-by-volume-fraction
/ f(Δg)-by-area-fraction convention (see below and orientation/odf.py).
The difference from that soft error term: our cap is a HARD per-swap
bound -- every accepted assignment satisfies d_TV <= epsilon, and by
Markov contraction the same bound holds for the kernel-smoothed ODF at
every half-width (orientation/odf.py) -- whereas a soft penalty gives no
such guarantee; and our objective remains the angle marginal, never the
full MDF Saylor et al. optimise.

``anneal_assignment``'s optional ``odf_drift_max`` argument bounds this
drift: with it set, every ACCEPTED assignment along the walk satisfies
d_TV(f_V(π), f_V(π₀)) <= odf_drift_max, tracked in O(1) per swap via
:class:`~grainsmith.orientation.odf.DriftTracker`.  Because convolving a
measure with a normalised, non-negative kernel is a Markov operator (a
total-variation CONTRACTION), that same bound holds for the kernel-
smoothed ODF at EVERY half-width, not only the un-smoothed one -- which is
what makes "f_V is preserved to within an explicit epsilon by
construction" a sound claim rather than one merely checked after the
fact.  See ``orientation/odf.py``'s module docstring for the full
derivation and the measurement/control machinery
(:func:`~grainsmith.orientation.odf.atomic_drift`,
:class:`~grainsmith.orientation.odf.DriftTracker`) -- this module only
wires that machinery into the annealing loop below.

The ODF f(g) is defined by VOLUME fraction; a full MDF f(dg) is a density
over misorientation space weighted by boundary AREA. The code below
controls only its one-dimensional ANGLE marginal, not the full MDF.

Targets and the generated histogram live on a common bin grid over
[0, θ_max] of the point group:

- ``haar_random`` (deprecated alias ``mackenzie``) — random-pair
  disorientation-ANGLE reference of the ACTUAL point group, computed once
  per run by a deterministic Monte-Carlo with a fixed internal seed
  (constants MDF_REFERENCE_SEED/SAMPLES — a point-group property, not a
  run property). Equals the Mackenzie 1958 law only for the cubic proper
  point group; for any other group it is a different curve Mackenzie
  never derived.
- ``sigma3_angle_enriched`` (deprecated alias ``csl_enriched``) —
  ``haar_random`` base with ``sigma3_fraction`` of the mass moved into
  the Σ3 Brandon ANGULAR window 60° ± 15°/√3 (cubic only). Honest
  limitation: the energy sees the angle histogram only — the 60°-window
  is a NECESSARY condition for Σ3, so true Σ3 enrichment requires an
  orientation set containing twin-related pairs (e.g. two odf_components
  related by 60° ⟨111⟩); documented in the example config.
- ``histogram``     — user bin edges + densities.

Energy: symmetric χ² distance between bin-mass vectors,
χ²(P, T) = Σ_b (P_b − T_b)² / (P_b + T_b), bins with P_b + T_b = 0
contribute 0 (bounded in [0, 2], robust to empty target bins).

Angle metric: the rotation angle is conjugation-invariant, so the
symmetry reduction needs only the one-sided orbit {m ⊗ S}
(misorientation.disorientation_angles); since assignments are
permutations of a fixed set, angles are cached per orientation-index
pair — repeat lookups during annealing are free.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from grainsmith.constants import (
    CSL_BRANDON_FACTOR,
    CSL_TABLE,
    MDF_REFERENCE_SAMPLES,
    MDF_REFERENCE_SEED,
    MDF_THETA_MAX_ROUND_DEG,
)
from grainsmith.errors import ConfigError
from grainsmith.orientation.misorientation import disorientation_angles
from grainsmith.orientation.odf import (
    DriftTracker,
    atomic_drift,
    orientation_classes,
    orientation_weights,
)

log = logging.getLogger(__name__)


@dataclass
class MDFResult:
    """Outcome of the MDF annealing stage (summary.csv rows + G12 input)."""
    permutation: np.ndarray        # (N,) grain → orientation index
    bin_edges: np.ndarray          # (B+1,) degrees
    target_masses: np.ndarray      # (B,) Σ = 1
    reference_masses: np.ndarray   # (B,) random-pair reference, Σ = 1
    chi2_initial: float
    chi2_final: float              # annealer's claim; G12 re-measures
    n_steps: int
    n_accepted: int
    # Volume-weighted ODF trust region (see the module docstring and
    # orientation/odf.py) -- populated only when `grain_volumes` was
    # supplied to anneal_assignment; all three default so existing
    # construction sites (and the both-None mode) are unaffected.
    odf_drift_final: float | None = None   # d_TV(f_V(permutation), f_V(pi0))
    odf_drift_max: float | None = None     # configured cap, or None
    n_drift_vetoed: int = 0                # trial moves rejected by the cap


# ---------------------------------------------------------------------------
# Random-pair reference (Mackenzie law for cubic)
# ---------------------------------------------------------------------------

def reference_angles(sym_quats: np.ndarray) -> np.ndarray:
    """Disorientation angles (deg) of MDF_REFERENCE_SAMPLES Haar-random
    pairs reduced by the given point group.

    The relative rotation of two independent Haar-random orientations is
    itself Haar-random, so one random rotation per sample suffices.  The
    generator is seeded with the fixed MDF_REFERENCE_SEED — the reference
    is a deterministic curve of the point group, identical for every run
    seed (np.random module functions remain unused; this is a local
    Generator construction).
    """
    from scipy.spatial.transform import Rotation as _Rotation

    from grainsmith.orientation.quaternion import from_scipy

    gen = np.random.Generator(np.random.PCG64(MDF_REFERENCE_SEED))
    rot = _Rotation.random(MDF_REFERENCE_SAMPLES, random_state=gen)
    q = from_scipy(rot).reshape(-1, 4)
    # Chunked reduction: the (chunk, Nsym, 4) candidate array stays ~15 MB
    # instead of ~150 MB+ for the full sample set at once.
    chunk = 20_000
    out = np.empty(len(q), dtype=np.float64)
    identity = np.zeros((chunk, 4), dtype=np.float64)
    identity[:, 0] = 1.0
    for lo in range(0, len(q), chunk):
        block = q[lo:lo + chunk]
        out[lo:lo + len(block)] = disorientation_angles(
            identity[:len(block)], block, sym_quats)
    return out


def theta_max_deg(ref_angles: np.ndarray) -> float:
    """MC maximum rounded up to MDF_THETA_MAX_ROUND_DEG, not a proven
    fundamental-zone bound (cubic: about 62.8 to 63.0 degrees)."""
    quantum = MDF_THETA_MAX_ROUND_DEG
    return quantum * math.ceil(float(np.max(ref_angles)) / quantum)


def histogram_masses(
    angles_deg: np.ndarray,
    weights: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Weight-normalized bin masses of *angles_deg* on *bin_edges*.

    Angles above the last edge are clipped into the last bin (the
    empirical θ_max can undershoot the true fundamental-zone maximum by
    ≲1°; the clipped tail mass is ≲1e-5, see constants).
    """
    idx = np.clip(np.digitize(angles_deg, bin_edges) - 1,
                  0, len(bin_edges) - 2)
    masses = np.bincount(idx, weights=weights, minlength=len(bin_edges) - 1)
    total = float(np.sum(masses))
    return masses / total if total > 0.0 else masses


def chi2_distance(p_masses: np.ndarray, t_masses: np.ndarray) -> float:
    """Symmetric χ² distance Σ (P−T)²/(P+T) between bin-mass vectors
    (∈ [0, 2]; empty bins on both sides contribute 0)."""
    p = np.asarray(p_masses, dtype=np.float64)
    t = np.asarray(t_masses, dtype=np.float64)
    denom = p + t
    mask = denom > 0.0
    return float(np.sum((p[mask] - t[mask]) ** 2 / denom[mask]))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

_TARGET_ALIASES: dict[str, str] = {
    "mackenzie": "haar_random",
    "csl_enriched": "sigma3_angle_enriched",
}
"""Deprecated mdf_target ``type`` name -> canonical replacement.

config/resolve.py deliberately never rewrites the user's configured value
(config_sha256 is the sha256 of the dumped resolved config, so normalising
a deprecated alias in place would change the recorded hash of every
archived run that used it). This mapping is instead applied HERE, at the
single point of use, so the branch logic below tests canonical names
only while every existing config -- spelled either way -- keeps producing
bit-identical output.
"""


def canonical_target_type(raw: str) -> str:
    """Canonical mdf_target ``type`` name for *raw*.

    Maps the deprecated aliases (``"mackenzie"`` -> ``"haar_random"``,
    ``"csl_enriched"`` -> ``"sigma3_angle_enriched"``) to their canonical
    replacement; every other name -- the canonical names themselves,
    ``"histogram"``, or anything else -- passes through unchanged.
    """
    return _TARGET_ALIASES.get(raw, raw)


def build_bins_and_target(
    mdf_cfg,
    sym_quats: np.ndarray,
    mdf_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin_edges, target_masses, reference_masses) for an MdfTargetConfig.

    haar_random / sigma3_angle_enriched (or their deprecated aliases
    mackenzie / csl_enriched, normalised via :func:`canonical_target_type`)
    use *mdf_bins* uniform bins over [0, θ_max(point group)]; histogram
    uses the user's edges.  All mass vectors sum to 1.
    """
    ref = reference_angles(sym_quats)
    kind = canonical_target_type(mdf_cfg.type)

    if kind == "histogram":
        edges = np.asarray(mdf_cfg.bin_edges, dtype=np.float64)
        dens = np.asarray(mdf_cfg.densities, dtype=np.float64)
        masses = dens * np.diff(edges)
        target = masses / float(np.sum(masses))
        return edges, target, histogram_masses(ref, np.ones_like(ref), edges)

    edges = np.linspace(0.0, theta_max_deg(ref), mdf_bins + 1)
    ref_masses = histogram_masses(ref, np.ones_like(ref), edges)

    if kind == "haar_random":
        return edges, ref_masses, ref_masses

    # sigma3_angle_enriched: move sigma3_fraction of the mass into the Σ3
    # Brandon angular window, scale the haar_random base into the remainder.
    theta3, _axis3 = CSL_TABLE["3"]
    brandon3 = CSL_BRANDON_FACTOR / np.sqrt(3.0)
    lo, hi = theta3 - brandon3, min(theta3 + brandon3, float(edges[-1]))
    overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo),
                      0.0, None)
    window = overlap / float(np.sum(overlap))
    f = float(mdf_cfg.sigma3_fraction)
    target = (1.0 - f) * ref_masses + f * window
    return edges, target / float(np.sum(target)), ref_masses


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------

class _AngleCache:
    """Disorientation angles keyed by ORIENTATION-index pairs.

    Assignments are permutations of the fixed orientation set, so angles
    seen once never need recomputing; misses are evaluated in one
    vectorized batch.
    """

    def __init__(self, quats: np.ndarray, sym_quats: np.ndarray) -> None:
        self._quats = quats
        self._sym = sym_quats
        self._cache: dict[tuple[int, int], float] = {}

    def angles(self, k_idx: np.ndarray, l_idx: np.ndarray) -> np.ndarray:
        """(M,) angles for orientation-index pairs (k_idx[m], l_idx[m])."""
        keys = [(int(min(k, ll)), int(max(k, ll)))
                for k, ll in zip(k_idx, l_idx, strict=True)]
        miss = [key for key in dict.fromkeys(keys) if key not in self._cache]
        if miss:
            ka = np.array([k for k, _ in miss], dtype=np.intp)
            la = np.array([ll for _, ll in miss], dtype=np.intp)
            vals = disorientation_angles(
                self._quats[ka], self._quats[la], self._sym)
            for key, v in zip(miss, vals, strict=True):
                self._cache[key] = float(v)
        return np.array([self._cache[key] for key in keys], dtype=np.float64)


def anneal_assignment(
    quats: np.ndarray,
    pairs: list[tuple[int, int]],
    pair_areas: np.ndarray,
    bin_edges: np.ndarray,
    target_masses: np.ndarray,
    reference_masses: np.ndarray,
    sym_quats: np.ndarray,
    n_steps: int,
    t0: float,
    cooling: float,
    rng: np.random.Generator,
    *,
    grain_volumes: np.ndarray | None = None,
    odf_drift_max: float | None = None,
) -> MDFResult:
    """Simulated annealing over the orientation-assignment permutation.

    Move: swap π(a), π(b) for two random distinct grains; ΔE from the
    affected adjacency pairs only (the (a, b) pair itself is angle-
    invariant under the swap and need not be excluded — its contribution
    cancels).  Schedule: T_k = t0 · cooling^k; Metropolis acceptance.
    One uniform variate is drawn per step regardless of ΔE sign so the
    stream consumption is move-independent, keeping runs reproducible.

    Returns the best-energy assignment encountered (annealing is a
    heuristic — gate G12 re-measures the final structure and only warns).

    Optional volume-weighted ODF trust region (module docstring above and
    ``orientation/odf.py``): ``grain_volumes`` (per-grain volume, raw or
    already-normalised) and ``odf_drift_max`` (an epsilon on
    d_TV(f_V(π), f_V(π₀))) together add a hard veto on top of the
    Metropolis acceptance above, in three modes:

    * both omitted (default) -- exactly today's behaviour;
      ``MDFResult.odf_drift_final`` is ``None``.
    * ``grain_volumes`` given, ``odf_drift_max=None`` -- MEASURE ONLY: the
      drift is tracked throughout but never vetoed, and
      ``odf_drift_final`` reports the drift of the RETURNED assignment.
      The returned permutation is bit-identical to the both-omitted case.
    * both given -- MEASURE AND VETO: a trial move is rejected outright
      (before its χ² cost is even computed, since the O(1) drift check is
      strictly cheaper than the O(|affected|) χ² update) whenever it would
      push the tracked drift above ``odf_drift_max``.  Every ACCEPTED
      assignment along the walk -- and therefore the returned
      ``best_perm`` -- satisfies the cap.
    * ``odf_drift_max`` given without ``grain_volumes`` raises
      :class:`~grainsmith.errors.ConfigError` (nothing to measure the cap
      against).

    Crucially, the trust region never changes which random numbers are
    drawn: the move proposal (``a``, ``b``, ``u``) is drawn
    unconditionally, exactly as without it, and the veto is applied
    strictly AFTER those draws -- so every value of ``odf_drift_max``,
    including ``None``, consumes the ``rng`` stream identically.
    """
    n = len(quats)

    # Validated BEFORE the early return below, so bad input always raises
    # regardless of adjacency (an empty/degenerate `pairs` must not mask a
    # misconfiguration of the trust region).
    if odf_drift_max is not None and grain_volumes is None:
        raise ConfigError(
            "odf_drift_max requires grain_volumes to be given as well -- "
            "there is nothing to measure the volume-weighted ODF drift "
            f"against; got grain_volumes=None, odf_drift_max={odf_drift_max!r}."
        )
    drift_cap: float | None = None
    if odf_drift_max is not None:
        drift_cap = float(odf_drift_max)
        if not math.isfinite(drift_cap) or drift_cap < 0.0:
            raise ConfigError(
                "odf_drift_max must be a finite value >= 0; got "
                f"{odf_drift_max!r}."
            )
    if grain_volumes is not None and len(grain_volumes) != n:
        raise ConfigError(
            f"grain_volumes must have length {n} (one entry per grain, "
            f"matching quats); got length {len(grain_volumes)}."
        )

    perm = np.arange(n, dtype=np.intp)
    class_of: np.ndarray | None = None
    n_classes: int | None = None
    tracker: DriftTracker | None = None
    if grain_volumes is not None:
        class_of, n_classes = orientation_classes(quats)
        tracker = DriftTracker(perm, grain_volumes, class_of, n_classes)

    n_bins = len(bin_edges) - 1
    pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
    pair_j = np.array([j for _, j in pairs], dtype=np.intp)
    pair_areas = np.asarray(pair_areas, dtype=np.float64)
    a_total = float(np.sum(pair_areas))
    if len(pairs) == 0 or a_total <= 0.0:
        # No adjacency area to anneal: an empty histogram gives 0/0 = NaN that
        # chi2_distance silently masks to a *falsely perfect* chi2 = 0.0. Return
        # the identity assignment with NaN chi2 instead — honest "not
        # measurable" (gate G12 re-measures and only warns).
        nan = float("nan")
        return MDFResult(
            permutation=np.arange(n, dtype=np.intp),
            bin_edges=np.asarray(bin_edges, dtype=np.float64),
            target_masses=np.asarray(target_masses, dtype=np.float64),
            reference_masses=np.asarray(reference_masses, dtype=np.float64),
            chi2_initial=nan, chi2_final=nan, n_steps=0, n_accepted=0,
            odf_drift_final=(0.0 if grain_volumes is not None else None),
            odf_drift_max=odf_drift_max, n_drift_vetoed=0)

    # grain → indices of pairs touching it
    touching: list[list[int]] = [[] for _ in range(n)]
    for p_idx, (i, j) in enumerate(pairs):
        touching[i].append(p_idx)
        touching[j].append(p_idx)
    touch_arr = [np.array(t, dtype=np.intp) for t in touching]

    cache = _AngleCache(quats, sym_quats)

    def pair_bins(p_idx: np.ndarray, perm_: np.ndarray) -> np.ndarray:
        ang = cache.angles(perm_[pair_i[p_idx]], perm_[pair_j[p_idx]])
        return np.clip(np.digitize(ang, bin_edges) - 1, 0, n_bins - 1)

    all_pairs = np.arange(len(pairs), dtype=np.intp)
    bins = pair_bins(all_pairs, perm)
    hist = np.bincount(bins, weights=pair_areas, minlength=n_bins)

    chi2 = chi2_distance(hist / a_total, target_masses)
    chi2_initial = chi2
    best_chi2 = chi2
    best_perm = perm.copy()
    n_accepted = 0
    n_drift_vetoed = 0

    for k in range(n_steps):
        a = int(rng.integers(n))
        b = int(rng.integers(n - 1))
        if b >= a:
            b += 1
        u = float(rng.random())

        affected = np.unique(np.concatenate([touch_arr[a], touch_arr[b]]))
        if len(affected) == 0:
            continue

        # k_a/k_b are read from `perm` BEFORE the trial swap below -- at
        # this point `perm` IS the last ACCEPTED state (every acceptance
        # this loop makes either commits the tracker in lockstep, below,
        # or reverts `perm` without ever having touched the tracker), so
        # this matches DriftTracker's own internal permutation exactly
        # and its desync guard in `trial()`/`commit()` is satisfied.
        # `trial()` is O(1) versus the O(|affected|) chi2 update below, so
        # vetoing here first is strictly cheaper and changes nothing else.
        k_a = int(perm[a])
        k_b = int(perm[b])
        if drift_cap is not None:
            # drift_cap is not None ⇒ odf_drift_max was given ⇒ (by the
            # mode-conflict check above) grain_volumes was given too ⇒
            # tracker was constructed -- spelled out for the type checker.
            assert tracker is not None
            if tracker.trial(a, b, k_a, k_b) > drift_cap:
                n_drift_vetoed += 1
                continue

        perm[a], perm[b] = perm[b], perm[a]
        new_bins = pair_bins(affected, perm)
        hist_new = hist.copy()
        np.subtract.at(hist_new, bins[affected], pair_areas[affected])
        np.add.at(hist_new, new_bins, pair_areas[affected])
        chi2_new = chi2_distance(hist_new / a_total, target_masses)

        d_e = chi2_new - chi2
        # cooling**k underflows to 0.0 for very long schedules; -746 is
        # below log(min subnormal), so the exp underflows to 0.0 (reject).
        temp = t0 * cooling ** k
        log_accept = -d_e / temp if temp > 0.0 else -math.inf
        if d_e <= 0.0 or u < math.exp(max(log_accept, -746.0)):
            n_accepted += 1
            hist = hist_new
            bins[affected] = new_bins
            chi2 = chi2_new
            if tracker is not None:
                # commit() performs its own swap, re-synchronising the
                # tracker's internal permutation with `perm` above.
                tracker.commit(a, b, k_a, k_b)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_perm = perm.copy()
        else:
            perm[a], perm[b] = perm[b], perm[a]  # revert

    log.info("mdf annealing: chi2 %.4f -> %.4f (best %.4f), "
             "%d/%d swaps accepted",
             chi2_initial, chi2, best_chi2, n_accepted, n_steps)

    # Honest re-measurement of the RETURNED assignment (best_perm), never
    # `tracker.value` (the walk's LAST state, which can differ from
    # best_perm) -- best_perm is only ever updated on acceptance, and
    # every acceptance already passed the veto above, so this is expected
    # (not merely asserted) to satisfy odf_drift_max.
    odf_drift_final: float | None = None
    if tracker is not None:
        assert grain_volumes is not None
        assert class_of is not None and n_classes is not None
        v_best = orientation_weights(best_perm, grain_volumes)
        v_ref = orientation_weights(np.arange(n), grain_volumes)
        odf_drift_final = atomic_drift(v_best, v_ref, class_of, n_classes)
        log.info(
            "mdf annealing (ODF trust region): drift %.6f (cap %s), "
            "%d/%d trial moves vetoed",
            odf_drift_final,
            "none" if drift_cap is None else f"{drift_cap:.6f}",
            n_drift_vetoed, n_steps,
        )

    return MDFResult(
        permutation=best_perm,
        bin_edges=bin_edges,
        target_masses=target_masses,
        reference_masses=reference_masses,
        chi2_initial=chi2_initial,
        chi2_final=best_chi2,
        n_steps=n_steps,
        odf_drift_final=odf_drift_final,
        odf_drift_max=odf_drift_max,
        n_drift_vetoed=n_drift_vetoed,
        n_accepted=n_accepted,
    )
