"""Volume-weighted ODF diagnostics and assignment-drift control (§6.10).

BACKGROUND -- why this module exists
-------------------------------------
orientation/mdf.py's assignment annealing permutes WHICH grain gets WHICH
orientation to pull the boundary-AREA-weighted misorientation histogram
toward a target.  Because it only permutes the assignment, the NUMBER-
weighted orientation distribution

    f_n = (1/N) sum_k delta_{q_k}

is exactly invariant -- but the physically meaningful orientation
distribution function (ODF) is VOLUME-weighted:

    f_V(pi) = sum_i (V_i / sum V) * delta_{q_pi(i)}

Settled convention for this codebase (do not contradict it elsewhere): the
ODF f(g) is defined by VOLUME fraction; the MDF f(dg) is defined by AREA
fraction (the MDF histogram in orientation/mdf.py is already boundary-area
weighted).  Only the ODF side had a defect -- and only in the claim that
annealing leaves it invariant, which is false in general: swapping which
grain (of unequal volume) carries which orientation moves mass between
orientations of f_V even though f_n never moves.  This module supplies the
measurement and control primitives that make the corrected claim
("f_V is preserved to within an explicit epsilon") checkable and enforceable.

TWO FACTS THAT DRIVE THE DESIGN
--------------------------------
(A) For a single transposition of grains a and b carrying distinct
    orientation classes (all other grains fixed),

        d_TV(f_V(pi), f_V(pi o (ab))) = |V_a - V_b| / sum_i V_i.

    (Verified: an 80/20-volume, two-orientation example gives exactly 0.6.)
    This is why the incremental swap update below is exact in closed form:
    a transposition moves EXACTLY |V_a - V_b|/sum V of TV mass, and only
    between the two orientation classes involved.

(B) On the LABELLED orientation-index measures (before merging duplicate
    or symmetry-equivalent orientations), the atomic TV distance

        d_TV(f_V, f_n) = 0.5 * sum_i |w_i - 1/N|

    is INDEPENDENT of the assignment pi.  It is a property of the
    grain-size distribution ALONE. After merging equivalent orientations
    this is an upper bound, not necessarily an equality or an invariant.
    The annealer cannot change this labelled gap -- it can change WHICH
    orientations end up carrying the excess mass, and that redistribution
    is invisible to the atomic distance (a permutation of which atom holds
    which weight does not change sum_i |w_i - 1/N|) and visible only after
    smoothing in orientation space.  This is why the module offers BOTH:

      * an exact ATOMIC drift metric (:func:`atomic_drift` /
        :class:`DriftTracker`) for O(1)-per-swap CONTROL during annealing
        (it bounds the true, kernel-smoothed drift at every smoothing
        scale -- see below -- so it is a sound thing to cap), and
      * a kernel discrepancy (:func:`mmd` against the
        :func:`symmetrized_gram` matrix) for MEASUREMENT of where the
        excess mass actually landed.

    Verified numerically: a deliberately size-biased assignment and a
    neutral random one give the IDENTICAL atomic distance to f_n, while the
    kernel statistic (:func:`mmd` vs. a null built by :func:`null_mmd`)
    separates them by a factor of ~4.

WHY THE ATOMIC DRIFT BOUNDS THE KERNEL-SMOOTHED DRIFT AT EVERY SCALE
---------------------------------------------------------------------
Convolving a measure with a normalised, non-negative kernel (the de la
Vallee Poussin kernel below is exactly that) is a Markov operator, and
Markov operators are TV CONTRACTIONS: smoothing two measures can only
shrink their TV distance, never grow it.  So a bound on the atomic drift
``drift(pi) <= eps`` bounds the drift of the kernel-smoothed ODF at EVERY
half-width kappa simultaneously.  This is what makes an
"f_V preserved to within eps by construction" claim valid without having to
recompute a Gram-matrix quadratic form on every annealing step -- the O(1)
atomic update (:class:`DriftTracker`) is a sound, cheap proxy to cap during
the anneal, and the kernel discrepancy is computed once at the end (or a
handful of times) to report WHERE, not whether, drift occurred.

THE de la VALLEE POUSSIN KERNEL
---------------------------------
psi_kappa(omega) = C(kappa) * cos^(2*kappa)(omega/2), with positive INTEGER
kappa. The even integer quaternion-dot power is a tensor-product kernel;
averaging over the crystal group preserves positive semidefiniteness.
Fractional kappa does NOT have that guarantee and gives negative Gram
eigenvalues for broad kernels. Two closed forms are needed:

* :func:`vp_kappa` rounds the continuous solution
    -ln(2) / (2*ln(cos(radians(halfwidth)/2))) to the nearest positive
    integer (at least 1). :func:`vp_halfwidth` reports the ACTUAL half-width
    after this quantisation. The maximum effective half-width is 90 degrees
    (kappa=1); broad requested kernels may differ substantially from it.

* :func:`vp_norm` normalises C(kappa) so psi integrates to 1 over SO(3)
  under NORMALISED Haar measure, so Gram-matrix / density values below are
  in "multiples of a random distribution" (mrd) -- the standard texture
  unit where a perfectly random (Haar) orientation set reads 1.0
  everywhere.  Derivation: the Haar ANGLE density is p(omega) =
  (1 - cos omega)/pi on [0, pi], and substituting t = omega/2,
      integral over SO(3) of psi
        = integral_0^pi psi(omega) p(omega) d(omega)
        = (4C/pi) * integral_0^(pi/2) cos^(2k)(t) sin^2(t) dt
        = (2C/pi) * Beta(3/2, k + 1/2).
  Setting this to 1 and solving for C = vp_norm(k) gives
      vp_norm(k) = pi / (2 * Beta(1.5, k + 0.5)).
  Implemented via ``scipy.special.betaln`` (log-Beta) for numerical
    robustness at large kappa (kappa=91 for a requested 10 deg half-width).

QUATERNION EVALUATION SHORTCUT
---------------------------------
For unit quaternions, the scalar (real) part of g^{-1} (x) x equals the
plain R^4 dot product <g, x> (both encode the same "how aligned are these
two rotations" cosine), and cos(omega/2) = |scalar part| for the rotation
angle omega between them.  So evaluating psi_kappa at the angle between two
ORIENTATIONS never requires a quaternion product: it is a plain dot product
against a PRECOMPUTED orbit array -- :func:`symmetrized_gram` never calls
:func:`~grainsmith.orientation.quaternion.quat_mul` in its inner loop, only
matrix multiplication (dot products), which is what makes an O(N^2 * Nsym)
Gram build fast enough to run inside a QA gate.

SYMMETRY SIDE -- crystal symmetry acts on the RIGHT
-----------------------------------------------------
In this codebase, the physically equivalent orientations of a stored
quaternion q are {q (x) S : S in point group} -- symmetry acts on the
RIGHT.  :func:`symmetrized_gram` therefore builds the orbit as
``quat_mul_batch(quats[:, None, :], sym_quats[None, :, :])`` (q first,
S second).  Building it the other way round, S (x) q, would be WRONG --
and would not even be symmetry-invariant, since {S (x) q} is a different
(left) orbit than the physically equivalent set.  The right-sided orbit
also gives invariance in the FIRST argument for free: the rotation angle
between two quaternions is conjugation-invariant (the same fact already
used in orientation/misorientation.py's one-sided-orbit disorientation
reduction), so it does not matter which of q_k, q_l is the one whose orbit
gets enumerated -- only that exactly one of them is.

THE SYMMETRIZED GRAM MATRIX
-------------------------------
    M[k, l] = (1/Nsym) * sum_S psi_kappa( <q_k, q_l (x) S> )

Verified properties (pinned in tests/test_odf_guard.py):

* M is symmetric and POSITIVE SEMIDEFINITE. Wide kernels have finite
    bandwidth and can be rank-deficient even for distinct orientations.
* M[k, k] is independent of the orientation: it is the sum of kernel
    values at the symmetry-operation angles, divided by Nsym. For a narrow
    kernel nonidentity terms are negligible, giving approximately
    vp_norm(kappa)/Nsym (~64.9 mrd for requested 10 deg and cubic symmetry).
* The OFF-DIAGONAL mean tends to 1.0 mrd for a Haar-random orientation set
  (the normalisation is exactly calibrated so a uniformly random texture
  reads 1 mrd almost everywhere) -- the FULL mean (diagonal included) is
  biased high by the large diagonal terms, so this must be measured off
  the off-diagonal entries only.

Because M is symmetric positive (semi)definite, MMD(v_a, v_b) =
sqrt((v_a - v_b)^T M (v_a - v_b)) (:func:`mmd`) is a pseudometric (never
negative, triangle inequality holds): distinct distributions can have
identical moments within this finite bandwidth. Since psi_kappa is a positive
definite kernel it factors as psi = h * h for some h (a standard fact about
positive-definite kernels), so MMD is EXACTLY the L2 distance between the
two measures after being smoothed by h.  It is NOT (and must not be
described as) the L2 distance between the two psi-SMOOTHED densities --
that would require psi to be self-reproducing under convolution
(psi * psi == psi up to scale), which was checked numerically for the de la
Vallee Poussin kernel and found FALSE.

ORIENTATION CLASSES -- merging exact duplicates
---------------------------------------------------
:func:`orientation_classes` merges orientations that are BIT-IDENTICAL
after a canonical sign choice into a single atom.  Without this, swapping
two grains that happen to carry the SAME orientation (e.g. the shipped
``cu_twin_odf_mdf.yaml`` example, which uses ``spread_deg: 0`` and
therefore produces bit-identical duplicates via ``np.tile``) would be
charged a nonzero drift cost for a move that changes NOTHING about f_V.

Merging is done on EXACT equality after sign canonicalisation (unit
quaternions q and -q represent the same rotation): flip so w >= 0; in the
knife-edge case w == 0 exactly (a 180 deg rotation), canonicalise on the
sign of the first nonzero of (x, y, z).  Equality is EXACT, never rounded:
failing to merge two orientations that are in fact identical only makes the
reported drift an OVER-estimate (a real zero-cost swap gets counted as
nonzero), so the "drift <= eps" UPPER-BOUND guarantee the annealer relies
on still holds.  Rounding to merge near-duplicates, by contrast, could
merge orientations that are genuinely distinct and would silently UNDER-
count real drift, breaking that guarantee -- so it is deliberately not
done here.

SYMMETRY-QUOTIENT CLASSES -- the physical ODF lives on SO(3)/Sym
-----------------------------------------------------------------
For a single-phase material, orientations q and q (x) S (S any crystal
point-group operation) are the SAME physical orientation: the physical
ODF is a measure on the QUOTIENT of orientation space by the point group,
not on the labelled orientation set.  :func:`orientation_classes`
deliberately does NOT merge such pairs (only bit-identical ones): merging
classes can only SHRINK a total-variation distance (triangle inequality),
so any tolerance-based merge inside the annealer's control loop could
silently UNDER-count real drift and break the cap guarantee -- exact
equality is the tightest merge that provably cannot.  The consequence is
a deliberate direction of conservatism: the atomic drift on the labelled
set is an UPPER BOUND on the physical drift on the quotient (quotienting
is a pushforward, and TV never grows under a pushforward),

    d_TV(f_V^quotient(pi), f_V^quotient(pi0)) <= d_TV^atomic(pi, pi0),

and equality need not hold: two grains carrying symmetry-equivalent
orientations (realisable in practice only via hand-built ``from_list``
inputs -- every continuous sampler hits exact equivalence with
probability zero) can be swapped at zero physical cost while the atomic
metric charges |V_a - V_b| / sum_i V_i for it (verified directly: a
cubic-symmetry-equivalent pair at V = (0.8, 0.2) reads atomic drift 0.6
for a swap whose physical drift is 0).  For MEASUREMENT (never for the
annealer's control loop), :func:`symmetry_classes` merges
quotient-equivalent orientations at a fixed angular tolerance
(:data:`~grainsmith.constants.ODF_QUOTIENT_TOL_DEG`) and gate G22 reports
the resulting quotient drift alongside the atomic one, so both the
controlled bound and the sharper physical number are visible.

DETERMINISM
--------------
:func:`null_mmd` never touches the module-level ``np.random`` functions; it
builds a fresh ``np.random.Generator(np.random.PCG64(seed))`` from a seed
argument that defaults to :data:`~grainsmith.constants.ODF_NULL_SEED`.
Exactly like ``MDF_REFERENCE_SEED`` in orientation/mdf.py, this null is a
property of the CODE (what does the kernel-smoothed volume-vs-count gap
look like under a NEUTRAL, non-adversarial assignment of these grain sizes
to this orientation set?), not of the run -- it must be the same
deterministic reference for every run seed, so it deliberately does not
draw from the run's own RNG bundle.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

from grainsmith.constants import (
    MEMORY_HARD_LIMIT_BYTES,
    ODF_KERNEL_MAX_ELEMS,
    ODF_NULL_SAMPLES,
    ODF_NULL_SEED,
    ODF_QUOTIENT_TOL_DEG,
)
from grainsmith.errors import ConfigError, GrainsmithError
from grainsmith.memory import build_memory_guard_message
from grainsmith.orientation.quaternion import quat_mul_batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# de la Vallee Poussin kernel
# ---------------------------------------------------------------------------

def vp_kappa(halfwidth_deg: float) -> float:
    """Nearest positive integer VP degree for a requested half-width.

    Integer degrees make the quaternion-dot kernel positive semidefinite;
    fractional powers need not be. Use :func:`vp_halfwidth` to recover the
    effective width. The log-cosine uses log1p(-2*sin(angle/2)**2) at small
    angles to avoid rounding cos(angle) to 1 and dividing by zero.
    """
    h = float(halfwidth_deg)
    if not math.isfinite(h) or not (0.0 < h < 180.0):
        raise ConfigError(
            "ODF kernel halfwidth_deg must be a finite value in the open "
            f"interval (0, 180) degrees; got {halfwidth_deg!r}."
        )
    angle = math.radians(h) / 2.0
    log_cos = (math.log1p(-2.0 * math.sin(angle / 2.0)**2)
               if h < 90.0 else math.log(math.cos(angle)))
    if log_cos == 0.0:
        raise ConfigError("ODF kernel halfwidth_deg is too small for float64.")
    continuous = -math.log(2.0) / (2.0 * log_cos)
    if not math.isfinite(continuous):
        raise ConfigError("ODF kernel degree is too large for float64.")
    return float(max(1, round(continuous)))


def vp_halfwidth(kappa: float) -> float:
    """Effective angular half-width (degrees) of a positive-degree kernel."""
    degree = float(kappa)
    if not math.isfinite(degree) or degree <= 0.0:
        raise ConfigError("ODF kernel degree must be finite and positive.")
    sine_squared = -math.expm1(-math.log(2.0) / (2.0 * degree)) / 2.0
    return math.degrees(4.0 * math.asin(math.sqrt(sine_squared)))


def vp_norm(kappa: float) -> float:
    """Normalising constant C(kappa) so the de la Vallee Poussin kernel
    integrates to 1 over SO(3) under normalised Haar measure (values are
    then in "multiples of a random distribution", mrd).

    vp_norm(kappa) = pi / (2 * Beta(1.5, kappa + 0.5)) -- see the module
    docstring for the Haar-measure derivation.  Implemented via
    ``scipy.special.betaln`` (log-Beta) rather than ``scipy.special.beta``
    directly for robustness at large kappa. The ordinary 5 and 10 degree
    kernels do not underflow in linear space.
    """
    k = float(kappa)
    return math.exp(math.log(math.pi) - math.log(2.0) - float(betaln(1.5, k + 0.5)))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_positive_finite(values, name: str) -> np.ndarray:
    """1-D float64 copy of *values*, or ConfigError if any entry is
    non-finite or non-positive (used for volumes / volume-fraction
    weights: a zero or negative "amount of material" is never valid)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(v)):
        raise ConfigError(f"{name} must be finite; got {values!r}.")
    if np.any(v <= 0.0):
        raise ConfigError(f"{name} must be strictly positive; got {values!r}.")
    return v


def _validate_permutation(perm, n: int, name: str = "perm") -> np.ndarray:
    """1-D intp copy of *perm*, or ConfigError unless it is EXACTLY a
    permutation of ``range(n)`` -- every index in ``[0, n)`` appearing
    exactly once.

    This check exists because ``orientation_weights`` and
    :class:`DriftTracker` build their output by scatter-assignment
    (``v[perm] = w``) into a ``np.empty`` array: a *repeated* index in
    *perm* leaves some other index NEVER written, so without this guard
    the function would silently return UNINITIALISED MEMORY for those
    entries -- unacceptable in a codebase whose central claim is
    bit-reproducibility.  An out-of-range index would raise its own
    (less informative) IndexError/negative-wraparound bug on the same
    scatter, so it is checked here too, before the write happens.

    Always returns a FRESH copy (``np.array(..., copy=True)``, never
    ``np.asarray``): :class:`DriftTracker` keeps this return value as its
    own mutable ``self._perm``, mutating it in place on every
    :meth:`~DriftTracker.commit`.  Returning a view/alias of the caller's
    own array here would let two independent trackers (or a tracker and
    its caller) silently corrupt each other's state the moment either one
    swaps -- exactly the kind of desync :meth:`DriftTracker._delta`
    separately guards against, so this helper must not introduce one of
    its own by aliasing.
    """
    raw = np.asarray(perm).reshape(-1)
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ConfigError(f"{name} must contain finite integer indices.")
    if raw.dtype.kind == "f" and np.any(raw != np.floor(raw)):
        raise ConfigError(f"{name} must contain integer indices, not fractions.")
    if len(raw) != n:
        raise ConfigError(f"{name} must have length {n}; got {len(raw)}.")
    if n == 0:
        return np.array(raw, dtype=np.intp, copy=True)
    if int(raw.min()) < 0 or int(raw.max()) >= n:
        raise ConfigError(
            f"{name} must contain only indices in [0, {n}); got values "
            f"ranging [{int(raw.min())}, {int(raw.max())}]."
        )
    p = np.array(raw, dtype=np.intp, copy=True)
    counts = np.bincount(p, minlength=n)
    if not np.all(counts == 1):
        bad = np.nonzero(counts != 1)[0]
        raise ConfigError(
            f"{name} must be a permutation of range({n}) (each orientation "
            "index must appear exactly once); index/indices with wrong "
            f"multiplicity: {bad[:10].tolist()}"
            f"{', ...' if len(bad) > 10 else ''}."
        )
    return p


# ---------------------------------------------------------------------------
# Orientation classes (exact-duplicate merging)
# ---------------------------------------------------------------------------

def _canonical_sign(q: np.ndarray) -> np.ndarray:
    """Sign-canonicalize unit quaternions so the two antipodal
    representations (q, -q) of the SAME rotation compare EQUAL: flip so
    w >= 0; when w == 0 exactly, flip so the first nonzero of (x, y, z) is
    positive (a 180 degree rotation still has two antipodal quaternion
    representatives even though w == 0 does not itself carry a sign)."""
    q = np.array(q, dtype=np.float64, copy=True)
    flip = q[:, 0] < 0.0
    zero_w = q[:, 0] == 0.0
    if np.any(zero_w):
        rest = q[zero_w][:, 1:]
        nonzero = rest != 0.0
        has_nonzero = np.any(nonzero, axis=1)
        first_idx = np.argmax(nonzero, axis=1)  # index of first True (0 if none)
        first_val = rest[np.arange(len(rest)), first_idx]
        flip[zero_w] = has_nonzero & (first_val < 0.0)
    q[flip] *= -1.0
    return q


def orientation_classes(quats: np.ndarray) -> tuple[np.ndarray, int]:
    """Merge bit-identical orientations (after sign canonicalisation) into
    classes.

    Returns ``(class_of, n_classes)``: ``class_of[k]`` is the class index
    of orientation ``k`` (0 <= class_of[k] < n_classes).  Uses EXACT
    equality (``numpy.unique`` on the sign-canonicalised rows), never
    rounding -- see the module docstring's "ORIENTATION CLASSES" section
    for why an exact-equality merge keeps the drift bound sound while a
    rounded one would not.

    NOTE -- symmetry-equivalent but bit-DISTINCT orientations (q vs.
    q (x) S) are NOT merged here, deliberately; the atomic drift built on
    these classes is an upper bound on the physical (quotient-space)
    drift.  See the module docstring's "SYMMETRY-QUOTIENT CLASSES"
    section and :func:`symmetry_classes` for the measurement-side merge.
    """
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    canon = _canonical_sign(q)
    _, inverse = np.unique(canon, axis=0, return_inverse=True)
    class_of = np.asarray(inverse, dtype=np.intp).reshape(-1)
    n_classes = int(class_of.max()) + 1 if len(class_of) else 0
    return class_of, n_classes


def symmetry_classes(
    quats: np.ndarray,
    sym_quats: np.ndarray,
    tol_deg: float = ODF_QUOTIENT_TOL_DEG,
) -> tuple[np.ndarray, int]:
    """Merge orientations related by a crystal point-group operation into
    quotient-space classes (SO(3)/Sym) -- for MEASUREMENT only, never for
    the annealer's control loop (see the module docstring's
    "SYMMETRY-QUOTIENT CLASSES" section for why control must stay on
    :func:`orientation_classes`' exact-equality classes).

    Two stored quaternions q_k, q_l land in the same class when their
    symmetry-reduced separation angle is at most *tol_deg*, i.e. when

        max_S |<q_k, q_l (x) S>| >= cos(radians(tol_deg) / 2)

    over the right-sided symmetry orbit (the same orbit convention as
    :func:`symmetrized_gram`; the |dot| subsumes the antipodal
    identification q ~ -q, exactly as in
    :func:`~grainsmith.orientation.misorientation.disorientation_angles`).
    The classes are the connected components of that pairwise graph
    (union-find), so the resulting partition is independent of row order
    and evaluation chunking.

    At the default tolerance (1e-3 deg) this coincides with EXACT
    symmetry-class merging except for physically absurd near-degeneracies:
    a genuine q' = q (x) S pair survives float round-trip at ~1e-13 deg
    separation (three orders of magnitude below the tolerance), while two
    generically distinct orientations sit orders of magnitude above it.
    Because merging can only shrink a total-variation distance, an
    :func:`atomic_drift` computed on these classes is guaranteed <= the
    labelled-set value -- the sharper physical number, reported by gate
    G22 next to the (controlled) atomic one.

    Cost: O(N^2 * Nsym) flops via row-chunked matrix multiplication
    against the same orbit array :func:`symmetrized_gram` builds (the
    transient stays under ``ODF_KERNEL_MAX_ELEMS`` elements); intended for
    the orientation-set sizes the ODF diagnostics target (hundreds to low
    thousands of grains), e.g. inside gate G22's memory-guarded block.
    """
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    sym = np.asarray(sym_quats, dtype=np.float64).reshape(-1, 4)
    n = len(q)
    if n == 0:
        return np.empty(0, dtype=np.intp), 0
    n_sym = len(sym)
    cos_tol = math.cos(math.radians(float(tol_deg)) / 2.0)

    # Union-find over the orientation indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    qs = quat_mul_batch(q[:, None, :], sym[None, :, :]).reshape(-1, 4)
    chunk = max(1, ODF_KERNEL_MAX_ELEMS // max(n * n_sym, 1))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = np.abs(q[lo:hi] @ qs.T)            # (hi-lo, n*n_sym)
        m = d.reshape(hi - lo, n, n_sym).max(axis=2)   # max over S
        rows, cols = np.nonzero(m >= cos_tol)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            union(lo + r, c)

    roots = np.array([find(i) for i in range(n)], dtype=np.intp)
    _, class_of = np.unique(roots, return_inverse=True)
    class_of = np.asarray(class_of, dtype=np.intp).reshape(-1)
    n_classes = int(class_of.max()) + 1
    return class_of, n_classes


# ---------------------------------------------------------------------------
# Symmetrized Gram matrix
# ---------------------------------------------------------------------------

def symmetrized_gram(
    quats: np.ndarray,
    sym_quats: np.ndarray,
    kappa: float,
    *,
    memory_limit_bytes: float | None = None,
    memory_limit_source: str = "config",
) -> np.ndarray:
    """Symmetrized de la Vallee Poussin Gram matrix
    ``M[k, l] = (1/Nsym) * sum_S psi_kappa(<q_k, q_l (x) S>)`` (mrd units).

    Builds the right-sided symmetry orbit ``{q (x) S}`` ONCE (see the
    module docstring's "SYMMETRY SIDE" section for why the orbit must be
    right-sided) and evaluates psi via the plain-dot-product shortcut (no
    per-pair quaternion product) in ROW CHUNKS sized so the transient
    candidate array stays under ``ODF_KERNEL_MAX_ELEMS`` elements
    regardless of how large N or Nsym grows.

    A dot product between unit vectors can exceed 1.0 by a float ULP or
    two; the mandatory clip to [0, 1] BEFORE raising to the ``2*kappa``
    power prevents that from exploding into a huge (or NaN, for a
    fractional-looking negative base) value at the kappa a narrow
    half-width demands (kappa ~ 91 at 10 deg).

    §13 memory guard: the estimated peak allocation (the ``N x N`` output
    plus one row-chunk's transient candidate array) is computed BEFORE any
    of it is allocated; if it exceeds *memory_limit_bytes* (default
    :data:`~grainsmith.constants.MEMORY_HARD_LIMIT_BYTES` when not given --
    the un-clamped module default, since this diagnostic is not wired
    through the per-run RAM-clamped budget the way tessellation
    construction is) a :class:`~grainsmith.errors.GrainsmithError` is
    raised with the standard guard message instead of risking an OOM kill.
    """
    degree = float(kappa)
    if not math.isfinite(degree) or degree < 0.0 or not degree.is_integer():
        raise ConfigError(
            "ODF Gram kernel degree must be a finite non-negative integer; "
            "fractional powers are not positive semidefinite."
        )
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    sym = np.asarray(sym_quats, dtype=np.float64).reshape(-1, 4)
    n = len(q)
    n_sym = len(sym)

    limit_bytes = (MEMORY_HARD_LIMIT_BYTES if memory_limit_bytes is None
                   else float(memory_limit_bytes))

    chunk = max(1, ODF_KERNEL_MAX_ELEMS // max(n * n_sym, 1))
    est_bytes = 8.0 * n * n + 8.0 * chunk * n * n_sym
    log.info(
        "ODF symmetrized Gram: %d orientations x %d symmetry ops, "
        "row-chunk %d, estimated %.2f GB peak.",
        n, n_sym, chunk, est_bytes / 1e9,
    )
    if est_bytes > limit_bytes:
        raise GrainsmithError(build_memory_guard_message(
            f"ODF symmetrized Gram matrix ({n} orientations x {n_sym} "
            "symmetry ops)",
            est_bytes, limit_bytes, memory_limit_source,
            extra_advice=(
                "Reduce the number of distinct orientations, or merge "
                "duplicates via orientation_classes before building the "
                "Gram matrix."
            ),
        ))

    qs = quat_mul_batch(q[:, None, :], sym[None, :, :]).reshape(-1, 4)  # (n*n_sym, 4)
    scale = vp_norm(kappa) / n_sym
    two_kappa = 2.0 * float(kappa)
    out = np.empty((n, n), dtype=np.float64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = q[lo:hi] @ qs.T                     # (hi-lo, n*n_sym)
        np.abs(d, out=d)
        np.clip(d, 0.0, 1.0, out=d)
        d **= two_kappa
        out[lo:hi] = scale * d.reshape(hi - lo, n, n_sym).sum(axis=2)
    return out


# ---------------------------------------------------------------------------
# Volume-weighted measures on the fixed orientation set
# ---------------------------------------------------------------------------

def orientation_weights(perm: np.ndarray, grain_weights: np.ndarray) -> np.ndarray:
    """Volume-fraction vector over ORIENTATION-index space for assignment
    *perm* (``perm[i]`` = orientation index carried by grain ``i``, the
    same "grain -> orientation index" convention as
    :attr:`~grainsmith.orientation.mdf.MDFResult.permutation`).

    ``v[perm[i]] = grain_weights[i] / sum(grain_weights)`` for every grain
    ``i`` -- this is exactly ``f_V(pi)`` as an ``(N,)`` weight vector
    summing to 1.  *grain_weights* may be RAW volumes (e.g. cubic
    angstroms, as returned by ``analysis.grains.grain_volumes``) or
    already-normalised volume fractions -- both are accepted and
    normalised internally, so callers never need to remember to divide by
    the total themselves (a real failure mode: every weight-consuming
    function in this module implicitly assumed ``sum(w) == 1``, and
    nothing enforced it before this fix).

    Raises ConfigError if *grain_weights* has a non-finite or non-positive
    entry, or if *perm* is not EXACTLY a permutation of ``range(N)``
    (:func:`_validate_permutation` -- a repeated index would otherwise
    leave part of the output as UNINITIALISED MEMORY from ``np.empty``).
    """
    w = _validate_positive_finite(grain_weights, "grain_weights")
    w = w / float(np.sum(w))
    perm = _validate_permutation(perm, len(w), "perm")
    v = np.empty(len(w), dtype=np.float64)
    v[perm] = w
    return v


def class_masses(v: np.ndarray, class_of: np.ndarray, n_classes: int) -> np.ndarray:
    """Total weight ``m_c = sum_{k: class_of[k]==c} v[k]`` carried by each
    orientation CLASS (see :func:`orientation_classes`) -- the object the
    atomic drift (:func:`atomic_drift`, :class:`DriftTracker`) is defined
    on, so that two grains sharing the same orientation never contribute a
    spurious nonzero cost when swapped."""
    v = np.asarray(v, dtype=np.float64)
    class_of = np.asarray(class_of, dtype=np.intp)
    return np.bincount(class_of, weights=v, minlength=n_classes)


def component_volume_fractions(
    component_of: np.ndarray,
    grain_weights: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """VOLUME-weighted fraction of material realised in each ODF texture
    component: ``sum_{i: component_of[i]==c} (w_i / sum w)`` for every
    component ``c``.

    Why this belongs next to the rest of the module: the same count-vs-
    volume confusion that arises for assignment annealing (the
    module docstring's fact (A)/(B)) recurs, in a SECOND place, in the ODF
    sampler itself.  ``orientation.samplers.odf_components`` realises the
    user's configured per-component ``weight`` fractions as a
    GRAIN-COUNT (Categorical draw) fraction, while ``odf_mtex.txt``
    exports VOLUME fractions -- so the configured weight and the
    material actually realised in a component can differ, exactly the
    way f_n and f_V differ elsewhere.  A QA gate can report the
    configured weights next to these realised volume fractions so the
    gap is visible directly, rather than only inferable.

    *grain_weights* may be RAW volumes or already-normalised fractions
    (normalised internally by the sum, same convention as
    :func:`orientation_weights` / :func:`count_vs_volume_gap`).
    *component_of* is a MANY-TO-ONE labelling (the texture-component index
    of each grain), NOT a permutation -- unlike *perm* in
    :func:`orientation_weights`, a repeated or entirely absent component
    index is the ordinary case (most components contain more than one
    grain, and a component can legitimately end up empty), so it is
    validated only for being finite, length-matched to *grain_weights*,
    and in ``[0, n_components)`` -- never for bijectivity.

    Raises ConfigError if *grain_weights* has a non-finite or non-positive
    entry, if *component_of* is non-finite or a different length than
    *grain_weights*, or if any *component_of* entry falls outside
    ``[0, n_components)``.
    """
    w = _validate_positive_finite(grain_weights, "grain_weights")
    w = w / float(np.sum(w))
    comp = np.asarray(component_of)
    if len(comp) != len(w):
        raise ConfigError(
            "component_of and grain_weights must have the same length; "
            f"got {len(comp)} and {len(w)}."
        )
    if not np.all(np.isfinite(np.asarray(comp, dtype=np.float64))):
        raise ConfigError(f"component_of must be finite; got {component_of!r}.")
    if comp.dtype.kind not in "iuf" or np.any(comp != np.floor(comp)):
        raise ConfigError("component_of must contain integer indices, not fractions.")
    if len(comp) and (int(comp.min()) < 0 or int(comp.max()) >= n_components):
        raise ConfigError(
            f"component_of must contain only indices in [0, {n_components}); "
            f"got values ranging [{int(comp.min())}, {int(comp.max())}]."
        )
    comp = comp.astype(np.intp)
    return np.bincount(comp, weights=w, minlength=n_components)


# ---------------------------------------------------------------------------
# Volume-balanced component assignment (weight_basis="volume")
# ---------------------------------------------------------------------------

def volume_balanced_partition(
    weights: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Deterministic assignment of grains to texture components so the
    VOLUME fraction of material in component ``c`` matches ``weights[c]``.

    This is the ``weight_basis="volume"`` counterpart to the categorical
    (grain-COUNT) draw ``orientation.samplers.odf_components`` uses under
    ``weight_basis="count"`` -- see that function and
    ``config.schema.OrientationConfig.component_weight_basis`` for the
    two realisations' respective trade-offs.

    Algorithm -- EXACTLY this; it is deterministic and MUST stay so (do
    not replace the stable descending sort or ``np.argmax``'s first-max
    tie rule with anything else -- an unstable sort or a random
    tie-break would silently break bit-reproducibility)::

        w       = weights / weights.sum()
        T       = volumes.sum()
        deficit = w * T
        for i in argsort(-volumes, kind="stable"):  # desc, ties -> lower grain index
            c = argmax(deficit)                      # ties -> lowest component index
            comp[i] = c
            deficit[c] -= volumes[i]

    No RNG is used anywhere in this function.

    PROVEN INVARIANT
    -----------------
    Lemma.  When grain i is assigned to c, the sum of all deficits equals
    the volume not yet assigned, which is >= volumes[i] > 0, so the
    maximum deficit is >= remaining/K > 0.  Hence after the assignment
    deficit_c > -volumes[i].  A deficit only ever changes again when that
    component receives another grain, which by the descending order is
    smaller.  THEREFORE no component ever overshoots its volume target by
    more than the SMALLEST grain it received.

    THE HONEST COST -- not hidden
    -------------------------------
    Demanding an exact volume match on a finite grain set with a broad
    size distribution necessarily induces a correlation between grain
    size and component (the largest grain deterministically lands in the
    largest-deficit component).  That correlation vanishes as N grows and
    is REPORTED by gate G25 (per-component mean grain volume).  It is a
    stated trade-off, not a bug.  Callers that cannot tolerate a
    size-component correlation should use ``weight_basis="count"``
    instead, which pays the opposite cost: no size correlation, but a
    volume-fraction mismatch set by the grain-size distribution (measured:
    a configured 0.500/0.500 realises as 0.745/0.255 by volume for 24
    grains drawn from a lognormal size distribution with sigma_log 0.5 --
    the counter-example that motivated the volume basis).

    Parameters
    ----------
    weights : (K,) array
        Target (relative) volume fraction of each of K components; must
        be finite and strictly positive (normalised internally by the
        sum -- need not already sum to 1).
    volumes : (N,) array
        Per-grain volume (raw or already a fraction, any consistent
        positive unit); must be finite and strictly positive.

    Returns
    -------
    (N,) np.intp array
        ``comp[i]`` is the component index (in ``[0, K)``) grain ``i`` is
        assigned to.  A component can legitimately end up with ZERO
        grains (its deficit never becomes the largest remaining one
        before it is exhausted by other components) -- callers must not
        assume every component receives at least one grain.

    Raises
    ------
    ConfigError
        If *weights* or *volumes* is non-finite or non-positive anywhere
        (same style as the other validators in this module -- see
        :func:`_validate_positive_finite`).
    """
    w = _validate_positive_finite(weights, "weights")
    v = _validate_positive_finite(volumes, "volumes")
    w = w / float(np.sum(w))
    total = float(np.sum(v))
    deficit = w * total
    comp = np.empty(len(v), dtype=np.intp)
    for i in np.argsort(-v, kind="stable"):
        c = int(np.argmax(deficit))
        comp[i] = c
        deficit[c] -= v[i]
    return comp


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mmd(v_a: np.ndarray, v_b: np.ndarray, gram: np.ndarray) -> float:
    """Kernel discrepancy ``sqrt((v_a - v_b)^T M (v_a - v_b))`` between two
    weight vectors over the SAME orientation set, using the symmetrized
    Gram matrix *gram* (:func:`symmetrized_gram`).

    Because *gram* is symmetric positive (semi)definite, this quadratic
    form is never negative in exact arithmetic; the ``max(..., 0.0)`` guard
    only absorbs float roundoff for near-identical vectors, it is not a
    correction of the underlying math (see the module docstring's
    "SYMMETRIZED GRAM MATRIX" section for why MMD is a pseudometric, and
    what it is -- and is NOT -- exactly equal to).
    """
    d = np.asarray(v_a, dtype=np.float64) - np.asarray(v_b, dtype=np.float64)
    quad = float(d @ np.asarray(gram, dtype=np.float64) @ d)
    return math.sqrt(max(quad, 0.0))


def atomic_drift(
    v: np.ndarray, v_ref: np.ndarray, class_of: np.ndarray, n_classes: int,
) -> float:
    """Exact atomic (un-smoothed) TV drift ``0.5 * sum_c |m_c(v) -
    m_c(v_ref)|`` between two weight vectors' CLASS masses.

    This is the control quantity: because kernel smoothing is a TV
    contraction (module docstring), a bound on this atomic quantity bounds
    the TV of kernel-smoothed densities at EVERY half-width at once. The
    MMD diagnostic is a different norm, not numerically capped by epsilon.

    With classes from :func:`orientation_classes` (the annealer's control
    path) the value is defined on the LABELLED orientation set and is an
    upper bound on the physical quotient-space drift (module docstring,
    "SYMMETRY-QUOTIENT CLASSES"); pass classes from
    :func:`symmetry_classes` to measure the quotient-space value instead.
    """
    m = class_masses(v, class_of, n_classes)
    m_ref = class_masses(v_ref, class_of, n_classes)
    return 0.5 * float(np.sum(np.abs(m - m_ref)))


def count_vs_volume_gap(grain_weights: np.ndarray) -> float:
    """Atomic TV distance ``0.5 * sum_i |w_i - 1/N|`` between the
    volume-weighted and number-weighted measures on N grains -- fact (B)
    of the module docstring: this is a property of the grain-SIZE
    distribution alone and is the same for every assignment pi (a
    permutation only relabels which atom carries which w_i, it cannot
    change the multiset of |w_i - 1/N| terms being summed).

    *grain_weights* may be RAW volumes or already-normalised volume
    fractions -- normalised internally (divided by their sum) before the
    gap is computed, so e.g. passing the pipeline's raw per-grain volumes
    in cubic angstroms gives the same answer as passing volume fractions
    (a real, measured failure mode before this fix: raw lognormal volumes
    for N=250 gave 175.6 instead of the correct 0.319).
    """
    w = _validate_positive_finite(grain_weights, "grain_weights")
    w = w / float(np.sum(w))
    n = len(w)
    return 0.5 * float(np.sum(np.abs(w - 1.0 / n)))


def effective_sample_size(volumes: np.ndarray) -> float:
    """Kish effective sample size ``(sum V)^2 / sum V^2`` of a grain-volume
    distribution: N for equal volumes, tending to 1 as a single grain
    comes to dominate the total volume.  Scale-invariant (raw volumes or
    normalised fractions give the same answer), used to gauge how much
    the volume-weighted measure can differ from the number-weighted one --
    a small effective sample size is exactly the regime where
    :func:`count_vs_volume_gap` is large.
    """
    v = _validate_positive_finite(volumes, "volumes")
    total = float(np.sum(v))
    total_sq = float(np.sum(v * v))
    return (total * total) / total_sq


def null_mmd(
    gram: np.ndarray,
    grain_weights: np.ndarray,
    n_samples: int = ODF_NULL_SAMPLES,
    seed: int = ODF_NULL_SEED,
) -> np.ndarray:
    """Null distribution of :func:`mmd` between a NEUTRAL random
    assignment's volume-weighted vector and the count-weighted (uniform)
    vector, for the fixed *grain_weights* multiset and *gram* matrix.

    Each of the *n_samples* draws assigns the SAME multiset of grain
    volume-fraction weights to a uniformly random permutation of the
    orientation slots, and measures its kernel discrepancy against the
    perfectly-uniform count-weighted measure ``1/N`` -- i.e. "how large
    does the kernel-smoothed volume-vs-count gap look under an assignment
    that is not adversarially biased toward any particular orientation".
    Comparing an ANNEALED assignment's MMD against the upper quantiles of
    this null is what distinguishes "this much smoothed drift is just the
    unavoidable atomic count-vs-volume gap (fact B), seen through the
    kernel" from "the annealer specifically concentrated volume onto a
    subset of orientations".

    THIS IS THE NULL FOR ``mmd(v, uniform, gram)`` ONLY -- i.e. for
    :attr:`ODFDiagnostics.mmd_vs_count` -- and for NOTHING ELSE.  In
    particular it is NOT a null for ``mmd(v_final, v_initial, gram)``
    (:attr:`ODFDiagnostics.mmd_vs_initial`): that statistic measures
    distance to a *specific* pre-annealing assignment, a single draw that
    carries its own sampling noise, not to the deterministic f_n this
    function's samples are all measured against.  Placing a statistic in
    a null distribution is only a valid test when both are measured
    against the SAME reference point -- comparing mismatched references
    was a real defect caught in review of this module, so a future caller
    must not repeat it: always pair this null with ``mmd(v, uniform,
    gram)`` for the SAME ``v``, never with a distance to any other
    assignment.

    Uses a LOCAL ``np.random.Generator(np.random.PCG64(seed))`` -- never a
    module-level ``np.random.*`` call -- so two calls with the same
    arguments return a BIT-IDENTICAL array regardless of any other RNG
    activity in the process (see the module docstring's "DETERMINISM"
    section; the default *seed* is a property of the CODE, matching
    ``MDF_REFERENCE_SEED``'s reasoning in orientation/mdf.py).
    """
    w = _validate_positive_finite(grain_weights, "grain_weights")
    gram = np.asarray(gram, dtype=np.float64)
    n = len(w)
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    gen = np.random.Generator(np.random.PCG64(seed))
    out = np.empty(int(n_samples), dtype=np.float64)
    for s in range(int(n_samples)):
        perm = gen.permutation(n)
        v = orientation_weights(perm, w)
        out[s] = mmd(v, uniform, gram)
    return out


# ---------------------------------------------------------------------------
# Incremental drift control
# ---------------------------------------------------------------------------

class DriftTracker:
    """O(1)-per-swap tracker of the atomic drift (:func:`atomic_drift`)
    from a fixed reference assignment ``pi0``, for use inside an annealing
    inner loop (orientation/mdf.py's ``anneal_assignment``) without paying
    an O(N) :func:`class_masses` recomputation on every trial move.

    Maintains ``dev[c] = m_c(current) - m_c(pi0)`` per class.  A swap of
    grains ``a`` and ``b`` currently holding orientation indices
    ``k_a = pi(a)``, ``k_b = pi(b)`` changes AT MOST two classes' masses
    (fact A, generalised to classes): if ``class(k_a) == class(k_b)`` the
    swap trades weight WITHIN one class and drift is exactly unchanged;
    otherwise exactly ``dev[class(k_a)]`` and ``dev[class(k_b)]`` shift by
    ``+-(w_b - w_a)`` and the drift (``0.5 * sum |dev|``) updates by only
    the two changed terms.  Verified exact to 4e-15 against a from-scratch
    recomputation over 20000 random swaps.

    Parameters
    ----------
    perm0 : (N,) int
        Reference "grain -> orientation index" assignment (drift is
        measured relative to this).  Must be EXACTLY a permutation of
        ``range(N)`` (:func:`_validate_permutation`).
    grain_weights : (N,) float64
        Weight of each grain -- RAW volumes or already-normalised volume
        fractions, either is accepted: normalised internally (divided by
        their sum) once, here, and the normalised form is what every
        internal computation (including the incremental swap update) uses
        (a real, measured failure mode before this fix: raw volumes fed
        straight through gave a drift of 21.48 instead of the correct
        0.0610 for a 250-grain lognormal example).
    class_of : (N,) int
        Orientation class of each orientation INDEX (:func:`orientation_classes`).
    n_classes : int
    """

    def __init__(
        self,
        perm0: np.ndarray,
        grain_weights: np.ndarray,
        class_of: np.ndarray,
        n_classes: int,
    ) -> None:
        w = _validate_positive_finite(grain_weights, "grain_weights")
        self._w = w / float(np.sum(w))  # normalised once; see class docstring
        self._perm = _validate_permutation(perm0, len(self._w), "perm0")
        v0 = np.empty(len(self._w), dtype=np.float64)
        v0[self._perm] = self._w
        self._class_of = np.asarray(class_of, dtype=np.intp).reshape(-1)
        self._n_classes = int(n_classes)
        self._m0 = class_masses(v0, self._class_of, self._n_classes)
        self._dev = np.zeros(self._n_classes, dtype=np.float64)
        self._value = 0.0

    @property
    def value(self) -> float:
        """Current tracked drift relative to ``pi0``."""
        return self._value

    def _delta(self, a: int, b: int, k_a: int, k_b: int):
        """Shared math for :meth:`trial`/:meth:`commit`: returns
        ``(c_a, c_b, new_a, new_b, delta)`` where *delta* is the change in
        ``.value`` a swap of grains a, b (holding orientation indices
        k_a, k_b) would cause; ``new_a``/``new_b`` are ``None`` when
        ``c_a == c_b`` (nothing to update).

        Verifies ``k_a``/``k_b`` against this tracker's OWN copy of the
        current assignment before touching anything: the caller (the
        annealer) maintains its own permutation and passes k_a/k_b in
        rather than have this method look them up, purely to avoid a
        redundant array read on the hot path -- but if the caller's
        permutation and this tracker's copy have desynced (e.g. a swap
        applied to one but not the other, or applied and then reverted
        inconsistently), trusting a wrong k_a/k_b would silently apply the
        deviation update to the WRONG orientation classes. A single swap
        can even give the numerically same delta by coincidence, so a
        desync would otherwise hide until it had silently corrupted many
        steps -- hence a hard, immediate failure here instead."""
        if int(self._perm[a]) != int(k_a) or int(self._perm[b]) != int(k_b):
            raise GrainsmithError(
                "DriftTracker desync: internal perm has "
                f"perm[{a}]={int(self._perm[a])}, perm[{b}]={int(self._perm[b])}, "
                f"but was called with k_a={k_a}, k_b={k_b}. The caller's "
                "tracked assignment has drifted out of sync with "
                "DriftTracker's own copy -- every commit() must be paired "
                "with the SAME swap applied to the caller's permutation."
            )
        c_a = int(self._class_of[k_a])
        c_b = int(self._class_of[k_b])
        if c_a == c_b:
            return c_a, c_b, None, None, 0.0
        w_a = self._w[a]
        w_b = self._w[b]
        dev_a = self._dev[c_a]
        dev_b = self._dev[c_b]
        new_a = dev_a + (w_b - w_a)
        new_b = dev_b + (w_a - w_b)
        delta = 0.5 * ((abs(new_a) - abs(dev_a)) + (abs(new_b) - abs(dev_b)))
        return c_a, c_b, new_a, new_b, delta

    def trial(self, a: int, b: int, k_a: int, k_b: int) -> float:
        """Drift value AFTER hypothetically swapping grains a and b
        (currently at orientation indices k_a, k_b), WITHOUT mutating any
        tracked state -- safe to call any number of times, e.g. to
        evaluate a Metropolis move before deciding whether to
        :meth:`commit` it."""
        _, _, _, _, delta = self._delta(a, b, k_a, k_b)
        return self._value + delta

    def commit(self, a: int, b: int, k_a: int, k_b: int) -> None:
        """Apply the swap of grains a and b (at orientation indices
        k_a, k_b) to the tracked state: updates ``.value``, the per-class
        deviations, and the internal copy of the assignment (needed so
        :meth:`recompute` can rebuild the exact state from scratch)."""
        c_a, c_b, new_a, new_b, delta = self._delta(a, b, k_a, k_b)
        if new_a is not None:
            self._dev[c_a] = new_a
            self._dev[c_b] = new_b
            self._value += delta
        self._perm[a], self._perm[b] = self._perm[b], self._perm[a]

    def recompute(self) -> float:
        """Recompute ``.value`` EXACTLY from the current assignment (a
        fresh :func:`class_masses` pass), replacing the incrementally
        tracked value.  Call this once at the end of an anneal so the
        REPORTED drift never carries accumulated floating-point error from
        thousands of incremental :meth:`commit` updates.

        Builds ``v`` directly from the tracker's own (already normalised
        in ``__init__``) ``self._perm``/``self._w`` rather than via
        :func:`orientation_weights`, which would re-validate and
        re-normalise (dividing by a sum already extremely close to 1) on
        every call -- harmless numerically but wasted work on what is
        meant to be a cheap, exact closing check.
        """
        v = np.empty(len(self._w), dtype=np.float64)
        v[self._perm] = self._w
        m = class_masses(v, self._class_of, self._n_classes)
        self._dev = m - self._m0
        self._value = 0.5 * float(np.sum(np.abs(self._dev)))
        return self._value


# ---------------------------------------------------------------------------
# Diagnostics record (plain data holder for the pipeline gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ODFDiagnostics:
    """Plain data holder summarising the volume-weighted ODF diagnostics
    for one run, at one kernel half-width -- assembled by a later pipeline
    stage (the QA gate that consumes this module) and reported alongside
    the other QA gates. Carries no behaviour of its own.

    Attributes
    ----------
    halfwidth_deg : float
        de la Vallee Poussin kernel half-width these diagnostics were
        computed at.
    n_eff : float
        :func:`effective_sample_size` of the run's grain volumes.
    count_vs_volume_gap : float
        :func:`count_vs_volume_gap` of the run's grain volume fractions
        (assignment-independent, fact B).
    drift : float
        Final :func:`atomic_drift` of the annealed assignment relative to
        its pre-annealing reference (:class:`DriftTracker`'s reported
        value after :meth:`~DriftTracker.recompute`).
    drift_max : float | None
        Configured cap on ``drift`` during annealing, or ``None`` when no
        cap was configured.
    mmd_vs_count : float
        :func:`mmd` between the annealed volume-weighted vector and the
        count-weighted (uniform 1/N) vector, at ``halfwidth_deg`` -- THE
        SAME reference point :func:`null_mmd` samples its null
        distribution against.  ``null_quantile`` refers to THIS field:
        placing a statistic in a null distribution is only a valid test
        when both are measured against the same reference, and f_n
        (deterministic, the discrete orientation distribution the sampler
        intended) rather than pi_0 (a single neutral draw, itself subject
        to sampling noise) is that common reference -- see
        :func:`null_mmd`'s docstring.
    mmd_vs_initial : float
        :func:`mmd` between the annealed and pre-annealing volume-weighted
        vectors, at ``halfwidth_deg``.  DESCRIPTIVE ONLY -- how far the
        anneal moved the smoothed ODF from its starting assignment.  It
        has NO null distribution here (``null_mmd`` samples distances to
        f_n, not to pi_0) and must NOT be compared against
        ``null_median``/``null_p95``/``null_quantile``.
    null_median, null_p95 : float
        Median and 95th percentile of :func:`null_mmd` for this run's
        grain weights and Gram matrix -- the null distribution of
        ``mmd_vs_count``, NOT of ``mmd_vs_initial`` (see above).
    null_quantile : float
        Quantile of the null distribution that ``mmd_vs_count`` (NOT
        ``mmd_vs_initial`` -- see above) falls at (0 = at or below the
        null's minimum sample, 1 = at or above its maximum).
    """
    halfwidth_deg: float
    n_eff: float
    count_vs_volume_gap: float
    drift: float
    drift_max: float | None
    mmd_vs_count: float
    mmd_vs_initial: float
    null_median: float
    null_p95: float
    null_quantile: float
