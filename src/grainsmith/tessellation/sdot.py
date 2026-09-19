"""Semi-discrete optimal transport: power-diagram weights for prescribed
grain volumes.

Damped Newton on the Kantorovich dual (Kitagawa–Mérigot–Thibert, JEMS 2019;
applied to polycrystals by Bourne–Kok–Roper–Spanjer, Phil. Mag. 2020 and,
periodically, Bourne et al. 2022):

    F(w) = V(w) − V^target,   J_ij = ∂V_i/∂w_j
    J_ii = +Σ_j A_ij/(2 d_ij),  J_ij = −A_ij/(2 d_ij)   (i ≠ j)

with A_ij the EXACT interface area and d_ij = ‖replica_j − c_i‖ — both read
off the polyhedral cell faces, which is why the power backend keeps exact
geometry rather than voxel estimates.  J is a graph Laplacian (rows sum to
0, gauge w → w + c·1); the gauge is fixed by pinning w_0 = 0.

Damping (KMT): backtrack t ← t/2 until min_i V_i(w + tδ) ≥ ε₀ with
ε₀ = ½·min(min_i V_i(w⁰), min_i V_i^target) and the residual decreases —
no cell ever vanishes, global convergence, quadratic near the solution.
An empty-cell TessellationError during a trial build counts as a rejected
step (halve t).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from grainsmith.errors import ConfigError, TessellationError
from grainsmith.tessellation.power import PowerTessellation

log = logging.getLogger(__name__)

_MAX_BACKTRACK = 25
"""Newton backtracking halvings before declaring the step failed: 2^-25
≈ 3e-8 of the full step — far below any useful progress."""


@dataclass
class SDOTResult:
    """Outcome of the volume fit (recorded in summary.csv)."""
    tess: PowerTessellation
    weights: np.ndarray
    target_volumes: np.ndarray
    achieved_volumes: np.ndarray
    max_rel_error: float
    iterations: int
    converged: bool


def sample_target_volumes(
    dist_type: str,
    n: int,
    box_volume: float,
    rng: np.random.Generator,
    sigma_log: float = 0.35,
    volumes: list[float] | None = None,
) -> np.ndarray:
    """Per-grain target volumes summing EXACTLY to box_volume.

    lognormal: equivalent-sphere diameters d ~ LogN(0, sigma_log), V ∝ d³,
    then normalized — the user prescribes the SHAPE (sigma_log); the scale
    is fixed by V_box/N.
    """
    if dist_type == "equal":
        v = np.full(n, 1.0, dtype=np.float64)
    elif dist_type == "lognormal":
        if sigma_log <= 0.0:
            raise ConfigError(
                f"size_distribution.sigma_log must be > 0, got {sigma_log}.")
        d = rng.lognormal(mean=0.0, sigma=sigma_log, size=n)
        v = d ** 3
    elif dist_type == "volumes":
        if volumes is None or len(volumes) != n:
            got = 0 if volumes is None else len(volumes)
            raise ConfigError(
                f"size_distribution.volumes must list exactly "
                f"grains.number={n} relative volumes, got {got}.")
        v = np.asarray(volumes, dtype=np.float64)
        if np.any(v <= 0.0):
            raise ConfigError(
                "size_distribution.volumes must all be positive.")
    else:
        raise ConfigError(
            f"Unknown size_distribution.type: {dist_type!r} "
            "(lognormal | equal | volumes).")
    return v * (box_volume / float(np.sum(v)))


def fit_power_weights(
    seeds: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    target_volumes: np.ndarray,
    vol_tol: float = 1e-3,
    max_iter: int = 30,
    centroidal_iterations: int = 0,
) -> SDOTResult:
    """Fit power-diagram weights so cell volumes match *target_volumes*.

    Optional outer centroidal loop (Kuhn et al., CMAME 2020): after each
    volume fit, move seeds to their power-cell centroids (min-image
    rewrap) and refit — equiaxed grains WITH prescribed volumes.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64).copy()
    targets = np.asarray(target_volumes, dtype=np.float64)
    n = len(seeds)
    if targets.shape != (n,):
        raise ConfigError(
            f"target_volumes shape {targets.shape} != ({n},).")
    box_vol = float(np.prod(L))
    if abs(float(np.sum(targets)) - box_vol) > 1e-9 * box_vol:
        raise ConfigError(
            "target volumes must sum to the box volume "
            f"({np.sum(targets):.6g} vs {box_vol:.6g} Å³).")

    result = _newton_fit(seeds, L, periodic, targets, vol_tol, max_iter)
    for k in range(centroidal_iterations):
        seeds = _power_centroids(result.tess, seeds, L, periodic)
        log.info("sdot: centroidal iteration %d/%d",
                 k + 1, centroidal_iterations)
        result = _newton_fit(seeds, L, periodic, targets, vol_tol, max_iter)
    return result


def _newton_fit(
    seeds: np.ndarray,
    L: np.ndarray,
    periodic: list[bool],
    targets: np.ndarray,
    vol_tol: float,
    max_iter: int,
) -> SDOTResult:
    n = len(seeds)
    w = np.zeros(n, dtype=np.float64)
    tess = PowerTessellation(seeds, L, periodic, weights=w)
    vols = _volumes(tess)
    eps0 = 0.5 * min(float(np.min(vols)), float(np.min(targets)))

    it = 0
    for it in range(1, max_iter + 1):
        rel = np.abs(vols - targets) / targets
        if float(np.max(rel)) <= vol_tol:
            return SDOTResult(tess, w, targets, vols,
                              float(np.max(rel)), it - 1, True)

        J = _jacobian(tess, n)
        F = vols - targets
        delta = _solve_gauge_fixed(J, -F)

        # KMT damping: never let a cell shrink below eps0; require the
        # residual to decrease.
        t = 1.0
        accepted = False
        f_norm = float(np.max(np.abs(F)))
        for _ in range(_MAX_BACKTRACK):
            try:
                trial = PowerTessellation(seeds, L, periodic,
                                          weights=w + t * delta)
            except TessellationError:
                t *= 0.5
                continue
            tvols = _volumes(trial)
            if (float(np.min(tvols)) >= eps0
                    and float(np.max(np.abs(tvols - targets))) < f_norm):
                tess, vols, w = trial, tvols, w + t * delta
                accepted = True
                break
            t *= 0.5
        if not accepted:
            raise TessellationError(
                f"SDOT Newton step rejected after {_MAX_BACKTRACK} "
                f"halvings at iteration {it} (max rel err "
                f"{float(np.max(rel)):.3e}). The target spread may be "
                "infeasible for this seed configuration — reduce "
                "sigma_log or increase grains.")
        log.debug("sdot it %d: max rel err %.3e (step %.3g)",
                  it, float(np.max(np.abs(vols - targets) / targets)), t)

    rel = np.abs(vols - targets) / targets
    return SDOTResult(tess, w, targets, vols,
                      float(np.max(rel)), it, bool(np.max(rel) <= vol_tol))


def _volumes(tess: PowerTessellation) -> np.ndarray:
    return np.array([c.volume for c in tess.cells], dtype=np.float64)


def _jacobian(tess: PowerTessellation, n: int) -> np.ndarray:
    """J_ij = ∂V_i/∂w_j from exact face areas and replica distances.

    Self-image faces (i == j) are excluded: both sides share w_i, the
    plane never moves.  Wall (mirror) faces likewise (weights cancel).
    Periodic multi-faces between the same pair accumulate."""
    J = np.zeros((n, n), dtype=np.float64)
    for cell in tess.cells:
        i = cell.grain_id
        for face in cell.faces:
            j = face.neighbor_id
            if j < 0 or j == i:
                continue
            coef = face.area / (2.0 * face.seed_distance)
            J[i, i] += coef
            J[i, j] -= coef
    return J


def _solve_gauge_fixed(J: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve J δ = rhs on the quotient by constants: pin δ_0 = 0.

    J restricted to indices 1..N−1 is a nonsingular M-matrix when the
    adjacency graph is connected (always, for a tessellation of a box)."""
    n = len(rhs)
    delta = np.zeros(n, dtype=np.float64)
    A = J[1:, 1:]
    b = rhs[1:]
    try:
        delta[1:] = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        delta[1:] = np.linalg.lstsq(A, b, rcond=None)[0]
    return delta


def _power_centroids(
    tess: PowerTessellation,
    seeds: np.ndarray,
    L: np.ndarray,
    periodic: list[bool],
) -> np.ndarray:
    """Exact polyhedral centroids (signed tetra fan over outward faces),
    rewrapped into the box on periodic axes."""
    new = np.empty_like(seeds)
    for cell in tess.cells:
        origin = cell.vertices.mean(axis=0) + seeds[cell.grain_id]
        vol = 0.0
        cen = np.zeros(3)
        for face in cell.faces:
            v = face.vertices
            for t in range(1, len(v) - 1):
                a, b, c = v[0] - origin, v[t] - origin, v[t + 1] - origin
                v6 = float(np.dot(a, np.cross(b, c)))   # 6 × signed volume
                # tetra centroid (absolute) = origin + (a+b+c)/4
                cen += v6 * (origin + (a + b + c) / 4.0)
                vol += v6
        cen /= vol
        new[cell.grain_id] = cen
    for ax in range(3):
        if periodic[ax]:
            new[:, ax] %= L[ax]
            col = new[:, ax]
            col[col >= L[ax]] = 0.0
    return new
