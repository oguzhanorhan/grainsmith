"""Grain orientation samplers (§6.10).

All samplers return (N, 4) float64 array of scalar-first unit quaternions.
Uses ONLY the rng stream passed in — no np.random.* module calls (keeps RNG
streams deterministic and reproducible).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from grainsmith.constants import PARALLEL_DOT_TOL, RNG_MAX_ATTEMPTS_FACTOR, ZERO_VECTOR_TOL
from grainsmith.errors import ConfigError
from grainsmith.orientation.quaternion import (
    axis_angle_to_quat,
    from_scipy,
    quat_normalize,
)

SAMPLE_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
"""Sample-direction strings accepted by fiber specs (§7)."""


def random_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample N orientations uniformly over SO(3) (Haar measure, Shoemake 1992).

    Uses scipy.spatial.transform.Rotation.random which implements the correct
    uniform distribution (angle pdf p(θ) ∝ (1−cos θ) on [0, π]).

    Returns (N, 4) float64 scalar-first quaternions.
    """
    # scipy uses random_state as a numpy Generator — compatible.
    # from_scipy handles rotation stacks vectorized (no per-element loop).
    rot = _Rotation.random(n, random_state=rng)
    return from_scipy(rot).reshape(-1, 4)


def fixed_orientation(
    n: int,
    spec: dict,
    cell_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Return N copies of one fixed orientation given by spec dict (§6.10, §7).

    Spec must have exactly one of:
        euler_bunge_deg: [phi1, Phi, phi2]
        axis_angle: {axis: [h,k,l], angle_deg: float}
        quaternion: [w, x, y, z]
        hkl_uvw: {plane: [h,k,l], direction: [u,v,w]}  (requires cell_matrix)
    """
    keys = [k for k in spec if spec[k] is not None]
    if len(keys) != 1:
        raise ConfigError(
            f"orientation.fixed must have exactly one sub-key. Got: {keys}"
        )
    key = keys[0]

    if key == "euler_bunge_deg":
        phi1, Phi, phi2 = spec[key]
        # Standard Bunge (§1): R(q) = intrinsic-ZXZ(phi1,Phi,phi2) so the
        # Bunge matrix g = R(q).T factorizes as Rz(phi2)Rx(Phi)Rz(phi1)
        # (MTEX/EBSD convention — not the inverse rotation).
        rot_R = _Rotation.from_euler("ZXZ", [phi1, Phi, phi2], degrees=True)
        q = from_scipy(rot_R)
    elif key == "axis_angle":
        axis = np.array(spec[key]["axis"], dtype=np.float64)
        angle = float(spec[key]["angle_deg"])
        q = axis_angle_to_quat(axis, angle)
    elif key == "quaternion":
        q = quat_normalize(np.array(spec[key], dtype=np.float64))
    elif key == "hkl_uvw":
        if cell_matrix is None:
            raise ConfigError("hkl_uvw orientation requires cell_matrix to be provided.")
        plane = np.array(spec[key]["plane"], dtype=np.float64)
        direction = np.array(spec[key]["direction"], dtype=np.float64)
        q = _hkl_uvw_to_quat(plane, direction, cell_matrix)
    else:
        raise ConfigError(f"Unknown orientation.fixed key: {key!r}")

    return np.tile(q, (n, 1))


def _hkl_uvw_to_quat(
    plane: np.ndarray,
    direction: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Construct quaternion so that crystal plane (hkl) ∥ lab z and [uvw] ∥ lab x.

    Uses crystallographic metric:
        n_cart ∝ A^{-T} @ hkl   (reciprocal-space normal)
        d_cart ∝ A @ uvw         (direct-space direction)
    """
    Ainv = np.linalg.inv(A)
    # Cartesian normal to (hkl): n = A^{-T} @ hkl
    n = (Ainv.T @ plane)
    n_norm = float(np.linalg.norm(n))
    # Cartesian direction [uvw]: d = A @ uvw
    d = A @ direction
    d_norm = float(np.linalg.norm(d))
    if n_norm < ZERO_VECTOR_TOL or d_norm < ZERO_VECTOR_TOL:
        raise ConfigError(
            f"hkl_uvw: plane {plane.tolist()} and direction {direction.tolist()} "
            "must both be non-zero vectors.")
    n /= n_norm
    d /= d_norm
    # Check orthogonality
    dot = float(np.dot(n, d))
    if abs(dot) > 1e-6:
        raise ConfigError(
            f"hkl_uvw: plane {plane.tolist()} and direction {direction.tolist()} "
            f"are not orthogonal (dot product = {dot:.6g}). "
            "For cubic crystals [uvw] must be perpendicular to (hkl)."
        )
    # Active rotation v_lab = R @ v_crystal with R @ d = x̂, R @ (n×d) = ŷ,
    # R @ n = ẑ requires d, n×d, n as the ROWS of R (right-handed: det = +1).
    # column_stack here would build the transpose (= inverse) instead;
    # pinned by tests/test_samplers.py.
    y_ax = np.cross(n, d)
    y_ax /= np.linalg.norm(y_ax)
    R = np.vstack([d, y_ax, n])  # (3,3) rows
    from grainsmith.orientation.quaternion import matrix_to_quat
    return matrix_to_quat(R)


def fiber_texture(
    n: int,
    crystal_axis: np.ndarray,
    sample_direction: np.ndarray,
    spread_deg: float,
    rng: np.random.Generator,
    cell_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Sample N orientations from a fiber texture (§6.10).

    crystal_axis ⟨uvw⟩ is aligned to sample_direction with Gaussian angular spread;
    azimuthal rotation is uniform.

    Returns (N, 4) float64 scalar-first quaternions.
    """
    from grainsmith.orientation.quaternion import axis_angle_to_quat, quat_mul

    # Cartesian crystal axis direction
    if cell_matrix is not None:
        ca = cell_matrix @ np.array(crystal_axis, dtype=np.float64)
    else:
        ca = np.array(crystal_axis, dtype=np.float64)
    ca_norm = float(np.linalg.norm(ca))

    sd = np.array(sample_direction, dtype=np.float64)
    sd_norm = float(np.linalg.norm(sd))
    if ca_norm < ZERO_VECTOR_TOL or sd_norm < ZERO_VECTOR_TOL:
        raise ConfigError(
            "fiber: crystal_axis and sample_direction must be non-zero vectors "
            f"(got crystal_axis={np.asarray(crystal_axis).tolist()}).")
    ca /= ca_norm
    sd /= sd_norm

    # Base rotation: ca → sd
    q_base = _rotation_between(ca, sd)

    quats = []
    spread_rad = np.radians(spread_deg)
    for _ in range(n):
        # Tilt by Gaussian spread around a perpendicular axis
        tilt_rad = rng.standard_normal() * spread_rad
        perp = _perp_to(sd, rng)
        q_tilt = axis_angle_to_quat(perp, np.degrees(tilt_rad))
        # Azimuthal rotation around sd
        az_deg = rng.uniform(0.0, 360.0)
        q_az = axis_angle_to_quat(sd, az_deg)
        q = quat_mul(q_az, quat_mul(q_tilt, q_base))
        if q[0] < 0:
            q = -q
        quats.append(q)
    return np.array(quats, dtype=np.float64)


def odf_components(
    n: int,
    components: list[dict],
    rng: np.random.Generator,
    cell_matrix: np.ndarray | None = None,
    *,
    return_component_index: bool = False,
    grain_volumes: np.ndarray | None = None,
    weight_basis: str = "volume",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample N orientations from weighted ODF texture components.

    Each component dict carries ``weight`` (> 0, relative) and exactly one
    of:

    - ``euler_bunge_deg: [φ1, Φ, φ2]`` — a discrete component centre; an
      optional ``spread_deg`` adds an isotropic angular spread (see
      :func:`_haar_gaussian_angles`),
    - ``fiber: {crystal_axis, sample_direction, spread_deg}`` — reuses the
      :func:`fiber_texture` machinery,
    - ``random: true`` — Haar-uniform fraction for partially textured
      states.

    Grain → component assignment depends on *weight_basis*:

    - ``"count"`` — one categorical draw ∝ weight per grain
      (``rng.choice``), byte-identical to the pre-1.2 behaviour and same
      RNG consumption.  The configured ``weight`` fractions are realised
      as GRAIN-COUNT fractions; the VOLUME fraction of material each
      component ends up representing is generally different (grain
      volumes are unequal) and is additionally reshuffled by any later
      assignment annealing (orientation/mdf.py permutes orientations —
      hence component labels — among grains of unequal volume).
    - ``"volume"`` (the physical ODF definition) — deterministic
      assignment via :func:`~grainsmith.orientation.odf.volume_balanced_partition`
      on *grain_volumes*; NO ``rng.choice`` call happens, so this mode
      consumes no RNG draws for the assignment step itself.  Requires
      *grain_volumes* (length ``n``); raises :class:`ConfigError`
      otherwise, or for any *weight_basis* other than ``"count"``/
      ``"volume"``.

    In BOTH modes, orientations are then sampled component-by-component in
    list order (deterministic stream consumption) — the per-component
    sampling loop itself does not depend on *weight_basis*.  Returns
    (N, 4) float64 scalar-first quaternions, w ≥ 0.

    Gate G25 reports configured weight vs realised count vs realised
    volume fraction side by side so any remaining gap is visible directly,
    and ``orientation.odf.component_volume_fractions`` computes the volume
    side from the per-grain component labels this sampler can return.

    With ``return_component_index=True``, returns ``(quats, comp_idx)``
    where ``comp_idx[i]`` is the component index grain ``i`` drew — the
    sampler's EXACT assignment, never a post-hoc nearest-centre
    reconstruction (which could not attribute overlapping spreads or the
    random component at all).  Default ``False`` keeps the historical
    return contract (quats only).
    """
    from grainsmith.orientation.odf import volume_balanced_partition
    from grainsmith.orientation.quaternion import quat_mul_batch

    weights = np.array([float(c["weight"]) for c in components],
                       dtype=np.float64)
    if weight_basis == "count":
        probs = weights / weights.sum()
        comp_idx = rng.choice(len(components), size=n, p=probs)
    elif weight_basis == "volume":
        if grain_volumes is None:
            raise ConfigError(
                "odf_components: weight_basis='volume' requires "
                "grain_volumes (per-grain volumes, length n); got None."
            )
        gv = np.asarray(grain_volumes, dtype=np.float64).reshape(-1)
        if len(gv) != n:
            raise ConfigError(
                f"odf_components: grain_volumes must have length n={n}; "
                f"got {len(gv)}."
            )
        comp_idx = volume_balanced_partition(weights, gv)
    else:
        raise ConfigError(
            "odf_components: weight_basis must be 'count' or 'volume'; "
            f"got {weight_basis!r}."
        )

    quats = np.empty((n, 4), dtype=np.float64)
    for k, comp in enumerate(components):
        rows = np.flatnonzero(comp_idx == k)
        m = len(rows)
        if m == 0:
            continue
        if comp.get("random"):
            quats[rows] = random_uniform(m, rng)
            continue
        if comp.get("fiber") is not None:
            fib = comp["fiber"]
            direction = SAMPLE_DIRECTIONS.get(fib["sample_direction"])
            if direction is None:
                raise ConfigError(
                    f"fiber component sample_direction must be one of "
                    f"{sorted(SAMPLE_DIRECTIONS)}, "
                    f"got {fib['sample_direction']!r}.")
            quats[rows] = fiber_texture(
                m, np.asarray(fib["crystal_axis"], dtype=np.float64),
                np.asarray(direction, dtype=np.float64),
                float(fib["spread_deg"]), rng, cell_matrix,
            )
            continue
        # Discrete Euler component (+ optional isotropic spread)
        q_base = fixed_orientation(
            1, {"euler_bunge_deg": comp["euler_bunge_deg"]}, cell_matrix)[0]
        spread = float(comp.get("spread_deg") or 0.0)
        if spread == 0.0:
            quats[rows] = np.tile(q_base, (m, 1))
            continue
        theta = _haar_gaussian_angles(m, np.radians(spread), rng)
        axes = _uniform_axes(m, rng)
        half = 0.5 * theta
        q_pert = np.concatenate(
            [np.cos(half)[:, None], np.sin(half)[:, None] * axes], axis=1)
        q = quat_mul_batch(q_pert, q_base[None, :])
        q[q[:, 0] < 0] *= -1.0
        quats[rows] = q
    if return_component_index:
        return quats, np.asarray(comp_idx, dtype=np.intp)
    return quats


def _uniform_axes(m: int, rng: np.random.Generator) -> np.ndarray:
    """(m, 3) unit vectors uniform on S² (normalized Gaussian triples)."""
    v = rng.standard_normal((m, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # An exactly-zero triple has probability 0; regenerate defensively.
    while np.any(norms < 1e-300):
        bad = norms[:, 0] < 1e-300
        v[bad] = rng.standard_normal((int(np.sum(bad)), 3))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _haar_gaussian_angles(
    m: int,
    sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample m rotation angles from the Haar-corrected Gaussian spread law
    p(θ) ∝ exp(−θ²/(2σ²)) · sin²(θ/2) on [0, π].

    This is the angle marginal of the isotropic SO(3) von Mises–Fisher
    small-angle surrogate: a Gaussian angular profile times the Haar
    measure factor sin²(θ/2).

    Exact rejection sampling with the Maxwell proposal θ = σ·‖N(0, I₃)‖
    (density ∝ θ² exp(−θ²/(2σ²))): accept with probability
    (sin(θ/2) / (θ/2))² ≤ 1 and reject θ > π.  The proposal carries the
    leading θ² Jacobian of the Haar factor, so acceptance → 1 as σ → 0
    (a plain half-normal proposal would collapse to acceptance ~σ²/4 —
    same target law, chosen for robustness at small spreads).
    """
    out = np.empty(m, dtype=np.float64)
    filled = 0
    proposals = 0
    max_proposals = max(10_000, RNG_MAX_ATTEMPTS_FACTOR * m)
    while filled < m:
        k = m - filled
        theta = sigma_rad * np.linalg.norm(rng.standard_normal((k, 3)),
                                           axis=1)
        u = rng.random(k)
        half = 0.5 * theta
        # u·(θ/2)² ≤ sin²(θ/2) accepts θ→0 with probability → 1 (0 ≤ 0).
        ok = (theta <= np.pi) & (u * half**2 <= np.sin(half) ** 2)
        n_ok = int(np.sum(ok))
        out[filled:filled + n_ok] = theta[ok]
        filled += n_ok
        proposals += k
        if proposals >= max_proposals and filled < m:
            raise ConfigError(
                f"_haar_gaussian_angles: rejection sampling failed to collect "
                f"{m} samples after {proposals} proposals "
                f"(spread_deg={np.degrees(sigma_rad):.1f}°). "
                "The spread is too large for this rejection sampler — "
                "reduce spread_deg below ~360°."
            )
    return out


def from_list(orientations: list, cell_matrix: np.ndarray | None = None) -> np.ndarray:
    """Build quaternion array from a list of per-grain orientation dicts (§6.10, §7).

    Each entry in the list may have any of the same keys as orientation.fixed.
    Returns (N, 4) float64.
    """
    quats = []
    for spec in orientations:
        arr = fixed_orientation(1, spec, cell_matrix)
        quats.append(arr[0])
    return np.array(quats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Quaternion that rotates unit vector v1 onto unit vector v2."""
    from grainsmith.orientation.quaternion import axis_angle_to_quat
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if abs(dot - 1.0) < PARALLEL_DOT_TOL:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if abs(dot + 1.0) < PARALLEL_DOT_TOL:
        perp = _any_perp(v1)
        return axis_angle_to_quat(perp, 180.0)
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)
    angle = np.degrees(np.arccos(dot))
    return axis_angle_to_quat(axis, angle)


def _any_perp(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to v."""
    if abs(v[0]) < 0.9:
        perp = np.cross(v, [1.0, 0.0, 0.0])
    else:
        perp = np.cross(v, [0.0, 1.0, 0.0])
    return perp / np.linalg.norm(perp)


def _perp_to(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random unit vector perpendicular to v."""
    # Generate random vector, project out v component
    r = rng.standard_normal(3)
    r -= np.dot(r, v) * v
    norm = np.linalg.norm(r)
    if norm < ZERO_VECTOR_TOL:
        return _any_perp(v)
    return r / norm
