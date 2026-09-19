"""Quaternion arithmetic — scalar-first (w, x, y, z) convention.

Convention (§1):
    q = (w, x, y, z), unit norm.
    Active rotation: v_lab = R(q) @ v_crystal.
    R(q) per standard formula (see quat_to_matrix).

scipy bridge:
    scipy.spatial.transform.Rotation stores scalar-LAST (x,y,z,w).
    Use only to_scipy(q) / from_scipy(rot) — never .as_quat()/.from_quat() directly.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from grainsmith.constants import IDENTITY_ANGLE_TOL, ZERO_VECTOR_TOL
from grainsmith.errors import ConfigError

# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first unit quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_mul_batch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product with NumPy broadcasting over leading axes.

    ``p`` and ``q`` are (..., 4) scalar-first quaternion arrays with
    mutually broadcastable leading shapes; returns the broadcast (..., 4)
    product.  Identical algebra to :func:`quat_mul` (which it reduces to
    for two single quaternions) — used by the vectorized disorientation
    and CSL class-deviation loops (§6.10 performance mandate).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    pw, pv = p[..., :1], p[..., 1:]
    qw, qv = q[..., :1], q[..., 1:]
    w = pw * qw - np.sum(pv * qv, axis=-1, keepdims=True)
    v = pw * qv + qw * pv + np.cross(pv, qv)
    return np.concatenate([w, v], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a unit quaternion: (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (= conjugate for unit quats)."""
    return quat_conj(q)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return unit-normalized quaternion."""
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Rotation matrix conversion
# ---------------------------------------------------------------------------

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert scalar-first unit quaternion to 3x3 rotation matrix.
    Active rotation convention: v_lab = R @ v_crystal.

    R = I + 2w*[v]× + 2*[v]×²    where [v]× is the skew-symmetric matrix of (x,y,z).
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to scalar-first quaternion (w>=0 canonical form).
    Uses Shepperd's method."""
    from grainsmith.crystal.pointgroup import _canonical_quat, _matrix_to_quat
    return _canonical_quat(_matrix_to_quat(R))


# ---------------------------------------------------------------------------
# Axis-angle conversion
# ---------------------------------------------------------------------------

def axis_angle_to_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Convert axis (3,) + angle (degrees) to scalar-first unit quaternion.

    Raises ConfigError for a zero (or numerically-zero) axis — normalizing it
    would yield a NaN quaternion that silently poisons all downstream
    orientation analysis."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < ZERO_VECTOR_TOL:
        raise ConfigError(
            f"rotation axis must be a non-zero vector, got {axis.tolist()}")
    axis = axis / norm
    half = np.radians(angle_deg) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (axis (3,), angle_deg) from scalar-first unit quaternion.
    Returns (z-hat, 0.0) for the identity."""
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * np.degrees(np.arccos(abs(w)))
    if angle < IDENTITY_ANGLE_TOL:
        return np.array([0.0, 0.0, 1.0]), 0.0
    s = np.sqrt(1.0 - w*w)
    axis = q[1:] / s
    if q[0] < 0:
        axis = -axis
    return axis / np.linalg.norm(axis), angle


# ---------------------------------------------------------------------------
# scipy bridge — ONLY place where .as_quat()/.from_quat() are allowed
# ---------------------------------------------------------------------------

def to_scipy(q: np.ndarray) -> _Rotation:
    """Convert scalar-first (w,x,y,z) to scipy Rotation (which stores scalar-last)."""
    # scipy scalar-last order: (x, y, z, w)
    return _Rotation.from_quat([q[1], q[2], q[3], q[0]])


def from_scipy(rot: _Rotation) -> np.ndarray:
    """Convert scipy Rotation to scalar-first (w,x,y,z) unit quaternion(s).

    Accepts a single rotation (returns (4,)) or a rotation stack
    (returns (N, 4)) — the batch path keeps samplers free of per-element
    Python loops.  Output is sign-canonicalized to w ≥ 0.
    """
    xyzw = np.atleast_2d(rot.as_quat())  # scalar-last: (x, y, z, w)
    q = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1).astype(np.float64)
    q[q[:, 0] < 0] *= -1.0
    return q[0] if rot.single else q


# ---------------------------------------------------------------------------
# Bunge Euler extraction (§1 convention)
# ---------------------------------------------------------------------------

def quat_to_bunge(q: np.ndarray) -> tuple[float, float, float]:
    """
    Extract Bunge Euler angles (phi1, Phi, phi2) in degrees from a unit quaternion.

    Convention (§1, standard Bunge / MTEX / EBSD):
        The Bunge orientation matrix g = R(q).T is the specimen→crystal
        coordinate transform, whose canonical factorization is
        g = Rz(phi2) Rx(Phi) Rz(phi1) = intrinsic-ZXZ(phi1,Phi,phi2).T.
        Therefore intrinsic-ZXZ(phi1,Phi,phi2) = g.T = R(q), so the standard
        Bunge angles are the intrinsic-ZXZ Euler angles of R(q) ITSELF — NOT
        of g = R.T.  (Extracting from g instead yields the *inverse*
        orientation, (phi1,Phi,phi2) -> (180-phi2,Phi,180-phi1); this was a
        bug that has since been fixed.)  The angles produced here match
        MTEX ``loadOrientation_generic(...,'Bunge')`` and every EBSD device /
        textbook — verified by an MTEX-convention round-trip.

    Degenerate case: for Phi ≈ 0 or 180° only phi1 ± phi2 is defined; the
    convention here (scipy's) is the deterministic choice phi2 := 0, with
    phi1 carrying the whole z-rotation.  scipy's gimbal-lock UserWarning is
    suppressed because this resolution is intentional and documented.
    """
    import warnings

    R = quat_to_matrix(q)
    rot_R = _Rotation.from_matrix(R)
    # Standard Bunge angles = intrinsic-ZXZ Euler angles of R(q) (see above).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Gimbal lock", UserWarning)
        angles = rot_R.as_euler('ZXZ', degrees=True)
    phi1, Phi, phi2 = float(angles[0]), float(angles[1]), float(angles[2])
    # Bring to [0, 360) x [0, 180] x [0, 360)
    phi1 = phi1 % 360.0
    Phi = float(np.clip(Phi, 0.0, 180.0))
    phi2 = phi2 % 360.0
    return phi1, Phi, phi2
