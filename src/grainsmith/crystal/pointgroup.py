"""Extract proper-rotation quaternions of the point group from spglib. Used for disorientation."""
import numpy as np

from grainsmith.errors import CrystalError


def proper_rotation_quaternions(cell_matrix: np.ndarray,
                                rotations_frac: np.ndarray) -> np.ndarray:
    """
    Convert fractional rotation matrices (int, from spglib) to Cartesian unit quaternions.

    Steps (§6.3):
        R_cart = A @ R_frac @ A^{-1}
        Keep only det = +1 (proper rotations)
        Orthonormalize via SVD: U @ V^T
        Deduplicate (angular tolerance 1e-4 rad)
        Convert to quaternions, canonicalize w >= 0

    Returns (Nsym, 4) float64 array, scalar-first convention (w,x,y,z).
    """
    A = cell_matrix
    Ainv = np.linalg.inv(A)
    quats: list[np.ndarray] = []

    for Rf in rotations_frac:
        Rc = A @ Rf.astype(np.float64) @ Ainv
        # Keep only proper rotations (det = +1)
        det = np.linalg.det(Rc)
        if abs(det - 1.0) > 0.1:
            continue
        # Orthonormalize via SVD
        U, _, Vt = np.linalg.svd(Rc)
        Rc_ortho = U @ Vt
        if np.linalg.det(Rc_ortho) < 0:
            Vt[-1] *= -1
            Rc_ortho = U @ Vt
        q = _matrix_to_quat(Rc_ortho)
        quats.append(q)

    if not quats:
        raise CrystalError("No proper rotations found in provided symmetry operations.")

    # Deduplicate
    unique = _dedupe_quats(quats)
    return np.array(unique, dtype=np.float64)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to scalar-first quaternion (w,x,y,z), w >= 0."""
    # Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return _canonical_quat(q)


def _canonical_quat(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Sign-canonical form: flip q so its first significant component is
    positive (w first; for 180° rotations w ≈ 0 falls through to x, y, z).

    Invariants downstream code relies on:
    - w ≥ −eps always (axis-angle extraction stays in [0°, 180°]);
    - q and −q map to the same representative.
    The w ≈ 0 edge is safe: a true 180° rotation gives |w| ≲ 1e−15 (machine
    noise), three orders below eps = 1e−9, while genuinely nonzero w of any
    crystallographic rotation is ≥ cos(90°/2)·noise-free ≈ 0.5 — eps never
    misclassifies; for a unit quaternion all four components cannot be
    below eps simultaneously.
    """
    for i in range(4):
        if q[i] < -eps:
            return -q.copy()
        if q[i] > eps:
            return q.copy()
    return q.copy()


def _dedupe_quats(quats: list[np.ndarray], tol: float = 1e-4) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for q in quats:
        is_dup = False
        for uq in unique:
            # quaternions q and -q represent the same rotation
            diff1 = float(np.linalg.norm(q - uq))
            diff2 = float(np.linalg.norm(q + uq))
            if min(diff1, diff2) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(q)
    return unique
