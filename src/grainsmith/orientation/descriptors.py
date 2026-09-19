"""Crystallographic descriptors: Bunge Euler, Miller rationalization (§6.10).

Miller rationalization algorithm:
    Normalize so max|component| = 1.
    Scan denominators 1..max_index.
    Score by angular deviation from exact.
    Tie-break by smallest ‖candidate‖∞.
    Reduce by gcd. Report deviation (never present as exact).
"""
from __future__ import annotations

import math

import numpy as np

from grainsmith.constants import ZERO_VECTOR_TOL
from grainsmith.orientation.quaternion import quat_to_bunge, quat_to_matrix


def rationalize(
    v: np.ndarray,
    max_index: int = 12,
) -> tuple[list[int], float]:
    """
    Rationalize a real-valued direction into integer Miller indices.

    Parameters
    ----------
    v : (3,) float
        Direction vector (need not be unit).
    max_index : int
        Maximum absolute integer index to scan.

    Returns
    -------
    (ints, deviation_deg)
        ints : [h, k, l] integer Miller indices (reduced by gcd).
        deviation_deg : angular deviation of the rationalized direction from v.
    """
    v = np.asarray(v, dtype=np.float64)
    m = float(np.max(np.abs(v)))
    if m < 1e-15:
        return [0, 0, 0], 0.0
    v = v / m
    # Unit-L2 copy for angle computation; L∞-normalised v drives the scan so
    # that denom == max_index reaches the high-index corner exactly.
    v_unit = v / float(np.linalg.norm(v))

    # Vectorise the denominator scan: build all max_index candidates at
    # once instead of looping with ~6 numpy dispatches per denominator.  The
    # reductions are chosen to be BIT-IDENTICAL to the old per-row scalar ones
    # (verified): np.linalg.norm(...,axis=1) == per-row norm, and
    # (cand·v_unit).sum(axis=1) == per-row np.dot — so the deviation float that
    # lands in the golden-pinned CSVs is unchanged.  The tie-break stays a cheap
    # scalar loop over the ≤max_index candidates to preserve exact selection.
    denoms = np.arange(1, max_index + 1, dtype=np.float64)          # (D,)
    cand_i = np.round(v[None, :] * denoms[:, None]).astype(int)     # (D, 3)
    cand_f = cand_i.astype(np.float64)                             # (D, 3)
    cand_norm = np.linalg.norm(cand_f, axis=1)                     # (D,)
    valid = (~np.all(cand_i == 0, axis=1)) & (cand_norm >= ZERO_VECTOR_TOL)
    dots = np.abs((cand_f * v_unit[None, :]).sum(axis=1))           # (D,)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dots / cand_norm, 0.0, 1.0)
    dev = np.degrees(np.arccos(cos))                              # (D,) nan if invalid
    maxabs = np.max(np.abs(cand_i), axis=1)                        # (D,)

    best_ints: list[int] = [1, 0, 0]
    best_dev = np.inf
    best_maxabs = 1
    for d in range(max_index):
        if not valid[d]:
            continue
        dv = float(dev[d])
        if (dv < best_dev or
                (abs(dv - best_dev) < 1e-8 and int(maxabs[d]) < best_maxabs)):
            best_dev = dv
            best_ints = cand_i[d].tolist()
            best_maxabs = int(maxabs[d])

    # Reduce by gcd
    g = math.gcd(*[abs(x) for x in best_ints if x != 0])
    if g > 1:
        best_ints = [x // g for x in best_ints]

    return best_ints, best_dev


def plane_miller(n_cart: np.ndarray, A: np.ndarray, max_index: int = 12) -> tuple[list[int], float]:
    """
    Convert Cartesian boundary-plane normal to Miller (hkl).

    m ∝ A^T @ n_cart  (reciprocal space), then rationalize.
    """
    m = A.T @ n_cart
    return rationalize(m, max_index)


def direction_miller(d_cart: np.ndarray, A: np.ndarray, max_index: int = 12) -> tuple[list[int], float]:
    """
    Convert Cartesian direction to Miller [uvw].

    u ∝ A^{-1} @ d_cart, then rationalize.
    """
    Ainv = np.linalg.inv(A)
    u = Ainv @ d_cart
    return rationalize(u, max_index)


def grain_descriptors(q: np.ndarray, A: np.ndarray, max_index: int = 12) -> dict:
    """
    Compute per-grain crystallographic descriptors (§6.10, grains.csv columns).

    Returns dict with:
        q_w, q_x, q_y, q_z, euler_phi1_deg, euler_Phi_deg, euler_phi2_deg,
        axis_x, axis_y, axis_z, angle_deg,
        z_plane_hkl (str), z_plane_dev_deg,
        x_dir_uvw (str), x_dir_dev_deg
    """
    from grainsmith.orientation.quaternion import quat_to_axis_angle
    R = quat_to_matrix(q)
    phi1, Phi, phi2 = quat_to_bunge(q)
    ax, ang = quat_to_axis_angle(q)

    # Crystal plane facing lab +z: lab z in crystal frame = R.T @ [0,0,1] = R[2, :]
    z_lab = np.array([0.0, 0.0, 1.0])
    n_cryst = R.T @ z_lab          # plane normal in crystal Cartesian
    hkl, z_dev = plane_miller(n_cryst, A, max_index)

    # Crystal direction along lab +x
    x_lab = np.array([1.0, 0.0, 0.0])
    d_cryst = R.T @ x_lab
    uvw, x_dev = direction_miller(d_cryst, A, max_index)

    return {
        "q_w": float(q[0]), "q_x": float(q[1]), "q_y": float(q[2]), "q_z": float(q[3]),
        "euler_phi1_deg": phi1, "euler_Phi_deg": Phi, "euler_phi2_deg": phi2,
        "axis_x": float(ax[0]), "axis_y": float(ax[1]), "axis_z": float(ax[2]),
        "angle_deg": float(ang),
        "z_plane_hkl": "(" + " ".join(str(v) for v in hkl) + ")",
        "z_plane_dev_deg": z_dev,
        "x_dir_uvw": "[" + " ".join(str(v) for v in uvw) + "]",
        "x_dir_dev_deg": x_dev,
    }
