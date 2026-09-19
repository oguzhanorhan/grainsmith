"""Misorientation and disorientation calculations (§6.10).

Convention:
    Δq_ij = q_i^{-1} ⊗ q_j  (crystal frame of grain i).
    Disorientation = symmetry-reduced minimum angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grainsmith.orientation.quaternion import (
    quat_inv,
    quat_mul,
    quat_mul_batch,
    quat_to_axis_angle,
)


def disorientation_angles(
    q_i: np.ndarray,
    q_j: np.ndarray,
    sym_quats: np.ndarray,
) -> np.ndarray:
    """Symmetry-reduced misorientation ANGLES for M grain pairs at once.

    Parameters
    ----------
    q_i, q_j : (M, 4) float64
        Scalar-first unit quaternions of the pair members.
    sym_quats : (Nsym, 4) float64
        Proper-rotation symmetry quaternions of the point group.

    Returns
    -------
    (M,) float64 disorientation angles in degrees.

    The rotation ANGLE is conjugation-invariant, so the two-sided orbit
    S_a ⊗ m ⊗ S_b collapses to the one-sided orbit {m ⊗ S} for angle
    purposes (S_a m S_b is conjugate to m S_b S_a; see
    analysis/boundaries.py), and the grain-exchange branch m⁻¹ shares the
    angle of m.  Hence Nsym candidates per pair instead of 2·Nsym² — this
    matters because this is the annealing inner loop, where only angles
    enter the MDF histogram.  Axes are NOT defined by this reduction; use
    :func:`disorientation` when the axis is needed.
    """
    q_i = np.asarray(q_i, dtype=np.float64).reshape(-1, 4)
    q_j = np.asarray(q_j, dtype=np.float64).reshape(-1, 4)
    sym = np.asarray(sym_quats, dtype=np.float64)
    # m = q_i^{-1} ⊗ q_j, vectorized (conjugate of unit quats = inverse)
    inv_i = q_i * np.array([1.0, -1.0, -1.0, -1.0])
    m = quat_mul_batch(inv_i, q_j)                          # (M, 4)
    cand = quat_mul_batch(m[:, None, :], sym[None, :, :])   # (M, Nsym, 4)
    w_max = np.max(np.abs(cand[..., 0]), axis=1)
    return 2.0 * np.degrees(np.arccos(np.clip(w_max, 0.0, 1.0)))


@dataclass
class Disorientation:
    """Result of disorientation computation between two grains."""
    angle_deg: float
    axis_crystal: np.ndarray  # (3,) unit vector in crystal frame of grain i
    axis_uvw: list[int] | None = None  # rationalized Miller indices (set by descriptors)
    axis_dev_deg: float = 0.0          # deviation of rationalized axis


def disorientation(
    q_i: np.ndarray,
    q_j: np.ndarray,
    sym_quats: np.ndarray,
) -> Disorientation:
    """
    Compute disorientation between grain i and grain j.

    Parameters
    ----------
    q_i, q_j : (4,) float64
        Unit quaternions (scalar-first) for grains i and j.
    sym_quats : (Nsym, 4) float64
        Proper-rotation symmetry quaternions of the point group (§6.3).

    Returns
    -------
    Disorientation
        Minimum-angle disorientation, axis in crystal frame of grain i.

    Algorithm (§6.10):
        m = q_i^{-1} ⊗ q_j
        Minimize angle over all S_a ⊗ m ⊗ S_b and m^{-1} (grain exchange).
        Fully vectorized over the 2 × Nsym × Nsym candidate array; ties
        resolve to the first candidate in (base, S_b, S_a) order — the
        same deterministic representative as the original scan.
    """
    sym = np.asarray(sym_quats, dtype=np.float64)
    m = quat_mul(quat_inv(q_i), q_j)
    bases = np.stack([m, quat_inv(m)])                     # (2, 4)

    # left[base, a] = S_a ⊗ base
    left = quat_mul_batch(sym[None, :, :], bases[:, None, :])
    # cand[base, b, a] = (S_a ⊗ base) ⊗ S_b
    cand = quat_mul_batch(left[:, None, :, :], sym[None, :, None, :])

    ws = np.clip(np.abs(cand[..., 0]), 0.0, 1.0)
    angles = 2.0 * np.degrees(np.arccos(ws))
    idx = np.unravel_index(int(np.argmin(angles)), angles.shape)
    best_q = cand[idx]

    axis, angle = quat_to_axis_angle(best_q)
    return Disorientation(angle_deg=angle, axis_crystal=axis)
