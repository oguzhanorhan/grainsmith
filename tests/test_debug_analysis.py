"""Scientific debug wave — analysis + orientation tracks.

Numerically pins the claims audited this round:

- quaternion <-> matrix round trip and Hamilton-product composition
  (q1 * q2 == R(q1) @ R(q2), the active-rotation, scalar-first convention
  documented in orientation/quaternion.py);
- ``random_uniform`` is Haar-uniform on SO(3): the rotation-angle marginal
  matches p(theta) ~ (1 - cos theta) on [0, pi] (KS test) and the axis is
  uniform on S^2 (KS test on the polar cosine);
- the Mackenzie (1958) disorientation-angle law for the cubic point group:
  mean ~= 40.73 deg, max ~= 62.8 deg, reproduced from
  ``orientation.misorientation.disorientation_angles`` on independent
  Haar-random pairs (not read off ``mdf.reference_angles``, which is
  built from the same primitive and would make the check circular);
- the batched one-sided disorientation reduction
  (``disorientation_angles``) agrees with the full two-sided
  ``disorientation()`` scan to float64 precision;
- the voxel staircase-area overcount factor ||n||_1, whose documented
  maximum is sqrt(3) at the cubic body diagonal (1,1,1)/sqrt(3) —
  analysis/statistics.py's sphericity/S_V/L_V voxel-estimator docstrings.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.transform import Rotation as _Rotation

from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.orientation.misorientation import disorientation, disorientation_angles
from grainsmith.orientation.quaternion import (
    axis_angle_to_quat,
    from_scipy,
    matrix_to_quat,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_to_axis_angle,
    quat_to_matrix,
    to_scipy,
)
from grainsmith.orientation.samplers import random_uniform


def _cubic_sym_quats() -> np.ndarray:
    A = cell_matrix(3.615, 3.615, 3.615, 90.0, 90.0, 90.0)
    rots, _ = symmetry_ops(hall_from_international(221))
    return proper_rotation_quaternions(A, rots)


# --------------------------------------------------------------- quaternion


def test_quat_matrix_roundtrip():
    """matrix_to_quat(quat_to_matrix(q)) reproduces R to float64 precision
    for 2000 independent Haar-random rotations."""
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(2000):
        Rm = _Rotation.random(random_state=rng).as_matrix()
        q = matrix_to_quat(Rm)
        max_err = max(max_err, float(np.max(np.abs(Rm - quat_to_matrix(q)))))
    assert max_err < 1e-12, f"matrix round-trip error {max_err:.3e}"


def test_quat_mul_matches_matrix_composition():
    """Hamilton product q1*q2 represents the SAME rotation as R1 @ R2 —
    the active, scalar-first convention documented in quaternion.py's
    module docstring (v_lab = R(q) @ v_crystal)."""
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(2000):
        q1 = quat_normalize(rng.standard_normal(4))
        q2 = quat_normalize(rng.standard_normal(4))
        R12 = quat_to_matrix(quat_mul(q1, q2))
        R1R2 = quat_to_matrix(q1) @ quat_to_matrix(q2)
        max_err = max(max_err, float(np.max(np.abs(R12 - R1R2))))
    assert max_err < 1e-12, f"composition error {max_err:.3e}"


def test_quat_inv_is_transpose():
    """quat_inv(q) (= conjugate for a unit quaternion) maps to R.T."""
    rng = np.random.default_rng(2)
    for _ in range(500):
        q = quat_normalize(rng.standard_normal(4))
        Rm = quat_to_matrix(q)
        assert np.allclose(quat_to_matrix(quat_inv(q)), Rm.T, atol=1e-12)


def test_axis_angle_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(1000):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.5, 179.5)  # avoid the antipodal/identity edges
        q = axis_angle_to_quat(axis, angle)
        axis2, angle2 = quat_to_axis_angle(q)
        d = min(np.linalg.norm(axis2 - axis), np.linalg.norm(axis2 + axis))
        assert d < 1e-9
        assert abs(angle2 - angle) < 1e-8


def test_scipy_bridge_roundtrip():
    """to_scipy/from_scipy is the ONLY place scalar-last <-> scalar-first
    reindexing happens; round trip must be exact."""
    rng = np.random.default_rng(4)
    for _ in range(1000):
        q = quat_normalize(rng.standard_normal(4))
        if q[0] < 0:
            q = -q
        q2 = from_scipy(to_scipy(q))
        assert np.allclose(q, q2, atol=1e-14)


def test_axis_angle_zero_vector_raises():
    from grainsmith.errors import ConfigError

    with pytest.raises(ConfigError):
        axis_angle_to_quat(np.array([0.0, 0.0, 0.0]), 45.0)


# ------------------------------------------------------------ Haar-uniformity


def test_random_uniform_angle_matches_haar_law():
    """The rotation-angle marginal of Haar measure on SO(3) is
    p(theta) ~ (1 - cos theta), theta in [0, pi]; CDF F(theta) =
    (theta - sin theta)/pi.  KS test against 2*10^5 samples."""
    rng = np.random.default_rng(42)
    q = random_uniform(200_000, rng)
    theta = 2.0 * np.arccos(np.clip(np.abs(q[:, 0]), 0.0, 1.0))

    def cdf_haar(t):
        return (t - np.sin(t)) / np.pi

    ks = stats.kstest(theta, cdf_haar)
    assert ks.pvalue > 0.01, f"angle marginal fails Haar KS test: {ks}"


def test_random_uniform_axis_is_isotropic():
    """The rotation axis of a Haar-random orientation is uniform on S^2:
    its polar cosine (w.r.t. an arbitrary fixed pole) is Uniform[-1, 1]."""
    rng = np.random.default_rng(43)
    q = random_uniform(200_000, rng)
    axis = q[:, 1:]
    norms = np.linalg.norm(axis, axis=1)
    mask = norms > 1e-9
    axis_unit = axis[mask] / norms[mask, None]
    cos_polar = axis_unit[:, 2]
    ks = stats.kstest(cos_polar, "uniform", args=(-1.0, 2.0))
    assert ks.pvalue > 0.01, f"axis isotropy fails Haar KS test: {ks}"


def test_random_uniform_deterministic_seed():
    q1 = random_uniform(50, np.random.default_rng(7))
    q2 = random_uniform(50, np.random.default_rng(7))
    assert np.array_equal(q1, q2)


# -------------------------------------------------------- Mackenzie (cubic)


def test_cubic_disorientation_matches_mackenzie_1958():
    """Random-pair cubic disorientation reproduces the Mackenzie (1958)
    law: mean ~= 40.73 deg, max <= 62.8 deg (m-3m fundamental-zone bound),
    computed independently of orientation/mdf.py's cached reference
    curve — straight from disorientation_angles on fresh Haar samples."""
    sym = _cubic_sym_quats()
    assert len(sym) == 24
    rng = np.random.default_rng(123)
    n = 300_000
    qi = random_uniform(n, rng)
    qj = random_uniform(n, rng)
    angles = disorientation_angles(qi, qj, sym)

    assert angles.min() >= 0.0
    assert angles.max() <= 62.85, f"exceeds cubic fundamental-zone bound: {angles.max()}"
    assert abs(angles.mean() - 40.73) < 0.15, f"mean {angles.mean():.3f} != 40.73 deg"


def test_disorientation_batched_matches_full_scan():
    """disorientation_angles (one-sided orbit, vectorized) must agree with
    the full two-sided disorientation() scan to float64 precision — the
    module docstring's claim that the two reductions coincide for the
    ANGLE (not axis)."""
    sym = _cubic_sym_quats()
    rng = np.random.default_rng(55)
    m = 50
    qi = random_uniform(m, rng)
    qj = random_uniform(m, rng)
    fast = disorientation_angles(qi, qj, sym)
    full = np.array([disorientation(qi[k], qj[k], sym).angle_deg for k in range(m)])
    assert np.max(np.abs(fast - full)) < 1e-9


def test_disorientation_identity_pair_is_zero():
    sym = _cubic_sym_quats()
    q = axis_angle_to_quat([1, 0, 0], 45.0)
    d = disorientation(q, q, sym)
    assert d.angle_deg < 1e-8


def test_disorientation_exchange_symmetry():
    sym = _cubic_sym_quats()
    rng = np.random.default_rng(9)
    q = random_uniform(2, rng)
    d_ij = disorientation(q[0], q[1], sym)
    d_ji = disorientation(q[1], q[0], sym)
    assert abs(d_ij.angle_deg - d_ji.angle_deg) < 1e-6


# --------------------------------------------------------- voxel area bias


def test_l1_norm_max_at_body_diagonal_is_sqrt3():
    """The voxel staircase-area overcount factor ||n||_1 for a unit
    normal n attains its documented maximum sqrt(3) at the cubic body
    diagonal (1,1,1)/sqrt(3), not at a single-axis 45 deg tilt (SCIENCE_
    AUDIT #12) — 200,000 random unit directions (Gaussian-normalized)
    plus the exact stationary point."""
    rng = np.random.default_rng(11)
    n = rng.standard_normal((200_000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    l1 = np.abs(n).sum(axis=1)
    assert l1.max() < np.sqrt(3.0) + 1e-9
    # the exact stationary point reaches it
    diag = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert abs(np.abs(diag).sum() - np.sqrt(3.0)) < 1e-12
    # a single-axis 45 deg tilt, e.g. (110)/sqrt(2), reaches only sqrt(2)
    tilt45 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    assert abs(np.abs(tilt45).sum() - np.sqrt(2.0)) < 1e-12
