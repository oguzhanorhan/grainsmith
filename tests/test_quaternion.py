"""Tests for orientation/quaternion.py — §10 test_quaternion.py."""
from __future__ import annotations
import numpy as np
from scipy.stats import kstest
from grainsmith.orientation.quaternion import (
    quat_mul, quat_inv, quat_to_matrix, matrix_to_quat,
    axis_angle_to_quat, quat_to_axis_angle,
    to_scipy, from_scipy, quat_to_bunge,
)


def test_identity_matrix():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_matrix(q_id)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-14)


def test_quat_mul_identity():
    q = np.array([0.6, 0.4, 0.5, 0.5])
    q /= np.linalg.norm(q)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_mul(q_id, q), q, atol=1e-14)
    np.testing.assert_allclose(quat_mul(q, q_id), q, atol=1e-14)


def test_quat_mul_inverse():
    q = np.array([0.6, 0.4, 0.5, 0.5])
    q /= np.linalg.norm(q)
    prod = quat_mul(q, quat_inv(q))
    np.testing.assert_allclose(abs(prod[0]), 1.0, atol=1e-14)
    np.testing.assert_allclose(prod[1:], 0.0, atol=1e-14)


def test_rotation_z90():
    """90° rotation about z maps x→y."""
    q = axis_angle_to_quat([0, 0, 1], 90.0)
    R = quat_to_matrix(q)
    result = R @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)


def test_matrix_roundtrip():
    """matrix_to_quat(quat_to_matrix(q)) == q."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = from_scipy(__import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.random(random_state=rng))
        R = quat_to_matrix(q)
        q2 = matrix_to_quat(R)
        # q and -q represent the same rotation
        diff = min(np.linalg.norm(q - q2), np.linalg.norm(q + q2))
        assert diff < 1e-10, f"Round-trip failed: {q} → {q2}"


def test_scipy_bridge_roundtrip():
    """to_scipy(from_scipy(rot)) round-trip."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(99)
    for _ in range(10):
        rot = Rotation.random(random_state=rng)
        q = from_scipy(rot)
        rot2 = to_scipy(q)
        np.testing.assert_allclose(
            rot2.as_matrix(), rot.as_matrix(), atol=1e-12,
            err_msg="scipy bridge round-trip failed"
        )


def test_euler_identity():
    """Identity quaternion → Bunge (0, 0, 0)."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    phi1, Phi, phi2 = quat_to_bunge(q_id)
    assert Phi < 1e-10, f"Expected Phi=0 for identity, got {Phi}"


def test_euler_pure_z():
    """Pure rotation about lab z: phi1 changes, Phi=0."""
    q = axis_angle_to_quat([0, 0, 1], 30.0)
    phi1, Phi, phi2 = quat_to_bunge(q)
    assert Phi < 1e-8, f"Expected Phi≈0, got {Phi}"


def _bunge_matrix_textbook(phi1, Phi, phi2):
    """Canonical Bunge orientation matrix g (specimen->crystal), the
    Engler & Randle / MTEX / EBSD convention.  SCIENCE_AUDIT #1 ground truth."""
    p1, P, p2 = np.radians([phi1, Phi, phi2])
    c1, s1 = np.cos(p1), np.sin(p1)
    c2, s2 = np.cos(p2), np.sin(p2)
    cP, sP = np.cos(P), np.sin(P)
    return np.array([
        [c1 * c2 - s1 * s2 * cP,  s1 * c2 + c1 * s2 * cP,  s2 * sP],
        [-c1 * s2 - s1 * c2 * cP, -s1 * s2 + c1 * c2 * cP,  c2 * sP],
        [s1 * sP,                 -c1 * sP,                 cP],
    ])


def test_bunge_matches_standard_textbook_matrix():
    """SCIENCE_AUDIT #1 (the load-bearing pin): quat_to_bunge must produce the
    STANDARD Bunge angles, i.e. g = R(q).T must equal the canonical Bunge
    matrix Rz(phi2)Rx(Phi)Rz(phi1) for the *extracted* angles.  This is the
    external-truth pin a self-consistent round-trip alone cannot provide:
    extracting from g (instead of R) would return the inverse orientation
    (180-phi2, Phi, 180-phi1) and silently break MTEX/EBSD export."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20260613)))
    for _ in range(500):
        v = rng.normal(size=4)
        q = v / np.linalg.norm(v)
        q *= np.sign(q[0]) if q[0] != 0 else 1.0
        R = quat_to_matrix(q)
        phi1, Phi, phi2 = quat_to_bunge(q)
        g_standard = _bunge_matrix_textbook(phi1, Phi, phi2)
        np.testing.assert_allclose(
            g_standard, R.T, atol=1e-9,
            err_msg="exported Bunge angles do not match the standard "
                    "specimen->crystal Bunge matrix g = R(q).T",
        )


def test_bunge_ideal_brass_component():
    """SCIENCE_AUDIT #1: the Brass texture component {011}<2-11> exports as the
    published standard Bunge triple (35.26, 45, 0) (degrees)."""
    from grainsmith.orientation.samplers import _hkl_uvw_to_quat
    q = _hkl_uvw_to_quat(np.array([0.0, 1.0, 1.0]),
                         np.array([2.0, -1.0, 1.0]), np.eye(3))
    phi1, Phi, phi2 = quat_to_bunge(q)
    np.testing.assert_allclose([phi1, Phi, phi2], [35.2644, 45.0, 0.0], atol=1e-3)


def test_axis_angle_roundtrip():
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    angle = 73.5
    q = axis_angle_to_quat(axis, angle)
    ax2, ang2 = quat_to_axis_angle(q)
    assert abs(ang2 - angle) < 1e-8
    np.testing.assert_allclose(ax2, axis, atol=1e-8)


def test_uniform_sampler_angle_pdf():
    """Uniform SO(3) sampler: angle pdf ∝ (1 - cos θ) on [0, π]. KS test."""
    from grainsmith.orientation.samplers import random_uniform
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(12345)))
    N = 200_000  # N = 2e5, fixed seed
    quats = random_uniform(N, rng)
    # Extract angles
    ws = np.clip(np.abs(quats[:, 0]), 0.0, 1.0)
    angles = 2.0 * np.arccos(ws)  # in [0, π]

    # Theoretical CDF: F(θ) = (θ - sin θ) / π  for θ ∈ [0, π]
    def cdf(t):
        return (t - np.sin(t)) / np.pi

    stat, pval = kstest(angles, cdf)
    assert pval > 0.01, f"KS test failed: stat={stat:.4f}, p={pval:.4f} (expected uniform SO(3))"
