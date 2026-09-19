"""Tests for orientation/samplers.py fixed-orientation forms (§6.10).

Pins the hkl_uvw rotation construction: the crystal plane normal must land
on lab +z and the crystal direction on lab +x under the ACTIVE rotation
v_lab = R(q) @ v_crystal.  The legacy-class-C4 transpose bug (column_stack
instead of rows) produced R⁻¹ here — AUDIT C2 regression.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.crystal.cell import cell_matrix
from grainsmith.errors import ConfigError
from grainsmith.orientation.quaternion import quat_to_bunge, quat_to_matrix
from grainsmith.orientation.samplers import _haar_gaussian_angles, fixed_orientation


def test_hkl_uvw_alignment_cubic():
    """(111) ∥ lab z and [1-10] ∥ lab x for a cubic cell."""
    A = cell_matrix(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    spec = {"hkl_uvw": {"plane": [1, 1, 1], "direction": [1, -1, 0]}}
    q = fixed_orientation(1, spec, A)[0]
    R = quat_to_matrix(q)

    n_c = np.linalg.inv(A).T @ np.array([1.0, 1.0, 1.0])
    n_c /= np.linalg.norm(n_c)
    d_c = A @ np.array([1.0, -1.0, 0.0])
    d_c /= np.linalg.norm(d_c)

    np.testing.assert_allclose(R @ n_c, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R @ d_c, [1.0, 0.0, 0.0], atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12  # proper rotation


def test_hkl_uvw_alignment_hexagonal():
    """(0001) ∥ z with [100] ∥ x for an HCP cell (non-cubic metric path)."""
    A = cell_matrix(2.95, 2.95, 4.68, 90.0, 90.0, 120.0)
    spec = {"hkl_uvw": {"plane": [0, 0, 1], "direction": [1, 0, 0]}}
    q = fixed_orientation(1, spec, A)[0]
    R = quat_to_matrix(q)

    n_c = np.linalg.inv(A).T @ np.array([0.0, 0.0, 1.0])
    n_c /= np.linalg.norm(n_c)
    d_c = A @ np.array([1.0, 0.0, 0.0])
    d_c /= np.linalg.norm(d_c)

    np.testing.assert_allclose(R @ n_c, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R @ d_c, [1.0, 0.0, 0.0], atol=1e-12)


def test_hkl_uvw_nonorthogonal_raises():
    """(111) with [100] is not an orthogonal pair → ConfigError with the
    offending dot product (§6.10)."""
    A = cell_matrix(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    spec = {"hkl_uvw": {"plane": [1, 1, 1], "direction": [1, 0, 0]}}
    with pytest.raises(ConfigError):
        fixed_orientation(1, spec, A)


def test_euler_bunge_fixed_round_trip():
    """fixed_orientation(euler_bunge_deg) → quat_to_bunge round-trips."""
    spec = {"euler_bunge_deg": [10.0, 20.0, 30.0]}
    q = fixed_orientation(1, spec)[0]
    phi1, Phi, phi2 = quat_to_bunge(q)
    np.testing.assert_allclose([phi1, Phi, phi2], [10.0, 20.0, 30.0],
                               atol=1e-9)


# ---------------------------------------------------------------------------
# Fix #11: _haar_gaussian_angles raises ConfigError on pathological spread
# ---------------------------------------------------------------------------


def test_haar_gaussian_angles_normal_spread():
    """Sanity check: small spread (5°) returns m valid angles in [0, π]."""
    rng = np.random.Generator(np.random.PCG64(42))
    sigma_rad = np.radians(5.0)
    angles = _haar_gaussian_angles(10, sigma_rad, rng)
    assert angles.shape == (10,)
    assert np.all(angles >= 0.0) and np.all(angles <= np.pi)


def test_haar_gaussian_angles_pathological_spread_raises():
    """A pathologically large spread collapses acceptance probability to
    essentially zero; the bounded guard must raise ConfigError rather than
    hanging.

    sigma = 36000° → sigma_rad ≈ 628 rad → need ||N(0,I3)|| ≤ π/628 ≈ 0.005
    to even be ≤ π.  P(Maxwell ≤ 0.005) ≈ 10^-8; the budget of 10 000
    proposals will never find a sample.  We use m=1 to minimise the budget.
    """
    rng = np.random.Generator(np.random.PCG64(99))
    sigma_rad = np.radians(36_000.0)
    with pytest.raises(ConfigError, match="spread"):
        _haar_gaussian_angles(1, sigma_rad, rng)
