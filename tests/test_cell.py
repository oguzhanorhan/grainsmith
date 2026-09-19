"""Tests for crystal/cell.py — §10 test_cell.py. Regression for defect O1."""
import numpy as np
import pytest
from grainsmith.crystal.cell import cell_matrix, validate_cellpar
from grainsmith.errors import ConfigError


def test_cubic_identity():
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    assert A.dtype == np.float64
    np.testing.assert_allclose(A, np.diag([3.0, 3.0, 3.0]), atol=1e-12)


def test_cubic_volume():
    a = 3.615
    A = cell_matrix(a, a, a, 90.0, 90.0, 90.0)
    assert abs(np.linalg.det(A) - a**3) < 1e-10


def test_hexagonal():
    # HCP Ti: a=2.9508, c=4.6855, alpha=beta=90, gamma=120
    A = cell_matrix(2.9508, 2.9508, 4.6855, 90.0, 90.0, 120.0)
    assert A.dtype == np.float64
    # Volume = a^2 * c * sin(60°)
    expected_vol = 2.9508**2 * 4.6855 * np.sin(np.radians(60.0))
    assert abs(np.linalg.det(A) - expected_vol) < 1e-4


def test_monoclinic_O1_regression():
    """O1 regression: monoclinic beta=100° must give a valid non-diagonal matrix."""
    A = cell_matrix(5.0, 4.0, 6.0, 90.0, 100.0, 90.0)
    assert A.dtype == np.float64
    # a3 must have an x component due to cos(beta)
    assert abs(A[0, 2]) > 0.1  # non-zero x-component of c vector
    # Volume: a*b*c*sqrt(1-cos^2(beta))
    expected_vol = 5.0 * 4.0 * 6.0 * np.sin(np.radians(100.0))
    assert abs(np.linalg.det(A) - expected_vol) < 1e-4


def test_rhombohedral_O1_regression():
    """O1 regression: rhombohedral (trigonal) cell must be valid."""
    # Rhombohedral: a=b=c, alpha=beta=gamma=70°
    A = cell_matrix(4.0, 4.0, 4.0, 70.0, 70.0, 70.0)
    assert A.dtype == np.float64
    assert np.linalg.det(A) > 0


def test_triclinic():
    A = cell_matrix(5.0, 6.0, 7.0, 80.0, 95.0, 110.0)
    assert A.dtype == np.float64
    assert np.linalg.det(A) > 0


def test_unphysical_params():
    with pytest.raises(ConfigError):
        cell_matrix(1.0, 1.0, 1.0, 90.0, 90.0, 0.0)  # sin(gamma)=0


def test_validate_cellpar_cubic():
    cp = validate_cellpar("cubic", {"a": 3.52})
    assert cp.b == cp.a and cp.c == cp.a
    assert cp.alpha == 90.0 and cp.beta == 90.0 and cp.gamma == 90.0


def test_validate_cellpar_cubic_overspec():
    with pytest.raises(ConfigError, match="Unexpected"):
        validate_cellpar("cubic", {"a": 3.52, "b": 3.52})


def test_validate_cellpar_cubic_underspec():
    with pytest.raises(ConfigError, match="Missing"):
        validate_cellpar("cubic", {})


def test_validate_cellpar_monoclinic():
    cp = validate_cellpar("monoclinic", {"a": 5.0, "b": 4.0, "c": 6.0, "beta": 100.0})
    assert cp.alpha == 90.0 and cp.gamma == 90.0
    assert cp.beta == 100.0


def test_validate_cellpar_hexagonal():
    cp = validate_cellpar("hexagonal", {"a": 2.95, "c": 4.68})
    assert cp.gamma == 120.0 and cp.b == cp.a


def test_validate_cellpar_rhombohedral_R_setting():
    """Rhombohedral (R) axis setting: free (a, alpha); b=c=a, beta=gamma=alpha."""
    cp = validate_cellpar("rhombohedral", {"a": 5.0, "alpha": 55.0})
    assert (cp.a, cp.b, cp.c) == (5.0, 5.0, 5.0)
    assert (cp.alpha, cp.beta, cp.gamma) == (55.0, 55.0, 55.0)
    A = cell_matrix(cp.a, cp.b, cp.c, cp.alpha, cp.beta, cp.gamma)
    # all three lattice vectors have equal length a and pairwise angle alpha
    lengths = np.linalg.norm(A, axis=0)
    np.testing.assert_allclose(lengths, 5.0, atol=1e-12)
    for i in range(3):
        j = (i + 1) % 3
        cos_ij = A[:, i] @ A[:, j] / (lengths[i] * lengths[j])
        assert abs(cos_ij - np.cos(np.radians(55.0))) < 1e-12


def test_validate_cellpar_rhombohedral_rejects_hexagonal_params():
    with pytest.raises(ConfigError):
        validate_cellpar("rhombohedral", {"a": 5.0, "c": 14.0})
