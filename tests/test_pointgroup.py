"""Tests for crystal/pointgroup.py — §10."""
import numpy as np
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.crystal.pointgroup import proper_rotation_quaternions


def _get_sym_quats(sg_number, a, b=None, c=None, alpha=90., beta=90., gamma=90.):
    if b is None:
        b = a
    if c is None:
        c = a
    A = cell_matrix(a, b, c, alpha, beta, gamma)
    hall = hall_from_international(sg_number)
    rots, _ = symmetry_ops(hall)
    return proper_rotation_quaternions(A, rots)


def test_cubic_m3m_24_proper_rotations():
    """m-3m point group has 24 proper rotations."""
    quats = _get_sym_quats(221, 3.0)
    assert len(quats) == 24


def test_hexagonal_6mmm_12_proper_rotations():
    """6/mmm point group has 12 proper rotations."""
    quats = _get_sym_quats(194, 3.0, c=5.0, gamma=120.0)
    assert len(quats) == 12


def test_quaternion_unit_norm():
    quats = _get_sym_quats(221, 3.0)
    norms = np.linalg.norm(quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_quaternion_closure():
    """Product of any two symmetry quaternions must be in the set (closure).
    
    Two unit quaternions represent the same rotation iff q1 = ±q2,
    so comparison uses min(‖q-prod‖, ‖q+prod‖).
    """
    from grainsmith.orientation.quaternion import quat_mul
    quats = _get_sym_quats(221, 3.0)
    for i in range(min(5, len(quats))):
        for j in range(min(5, len(quats))):
            prod = quat_mul(quats[i], quats[j])
            # q and -q represent the same rotation — compare both
            diffs = np.minimum(
                np.linalg.norm(quats - prod, axis=1),
                np.linalg.norm(quats + prod, axis=1),
            )
            assert np.min(diffs) < 1e-6, f"Product q[{i}]*q[{j}] not in group"


def test_scalar_w_non_negative():
    """All returned quaternions must have w >= 0 (canonical form)."""
    quats = _get_sym_quats(221, 3.0)
    assert np.all(quats[:, 0] >= -1e-12)
