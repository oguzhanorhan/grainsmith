"""Tests for orientation/misorientation.py — §10 test_misorientation.py."""
from __future__ import annotations
import numpy as np
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.orientation.quaternion import axis_angle_to_quat
from grainsmith.orientation.misorientation import disorientation


def _sym_quats_cubic():
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    hall = hall_from_international(221)
    rots, _ = symmetry_ops(hall)
    return proper_rotation_quaternions(A, rots)


def _sym_quats_hex():
    A = cell_matrix(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
    hall = hall_from_international(194)
    rots, _ = symmetry_ops(hall)
    return proper_rotation_quaternions(A, rots)


def test_identical_grains_zero():
    """Disorientation of identical grains = 0°."""
    sym = _sym_quats_cubic()
    q = axis_angle_to_quat([1, 0, 0], 45.0)
    d = disorientation(q, q, sym)
    assert d.angle_deg < 1e-8, f"Expected 0°, got {d.angle_deg}"


def test_sigma3_cubic():
    """Σ3 = 60° ⟨111⟩ for cubic (atol 0.05°)."""
    sym = _sym_quats_cubic()
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    q_i = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    q_j = axis_angle_to_quat(axis, 60.0)
    d = disorientation(q_i, q_j, sym)
    assert abs(d.angle_deg - 60.0) < 0.05, f"Σ3: expected 60°, got {d.angle_deg:.4f}°"


def test_sigma5_cubic():
    """Σ5 = 36.87° ⟨100⟩ for cubic (atol 0.05°)."""
    sym = _sym_quats_cubic()
    axis = np.array([1.0, 0.0, 0.0])
    q_i = np.array([1.0, 0.0, 0.0, 0.0])
    q_j = axis_angle_to_quat(axis, 36.87)
    d = disorientation(q_i, q_j, sym)
    assert abs(d.angle_deg - 36.87) < 0.05, f"Σ5: expected 36.87°, got {d.angle_deg:.4f}°"


def test_exchange_symmetry():
    """Disorientation(i,j) == Disorientation(j,i) (exchange symmetry)."""
    sym = _sym_quats_cubic()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(7)))
    from grainsmith.orientation.samplers import random_uniform
    quats = random_uniform(2, rng)
    q_i, q_j = quats[0], quats[1]
    d_ij = disorientation(q_i, q_j, sym)
    d_ji = disorientation(q_j, q_i, sym)
    assert abs(d_ij.angle_deg - d_ji.angle_deg) < 1e-6, (
        f"Exchange symmetry violated: {d_ij.angle_deg:.6f} vs {d_ji.angle_deg:.6f}"
    )


def test_hexagonal_max_angle():
    """Hexagonal disorientation angle ≤ 93.84° (Mackenzie distribution max)."""
    sym = _sym_quats_hex()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(42)))
    from grainsmith.orientation.samplers import random_uniform
    quats = random_uniform(20, rng)
    for i in range(len(quats)):
        for j in range(i + 1, len(quats)):
            d = disorientation(quats[i], quats[j], sym)
            assert d.angle_deg <= 93.85, (
                f"Hexagonal disorientation {d.angle_deg:.2f}° exceeds theoretical max"
            )


def test_class_deviation_double_coset_invariant():
    """SCIENCE_AUDIT #5: the CSL/misorientation class deviation is a
    double-coset (Sym·m·Sym + exchange) invariant.  Applying arbitrary
    symmetry operators to BOTH operands, swapping the grains, and inverting
    must leave the deviation from an exact Σ-relationship at ≈0 — whereas a
    naive relative-rotation reduction would report a large spurious angle."""
    from grainsmith.orientation.quaternion import quat_mul, quat_inv
    from grainsmith.analysis.boundaries import _misorientation_class_deviation

    sym = _sym_quats_cubic()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(1958)))
    # exact Σ5 relationship as the reference class
    q_sigma = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 36.86989764584402)
    for _ in range(200):
        # two independent symmetry-equivalent representatives of the SAME class
        q_a = quat_mul(sym[rng.integers(len(sym))],
                       quat_mul(q_sigma, sym[rng.integers(len(sym))]))
        q_b = quat_mul(sym[rng.integers(len(sym))],
                       quat_mul(q_sigma, sym[rng.integers(len(sym))]))
        if rng.random() < 0.5:
            q_b = quat_inv(q_b)                    # grain-exchange branch
        dev = _misorientation_class_deviation(q_a, q_b, sym)
        assert dev < 1e-4, f"class deviation not invariant: {dev:.6e}°"


def test_cubic_max_angle():
    """Cubic disorientation angle ≤ 62.8° (Mackenzie distribution max for m-3m)."""
    sym = _sym_quats_cubic()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(99)))
    from grainsmith.orientation.samplers import random_uniform
    quats = random_uniform(30, rng)
    for i in range(len(quats)):
        for j in range(i + 1, len(quats)):
            d = disorientation(quats[i], quats[j], sym)
            assert d.angle_deg <= 62.81, (
                f"Cubic disorientation {d.angle_deg:.2f}° exceeds theoretical max"
            )
