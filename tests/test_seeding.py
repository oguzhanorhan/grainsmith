"""Tests for seeding.py — §10 test_seeding.py.

RSA placement (min-image, mixed periodicity, determinism, failure mode)
and Lloyd centroidal relaxation (§6.4).
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.errors import TessellationError
from grainsmith.seeding import lloyd_relax, seed_grains, wigner_seitz_radius
from grainsmith.tessellation.flat import FlatTessellation


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# ------------------------------------------------------------------ RSA

def test_seeding_min_distance_periodic():
    """All seeds respect min_seed_distance under periodic min-image."""
    L = np.array([50.0, 50.0, 50.0])
    n = 8
    seeds = seed_grains(n, L, [True, True, True], _rng(1))
    assert seeds.shape == (n, 3)
    sinv = 1.0 / L
    r_ws = wigner_seitz_radius(float(np.prod(L)), n)
    for i in range(n):
        for j in range(i + 1, n):
            dr = seeds[i] - seeds[j]
            dr -= np.round(dr * sinv) * L
            d = np.linalg.norm(dr)
            assert d >= r_ws * 0.99, f"Seeds {i},{j} too close: {d:.4f}"


def test_seeding_mixed_periodicity():
    """Free-axis coordinates stay within the box."""
    L = np.array([60.0, 60.0, 30.0])
    seeds = seed_grains(5, L, [True, True, False], _rng(7))
    assert np.all(seeds[:, 2] >= 0.0)
    assert np.all(seeds[:, 2] <= L[2])


def test_seeding_determinism():
    L = np.array([40.0, 40.0, 40.0])
    s1 = seed_grains(6, L, [True, True, True], _rng(99))
    s2 = seed_grains(6, L, [True, True, True], _rng(99))
    np.testing.assert_array_equal(s1, s2)


def test_seeding_fails_too_dense():
    L = np.array([10.0, 10.0, 10.0])
    with pytest.raises(TessellationError):
        seed_grains(20, L, [True, True, True], _rng(5), min_seed_distance=8.0)


# ------------------------------------------------------------------ Lloyd

def test_lloyd_zero_iterations_is_identity():
    L = np.array([40.0, 40.0, 40.0])
    seeds = seed_grains(5, L, [True, True, True], _rng(3))
    out = lloyd_relax(seeds, L, [True, True, True], 0)
    np.testing.assert_array_equal(out, seeds)


def test_lloyd_deterministic_and_in_box():
    L = np.array([50.0, 50.0, 50.0])
    seeds = seed_grains(8, L, [True, True, True], _rng(13))
    a = lloyd_relax(seeds, L, [True, True, True], 2)
    b = lloyd_relax(seeds, L, [True, True, True], 2)
    np.testing.assert_array_equal(a, b)
    assert np.all(a >= 0.0) and np.all(a < L)


def test_lloyd_equalizes_cell_volumes():
    """Lloyd relaxation moves seeds toward centroids → the spread of cell
    volumes shrinks (the §6.4 'more equiaxed grains' property)."""
    L = np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    seeds = seed_grains(8, L, periodic, _rng(17))
    relaxed = lloyd_relax(seeds, L, periodic, 3)

    vol0 = np.array([c.volume for c in FlatTessellation(seeds, L, periodic).cells])
    vol1 = np.array([c.volume for c in FlatTessellation(relaxed, L, periodic).cells])
    assert float(np.std(vol1)) < float(np.std(vol0)), (
        f"volume spread did not shrink: {np.std(vol0):.1f} → {np.std(vol1):.1f}"
    )


def test_lloyd_mixed_periodicity_stays_in_box():
    L = np.array([60.0, 60.0, 30.0])
    periodic = [True, True, False]
    seeds = seed_grains(5, L, periodic, _rng(19))
    relaxed = lloyd_relax(seeds, L, periodic, 2)
    assert np.all(relaxed[:, 2] > 0.0) and np.all(relaxed[:, 2] < L[2])


def test_seed_grains_lloyd_integration():
    """seed_grains(lloyd_iterations=k) is wired (the config field is no
    longer a silent no-op) and equals RSA + explicit lloyd_relax."""
    L = np.array([50.0, 50.0, 50.0])
    periodic = [True, True, True]
    combined = seed_grains(8, L, periodic, _rng(23), lloyd_iterations=2)
    manual = lloyd_relax(seed_grains(8, L, periodic, _rng(23)), L, periodic, 2)
    np.testing.assert_array_equal(combined, manual)
