"""Mandatory equality tests for B4 (RSA spatial-hash) and B5 (Lloyd grouped
reduction) optimisations in seeding.py.

B4: new seed_grains must be BIT-IDENTICAL to the reference brute-force RSA.
B5: new lloyd_relax must match the reference per-grain-loop implementation
    to np.allclose(atol=1e-9).

The reference implementations are pasted verbatim from the pre-optimisation
code so this test is fully self-contained.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.errors import TessellationError
from grainsmith.seeding import lloyd_relax, seed_grains


# ---------------------------------------------------------------------------
# Reference implementations (pre-optimisation verbatim copies)
# ---------------------------------------------------------------------------

def _ref_seed_grains(
    n: int,
    box_lengths,
    periodic,
    rng,
    min_seed_distance=None,
    lloyd_iterations: int = 0,
):
    """Brute-force O(N²) RSA — verbatim copy of the pre-B4 implementation."""
    from grainsmith.constants import RNG_MAX_ATTEMPTS_FACTOR
    from grainsmith.seeding import wigner_seitz_radius

    L = np.asarray(box_lengths, dtype=np.float64)
    volume = float(np.prod(L))
    r_ws = wigner_seitz_radius(volume, n)
    if min_seed_distance is None:
        min_seed_distance = r_ws
    r2_min = min_seed_distance ** 2
    sinv = np.where(periodic, 1.0 / L, 0.0)

    max_attempts = RNG_MAX_ATTEMPTS_FACTOR * n
    seeds = np.empty((n, 3), dtype=np.float64)
    placed = 0

    for _ in range(max_attempts):
        if placed == n:
            break
        p = rng.uniform(0.0, 1.0, size=3) * L
        ok = True
        for k in range(placed):
            dr = p - seeds[k]
            dr = dr - np.round(dr * sinv) * L
            if np.dot(dr, dr) < r2_min:
                ok = False
                break
        if ok:
            seeds[placed] = p
            placed += 1

    if placed < n:
        raise TessellationError(
            f"RSA seeding failed: could only place {placed}/{n} grains "
            f"with min_seed_distance={min_seed_distance:.4g} Å "
            f"(box={L.tolist()}, r_ws={r_ws:.4g} Å). "
            "Try a smaller min_seed_distance or fewer grains."
        )
    # skip lloyd in reference so we isolate RSA comparison
    return seeds


def _ref_lloyd_relax(
    seeds,
    box_lengths,
    periodic,
    iterations: int,
    grid_size="auto",
):
    """Per-grain-loop O(N_grains·N_voxels) Lloyd — verbatim copy of
    the pre-B5 implementation."""
    from grainsmith.constants import VOXEL_GRID_MAX
    from grainsmith.seeding import _nearest_seed_min_image, wigner_seitz_radius
    from grainsmith.tessellation.voxel import _voxel_centers

    if iterations <= 0:
        return np.asarray(seeds, dtype=np.float64).copy()

    L = np.asarray(box_lengths, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64).copy()
    n = len(seeds)
    r_ws = wigner_seitz_radius(float(np.prod(L)), n)
    L_min = float(np.min(L))

    if grid_size == "auto":
        n_short = min(int(np.ceil(L_min / (r_ws / 10.0))), VOXEL_GRID_MAX)
    else:
        n_short = int(grid_size)
    h = L_min / n_short
    shape = tuple(min(int(np.ceil(L[k] / h)), VOXEL_GRID_MAX) for k in range(3))
    centers = _voxel_centers(L, shape)

    sinv = np.where(periodic, 1.0 / L, 0.0)
    for _ in range(iterations):
        labels = _nearest_seed_min_image(centers, seeds, periodic, L)
        for i in range(n):
            pts = centers[labels == i]
            if len(pts) == 0:
                raise TessellationError(
                    f"Lloyd relaxation: grain {i} owns zero voxels — grid "
                    "too coarse for this seed density."
                )
            dr = pts - seeds[i]
            dr -= np.round(dr * sinv) * L
            new = seeds[i] + dr.mean(axis=0)
            for ax in range(3):
                if periodic[ax]:
                    new[ax] %= L[ax]
                    if new[ax] >= L[ax]:
                        new[ax] = 0.0
            seeds[i] = new
    return seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# ---------------------------------------------------------------------------
# B4 — bit-identical RSA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [
    # (n, box_lengths, periodic, min_seed_distance, rng_seed)
    # fully periodic, cubic
    (5,   [50.0, 50.0, 50.0], [True,  True,  True],  None,  1),
    (50,  [80.0, 80.0, 80.0], [True,  True,  True],  None,  2),
    (500, [200., 200., 200.], [True,  True,  True],  None,  3),
    # slab (free z)
    (5,   [60.0, 60.0, 30.0], [True,  True,  False], None,  4),
    (50,  [80.0, 80.0, 40.0], [True,  True,  False], None,  5),
    # non-cubic box
    (5,   [40.0, 60.0, 80.0], [True,  True,  True],  None,  6),
    (50,  [40.0, 60.0, 80.0], [True,  True,  True],  None,  7),
    # explicit min_seed_distance
    (5,   [50.0, 50.0, 50.0], [True,  True,  True],  5.0,   8),
    (5,   [60.0, 60.0, 30.0], [True,  True,  False], 4.0,   9),
    # all-free axes (no periodic wrapping)
    (5,   [50.0, 50.0, 50.0], [False, False, False], None, 10),
])
def test_b4_bit_identical(params):
    n, box, per, msd, rseed = params
    L = np.array(box, dtype=np.float64)
    got = seed_grains(n, L, per, _rng(rseed), min_seed_distance=msd)
    ref = _ref_seed_grains(n, L, per, _rng(rseed), min_seed_distance=msd)
    assert np.array_equal(got, ref), (
        f"B4 NOT bit-identical for n={n}, box={box}, per={per}, "
        f"msd={msd}, seed={rseed}.\n"
        f"  max |diff| = {np.max(np.abs(got - ref))}"
    )


# ---------------------------------------------------------------------------
# B5 — allclose lloyd_relax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [
    # (n, box_lengths, periodic, iterations, rng_seed)
    # fully periodic
    (4,  [40.0, 40.0, 40.0], [True,  True,  True],  1, 11),
    (8,  [50.0, 50.0, 50.0], [True,  True,  True],  2, 12),
    (8,  [50.0, 50.0, 50.0], [True,  True,  True],  5, 13),
    # slab
    (5,  [60.0, 60.0, 30.0], [True,  True,  False], 2, 14),
    (5,  [60.0, 60.0, 30.0], [True,  True,  False], 4, 15),
    # non-cubic
    (6,  [40.0, 60.0, 80.0], [True,  True,  True],  3, 16),
    # zero iterations → identity (trivial but exercises early-return path)
    (4,  [40.0, 40.0, 40.0], [True,  True,  True],  0, 17),
])
def test_b5_allclose(params):
    n, box, per, iters, rseed = params
    L = np.array(box, dtype=np.float64)
    seeds = seed_grains(n, L, per, _rng(rseed))
    got = lloyd_relax(seeds, L, per, iters)
    ref = _ref_lloyd_relax(seeds, L, per, iters)
    assert np.allclose(got, ref, atol=1e-9), (
        f"B5 NOT allclose(atol=1e-9) for n={n}, box={box}, per={per}, "
        f"iters={iters}, seed={rseed}.\n"
        f"  max |diff| = {np.max(np.abs(got - ref))}"
    )
