"""Tests for tessellation/flat.py — §10 (seeding tests: test_seeding.py)."""
from __future__ import annotations
import numpy as np
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation


# Seeding tests live in tests/test_seeding.py (§10 layout).

def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# ------------------------------------------------------------------
# Flat tessellation tests
# ------------------------------------------------------------------

def _flat_tess(n=6, Lval=50.0, seed=42, periodic=None):
    if periodic is None:
        periodic = [True, True, True]
    L = np.array([Lval, Lval, Lval])
    rng = _rng(seed)
    seeds = seed_grains(n, L, periodic, rng)
    return FlatTessellation(seeds, L, periodic), seeds, L


def test_volume_sum_periodic():
    """Sum of cell volumes == box volume (G3: rel err ≤ 1e-6)."""
    tess, seeds, L = _flat_tess(n=8, Lval=60.0)
    box_vol = float(np.prod(L))
    total = tess.total_volume()
    rel_err = abs(total - box_vol) / box_vol
    assert rel_err < 1e-6, f"Volume sum error {rel_err:.2e} (G3 target ≤ 1e-6)"


def test_membership_at_seeds():
    """grain_of(seeds[i]) == i for all grains."""
    tess, seeds, L = _flat_tess(n=6)
    gids = tess.grain_of(seeds)
    for i in range(len(seeds)):
        assert int(gids[i]) == i, f"Seed {i} assigned to grain {int(gids[i])}"


def test_adjacency_symmetric():
    """Adjacency list contains each pair exactly once, i < j."""
    tess, _, _ = _flat_tess(n=8)
    adj = tess.adjacency()
    for i, j in adj:
        assert i < j, f"Adjacency pair ({i},{j}) not sorted"
    # No duplicates
    assert len(adj) == len(set(adj))


def test_bounding_radius_covers_vertices():
    """bounding_radius(i) >= distance from seeds[i] to any vertex of cell i."""
    tess, seeds, L = _flat_tess(n=5)
    for _i, cell in enumerate(tess.cells):
        if len(cell.vertices) == 0:
            continue
        dists = np.linalg.norm(cell.vertices, axis=1)  # already relative to seed
        assert float(np.max(dists)) <= cell.bounding_radius + 1e-10


def test_flat_slab_clipping():
    """Slab (non-periodic z): cells are clipped EXACTLY at the box walls.

    Pins the AUDIT C4 regression — the previous implementation skipped
    unbounded ridges, giving a volume sum 6× the box and cells extending
    to z ≈ −39…+452 Å for this very configuration."""
    L = np.array([60.0, 60.0, 30.0])
    rng = _rng(3)
    seeds = seed_grains(5, L, [True, True, False], rng)
    tess = FlatTessellation(seeds, L, [True, True, False])

    # Membership of seeds
    gids = tess.grain_of(seeds)
    for i in range(5):
        assert int(gids[i]) == i

    # G3 for the slab: cells tile the box exactly
    box_vol = float(np.prod(L))
    rel_err = abs(tess.total_volume() - box_vol) / box_vol
    assert rel_err < 1e-6, f"slab volume sum error {rel_err:.2e}"

    # Every cell stays inside [0, Lz] on the free axis; wall faces exist
    n_wall = 0
    for i, cell in enumerate(tess.cells):
        z_abs = cell.vertices[:, 2] + seeds[i, 2]
        assert np.all(z_abs >= -1e-9) and np.all(z_abs <= L[2] + 1e-9), (
            f"cell {i} extends outside the slab: z ∈ "
            f"[{z_abs.min():.3g}, {z_abs.max():.3g}]"
        )
        n_wall += sum(1 for f in cell.faces if f.neighbor_id == -1)
    assert n_wall > 0, "no wall faces found in slab geometry"

    # Adjacency contains no self pairs and no wall sentinels
    for i, j in tess.adjacency():
        assert 0 <= i < j < 5


def test_margin_signed_and_finite():
    """margin() is a finite SIGNED boundary distance: positive inside the
    grain, negative outside, exact magnitude for a symmetric bicrystal.
    Pins the AUDIT C3 regression (NaN from periodic self-image faces)."""
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    pts = np.array([
        [9.5, 10.0, 10.0],   # 0.5 inside grain 0 (GB at x = 10)
        [10.5, 10.0, 10.0],  # 0.5 outside grain 0
        [5.0, 10.0, 10.0],   # seed of grain 0 (GB planes at x = 0 and 10)
    ])
    m = tess.margin(pts, 0)
    assert np.all(np.isfinite(m))
    np.testing.assert_allclose(m, [0.5, -0.5, 5.0], atol=1e-12)


def test_owns_tiles_torus_exactly_once():
    """Each torus point is owned exactly once over all grains AND all
    periodic lifts (the §6.8 fill-correctness invariant).

    The compact home cells tile the TORUS, not the box: a cell straddling a
    box face owns the wrapped part of its region at an out-of-box lift, so
    the sum must run over the {-1,0,1}³ lifts of each point."""
    tess, seeds, L = _flat_tess(n=6, Lval=50.0, seed=11)
    rng = _rng(123)
    pts = rng.uniform(0.0, 1.0, size=(200, 3)) * L
    owned = np.zeros(len(pts), dtype=int)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                lift = pts + np.array([sx, sy, sz]) * L
                for i in range(len(seeds)):
                    owned += tess.owns(lift, i).astype(int)
    assert np.all(owned == 1), (
        f"torus tiling violated: min={owned.min()}, max={owned.max()}"
    )


def test_owns_rejects_periodic_image():
    """owns() distinguishes the home cell from its periodic image."""
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[10.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    assert bool(tess.owns(seeds, 0)[0]) is True
    assert bool(tess.owns(seeds + L, 0)[0]) is False
