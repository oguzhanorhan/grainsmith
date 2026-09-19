"""RSA seeding + optional Lloyd relaxation (§6.4).

Uses ONLY the rng.seeding stream — no np.random.* module calls.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from grainsmith.constants import RNG_MAX_ATTEMPTS_FACTOR
from grainsmith.errors import TessellationError


def wigner_seitz_radius(volume: float, n: int) -> float:
    """r_ws = (3V / (4πN))^{1/3}  (Å)."""
    return (3.0 * volume / (4.0 * np.pi * n)) ** (1.0 / 3.0)


def seed_grains(
    n: int,
    box_lengths: np.ndarray,
    periodic: list[bool],
    rng: np.random.Generator,
    min_seed_distance: float | None = None,
    lloyd_iterations: int = 0,
) -> np.ndarray:
    """
    Place N grain seeds using Rejection Sampling (RSA), optionally followed
    by `lloyd_iterations` rounds of centroidal (Lloyd) relaxation (§6.4) for
    more equiaxed grains.

    Periodic axes use minimum-image criterion; free axes use Euclidean distance.
    Returns (N, 3) float64 array of seed coordinates in [0, L) (periodic axes)
    or [0, L] (free axes).

    Raises TessellationError if placement fails after max_attempts.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    volume = float(np.prod(L))
    r_ws = wigner_seitz_radius(volume, n)
    if min_seed_distance is None:
        min_seed_distance = r_ws  # §6.4: default 1.0 * r_ws
    r2_min = min_seed_distance ** 2
    sinv = np.where(periodic, 1.0 / L, 0.0)  # for min-image on periodic axes

    max_attempts = RNG_MAX_ATTEMPTS_FACTOR * n  # total budget (§6.4)
    seeds = np.empty((n, 3), dtype=np.float64)
    placed = 0

    # B4: spatial-hash grid — cell size = min_seed_distance so the 3×3×3
    # neighbourhood covers exactly all seeds within min_seed_distance.
    #
    # We use floor(L / msd) cells per axis, then CLAMP any coordinate that
    # overshoots (i.e. falls in [n_cells*msd, L)) into the last cell.
    # On periodic axes this guarantees that seeds very near L land in cell
    # n_cells-1, which is the direct periodic-image neighbour of cell 0
    # (reached by wrapping cell -1 → n_cells-1).  With ±1 cell neighbourhood:
    #
    #   • seeds in cells c_p-1..c_p+1 (mod n_cells on periodic axes) are
    #     always checked;
    #   • any two points whose min-image cell distance ≥ 2 have real
    #     min-image distance ≥ msd (so no false negatives);
    #   • the explicit distance check inside the loop prevents false positives.
    #
    # This yields the SAME accept/reject decision as the brute-force loop
    # for every candidate.
    n_cells = [
        max(1, int(L[ax] / min_seed_distance)) for ax in range(3)
    ]
    # Map from cell-index tuple -> list of placed seed indices
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _cell(p: np.ndarray) -> tuple[int, int, int]:
        """Cell index for point p — clamped to [0, n_cells[ax]-1]."""
        return (
            min(int(p[0] / min_seed_distance), n_cells[0] - 1),
            min(int(p[1] / min_seed_distance), n_cells[1] - 1),
            min(int(p[2] / min_seed_distance), n_cells[2] - 1),
        )

    def _overlaps(p: np.ndarray) -> bool:
        """Return True iff p is within min_seed_distance of any placed seed,
        using the SAME per-axis min-image test as the original brute-force."""
        cx, cy, cz = _cell(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    # Compute neighbour cell index with optional wrap
                    nx_ = cx + dx
                    ny_ = cy + dy
                    nz_ = cz + dz
                    if periodic[0]:
                        nx_ %= n_cells[0]
                    else:
                        if nx_ < 0 or nx_ >= n_cells[0]:
                            continue
                    if periodic[1]:
                        ny_ %= n_cells[1]
                    else:
                        if ny_ < 0 or ny_ >= n_cells[1]:
                            continue
                    if periodic[2]:
                        nz_ %= n_cells[2]
                    else:
                        if nz_ < 0 or nz_ >= n_cells[2]:
                            continue
                    for k in grid[(nx_, ny_, nz_)]:
                        dr = p - seeds[k]
                        # Minimum image on periodic axes only (identical to
                        # original: dr - round(dr * sinv) * L)
                        dr = dr - np.round(dr * sinv) * L
                        if np.dot(dr, dr) < r2_min:
                            return True
        return False

    for _ in range(max_attempts):
        if placed == n:
            break
        # Random point in box — SAME call as original, preserves RNG stream
        p = rng.uniform(0.0, 1.0, size=3) * L
        if not _overlaps(p):
            seeds[placed] = p
            cx, cy, cz = _cell(p)
            # Insert into grid using the raw (unwrapped) cell index so that
            # lookup from wrapped neighbours finds it correctly
            grid[(cx, cy, cz)].append(placed)
            placed += 1

    if placed < n:
        raise TessellationError(
            f"RSA seeding failed: could only place {placed}/{n} grains "
            f"with min_seed_distance={min_seed_distance:.4g} Å "
            f"(box={L.tolist()}, r_ws={r_ws:.4g} Å). "
            "Try a smaller min_seed_distance or fewer grains."
        )
    if lloyd_iterations > 0:
        seeds = lloyd_relax(seeds, L, periodic, lloyd_iterations)
    return seeds


def lloyd_relax(
    seeds: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    iterations: int,
    grid_size: int | str = "auto",
) -> np.ndarray:
    """Centroidal (Lloyd) relaxation of grain seeds (§6.4).

    Each iteration assigns voxel centers to their nearest seed (min-image on
    periodic axes) and moves every seed to the centroid of its cell.  The
    centroid of a cell that wraps a periodic face is computed in the seed's
    local frame (min-image displacements averaged, then rewrapped), so
    wrapping cells relax correctly.  Deterministic: no randomness involved.

    Parameters
    ----------
    seeds : (N, 3) float64, modified copy returned
    grid_size : voxels along the shortest edge; "auto" → h = r_ws / 10
        clamped to ≤ VOXEL_GRID_MAX (the §6.7 auto rule without the ℓ term).
    """
    from grainsmith.constants import VOXEL_GRID_MAX
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

        # B5: grouped reduction — O(N_voxels) instead of O(N_grains·N_voxels).
        # Compute per-voxel displacement in the seed's local min-image frame.
        dr = centers - seeds[labels]          # (V, 3) gather
        # Min-image on periodic axes only (vectorised, same formula as before)
        dr = dr - np.round(dr * sinv) * L    # (V, 3)

        # Accumulate per-label sums and counts via bincount
        counts = np.bincount(labels, minlength=n).astype(np.float64)  # (n,)

        # Check for empty grains before dividing
        empty = np.where(counts == 0)[0]
        if len(empty) > 0:
            i = int(empty[0])
            raise TessellationError(
                f"Lloyd relaxation: grain {i} owns zero voxels — grid "
                "too coarse for this seed density."
            )

        # Sum dr components per label
        sum_dr = np.empty((n, 3), dtype=np.float64)
        for ax in range(3):
            sum_dr[:, ax] = np.bincount(labels, weights=dr[:, ax], minlength=n)

        # New seed positions: seeds[i] + mean(dr_i)
        new_seeds = seeds + sum_dr / counts[:, None]

        # Re-wrap on periodic axes (same as original per-axis logic)
        for ax in range(3):
            if periodic[ax]:
                new_seeds[:, ax] %= L[ax]
                new_seeds[new_seeds[:, ax] >= L[ax], ax] = 0.0

        seeds = new_seeds

    return seeds


def _nearest_seed_min_image(
    X: np.ndarray,
    seeds: np.ndarray,
    periodic: list[bool],
    L: np.ndarray,
) -> np.ndarray:
    """Nearest-seed labels under per-axis min-image (grain_of semantics)."""
    from scipy.spatial import KDTree

    if all(periodic):
        s = seeds % L
        s[s >= L] = 0.0
        _, idx = KDTree(s, boxsize=L).query(X)
        return np.asarray(idx, dtype=np.int32)

    # Mixed periodicity: explicit translations on periodic axes only.
    from itertools import product as iproduct
    opts = [([-1.0, 0.0, 1.0] if periodic[ax] else [0.0]) for ax in range(3)]
    offsets = np.array(list(iproduct(*opts)), dtype=np.float64) * L[None, :]
    rep = (seeds[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    _, idx = KDTree(rep).query(X)
    return (np.asarray(idx) % len(seeds)).astype(np.int32)
