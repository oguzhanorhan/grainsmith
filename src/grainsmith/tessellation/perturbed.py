"""Periodic perturbed-distance (level-set) self-affine grain-boundary
tessellation backend (§6.6 extension, method M6).

Level-set membership:

    grain_of(x) = argmin_i [ d_pbc(x, s_i) − A · η_i(x) ]

where ``d_pbc`` is the periodic (minimum-image) Euclidean distance to seed
``s_i`` and ``η_i`` is a grain-INDEPENDENT, periodic, band-limited scalar
random field with the SAME spectral conventions ``warp.synthesize_grf``
already implements for the vector warp field (``gaussian`` single-scale,
or ``self_affine`` band-limited power law √Φ(k) ∝ k^(−(3+2H)/2)) — the
only difference is one scalar component per grain instead of three vector
components per box.

Why this is not ``warp``
-------------------------
``WarpTessellation`` evaluates ``base.grain_of(x + u(x))`` with a hard
bijectivity guard ``max‖∇u‖ < WARP_GRAD_MAX``: a bijective coordinate map
cannot change the box-counting dimension of the flat-Voronoi
surface it warps — the ``self_affine`` spectrum only recolors the
band-limited waviness riding on an otherwise D=2 (3D) / D_b=1 (planar
section) surface. Perturbing the ASSIGNMENT RULE itself (rather than the
coordinate) removes the injectivity constraint: the zero level set of
``[d(x,s_i) − A η_i(x)] − [d(x,s_j) − A η_j(x)]`` is a genuine implicit
surface whose roughness spectrum is inherited directly from ``η_i − η_j``,
producing a real, tunable self-affine boundary: box-counting D_b responds
continuously and monotonically to the Hurst exponent, collapsing to the
flat-Voronoi value D_b=1.000 as A → 0.

Field allocation — graph coloring, not N independent grids
------------------------------------------------------------
Non-adjacent (non-competing) grains can safely SHARE one underlying field
— the argmin only ever compares grains whose perturbed distances can
plausibly cross, so two grains that never compete are unaffected by
sharing noise. This is graph coloring: a conservative "conflict graph"
(the UNION of the actual flat-Voronoi adjacency — a required safety net,
see :func:`_flat_conflict_edges` — and a generous circumradius-overlap
test) is greedy-colored (Welsh–Powell order), and each color gets its own
synthesized field. Sharing a field between two ADJACENT grains would
silently degenerate their shared boundary to the flat bisector (the
shared term cancels algebraically in the argmin difference) — this is why
the conflict graph must be a proven superset of true adjacency, not a
convenient guess.

Guards
------
1. Seed containment (ConfigError, hard, pre-flight only — see below):
   ``A ≤ PERTURBED_DISTANCE_SAFETY · min_seed_distance / (2 · ETA_CLIP)``.
   Every synthesized field is hard-clipped to ``±ETA_CLIP`` (a
   deterministic bound, not an empirical tail estimate) BEFORE this ratio
   is used anywhere, so the bound is a proof, not a heuristic. This
   replaces warp's bijectivity guard (G6), which does not apply here (no
   diffeomorphism exists to check) — resolve.py keeps G6 warp-specific.
2. Exact seed-ownership (ConfigError, hard, POST-hoc): after field
   synthesis, ``grain_of(seeds[i])`` must equal ``i`` for every grain.
   This is the actual property guard (1) exists to make plausible; it is
   checked directly and exactly rather than only inferred from an
   amplitude bound, so guard (1) only needs to be a SUFFICIENT ceiling
   (msd/12 at the documented constants), not a tight one.
3. Band limits, shared textually with warp (config/resolve.py):
   ``l_min < l_max ≤ min(L)/2`` (resolve-time) and
   ``l_min ≥ 2·h_field`` (checked here, against the actual grid).
4. Connectivity (G5) is a REPAIR-AND-REPORT gate here, not a hard fail:
   see :func:`repair_connectivity`. ``reassigned_fraction`` is exposed as
   a property and written to summary.csv (pipeline.py) as a genuine
   quality metric, not a cosmetic one — the repaired label field is
   applied as a sparse voxel-snapped override consulted by
   ``grain_of``/``owns``/``margin`` for any query point landing in a
   repaired voxel (see :class:`PerturbedDistanceTessellation._override`),
   so the reported fraction describes what atoms actually get filled with.

Candidate-restricted assignment (no N_grains × N_voxels product)
-------------------------------------------------------------------
Built on the same periodic-replica ``cKDTree`` pattern used throughout
this package (``flat.py``'s local-competitor sets, ``weighted.py``'s
``_periodic_translations``): for a query point at nearest-seed distance
``d_1``, any replica farther than ``d_1 + 2·A·ETA_CLIP`` can be proven
(since ``|η| ≤ ETA_CLIP`` everywhere) never to win the argmin, so
``cKDTree.query_ball_point`` restricts the competitor set to a handful of
candidates per point instead of scanning every grain.

References: a worked example is documented in
``examples/advanced/adv_pdau_perturbed.yaml``. The general
self-affine grain-boundary phenomenology this method targets is C. Braun
et al., Sci. Rep. 8, 1592 (2018) (``METHODS_BIBLIOGRAPHY["fractal_gb"]``)
— a DIFFERENT paper from the D_b = 1.174 ± 0.004 box-counting target and
protocol this method's gate G20 compares against, which is C. Braun et
al., J. Appl. Phys. 128, 105102 (2020)
(``METHODS_BIBLIOGRAPHY["braun2020fractal"]``); io/methods.py cites both
of these, plus ``["selfaffine_psd"]``, wherever it generates the
self_affine paragraph (``warp.py`` itself cites neither — its
``gaussian``-only spectrum does not accept ``self_affine`` at all, see
its own module docstring).
"""
from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import label as _cc_label
from scipy.spatial import cKDTree

from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY, VOXEL_GRID_MAX
from grainsmith.errors import ConfigError, TessellationError
from grainsmith.tessellation.base import Tessellation
from grainsmith.tessellation.warp import synthesize_grf
from grainsmith.tessellation.weighted import _periodic_translations

if TYPE_CHECKING:
    from grainsmith.tessellation.voxel import VoxelGrid

log = logging.getLogger(__name__)

_CHUNK = 50_000
"""Query-point chunk size for candidate-restricted evaluation — same
memory-bounding role as flat.py's ``_OWNS_CHUNK`` / weighted.py's
``_CHUNK``."""

_MAX_REPAIR_PASSES = 25
"""Hard cap on G5 majority-vote repair sweeps (§ repair_connectivity).
Every case exercised during design converged in 1–3 passes (isolated
fragments only ever need to propagate across their own diameter in
voxels); 25 is generous headroom before raising, mirroring the
``atoms/overlap.py`` convergence-loop-with-cap pattern (that module's own
cap is 100 for a finer-grained per-pair process; ours operates on whole
fragments per pass, so far fewer passes are needed for a comparable
safety margin)."""


# ---------------------------------------------------------------------------
# Graph coloring: safe field-sharing among non-adjacent grains
# ---------------------------------------------------------------------------


def _flat_conflict_edges(
    seeds: np.ndarray, L: np.ndarray, periodic: list[bool]
) -> set[tuple[int, int]]:
    """Actual flat-Voronoi adjacency of *seeds*, used as a conflict-graph
    safety net (see module docstring: a pure circumradius-overlap test on
    HOME-replica distances is not always a superset of true adjacency —
    ``FlatTessellation.adjacency`` scans ridges across every periodic
    replica, occasionally realizing an edge between two non-home
    replicas whose home-to-home distance exceeds the circumradius sum).
    Reused verbatim rather than re-derived geometrically."""
    from grainsmith.tessellation.flat import FlatTessellation

    ft = FlatTessellation(seeds, L, periodic)
    return set(ft.adjacency())


def _min_image_dist_matrix(
    seeds: np.ndarray, L: np.ndarray, periodic: list[bool]
) -> np.ndarray:
    """(n, n) periodic minimum-image seed-to-seed distance matrix."""
    diff = seeds[:, None, :] - seeds[None, :, :]
    for ax in range(3):
        if periodic[ax]:
            diff[..., ax] -= np.round(diff[..., ax] / L[ax]) * L[ax]
    return np.sqrt((diff**2).sum(axis=2))


def build_conflict_graph(
    seeds: np.ndarray,
    L: np.ndarray,
    periodic: list[bool],
    flat_radii: np.ndarray,
    a_max: float,
) -> set[tuple[int, int]]:
    """Conservative conflict graph over grains: the union of (1) the
    actual flat-Voronoi adjacency and (2) a circumradius-overlap test
    dilated by ``4·a_max·ETA_CLIP`` — the worst-case boundary
    displacement any config-legal amplitude (up to the seed-containment
    ceiling ``a_max``) could produce. Built as a pure function of
    (seeds, box, guard ceiling) — independent of the run's actual
    ``amplitude``/``hurst``/``spectrum`` — so it is safe (a superset of
    the TRUE measured adjacency) for any amplitude the config could
    legally request, and reusable unchanged across a parameter sweep at
    fixed geometry. Verified against the measured adjacency of an actual
    synthesized tessellation at the guard ceiling (zero missed edges)."""
    edges = _flat_conflict_edges(seeds, L, periodic)
    D = _min_image_dist_matrix(seeds, L, periodic)
    margin = 4.0 * a_max * ETA_CLIP
    thresh = flat_radii[:, None] + flat_radii[None, :] + margin
    hits = np.argwhere(np.triu(D <= thresh, k=1))
    edges.update((int(i), int(j)) for i, j in hits)
    return edges


def greedy_color_grains(n_grains: int, conflict_edges: set[tuple[int, int]]) -> np.ndarray:
    """Welsh–Powell greedy coloring of the conflict graph: process grains
    in descending-degree order, assign the lowest color not already used
    by an already-colored neighbor. Deterministic given a deterministic
    edge set (ties in degree broken by grain index via ``np.argsort``'s
    stable sort) — no RNG involvement, so the coloring (and hence RNG
    consumption downstream) depends only on (seeds, box, guard
    constants), never on ``hurst``/``amplitude``/``spectrum``.

    Returns (n_grains,) int32 color ids, 0 … n_colors-1.
    """
    neighbors: list[set[int]] = [set() for _ in range(n_grains)]
    for i, j in conflict_edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    degree = np.array([len(neighbors[v]) for v in range(n_grains)])
    order = np.argsort(-degree, kind="stable")
    color = -np.ones(n_grains, dtype=np.int32)
    for v in order:
        used = {int(color[u]) for u in neighbors[int(v)] if color[u] >= 0}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


# ---------------------------------------------------------------------------
# Periodic scalar-field interpolation (trilinear; scalar analogue of
# warp.WarpTessellation._interpolate_u)
# ---------------------------------------------------------------------------


def _interpolate_scalar_field(
    field: np.ndarray,
    X: np.ndarray,
    L: np.ndarray,
    periodic: list[bool],
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trilinear interpolation of a single periodic scalar field at lab
    positions X. Boundary handling mirrors
    ``warp.WarpTessellation._interpolate_u`` exactly (true periodic wrap
    on periodic axes — NOT scipy's ``mode='wrap'``, which shortens the
    period by one sample — and edge-clamp on free axes), applied to one
    scalar component instead of three."""
    n_pts = len(X)
    i0 = np.empty((3, n_pts), dtype=np.int64)
    i1 = np.empty((3, n_pts), dtype=np.int64)
    frac = np.empty((3, n_pts), dtype=np.float64)
    for ax in range(3):
        N = grid_shape[ax]
        t = X[:, ax] / L[ax] * N - 0.5
        if periodic[ax]:
            t = t % N
            lo = np.floor(t).astype(np.int64)
            frac[ax] = t - lo
            i0[ax] = lo % N
            i1[ax] = (lo + 1) % N
        else:
            t = np.clip(t, 0.0, N - 1.0)
            lo = np.minimum(np.floor(t).astype(np.int64), N - 2)
            frac[ax] = t - lo
            i0[ax] = lo
            i1[ax] = lo + 1
    fx, fy, fz = frac
    gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
    F = field
    return (
        F[i0[0], i0[1], i0[2]] * gx * gy * gz
        + F[i1[0], i0[1], i0[2]] * fx * gy * gz
        + F[i0[0], i1[1], i0[2]] * gx * fy * gz
        + F[i0[0], i0[1], i1[2]] * gx * gy * fz
        + F[i1[0], i1[1], i0[2]] * fx * fy * gz
        + F[i1[0], i0[1], i1[2]] * fx * gy * fz
        + F[i0[0], i1[1], i1[2]] * gx * fy * fz
        + F[i1[0], i1[1], i1[2]] * fx * fy * fz
    )


def _eta_at(
    grain_ids: np.ndarray,
    X: np.ndarray,
    color_of_grain: np.ndarray,
    fields: np.ndarray,
    L: np.ndarray,
    periodic: list[bool],
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """η at (grain_id[k], X[k]) for parallel arrays *grain_ids*/*X*,
    batched per color (each color's field is interpolated once for all
    rows that need it, rather than row-by-row)."""
    colors = color_of_grain[grain_ids]
    out = np.empty(len(X), dtype=np.float64)
    for c in np.unique(colors):
        m = colors == c
        out[m] = _interpolate_scalar_field(
            fields[c], X[m], L, periodic, grid_shape
        )
    return out


# ---------------------------------------------------------------------------
# Numba acceleration: fused candidate-evaluation kernels for owns() /
# margin() / grain_of(), proven bit-identical to the NumPy reference paths
# below.
#
# All three kernels share one fused per-candidate primitive, ``_interp_eta``
# — the numba scalar equivalent of ``_interpolate_scalar_field`` above —
# plus an inline ``draw = sqrt(dx*dx+dy*dy+dz*dz)`` distance: the SAME
# ``val = draw - A*eta`` quantity every NumPy call site in this module
# computes, assembled without ever materializing the (K,)/(K,3)
# candidate-sized temporaries the NumPy path builds (``row``, ``gid``,
# ``dr``, ``draw``, ``eta``, ``val``, plus ``_segment_argmin``'s lexsort
# scratch).
#
# K1 (owns): short-circuit accept/reject against a directly-computed
#   ``val_home``, reproducing ``_segment_argmin``'s stable first-minimum
#   tie rule restated on replica-index-vs-``home_idx``.
# K2 (margin): full-scan running minimum, no short-circuit (there is no
#   privileged candidate to short-circuit against) — ``val`` can never
#   land on exactly ``-0.0``, which makes the reduction order-
#   independent, so a plain strict-``<`` running min is exact.
# K3 (grain_of / ``_candidates_home``): full-scan payload argmin (the
#   grain id at the row minimum), the same tie rule as K1 restated
#   without a home distinction.
#
# Numba is a SOFT dependency; ``GRAINSMITH_NO_NUMBA`` (or no numba
# installed) falls back to the exact NumPy paths below, kept byte-for-byte
# verbatim as the reference — same contract as weighted.py's "Numba
# acceleration" section and tessellation/_local_owns.py's ``_USE_NUMBA``
# flag.
# ---------------------------------------------------------------------------

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - environment without numba
    _HAVE_NUMBA = False


def _numba_disabled_by_env() -> bool:
    """``GRAINSMITH_NO_NUMBA=1`` (or ``true``/``yes``/``on``, case-
    insensitive) forces the NumPy owns()/margin()/grain_of() path — the
    same escape hatch, same function name, as weighted.py's own."""
    return os.environ.get("GRAINSMITH_NO_NUMBA", "").strip().lower() in (
        "1", "true", "yes", "on")


_USE_NUMBA: bool = _HAVE_NUMBA and not _numba_disabled_by_env()
"""Module-level dispatch flag (mirrors weighted.py's ``_USE_NUMBA``
contract exactly): equality tests monkeypatch this directly to force
either path for a direct A/B comparison. Nothing in the engine mutates it
after import except the environment read above."""


if _HAVE_NUMBA:

    @numba.njit(cache=True)
    def _interp_eta(x, y, z, field, Nx, Ny, Nz, Lx, Ly, Lz,
                    per_x, per_y, per_z):
        """Numba scalar equivalent of ``_interpolate_scalar_field`` at one
        lab point: same trilinear 8-term sum in the SAME textual term
        order (corner 000,100,010,001,110,101,011,111), same per-axis
        periodic-wrap / free-axis-clamp logic, same float32 -> float64
        widening AT THE READ (never an accumulator that stays float32)."""
        # axis 0 (x)
        t = x / Lx * Nx - 0.5
        if per_x:
            t = t % Nx
            lo = np.int64(math.floor(t))
            fx = t - lo
            i0x = lo % Nx
            i1x = (lo + 1) % Nx
        else:
            t = min(max(t, 0.0), Nx - 1.0)
            lo = min(np.int64(math.floor(t)), Nx - 2)
            fx = t - lo
            i0x = lo
            i1x = lo + 1
        # axis 1 (y)
        t = y / Ly * Ny - 0.5
        if per_y:
            t = t % Ny
            lo = np.int64(math.floor(t))
            fy = t - lo
            i0y = lo % Ny
            i1y = (lo + 1) % Ny
        else:
            t = min(max(t, 0.0), Ny - 1.0)
            lo = min(np.int64(math.floor(t)), Ny - 2)
            fy = t - lo
            i0y = lo
            i1y = lo + 1
        # axis 2 (z)
        t = z / Lz * Nz - 0.5
        if per_z:
            t = t % Nz
            lo = np.int64(math.floor(t))
            fz = t - lo
            i0z = lo % Nz
            i1z = (lo + 1) % Nz
        else:
            t = min(max(t, 0.0), Nz - 1.0)
            lo = min(np.int64(math.floor(t)), Nz - 2)
            fz = t - lo
            i0z = lo
            i1z = lo + 1

        gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        v000 = np.float64(field[i0x, i0y, i0z])
        acc = v000 * gx * gy * gz
        v100 = np.float64(field[i1x, i0y, i0z])
        acc = acc + v100 * fx * gy * gz
        v010 = np.float64(field[i0x, i1y, i0z])
        acc = acc + v010 * gx * fy * gz
        v001 = np.float64(field[i0x, i0y, i1z])
        acc = acc + v001 * gx * gy * fz
        v110 = np.float64(field[i1x, i1y, i0z])
        acc = acc + v110 * fx * fy * gz
        v101 = np.float64(field[i1x, i0y, i1z])
        acc = acc + v101 * fx * gy * fz
        v011 = np.float64(field[i0x, i1y, i1z])
        acc = acc + v011 * gx * fy * fz
        v111 = np.float64(field[i1x, i1y, i1z])
        acc = acc + v111 * fx * fy * fz
        return acc

    @numba.njit(cache=True)
    def _eval_candidate(x, y, z, rep_x, rep_y, rep_z, field, Nx, Ny, Nz,
                        Lx, Ly, Lz, per_x, per_y, per_z, A):
        # Shared by K1/K2/K3: all three evaluate draw before eta, and this
        # shared helper enforces that order.
        dx, dy, dz = x - rep_x, y - rep_y, z - rep_z
        draw = math.sqrt(dx * dx + dy * dy + dz * dz)
        eta = _interp_eta(x, y, z, field, Nx, Ny, Nz, Lx, Ly, Lz,
                          per_x, per_y, per_z)
        return draw - A * eta

    @numba.njit(cache=True)
    def _owns_kernel_perturbed(X, offsets, col_flat, rep_seeds,
                               color_of_grain, fields, Nx, Ny, Nz,
                               Lx, Ly, Lz, per_x, per_y, per_z,
                               A, n_grains, home_idx):
        """K1 — ``owns(X, i)`` fused with short-circuit rejection.
        ``val_home`` is computed once per row, O(1), independent of
        the candidate scan (home is never a candidate — see
        ``if col == home_idx``); the scan then rejects the instant any
        OTHER candidate beats-or-ties (before home is seen) or strictly
        beats (after home is seen) ``val_home`` — exactly
        ``_segment_argmin``'s stable first-minimum tie rule restated on
        scan position, which is index-sorted in this codebase's
        ``query_ball_point`` usage. ``seen_home`` makes the "home absent
        from the candidate list" case correctly return False regardless
        of how the comparisons went."""
        n_pts = X.shape[0]
        out = np.empty(n_pts, dtype=np.bool_)
        for r in range(n_pts):
            x, y, z = X[r, 0], X[r, 1], X[r, 2]

            rx = rep_seeds[home_idx, 0]
            ry = rep_seeds[home_idx, 1]
            rz = rep_seeds[home_idx, 2]
            c_home = color_of_grain[home_idx % n_grains]
            val_home = _eval_candidate(x, y, z, rx, ry, rz, fields[c_home],
                                       Nx, Ny, Nz, Lx, Ly, Lz,
                                       per_x, per_y, per_z, A)

            seen_home = False
            home_wins = True
            for k in range(offsets[r], offsets[r + 1]):
                col = col_flat[k]
                if col == home_idx:
                    seen_home = True
                    continue
                gid = col % n_grains
                rx = rep_seeds[col, 0]
                ry = rep_seeds[col, 1]
                rz = rep_seeds[col, 2]
                c = color_of_grain[gid]
                val = _eval_candidate(x, y, z, rx, ry, rz, fields[c],
                                     Nx, Ny, Nz, Lx, Ly, Lz,
                                     per_x, per_y, per_z, A)
                if not seen_home:
                    if val <= val_home:      # earlier in scan order,
                        home_wins = False    # tie-or-beat -> home loses
                        break
                else:
                    if val < val_home:       # later in scan order,
                        home_wins = False    # strict beat only
                        break
            out[r] = home_wins and seen_home
        return out

    @numba.njit(cache=True)
    def _margin_kernel_perturbed(X, offsets, col_flat, seeds, rep_seeds,
                                 color_of_grain, fields, Nx, Ny, Nz,
                                 Lx, Ly, Lz, per_x, per_y, per_z,
                                 A, n_grains, i):
        """K2 — ``margin(X, i)`` fused, exact values, no short-circuit:
        ``val_i`` is a direct O(1) per-row min-image evaluation against
        grain i's own seed (via ``np.rint`` for the nearest-integer wrap,
        not the builtin ``round()``); ``val_other`` is a plain full-scan
        running minimum over every non-i candidate — no tie-break rule to
        reproduce (``val`` can never be exactly ``-0.0``, so the
        reduction is order-independent) and no early exit (K2 needs the
        actual minimum VALUE, not just an argmin identity)."""
        n_pts = X.shape[0]
        out = np.empty(n_pts, dtype=np.float64)
        sx, sy, sz = seeds[i, 0], seeds[i, 1], seeds[i, 2]
        c_i = color_of_grain[i]
        for r in range(n_pts):
            x, y, z = X[r, 0], X[r, 1], X[r, 2]

            dxi, dyi, dzi = x - sx, y - sy, z - sz
            if per_x:
                dxi -= np.rint(dxi / Lx) * Lx
            if per_y:
                dyi -= np.rint(dyi / Ly) * Ly
            if per_z:
                dzi -= np.rint(dzi / Lz) * Lz
            d_i = math.sqrt(dxi * dxi + dyi * dyi + dzi * dzi)
            eta_i = _interp_eta(x, y, z, fields[c_i], Nx, Ny, Nz,
                                Lx, Ly, Lz, per_x, per_y, per_z)
            val_i = d_i - A * eta_i

            best = np.inf
            for k in range(offsets[r], offsets[r + 1]):
                col = col_flat[k]
                gid = col % n_grains
                if gid == i:
                    continue
                rx = rep_seeds[col, 0]
                ry = rep_seeds[col, 1]
                rz = rep_seeds[col, 2]
                c = color_of_grain[gid]
                val = _eval_candidate(x, y, z, rx, ry, rz, fields[c],
                                     Nx, Ny, Nz, Lx, Ly, Lz,
                                     per_x, per_y, per_z, A)
                if val < best:
                    best = val

            out[r] = (best - val_i) / 2.0
        return out

    @numba.njit(cache=True)
    def _grain_of_kernel_perturbed(X, offsets, col_flat, rep_seeds,
                                   color_of_grain, fields, Nx, Ny, Nz,
                                   Lx, Ly, Lz, per_x, per_y, per_z,
                                   A, n_grains):
        """K3 — ``grain_of(X)`` / ``_candidates_home``: a full-scan
        PAYLOAD argmin (the winning grain id, not just a boolean or a bare
        minimum value). Structurally between K1 and K2: like K2, every
        candidate is symmetric so there is no "home" to short-circuit
        against; like K1, the tie rule (first-minimum wins, strict ``<``
        over the index-sorted scan order) determines which PAYLOAD
        survives a tie, not just which value does."""
        n_pts = X.shape[0]
        out = np.empty(n_pts, dtype=np.int32)
        for r in range(n_pts):
            x, y, z = X[r, 0], X[r, 1], X[r, 2]
            best_val = np.inf
            best_gid = -1
            for k in range(offsets[r], offsets[r + 1]):
                col = col_flat[k]
                gid = col % n_grains
                rx = rep_seeds[col, 0]
                ry = rep_seeds[col, 1]
                rz = rep_seeds[col, 2]
                c = color_of_grain[gid]
                val = _eval_candidate(x, y, z, rx, ry, rz, fields[c],
                                     Nx, Ny, Nz, Lx, Ly, Lz,
                                     per_x, per_y, per_z, A)
                if val < best_val:
                    best_val = val
                    best_gid = gid
            out[r] = best_gid
        return out
else:  # pragma: no cover - environment without numba
    _interp_eta = None
    _eval_candidate = None
    _owns_kernel_perturbed = None
    _margin_kernel_perturbed = None
    _grain_of_kernel_perturbed = None


# ---------------------------------------------------------------------------
# G5: periodic-connectivity repair-and-report (voxel labels)
# ---------------------------------------------------------------------------


def _periodic_merge_roots(comp: np.ndarray, periodic: list[bool]) -> np.ndarray:
    """Union-find merge of a non-periodic ``scipy.ndimage.label`` output
    across periodic face pairs. Returns a (ncomp+1,) root-id array
    (index 0 = background). Same merge logic as
    ``voxel._component_sizes``, generalized to return per-component root
    ids (needed by the repair pass) rather than only merged sizes."""
    ncomp = int(comp.max())
    parent = np.arange(ncomp + 1, dtype=np.intp)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for ax in range(3):
        if not periodic[ax]:
            continue
        first: list[slice | int] = [slice(None)] * 3
        last: list[slice | int] = [slice(None)] * 3
        first[ax] = 0
        last[ax] = -1
        a_face = comp[tuple(first)].ravel()
        b_face = comp[tuple(last)].ravel()
        both = (a_face > 0) & (b_face > 0)
        pairs = np.unique(np.stack([a_face[both], b_face[both]], axis=1), axis=0)
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    return np.array([find(c) for c in range(ncomp + 1)])


def _find_fragment_mask(
    labels: np.ndarray, n_grains: int, periodic: list[bool]
) -> np.ndarray:
    """Boolean mask of voxels NOT in their grain's largest periodically-
    connected component. Raises TessellationError for a zero-voxel grain
    (empty-cell guard, matching ``voxel.check_connectivity``'s existing
    contract — a genuinely empty grain is not a "fragment", it is a
    construction failure with no home to repair into)."""
    fragment_mask = np.zeros(labels.shape, dtype=bool)
    for g in range(n_grains):
        mask = labels == g
        n_total = int(mask.sum())
        if n_total == 0:
            raise TessellationError(
                f"Grain {g} has zero voxels (empty cell, G5) under the "
                "perturbed_distance assignment. Reduce amplitude, refine "
                "the voxel grid, or use fewer grains."
            )
        comp, ncomp = _cc_label(mask)
        if ncomp <= 1:
            continue
        roots = _periodic_merge_roots(comp, periodic)
        sizes = np.bincount(comp.ravel(), minlength=ncomp + 1)
        merged_sizes: dict[int, int] = {}
        for c in range(1, ncomp + 1):
            r = int(roots[c])
            merged_sizes[r] = merged_sizes.get(r, 0) + int(sizes[c])
        if len(merged_sizes) <= 1:
            continue  # single periodic component after merge: fine
        largest_root = max(merged_sizes, key=lambda r: merged_sizes[r])
        comp_roots = roots[comp]
        fragment_mask |= mask & (comp_roots != largest_root)
    return fragment_mask


def _majority_vote_reassign(
    labels: np.ndarray, fragment_mask: np.ndarray, periodic: list[bool]
) -> tuple[np.ndarray, int]:
    """One Jacobi sweep: each fragment voxel is reassigned to the modal
    label among its ≤6 face-neighbors that are NOT themselves fragment
    voxels this pass. O(M) in the number of fragment voxels M — no
    (M, n_grains) dense array (validated equivalent to that reference
    approach, but independent of n_grains). Voxels with no valid
    (non-fragment) neighbor this pass keep their old label; the caller's
    loop iterates until nothing changes."""
    shape = labels.shape
    idx = np.argwhere(fragment_mask)
    M = len(idx)
    if M == 0:
        return labels, 0

    neighbor_vals = np.full((M, 6), -1, dtype=np.int64)
    k = 0
    for ax in range(3):
        for direction in (-1, 1):
            nb = idx.copy()
            nb[:, ax] += direction
            if periodic[ax]:
                nb[:, ax] %= shape[ax]
                valid = np.ones(M, dtype=bool)
            else:
                valid = (nb[:, ax] >= 0) & (nb[:, ax] < shape[ax])
                nb[:, ax] = np.clip(nb[:, ax], 0, shape[ax] - 1)
            nb_frag = fragment_mask[nb[:, 0], nb[:, 1], nb[:, 2]]
            nb_label = labels[nb[:, 0], nb[:, 1], nb[:, 2]]
            neighbor_vals[:, k] = np.where(valid & ~nb_frag, nb_label, -1)
            k += 1

    sorted_vals = np.sort(neighbor_vals, axis=1)
    counts = np.zeros_like(sorted_vals)
    for c in range(6):
        counts[:, c] = np.sum(sorted_vals == sorted_vals[:, c:c + 1], axis=1)
    counts_masked = np.where(sorted_vals < 0, -1, counts)
    best_col = np.argmax(counts_masked, axis=1)
    rows = np.arange(M)
    best_val = sorted_vals[rows, best_col]
    has_valid = counts_masked[rows, best_col] > 0

    old_vals = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    new_vals = np.where(has_valid, best_val, old_vals)
    new_labels = labels.copy()
    new_labels[idx[:, 0], idx[:, 1], idx[:, 2]] = new_vals
    n_changed = int(np.sum(new_vals != old_vals))
    return new_labels, n_changed


def repair_connectivity(
    labels: np.ndarray,
    n_grains: int,
    periodic: list[bool],
    max_passes: int = _MAX_REPAIR_PASSES,
) -> tuple[np.ndarray, float]:
    """G5 repair-and-report for ``perturbed_distance`` (§6.6 extension):
    unlike ``voxel.check_connectivity`` (hard TessellationError on any
    macroscopic fragment), disconnected fragments are iteratively
    reassigned to whichever neighboring grain shares the most contact
    (majority-vote Jacobi sweeps) until every grain is one periodically-
    connected component.

    Returns ``(repaired_labels, reassigned_fraction)`` where
    ``reassigned_fraction`` is the total voxel count changed across all
    passes divided by the total voxel count — a QA metric, not a
    pass/fail gate (recorded, never required).

    Raises TessellationError if repair does not converge within
    *max_passes* (pathological fragmentation — reduce amplitude) or if
    any grain has zero voxels (empty-cell guard, see
    :func:`_find_fragment_mask`).
    """
    labels = labels.copy()
    n_vox_total = labels.size
    total_reassigned = 0
    for _ in range(max_passes):
        fragment_mask = _find_fragment_mask(labels, n_grains, periodic)
        if not np.any(fragment_mask):
            return labels, total_reassigned / n_vox_total
        labels, n_changed = _majority_vote_reassign(labels, fragment_mask, periodic)
        total_reassigned += n_changed
        if n_changed == 0:
            # Isolated pocket(s) with no non-fragment neighbor this pass:
            # further sweeps cannot help (the same fragment_mask would be
            # rebuilt identically) — surface the same information a hard
            # gate would, rather than silently returning an unrepaired field.
            raise TessellationError(
                "perturbed_distance G5 repair stalled: "
                f"{int(fragment_mask.sum())} voxel(s) have no non-fragment "
                "neighbor to adopt (isolated pocket). Reduce amplitude or "
                "refine the voxel grid."
            )
    raise TessellationError(
        f"perturbed_distance G5 repair did not converge within "
        f"{max_passes} passes (pathological fragmentation — reduce "
        "amplitude)."
    )


# ---------------------------------------------------------------------------
# D_b box-counting estimate (G13-analogue, warn-only)
# ---------------------------------------------------------------------------


def box_count_dimension(mask: np.ndarray, eps_min_px: int = 3,
                        eps_ratio: int = 10,
                        eps_max_px: int | None = None) -> float | None:
    """2D box-counting roughness-index fit of a binary mask's PERIMETER,
    following the Braun et al. 2020 protocol (J. Appl. Phys. 128, 105102):
    boxes counted as those intersecting the
    boundary (a pixel with ≥1 opposite-value 4-neighbor), the exponent
    read from the slope of a ``log N(ε)`` vs. ``log(1/ε)`` linear fit
    over box sizes ``ε ∈ [eps_min_px, eps_max_px]``.

    ``eps_max_px``, when given explicitly, OVERRIDES ``eps_ratio``:
    :meth:`PerturbedDistanceTessellation.d_b_estimate`
    passes the synthesis band's own ``[l_min, l_max]`` converted to the
    section grid's pixel units, so the fit window covers exactly the
    scales the synthesized η field has spectral content at — fitting
    ``ε`` outside that band (finer than ``l_min``, coarser than
    ``l_max``) measures pixelation noise or finite-object-size cutoff,
    not the target self-affine law, and is excluded rather than
    silently averaged in. ``eps_max_px=None`` (the default) falls back to
    the original one-decade convention ``eps_max_px = eps_ratio ·
    eps_min_px`` — an arbitrary decade unrelated to any particular
    band, kept as the default for any caller (including this module's
    own direct unit tests) that does not have a synthesis band to
    anchor to.

    Returns ``None`` (caller reports "n/a") when *mask*'s OWN ARRAY
    EXTENT is too small for even the minimum box range (``2·eps_max_px``
    must fit inside ``min(mask.shape)``). This is a section-GRID-
    resolution guard, not a per-grain size exclusion: in
    :meth:`PerturbedDistanceTessellation.d_b_estimate`, every grain's
    mask for a given cross-section is cut from the SAME
    ``SECTION_GRID``-derived ``(n_u, n_v)`` array, so this check is
    identical for every grain in that call — it protects against an
    ``eps`` range that does not fit the CHOSEN SAMPLING RESOLUTION, not
    against a genuinely small grain (a small grain instead yields a
    small or empty `boundary` array, caught separately below). Braun et
    al.'s own minimum-grain-size criterion (their Eq. 2, a per-grain
    equivalent-diameter threshold, d_min = ε_max/0.4 with ε_min ≥ 3s) is
    a DIFFERENT check on a different quantity and is not implemented
    here — see the module-level ``d_b_estimate`` docstring and
    docs/physics.md §5b for how that criterion bears on what MD-typical
    grain sizes can and cannot resolve.
    """
    ny, nx = mask.shape
    if eps_max_px is None:
        eps_max_px = eps_ratio * eps_min_px
    if eps_min_px >= eps_max_px:
        return None  # degenerate/inverted band at this section's pixel scale
    if min(ny, nx) < 2 * eps_max_px:
        return None
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    if not np.any(boundary):
        return None  # grain fills (or misses) the whole section: no perimeter

    eps_list = np.arange(eps_min_px, eps_max_px + 1, 2)  # Braun "odd sizes" cadence
    counts = []
    for eps in eps_list:
        n_by = int(np.ceil(ny / eps))
        n_bx = int(np.ceil(nx / eps))
        cnt = 0
        for by in range(n_by):
            for bx in range(n_bx):
                if np.any(boundary[by * eps:(by + 1) * eps, bx * eps:(bx + 1) * eps]):
                    cnt += 1
        counts.append(cnt)
    log_inv_eps = np.log(1.0 / eps_list.astype(np.float64))
    log_n = np.log(np.asarray(counts, dtype=np.float64))
    slope = float(np.polyfit(log_inv_eps, log_n, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# PerturbedDistanceTessellation
# ---------------------------------------------------------------------------


class PerturbedDistanceTessellation(Tessellation):
    """Level-set / perturbed-distance tessellation (§6.6 extension, M6).

    grain_of(x) = argmin_i [ d_pbc(x, s_i) − A · η_i(x) ]

    See the module docstring for the full physics rationale, the guard
    set, and the field-sharing (graph-coloring) memory scheme.
    """

    def __init__(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
        amplitude: float,
        min_seed_distance: float,
        rng: np.random.Generator,
        grid_size: int | str = "auto",
        connectivity_check: bool = True,
        spectrum: str = "gaussian",
        hurst: float = 0.8,
        correlation_length: float = 15.0,
        l_min: float = 8.0,
        l_max: float = 60.0,
        amplitude_convention: str = "total_rms",
        reference_wavelength: float | None = None,
        *,
        memory_limit_bytes: float | None = None,
        memory_limit_source: str = "config",
    ) -> None:
        # §13: stamp the per-run memory budget FIRST, before
        # _repair_and_build_override (called near the end of this
        # constructor when connectivity_check=True) builds a voxel grid
        # (see tessellation/base.py's Tessellation.memory_limit_bytes
        # docstring). Only overwrite the class default when the caller
        # actually passed one, so direct construction (tests) keeps the
        # 16 GB class default.
        if memory_limit_bytes is not None:
            self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_source = memory_limit_source

        self._seeds = np.asarray(seeds, dtype=np.float64)
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        self._n = len(self._seeds)
        self._spectrum = spectrum
        self._hurst = hurst
        self._correlation_length = correlation_length
        self._l_min = l_min
        self._l_max = l_max
        self._amplitude_convention = amplitude_convention
        self._reference_wavelength = (l_max if reference_wavelength is None
                                      else float(reference_wavelength))

        # --- Amplitude-convention resolution (opt-in; the
        # default "total_rms" is BIT-IDENTICAL to every other call
        # site — this branch is simply never taken there). Everything
        # downstream (guard 1/2, field scaling `A·η_i`, the ETA_CLIP-based
        # candidate-search radius) consumes ``self._amplitude`` as the
        # TOTAL-RMS-equivalent amplitude regardless of which convention
        # the caller used — a single conversion here keeps every other
        # method in this class unaware the reference_wavelength
        # convention exists at all. See
        # ``tessellation.warp.reference_shell_kappa`` for the κ
        # definition and the exact/invertible mapping formula.
        # Deliberately the CLOSED-FORM (grid-independent) κ, not a grid-
        # exact discrete one: this runs before the grid is resolved
        # below, and matches (bit-for-bit) the same pre-flight formula
        # ``config/resolve.py`` Rule 29(d) already checked — so this
        # constructor's own guard 1 immediately below re-derives exactly
        # the same inequality resolve_config already verified, rather
        # than silently checking a slightly different number. Like
        # PERTURBED_DISTANCE_SAFETY's own bound, this is a permissive,
        # sufficient (not tight-to-the-realized-grid) pre-flight
        # conversion — guard 2 (exact seed-ownership, post-hoc) is the
        # real correctness backstop regardless of convention.
        self._kappa = 1.0
        if amplitude_convention == "reference_wavelength":
            from grainsmith.tessellation.warp import (
                amplitude_to_total_rms,
                reference_shell_kappa,
            )

            self._kappa = reference_shell_kappa(
                hurst, l_min, l_max, reference_wavelength)
            self._amplitude_reference: float | None = float(amplitude)
            self._amplitude = amplitude_to_total_rms(amplitude, self._kappa)
        else:
            self._amplitude_reference = None
            self._amplitude = float(amplitude)

        # --- Guard 1 (ConfigError, hard, pre-flight): seed containment.
        # A sufficient (not tight) bound — guard 2 below is the exact
        # check; see constants.PERTURBED_DISTANCE_SAFETY / ETA_CLIP.
        # Binds ``self._amplitude`` — the REALIZED total-RMS-equivalent
        # amplitude — so a convention switch can only make this check
        # STRICTER (κ ≤ 1 ⇒ amplitude/κ ≥ amplitude), never weaker.
        a_max = PERTURBED_DISTANCE_SAFETY * min_seed_distance / (2.0 * ETA_CLIP)
        if self._amplitude > a_max:
            if amplitude_convention == "reference_wavelength":
                raise ConfigError(
                    f"perturbed_distance amplitude={amplitude:.4g} Å under "
                    "amplitude_convention: reference_wavelength is "
                    f"equivalent to a total-RMS amplitude of "
                    f"{self._amplitude:.4g} Å (÷ κ={self._kappa:.4g}), "
                    f"which exceeds A_max = {PERTURBED_DISTANCE_SAFETY:g}·"
                    f"min_seed_distance/(2·ETA_CLIP) = {a_max:.4g} Å "
                    "(seed-containment guard, binding on the REALIZED "
                    "field amplitude regardless of convention). Reduce "
                    "amplitude, or increase min_seed_distance / reduce "
                    "grains.number."
                )
            raise ConfigError(
                f"perturbed_distance amplitude={self._amplitude:.4g} Å "
                f"exceeds A_max = {PERTURBED_DISTANCE_SAFETY:g}·"
                f"min_seed_distance/(2·ETA_CLIP) = {a_max:.4g} Å "
                "(seed-containment guard). Reduce boundaries.curved."
                "amplitude, or increase min_seed_distance / reduce "
                "grains.number."
            )
        self._a_max = a_max

        # --- Periodic replica layout (weighted.py convention: index =
        # block·N + gid) — shared candidate-restricted machinery for
        # grain_of / owns / margin.
        offsets, identity_block = _periodic_translations(self._L, self._periodic)
        self._rep_seeds = (
            self._seeds[None, :, :] + offsets[:, None, :]
        ).reshape(-1, 3)
        self._identity_block = identity_block
        self._tree = cKDTree(self._rep_seeds)
        self._home_tree = cKDTree(self._seeds) if not all(periodic) else None
        # Fully periodic boxes route grain_of through the home-block
        # KDTree with an explicit modulo wrap below (mirrors flat.py's
        # boxsize-KDTree shortcut); mixed/free axes fall back to the
        # replica tree directly since scipy's KDTree ``boxsize`` only
        # supports fully-periodic boxes.

        # --- Field allocation: graph coloring over a conflict graph that
        # is a proven superset of true adjacency for ANY amplitude up to
        # a_max (independent of the amplitude actually requested).
        from grainsmith.tessellation.flat import FlatTessellation

        flat_tess = FlatTessellation(self._seeds, self._L, self._periodic)
        flat_radii = np.array(
            [flat_tess.bounding_radius(i) for i in range(self._n)], dtype=np.float64
        )
        conflict_edges = build_conflict_graph(
            self._seeds, self._L, self._periodic, flat_radii, a_max
        )
        self._color_of_grain = greedy_color_grains(self._n, conflict_edges)
        n_colors = int(self._color_of_grain.max()) + 1 if self._n else 0
        self._n_colors = n_colors + 2  # cheap slack (a fixed margin
        # against the greedy heuristic being non-optimal on a particular
        # seed layout; two extra fields cost little and are never assigned
        # to a grain).

        # --- Grid resolution: same heuristic as WarpTessellation (§6.6),
        # ell = correlation_length for gaussian, l_min for self_affine
        # (the gradient/roughness of a band-limited power law is
        # dominated by its shortest wavelength).
        ell = correlation_length if spectrum == "gaussian" else l_min
        L_min = float(np.min(self._L))
        if grid_size == "auto":
            n_short = max(16, int(np.ceil(L_min / (ell / 4.0))))
        else:
            n_short = int(grid_size)
        n_short = min(n_short, VOXEL_GRID_MAX)
        h_t = L_min / n_short
        nx, ny, nz = (int(np.ceil(self._L[ax] / h_t)) for ax in range(3))
        if max(nx, ny, nz) > VOXEL_GRID_MAX:
            log.warning(
                f"PerturbedDistanceTessellation: grid {(nx, ny, nz)} "
                f"clamped to ≤ {VOXEL_GRID_MAX} per axis; field resolution "
                "is coarser than the ℓ/4 (or l_min/4) target."
            )
            nx, ny, nz = (min(g, VOXEL_GRID_MAX) for g in (nx, ny, nz))
        self._grid_shape: tuple[int, int, int] = (nx, ny, nz)

        if spectrum == "self_affine":
            h_field = float(max(self._L[ax] / self._grid_shape[ax] for ax in range(3)))
            if l_min < 2.0 * h_field:
                raise ConfigError(
                    f"perturbed_distance self_affine l_min={l_min:g} Å is "
                    f"below 2·h_field={2.0 * h_field:g} Å (grid "
                    f"{self._grid_shape} cannot resolve it; Nyquist). "
                    "Increase l_min or refine the grid."
                )

        # --- Synthesize one clipped, unit-RMS scalar field per color.
        # ETA_CLIP is a fixed sup-norm bound (constants.py), NOT baked
        # into the stored field — A is applied at evaluation time, so the
        # candidate-search radius (2·A·ETA_CLIP) and the seed-containment
        # guard both reason about the SAME clip constant directly.
        # numpy types bit_generator.seed_seq as the ISeedSequence protocol;
        # the concrete object is always a SeedSequence, which carries the
        # entropy/spawn_key this stateless per-color derivation needs.
        seed_seq = cast("np.random.SeedSequence", rng.bit_generator.seed_seq)
        fields = np.empty((self._n_colors,) + self._grid_shape, dtype=np.float32)
        for c in range(self._n_colors):
            color_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                entropy=seed_seq.entropy, spawn_key=seed_seq.spawn_key + (c,),
            )))
            raw = synthesize_grf(
                self._grid_shape, correlation_length, self._L, color_rng,
                spectrum=spectrum, hurst=hurst, l_min=l_min, l_max=l_max,
                n_components=1,
            )[0]
            fields[c] = np.clip(raw, -ETA_CLIP, ETA_CLIP).astype(np.float32)
        self._fields = fields

        # --- G5 attribute placeholders MUST exist before the first
        # grain_of() call below (guard 2 uses it) — grain_of()/owns()
        # unconditionally check ``self._override_labels is None`` as
        # their fast-path no-op test. self._touched_mask is set only
        # alongside self._override_labels (see _repair_and_build_override)
        # and is never read otherwise.
        self._reassigned_fraction = 0.0
        self._override_labels: np.ndarray | None = None
        self._override_h: np.ndarray | None = None
        self._override_shape: tuple[int, int, int] | None = None
        self._voxel: VoxelGrid | None = None

        # --- Guard 2 (ConfigError, hard, POST-hoc, exact): every seed
        # must remain self-owned. This is the property guard 1 exists to
        # make plausible, checked directly rather than only inferred.
        # Deliberately evaluated BEFORE G5 repair: repair only touches
        # voxel-grid cells, never the seed points themselves, so the
        # ANALYTIC (pre-repair) seed-ownership property is the one that
        # must hold — checking it first also fails fast, before paying
        # for voxelization, on a config whose amplitude captures a seed
        # outright.
        g_of_seeds = self.grain_of(self._seeds)
        bad = np.where(g_of_seeds != np.arange(self._n))[0]
        if len(bad):
            raise ConfigError(
                f"perturbed_distance: {len(bad)} seed(s) (e.g. grain "
                f"{int(bad[0])}) are not self-owned after field synthesis "
                "— a competing grain's perturbation captured the seed "
                "point. Reduce boundaries.curved.amplitude or hurst, or "
                "increase min_seed_distance."
            )

        # --- Bounding radius (uniform, not per-grain-tightened — a
        # possible future performance optimization, not a correctness
        # concern): proven and empirically confirmed that every point
        # assigned to grain i lies within this radius of seed i.
        #
        # THE SLACK TERM MUST USE self._a_max, NOT self._amplitude.
        # This looks like an obvious tightening -- a_max is the config-LEGAL
        # CEILING (PERTURBED_DISTANCE_SAFETY·min_seed_distance/(2·ETA_CLIP)),
        # independent of the amplitude actually requested, so a run well
        # under the ceiling appears to pay slack it does not need -- and
        # fill_grain CUBES this radius to size its lattice grid, so the
        # apparent waste is large (on the 12-grain 1925 Å PdAu config:
        # 260.80 Å of slack instead of 110.45 Å, a 619³ grid instead of
        # 541³, 11.4 GB per grain instead of 7.6 GB).
        #
        # It is WRONG. Substituting self._amplitude UNDER-COVERS and
        # silently loses owned atoms. Measured on
        # examples/self_affine_gb/pdau_perturbed_self_affine.yaml
        # (20 grains, 200 Å, amplitude 3.6 Å):
        #     a_max slack      -> fill: 535,789 atoms
        #     amplitude slack  -> fill: 535,769 atoms   (20 LOST)
        # The final atom counts hide this (508,065 vs 508,068): the missing
        # atoms also remove overlap partners, so overlap removal deletes 23
        # fewer and the totals end up looking close. Only the FILL count
        # exposes it. See tests/test_perturbed.py::
        # test_bound_radius_uses_a_max_not_amplitude.
        #
        # Why 2·A·ETA_CLIP with the RUN's amplitude is not a bound here:
        # that quantity bounds the perturbation of the distance VALUE (it
        # is the correct search_pad in owns()/margin(), which ask "which
        # seeds could win at this point?"). This radius asks the different
        # question "how far from its seed can an owned point be?", whose
        # answer depends on how slowly d(x,s_i) - d(x,s_j) grows along the
        # outward direction -- for a near-parallel bisector the ownership
        # region reaches much further than the distance perturbation
        # itself. The a_max-based bound is the one this file proves and
        # that build_conflict_graph() above is consistent with (it also
        # takes a_max, deliberately "independent of the amplitude actually
        # requested").
        self._bound_radius = float(np.max(flat_radii)) + 2.0 * self._a_max * ETA_CLIP

        # --- G5: repair-and-report connectivity (not hard-fail).
        if connectivity_check:
            self._repair_and_build_override()

        self._adjacency: list[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _repair_and_build_override(self) -> None:
        """Voxelize the analytic assignment, repair periodic connectivity
        (majority-vote reassignment), and store the sparse override that
        grain_of/owns/margin consult for any query point whose voxel was
        touched by repair. Also builds the shared VoxelGrid (reused by
        the analysis stage's adjacency/volumes, matching warp/weighted's
        ``voxel_grid`` property contract)."""
        from grainsmith.tessellation.voxel import build_voxel_grid

        vg = build_voxel_grid(self, self._L, "auto", r_ws=None,
                              correlation_length=self._l_min
                              if self._spectrum == "self_affine"
                              else self._correlation_length)
        raw_labels = vg.labels
        repaired, frac = repair_connectivity(raw_labels, self._n, self._periodic)
        self._reassigned_fraction = frac
        diff = repaired != raw_labels
        if np.any(diff):
            self._override_labels = repaired
            self._override_h = vg.h_vec
            self._override_shape = vg.shape
            self._touched_mask = diff
            vg.labels = repaired  # keep the shared voxel grid consistent
        self._voxel = vg
        log.info(
            "PerturbedDistanceTessellation: G5 repair reassigned %.4f%% "
            "of voxels.", 100.0 * frac,
        )

    def _apply_override(self, X: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Overwrite *out* (already-computed analytic grain ids) with the
        repaired label wherever the query point's voxel was touched by
        G5 repair. No-op (returns *out* unchanged) when repair changed
        nothing or connectivity_check was disabled."""
        if self._override_labels is None:
            return out
        h = self._override_h
        shape = self._override_shape
        assert h is not None and shape is not None
        Xw = np.array(X, dtype=np.float64, copy=True)
        for ax in range(3):
            if self._periodic[ax]:
                Xw[:, ax] = np.mod(Xw[:, ax], self._L[ax])
            else:
                Xw[:, ax] = np.clip(Xw[:, ax], 0.0, self._L[ax])
        vidx = [
            np.clip(np.floor(Xw[:, ax] / h[ax]).astype(np.int64), 0, shape[ax] - 1)
            for ax in range(3)
        ]
        override_lab = self._override_labels[vidx[0], vidx[1], vidx[2]]
        # A voxel is "touched" iff its stored (post-repair) label differs
        # from what the ANALYTIC rule alone would give it. That mask is
        # captured once at construction time (self._touched_mask, built
        # in _repair_and_build_override from the repaired-vs-raw label
        # arrays) rather than recomputed per query.
        touched = self._touched_mask[vidx[0], vidx[1], vidx[2]]
        out = out.copy()
        out[touched] = override_lab[touched]
        return out

    # ------------------------------------------------------------------
    # Candidate-restricted core (shared by grain_of / owns / margin)
    # ------------------------------------------------------------------

    def _candidates_home(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidate (row, grain_id, perturbed_value) arrays restricted to
        HOME replicas (grain ids 0..n-1 under min-image), used by
        grain_of/margin. Exact: any grain farther than
        ``d_1 + 2·A·ETA_CLIP`` from the nearest seed cannot win, since
        ``|η| ≤ ETA_CLIP`` bounds the perturbation on every side of the
        comparison uniformly."""
        n_pts = len(X)
        row_all: list[np.ndarray] = []
        gid_all: list[np.ndarray] = []
        val_all: list[np.ndarray] = []
        seeds = self._seeds
        L = self._L
        periodic = self._periodic
        A = self._amplitude
        search_pad = 2.0 * A * ETA_CLIP + 1e-9
        for start in range(0, n_pts, _CHUNK):
            Xc = X[start:start + _CHUNK]
            if all(periodic):
                d1, _ = self._home_tree_query(Xc)
            else:
                diff = Xc[:, None, :] - seeds[None, :, :]
                for ax in range(3):
                    if periodic[ax]:
                        diff[..., ax] -= np.round(diff[..., ax] / L[ax]) * L[ax]
                d1 = np.min(np.linalg.norm(diff, axis=2), axis=1)
            radius = d1 + search_pad
            cand = self._tree.query_ball_point(Xc, r=radius)
            lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(Xc))
            row = np.repeat(np.arange(start, start + len(Xc)), lens)
            col = np.concatenate(cand).astype(np.int64) if len(row) else np.empty(0, dtype=np.int64)
            gid = col % self._n
            dr = X[row] - self._rep_seeds[col]
            draw = np.linalg.norm(dr, axis=1)
            eta = _eta_at(gid, X[row], self._color_of_grain, self._fields,
                          L, periodic, self._grid_shape)
            val = draw - A * eta
            row_all.append(row)
            gid_all.append(gid)
            val_all.append(val)
        return (
            np.concatenate(row_all) if row_all else np.empty(0, dtype=np.int64),
            np.concatenate(gid_all) if gid_all else np.empty(0, dtype=np.int64),
            np.concatenate(val_all) if val_all else np.empty(0, dtype=np.float64),
        )

    def _home_tree_query(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-seed distance under full periodicity via a boxsize
        KDTree (fast path; mirrors flat.py's ``_home_tree``)."""
        if self._home_tree is None:
            seeds_wrapped = self._seeds % self._L
            seeds_wrapped[seeds_wrapped >= self._L] = 0.0
            self._home_tree = cKDTree(seeds_wrapped, boxsize=self._L)
        Xw = X % self._L
        Xw[Xw >= self._L] = 0.0
        return self._home_tree.query(Xw)

    @staticmethod
    def _offsets_and_col(cand: list) -> tuple[np.ndarray, np.ndarray]:
        """CSR-style ``(offsets, col_flat)`` pair from scipy's ragged
        ``query_ball_point`` output, in THE SAME per-point order it was
        returned — load-bearing for the tie-break contract:
        ``col_flat[offsets[p]:offsets[p+1]]`` is point *p*'s candidate
        slice, visited by the numba kernels in exactly this order. Used
        by all three numba dispatch paths (K1/K2/K3) in place of the
        NumPy reference's ``row``/``col`` arrays."""
        lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
        offsets = np.empty(len(cand) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lens, out=offsets[1:])
        total = int(offsets[-1]) if len(offsets) else 0
        if total == 0:
            col_flat = np.empty(0, dtype=np.int64)
        else:
            col_flat = np.concatenate(cand).astype(np.int64)
        return offsets, col_flat

    def _grain_of_numba(self, X: np.ndarray) -> np.ndarray:
        """K3 dispatch: same candidate-restriction anchor
        (``_home_tree_query`` / direct diff-based ``d1``, unchanged scipy)
        as ``_candidates_home``, but the post-query evaluation is the
        fused ``_grain_of_kernel_perturbed`` call instead of building
        (row, gid, val) temporaries for ``_segment_argmin``."""
        n_pts = len(X)
        out = np.empty(n_pts, dtype=np.int32)
        if n_pts == 0:
            return out
        seeds = self._seeds
        L = self._L
        periodic = self._periodic
        A = self._amplitude
        search_pad = 2.0 * A * ETA_CLIP + 1e-9
        Nx, Ny, Nz = self._grid_shape
        per_x, per_y, per_z = periodic
        Lx, Ly, Lz = L
        for start in range(0, n_pts, _CHUNK):
            Xc = np.ascontiguousarray(X[start:start + _CHUNK])
            if all(periodic):
                d1, _ = self._home_tree_query(Xc)
            else:
                diff = Xc[:, None, :] - seeds[None, :, :]
                for ax in range(3):
                    if periodic[ax]:
                        diff[..., ax] -= np.round(diff[..., ax] / L[ax]) * L[ax]
                d1 = np.min(np.linalg.norm(diff, axis=2), axis=1)
            radius = d1 + search_pad
            cand = self._tree.query_ball_point(Xc, r=radius)
            offsets, col_flat = self._offsets_and_col(cand)
            out[start:start + len(Xc)] = _grain_of_kernel_perturbed(
                Xc, offsets, col_flat, self._rep_seeds, self._color_of_grain,
                self._fields, Nx, Ny, Nz, Lx, Ly, Lz, per_x, per_y, per_z,
                A, self._n)
        return out

    @staticmethod
    def _segment_argmin(row: np.ndarray, key: np.ndarray, payload: np.ndarray,
                        n_rows: int) -> np.ndarray:
        """Per-row argmin of *key*, returning the winning *payload* value
        for each of *n_rows* rows (rows with no candidates are left at
        the caller's own sentinel — never occurs here since a point's own
        nearest replica is always within its own search radius)."""
        order = np.lexsort((key, row))
        row_sorted = row[order]
        first = np.empty(len(order), dtype=bool)
        if len(order):
            first[0] = True
            first[1:] = row_sorted[1:] != row_sorted[:-1]
        out = np.empty(n_rows, dtype=payload.dtype)
        out[row_sorted[first]] = payload[order[first]]
        return out

    # ------------------------------------------------------------------
    # Tessellation ABC
    # ------------------------------------------------------------------

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        if _USE_NUMBA and _grain_of_kernel_perturbed is not None:
            out = self._grain_of_numba(X)
        else:
            row, gid, val = self._candidates_home(X)
            out = self._segment_argmin(row, val, gid, len(X)).astype(np.int32)
        if self._override_labels is not None:
            out = self._apply_override(X, out)
        return out

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact home-cell ownership (§6.8) via the full periodic
        replica set (weighted.py convention) — exact tiling for the atom
        fill, validated against a dense brute-force reference including
        out-of-box query points (needed since ``fill_grain`` evaluates
        unwrapped lattice points that can extend past the box edge)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        n_pts = len(X)
        if n_pts == 0:
            return np.zeros(0, dtype=bool)
        A = self._amplitude
        search_pad = 2.0 * A * ETA_CLIP + 1e-9
        home_idx = self._identity_block * self._n + i

        if _USE_NUMBA and _owns_kernel_perturbed is not None:
            out = np.empty(n_pts, dtype=bool)
            Nx, Ny, Nz = self._grid_shape
            per_x, per_y, per_z = self._periodic
            Lx, Ly, Lz = self._L
            for start in range(0, n_pts, _CHUNK):
                Xc = np.ascontiguousarray(X[start:start + _CHUNK])
                d1, _ = self._tree.query(Xc, k=1)
                radius = d1 + search_pad
                cand = self._tree.query_ball_point(Xc, r=radius)
                offsets, col_flat = self._offsets_and_col(cand)
                out[start:start + len(Xc)] = _owns_kernel_perturbed(
                    Xc, offsets, col_flat, self._rep_seeds,
                    self._color_of_grain, self._fields, Nx, Ny, Nz,
                    Lx, Ly, Lz, per_x, per_y, per_z, A, self._n, home_idx)
            if self._override_labels is not None:
                out = self._apply_override_owns(X, i, out)
            return out

        row_all: list[np.ndarray] = []
        replica_all: list[np.ndarray] = []
        val_all: list[np.ndarray] = []
        for start in range(0, n_pts, _CHUNK):
            Xc = X[start:start + _CHUNK]
            d1, _ = self._tree.query(Xc, k=1)
            radius = d1 + search_pad
            cand = self._tree.query_ball_point(Xc, r=radius)
            lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(Xc))
            row = np.repeat(np.arange(start, start + len(Xc)), lens)
            col = np.concatenate(cand).astype(np.int64) if len(row) else np.empty(0, dtype=np.int64)
            gid = col % self._n
            dr = X[row] - self._rep_seeds[col]
            draw = np.linalg.norm(dr, axis=1)
            eta = _eta_at(gid, X[row], self._color_of_grain, self._fields,
                          self._L, self._periodic, self._grid_shape)
            val = draw - A * eta
            row_all.append(row)
            replica_all.append(col)
            val_all.append(val)
        row_cat = np.concatenate(row_all) if row_all else np.empty(0, dtype=np.int64)
        replica_cat = np.concatenate(replica_all) if replica_all else np.empty(0, dtype=np.int64)
        val_cat = np.concatenate(val_all) if val_all else np.empty(0, dtype=np.float64)
        winning_replica = self._segment_argmin(row_cat, val_cat, replica_cat, n_pts)
        out = winning_replica == home_idx
        if self._override_labels is not None:
            out = self._apply_override_owns(X, i, out)
        return out

    def _apply_override_owns(self, X: np.ndarray, i: int, out: np.ndarray) -> np.ndarray:
        """owns() override consistent with the grain_of() override: a
        point in a repaired voxel is owned by grain i iff the repaired
        label at that voxel is i (box-covering voxel grid = the identity
        replica's domain for the periodic axes G5 repair operates over)."""
        assert (self._override_h is not None
                and self._override_shape is not None
                and self._override_labels is not None)
        Xw = np.array(X, dtype=np.float64, copy=True)
        for ax in range(3):
            if self._periodic[ax]:
                Xw[:, ax] = np.mod(Xw[:, ax], self._L[ax])
            else:
                Xw[:, ax] = np.clip(Xw[:, ax], 0.0, self._L[ax])
        h = self._override_h
        shape = self._override_shape
        vidx = [
            np.clip(np.floor(Xw[:, ax] / h[ax]).astype(np.int64), 0, shape[ax] - 1)
            for ax in range(3)
        ]
        touched = self._touched_mask[vidx[0], vidx[1], vidx[2]]
        override_lab = self._override_labels[vidx[0], vidx[1], vidx[2]]
        out = out.copy()
        out[touched] = override_lab[touched] == i
        return out

    def _d1_excluding(self, X: np.ndarray, i: int) -> np.ndarray:
        """Nearest-REPLICA distance EXCLUDING grain i's own replicas —
        the correct search-radius anchor for margin()'s "best competing
        grain" query (``_candidates_home``'s GLOBAL-nearest anchor is
        provably safe for grain_of/owns's global argmin, but NOT for a
        query that must exclude one specific grain: when i itself is the
        global nearest, the true best competitor can sit farther than
        ``d1_global + 2·A·ETA_CLIP``).

        Pigeonhole argument: grain i owns exactly ``n_blocks`` replicas
        (one per periodic shift), so among the ``n_blocks+1`` nearest
        replicas of any query point, at least one belongs to a DIFFERENT
        grain — validated against a dense brute-force ``d1_excl`` to
        float64 roundoff.

        Ascending k-escalation (k=2, then 4, then the full pigeonhole
        bound n_blocks+1, each capped to the available replica count):
        cKDTree.query returns ``dists`` in ascending order, so for any k
        that contains at least one non-i replica, the first such column
        IS the true nearest non-i replica — a small k either already
        contains that replica (cheap answer, exact) or doesn't (escalate
        to a larger k), there is no k for which a small query could return
        a value different from the k=n_blocks+1 query once both resolve a
        row. Only rows still unresolved (``todo``) are re-queried at each
        step, so the escalation changes cost, never which replica wins.

        Ties cannot change the float64 result either: it is a SET
        MINIMUM, min{d(x, S_k) : gid(k) != i}, and cKDTree computes every
        distance as sqrt of the same squared-distance formula regardless
        of k or query batch size — no new arithmetic, no reduction-order
        dependence.
        * i-replica tied with a non-i replica (same distance, different
          rank in the returned order): whichever k first exposes a non-i
          column, ``first_col`` selects that non-i replica; its distance
          equals the tied i-replica's distance to float64 bit-for-bit, so
          the returned value is the same regardless of which k resolved
          the row.
        * two non-i replicas tied (e.g. a point exactly on their
          bisector): both carry the identical sqrt(d²) value, so whichever
          of the two lands in the first non-i column is irrelevant — the
          returned float64 is the same value either way.
        * a box-corner point equidistant from up to 8 replicas: same
          argument per pair — every tied replica shares one float64
          distance, so the escalation can only change WHICH tied replica
          is read, never the value read, and a non-i replica is picked
          over an i-replica the instant one appears in the returned
          columns at any k.
        Rows where every column at every escalation step (up to and
        including the full pigeonhole bound) belongs to grain i are
        genuinely unresolvable and still raise ``TessellationError`` with
        the unchanged message — same degenerate-system contract as the
        single k=n_blocks+1 query this replaces."""
        n_rep = len(self._rep_seeds)
        n_blocks = n_rep // self._n
        n_pts = len(X)
        out = np.empty(n_pts, dtype=np.float64)
        todo = np.arange(n_pts)
        # Every step is capped by the pigeonhole bound (n_blocks + 1) as
        # well as the available replica count: querying past n_blocks + 1
        # neighbours can never expose a different competing-grain replica
        # than the frozen k=n_blocks+1 reference already would (see the
        # escalation argument above), so no step should ever query more
        # neighbours than that reference does. The (2, 4, n_blocks+1)
        # sequence is non-decreasing before clipping, so clipping can only
        # introduce CONSECUTIVE duplicate k_eff values (e.g. n_blocks=3
        # makes both the k=4 and k=n_blocks+1 steps clip to 4) -- drop
        # those so a step is never repeated.
        k_steps: list[int] = []
        for k in (2, 4, n_blocks + 1):
            k_eff = min(k, n_blocks + 1, n_rep)
            if not k_steps or k_steps[-1] != k_eff:
                k_steps.append(k_eff)
        for k_eff in k_steps:
            # todo == arange(n_pts) on the first step, and on any later
            # step where nothing has resolved yet -- use X directly rather
            # than pay for a fancy-index copy that would just reproduce X.
            X_todo = X if len(todo) == n_pts else X[todo]
            dists, idxs = self._tree.query(X_todo, k=k_eff)
            if k_eff == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            grain_ids_k = idxs % self._n
            not_i = grain_ids_k != i
            ok = np.any(not_i, axis=1)
            first_col = np.argmax(not_i, axis=1)
            resolved = dists[np.arange(len(todo)), first_col]
            out[todo[ok]] = resolved[ok]
            todo = todo[~ok]
            if len(todo) == 0:
                break
        else:
            raise TessellationError(
                f"perturbed_distance margin(): no competing grain found "
                f"for grain {i} within its own replica count nearest "
                "neighbors — degenerate system (e.g. n_grains=1, which "
                "has no grain boundary to measure margin against)."
            )
        return out

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed boundary distance for grain i: half the perturbed-
        distance deficit to the nearest COMPETING grain, matching
        weighted.py's convention (Å-scaled, positive inside, zero on the
        boundary) — validated against dense brute force to float64
        roundoff, INCLUDING the case where grain i is the globally
        nearest seed (see :meth:`_d1_excluding`)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        n_pts = len(X)
        if n_pts == 0:
            return np.empty(0, dtype=np.float64)

        A = self._amplitude
        search_pad = 2.0 * A * ETA_CLIP + 1e-9

        if _USE_NUMBA and _margin_kernel_perturbed is not None:
            out = np.empty(n_pts, dtype=np.float64)
            Nx, Ny, Nz = self._grid_shape
            per_x, per_y, per_z = self._periodic
            Lx, Ly, Lz = self._L
            for start in range(0, n_pts, _CHUNK):
                Xc = np.ascontiguousarray(X[start:start + _CHUNK])
                d1_excl = self._d1_excluding(Xc, i)
                radius = d1_excl + search_pad
                cand = self._tree.query_ball_point(Xc, r=radius)
                offsets, col_flat = self._offsets_and_col(cand)
                out[start:start + len(Xc)] = _margin_kernel_perturbed(
                    Xc, offsets, col_flat, self._seeds, self._rep_seeds,
                    self._color_of_grain, self._fields, Nx, Ny, Nz,
                    Lx, Ly, Lz, per_x, per_y, per_z, A, self._n, i)
            return out

        row_all: list[np.ndarray] = []
        gid_all: list[np.ndarray] = []
        val_all: list[np.ndarray] = []
        for start in range(0, n_pts, _CHUNK):
            Xc = X[start:start + _CHUNK]
            d1_excl = self._d1_excluding(Xc, i)
            radius = d1_excl + search_pad
            cand = self._tree.query_ball_point(Xc, r=radius)
            lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(Xc))
            row = np.repeat(np.arange(start, start + len(Xc)), lens)
            col = np.concatenate(cand).astype(np.int64) if len(row) else np.empty(0, dtype=np.int64)
            gid = col % self._n
            dr = X[row] - self._rep_seeds[col]
            draw = np.linalg.norm(dr, axis=1)
            eta = _eta_at(gid, X[row], self._color_of_grain, self._fields,
                          self._L, self._periodic, self._grid_shape)
            row_all.append(row)
            gid_all.append(gid)
            val_all.append(draw - A * eta)
        row_cat = np.concatenate(row_all) if row_all else np.empty(0, dtype=np.int64)
        gid_cat = np.concatenate(gid_all) if gid_all else np.empty(0, dtype=np.int64)
        val_cat = np.concatenate(val_all) if val_all else np.empty(0, dtype=np.float64)
        val_other_masked = np.where(gid_cat == i, np.inf, val_cat)
        val_other = self._segment_argmin(row_cat, val_other_masked, val_other_masked, n_pts)

        # val_i: direct single-grain evaluation (grain i is always its
        # own candidate within its own search radius by construction).
        seeds = self._seeds
        L = self._L
        periodic = self._periodic
        diff_i = X - seeds[i][None, :]
        for ax in range(3):
            if periodic[ax]:
                diff_i[:, ax] -= np.round(diff_i[:, ax] / L[ax]) * L[ax]
        d_i = np.linalg.norm(diff_i, axis=1)
        eta_i = _eta_at(np.full(n_pts, i, dtype=np.int64), X,
                        self._color_of_grain, self._fields, L, periodic,
                        self._grid_shape)
        val_i = d_i - self._amplitude * eta_i
        return (val_other - val_i) / 2.0

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> None:
        """No certified GB-shell bound: the perturbed generalized distance
        ``val = draw - A * eta`` (module-level ``margin()``/
        ``_d1_excluding`` above) has Lipschitz constant ``1 + A * |grad
        eta|`` in ``x`` -- ``> 1`` whenever the perturbation amplitude
        ``A`` and field gradient are nonzero, which is the entire point of
        this backend. A Lipschitz-1 margin filter would not certify a
        superset of the true pair endpoints for this distance. Always
        returns ``None`` (unchanged full-N overlap behaviour)."""
        return None

    def adjacency(self) -> list[tuple[int, int]]:
        """Adjacency measured on the actual (post-repair) curved diagram
        via the shared voxel grid — matching weighted.py's convention
        (a flat-Voronoi proxy can mis-report curved adjacency)."""
        if self._adjacency is None:
            if self._voxel is None:
                from grainsmith.tessellation.voxel import build_voxel_grid
                self._voxel = build_voxel_grid(self, self._L, "auto")
            assert self._voxel is not None  # just built above if it was None
            self._adjacency = self._voxel.adjacency(self._periodic)
        return list(self._adjacency)

    def bounding_radius(self, i: int) -> float:
        return self._bound_radius

    @property
    def seeds(self) -> np.ndarray:
        return self._seeds.copy()

    @property
    def n_grains(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # perturbed_distance-specific accessors
    # ------------------------------------------------------------------

    @property
    def reassigned_fraction(self) -> float:
        """Fraction of analysis-voxel-grid voxels reassigned by G5 repair
        (0.0 when connectivity_check=False or nothing needed repair)."""
        return self._reassigned_fraction

    @property
    def n_colors(self) -> int:
        """Number of distinct scalar fields synthesized (graph-coloring
        result + a fixed 2-color slack)."""
        return self._n_colors

    @property
    def color_of_grain(self) -> np.ndarray:
        return self._color_of_grain.copy()

    @property
    def spectrum(self) -> str:
        return self._spectrum

    @property
    def amplitude_convention(self) -> str:
        """'total_rms' (default) or 'reference_wavelength' — see
        ``boundaries.curved.amplitude_convention``."""
        return self._amplitude_convention

    @property
    def reference_wavelength(self) -> float:
        """Resolved reference-octave anchor wavelength (Å); equals
        ``l_max`` when the config left it unset. Meaningful only under
        ``amplitude_convention: reference_wavelength``, but always
        resolved to a concrete value (never None) for reporting."""
        return self._reference_wavelength

    @property
    def kappa(self) -> float:
        """κ = σ(reference octave)/σ(whole band) — the closed-form
        conversion factor actually used to resolve this instance's
        total-RMS-equivalent amplitude (``tessellation.warp.
        reference_shell_kappa``). Exactly 1.0 under
        ``amplitude_convention: total_rms`` (no conversion applied)."""
        return self._kappa

    @property
    def amplitude_reference(self) -> float | None:
        """The config-file ``amplitude`` value AS GIVEN under
        ``amplitude_convention: reference_wavelength`` (Å-RMS at
        ``reference_wavelength``); ``None`` under ``total_rms`` (no
        separate reference-octave amplitude exists in that
        convention — ``amplitude`` below already IS a total-RMS
        value)."""
        return self._amplitude_reference

    @property
    def amplitude_total_rms(self) -> float:
        """The REALIZED total-RMS-equivalent amplitude (Å) — what
        actually multiplies the unit-RMS synthesized field
        (``d_i − amplitude_total_rms · η_i``) and what the
        seed-containment guard (``a_max``) binds, in EITHER convention.
        Equals the config's ``amplitude`` field verbatim under
        ``total_rms``; equals ``amplitude_reference / kappa`` under
        ``reference_wavelength``."""
        return self._amplitude

    @property
    def fields(self) -> np.ndarray:
        """The synthesized (n_colors, Nx, Ny, Nz) per-color scalar η
        fields — the perturbed_distance analogue of
        ``WarpTessellation.field`` (that property exposes ONE 3-component
        vector field; this exposes ``n_colors`` independent scalar
        fields). Public accessor for external verification (e.g. an
        independent PSD/Hurst check), mirroring the existing warp-side
        pattern rather than reaching into ``self._fields`` directly."""
        return self._fields

    def hurst_estimate(self) -> float:
        """Back-estimated Hurst exponent of the synthesized per-color η
        fields (G13; owned here rather than by WarpTessellation — see
        ``tessellation/warp.py``'s module docstring for why warp does not
        accept spectrum: self_affine at all, and hence does not own G13).
        Only meaningful for the self_affine spectrum.

        Uses the same :func:`grainsmith.tessellation.warp.estimate_hurst`
        3D-PSD fit warp used, applied to ``self._fields`` (n_colors
        components instead of warp's 3) — a larger ensemble of
        independent realizations of the SAME target spectral law, not a
        different estimator (see ``estimate_hurst``'s docstring).
        """
        if self._spectrum != "self_affine":
            raise TessellationError(
                "hurst_estimate() is defined for the self_affine spectrum "
                "only.")
        from grainsmith.tessellation.warp import estimate_hurst

        return estimate_hurst(self._fields, self._L, self._l_min,
                              self._l_max,
                              memory_limit_bytes=self.memory_limit_bytes,
                              memory_limit_source=self.memory_limit_source)

    @property
    def voxel_grid(self):
        """VoxelGrid built for the G5 repair pass (None only if
        connectivity_check=False); reusable by the analysis stage,
        matching warp/weighted's ``voxel_grid`` property contract."""
        return self._voxel

    def d_b_estimate(self, k_largest: int = 5, n_sections: int = 3) -> float | None:
        """Box-counting roughness-index estimate (gate **G20** — a
        warn-only DIAGNOSTIC, not a fractal-dimension certificate; see
        :func:`box_count_dimension` and docs/physics.md §5b), averaged
        over up to *n_sections* in-box cross-sections (mid-plane
        sections along each periodic axis present, capped at
        *n_sections*) of the *k_largest* largest grains by voxel count.

        Fit window: box sizes are restricted to the
        SYNTHESIS BAND itself, ``ε ∈ [l_min, l_max]`` (converted to this
        section's own pixel scale ``h`` — see below), rather than an
        arbitrary fixed-pixel decade. A box size finer than ``l_min`` or
        coarser than ``l_max`` probes a scale the η field has no
        designed spectral content at (finer: dominated by voxel/section
        pixelation noise; coarser: dominated by the finite grain/section
        extent, not the target power law), so admitting it would let an
        out-of-band artifact bias the reported exponent. Falls back to
        ``box_count_dimension``'s original fixed-decade default only for
        the (pipeline-unreachable — ``pipeline.py`` only calls this
        method when ``spectrum == "self_affine"``) ``gaussian``-spectrum
        case, which has no ``[l_min, l_max]`` band to anchor to.

        Why this is a roughness INDEX, not a fractal-dimension estimate
        -----------------------------------------------------------------
        At MD-typical grain sizes (≲50 nm), the synthesis band spans too
        few octaves for box-counting to separate the Hurst exponent's
        effect from finite-band/finite-size noise at a LARGE effect
        size — this is a GRADIENT with octave count, not a sharp cliff.
        Measured: η² ≈ 5–7% (not distinguishable from noise, p ≈ 0.14)
        at the ~1.7-octave band
        ``examples/self_affine_gb/pdau_perturbed_self_affine.yaml`` uses; a
        synthetic-field Monte Carlo shows the effect already
        statistically SIGNIFICANT (p ≪ 0.01) by ~3.2 octaves, but only
        reaching a conventionally LARGE effect size (η² > 80%) at ≈50 nm
        grain diameter / ~5.5 octaves — significance and effect-size
        magnitude cross their respective thresholds at different octave
        counts (see docs/physics.md §5b(g) for the full octave/η²
        table). This mirrors Braun et al. 2020's own protocol constraint
        that a usable box-counting range needs the analogous ε_min ≥ 3s
        floor and an order-of-magnitude ε span.
        Below the ~5.5-octave regime, ``d_b_estimated`` is honest
        evidence of LOCAL boundary roughness at the resolved scales, not
        a converged, large-effect-size fractal dimension comparable to
        Braun's D_b = 1.174 ± 0.004 — see docs/physics.md §5b for the
        full argument. G20 is warn-only and reports exactly this number
        without editorializing on it; the caveat belongs in the gate
        message/docs, not a threshold this method enforces.

        Returns ``None`` when no (grain, section) combination yields a
        usable estimate (e.g. every sampled grain is too small relative
        to the section resolution, or the band collapses at this
        section's pixel scale) — reported as "n/a", never as a
        fabricated number.
        """
        if self._voxel is None:
            return None
        vol = self._voxel.volumes()
        k = min(k_largest, self._n)
        largest = np.argsort(-vol)[:k]

        from grainsmith.constants import SECTION_GRID

        axes_present = [ax for ax in range(3) if self._periodic[ax]] or [0, 1, 2]
        section_axes = axes_present[:n_sections]

        estimates: list[float] = []
        for ax in section_axes:
            u_ax, v_ax = (k2 for k2 in range(3) if k2 != ax)
            n_short = SECTION_GRID
            h = float(min(self._L[u_ax], self._L[v_ax])) / n_short
            n_u = int(np.ceil(self._L[u_ax] / h))
            n_v = int(np.ceil(self._L[v_ax] / h))
            u = (np.arange(n_u) + 0.5) * (self._L[u_ax] / n_u)
            v = (np.arange(n_v) + 0.5) * (self._L[v_ax] / n_v)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            pts = np.empty((n_u * n_v, 3), dtype=np.float64)
            pts[:, u_ax] = uu.ravel()
            pts[:, v_ax] = vv.ravel()
            pts[:, ax] = 0.5 * self._L[ax]
            gid = self.grain_of(pts).reshape(n_u, n_v)
            # Band-limited fit window: convert the synthesis band's own
            # [l_min, l_max] (Å) to THIS section's pixel units (h Å/px)
            # -- h varies section to section only through floating-point
            # rounding of n_u/n_v, so this is effectively one shared
            # window, computed per-section for exactness.
            if self._spectrum == "self_affine":
                eps_min_px = max(1, int(round(self._l_min / h)))
                eps_max_px = max(eps_min_px + 1, int(round(self._l_max / h)))
            else:
                eps_min_px, eps_max_px = 3, None
            for g in largest:
                mask = gid == g
                if not np.any(mask):
                    continue
                db = box_count_dimension(mask, eps_min_px=eps_min_px,
                                         eps_max_px=eps_max_px)
                if db is not None:
                    estimates.append(db)
        return float(np.mean(estimates)) if estimates else None
