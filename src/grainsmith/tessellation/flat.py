"""Flat Voronoi tessellation via scipy.spatial.Voronoi (Qhull).

Algorithm (§6.5):
1. Replicate seeds along each axis:
   - periodic axis  → translated images, shifts {-1, 0, +1}·L;
   - free axis      → MIRROR images reflected across the two box walls
                      (c → -c and c → 2L - c).
   For any point x inside the box and any seed s, a wall mirror is never
   closer than the unreflected seed (d² grows by 4·x_ax·s_ax ≥ 0 at the low
   wall, 4·(L-x_ax)(L-s_ax) ≥ 0 at the high wall), so the interior cell
   structure is untouched while every home cell is clipped EXACTLY at the
   wall planes — the bisector between a seed and its own mirror IS the wall.
   This implements §6.5 step 4 (box-wall half-space clipping) without
   explicit clipping code.
2. scipy.spatial.Voronoi on the replica set (float64; Qhull is robust —
   replaces the legacy MNN cap and the 2.8·r_ws cutoff entirely).
3. For each home seed: collect ridges, order each ridge polygon cyclically
   in its plane (Qhull does NOT guarantee cyclic vertex order in 3D),
   build FlatCell faces (area, outward unit normal) and the cell volume
   via the divergence theorem.
4. Gates enforced on construction:
   - G3: Σ cell volumes == box volume, rel. err ≤ VOL_REL_TOL (hard);
   - G4: per-cell Euler relation V − E + F = 2 with vertex deduplication
         tolerance EULER_TOL × max(L) (hard).

Face taxonomy:
   neighbor_id >= 0  → grain-boundary face (neighbor_id == grain_id is a
                       legitimate GB with the grain's own periodic image);
   neighbor_id == -1 → box-wall face (free surface; slab geometry only).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product as iproduct

import numpy as np
from scipy.spatial import KDTree, Voronoi

from grainsmith.constants import EULER_TOL, OWNS_TIE_TOL, VOL_REL_TOL
from grainsmith.errors import TessellationError
from grainsmith.tessellation._local_owns import local_owns
from grainsmith.tessellation._shell_knn import SHELL_KNN_K as _SHELL_KNN_K
from grainsmith.tessellation._shell_knn import certified_knn_shell_mask
from grainsmith.tessellation.base import Tessellation

WALL_NEIGHBOR_ID: int = -1
"""Sentinel neighbor_id for box-wall (free-surface) faces."""

_OWNS_CHUNK: int = 50_000
"""Row-chunk size for the vectorised owns() inner loop.

Bounds the peak (rows × deg) working array at ~50 000 × ~20 × 8 bytes ≈ 8 MB
per chunk regardless of the total fill size (mirrors weighted.py's _CHUNK)."""


@dataclass
class FlatFace:
    neighbor_id: int         # neighboring grain id; WALL_NEIGHBOR_ID for box wall
    vertices: np.ndarray     # (M, 3) float64 vertex positions, cyclically ordered
    area: float              # face area in Å²
    unit_normal: np.ndarray  # (3,) outward unit normal from this grain
    seed_distance: float = 0.0  # ‖replica seed_j − seed_i‖ (Å); 0 for walls.
    #                             Drives the SDOT Hessian ∂V_i/∂w_j =
    #                             −A_ij/(2 d_ij) (power diagrams).


def replicate_seeds(
    seeds: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
) -> tuple[np.ndarray, int, np.ndarray]:
    """Replicate seeds: translations on periodic axes, wall MIRRORS on free
    axes (module docstring step 1).  Shared by the flat-Voronoi and power
    (Laguerre) backends — mirrors carry the same weight as their source, so
    the exact-wall-clipping proof holds for both diagrams.

    Returns (points (B·N, 3), identity_block_index, is_mirror flags);
    replica layout: index = block·N + gid, so index % N == gid.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64)
    n = len(seeds)
    # Per-axis affine options (a, b): coordinate c → a·c + b
    options: list[list[tuple[float, float]]] = []
    for ax in range(3):
        if periodic[ax]:
            options.append([(1.0, -L[ax]), (1.0, 0.0), (1.0, +L[ax])])
        else:
            options.append([(-1.0, 0.0), (1.0, 0.0), (-1.0, 2.0 * L[ax])])

    blocks = list(iproduct(*options))
    identity = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    identity_block = blocks.index(identity)

    all_pts = np.empty((len(blocks) * n, 3), dtype=np.float64)
    is_mirror = np.zeros(len(blocks) * n, dtype=bool)
    for b, opts in enumerate(blocks):
        sl = slice(b * n, (b + 1) * n)
        for ax, (a, off) in enumerate(opts):
            all_pts[sl, ax] = a * seeds[:, ax] + off
        if any(a < 0 for a, _ in opts):
            is_mirror[sl] = True
    return all_pts, identity_block, is_mirror


@dataclass
class FlatCell:
    grain_id: int
    vertices: np.ndarray         # (Nv, 3) float64, relative to the seed
    faces: list[FlatFace]
    volume: float                # Å³
    _bounding_radius: float = field(init=False)

    def __post_init__(self) -> None:
        if len(self.vertices) > 0:
            self._bounding_radius = float(np.max(
                np.linalg.norm(self.vertices, axis=1)
            ))
        else:
            self._bounding_radius = 0.0

    @property
    def bounding_radius(self) -> float:
        return self._bounding_radius


class FlatTessellation(Tessellation):
    """Voronoi tessellation of a periodic/slab box using Qhull (§6.5)."""

    def __init__(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
    ) -> None:
        """
        Parameters
        ----------
        seeds : (N, 3) float64  seed positions, inside [0, L) per axis
        box_lengths : (3,) float64  box dimensions in Å
        periodic : list of 3 bool  per-axis periodicity
        """
        self._seeds = np.asarray(seeds, dtype=np.float64)
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        n = len(self._seeds)
        self._n = n

        all_pts, identity_block, is_mirror = self._replicate()
        self._all_pts = all_pts
        self._identity_block = identity_block
        self._is_mirror = is_mirror
        self._home_ids = np.tile(np.arange(n, dtype=np.int32),
                                 len(all_pts) // n)

        vor = Voronoi(all_pts, qhull_options="Qbb Qc Qz")

        # Index ridges by point once (avoids O(N·R) rescans per cell)
        ridges_by_point: dict[int, list[int]] = defaultdict(list)
        for r_idx, (p1, p2) in enumerate(vor.ridge_points):
            ridges_by_point[int(p1)].append(r_idx)
            ridges_by_point[int(p2)].append(r_idx)

        # Per-grain local competitor sets for the fast owns() path.
        # _cell_nbr_idx[gid] = int64 array of replica indices in self._all_pts
        # that are the home replica plus every face-neighbour replica.
        self._cell_nbr_idx: dict[int, np.ndarray] = {}

        self._cells: list[FlatCell] = []
        for gid in range(n):
            cell = self._build_cell(gid, vor, ridges_by_point)
            self._euler_check(cell)          # gate G4
            self._cells.append(cell)

        self._volume_check()                 # gate G3

        self._adj = self._build_adjacency(vor)

        # Trees for membership queries (grain_of still uses these)
        self._tree = KDTree(all_pts)
        if all(self._periodic):
            seeds_wrapped = self._seeds % self._L
            seeds_wrapped[seeds_wrapped >= self._L] = 0.0
            self._home_tree: KDTree | None = KDTree(seeds_wrapped,
                                                    boxsize=self._L)
        else:
            self._home_tree = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _replicate(self) -> tuple[np.ndarray, int, np.ndarray]:
        return replicate_seeds(self._seeds, self._L, self._periodic)

    def _build_cell(
        self,
        gid: int,
        vor: Voronoi,
        ridges_by_point: dict[int, list[int]],
    ) -> FlatCell:
        """Build FlatCell for grain gid from Voronoi output."""
        home_pt_idx = self._identity_block * self._n + gid

        vertex_idx_set: set[int] = set()
        faces: list[FlatFace] = []
        # Collect every replica index that shares a ridge (face) with home.
        nbr_idxs: list[int] = []

        for r_idx in ridges_by_point.get(home_pt_idx, []):
            p1, p2 = (int(v) for v in vor.ridge_points[r_idx])
            neighbor_idx = p2 if p1 == home_pt_idx else p1
            ridge_verts = vor.ridge_vertices[r_idx]

            if -1 in ridge_verts:
                # With wall mirrors on free axes every home cell is interior
                # and bounded; an unbounded home ridge is an internal error.
                raise TessellationError(
                    f"Unbounded Voronoi ridge on home cell of grain {gid} — "
                    "replica construction failed (internal error)."
                )

            verts = vor.vertices[ridge_verts]  # (M, 3)
            vertex_idx_set.update(int(v) for v in ridge_verts)

            # Outward normal: Voronoi face is the bisector plane, normal to
            # the seed→neighbor direction (replica coordinates).
            normal = self._all_pts[neighbor_idx] - self._all_pts[home_pt_idx]
            n_len = float(np.linalg.norm(normal))
            if n_len < 1e-12 * float(np.max(self._L)):
                raise TessellationError(
                    f"Degenerate seed pair for grain {gid}: replica "
                    f"{neighbor_idx} coincides with the home seed. "
                    "A seed lies exactly on a box wall or two seeds coincide."
                )
            unit_normal = normal / n_len

            verts_ordered = _order_polygon(verts, unit_normal)
            area = _polygon_area(verts_ordered, unit_normal)

            if self._is_mirror[neighbor_idx]:
                neighbor_gid = WALL_NEIGHBOR_ID
            else:
                neighbor_gid = int(self._home_ids[neighbor_idx])

            faces.append(FlatFace(
                neighbor_id=neighbor_gid,
                vertices=verts_ordered,
                area=area,
                unit_normal=unit_normal,
                seed_distance=n_len,
            ))
            nbr_idxs.append(neighbor_idx)

        # Store local competitor set: home replica + every face-neighbour replica.
        # Proof of exactness: see module docstring and the owns() implementation.
        self._cell_nbr_idx[gid] = np.unique(
            np.array([home_pt_idx, *nbr_idxs], dtype=np.int64)
        )

        if not vertex_idx_set:
            all_cell_verts = np.empty((0, 3))
        else:
            all_cell_verts = vor.vertices[sorted(vertex_idx_set)]

        cell_verts_rel = all_cell_verts - self._seeds[gid]
        vol = _cell_volume(faces, self._seeds[gid])

        return FlatCell(
            grain_id=gid,
            vertices=cell_verts_rel,
            faces=faces,
            volume=vol,
        )

    def _euler_check(self, cell: FlatCell) -> None:
        check_euler_cell(cell, self._L)

    def _volume_check(self) -> None:
        """Gate G3: Σ cell volumes == box volume (rel. err ≤ VOL_REL_TOL)."""
        box_vol = float(np.prod(self._L))
        total = self.total_volume()
        rel_err = abs(total - box_vol) / box_vol
        if rel_err > VOL_REL_TOL:
            raise TessellationError(
                f"Volume sum check failed: Σ V_i = {total:.6g} Å³ vs box "
                f"{box_vol:.6g} Å³ (rel. err {rel_err:.3e} > {VOL_REL_TOL:g}, G3)."
            )

    def _build_adjacency(self, vor: Voronoi) -> list[tuple[int, int]]:
        """Grain adjacency from ridge topology.  Wall ridges (mirror
        replicas) and self-image ridges are not grain pairs."""
        adj: set[tuple[int, int]] = set()
        for p1, p2 in vor.ridge_points:
            p1, p2 = int(p1), int(p2)
            if self._is_mirror[p1] or self._is_mirror[p2]:
                continue
            g1 = int(self._home_ids[p1])
            g2 = int(self._home_ids[p2])
            if g1 != g2:
                adj.add((min(g1, g2), max(g1, g2)))
        return sorted(adj)

    # ------------------------------------------------------------------
    # Tessellation ABC methods
    # ------------------------------------------------------------------

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        """Return grain index for each point.

        Wraps X to [0, L) on periodic axes first, then finds the nearest
        seed under min-image PBC, so grain_of(X) = grain_of(X + n·L).
        Periodic images of a grain map back to its home id.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        X_wrapped = X.copy()
        for ax in range(3):
            if self._periodic[ax]:
                X_wrapped[:, ax] = X_wrapped[:, ax] % self._L[ax]
                # x % L can return exactly L for tiny negative x
                col = X_wrapped[:, ax]
                col[col >= self._L[ax]] = 0.0
        if self._home_tree is not None:
            _, idx = self._home_tree.query(X_wrapped)
            return np.asarray(idx, dtype=np.int32)
        _, idx_rep = self._tree.query(X_wrapped)
        return self._home_ids[np.asarray(idx_rep)].astype(np.int32)

    def _owns_global(self, X: np.ndarray, i: int) -> np.ndarray:
        """O(N log N) reference implementation of owns() using the global KDTree.

        Retained verbatim for regression testing: the new owns() must produce
        an identical boolean mask for every grain and every query set.
        Do NOT call this on large fills — it is O(rows · log N_replicas) and
        triggers a per-ambiguous-point Python loop.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.zeros(0, dtype=bool)
        home_idx = self._identity_block * self._n + i
        tol = OWNS_TIE_TOL * float(np.max(self._L))

        dists, idxs = self._tree.query(X, k=2)
        d1 = dists[:, 0]
        d_home = np.linalg.norm(X - self._all_pts[home_idx], axis=1)

        out = np.zeros(len(X), dtype=bool)
        clear_win = (idxs[:, 0] == home_idx) & (dists[:, 1] - d1 > tol)
        clear_loss = (idxs[:, 0] != home_idx) & (d_home - d1 > tol)
        out[clear_win] = True

        ambiguous = np.where(~(clear_win | clear_loss))[0]
        for j in ambiguous:
            cand = self._tree.query_ball_point(X[j], r=d1[j] + tol)
            dd = np.linalg.norm(self._all_pts[cand] - X[j], axis=1)
            # Wall MIRROR replicas clip the cell but never own atoms (a real
            # replica always ties them at the wall), so exclude them from the
            # tie-break — matching local_owns' is_mirror handling.
            owner = min(int(c) for c, d in zip(cand, dd, strict=True)
                        if d <= d1[j] + tol and not self._is_mirror[c])
            out[j] = (owner == home_idx)
        return out

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact home-cell ownership (§6.8; see Tessellation.owns).

        Fast O(rows · deg) path: queries only the LOCAL competitor set of
        grain i — its home replica plus every face-neighbour replica stored
        in self._cell_nbr_idx[i] at construction time.

        Exactness proof (see module docstring): any replica that could be
        nearer to a query point than the home replica must share a face with
        cell i (it lies on the other side of that bisector plane).  Points on
        a face are equidistant only to the face's two neighbours (both in the
        local set).  So restricting the search to the local set reproduces the
        global argmin exactly, including the lowest-replica-index tie-break.

        Implementation: delegates the per-row distance/argmin to
        :func:`grainsmith.tessellation._local_owns.local_owns` — a chunked,
        fully-vectorised NumPy reduction with an optional, bit-identical Numba
        kernel.  The (chunk × deg) NumPy working array stays bounded for
        very large fills.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.zeros(0, dtype=bool)

        home_idx = int(self._identity_block * self._n + i)
        nbr = self._cell_nbr_idx[i]           # (deg,) int64 replica indices
        P = self._all_pts[nbr]                 # (deg, 3) competitor positions
        tol = OWNS_TIE_TOL * float(np.max(self._L))
        return local_owns(X, P, nbr, home_idx, tol, len(self._all_pts),
                          _OWNS_CHUNK, is_mirror=self._is_mirror[nbr])

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed distance to the nearest grain-boundary face of grain i
        (positive = inside grain i).

        Uses the cell's stored faces (bisector planes), so it is exact for
        every face type including periodic self-image faces.  Box-wall faces
        (free surfaces, neighbor_id == -1) are NOT grain boundaries and are
        excluded; a single-grain slab therefore has margin = +inf.

        Points are reduced to the cell's compact frame by min-image relative
        to the seed, so wrapped and unwrapped inputs give identical results.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        cell = self._cells[i]
        center = self._seeds[i]

        dx = X - center
        for ax in range(3):
            if self._periodic[ax]:
                dx[:, ax] -= np.round(dx[:, ax] / self._L[ax]) * self._L[ax]
        Xl = center + dx

        margins = np.full(len(X), np.inf, dtype=np.float64)
        for face in cell.faces:
            if face.neighbor_id == WALL_NEIGHBOR_ID:
                continue
            n_hat = face.unit_normal
            p0 = face.vertices[0]
            signed = (p0 - Xl) @ n_hat
            margins = np.minimum(margins, signed)
        return margins

    def _shell_replica_tree(self) -> KDTree:
        """Plain (non-periodic) cKDTree over every TRANSLATED replica seed
        (``self._all_pts``), used only by ``gb_shell_lower_bound``'s
        fully-periodic k-NN fast path. When every axis is periodic,
        ``replicate_seeds`` never produces a wall MIRROR (module docstring
        step 1), so ``self._all_pts`` already IS exactly the periodic-
        translation replica set the certificate needs -- no extra
        filtering. This is byte-for-byte the same plain ``KDTree(all_pts)``
        that ``__init__`` already builds and caches as ``self._tree`` (used
        by ``grain_of``/``_owns_global``), so it is reused directly here
        instead of building a second, identical tree."""
        return self._tree

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        """Certified GB-shell mask (see ``Tessellation.gb_shell_lower_bound``).

        Fully-periodic case (measured production path: 500 Å periodic
        boxes): delegates to the SAME certified k-NN construction as
        ``WeightedTessellation`` (``tessellation._shell_knn``), with every
        replica weight zero -- Flat's plain-Euclidean-nearest-seed
        membership is exactly the additive diagram's ``w == 0`` special
        case, so the identical 1-Lipschitz proof applies verbatim, and
        ``self._all_pts`` (translations only, no wall mirrors when every
        axis is periodic -- see ``_shell_replica_tree``) is exactly the
        replica set the certificate needs.  Measured: this is
        substantially cheaper than the alternative below at production
        scale (evaluating the per-grain face-plane margin -- also exact,
        see ``margin()`` -- over EVERY atom, not just pair endpoints,
        costs O(N * faces_per_cell) and was measured to erase most of this
        pre-filter's own saving on a 1.37M-atom / 140-grain system: 0.57 s
        for the margin pass alone against a 0.96 s unfiltered baseline).

        Mixed/non-periodic case (a free axis has no k-NN-friendly replica
        set -- ``_periodic_translations``-style replication does not
        apply to wall mirrors the same way): falls back to the exact
        per-grain face-plane margin (module-level
        ``_exact_margin_shell_mask``) -- for a point interior to a convex
        polytope, the distance to the boundary equals the minimum
        distance to its supporting hyperplanes, and ``margin()`` computes
        exactly that minimum over every stored face (see ``margin()``'s
        docstring above) -- INCLUDING self-image faces (``neighbor_id ==
        grain_id`` is a legitimate GB face with the grain's own periodic
        image, and only ``WALL_NEIGHBOR_ID`` faces are excluded; a
        free-surface wall cannot host an overlap pair, so excluding it
        from the margin cannot drop a pair). This path is not the
        production-critical one (not fully periodic), so its extra cost
        is accepted in exchange for reusing ``margin()`` directly rather
        than re-deriving a second wall-aware replica set.

        ``grain`` must be each atom's OWN grain id for the fallback path
        (fill only ever places an atom inside the cell that owns it, so
        this is always an interior point of ``grain[k]``'s cell -- the
        precondition the support-hyperplane argument needs); the
        fully-periodic k-NN path ignores it entirely (home ownership is
        the queried argmin replica, never the recorded grain id, exactly
        as in the additive-weighted backend).
        """
        if all(self._periodic):
            return certified_knn_shell_mask(
                pos, self._shell_replica_tree(),
                np.zeros(len(self._all_pts), dtype=np.float64), cutoff,
                workers, k=_SHELL_KNN_K)
        return _exact_margin_shell_mask(self, pos, grain, cutoff)

    def adjacency(self) -> list[tuple[int, int]]:
        return list(self._adj)

    def bounding_radius(self, i: int) -> float:
        return self._cells[i].bounding_radius

    @property
    def seeds(self) -> np.ndarray:
        return self._seeds.copy()

    @property
    def n_grains(self) -> int:
        return self._n

    @property
    def cells(self) -> list[FlatCell]:
        return self._cells

    def cell_vertices_rel(self, i: int) -> np.ndarray | None:
        """Cell vertices of grain i, relative to seeds[i], in Cartesian Å.

        FlatCell.vertices is already stored relative to the seed at construction
        time (see _build_cell: ``cell_verts_rel = all_cell_verts - self._seeds[gid]``).
        PowerTessellation inherits this method unchanged — its _cells list also
        contains FlatCell objects with the same relative-to-seed convention.
        """
        return self._cells[i].vertices

    def total_volume(self) -> float:
        return sum(c.volume for c in self._cells)


def _exact_margin_shell_mask(
    tess: FlatTessellation, pos: np.ndarray, grain: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Exact per-grain face-plane GB-shell mask, shared by
    ``FlatTessellation.gb_shell_lower_bound``'s non-fully-periodic
    fallback and ``PowerTessellation.gb_shell_lower_bound`` (the power
    diagram's cells are ALSO convex polytopes bounded by exactly this kind
    of plane, built via halfspace intersection instead of Voronoi -- see
    power.py's module docstring -- so the same support-hyperplane
    exactness argument applies verbatim; Power's generalized distance
    ``‖x-c‖² - w`` is NOT 1-Lipschitz, so it cannot use the k-NN
    certificate ``FlatTessellation`` uses for the fully-periodic case,
    only this exact-margin path).

    Groups *pos* by *grain* and calls ``tess.margin`` once per distinct
    grain (batched, matching ``atoms.overlap._memoize_margins``'s
    per-grain batching convention) rather than once per atom.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    grain = np.asarray(grain)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=bool)
    m = np.empty(n, dtype=np.float64)
    for g in np.unique(grain):
        sel = grain == g
        m[sel] = tess.margin(pos[sel], int(g))
    return m <= cutoff


# ---------------------------------------------------------------------------
# Geometry helpers (shared with the power/Laguerre backend)
# ---------------------------------------------------------------------------

def check_euler_cell(cell: FlatCell, box_lengths: np.ndarray) -> None:
    """Gate G4: V − E + F = 2 per convex cell (vertex dedupe at
    EULER_TOL·max(L) via union-find on tol-pairs — robust against
    degenerate vertices that Qhull splits into near-duplicates)."""
    if not cell.faces:
        raise TessellationError(
            f"Grain {cell.grain_id} has no faces (G4)."
        )
    tol = EULER_TOL * float(np.max(box_lengths))

    all_verts = np.vstack([f.vertices for f in cell.faces])
    parent = list(range(len(all_verts)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = KDTree(all_verts)
    for a, b in tree.query_pairs(tol):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cluster = np.array([find(k) for k in range(len(all_verts))])
    v_count = len(np.unique(cluster))

    edges: set[frozenset[int]] = set()
    offset = 0
    for f in cell.faces:
        m = len(f.vertices)
        ids = cluster[offset:offset + m]
        offset += m
        for t in range(m):
            e = frozenset((int(ids[t]), int(ids[(t + 1) % m])))
            if len(e) == 2:
                edges.add(e)
    e_count = len(edges)
    f_count = len(cell.faces)

    euler = v_count - e_count + f_count
    if euler != 2:
        raise TessellationError(
            f"Euler check failed for grain {cell.grain_id}: "
            f"V−E+F = {v_count}−{e_count}+{f_count} = {euler} ≠ 2 (G4)."
        )

def _order_polygon(verts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the polygon vertices sorted cyclically (CCW about `normal`).

    Qhull's ridge_vertices carry no cyclic-order guarantee in 3D; an
    unordered fan triangulation gives wrong areas (up to tens of percent).
    """
    if len(verts) < 3:
        return verts
    centroid = verts.mean(axis=0)
    rel = verts - centroid
    # In-plane orthonormal basis (u, v) with u from the largest offset
    u = rel[int(np.argmax(np.linalg.norm(rel, axis=1)))]
    u = u - np.dot(u, normal) * normal
    u_len = float(np.linalg.norm(u))
    if u_len < 1e-30:
        return verts
    u /= u_len
    v = np.cross(normal, u)
    angles = np.arctan2(rel @ v, rel @ u)
    return verts[np.argsort(angles)]


def _polygon_area(verts: np.ndarray, normal: np.ndarray) -> float:
    """Area of a planar polygon with cyclically ordered vertices.

    Projection form of the shoelace formula:
        A = ½ |n̂ · Σ_k (v_k × v_{k+1})|
    Exact for planar polygons; requires ordered vertices (see
    _order_polygon).
    """
    if len(verts) < 3:
        return 0.0
    cross_sum = np.cross(verts, np.roll(verts, -1, axis=0)).sum(axis=0)
    return float(0.5 * abs(np.dot(normal, cross_sum)))


def _cell_volume(faces: list[FlatFace], center: np.ndarray) -> float:
    """Cell volume via divergence theorem: V = (1/3) Σ_f area_f · |n̂_f · r_f|
    where r_f is the vector from the seed to the face centroid (the seed is
    interior, so n̂_f · r_f > 0 for outward normals; abs() is a safety net)."""
    vol = 0.0
    for face in faces:
        face_centroid = face.vertices.mean(axis=0)
        r = face_centroid - center
        vol += face.area * abs(np.dot(face.unit_normal, r))
    return vol / 3.0
