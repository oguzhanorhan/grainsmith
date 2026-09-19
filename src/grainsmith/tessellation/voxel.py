"""Voxel membership grid for curved tessellations (§6.7).

Provides:
- Per-grain volumes (voxel count × voxel volume)
- Adjacency (face-neighbor label pairs, periodicity-aware)
- Boundary voxel extraction (gb_points)
- Connectivity check (periodic flood fill, gate G5)
- GB areas (voxel-face estimator; marching-cubes mesh refinement optional)
- Local boundary normals (central differences of margin_i − margin_j)
- Optional marching-cubes GB mesh per grain pair (scikit-image, extra [mesh])

Area-estimator bias (§6.7): the voxel-face count
estimator is exact for axis-aligned boundaries but OVERESTIMATES oblique
ones (a 45° plane by √2, up to √3 in the worst corner orientation) — it is
the fast, PBC-correct fallback.  The marching-cubes mesh estimator is
asymptotically unbiased for smooth boundaries but, being computed on the
non-periodic label array, only captures zero-crossings interior to the box
(a boundary sheet that coincides with the box face is reported by the
voxel-face estimator, not the mesh).
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np

from grainsmith.constants import VOXEL_GRID_MAX
from grainsmith.errors import GrainsmithError, TessellationError
from grainsmith.memory import build_memory_guard_message
from grainsmith.tessellation.base import Tessellation

log = logging.getLogger(__name__)


def voxel_grid_shape(box_lengths: np.ndarray, grid_size: int) -> tuple[int, int, int]:
    """Requested grid shape with the same per-axis cap used by the builder."""
    lengths = np.asarray(box_lengths, dtype=np.float64)
    spacing = float(np.min(lengths)) / grid_size
    requested = tuple(int(np.ceil(length / spacing)) for length in lengths)
    clamped = tuple(min(size, VOXEL_GRID_MAX) for size in requested)
    if clamped != requested:
        log.warning(
            "Voxel grid %s clamped to %s (VOXEL_GRID_MAX=%d per axis); "
            "analysis resolution on the clamped axis is coarser than requested.",
            requested, clamped, VOXEL_GRID_MAX,
        )
    return cast(tuple[int, int, int], clamped)


def build_voxel_grid(
    tess: Tessellation,
    box_lengths: np.ndarray,
    grid_size: int | str = "auto",
    r_ws: float | None = None,
    correlation_length: float | None = None,
) -> VoxelGrid:
    """Build a voxel membership grid for the given tessellation.

    Parameters
    ----------
    tess : Tessellation
    box_lengths : (3,) float64
    grid_size : int or "auto"
        Number of voxels along the shortest box edge (others scaled
        proportionally).  Auto: h = min(ℓ/4, r_ws/10), clamped to
        ≤ VOXEL_GRID_MAX per axis (warns when clamped).
    r_ws, correlation_length : used for auto grid resolution.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    L_min = float(np.min(L))

    if grid_size == "auto":
        h_target = L_min / 20.0  # conservative default
        if r_ws is not None:
            h_target = min(h_target, r_ws / 10.0)
        if correlation_length is not None:
            h_target = min(h_target, correlation_length / 4.0)
        n_short = int(np.ceil(L_min / h_target))
        if n_short > VOXEL_GRID_MAX:
            log.warning(
                f"Auto voxel grid ({n_short} per shortest edge) clamped to "
                f"{VOXEL_GRID_MAX}; analysis resolution is coarser than the "
                "h = min(ℓ/4, r_ws/10) target."
            )
            n_short = VOXEL_GRID_MAX
    else:
        n_short = int(grid_size)

    grid_shape = voxel_grid_shape(L, n_short)

    # §13: memory estimate BEFORE allocation (centers f64 ×3, labels i32,
    # ×2 safety for transients), hard error above the per-run limit. Read
    # the limit OFF THE TESSELLATION (tess.memory_limit_bytes) rather than
    # the constants.py module default -- the pipeline stamps the run's
    # runtime.memory_limit_gb value there (tessellation/base.py) so this
    # check honors it while ProcessPoolExecutor 'spawn' workers elsewhere
    # (fill.py) still see the value via the pickled instance, not a global.
    n_vox = int(np.prod(grid_shape))
    est_bytes = n_vox * (3 * 8 + 4) * 2
    log.info(
        f"Voxel grid {grid_shape}: {n_vox:,} voxels, "
        f"estimated {est_bytes / 1e9:.2f} GB peak."
    )
    limit_bytes = tess.memory_limit_bytes
    if est_bytes > limit_bytes:
        # getattr, not a plain attribute read: a handful of tests exercise
        # build_voxel_grid against duck-typed stub tessellations that don't
        # subclass Tessellation and therefore never picked up
        # memory_limit_source (tests/test_curvature.py) — "config" (the
        # base-class default, see tessellation/base.py) is the correct
        # fallback for those.
        source = getattr(tess, "memory_limit_source", "config")
        raise TessellationError(build_memory_guard_message(
            f"Voxel grid {grid_shape}",
            est_bytes, limit_bytes, source,
            # NO --jobs sentence here: build_voxel_grid builds exactly one
            # whole-box grid in the single-threaded driver (never inside a
            # ProcessPoolExecutor worker) — see build_memory_guard_message's
            # extra_advice docs for why that sentence would be false here.
            extra_advice="Use a smaller analysis.voxel_grid.",
        ))

    centers = _voxel_centers(L, grid_shape)  # (Nx*Ny*Nz, 3)

    # Query tessellation (chunked)
    CHUNK = 500_000
    labels = np.empty(n_vox, dtype=np.int32)
    for start in range(0, n_vox, CHUNK):
        labels[start:start + CHUNK] = tess.grain_of(centers[start:start + CHUNK])

    labels_3d = labels.reshape(grid_shape)
    return VoxelGrid(labels_3d, L, grid_shape, tess.n_grains, tess=tess)


def _voxel_centers(L: np.ndarray, shape: tuple) -> np.ndarray:
    """Generate (N, 3) array of voxel center coordinates."""
    Nx, Ny, Nz = shape
    hx, hy, hz = L[0] / Nx, L[1] / Ny, L[2] / Nz
    xs = np.arange(Nx) * hx + 0.5 * hx
    ys = np.arange(Ny) * hy + 0.5 * hy
    zs = np.arange(Nz) * hz + 0.5 * hz
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Total area (Å²) of a triangle mesh."""
    if len(faces) == 0:
        return 0.0
    tri = verts[faces]  # (M, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))


def _gb_pair_normals(
    tess: Tessellation, h_vec: np.ndarray, i: int, j: int, pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference local normals for ONE grain pair — pure function
    of (tess, h_vec, i, j, pts), with no dependence on a VoxelGrid instance.

    This is the per-pair kernel VoxelGrid.gb_normals' loop calls (extracted
    so it can also be farmed out per-pair to worker processes by
    analysis/boundaries.py's process-parallel driver — mirrors analysis/
    curvature.py's _pair_curvature: both the serial gb_normals loop and the
    parallel driver call this SAME function, so results are byte-identical
    by construction rather than by two hand-kept-in-sync copies of the
    math).

    Returns (normals (M,3) unfiltered, ok (M,) bool — non-degenerate
    gradient mask, see gb_normals docstring); the caller filters
    points/normals/areas by `ok`.
    """
    grad = np.empty((len(pts), 3), dtype=np.float64)
    for ax in range(3):
        step = np.zeros(3)
        step[ax] = h_vec[ax]
        phi_p = tess.margin(pts + step, i) - tess.margin(pts + step, j)
        phi_m = tess.margin(pts - step, i) - tess.margin(pts - step, j)
        grad[:, ax] = (phi_p - phi_m) / (2.0 * h_vec[ax])
    norm = np.linalg.norm(grad, axis=1)
    ok = norm > 1e-30
    normals = np.zeros_like(grad)
    normals[ok] = -grad[ok] / norm[ok, None]
    return normals, ok


class VoxelGrid:
    """Voxel membership grid for analysis and curved-boundary operations (§6.7)."""

    def __init__(
        self,
        labels: np.ndarray,   # (Nx, Ny, Nz) int32
        box_lengths: np.ndarray,
        shape: tuple,
        n_grains: int,
        tess: Tessellation | None = None,
    ) -> None:
        self.labels = labels  # via the property below: also (re)seeds the
                              # empty _gb_faces_cache for this instance
        self.L = np.asarray(box_lengths, dtype=np.float64)
        self.shape = shape
        self.n_grains = n_grains
        self.tess = tess
        self.h_vec = np.array([self.L[k] / shape[k] for k in range(3)],
                              dtype=np.float64)
        self.h = float(np.max(self.h_vec))  # nominal (≈ isotropic) spacing

    # ------------------------------------------------------------------
    # labels: a property so any rebind (not just __init__) invalidates
    # the memoized GB-face cache below.
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """(Nx, Ny, Nz) int32 grain-id membership grid.

        A plain attribute in behavior (get/set, no other side effects) —
        it is a property only so that assigning a new array also clears
        `_gb_faces_cache`. The one rebind outside __init__ is
        perturbed.py's G5 connectivity repair (``vg.labels = repaired``);
        without invalidation here, a grid whose GB faces were already
        computed and cached under the pre-repair labels would silently
        keep serving stale faces after the rebind.
        """
        return self._labels

    @labels.setter
    def labels(self, value: np.ndarray) -> None:
        self._labels = value
        # _gb_faces_cache: {periodic tuple: {(i, j): (midpoints, areas)}}.
        # Reset (not mutated) on every labels rebind, __init__ included.
        # Once (re)filled per periodic key, entries pin their face arrays
        # for the instance's remaining lifetime — bounded by the grid's
        # own adjacency-pair count; measured no RSS regression at
        # benchmark scale.
        self._gb_faces_cache: dict[
            tuple[bool, bool, bool],
            dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
        ] = {}

    # ------------------------------------------------------------------
    # Volumes / adjacency / boundary extraction
    # ------------------------------------------------------------------

    def volumes(self) -> np.ndarray:
        """Per-grain volumes (Å³). Returns (n_grains,) float64."""
        vox_vol = float(np.prod(self.h_vec))
        counts = np.bincount(self.labels.ravel(), minlength=self.n_grains)
        return counts[:self.n_grains].astype(np.float64) * vox_vol

    def adjacency(self, periodic: list[bool] | None = None) -> list[tuple[int, int]]:
        """Adjacency list from face-neighbor label pairs (6-connectivity).

        On non-periodic axes the wrap-around pair produced by np.roll
        (last slice ↔ first slice) is NOT a physical contact and is masked
        out; pass the box periodicity to get correct slab adjacency.
        """
        if periodic is None:
            periodic = [True, True, True]
        adj: set[tuple[int, int]] = set()
        lab = self.labels
        for axis in range(3):
            shifted = np.roll(lab, -1, axis=axis)
            mask = lab != shifted
            if not periodic[axis]:
                sl: list[slice | int] = [slice(None)] * 3
                sl[axis] = -1
                mask[tuple(sl)] = False
            a = lab[mask]
            b = shifted[mask]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            if len(lo):
                for p, q in np.unique(np.stack([lo, hi], axis=1), axis=0):
                    adj.add((int(p), int(q)))
        return sorted(adj)

    def boundary_voxels(self, periodic: list[bool] | None = None) -> np.ndarray:
        """Return (M, 3) positions of voxels on grain boundaries.

        Wrap-around comparisons are masked on non-periodic axes (a free
        surface is not a grain boundary)."""
        if periodic is None:
            periodic = [True, True, True]
        lab = self.labels
        Nx, Ny, Nz = self.shape
        boundary = np.zeros((Nx, Ny, Nz), dtype=bool)
        for axis in range(3):
            shifted = np.roll(lab, -1, axis=axis)
            diff = lab != shifted
            if not periodic[axis]:
                sl: list[slice | int] = [slice(None)] * 3
                sl[axis] = -1
                diff[tuple(sl)] = False
            boundary |= diff
        idx = np.argwhere(boundary)  # (M, 3) voxel indices
        return (idx + 0.5) * self.h_vec

    # ------------------------------------------------------------------
    # GB faces: areas and local normals (§6.7)
    # ------------------------------------------------------------------

    def _gb_faces(
        self, periodic: list[bool] | None
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Voxel faces where the label changes, grouped by grain pair.

        Returns {(i, j): (midpoints (M,3), face_areas (M,))} with i < j.
        A face between voxel v and its +1 neighbor along axis k has its
        midpoint on the shared face and area ∏h / h_k.

        Memoized on the instance keyed by the (resolved) periodic tuple:
        gb_normals and gb_face_samples both call this for the same grid,
        and callers must treat the cached arrays as read-only (none of
        the current consumers mutate them in place).
        """
        if periodic is None:
            periodic = [True, True, True]
        # box.periodic is a 3-list by schema and the default above is a
        # 3-list, so the generator always yields exactly three bools; cast
        # tells mypy the fixed arity it cannot infer from a genexp.
        key = cast("tuple[bool, bool, bool]",
                   tuple(bool(p) for p in periodic))
        cached = self._gb_faces_cache.get(key)
        if cached is not None:
            return cached
        lab = self.labels
        vox_vol = float(np.prod(self.h_vec))
        out: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}
        for axis in range(3):
            shifted = np.roll(lab, -1, axis=axis)
            mask = lab != shifted
            if not periodic[axis]:
                sl: list[slice | int] = [slice(None)] * 3
                sl[axis] = -1
                mask[tuple(sl)] = False
            idx = np.argwhere(mask)
            if len(idx) == 0:
                continue
            mids = (idx + 0.5) * self.h_vec
            mids[:, axis] += 0.5 * self.h_vec[axis]
            face_area = vox_vol / self.h_vec[axis]
            a = lab[mask]
            b = shifted[mask]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            keys = lo.astype(np.int64) * self.n_grains + hi.astype(np.int64)
            for pair_key in np.unique(keys):
                i, j = int(pair_key // self.n_grains), int(pair_key % self.n_grains)
                pts = mids[keys == pair_key]
                out.setdefault((i, j), []).append((pts, face_area))
        result = {
            pair: (
                np.concatenate([p for p, _ in chunks], axis=0),
                np.concatenate([np.full(len(p), ar) for p, ar in chunks]),
            )
            for pair, chunks in out.items()
        }
        self._gb_faces_cache[key] = result
        return result

    def gb_face_samples(
        self, periodic: list[bool] | None = None
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Boundary voxel-face sample points and area weights per pair.

        Public accessor over the voxel-face extraction shared with
        gb_areas / gb_normals: {(i, j): (midpoints (M, 3) Å,
        face_areas (M,) Å²)}, i < j.  Same staircase-area caveat as
        gb_areas (module docstring).

        The returned dict and its (midpoints, areas) arrays ARE
        `_gb_faces_cache`'s own shared state, not a copy — treat them as
        read-only. Mutating them in place would corrupt every other
        caller's view of the same cached pair for the rest of this
        instance's lifetime (or until a `labels` rebind clears it).
        """
        return self._gb_faces(periodic)

    def gb_areas(self, periodic: list[bool] | None = None
                 ) -> dict[tuple[int, int], float]:
        """Per-pair GB area via the voxel-face estimator (Å²).

        Exact for axis-aligned boundaries; overestimates oblique ones by up
        to √3 (staircase bias, see module docstring) — the fast, PBC-correct
        fallback of §6.7.  Use gb_mesh()/mesh_area() for smooth boundaries.
        """
        faces = self._gb_faces(periodic)
        return {pair: float(np.sum(areas)) for pair, (_, areas) in faces.items()}

    def gb_normals(
        self, periodic: list[bool] | None = None
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Local boundary normals per grain pair (§6.7).

        For each label-change voxel face between grains (i, j), the unit
        normal (pointing from grain i toward grain j, i < j) is the central
        difference of the level-set field φ(x) = margin_i(x) − margin_j(x):
        φ > 0 inside i, so n̂ = −∇φ/‖∇φ‖.

        Returns {(i, j): (face_midpoints (M,3), unit_normals (M,3),
        face_areas (M,))} — the areas are the weights for the area-weighted
        mean normal and normal-spread statistics of §6.10.
        Requires the grid to have been built with a tessellation reference.
        """
        if self.tess is None:
            raise GrainsmithError(
                "gb_normals requires the VoxelGrid to hold its source "
                "tessellation (build via build_voxel_grid)."
            )
        faces = self._gb_faces(periodic)
        result: dict[tuple[int, int],
                     tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for (i, j), (pts, areas) in faces.items():
            normals, ok = _gb_pair_normals(self.tess, self.h_vec, i, j, pts)
            result[(i, j)] = (pts[ok], normals[ok], areas[ok])
        return result

    def gb_mesh(
        self, pair: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Marching-cubes triangle mesh of the (i, j) grain boundary
        (§6.7, optional — requires the [mesh] extra / scikit-image).

        The zero level set of φ = margin_i − margin_j is meshed on the
        voxel grid and triangles are filtered to those whose centroid is
        owned by grain i or j (the φ = 0 surface continues into third-grain
        territory past triple lines).  Computed on the non-periodic array:
        boundary sheets coinciding with a box face are not captured (use
        gb_areas for the PBC-correct estimate).

        Returns (vertices (V,3) Å lab frame, faces (F,3) int).
        """
        try:
            from skimage import measure
        except ImportError as exc:
            raise GrainsmithError(
                'gb_mesh requires scikit-image: pip install ".[mesh]"'
            ) from exc
        if self.tess is None:
            raise GrainsmithError(
                "gb_mesh requires the VoxelGrid to hold its source "
                "tessellation (build via build_voxel_grid)."
            )
        i, j = pair
        centers = _voxel_centers(self.L, self.shape)
        phi = (self.tess.margin(centers, i)
               - self.tess.margin(centers, j)).reshape(self.shape)
        if not (phi.min() < 0.0 < phi.max()):
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
        verts, faces, _, _ = measure.marching_cubes(
            phi, level=0.0, spacing=tuple(self.h_vec))
        verts = verts + 0.5 * self.h_vec  # grid origin sits at voxel center
        if len(faces):
            cent = verts[faces].mean(axis=1)
            owner = self.tess.grain_of(cent)
            faces = faces[np.isin(owner, [i, j])]
        return verts, faces

    # ------------------------------------------------------------------
    # Connectivity (gate G5)
    # ------------------------------------------------------------------

    def check_connectivity(self, periodic: list[bool]) -> dict[int, bool]:
        """Check each grain is one connected component under periodic
        6-connectivity flood fill (gate G5).

        Components smaller than max(2, G5_SPURIOUS_VOXEL_FRACTION × grain
        voxels) are discretization slivers (see the constant's docstring) —
        even convex flat-Voronoi cells voxel-split at wedge corners thinner
        than the voxel spacing.  G5 fails on macroscopic fragments only.

        Returns dict grain_id -> bool (True = connected).
        Raises TessellationError for empty or fragmented grains.
        """
        from grainsmith.constants import G5_SPURIOUS_VOXEL_FRACTION

        results = {}
        for gid in range(self.n_grains):
            mask = (self.labels == gid)
            n_total = int(mask.sum())
            if n_total == 0:
                # Empty-cell guard (§3.2): a grain with zero voxels means the
                # weighted/warped diagram swallowed it — fail loudly.
                raise TessellationError(
                    f"Grain {gid} has zero voxels (empty cell, G5). "
                    "Reduce weight_sigma / amplitude, or refine the voxel grid."
                )
            sizes = sorted(_component_sizes(mask, periodic), reverse=True)
            sliver_cap = max(2.0, G5_SPURIOUS_VOXEL_FRACTION * n_total)
            macroscopic = [s for s in sizes[1:] if s > sliver_cap]
            results[gid] = not macroscopic
            if sizes[1:]:
                log.debug(
                    f"G5 grain {gid}: {len(sizes)} components, sizes {sizes[:5]} "
                    f"(sliver cap {sliver_cap:.1f})"
                )

        disconnected = [g for g, ok in results.items() if not ok]
        if disconnected:
            raise TessellationError(
                f"Grains {disconnected} are NOT connected (G5 gate failed). "
                "Reduce boundary amplitude or increase correlation_length."
            )
        return results


def _component_sizes(mask: np.ndarray, periodic: list[bool]) -> list[int]:
    """Sizes (voxel counts) of the 6-connected components of a 3D boolean
    mask under periodic connectivity.

    Implementation: non-periodic ``scipy.ndimage.label`` (C flood fill),
    then components touching through periodic faces are merged with a
    small union-find over the wrap face-pairs — O(n_vox) with vectorized
    inner loops, replacing the original pure-Python BFS that made G5 the
    bottleneck for imported 256³ fields.
    """
    from scipy.ndimage import label as _cc_label

    labeled, n_comp = _cc_label(mask)   # 6-connectivity default in 3D
    if n_comp == 0:
        return []
    sizes_arr = np.bincount(labeled.ravel(), minlength=n_comp + 1)[1:]

    parent = np.arange(n_comp + 1, dtype=np.intp)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]   # path halving
            a = int(parent[a])
        return a

    for ax in range(3):
        if not periodic[ax]:
            continue
        first: list[slice | int] = [slice(None)] * 3
        last: list[slice | int] = [slice(None)] * 3
        first[ax] = 0
        last[ax] = -1
        a_face = labeled[tuple(first)].ravel()
        b_face = labeled[tuple(last)].ravel()
        both = (a_face > 0) & (b_face > 0)
        pairs = np.unique(
            np.stack([a_face[both], b_face[both]], axis=1), axis=0)
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    merged: dict[int, int] = {}
    for comp in range(1, n_comp + 1):
        root = find(comp)
        merged[root] = merged.get(root, 0) + int(sizes_arr[comp - 1])
    return list(merged.values())
