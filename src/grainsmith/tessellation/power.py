"""Periodic power (Laguerre) diagram backend.

Cell_i = {x : ‖x−c_i‖² − w_i ≤ ‖x−c_j‖² − w_j ∀j}: CONVEX polyhedra whose
(i,j) interface is the plane 2(c_j−c_i)·x = ‖c_j‖²−‖c_i‖² + w_i − w_j.
This is the diagram whose cell volumes are controllable through the weights
(semi-discrete optimal transport, sdot.py) — NOT the Johnson–Mehl
``additive_weights`` (d − w) diagram, whose boundaries are curved.

Construction (per home cell): halfspace intersection over the flat-Voronoi
replica seed set (translations on periodic axes, MIRROR seeds on free axes;
flat.replicate_seeds).  Mirrors carry the same weight as their source, so
the power bisector with the mirror degenerates to the wall plane and the
flat-Voronoi exact-wall-clipping proof carries over verbatim (the weights cancel in
power_mirror(x) − power_seed(x) = ‖x−c′‖² − ‖x−c‖² ≥ 0 inside the box).

Interior point per cell: Chebyshev center via linprog (HiGHS) — a power
cell need NOT contain its seed, and may even be empty for extreme weights
(the SDOT damping prevents that; an empty cell raises TessellationError).

Halfspace pruning + a-posteriori certificate: candidate neighbors come from
a KDTree radius query; a dropped halfspace can only ENLARGE cells, so the
G3 volume-sum check (Σ V_i == V_box, rel ≤ 1e−6) certifies the pruning —
on failure the radius doubles and the diagram is rebuilt (deterministic;
final fallback uses ALL replicas).

Membership queries use the 4D LIFTING: with z_j = sqrt(w_max − w_j),
‖(x,0) − (c_j,z_j)‖² = ‖x−c_j‖² + w_max − w_j = power_j(x) + w_max, so the
power-nearest replica is the EUCLIDEAN-nearest lifted point — one 4D
KDTree serves grain_of/owns exactly, with the same tie-break semantics.

Inherits from FlatTessellation: margin (exact signed plane distances),
adjacency, cells, bounding radii, and every flat-only output (vertices.csv,
gnuplot edges, exact polyhedral volumes/areas) work unchanged — with zero
weights the diagram IS the flat Voronoi diagram (regression-pinned).
gb_shell_lower_bound is NOT inherited unchanged: it is deliberately
OVERRIDDEN below to force FlatTessellation's exact per-grain face-plane
margin path unconditionally, skipping its fully-periodic k-NN fast path
entirely. That fast path's certificate relies on the membership distance
being 1-Lipschitz, which holds for Flat's plain-Euclidean-nearest-seed
distance but not for the power (Laguerre) generalized distance
‖x−c‖² − w (its Lipschitz constant grows without bound as ‖x−c‖ grows),
so it cannot carry over to nonzero weights -- see
``PowerTessellation.gb_shell_lower_bound``'s own docstring.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, KDTree

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from grainsmith.constants import OWNS_TIE_TOL, POWER_PLANE_TOL
from grainsmith.errors import TessellationError
from grainsmith.tessellation._local_owns import local_owns
from grainsmith.tessellation.flat import (
    _OWNS_CHUNK,
    WALL_NEIGHBOR_ID,
    FlatCell,
    FlatFace,
    FlatTessellation,
    _exact_margin_shell_mask,
    _order_polygon,
    _polygon_area,
    check_euler_cell,
    replicate_seeds,
)

log = logging.getLogger(__name__)

_MAX_PRUNE_DOUBLINGS = 3
_MIN_CANDIDATES = 26


class PowerTessellation(FlatTessellation):
    """Periodic/slab power (Laguerre) diagram with per-grain weights (Å²)."""

    def __init__(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
        weights: np.ndarray | None = None,
    ) -> None:
        # NOTE: intentionally does NOT call super().__init__ — the parent
        # builds a Voronoi diagram; this class builds the same attribute
        # set from halfspace intersections, then reuses every inherited
        # method that reads those attributes (margin, adjacency, cells...).
        self._seeds = np.asarray(seeds, dtype=np.float64)
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        n = len(self._seeds)
        self._n = n
        if weights is None:
            weights = np.zeros(n, dtype=np.float64)
        self._weights = np.asarray(weights, dtype=np.float64).copy()
        if self._weights.shape != (n,):
            raise TessellationError(
                f"weights shape {self._weights.shape} != ({n},)")

        all_pts, identity_block, is_mirror = replicate_seeds(
            self._seeds, self._L, self._periodic)
        self._all_pts = all_pts
        self._identity_block = identity_block
        self._is_mirror = is_mirror
        self._home_ids = np.tile(np.arange(n, dtype=np.int32),
                                 len(all_pts) // n)
        # Replica weights: every image (translated or mirrored) carries the
        # weight of its source grain.
        self._rep_w = self._weights[self._home_ids.astype(np.int64)]

        # 4D lifting for membership queries (see module docstring).
        w_max = float(np.max(self._rep_w))
        lift = np.sqrt(w_max - self._rep_w)
        self._lifted = np.column_stack([all_pts, lift])
        self._tree4 = KDTree(self._lifted)

        # Per-grain local competitor sets for the fast owns() path.
        # _cell_nbr_idx[gid] = int64 array of replica indices in self._lifted
        # that are the home replica plus every candidate replica that generated
        # a real face (or the full pruned candidate set, which is a superset).
        self._cell_nbr_idx: dict[int, np.ndarray] = {}

        self._build_all_cells()              # gates G3 + G4 inside
        self._adj = self._adjacency_from_faces()
        # Parent attributes not used by this backend:
        self._tree = None
        self._home_tree = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_all_cells(self) -> None:
        """Build every home cell; G3 certifies the halfspace pruning —
        on volume mismatch the candidate radius doubles and we rebuild."""
        from grainsmith.seeding import wigner_seitz_radius

        tree3 = KDTree(self._all_pts)
        box_vol = float(np.prod(self._L))
        r_ws = wigner_seitz_radius(box_vol, self._n)
        radius = 4.0 * r_ws

        from grainsmith.constants import VOL_REL_TOL
        for attempt in range(_MAX_PRUNE_DOUBLINGS + 1):
            use_all = attempt == _MAX_PRUNE_DOUBLINGS
            # Reset the local-set dict each attempt so a doubled-radius
            # rebuild produces a fresh (possibly larger) neighbour set.
            self._cell_nbr_idx = {}
            cells = [
                self._build_cell_hs(gid, tree3, None if use_all else radius)
                for gid in range(self._n)
            ]
            total = sum(c.volume for c in cells)
            rel_err = abs(total - box_vol) / box_vol
            if rel_err <= VOL_REL_TOL:
                self._cells = cells
                for c in cells:
                    check_euler_cell(c, self._L)     # gate G4
                return
            log.debug(
                "power: pruning radius %.3g insufficient "
                "(volume rel err %.3e) — %s", radius, rel_err,
                "FULL set failed" if use_all else "doubling")
            radius *= 2.0
        raise TessellationError(
            f"Power-diagram volume sum failed even with the full replica "
            f"set: Σ V_i vs box rel. err {rel_err:.3e} (G3). Weights may "
            "be extreme (near-empty cells) — reduce the target volume "
            "spread.")

    def _build_cell_hs(
        self,
        gid: int,
        tree3: KDTree,
        radius: float | None,
    ) -> FlatCell:
        """One home cell via halfspace intersection."""
        home_idx = self._identity_block * self._n + gid
        c_i = self._all_pts[home_idx]
        w_i = self._rep_w[home_idx]

        if radius is None:
            cand = np.arange(len(self._all_pts))
        else:
            cand = np.asarray(
                tree3.query_ball_point(c_i, r=radius), dtype=np.int64)
            if len(cand) < _MIN_CANDIDATES + 1:
                _, knn = tree3.query(
                    c_i, k=min(_MIN_CANDIDATES + 1, len(self._all_pts)))
                cand = np.union1d(cand, np.asarray(knn, dtype=np.int64))
        cand = cand[cand != home_idx]

        cj = self._all_pts[cand]
        wj = self._rep_w[cand]
        # Halfspace a·x ≤ b:  a = 2(c_j − c_i),
        # b = ‖c_j‖² − ‖c_i‖² − w_j + w_i
        a = 2.0 * (cj - c_i)
        # A zero-norm bisector row means a candidate seed coincides with the
        # home seed; _chebyshev_center accepts the degenerate 0·x ≤ b row but
        # HalfspaceIntersection then raises a cryptic QhullError. Reject it with
        # a clean message instead (mirrors the flat backend).
        if np.any(np.linalg.norm(a, axis=1) < POWER_PLANE_TOL * float(np.max(self._L))):
            raise TessellationError(
                f"Power cell of grain {gid}: a neighbour seed coincides with "
                "the home seed (zero-separation bisector). Grain centers must "
                "be distinct.")
        b = (np.sum(cj * cj, axis=1) - float(c_i @ c_i) - wj + w_i)
        # scipy format: rows [A | -b] meaning A·x + (-b) ≤ 0
        halfspaces = np.column_stack([a, -b])

        x_int, cheby_r = _chebyshev_center(a, b)
        if x_int is None or cheby_r <= 0.0:
            raise TessellationError(
                f"Power cell of grain {gid} is empty or degenerate "
                f"(Chebyshev radius {cheby_r!r}). The weight spread is too "
                "large for this seed configuration (SDOT damping should "
                "prevent this — check target volumes).")

        try:
            hs = HalfspaceIntersection(halfspaces, x_int)
        except QhullError as e:
            raise TessellationError(
                f"Power cell of grain {gid}: halfspace intersection degenerate "
                f"({e}). SDOT damping should prevent this — check target volumes."
            ) from e
        verts = hs.intersections                       # (Nv, 3) absolute

        plane_tol = POWER_PLANE_TOL * float(np.max(self._L))
        faces: list[FlatFace] = []
        norms_a = np.linalg.norm(a, axis=1)
        # vertices on plane k: |a_k·v − b_k| ≤ ‖a_k‖·plane_tol
        residual = verts @ a.T - b[None, :]            # (Nv, Nc)
        on_plane = np.abs(residual) <= norms_a[None, :] * plane_tol

        for k in range(len(cand)):
            vk = verts[on_plane[:, k]]
            if len(vk) < 3:
                continue
            unit_normal = a[k] / norms_a[k]
            verts_ordered = _order_polygon(vk, unit_normal)
            area = _polygon_area(verts_ordered, unit_normal)
            if area <= 0.0:
                continue
            rep = int(cand[k])
            neighbor_gid = (WALL_NEIGHBOR_ID if self._is_mirror[rep]
                            else int(self._home_ids[rep]))
            faces.append(FlatFace(
                neighbor_id=neighbor_gid,
                vertices=verts_ordered,
                area=area,
                unit_normal=unit_normal,
                seed_distance=float(np.linalg.norm(cj[k] - c_i)),
            ))

        # Store local competitor set: home lifted point + all candidate lifted
        # points (the pruned candidate set is a guaranteed superset of the true
        # face-neighbours, so the exactness proof applies — see module docstring
        # and the owns() implementation).
        self._cell_nbr_idx[gid] = np.unique(
            np.concatenate(
                [np.array([home_idx], dtype=np.int64),
                 cand.astype(np.int64)]
            )
        )

        vol = _cell_volume_interior(faces, x_int)
        return FlatCell(
            grain_id=gid,
            vertices=verts - self._seeds[gid],
            faces=faces,
            volume=vol,
        )

    def _adjacency_from_faces(self) -> list[tuple[int, int]]:
        adj: set[tuple[int, int]] = set()
        for cell in self._cells:
            for face in cell.faces:
                j = face.neighbor_id
                if j >= 0 and j != cell.grain_id:
                    adj.add((min(cell.grain_id, j), max(cell.grain_id, j)))
        return sorted(adj)

    # ------------------------------------------------------------------
    # Membership (4D lifted tree) — overrides the Voronoi-tree versions
    # ------------------------------------------------------------------

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        """Power-nearest grain (min-image on periodic axes; mirrors never
        win inside the box — see module docstring)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        Xw = X.copy()
        for ax in range(3):
            if self._periodic[ax]:
                Xw[:, ax] = Xw[:, ax] % self._L[ax]
                col = Xw[:, ax]
                col[col >= self._L[ax]] = 0.0
        q = np.column_stack([Xw, np.zeros(len(Xw))])
        _, idx = self._tree4.query(q)
        return self._home_ids[np.asarray(idx)].astype(np.int32)

    def _owns_global(self, X: np.ndarray, i: int) -> np.ndarray:
        """O(N log N) reference implementation of owns() using the global 4D KDTree.

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

        q = np.column_stack([X, np.zeros(len(X))])
        dists, idxs = self._tree4.query(q, k=2)
        d1 = dists[:, 0]
        d_home = np.linalg.norm(q - self._lifted[home_idx], axis=1)

        out = np.zeros(len(X), dtype=bool)
        clear_win = (idxs[:, 0] == home_idx) & (dists[:, 1] - d1 > tol)
        clear_loss = (idxs[:, 0] != home_idx) & (d_home - d1 > tol)
        out[clear_win] = True

        ambiguous = np.where(~(clear_win | clear_loss))[0]
        for j in ambiguous:
            cand = self._tree4.query_ball_point(q[j], r=d1[j] + tol)
            dd = np.linalg.norm(self._lifted[cand] - q[j], axis=1)
            # Wall MIRROR replicas clip the cell but never own atoms (a real
            # replica always ties them at the wall), so exclude them from the
            # tie-break — matching local_owns' is_mirror handling.
            owner = min(int(c) for c, d in zip(cand, dd, strict=True)
                        if d <= d1[j] + tol and not self._is_mirror[c])
            out[j] = (owner == home_idx)
        return out

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact home-cell ownership in the power metric (exact tiling
        of the torus; ties → lowest replica index).

        Fast O(rows · deg) path: queries only the LOCAL competitor set of
        grain i — its home lifted point plus every candidate replica lifted
        point stored in self._cell_nbr_idx[i] at construction time.

        Exactness: in the 4D lifting the power diagram is a Euclidean Voronoi
        diagram of the lifted replicas, so the flat-backend proof carries over
        verbatim (module docstring): only face-neighbours can beat the home
        replica, and they are all in the local set.  The 4th coordinate
        (sqrt(w_max − w_j)) is translation-invariant, so the local set is the
        same in 3D and 4D.  The lowest-global-replica-index tie-break is
        reproduced exactly by taking the minimum of the within-tolerance indices.

        Implementation: lifts the query points to 4D (4th coordinate 0, as in
        grain_of) and delegates the per-row distance/argmin to the shared
        :func:`grainsmith.tessellation._local_owns.local_owns` driver — chunked
        NumPy with an optional, bit-identical Numba kernel.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        rows = len(X)
        if rows == 0:
            return np.zeros(0, dtype=bool)

        home_idx = int(self._identity_block * self._n + i)
        nbr = self._cell_nbr_idx[i]            # (deg,) int64 replica indices
        # Lifted 4D points for the local competitor set and the query points
        # (4th coordinate 0 — same lifting as grain_of).
        P = self._lifted[nbr]                   # (deg, 4)
        Q = np.column_stack([X, np.zeros(rows)])  # (rows, 4)
        tol = OWNS_TIE_TOL * float(np.max(self._L))
        return local_owns(Q, P, nbr, home_idx, tol, len(self._lifted),
                          _OWNS_CHUNK, is_mirror=self._is_mirror[nbr])

    # ------------------------------------------------------------------

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        """Certified GB-shell mask via the exact per-grain face-plane
        margin (``flat._exact_margin_shell_mask``), UNCONDITIONALLY --
        NOT ``FlatTessellation``'s inherited fully-periodic k-NN fast
        path.  The power (Laguerre) generalized distance ``‖x-c‖² - w``
        is quadratic, not 1-Lipschitz (its Lipschitz constant grows
        without bound as ``‖x-c‖`` grows), so the plain-Euclidean k-NN
        certificate ``FlatTessellation`` uses for zero-weight cells does
        NOT carry over to nonzero power weights -- a wrong margin here
        would silently drop true pairs, not merely cost extra time.
        Power's cells ARE still convex polytopes (built via halfspace
        intersection -- module docstring), so the support-hyperplane
        argument ``_exact_margin_shell_mask`` relies on applies exactly as
        it does for Flat; only the (cheaper, Lipschitz-1-only) k-NN
        shortcut is unavailable here, workers is unused for the same
        reason FlatTessellation's fallback path ignores it: this margin
        computation has no tree query to parallelize.
        """
        return _exact_margin_shell_mask(self, pos, grain, cutoff)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()


def _chebyshev_center(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Largest inscribed ball of {x : a·x ≤ b}: maximize r s.t.
    a_k·x + ‖a_k‖ r ≤ b_k.  Returns (center, radius) or (None, -inf)."""
    norms = np.linalg.norm(a, axis=1)
    A_ub = np.column_stack([a, norms])
    c = np.array([0.0, 0.0, 0.0, -1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b,
                  bounds=[(None, None)] * 3 + [(0.0, None)],
                  method="highs")
    if not res.success:
        return None, float("-inf")
    return res.x[:3], float(res.x[3])


def _cell_volume_interior(faces: list[FlatFace], x_int: np.ndarray) -> float:
    """Divergence-theorem volume with the INTERIOR point as base.

    The flat backend uses the seed as base point; a power cell need not
    contain its seed, so the Chebyshev center is used instead (every
    n̂·(centroid − x_int) is then positive for outward normals)."""
    vol = 0.0
    for face in faces:
        r = face.vertices.mean(axis=0) - x_int
        vol += face.area * float(np.dot(face.unit_normal, r))
    return vol / 3.0
