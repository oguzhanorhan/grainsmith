"""Weighted (additive and anisotropic) tessellation backends (§6.6, method M3/M4).

additive_weights: grain_of(x) = argmin_i (d_per(x, c_i) - w_i)  (Johnson-Mehl/Apollonius)
anisotropic:      grain_of(x) = argmin_i (x-c_i)^T M_i (x-c_i)  (GBPD family)

Both implement the full membership API (§3.3, §5):
- grain_of : torus membership (min-image), images folded onto home ids;
- owns     : compact home-cell membership over explicit periodic replicas
             (exact tiling for the atom fill, see Tessellation.owns);
- margin   : SIGNED, grain-specific boundary distance (positive inside
             grain i, negative outside);
- adjacency: measured on the actual curved diagram via the voxel grid
             (a flat-Voronoi proxy can mis-report curved adjacency).

Bounding radii are provable covers derived from the flat-Voronoi radii:
  additive : R_i ≤ max_j R_flat_j + (w_i − w_min)
             (x in cell i ⇒ d(x,c_i) ≤ d(x,c_j*) + w_i − w_j* with j* the
              flat owner of x, and d(x,c_j*) ≤ R_flat_j*),
  anisotropic: R_i ≤ (s_max,i / s_min,global) · max_j R_flat_j
             (sandwich |dr|/s_max ≤ √qf ≤ |dr|/s_min applied to both sides).

Numba acceleration of owns()
-----------------------------------------------------
``fill_grain`` (atoms/fill.py) has no ``cell_vertices_rel`` hook for either
backend here, so it enumerates a SPHERE-bounded candidate grid (§6.8) --
orders of magnitude more candidates than kept atoms for curved geometries
(measured: ~114:1 for anisotropic, ~6:1 for flat, on the Fig. 2 validation
system of the accompanying paper, at reduced scale) -- and pays the
full numpy ``owns()`` (a dense (C, B*N) distance matrix + ``argmin``) on
EVERY candidate. The dense evaluation is wasted work: rejection only
needs ONE replica that beats home, not all of them.

``_owns_kernel_aniso``/``_owns_kernel_additive`` below are optional numba
kernels expressing that short-circuit directly, exactly reproducing
``_WeightedBase.owns()``'s semantics (first-minimum ``argmin`` over
explicit replicas == home_idx) with two transformations that provably do
not change the result:

1. Short-circuit rejection: a point is rejected the instant ANY replica k
   beats home under the first-minimum tie rule (k < home_idx with
   d_k <= d_home, or k > home_idx with d_k < d_home) -- this is an exact
   restatement of "argmin != home_idx", not an approximation. Acceptance
   (the minority outcome -- see the candidate:kept ratios above) still
   requires the full scan, same asymptotic cost as before. Scan ORDER
   (``_build_scan_orders``: home first, then nearest-seed-first) affects
   only how quickly a rejection is found, never the outcome -- the
   accept/reject test at each step keys off the replica's fixed GLOBAL
   index vs. home_idx, not its position in the scan.
2. BOTH kernels compare in sqrt space, exactly like the numpy reference.
   A squared-space comparison for the anisotropic backend would be exact
   in REAL arithmetic (every M_i is SPD, sqrt is monotone) but is NOT
   exact in float64: IEEE-754 sqrt is a rounding map, and two quadratic
   forms one ulp apart frequently (~47% of adjacent-double pairs) round
   to the SAME double under sqrt. At such a pair the numpy path sees a
   TIE (first-minimum rule -> lowest replica index wins) while the
   squared-space path sees a strict inequality -- a genuine, engineered-
   counterexample-verified ownership flip on the exact tie surface
   (tests/test_owns_kernel.py::test_sqrt_rounding_collision_tie).
   ``_owns_kernel_aniso`` therefore takes ``np.sqrt`` of each quadratic
   form before comparing; the extra sqrt per candidate-replica pair is
   negligible next to the short-circuit's savings. The additive
   (Apollonius) backend's distance is (euclid - weight), where sqrt is
   structurally unavoidable, and ``_owns_kernel_additive`` has always
   computed it. Residual last-ulp difference in the quadratic-form
   SUMMATION order (einsum vs manual expansion) remains possible in
   principle exactly AT ownership ties; across 415,925,024 candidates from
   the Fig. 2 validation system it produced zero mask differences.

Numba is a SOFT dependency (see ``_HAVE_NUMBA``/``_USE_NUMBA`` below): if
unavailable, or if ``GRAINSMITH_NO_NUMBA`` is set, ``owns()`` falls back
to the exact numpy path in ``_WeightedBase.owns()`` -- kept verbatim as
the proven-correct reference implementation, so a numba-free environment
is always covered. ``grain_of()`` and
``margin()`` are UNCHANGED (numpy only): both need actual distance VALUES
(not just an argmin identity), so the short-circuit here does not apply
to them.

Exactness has been validated against the numpy reference -- random point
batteries, engineered tie cases, and a full replay of the Fig. 2
validation system's curved fill candidate grids -- see
tests/test_weighted.py / tests/test_owns_kernel.py's numba-vs-numpy
equality cases for the measured outcome (415,925,024 candidates checked
against a reduced-scale Cu curved-boundary validation tessellation,
0 mismatches; byte-identical polycrystal.data against
GRAINSMITH_NO_NUMBA=1 and across --jobs). The exactness argument above
depends on the short-circuit and the numpy reference agreeing at every
tie, so any mismatch a future change to this kernel introduces means that
argument no longer holds for the changed code; tests/test_owns_kernel.py's
opt-in fig2_validation marker (GRAINSMITH_RUN_FIG2_VALIDATION=1) re-runs
the full-system check above.
"""
from __future__ import annotations

import math
import os
from itertools import product as iproduct
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from grainsmith.errors import ConfigError
from grainsmith.seeding import wigner_seitz_radius
from grainsmith.tessellation._shell_knn import SHELL_KNN_K as _SHELL_KNN_K
from grainsmith.tessellation._shell_knn import certified_knn_shell_mask
from grainsmith.tessellation.base import Tessellation

if TYPE_CHECKING:
    from grainsmith.tessellation.voxel import VoxelGrid

_CHUNK = 50_000

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - environment without numba
    _HAVE_NUMBA = False


def _numba_disabled_by_env() -> bool:
    """``GRAINSMITH_NO_NUMBA=1`` (or ``true``/``yes``/``on``, case-
    insensitive) forces the numpy owns() path -- the explicit escape
    hatch for A/B validation and for CI environments without numba. Read
    once at import time into :data:`_USE_NUMBA`; nothing else in this
    module re-reads the environment."""
    return os.environ.get("GRAINSMITH_NO_NUMBA", "").strip().lower() in (
        "1", "true", "yes", "on")


_USE_NUMBA: bool = _HAVE_NUMBA and not _numba_disabled_by_env()
"""Module-level dispatch flag (mirrors tessellation/_local_owns.py's
_USE_NUMBA contract exactly): equality tests monkeypatch this directly to
force either path for a direct A/B comparison. Nothing in the engine
mutates it after import except the environment read above."""


if _HAVE_NUMBA:

    @numba.njit(nogil=True, cache=True)
    def _owns_kernel_aniso(X, rep_seeds, rep_metric_id, M, home_idx, scan_order):
        """Exact short-circuit ownership kernel, AnisotropicTessellation.

        Reproduces ``np.argmin(_wdist_replicas(X), axis=1) == home_idx``
        (this module's numpy reference): same per-replica metric
        (``rep_metric_id``), same first-minimum tie rule, and -- critically
        -- compared in SQRT space (``np.sqrt`` of the quadratic form, the
        same correctly-rounded IEEE sqrt numpy applies), NOT on raw squared
        forms. Squared-space comparison is NOT exact: two quadratic forms
        1 ulp apart can round to the SAME double under sqrt (~47% of
        adjacent-double pairs do), in which case numpy's sqrt-space argmin
        ties (first-minimum -> lowest replica index) while a squared-space
        comparison sees a strict inequality and can pick the other replica
        (regression-tested with an engineered collision pair in
        test_owns_kernel.py). The only remaining route difference is the
        summation ORDER inside the quadratic form (einsum vs this manual
        expansion), which can differ in the last ulp and matters only at
        exact ownership ties -- empirically zero mask mismatches across
        415,925,024 candidates from the Fig. 2 validation system.
        ``scan_order`` (home first, then
        nearest-seed-first,
        precomputed once per grain by :func:`_build_scan_orders`) is a
        pure speed heuristic -- the accept/reject test below keys off
        each replica's fixed global index ``k`` vs. ``home_idx``, never
        its position in the scan, so permuting ``scan_order`` can only
        change how fast a rejection is found, never the returned mask.

        Parameters
        ----------
        X : (C, 3) float64 query points (candidate lattice sites).
        rep_seeds : (B*N, 3) float64 all periodic replica seed positions.
        rep_metric_id : (B*N,) int64 grain id owning each replica's metric.
        M : (N, 3, 3) float64 stacked per-grain anisotropic metrics.
        home_idx : int64 global replica index of grain i's home (identity)
            block.
        scan_order : (B*N,) int64 replica visiting order (element 0 must
            be ``home_idx``'s row position -- see ``_build_scan_orders``).

        Returns
        -------
        (C,) bool -- True where grain i's home replica owns the point.
        """
        n_pts = X.shape[0]
        n_rep = rep_seeds.shape[0]
        out = np.empty(n_pts, dtype=np.bool_)
        home_k = scan_order[0]
        for r in range(n_pts):
            dx = X[r, 0] - rep_seeds[home_k, 0]
            dy = X[r, 1] - rep_seeds[home_k, 1]
            dz = X[r, 2] - rep_seeds[home_k, 2]
            mid = rep_metric_id[home_k]
            v0 = M[mid, 0, 0] * dx + M[mid, 0, 1] * dy + M[mid, 0, 2] * dz
            v1 = M[mid, 1, 0] * dx + M[mid, 1, 1] * dy + M[mid, 1, 2] * dz
            v2 = M[mid, 2, 0] * dx + M[mid, 2, 1] * dy + M[mid, 2, 2] * dz
            d_home = np.sqrt(v0 * dx + v1 * dy + v2 * dz)
            accept = True
            for si in range(1, n_rep):
                k = scan_order[si]
                dxk = X[r, 0] - rep_seeds[k, 0]
                dyk = X[r, 1] - rep_seeds[k, 1]
                dzk = X[r, 2] - rep_seeds[k, 2]
                mk = rep_metric_id[k]
                w0 = M[mk, 0, 0] * dxk + M[mk, 0, 1] * dyk + M[mk, 0, 2] * dzk
                w1 = M[mk, 1, 0] * dxk + M[mk, 1, 1] * dyk + M[mk, 1, 2] * dzk
                w2 = M[mk, 2, 0] * dxk + M[mk, 2, 1] * dyk + M[mk, 2, 2] * dzk
                d_k = np.sqrt(w0 * dxk + w1 * dyk + w2 * dzk)
                if k < home_idx:
                    if d_k <= d_home:
                        accept = False
                        break
                else:
                    if d_k < d_home:
                        accept = False
                        break
            out[r] = accept
        return out

    @numba.njit(nogil=True, cache=True)
    def _owns_kernel_additive(X, rep_seeds, rep_weights, home_idx, scan_order):
        """Exact short-circuit ownership kernel, WeightedTessellation
        (additive/Apollonius). Reproduces
        ``np.argmin(_wdist_replicas(X), axis=1) == home_idx`` exactly.
        Distances here are (euclid - weight): NOT squared-comparable (the
        per-replica additive weight breaks sqrt-monotonicity), so this
        kernel computes the true ``sqrt`` per replica -- only the
        short-circuit (see :func:`_owns_kernel_aniso`'s docstring) is
        exploited, not a squared-distance shortcut.

        Parameters
        ----------
        X : (C, 3) float64 query points.
        rep_seeds : (B*N, 3) float64 all periodic replica seed positions.
        rep_weights : (B*N,) float64 per-replica additive weight (tiled
            per-grain weight, same convention as ``self._rep_weights``).
        home_idx : int64 global replica index of grain i's home block.
        scan_order : (B*N,) int64 replica visiting order, home first.

        Returns
        -------
        (C,) bool -- True where grain i's home replica owns the point.
        """
        n_pts = X.shape[0]
        n_rep = rep_seeds.shape[0]
        out = np.empty(n_pts, dtype=np.bool_)
        home_k = scan_order[0]
        for r in range(n_pts):
            dx = X[r, 0] - rep_seeds[home_k, 0]
            dy = X[r, 1] - rep_seeds[home_k, 1]
            dz = X[r, 2] - rep_seeds[home_k, 2]
            d_home = math.sqrt(dx * dx + dy * dy + dz * dz) - rep_weights[home_k]
            accept = True
            for si in range(1, n_rep):
                k = scan_order[si]
                dxk = X[r, 0] - rep_seeds[k, 0]
                dyk = X[r, 1] - rep_seeds[k, 1]
                dzk = X[r, 2] - rep_seeds[k, 2]
                d_k = math.sqrt(dxk * dxk + dyk * dyk + dzk * dzk) - rep_weights[k]
                if k < home_idx:
                    if d_k <= d_home:
                        accept = False
                        break
                else:
                    if d_k < d_home:
                        accept = False
                        break
            out[r] = accept
        return out
else:  # pragma: no cover - environment without numba
    _owns_kernel_aniso = None
    _owns_kernel_additive = None


def _build_scan_orders(
    seeds: np.ndarray, rep_seeds: np.ndarray, identity_block: int, n: int
) -> np.ndarray:
    """(n, B*N) int64 scan-order table for the numba owns() kernels.

    Row i visits grain i's own home replica FIRST, then every other
    replica in ascending seed-to-seed distance from ``seeds[i]`` (ties --
    e.g. a symmetric seed layout -- broken by ascending global replica
    index, via ``argsort``'s stable sort over the natural 0..B*N-1
    order). Pure speed heuristic: a competing seed geometrically close to
    seed i is the one most likely to reject a candidate near grain i's
    own cell boundary, so visiting close competitors first maximizes the
    short-circuit hit rate. Provably CANNOT change either kernel's
    returned mask (see their docstrings -- the accept/reject test keys
    off each replica's fixed global index, not scan position).

    Computed once per grain at tessellation construction (O(n) argsorts
    of B*N elements each -- negligible next to field synthesis / the G5
    connectivity check), reused by every subsequent owns() call and every
    fill chunk for that grain.
    """
    n_rep = len(rep_seeds)
    orders = np.empty((n, n_rep), dtype=np.int64)
    for i in range(n):
        home_idx = identity_block * n + i
        d = np.linalg.norm(rep_seeds - seeds[i][None, :], axis=1)
        d[home_idx] = -1.0  # force home to sort first
        orders[i] = np.argsort(d, kind="stable")
    return orders


def _periodic_translations(
    L: np.ndarray, periodic: list[bool]
) -> tuple[np.ndarray, int]:
    """All {-1,0,1}-shift translation vectors on periodic axes (free axes
    get only 0).  Returns (offsets (B,3), identity_block_index)."""
    opts = [([-1.0, 0.0, 1.0] if periodic[ax] else [0.0]) for ax in range(3)]
    blocks = list(iproduct(*opts))
    identity_block = blocks.index((0.0, 0.0, 0.0))
    offsets = np.array(blocks, dtype=np.float64) * L[None, :]
    return offsets, identity_block


class _WeightedBase(Tessellation):
    """Shared scaffolding for the two function-based curved backends."""

    _seeds: np.ndarray
    _L: np.ndarray
    _periodic: list[bool]
    _n: int

    def _init_common(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
    ) -> None:
        self._seeds = np.asarray(seeds, dtype=np.float64)
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        self._n = len(self._seeds)
        self._r_ws = wigner_seitz_radius(float(np.prod(self._L)), self._n)

        offsets, identity_block = _periodic_translations(self._L, self._periodic)
        # Replica layout: index = block·N + gid  (same convention as flat.py)
        self._rep_seeds = (self._seeds[None, :, :] + offsets[:, None, :]
                           ).reshape(-1, 3)
        self._identity_block = identity_block
        self._adj: list[tuple[int, int]] | None = None
        self._voxel: VoxelGrid | None = None

        # Per-grain scan-order table for the numba owns() kernels (module
        # docstring's "Numba acceleration" section) — cheap (O(n) argsorts
        # of B·N elements), computed unconditionally so toggling
        # GRAINSMITH_NO_NUMBA / _USE_NUMBA at RUNTIME (as the equality
        # tests do) never needs a rebuild of the tessellation.
        self._scan_orders = _build_scan_orders(
            self._seeds, self._rep_seeds, identity_block, self._n)

    def _flat_radii(self) -> np.ndarray:
        """Per-grain flat-Voronoi circumradii (basis for the curved bounds)."""
        from grainsmith.tessellation.flat import FlatTessellation
        ft = FlatTessellation(self._seeds, self._L, self._periodic)
        return np.array([ft.bounding_radius(i) for i in range(self._n)],
                        dtype=np.float64)

    def _min_image(self, diff: np.ndarray) -> np.ndarray:
        """Min-image reduction of (..., 3) difference vectors (periodic axes)."""
        for ax in range(3):
            if self._periodic[ax]:
                diff[..., ax] -= np.round(diff[..., ax] / self._L[ax]) * self._L[ax]
        return diff

    # --- distance kernels supplied by subclasses --------------------------

    def _wdist_home(self, X: np.ndarray) -> np.ndarray:
        """(C, N) generalized distances to the home seeds under min-image."""
        raise NotImplementedError

    def _wdist_replicas(self, X: np.ndarray) -> np.ndarray:
        """(C, B·N) generalized distances to all explicit replicas (no
        min-image — replicas are explicit)."""
        raise NotImplementedError

    # --- shared API implementations ---------------------------------------

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        result = np.empty(len(X), dtype=np.int32)
        for start in range(0, len(X), _CHUNK):
            d = self._wdist_home(X[start:start + _CHUNK])
            # np.argmin takes the first minimum → lowest grain id on ties
            result[start:start + _CHUNK] = np.argmin(d, axis=1).astype(np.int32)
        return result

    # --- optional numba fast path (module docstring's "Numba
    # acceleration" section) — subclasses that ship a kernel override
    # this; the base implementation (None) means "no fast path, always
    # use the numpy _wdist_replicas/argmin reference below". ---------

    def _owns_numba(self, X: np.ndarray, home_idx: int,
                    scan_order: np.ndarray) -> np.ndarray | None:
        """Return the numba-computed owns() mask for chunk *X*, or None
        if this backend has no kernel / numba is unavailable / the numpy
        path was explicitly requested. Base implementation always
        returns None; :class:`AnisotropicTessellation` and
        :class:`WeightedTessellation` override it."""
        return None

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact home-cell ownership over explicit replicas (§6.8).

        Exact for the same reason as FlatTessellation.owns: the per-replica
        generalized distance is the Euclidean distance (shifted or scaled by
        shift-independent grain quantities), so beyond the {-1,0,1} shells
        it grows monotonically per axis and the replica set always contains
        the global minimizer on the owns-candidate region.  np.argmin's
        first-minimum rule gives the deterministic lowest-replica tie-break.

        Dispatches to a numba short-circuit kernel when available (module
        docstring's "Numba acceleration" section) — same exact semantics,
        proven never to change the returned mask; falls back to the numpy
        reference below when numba is unavailable, disabled
        (``GRAINSMITH_NO_NUMBA``), or the subclass has no kernel.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.zeros(0, dtype=bool)
        home_idx = self._identity_block * self._n + i
        if _USE_NUMBA:
            scan_order = self._scan_orders[i]
            fast = self._owns_numba(X, home_idx, scan_order)
            if fast is not None:
                return fast
        out = np.empty(len(X), dtype=bool)
        for start in range(0, len(X), _CHUNK):
            d = self._wdist_replicas(X[start:start + _CHUNK])
            out[start:start + _CHUNK] = (np.argmin(d, axis=1) == home_idx)
        return out

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed boundary distance for grain i (Å-scaled, §5).

        margin = (min_{j≠i} d_j − d_i) / 2 : positive iff x lies in grain i,
        zero on its boundary, negative outside.  For the additive backend
        this is exactly half the weighted-distance deficit; for the
        anisotropic backend distances are √(quadratic form) so the value
        stays Å-scaled."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), _CHUNK):
            d = self._wdist_home(X[start:start + _CHUNK])
            d_i = d[:, i].copy()
            d[:, i] = np.inf
            d_other = d.min(axis=1)
            out[start:start + _CHUNK] = (d_other - d_i) / 2.0
        return out

    def voxel_grid(self):
        """Lazily-built voxel grid of this curved diagram (shared by the
        G5 connectivity check and adjacency)."""
        if self._voxel is None:
            from grainsmith.tessellation.voxel import build_voxel_grid
            self._voxel = build_voxel_grid(self, self._L, "auto",
                                           r_ws=self._r_ws)
        return self._voxel

    def adjacency(self) -> list[tuple[int, int]]:
        """Adjacency measured on the actual curved diagram (voxel grid)."""
        if self._adj is None:
            self._adj = self.voxel_grid().adjacency(self._periodic)
        return list(self._adj)

    @property
    def seeds(self) -> np.ndarray:
        return self._seeds.copy()

    @property
    def n_grains(self) -> int:
        return self._n


class WeightedTessellation(_WeightedBase):
    """Additively-weighted (Johnson-Mehl / Apollonius) tessellation.

    grain_of(x) = argmin_i ( d_per(x, c_i) - w_i )
    where d_per is the periodic Euclidean distance (plain distance, NOT
    squared — squared-minus-weight would be a Laguerre diagram with flat
    faces).  Grain weights w_i ~ N(0, sigma_w²).

    Empty-cell guard: sigma_w ≤ min_seed_distance/6 (§6.6); defaults to
    r_ws/6 when no min_seed_distance is supplied.
    """

    def __init__(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
        weights: np.ndarray | None = None,
        sigma_w: float = 0.0,
        rng: np.random.Generator | None = None,
        min_seed_distance: float | None = None,
        connectivity_check: bool = True,
        *,
        memory_limit_bytes: float | None = None,
        memory_limit_source: str = "config",
    ) -> None:
        # §13: stamp the per-run memory budget FIRST, before the
        # connectivity_check below can build a voxel grid at construction
        # time (see tessellation/base.py's Tessellation.memory_limit_bytes
        # docstring). Only overwrite the class default when the caller
        # actually passed one, so direct construction (tests,
        # analysis/grains.py) keeps the 16 GB class default.
        if memory_limit_bytes is not None:
            self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_source = memory_limit_source

        self._init_common(seeds, box_lengths, periodic)
        n = self._n

        if weights is not None:
            self._weights = np.asarray(weights, dtype=np.float64)
        elif sigma_w > 0.0:
            limit = (min_seed_distance if min_seed_distance is not None
                     else self._r_ws) / 6.0
            if sigma_w > limit:
                raise ConfigError(
                    f"weight_sigma={sigma_w:.4g} Å > min_seed_distance/6 = "
                    f"{limit:.4g} Å. Reduce weight_sigma to avoid empty cells."
                )
            if rng is None:
                raise ConfigError("rng required when sigma_w > 0")
            self._weights = rng.normal(0.0, sigma_w, size=n)
        else:
            self._weights = np.zeros(n)

        n_blocks = len(self._rep_seeds) // n
        self._rep_weights = np.tile(self._weights, n_blocks)
        self._shell_tree: KDTree | None = None

        flat_r = self._flat_radii()
        w_min = float(np.min(self._weights))
        self._bound_radii = float(np.max(flat_r)) + np.maximum(
            self._weights - w_min, 0.0
        )

        # Gate G5 (§6.6): connectivity + empty-cell guard on the actual
        # weighted diagram.
        if connectivity_check:
            self.voxel_grid().check_connectivity(self._periodic)

    def _wdist_home(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self._seeds[None, :, :]
        diff = self._min_image(diff)
        return np.sqrt(np.sum(diff**2, axis=2)) - self._weights[None, :]

    def _wdist_replicas(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self._rep_seeds[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2)) - self._rep_weights[None, :]

    def _owns_numba(self, X: np.ndarray, home_idx: int,
                    scan_order: np.ndarray) -> np.ndarray | None:
        if _owns_kernel_additive is None:
            return None
        return np.asarray(
            _owns_kernel_additive(X, self._rep_seeds, self._rep_weights,
                                  home_idx, scan_order),
            dtype=bool)

    def _shell_replica_tree(self) -> KDTree:
        """Lazily-built plain (non-periodic) cKDTree over ALL ``B*N``
        periodic replica seeds (``self._rep_seeds``), used only by
        :meth:`gb_shell_lower_bound`'s k-NN certificate.  Built once (the
        replica seed positions never change after construction) and cached
        on the instance."""
        if self._shell_tree is None:
            self._shell_tree = KDTree(self._rep_seeds)
        return self._shell_tree

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        """Certified GB-shell lower bound, additive (Johnson-Mehl /
        Apollonius) diagram (see ``Tessellation.gb_shell_lower_bound`` and
        ``tessellation._shell_knn`` for the shared k-NN certificate this
        delegates to).

        Evaluating the replica-aware margin DENSELY over every replica
        costs O(N · B·N) (measured 36.8 s single-threaded at 7.3M atoms x
        140 seeds -- MORE than the overlap round it would accelerate); the
        shared k-NN certificate in ``_shell_knn`` is what makes this cheap.

        ``grain`` is accepted for interface parity with the abstract
        method but UNUSED here: home ownership is determined by the
        queried generalized-distance argmin over replicas, not by each
        atom's recorded grain id.
        """
        return certified_knn_shell_mask(
            pos, self._shell_replica_tree(), self._rep_weights, cutoff,
            workers, k=_SHELL_KNN_K)

    def bounding_radius(self, i: int) -> float:
        return float(self._bound_radii[i])

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()


class AnisotropicTessellation(_WeightedBase):
    """Anisotropic (GBPD) tessellation (§6.6, method M4).

    grain_of(x) = argmin_i (x-c_i)^T M_i (x-c_i)
    M_i = R_i^T diag(1/s_x², 1/s_y², 1/s_z²) R_i
    with independent random shape rotations R_i and per-grain semi-axis
    scalings s ∈ aspect_ratio_range.  Distances used in margin/owns are
    √(quadratic form) so that all reported quantities stay Å-scaled.
    """

    def __init__(
        self,
        seeds: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
        metrics: list[np.ndarray] | None = None,
        aspect_ratio_range: tuple[float, float] = (1.0, 2.0),
        rng: np.random.Generator | None = None,
        connectivity_check: bool = True,
        *,
        memory_limit_bytes: float | None = None,
        memory_limit_source: str = "config",
    ) -> None:
        # §13: stamp the per-run memory budget FIRST, before the
        # connectivity_check below can build a voxel grid at construction
        # time (see tessellation/base.py's Tessellation.memory_limit_bytes
        # docstring). Only overwrite the class default when the caller
        # actually passed one, so direct construction (tests,
        # analysis/grains.py) keeps the 16 GB class default.
        if memory_limit_bytes is not None:
            self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_source = memory_limit_source

        self._init_common(seeds, box_lengths, periodic)
        n = self._n

        if metrics is not None:
            self._M = [np.asarray(m, dtype=np.float64) for m in metrics]
        else:
            if rng is None:
                raise ConfigError("rng required for random anisotropic metrics")
            self._M = self._random_metrics(n, aspect_ratio_range, rng)

        # Stacked (n, 3, 3) metric array + per-replica metric-owner id, for
        # the numba owns() kernel (module docstring's "Numba acceleration"
        # section) — same self._M list, just laid out contiguously for
        # njit consumption. Built unconditionally (cheap: n 3x3 blocks);
        # unused when numba is unavailable/disabled.
        self._M_stack = np.stack(self._M, axis=0)
        n_blocks = len(self._rep_seeds) // n
        self._rep_metric_id = np.tile(np.arange(n, dtype=np.int64), n_blocks)

        # Semi-axis scalings from the metric spectra: eig(M) = 1/s².
        eigs = np.array([np.linalg.eigvalsh(M) for M in self._M])  # (n, 3) asc
        s_max_i = 1.0 / np.sqrt(eigs[:, 0])       # largest semi-axis per grain
        s_min_global = float(np.min(1.0 / np.sqrt(eigs[:, 2])))

        flat_r = self._flat_radii()
        self._bound_radii = (s_max_i / s_min_global) * float(np.max(flat_r))

        # Gate G5 (§6.6): connectivity + empty-cell guard.
        if connectivity_check:
            self.voxel_grid().check_connectivity(self._periodic)

    @staticmethod
    def _random_metrics(
        n: int,
        aspect_ratio_range: tuple[float, float],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        from scipy.spatial.transform import Rotation
        metrics = []
        lo, hi = aspect_ratio_range
        for _ in range(n):
            s = rng.uniform(lo, hi, size=3)
            R = Rotation.random(random_state=rng).as_matrix()
            D = np.diag(1.0 / s**2)
            metrics.append(R.T @ D @ R)
        return metrics

    def _metric_dists(self, X: np.ndarray, seeds: np.ndarray,
                      metric_ids: np.ndarray, min_image: bool) -> np.ndarray:
        """(C, len(seeds)) anisotropic distances √((x−c)ᵀ M (x−c))."""
        d = np.empty((len(X), len(seeds)), dtype=np.float64)
        for j in range(len(seeds)):
            dr = X - seeds[j][None, :]
            if min_image:
                dr = self._min_image(dr)
            Mj = self._M[int(metric_ids[j])]
            d[:, j] = np.einsum('ni,ij,nj->n', dr, Mj, dr)
        return np.sqrt(d)

    def _wdist_home(self, X: np.ndarray) -> np.ndarray:
        ids = np.arange(self._n)
        return self._metric_dists(X, self._seeds, ids, min_image=True)

    def _wdist_replicas(self, X: np.ndarray) -> np.ndarray:
        n_blocks = len(self._rep_seeds) // self._n
        ids = np.tile(np.arange(self._n), n_blocks)
        return self._metric_dists(X, self._rep_seeds, ids, min_image=False)

    def _owns_numba(self, X: np.ndarray, home_idx: int,
                    scan_order: np.ndarray) -> np.ndarray | None:
        if _owns_kernel_aniso is None:
            return None
        return np.asarray(
            _owns_kernel_aniso(X, self._rep_seeds, self._rep_metric_id,
                              self._M_stack, home_idx, scan_order),
            dtype=bool)

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> None:
        """No certified GB-shell bound: the anisotropic generalized distance
        ``sqrt((x-c)^T M (x-c))`` has Lipschitz constant ``1/s_min``
        (``s_min`` the smallest semi-axis scaling across all grains, from
        the metric spectrum), which is ``> 1`` for any non-degenerate
        aspect ratio (see ``AnisotropicTessellation``'s docstring:
        ``aspect_ratio_range`` defaults to ``(1.0, 2.0)``). A Lipschitz-1
        margin filter applied to a Lipschitz-(1/s_min) distance can
        OVERESTIMATE the true Euclidean margin and silently drop atoms
        whose true pair partner lies outside the shell. Always returns
        ``None`` (unchanged full-N overlap behaviour)."""
        return None

    def bounding_radius(self, i: int) -> float:
        return float(self._bound_radii[i])
