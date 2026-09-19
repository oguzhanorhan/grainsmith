"""Per-boundary GB curvature.

Local mean (H, 1/Å) and Gaussian (K, 1/Å²) curvature of the boundary
between grains i < j, from the level set φ = margin_i − margin_j
(margin is positive INSIDE its grain, §6.7).  The unit normal
n̂ = −∇φ/|∇φ| points from i toward j, and

    H = (∇φᵀ·Hφ·∇φ − |∇φ|²·tr Hφ) / (2|∇φ|³)     [Goldman 2005]
    K = (∇φᵀ·adj(Hφ)·∇φ) / |∇φ|⁴

so H > 0 ⇔ the center of curvature lies on the grain-i side ⇔ grain i
is locally convex (a spherical grain i of radius R has H = +1/R,
K = +1/R²).  K is invariant under the normal flip.

Derivatives are 19-point central-difference stencils with per-axis
steps h_vec, evaluated ONLY at the boundary sample points (no full-grid
φ — contrast gb_mesh).  margin() min-images internally, so stencil
points may leave [0, L).  Samples with |∇φ| < CURV_GRAD_MIN are
degenerate: dropped, counted, and reported through gate G16.

Flat runs never reach this kernel: FlatTessellation.margin is piecewise
planar (kinks at face edges), so the flat path writes literal zeros
from the exact face polygons instead (H = K = 0 by construction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grainsmith.constants import CURV_GRAD_MIN
from grainsmith.errors import GrainsmithError

# 19-point stencil: center, 6 axis neighbours, 12 edge neighbours
# (the 3×3×3 neighbourhood minus the 8 corners the central differences
# never touch).
_STENCIL = np.array([
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
], dtype=np.float64)
_C = 0
_P = {0: 1, 1: 3, 2: 5}                                # +e_a
_M = {0: 2, 1: 4, 2: 6}                                # −e_a
_PP = {(0, 1): 7, (0, 2): 11, (1, 2): 15}              # +e_a +e_b
_PM = {(0, 1): 8, (0, 2): 12, (1, 2): 16}              # +e_a −e_b
_MP = {(0, 1): 9, (0, 2): 13, (1, 2): 17}              # −e_a +e_b
_MM = {(0, 1): 10, (0, 2): 14, (1, 2): 18}             # −e_a −e_b


def _adjugate3(A: np.ndarray) -> np.ndarray:
    """Adjugate of a batch of 3×3 matrices, (M, 3, 3) → (M, 3, 3)."""
    adj = np.empty_like(A)
    for r in range(3):
        for c in range(3):
            i1, i2 = [k for k in range(3) if k != c]
            j1, j2 = [k for k in range(3) if k != r]
            minor = (A[:, i1, j1] * A[:, i2, j2]
                     - A[:, i1, j2] * A[:, i2, j1])
            adj[:, r, c] = (-1.0) ** (r + c) * minor
    return adj


def _hk_from_stencil(
    phi: np.ndarray, h_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, K, ok) from φ stencil values.

    Parameters
    ----------
    phi : (M, 19) float64 — φ at the _STENCIL offsets (scaled by h_vec).
    h_vec : (3,) float64 — per-axis steps (Å).

    Returns
    -------
    H (M,), K (M,) float64 (0.0 where degenerate), ok (M,) bool.
    """
    m = phi.shape[0]
    g = np.empty((m, 3), dtype=np.float64)
    hess = np.empty((m, 3, 3), dtype=np.float64)
    for a in range(3):
        g[:, a] = (phi[:, _P[a]] - phi[:, _M[a]]) / (2.0 * h_vec[a])
        hess[:, a, a] = ((phi[:, _P[a]] - 2.0 * phi[:, _C]
                          + phi[:, _M[a]]) / h_vec[a] ** 2)
    for (a, b), ipp in _PP.items():
        mixed = ((phi[:, ipp] - phi[:, _PM[(a, b)]]
                  - phi[:, _MP[(a, b)]] + phi[:, _MM[(a, b)]])
                 / (4.0 * h_vec[a] * h_vec[b]))
        hess[:, a, b] = mixed
        hess[:, b, a] = mixed

    g2 = np.einsum("ma,ma->m", g, g)
    gnorm = np.sqrt(g2)
    ok = gnorm > CURV_GRAD_MIN
    H = np.zeros(m, dtype=np.float64)
    K = np.zeros(m, dtype=np.float64)
    if np.any(ok):
        gHg = np.einsum("ma,mab,mb->m", g[ok], hess[ok], g[ok])
        tr = np.einsum("maa->m", hess[ok])
        gAg = np.einsum("ma,mab,mb->m", g[ok], _adjugate3(hess[ok]), g[ok])
        H[ok] = (gHg - g2[ok] * tr) / (2.0 * gnorm[ok] ** 3)
        K[ok] = gAg / gnorm[ok] ** 4
    return H, K, ok


_CHUNK = 50_000
"""Samples per margin batch (19 stencil points each) — memory guard."""


def _pair_curvature(
    tess, i: int, j: int, pts: np.ndarray, h_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, K, ok) at the boundary sample points of pair (i, j)."""
    h_vec = np.asarray(h_vec, dtype=np.float64)
    offsets = _STENCIL * h_vec[None, :]
    n = len(pts)
    H = np.empty(n, dtype=np.float64)
    K = np.empty(n, dtype=np.float64)
    ok = np.empty(n, dtype=bool)
    for s in range(0, n, _CHUNK):
        p = pts[s:s + _CHUNK]
        X = (p[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        phi = (tess.margin(X, i) - tess.margin(X, j)).reshape(len(p), 19)
        H[s:s + len(p)], K[s:s + len(p)], ok[s:s + len(p)] = \
            _hk_from_stencil(phi, h_vec)
    return H, K, ok


@dataclass
class PairCurvature:
    """Retained local curvature samples of one (i, j) boundary."""
    points: np.ndarray   # (M, 3) f64 lab Å, wrapped to [0, L)
    areas: np.ndarray    # (M,) f64 Å² (voxel-face / exact-polygon weight)
    H: np.ndarray        # (M,) f64 1/Å — H > 0: grain i locally convex
    K: np.ndarray        # (M,) f64 1/Å²


@dataclass
class CurvatureResult:
    """All pairs' samples + the degenerate-sample bookkeeping (G16)."""
    pairs: dict[tuple[int, int], PairCurvature]  # tess.adjacency() order
    n_raw: int
    n_dropped: int

    @property
    def dropped_fraction(self) -> float:
        return self.n_dropped / self.n_raw if self.n_raw else 0.0


_EMPTY = (np.empty((0, 3)), np.empty(0))


def _wrap(points: np.ndarray, box_lengths, periodic) -> np.ndarray:
    """Wrap coordinates to [0, L) on periodic axes (x == L → 0.0)."""
    w = np.array(points, dtype=np.float64, copy=True)
    for ax in range(3):
        if periodic[ax]:
            length = float(box_lengths[ax])
            w[:, ax] %= length
            w[w[:, ax] >= length, ax] = 0.0
    return w


def _face_centroid(vertices: np.ndarray) -> np.ndarray:
    """Exact area centroid of a planar convex polygon (fan triangulation).

    The vertex mean is biased toward vertex-dense edges; the spec's flat
    contract promises the exact face centroid.
    """
    v0 = vertices[0]
    cent = np.zeros(3, dtype=np.float64)
    # np.float64, not a bare 0.0: `a` below is np.floating (np.linalg.norm),
    # so after the first iteration `total` already IS an np.float64 at
    # runtime — initialising it as one changes no arithmetic, it just states
    # the type the accumulator actually has.
    total = np.float64(0.0)
    for k in range(1, len(vertices) - 1):
        tri_cent = (v0 + vertices[k] + vertices[k + 1]) / 3.0
        a = 0.5 * np.linalg.norm(
            np.cross(vertices[k] - v0, vertices[k + 1] - v0))
        cent += a * tri_cent
        total += a
    return cent / total if total > 0.0 else vertices.mean(axis=0)


def analyze_curvature(tess, box_lengths, periodic,
                      vg=None, jobs: int = 1) -> CurvatureResult:
    """Local (H, K) per boundary; dispatch mirrors analyze_boundaries.

    Flat backend (incl. power — planar boundaries): literal zeros, one
    sample per exact face polygon.  Everything else: level-set stencils
    at the voxel-face sample points of *vg*.  Empty adjacency (single
    crystal) returns an empty result before any dispatch — in the flat
    single-crystal case *vg* is legitimately None.

    jobs : worker processes for the voxel path's per-(i, j)-pair
        stencils (§13; CLI ``--jobs``). 1 = in-process serial (default;
        the flat path is always serial regardless — see _curvature_flat,
        trivial per-face work). Each pair's (H, K, ok) is a pure function
        of (tess, i, j, its own sample points, h_vec), and results are
        reassembled positionally in tess.adjacency() order, so n_raw/
        n_dropped and every pair's H/K/areas/points array are
        bit-identical for every jobs value — same guarantee and argument
        as atoms.fill.fill_grains's jobs.
    """
    if not tess.adjacency():
        return CurvatureResult(pairs={}, n_raw=0, n_dropped=0)
    from grainsmith.tessellation.flat import FlatTessellation
    if isinstance(tess, FlatTessellation):
        return _curvature_flat(tess, box_lengths, periodic)
    return _curvature_voxel(tess, box_lengths, periodic, vg, jobs=jobs)


def _curvature_flat(tess, box_lengths, periodic) -> CurvatureResult:
    pairs: dict[tuple[int, int], PairCurvature] = {}
    n_raw = 0
    for (i, j) in tess.adjacency():
        cents, areas = [], []
        for face in tess.cells[i].faces:
            if face.neighbor_id == j:
                cents.append(_face_centroid(face.vertices))
                areas.append(face.area)
        pts = np.array(cents, dtype=np.float64).reshape(-1, 3)
        n = len(pts)
        n_raw += n
        pairs[(i, j)] = PairCurvature(
            points=_wrap(pts, box_lengths, periodic),
            areas=np.array(areas, dtype=np.float64),
            H=np.zeros(n, dtype=np.float64),
            K=np.zeros(n, dtype=np.float64))
    return CurvatureResult(pairs=pairs, n_raw=n_raw, n_dropped=0)


def _assemble_pairs(
    adjacency, samples, results, box_lengths, periodic,
) -> tuple[dict[tuple[int, int], PairCurvature], int, int]:
    """Shared per-pair post-processing: mask by `ok`, wrap points to the
    box, build each PairCurvature, and accumulate n_raw/n_dropped, in
    adjacency order.

    *samples* is the per-pair (pts, areas) tuples in *adjacency* order
    (as returned by ``faces.get((i, j), _EMPTY)``). *results* supplies
    one (H, K, ok) tuple per NON-empty pair and None for an empty-pts
    pair, in that same order — it may be any iterable, not just a list:
    the parallel driver passes a generator that pulls from the worker
    pool's result stream lazily, so each pair's masking/wrapping runs as
    soon as its own result is ready instead of only after every worker
    result has been collected. Both the serial loop (_curvature_voxel)
    and the parallel driver (_curvature_voxel_parallel) funnel through
    this one function for the mask/wrap/PairCurvature-build/n_raw/
    n_dropped bookkeeping, so their outputs are byte-identical by
    construction rather than by two hand-kept-in-sync copies.
    """
    pairs: dict[tuple[int, int], PairCurvature] = {}
    n_raw = 0
    n_dropped = 0
    for (i, j), (pts, areas), result in zip(
            adjacency, samples, results, strict=True):
        n_raw += len(pts)
        if len(pts):
            H, K, ok = result
            n_dropped += int(np.count_nonzero(~ok))
            pairs[(i, j)] = PairCurvature(
                points=_wrap(pts[ok], box_lengths, periodic),
                areas=areas[ok], H=H[ok], K=K[ok])
        else:
            pairs[(i, j)] = PairCurvature(
                points=np.empty((0, 3)), areas=np.empty(0),
                H=np.empty(0), K=np.empty(0))
    return pairs, n_raw, n_dropped


def _curvature_voxel(tess, box_lengths, periodic, vg,
                     jobs: int = 1) -> CurvatureResult:
    """Level-set path: one _pair_curvature call per adjacency pair.

    jobs <= 1, or <= 1 boundary pair (parallelism has nothing to farm
    out): the original in-process loop, byte-for-byte (the per-pair
    results are gathered by a plain comprehension, then handed to
    _assemble_pairs). jobs > 1 with >= 2 pairs: the same per-pair calls
    run in a worker pool instead (_curvature_voxel_parallel), feeding the
    SAME _assemble_pairs — see analyze_curvature's docstring for why that
    keeps every output array bit-identical.
    """
    if vg is None or vg.tess is None:
        raise GrainsmithError(
            "gb_curvature on a curved run requires the analysis "
            "VoxelGrid built from its tessellation")
    faces = vg.gb_face_samples(periodic)
    adjacency = tess.adjacency()
    if jobs > 1 and len(adjacency) > 1:
        return _curvature_voxel_parallel(
            tess, box_lengths, periodic, vg, faces, adjacency, jobs)
    samples = [faces.get((i, j), _EMPTY) for (i, j) in adjacency]
    results = [
        _pair_curvature(tess, i, j, pts, vg.h_vec) if len(pts) else None
        for (i, j), (pts, _areas) in zip(adjacency, samples, strict=True)
    ]
    pairs, n_raw, n_dropped = _assemble_pairs(
        adjacency, samples, results, box_lengths, periodic)
    return CurvatureResult(pairs=pairs, n_raw=n_raw, n_dropped=n_dropped)


# ---------------------------------------------------------------------------
# Parallel driver (§13: curvature is embarrassingly parallel per boundary
# pair) — same ProcessPoolExecutor shape as atoms.fill's per-grain driver.
# ---------------------------------------------------------------------------

_CURVATURE_CTX: dict | None = None
"""Per-WORKER-PROCESS curvature context, set once by the pool initializer.

Each worker is a separate process (Windows spawn / POSIX fork), so this is
process-local plumbing for ProcessPoolExecutor — not shared mutable module
state in the §15 sense. It exists so the (potentially large) tessellation
and h_vec are pickled once per worker instead of once per boundary pair."""


def _init_curvature_worker(ctx: dict) -> None:
    global _CURVATURE_CTX
    _CURVATURE_CTX = ctx


def _curvature_pair_task(
    args: tuple[int, int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j, pts = args
    c = _CURVATURE_CTX
    assert c is not None  # initializer ran before any task
    return _pair_curvature(c["tess"], i, j, pts, c["h_vec"])


def _curvature_voxel_parallel(
    tess, box_lengths, periodic, vg, faces, adjacency, jobs: int,
) -> CurvatureResult:
    """Process-parallel counterpart of _curvature_voxel's loop.

    Sample points are computed here, parent-side, from *faces* — exactly
    as the serial loop does — so only (i, j, pts) crosses into each
    worker; the worker's only job is the pure _pair_curvature call.
    Empty-pts pairs are skipped when building *tasks* (nothing for a
    worker to do) rather than round-tripped through the pool; positional
    reassembly still works because the *results* generator below
    re-derives, pair by pair, which entries of *samples* were skipped
    (the identical `len(pts)` test *tasks* was filtered by) and only
    calls ``next(result_iter)`` for the ones that were actually sent out
    — so it advances in lockstep with pool.map's order-preserving output
    without needing a separate index.

    That generator is handed straight to _assemble_pairs (the SAME
    post-processing the serial loop uses) and consumed inside this
    function's own `with` block: each pair's mask/wrap/PairCurvature
    build runs as soon as its own pool result is available, so the raw
    (H, K, ok) arrays for every pair never all sit in the parent's
    memory at once the way ``list(pool.map(...))`` would — only the
    already-masked, typically much smaller PairCurvature survives each
    iteration. Same bytes as the serial path either way.
    """
    from concurrent.futures import ProcessPoolExecutor

    samples = [faces.get((i, j), _EMPTY) for (i, j) in adjacency]
    ctx = {"tess": tess, "h_vec": np.asarray(vg.h_vec, dtype=np.float64)}
    tasks = [(i, j, pts) for (i, j), (pts, _areas) in
             zip(adjacency, samples, strict=True) if len(pts)]
    with ProcessPoolExecutor(
        max_workers=min(jobs, len(adjacency)),
        initializer=_init_curvature_worker,
        initargs=(ctx,),
    ) as pool:
        result_iter = pool.map(_curvature_pair_task, tasks)
        results = (
            next(result_iter) if len(pts) else None
            for (pts, _areas) in samples
        )
        pairs, n_raw, n_dropped = _assemble_pairs(
            adjacency, samples, results, box_lengths, periodic)

    return CurvatureResult(pairs=pairs, n_raw=n_raw, n_dropped=n_dropped)


def _weighted_stats(values: np.ndarray,
                    weights: np.ndarray) -> tuple[float, float, float]:
    """(mean, std, abs_mean), area-weighted; NaNs when empty/degenerate."""
    w_total = float(np.sum(weights))
    if len(values) == 0 or w_total <= 0.0:
        nan = float("nan")
        return nan, nan, nan
    mean = float(np.sum(weights * values) / w_total)
    std = float(np.sqrt(np.sum(weights * (values - mean) ** 2) / w_total))
    abs_mean = float(np.sum(weights * np.abs(values)) / w_total)
    return mean, std, abs_mean


def attach_curvature(reports, result: CurvatureResult) -> None:
    """Fill the six curvature fields of each BoundaryReport in place."""
    for r in reports:
        pc = result.pairs.get((r.grain_i, r.grain_j))
        if pc is None or len(pc.H) == 0:
            nan = float("nan")
            r.H_mean_invA = r.H_std_invA = r.H_abs_mean_invA = nan
            r.K_mean_invA2 = r.K_std_invA2 = nan
            r.curv_n_samples = 0
            continue
        (r.H_mean_invA, r.H_std_invA,
         r.H_abs_mean_invA) = _weighted_stats(pc.H, pc.areas)
        r.K_mean_invA2, r.K_std_invA2, _ = _weighted_stats(pc.K, pc.areas)
        r.curv_n_samples = len(pc.H)


def global_curvature_rows(
    result: CurvatureResult,
) -> list[tuple[str, str, object]]:
    """summary.csv (section, key, value) rows for the curvature section."""
    if result.pairs:
        all_h = np.concatenate([p.H for p in result.pairs.values()])
        all_k = np.concatenate([p.K for p in result.pairs.values()])
        all_w = np.concatenate([p.areas for p in result.pairs.values()])
    else:
        all_h = all_k = all_w = np.empty(0)
    h_mean, _, h_abs = _weighted_stats(all_h, all_w)
    k_mean, _, _ = _weighted_stats(all_k, all_w)
    return [
        ("curvature", "H_mean_invA", h_mean),
        ("curvature", "H_abs_mean_invA", h_abs),
        ("curvature", "K_mean_invA2", k_mean),
        ("curvature", "n_samples", int(len(all_h))),
        ("curvature", "dropped_fraction", result.dropped_fraction),
    ]


def per_grain_gauss_bonnet(result: CurvatureResult,
                          n_grains: int) -> np.ndarray:
    """Per-grain face-interior Gauss-curvature integral Σ_faces K·dA
    (input to gate G21, qa.gate_g21_gauss_bonnet).

    Why a SMALL residual is the design intent, and why it isn't reality
    --------------------------------------------------------------------
    The Gauss-Bonnet theorem fixes ``∮_S K dA = 2π·χ(S) = 4π·(1 − g)``
    EXACTLY for any closed orientable surface S of genus g (every grain
    under periodic boundary conditions is one such closed 2-manifold).
    ``analyze_curvature`` samples K ONLY at boundary-face-INTERIOR
    points on purpose (module docstring above, §6.7): the level set φ
    is not twice-differentiable at a polyhedral-like grain's edges and
    vertices, so those samples are excluded rather than evaluated on a
    singular stencil. The DESIGN INTENT is that the excluded
    edges/vertices carry the dominant share of the budget (Dirac-mass
    angle defects — a cube's corners alone sum to 4π), leaving a small
    face-interior remainder.

    THIS FUNCTION'S OWN CALIBRATION (gate_g21_gauss_bonnet's validation
    sweep, full numbers in ``constants.CURV_G21_GAUSS_BONNET_TOL``)
    shows that intent is NOT met at the current estimator's noise/
    leakage floor: six near-FLAT baselines (amplitude 0.01) landed
    their worst-grain ratio at 0.977-1.041x the 4pi tolerance —
    clustered AT one full topological unit, not comfortably below it —
    and ordinary curved runs routinely reach 5-14x. A 2-grain bicrystal
    with ONE smooth boundary and NO triple junctions still measured
    8-13x across a 32-128 voxel_grid range, and that range shows no
    trend toward 0. So: do not read a returned value near or above 4pi
    as proof of a defect, and do not expect a well-behaved curved run
    to land near 0 — the realistic floor for this estimator is closer
    to "about one topological unit," with routine excursions several
    times higher.

    K is invariant under the (i, j) normal flip (module docstring), so
    one boundary sample contributes the SAME K·area term to both
    grain_i's and grain_j's own closed-surface total — each grain's
    surface is the union of all faces it owns, shared or not.

    Parameters
    ----------
    result : CurvatureResult
        Output of ``analyze_curvature`` (flat: exact zeros; curved:
        level-set samples with degenerate points already dropped).
    n_grains : int
        Total grain count (``tess.n_grains``) — fixes the output length
        even for grains with zero incident boundary samples (isolated
        single-crystal runs, or a grain whose neighbours were all
        degenerate).

    Returns
    -------
    (n_grains,) float64 — Σ_faces K·dA per grain; 0.0 for a grain with
    no retained samples (this is an EXACT 0.0 only for flat geometry,
    whose kernel writes literal zeros — see test_per_grain_gauss_bonnet_
    flat_is_zero; a curved grain with retained samples routinely lands
    at or above one multiple of 4π per the calibration above, not near
    0.0).
    """
    totals = np.zeros(n_grains, dtype=np.float64)
    for (i, j), pc in result.pairs.items():
        if len(pc.K) == 0:
            continue
        contrib = float(np.sum(pc.areas * pc.K))
        totals[i] += contrib
        totals[j] += contrib
    return totals
