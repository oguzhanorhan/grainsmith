"""Per-grain lattice fill (§6.8).

Correctness model
-----------------
The home cells of all grains (Tessellation.owns) tile the simulation domain
exactly once, so filling each grain's compact home cell generates every
torus point exactly once — for ANY lattice orientation, commensurate or not.
No wrap-and-deduplicate step exists: a wrapped-membership + rounding-dedup
scheme double-counts incommensurate lattices.

Positions are stored UNWRAPPED in absolute lab coordinates (§1): a cell that
straddles a periodic box face keeps its atoms on the cell's compact side.
Writers wrap to [0, L) at write time; the overlap stage wraps internally for
neighbor searches.

Parallel fill (§13)
-------------------
``fill_grain`` is a pure function of (inputs, rng stream), so grains fill
embarrassingly parallel: ``fill_grains(..., jobs=N)`` fans grains out over a
``ProcessPoolExecutor``.  Each grain consumes its OWN deterministic rng
stream (``RNGBundle.occupancy_streams``), and results are collected in grain
order — the output is bit-identical for every ``jobs`` value.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from grainsmith.constants import FILL_CHUNK, FILL_POOL_RAM_FRACTION
from grainsmith.errors import TessellationError
from grainsmith.memory import (
    build_memory_guard_message,
    detect_available_ram_bytes,
    format_bytes_adaptive,
)
from grainsmith.orientation.quaternion import quat_to_matrix
from grainsmith.tessellation.base import Tessellation

log = logging.getLogger(__name__)


@dataclass
class AtomBlock:
    """Struct-of-arrays atom container (§5)."""
    pos: np.ndarray                  # (N, 3) float64, unwrapped lab coords (Å)
    species: np.ndarray              # (N,) U2
    grain: np.ndarray                # (N,) int32
    gb_margin: np.ndarray | None = None  # (N,) float64, optional

    def __len__(self) -> int:
        return len(self.pos)

    @staticmethod
    def concatenate(blocks: list[AtomBlock]) -> AtomBlock:
        pos = np.concatenate([b.pos for b in blocks])
        species = np.concatenate([b.species for b in blocks])
        grain = np.concatenate([b.grain for b in blocks])
        has_margin = all(b.gb_margin is not None for b in blocks)
        margin = np.concatenate([b.gb_margin for b in blocks]) if has_margin else None
        return AtomBlock(pos=pos, species=species, grain=grain, gb_margin=margin)


FILL_GRID_BYTES_PER_POINT: int = 72
"""Peak bytes per lattice-grid point in :func:`fill_grain` (§13).

The three-step construction

    iu, iv, iw = np.meshgrid(*ranges, indexing='ij')          # 3 × int64
    T = np.stack([iu.ravel(), iv.ravel(), iw.ravel()],
                 axis=1).astype(np.float64)                   # int64 → float64

transiently holds THREE (n_grid,)/(n_grid,3) arrays at once: the meshgrid
triple (3×8 B), the ``np.stack`` int64 intermediate (3×8 B) and the
``astype`` float64 result (3×8 B) — the first two are still referenced
while the third is allocated.  That is 72 B/point, not the 48 B/point
(``n_grid * 6 * 8``) the guard used to charge, a 1.5× under-count
confirmed by ``tracemalloc`` (measured exactly 72.0 B/point).

After the stack intermediate is freed the resident cost settles at
48 B/point, because ``iu``/``iv``/``iw`` are never deleted and stay bound
to live locals for the whole chunk loop — but the guard must bound the
PEAK, which is what the OS OOM-killer reacts to."""


def _search_lattice(A: np.ndarray) -> np.ndarray:
    """Return a lattice matrix (columns = vectors) generating the SAME lattice
    as ``A`` but Niggli-reduced, so the nearest lattice translations lie within
    a small integer shell.  Falls back to ``A`` unchanged if reduction fails.

    ``cell_matrix`` builds A directly from (a, b, c, α, β, γ) with no reduction,
    so for a skewed triclinic/monoclinic cell the shortest lattice vector needs
    translation coefficients outside {-1, 0, 1} — invisible to a fixed ±1 shell.
    Reducing the lattice first restores the guarantee that a small shell is
    sufficient.  ``cell_matrix`` stores vectors as COLUMNS; spglib expects them
    as ROWS, hence the transposes.
    """
    try:
        import spglib
        reduced = spglib.niggli_reduce(np.ascontiguousarray(A.T))
    except Exception:  # pragma: no cover - spglib always present in practice
        reduced = None
    if reduced is None:
        return A
    return np.ascontiguousarray(np.asarray(reduced, dtype=np.float64).T)


def _compute_d_nn(frac_basis: np.ndarray, A: np.ndarray) -> float:
    """Minimum interatomic distance d_nn in the ideal crystal (Å).

    Considers all basis-pair distances over a neighbour shell of the (Niggli-
    reduced) lattice.  Only the true self-pair (same atom, zero shift) is
    excluded; an atom's distance to its own periodic image in a neighbouring
    cell is a valid d_nn candidate (it IS d_nn for a one-atom basis).

    Basis atoms are wrapped into the SAME reduced cell before searching.
    A lattice-column length supplies an upper bound on d_nn; reciprocal
    row norms then bound every integer image coefficient. The shell is
    at least +/-2 and expands if needed, including if reduction fails.
    """
    cart = (A @ frac_basis.T).T  # (N_basis, 3)
    n_basis = len(cart)
    A_search = _search_lattice(A)
    inverse = np.linalg.inv(A_search)
    search_fractional = (inverse @ cart.T).T
    translations = np.floor(search_fractional)
    cart = cart - translations @ A_search.T
    search_fractional -= translations
    upper_distance = float(np.min(np.linalg.norm(A_search, axis=0)))
    extent = (np.ptp(search_fractional, axis=0)
              + upper_distance * np.linalg.norm(inverse, axis=1))
    shells = np.maximum(2, np.ceil(np.nextafter(extent, np.inf))).astype(int)
    d_min = np.inf
    for index in np.ndindex(*(2 * shells + 1)):
        sv = np.asarray(index, dtype=np.float64) - shells
        img = cart + (A_search @ sv)
        for i in range(n_basis):
            diffs = img - cart[i]
            dists = np.linalg.norm(diffs, axis=1)
            if np.all(sv == 0.0):
                dists[i] = np.inf  # same atom, zero shift: not a pair
            d_min = min(d_min, float(np.min(dists)))
    return float(d_min)


def lattice_grid_extent(
    grain_id: int,
    tess: Tessellation,
    R_rot: np.ndarray,
    Ainv: np.ndarray,
    Ainv_rows: np.ndarray,
    r_basis_max: float,
    a_clip: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Lattice-index bounding box ``(lo, hi, n_grid)`` enumerated by
    :func:`fill_grain` for grain *grain_id* (§6.8).

    Split out of ``fill_grain`` so the DRIVER can price a grain's grid
    without filling it — :func:`estimate_fill_grid_bytes` uses this for
    the jobs-aware pre-flight in :func:`fill_grains`.  Pure geometry: no
    allocation proportional to ``n_grid`` happens here.
    """
    # Use getattr so that lightweight mock objects (e.g. in tests) that do not
    # inherit from Tessellation and therefore lack cell_vertices_rel fall through
    # to the sphere path rather than raising AttributeError.
    _cvr = getattr(tess, "cell_vertices_rel", None)
    V = _cvr(grain_id) if _cvr is not None else None  # (Nv,3) rel to seed, or None
    if V is not None and len(V) > 0:
        # Map cell vertices into lattice-index (fractional) space.
        # Owned atoms satisfy  X_lab - c_i = R_rot @ A @ (T+f),
        # so  T+f = A^{-1} @ R_rot.T @ (X_lab - c_i).
        # The cell vertices bound the set of X_lab - c_i values for owned
        # atoms; convexity is preserved by the linear map A^{-1} @ R_rot.T.
        Fv = (Ainv @ (np.array(V, dtype=np.float64) @ R_rot).T).T  # (Nv,3)
        # Pad by ceil((r_basis_max + a_clip) * ‖row_k(A^{-1})‖) + 1 per axis
        # to account for the maximum basis offset in index space.
        pad = np.ceil((r_basis_max + a_clip) * Ainv_rows).astype(np.int64) + 1
        lo = np.floor(Fv.min(axis=0)).astype(np.int64) - pad
        hi = np.ceil(Fv.max(axis=0)).astype(np.int64) + pad
    else:
        # Sphere fallback: symmetric cube from circumscribed-sphere radius.
        r_b = tess.bounding_radius(grain_id) + a_clip + r_basis_max
        n_max = np.ceil(r_b * Ainv_rows).astype(np.int64)
        lo, hi = -n_max, n_max

    n_grid = (int(hi[0] - lo[0] + 1) * int(hi[1] - lo[1] + 1)
              * int(hi[2] - lo[2] + 1))
    return lo, hi, n_grid


def estimate_fill_grid_bytes(
    tess: Tessellation,
    frac_basis: np.ndarray,
    A: np.ndarray,
    quats: np.ndarray,
    a_clip: float,
    grain_ids: list[int] | None = None,
) -> list[int]:
    """Peak lattice-grid bytes :func:`fill_grain` will allocate for each
    grain, computed in the driver WITHOUT filling anything (§13).

    Mirrors ``fill_grain``'s own geometry exactly (same
    :func:`lattice_grid_extent`, same ``FILL_GRID_BYTES_PER_POINT``), so
    the pre-flight in :func:`fill_grains` prices what the workers will
    really allocate rather than a separate approximation that could drift.
    """
    frac_basis = np.asarray(frac_basis, dtype=np.float64).reshape(-1, 3)
    A = np.asarray(A, dtype=np.float64)
    Ainv = np.linalg.inv(A)
    Ainv_rows = np.linalg.norm(Ainv, axis=1)
    cart_basis = (A @ frac_basis.T).T
    r_basis_max = (float(np.max(np.linalg.norm(cart_basis, axis=1)))
                   if len(frac_basis) else 0.0)
    ids = range(len(quats)) if grain_ids is None else grain_ids
    out = []
    for i in ids:
        R_rot = quat_to_matrix(np.asarray(quats[i], dtype=np.float64))
        _, _, n_grid = lattice_grid_extent(
            i, tess, R_rot, Ainv, Ainv_rows, r_basis_max, a_clip)
        out.append(n_grid * FILL_GRID_BYTES_PER_POINT)
    return out


def fill_grain(
    grain_id: int,
    tess: Tessellation,
    frac_basis: np.ndarray,
    basis_species: list[str],
    basis_occupancy: list[dict | None],
    A: np.ndarray,
    q_i: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    rng: np.random.Generator,
    store_margin: bool = False,
    a_clip: float = 0.0,
) -> AtomBlock:
    """
    Fill grain i with lattice atoms (§6.8).

    Algorithm:
        R_b = bounding_radius(i) + a_clip + max|A @ frac_basis|
        n_max[k] = ceil(R_b * ‖row_k(A^{-1})‖)        # exact per-axis bound
        Enumerate lattice translations T in ∏ [-n_max, n_max] (chunked §13)
        For each basis atom f: X_lab = R(q_i) @ A @ (T + f) + c_i
        Keep if tess.owns(X_lab, grain_id)            # compact home cell
                and box-clipped on non-periodic axes.
        Species: fixed per site, or sampled from the site occupancy dict
        (vectorized, one rng draw batch per basis site, deterministic order).

    Positions are returned unwrapped (§1); each torus point appears exactly
    once by the owns() tiling property.
    """
    R_rot = quat_to_matrix(q_i)
    c_i = tess.seeds[grain_id]
    Ainv = np.linalg.inv(A)
    L = np.asarray(box_lengths, dtype=np.float64)
    frac_basis = np.asarray(frac_basis, dtype=np.float64).reshape(-1, 3)
    n_basis = len(frac_basis)

    # Basis extent in Cartesian and in lattice-index space
    cart_basis = (A @ frac_basis.T).T
    r_basis_max = float(np.max(np.linalg.norm(cart_basis, axis=1))) if n_basis else 0.0

    # Per-row norm of A^{-1}: used in both the vertex and sphere paths.
    Ainv_rows = np.linalg.norm(Ainv, axis=1)

    lo, hi, n_grid = lattice_grid_extent(
        grain_id, tess, R_rot, Ainv, Ainv_rows, r_basis_max, a_clip)

    # The bounding-box grid T plus the three int meshgrid arrays dominate peak
    # memory here (the per-chunk expansion below is bounded by FILL_CHUNK, but T
    # itself is materialized once). Guard it against the §13 limit so a
    # pathologically large single grain raises cleanly instead of OOMing.
    est_bytes = n_grid * FILL_GRID_BYTES_PER_POINT
    # §13: read the LIMIT OFF THE TESSELLATION, not the constants.py module
    # default — the pipeline stamps the per-run runtime.memory_limit_gb value
    # onto tess.memory_limit_bytes (tessellation/base.py) before this runs,
    # and that value must survive a ProcessPoolExecutor 'spawn' worker's
    # re-import of this module, which a module-level global would not.
    limit_bytes = tess.memory_limit_bytes
    if est_bytes > limit_bytes:
        # getattr, not a plain attribute read: a handful of tests exercise
        # fill_grain against duck-typed stub tessellations that don't
        # subclass Tessellation and therefore never picked up
        # memory_limit_source (tests/test_fill.py, test_doping.py) — "config"
        # (the base-class default, see tessellation/base.py) is the correct
        # fallback for those.
        source = getattr(tess, "memory_limit_source", "config")
        raise TessellationError(build_memory_guard_message(
            f"fill_grain: lattice grid for grain {grain_id}",
            est_bytes, limit_bytes, source,
            # This IS the site that actually runs inside the per-grain
            # ProcessPoolExecutor (fill_grains, jobs > 1) -- see
            # build_memory_guard_message's extra_advice docs for why this
            # sentence is passed ONLY here and nowhere else.
            extra_advice=(
                "With --jobs N up to N grains fill concurrently, each "
                "allocating its own grid."),
        ))
    ranges = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
    iu, iv, iw = np.meshgrid(*ranges, indexing='ij')
    T = np.stack([iu.ravel(), iv.ravel(), iw.ravel()], axis=1).astype(np.float64)

    all_pos: list[np.ndarray] = []
    all_site: list[np.ndarray] = []

    # Process in chunks to bound memory (§13: ≤ FILL_CHUNK candidate rows)
    chunk_t = max(1, FILL_CHUNK // max(n_basis, 1))
    site_pattern = np.arange(n_basis, dtype=np.int64)

    for start in range(0, len(T), chunk_t):
        T_chunk = T[start:start + chunk_t]                     # (C, 3)
        F = T_chunk[:, None, :] + frac_basis[None, :, :]       # (C, Nb, 3)
        F_flat = F.reshape(-1, 3)

        X_cryst = (A @ F_flat.T).T
        X_lab = (R_rot @ X_cryst.T).T + c_i                    # unwrapped

        keep = tess.owns(X_lab, grain_id)
        for ax in range(3):
            if not periodic[ax]:
                keep &= (X_lab[:, ax] >= 0.0) & (X_lab[:, ax] <= L[ax])

        if not np.any(keep):
            continue

        all_pos.append(X_lab[keep])
        sites = np.tile(site_pattern, len(T_chunk))
        all_site.append(sites[keep])

    if not all_pos:
        return AtomBlock(
            pos=np.empty((0, 3), dtype=np.float64),
            species=np.empty(0, dtype="U2"),
            grain=np.empty(0, dtype=np.int32),
            gb_margin=np.empty(0, dtype=np.float64) if store_margin else None,
        )

    pos_arr = np.concatenate(all_pos, axis=0)
    site_arr = np.concatenate(all_site, axis=0)

    # Species: fixed symbol per site, occupancy-sampled where a dict is given.
    species_arr = np.asarray(basis_species, dtype="U2")[site_arr]
    for s in range(n_basis):
        occ = basis_occupancy[s]
        if occ is None:
            continue
        mask = site_arr == s
        count = int(mask.sum())
        if count == 0:
            continue
        symbols = list(occ.keys())
        probs = np.array(list(occ.values()), dtype=np.float64)
        probs /= probs.sum()
        species_arr[mask] = rng.choice(symbols, size=count, p=probs)

    grain_arr = np.full(len(pos_arr), grain_id, dtype=np.int32)
    margin_arr = tess.margin(pos_arr, grain_id) if store_margin else None

    return AtomBlock(
        pos=pos_arr,
        species=species_arr,
        grain=grain_arr,
        gb_margin=margin_arr,
    )


# ---------------------------------------------------------------------------
# Parallel driver (§13: fill is embarrassingly parallel per grain)
# ---------------------------------------------------------------------------

_FILL_CTX: dict | None = None
"""Per-WORKER-PROCESS fill context, set once by the pool initializer.

Each worker is a separate process (Windows spawn / POSIX fork), so this is
process-local plumbing for ProcessPoolExecutor — not shared mutable module
state in the §15 sense.  It exists so the (potentially large) tessellation
is pickled once per worker instead of once per grain."""


def _init_fill_worker(ctx: dict) -> None:
    global _FILL_CTX
    _FILL_CTX = ctx


def _fill_one(args: tuple[int, np.random.Generator]) -> AtomBlock:
    grain_id, rng = args
    c = _FILL_CTX
    assert c is not None  # initializer ran before any task
    return fill_grain(
        grain_id, c["tess"], c["frac_basis"], c["basis_species"],
        c["basis_occupancy"], c["A"], c["quats"][grain_id],
        c["box_lengths"], c["periodic"], rng,
        store_margin=c["store_margin"], a_clip=c["a_clip"],
    )


def _fill_one_phases(args: tuple[int, np.random.Generator]) -> AtomBlock:
    grain_id, rng = args
    c = _FILL_CTX
    assert c is not None  # initializer ran before any task
    p = int(c["phase_of"][grain_id])
    return fill_grain(
        grain_id, c["tess"], c["frac_list"][p], c["species_list"][p],
        c["occ_list"][p], c["A_list"][p], c["quats"][grain_id],
        c["box_lengths"], c["periodic"], rng,
        store_margin=c["store_margin"], a_clip=c["a_clip"],
    )


def _resolve_fill_workers(est_bytes: list[int], jobs: int, n: int,
                          tess: Tessellation) -> tuple[int, int]:
    """Worker count that actually FITS, plus the per-grain peak (§13).

    The in-worker guard in :func:`fill_grain` is per-ALLOCATION: it bounds
    one grain's grid against ``tess.memory_limit_bytes`` and is blind to
    the fact that ``--jobs N`` runs up to N of them at once.  On the
    12-grain 1925 Å PdAu config each grid was 11.4 GB against a 25 GB
    budget -- passing comfortably -- while ``--jobs 2`` needed 34 GB on a
    machine with 11 GB free, and the kernel OOM-killed a worker.

    Clamping (rather than raising) is correct here because the fill is
    order-independent: blocks come back in grain order and each grain
    draws its own rng stream, so the output is bit-identical for every
    worker count.  A clamped run is slower, never different.
    """
    workers = max(1, min(jobs, n))
    if not est_bytes:
        return workers, 0
    peak = max(est_bytes)
    if peak <= 0:
        return workers, peak

    budget = float(tess.memory_limit_bytes)
    basis = "runtime.memory_limit_gb"
    avail = detect_available_ram_bytes()
    if avail is not None:
        ram_budget = avail * FILL_POOL_RAM_FRACTION
        if ram_budget < budget:
            budget, basis = ram_budget, "available RAM"

    fits = max(1, int(budget // peak))
    if fits >= workers:
        return workers, peak

    log.warning(
        "fill: --jobs %d would need %s of concurrent lattice grids "
        "(%s per grain x %d workers) but only %s is usable (%s); "
        "clamping to --jobs %d. Output is unchanged (bit-identical for "
        "every jobs value), only slower. To go faster, raise "
        "grains.number (per-grain grid scales as box_volume/n_grains) "
        "or shrink box.lengths.",
        jobs, format_bytes_adaptive(peak * workers),
        format_bytes_adaptive(peak), workers,
        format_bytes_adaptive(budget), basis, fits,
    )
    return fits, peak


def _run_fill_pool(worker_fn, ctx: dict, rngs: list, n: int,
                   workers: int, peak: int) -> list[AtomBlock]:
    """Run the per-grain fill pool, translating a worker death into a
    memory diagnosis instead of a bare ``BrokenProcessPool`` (§13).

    A ``BrokenProcessPool`` from this pool means a worker vanished without
    returning -- in practice the OS OOM-killer (``SIGKILL``), which leaves
    no Python traceback and no clue in the driver's own RSS report, since
    that monitor samples only the driver.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_fill_worker,
            initargs=(ctx,),
        ) as pool:
            # map preserves input order → blocks come back in grain order.
            return list(pool.map(worker_fn, [(i, rngs[i]) for i in range(n)]))
    except BrokenProcessPool as exc:
        raise TessellationError(
            f"fill: a worker process died during the per-grain fill "
            f"({workers} workers). The usual cause is the OS OOM-killer: "
            f"each worker allocates its own lattice grid, peaking at "
            f"{format_bytes_adaptive(peak)} for the largest grain here, so "
            f"{workers} workers need about "
            f"{format_bytes_adaptive(peak * workers)} at once. "
            f"Confirm with: journalctl -k | grep -i 'killed process'. "
            f"Remedies, in order of effect: raise grains.number (the "
            f"per-grain grid scales as box_volume/n_grains), shrink "
            f"box.lengths, or re-run with --jobs 1 (output is "
            f"bit-identical for every jobs value, only slower)."
        ) from exc


def fill_grains(
    tess: Tessellation,
    frac_basis: np.ndarray,
    basis_species: list[str],
    basis_occupancy: list[dict | None],
    A: np.ndarray,
    quats: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    rngs: list[np.random.Generator],
    store_margin: bool = False,
    a_clip: float = 0.0,
    jobs: int = 1,
) -> list[AtomBlock]:
    """Fill every grain (§6.8), serial or process-parallel (§13).

    Parameters
    ----------
    rngs : list of per-grain Generators (RNGBundle.occupancy_streams) —
        one independent stream per grain, so the result does not depend
        on execution order.
    jobs : worker processes; 1 = in-process serial.  Results are returned
        in grain order and are bit-identical for every jobs value.
    """
    n = len(quats)
    if len(rngs) != n:
        raise ValueError(f"need one rng per grain: {len(rngs)} != {n}")

    if jobs <= 1:
        return [
            fill_grain(i, tess, frac_basis, basis_species, basis_occupancy,
                       A, quats[i], box_lengths, periodic, rngs[i],
                       store_margin=store_margin, a_clip=a_clip)
            for i in range(n)
        ]

    ctx = {
        "tess": tess,
        "frac_basis": np.asarray(frac_basis, dtype=np.float64),
        "basis_species": basis_species,
        "basis_occupancy": basis_occupancy,
        "A": np.asarray(A, dtype=np.float64),
        "quats": np.asarray(quats, dtype=np.float64),
        "box_lengths": np.asarray(box_lengths, dtype=np.float64),
        "periodic": list(periodic),
        "store_margin": store_margin,
        "a_clip": a_clip,
    }
    workers, peak = _resolve_fill_workers(
        estimate_fill_grid_bytes(tess, frac_basis, A, quats, a_clip),
        jobs, n, tess)
    return _run_fill_pool(_fill_one, ctx, rngs, n, workers, peak)


def fill_grains_phases(
    tess: Tessellation,
    phase_of: np.ndarray,
    frac_list: list[np.ndarray],
    species_list: list[list[str]],
    occ_list: list[list[dict | None]],
    A_list: list[np.ndarray],
    quats: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    rngs: list[np.random.Generator],
    store_margin: bool = False,
    a_clip: float = 0.0,
    jobs: int = 1,
) -> list[AtomBlock]:
    """Multiphase fill: like :func:`fill_grains` but each grain
    builds the crystal of ITS phase — ``phase_of[i]`` indexes the per-phase
    basis/cell lists.  Same per-grain rng streams and grain-order results,
    so the output stays bit-identical for every *jobs* value."""
    n = len(quats)
    if len(rngs) != n:
        raise ValueError(f"need one rng per grain: {len(rngs)} != {n}")

    if jobs <= 1:
        return [
            fill_grain(i, tess, frac_list[p], species_list[p], occ_list[p],
                       A_list[p], quats[i], box_lengths, periodic, rngs[i],
                       store_margin=store_margin, a_clip=a_clip)
            for i in range(n)
            for p in (int(phase_of[i]),)
        ]

    ctx = {
        "tess": tess,
        "phase_of": np.asarray(phase_of),
        "frac_list": [np.asarray(f, dtype=np.float64) for f in frac_list],
        "species_list": species_list,
        "occ_list": occ_list,
        "A_list": [np.asarray(A, dtype=np.float64) for A in A_list],
        "quats": np.asarray(quats, dtype=np.float64),
        "box_lengths": np.asarray(box_lengths, dtype=np.float64),
        "periodic": list(periodic),
        "store_margin": store_margin,
        "a_clip": a_clip,
    }
    # Per-grain grid cost, priced with THAT grain's phase cell/basis —
    # a multiphase run can mix a dense and a sparse lattice, and it is the
    # largest grid that has to fit N times over.
    phase_ids = np.asarray(phase_of).astype(int)
    est: list[int] = [0] * n
    for p in range(len(A_list)):
        members = [i for i in range(n) if phase_ids[i] == p]
        if not members:
            continue
        for i, b in zip(members, estimate_fill_grid_bytes(
                tess, frac_list[p], A_list[p], quats, a_clip,
                grain_ids=members), strict=True):
            est[i] = b
    workers, peak = _resolve_fill_workers(est, jobs, n, tess)
    return _run_fill_pool(_fill_one_phases, ctx, rngs, n, workers, peak)
