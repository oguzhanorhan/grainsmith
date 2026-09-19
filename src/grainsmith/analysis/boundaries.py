"""Per-boundary analysis: areas, normals, GB character, CSL (§6.10).

Produces exactly the boundaries.csv column set of §8.3; CSV serialization
lives in io/reports.py.

Geometry sources
----------------
flat backend   : exact face polygons of the Voronoi cells (areas + normals
                 per face; a pair adjacent through several periodic images
                 contributes all its faces).
curved backends: voxel-face midpoints/areas with central-difference local
                 normals (VoxelGrid.gb_normals, §6.7).

Mean normal & spread
--------------------
The mean unit normal n̂ is the normalized area-weighted resultant of the
local normals (oriented from grain i toward grain j, i < j).  When the
resultant is degenerate — e.g. a pair meeting through two opposite periodic
faces with antiparallel normals — the boundary has no meaningful mean plane:
character is reported as "undefined" and the plane indices as (000).  The
normal-spread angle (area-weighted RMS angle to n̂) qualifies the "mean
plane" claim for curved boundaries (§6.10).

GB character
------------
ψ = ∠(misorientation axis in the LAB frame, n̂); axes are unsigned so
ψ ∈ [0°, 90°].  twist: ψ < CHAR_TWIST_DEG; tilt: ψ > CHAR_TILT_DEG; else
mixed.

The lab axis must come from the PHYSICAL minimal-angle map
G = argmin_S angle(R_j · R(S) · R_iᵀ): the symmetry-REDUCED crystal-frame
axis from disorientation() is only defined up to the point-group orbit
(its two-sided candidates S_a·m·S_b are symmetry-frame re-expressions),
and mapping an arbitrary orbit representative to the lab frame would
misclassify e.g. a Σ5 ⟨100⟩ tilt boundary as twist.  The one-sided orbit
{R_j S R_iᵀ} is the set of physical lattice-to-lattice maps; its angles
coincide with the two-sided reduction (S_a m S_b is conjugate to m S_b S_a)
but its minimal-angle axis is the physically unique one.

CSL (cubic only)
-----------------
The disorientation is matched against the built-in cubic table under the
Brandon criterion Δθ ≤ CSL_BRANDON_FACTOR · Σ^(−1/2), where Δθ is the
symmetry-reduced misorientation between the boundary's disorientation and
the exact Σ rotation.  The smallest qualifying Σ wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grainsmith.constants import (
    CHAR_TILT_DEG,
    CHAR_TWIST_DEG,
    CSL_BRANDON_FACTOR,
    CSL_TABLE,
    IDENTITY_ANGLE_TOL,
)
from grainsmith.errors import ConfigError, GrainsmithError
from grainsmith.orientation.descriptors import direction_miller, plane_miller
from grainsmith.orientation.misorientation import disorientation
from grainsmith.orientation.quaternion import axis_angle_to_quat, quat_to_matrix
from grainsmith.tessellation.base import Tessellation

_EMPTY_NORMALS_AREAS: tuple[np.ndarray, np.ndarray] = (np.empty((0, 3)),
                                                        np.empty(0))
"""(normals, areas) default for a pair with no retained geometry samples —
already-extracted-normal shape (flat exact faces / voxel gb_normals output),
consumed directly by _pair_report_fields."""

_EMPTY_PTS_AREAS = _EMPTY_NORMALS_AREAS
"""(pts, areas) default for a pair with no retained voxel-face samples,
PRE-normal-extraction (vg.gb_face_samples output). Same empty shape as
_EMPTY_NORMALS_AREAS — aliased under its own name so each call site's
tuple meaning (raw face points vs. already-computed normals) is
unambiguous from the constant alone."""

_MIN_RESULTANT: float = 1e-6
"""Relative resultant length below which the area-weighted mean normal is
degenerate (antiparallel sheets) and no mean boundary plane exists."""

_CUBIC_N_PROPER: int = 24
"""Number of proper rotations of the cubic point group m-3m; the CSL
table is cubic-only (§6.10)."""


@dataclass
class BoundaryReport:
    """One boundaries.csv row (§8.3)."""
    grain_i: int
    grain_j: int
    seed_distance_A: float
    misorientation_deg: float
    axis_u: int
    axis_v: int
    axis_w: int
    axis_dev_deg: float
    area_A2: float
    mean_normal_x: float
    mean_normal_y: float
    mean_normal_z: float
    normal_spread_deg: float
    plane_i_hkl: str
    plane_i_dev_deg: float
    plane_j_hkl: str
    plane_j_dev_deg: float
    character: str
    character_angle_deg: float
    csl_sigma: str
    n_overlap_deleted: int
    # Phase columns (multiphase runs only)
    phase_i: str = ""
    phase_j: str = ""
    # GB curvature columns (analysis.gb_curvature runs only): area-weighted
    # stats over the boundary's retained local samples; flat runs carry
    # exact zeros.
    H_mean_invA: float = float("nan")
    H_std_invA: float = float("nan")
    H_abs_mean_invA: float = float("nan")
    K_mean_invA2: float = float("nan")
    K_std_invA2: float = float("nan")
    curv_n_samples: int = 0
    # Raw vectors for downstream use (not CSV columns)
    axis_crystal: np.ndarray | None = field(repr=False, default=None)
    mean_normal: np.ndarray | None = field(repr=False, default=None)


def _pair_geometry_flat(tess, pairs) -> dict[tuple[int, int],
                                              tuple[np.ndarray, np.ndarray]]:
    """{(i,j): (normals (M,3), areas (M,))} from exact cell faces."""
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (i, j) in pairs:
        normals = []
        areas = []
        for face in tess.cells[i].faces:
            if face.neighbor_id == j:
                normals.append(face.unit_normal)
                areas.append(face.area)
        out[(i, j)] = (np.array(normals, dtype=np.float64).reshape(-1, 3),
                       np.array(areas, dtype=np.float64))
    return out


def _pair_geometry_voxel(tess, box_lengths, periodic) -> dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """{(i,j): (normals, areas)} from the voxel grid (curved backends)."""
    from grainsmith.analysis.grains import get_voxel_grid
    vg = get_voxel_grid(tess, box_lengths)
    raw = vg.gb_normals(periodic)
    return {pair: (normals, areas) for pair, (_, normals, areas) in raw.items()}


def _pair_frames(
    i: int, j: int, multiphase: bool,
    pof: np.ndarray, syms: list[np.ndarray], As: list[np.ndarray],
    sym_quats: np.ndarray, A: np.ndarray,
) -> tuple[int, int, bool, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair phase/point-group/cell frame selection (§6.10).

    Pure lookup into the multiphase tables built once by the caller —
    same-phase pairs use their phase's point group and cell; interphase
    pairs have no common point group.  Shared by analyze_boundaries' serial
    loop and its voxel process-parallel driver so both pick frames
    identically. Returns (p_i, p_j, same_phase, sym_pair, A_i, A_j).
    """
    if multiphase:
        p_i, p_j = int(pof[i]), int(pof[j])
        same_phase = p_i == p_j
        sym_pair = syms[p_i]
        A_i, A_j = As[p_i], As[p_j]
    else:
        p_i = p_j = 0
        same_phase = True
        sym_pair = sym_quats
        A_i = A_j = A
    return p_i, p_j, same_phase, sym_pair, A_i, A_j


def _pair_report_fields(
    normals: np.ndarray, areas: np.ndarray,
    q_i: np.ndarray, q_j: np.ndarray, sym_pair: np.ndarray,
    A_i: np.ndarray, A_j: np.ndarray, same_phase: bool,
    csl: bool, character: bool,
) -> tuple:
    """Pure per-pair computation of every geometry/orientation-derived
    BoundaryReport field: misorientation, boundary-plane Miller indices,
    tilt/twist character and CSL Σ (§6.10).

    Takes the pair's ALREADY-EXTRACTED (normals, areas) — flat's exact
    face values or the voxel central-difference values — so this function
    itself never touches tess/the voxel grid and is cheap to run inside a
    worker process (mirrors analysis/curvature.py's _pair_curvature: the
    expensive tess-dependent extraction is a separate step, done either
    parent-side (flat, cheap) or in-worker (voxel, via
    tessellation.voxel._gb_pair_normals) before this pure function runs).

    Shared verbatim by analyze_boundaries' serial loop and its voxel
    ProcessPoolExecutor driver — same code, same floating-point operations
    in the same order, so every field is byte-identical for every jobs
    value (mirrors analysis/curvature.py's _assemble_pairs: one function,
    not two hand-kept-in-sync copies).

    Returns (misorientation_deg, axis_uvw, axis_dev_deg, area_A2,
    mean_normal, normal_spread_deg, plane_i_hkl, plane_i_dev_deg,
    plane_j_hkl, plane_j_dev_deg, character, character_angle_deg,
    csl_sigma, axis_crystal).
    """
    area_total = float(np.sum(areas))
    n_hat, spread = _mean_normal_and_spread(normals, areas)

    axis_crystal: np.ndarray | None
    if same_phase:
        dis = disorientation(q_i, q_j, sym_pair)
        mis_deg = float(dis.angle_deg)
        axis_crystal = dis.axis_crystal
        if mis_deg < IDENTITY_ANGLE_TOL:
            axis_crystal = None
            axis_uvw, axis_dev = [0, 0, 0], float("nan")
        else:
            axis_uvw, axis_dev = direction_miller(axis_crystal, A_i)
    else:
        dis = None
        mis_deg = float("nan")
        axis_crystal = None
        axis_uvw, axis_dev = [0, 0, 0], float("nan")

    if n_hat is None:
        char_str = "undefined"
        psi = float("nan")
        hkl_i, dev_i = [0, 0, 0], float("nan")
        hkl_j, dev_j = [0, 0, 0], float("nan")
        mean_n = np.zeros(3)
    elif not character:
        mean_n = n_hat
        char_str = "" if same_phase else "interphase"
        psi = float("nan")
        hkl_i, dev_i = [], float("nan")   # → empty hkl strings
        hkl_j, dev_j = [], float("nan")
    else:
        mean_n = n_hat
        # Boundary-plane indices in both crystal frames (§6.10):
        # n_crystal = R(q)^T n̂_lab, then m ∝ A^T n_crystal.  These
        # stay defined for interphase pairs (habit planes) — each
        # frame uses its OWN cell matrix.
        R_i = quat_to_matrix(q_i)
        R_j = quat_to_matrix(q_j)
        hkl_i, dev_i = plane_miller(R_i.T @ n_hat, A_i)
        hkl_j, dev_j = plane_miller(R_j.T @ n_hat, A_j)
        if same_phase and axis_crystal is not None:
            # Character angle ψ between the misorientation axis and
            # the boundary normal, both in the LAB frame.  The axis
            # comes from the physical minimal-angle map (see module
            # docstring) — NOT from mapping the symmetry-reduced
            # crystal representative.
            axis_lab, _ = _physical_min_axis_lab(q_i, q_j, sym_pair)
            cos_psi = abs(float(np.dot(axis_lab, n_hat)))
            psi = float(np.degrees(np.arccos(np.clip(cos_psi,
                                                     0.0, 1.0))))
            if psi < CHAR_TWIST_DEG:
                char_str = "twist"
            elif psi > CHAR_TILT_DEG:
                char_str = "tilt"
            else:
                char_str = "mixed"
        else:
            # tilt/twist needs a misorientation axis — undefined
            # across point groups.
            char_str = "undefined" if same_phase else "interphase"
            psi = float("nan")

    sigma = ""
    if csl and dis is not None:
        sigma = _csl_match(dis.angle_deg, dis.axis_crystal, sym_pair)

    return (mis_deg, axis_uvw, float(axis_dev), area_total, mean_n,
            float(spread), hkl_i, float(dev_i), hkl_j, float(dev_j),
            char_str, psi, sigma, axis_crystal)


def _build_report(
    i: int, j: int, seed_dist: float, fields: tuple, n_deleted: int,
    phase_i: str, phase_j: str,
) -> BoundaryReport:
    """Assemble one boundaries.csv row from _pair_report_fields' output
    plus the cheap parent-side bookkeeping (seed distance, overlap-ledger
    lookup, phase names) — shared by the serial and parallel paths."""
    (mis_deg, axis_uvw, axis_dev, area_total, mean_n, spread,
     hkl_i, dev_i, hkl_j, dev_j, char_str, psi, sigma,
     axis_crystal) = fields
    return BoundaryReport(
        grain_i=i,
        grain_j=j,
        seed_distance_A=seed_dist,
        misorientation_deg=mis_deg,
        axis_u=int(axis_uvw[0]),
        axis_v=int(axis_uvw[1]),
        axis_w=int(axis_uvw[2]),
        axis_dev_deg=axis_dev,
        area_A2=area_total,
        mean_normal_x=float(mean_n[0]),
        mean_normal_y=float(mean_n[1]),
        mean_normal_z=float(mean_n[2]),
        normal_spread_deg=spread,
        plane_i_hkl=("(" + " ".join(str(v) for v in hkl_i) + ")")
        if hkl_i else "",
        plane_i_dev_deg=dev_i,
        plane_j_hkl=("(" + " ".join(str(v) for v in hkl_j) + ")")
        if hkl_j else "",
        plane_j_dev_deg=dev_j,
        character=char_str,
        character_angle_deg=psi,
        csl_sigma=sigma,
        n_overlap_deleted=n_deleted,
        phase_i=phase_i,
        phase_j=phase_j,
        axis_crystal=axis_crystal,
        mean_normal=mean_n,
    )


# ---------------------------------------------------------------------------
# Parallel driver (voxel backend only — flat's exact-face geometry is cheap,
# same design choice as analysis/curvature.py's always-serial flat path).
# ---------------------------------------------------------------------------

_BOUNDARY_CTX: dict | None = None
"""Per-WORKER-PROCESS boundary-geometry context, set once by the pool
initializer. Each worker is a separate process (Windows spawn / POSIX
fork), so this is process-local plumbing for ProcessPoolExecutor — not
shared mutable module state in the §15 sense (same rationale as
analysis/curvature.py's _CURVATURE_CTX). Also carries syms_table/As_table
(the per-phase symmetry quaternions / cell matrices) so those are pickled
ONCE per worker process instead of once per task — tasks ship only the
(p_i, p_j) phase indices and look sym_pair/A_i/A_j back up here."""


def _init_boundary_worker(ctx: dict) -> None:
    global _BOUNDARY_CTX
    _BOUNDARY_CTX = ctx


def _boundary_pair_task(args: tuple) -> tuple:
    i, j, pts, areas, q_i, q_j, p_i, p_j = args
    c = _BOUNDARY_CTX
    assert c is not None  # initializer ran before any task
    from grainsmith.tessellation.voxel import _gb_pair_normals
    normals, ok = _gb_pair_normals(c["tess"], c["h_vec"], i, j, pts)
    sym_pair = c["syms_table"][p_i]
    A_i = c["As_table"][p_i]
    A_j = c["As_table"][p_j]
    return _pair_report_fields(
        normals[ok], areas[ok], q_i, q_j, sym_pair, A_i, A_j,
        p_i == p_j, c["csl"], c["character"])


def _boundary_fields_voxel_parallel(
    tess, box_lengths, periodic, quats,
    pairs: list[tuple[int, int]],
    frames: list[tuple[int, int, bool, np.ndarray, np.ndarray, np.ndarray]],
    syms_table: list[np.ndarray], As_table: list[np.ndarray],
    csl: bool, character: bool, jobs: int,
) -> list[tuple]:
    """Process-parallel counterpart of the serial per-pair loop's geometry
    + misorientation/character/CSL computation, for the voxel backend.

    Sample points are computed here, parent-side, from vg.gb_face_samples
    (cheap — memoized on the VoxelGrid instance, tessellation/voxel.py's
    _gb_faces cache), exactly as the serial path's _pair_geometry_voxel /
    vg.gb_normals does. *syms_table*/*As_table* (indexed by phase id — see
    analyze_boundaries) are pickled ONCE via the pool initializer instead
    of once per task: only (i, j, pts, areas, q_i, q_j, p_i, p_j) crosses
    into each worker, which looks sym_pair/A_i/A_j back up from its ctx
    by phase index (mirrors _pair_frames' lookup, done worker-side).

    Pairs with no retained face samples never reach a worker — mirrors
    analysis/curvature.py's _curvature_voxel_parallel idiom exactly:
    *tasks* is filtered by `if len(pts)`, and *results* is a lockstep
    generator that pulls the next pool result for a non-empty pair and,
    for an empty one, computes the (geometry-free, cheap) fields directly
    in the parent process via the SAME _pair_report_fields the workers
    call — so an empty pair never round-trips through the pool at all.
    pool.map preserves task order, and both *tasks* and *results* are
    built by iterating *pairs* (tess.adjacency() order) with the identical
    `len(pts)` test, so the returned list is already in adjacency order —
    positional zip with *pairs* in the caller needs no extra reordering.
    Every field stays byte-identical for every jobs value (see
    _pair_report_fields' docstring); this function only decides WHERE
    (worker vs. parent) each pair's identical computation runs.
    """
    from concurrent.futures import ProcessPoolExecutor

    from grainsmith.analysis.grains import get_voxel_grid
    vg = get_voxel_grid(tess, box_lengths)
    if vg.tess is None:
        raise GrainsmithError(
            "gb_normals requires the VoxelGrid to hold its source "
            "tessellation (build via build_voxel_grid)."
        )
    faces = vg.gb_face_samples(periodic)
    ctx = {"tess": vg.tess, "h_vec": np.asarray(vg.h_vec, dtype=np.float64),
           "csl": csl, "character": character,
           "syms_table": syms_table, "As_table": As_table}
    samples = [faces.get((i, j), _EMPTY_PTS_AREAS) for (i, j) in pairs]
    tasks = [
        (i, j, pts, areas, quats[i], quats[j], p_i, p_j)
        for (i, j), (p_i, p_j, *_rest), (pts, areas)
        in zip(pairs, frames, samples, strict=True)
        if len(pts)
    ]
    with ProcessPoolExecutor(
        max_workers=min(jobs, len(pairs)),
        initializer=_init_boundary_worker,
        initargs=(ctx,),
    ) as pool:
        result_iter = pool.map(_boundary_pair_task, tasks)
        results = (
            next(result_iter) if len(pts) else
            _pair_report_fields(
                pts, areas, quats[i], quats[j],
                syms_table[p_i], As_table[p_i], As_table[p_j],
                p_i == p_j, csl, character)
            for (i, j), (p_i, p_j, *_rest), (pts, areas)
            in zip(pairs, frames, samples, strict=True)
        )
        return list(results)


def pair_areas(
    tess: Tessellation,
    box_lengths: np.ndarray,
    periodic: list[bool],
) -> dict[tuple[int, int], float]:
    """Total GB area per adjacency pair {(i, j) : A_ij} (i < j).

    Same geometry sources as :func:`analyze_boundaries` (exact faces for
    flat/power, voxel faces for curved) — the MDF annealing weights
    its misorientation histogram with these areas, so the
    annealed energy and the re-measured gate G12 histogram agree by
    construction.
    """
    from grainsmith.tessellation.flat import FlatTessellation
    pairs = tess.adjacency()
    if isinstance(tess, FlatTessellation):
        geometry = _pair_geometry_flat(tess, pairs)
    else:
        geometry = _pair_geometry_voxel(tess, box_lengths, periodic)
    return {
        pair: float(np.sum(areas))
        for pair, (_normals, areas) in geometry.items()
    }


def _mean_normal_and_spread(
    normals: np.ndarray, areas: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Area-weighted mean unit normal and RMS angular spread (deg).

    Returns (None, 90.0) when the resultant is degenerate."""
    w_total = float(np.sum(areas))
    if w_total <= 0.0 or len(normals) == 0:
        return None, 90.0
    resultant = (areas[:, None] * normals).sum(axis=0) / w_total
    r_len = float(np.linalg.norm(resultant))
    if r_len < _MIN_RESULTANT:
        return None, 90.0
    n_hat = resultant / r_len
    cosines = np.clip(normals @ n_hat, -1.0, 1.0)
    theta = np.degrees(np.arccos(cosines))
    spread = float(np.sqrt(np.sum(areas * theta**2) / w_total))
    return n_hat, spread


def _physical_min_axis_lab(
    q_i: np.ndarray,
    q_j: np.ndarray,
    sym_quats: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Axis (lab frame, unit) and angle (deg) of the minimal-angle physical
    map G = R_j · R(S) · R_iᵀ over the proper point-group rotations S.

    See the module docstring: this axis — not the symmetry-reduced
    crystal-frame representative — is the well-defined input to the
    tilt/twist character angle ψ."""
    from grainsmith.crystal.pointgroup import _matrix_to_quat
    from grainsmith.orientation.quaternion import quat_to_axis_angle

    R_i = quat_to_matrix(q_i)
    R_j = quat_to_matrix(q_j)
    best_angle = np.inf
    best_G = None
    for s in sym_quats:
        G = R_j @ quat_to_matrix(s) @ R_i.T
        cos_t = np.clip((np.trace(G) - 1.0) / 2.0, -1.0, 1.0)
        ang = float(np.degrees(np.arccos(cos_t)))
        if ang < best_angle:
            best_angle = ang
            best_G = G
    assert best_G is not None  # sym_quats is never empty (identity ∈ group)
    axis, angle = quat_to_axis_angle(_matrix_to_quat(best_G))
    return axis, angle


def _misorientation_class_deviation(
    q_a: np.ndarray,
    q_b: np.ndarray,
    sym_quats: np.ndarray,
) -> float:
    """Angular distance (deg) between two misorientation CLASSES.

    Misorientations are double cosets Sym·m·Sym (plus grain exchange), so
    the representative-independent deviation is
        min_{S_a, S_b, ±} angle( q_a⁻¹ ⊗ S_a ⊗ q_b^{±1} ⊗ S_b ).
    Note S_a sits INSIDE the product — reducing only the relative rotation
    q_a⁻¹⊗q_b (a plain disorientation call) is NOT class-invariant and
    misses exact matches between equivalent representatives.

    Vectorized over the full 2 × Nsym × Nsym candidate array (the minimum
    angle equals 2·arccos of the maximum |w|).
    """
    from grainsmith.orientation.quaternion import quat_inv, quat_mul_batch

    sym = np.asarray(sym_quats, dtype=np.float64)
    bases = np.stack([q_b, quat_inv(q_b)])                       # (2, 4)
    # left[base, a] = q_a⁻¹ ⊗ S_a ⊗ base
    left = quat_mul_batch(
        quat_inv(q_a)[None, None, :],
        quat_mul_batch(sym[None, :, :], bases[:, None, :]),
    )
    # cand[base, a, b] = left ⊗ S_b
    cand = quat_mul_batch(left[:, :, None, :], sym[None, None, :, :])
    w_max = float(np.max(np.abs(cand[..., 0])))
    return 2.0 * float(np.degrees(np.arccos(min(w_max, 1.0))))


def _csl_match(
    angle_deg: float,
    axis_crystal: np.ndarray,
    sym_quats: np.ndarray,
) -> str:
    """Smallest Σ of the cubic table satisfying the Brandon criterion
    Δθ ≤ CSL_BRANDON_FACTOR · Σ^(−1/2)."""
    q_dis = axis_angle_to_quat(axis_crystal, angle_deg)
    best = ""
    best_sigma = np.inf
    for label, (theta_s, axis_s) in CSL_TABLE.items():
        sigma = float(int("".join(ch for ch in label if ch.isdigit())))
        if sigma >= best_sigma:
            continue
        brandon = CSL_BRANDON_FACTOR / np.sqrt(sigma)
        # Cheap angle prefilter: the class deviation is bounded below by
        # the difference of disorientation angles.
        if abs(angle_deg - theta_s) > brandon:
            continue
        q_sigma = axis_angle_to_quat(np.array(axis_s, dtype=np.float64), theta_s)
        dev = _misorientation_class_deviation(q_sigma, q_dis, sym_quats)
        if dev <= brandon:
            best = label
            best_sigma = sigma
    return best


def analyze_boundaries(
    tess: Tessellation,
    quats: np.ndarray,
    sym_quats: np.ndarray,
    A: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    ledger=None,
    csl: bool = False,
    character: bool = True,
    phase_of: np.ndarray | None = None,
    phase_names: list[str] | None = None,
    sym_quats_list: list[np.ndarray] | None = None,
    A_list: list[np.ndarray] | None = None,
    jobs: int = 1,
) -> list[BoundaryReport]:
    """Build the per-boundary report (§6.10 / boundaries.csv §8.3).

    Parameters
    ----------
    tess : Tessellation
    quats : (N, 4) grain orientations (scalar-first)
    sym_quats : (Nsym, 4) proper point-group rotations (§6.3)
    A : (3, 3) conventional cell matrix
    ledger : OverlapLedger | None → n_overlap_deleted column
    csl : match against the cubic CSL table (raises ConfigError for
        non-cubic point groups, §6.10).
    character : compute GB character (tilt/twist/mixed) and boundary-plane
        Miller indices (``analysis.gb_character``).  When False those
        columns are left empty/NaN; geometry and misorientation columns
        are always filled.
    phase_of, phase_names, sym_quats_list, A_list : multiphase runs.
        Same-phase pairs use THEIR phase's point group and
        cell; interphase pairs have no common point group, so the
        misorientation/axis columns are NaN/(000) and ``character`` is
        ``"interphase"`` — but the per-frame boundary-plane Miller
        indices stay defined (plane_i via A_i, plane_j via A_j: habit
        planes).  *sym_quats*/*A* are ignored when these are given; CSL
        is rejected at config time (resolve Rule 22).
    jobs : worker processes for the voxel backend's per-(i, j)-pair
        geometry + misorientation/character/CSL computation (§13; CLI
        ``--jobs``). 1 = in-process serial (default). The flat backend
        is always serial regardless of jobs — its exact-face geometry
        lookup is cheap (mirrors analyze_curvature's flat path: "trivial
        per-face work", nothing to gain from worker processes). >1 with
        >= 2 boundary pairs on a voxel-backed tessellation: the SAME
        per-pair kernel the serial loop below calls (_pair_frames +
        tessellation.voxel._gb_pair_normals + _pair_report_fields) runs
        in a worker pool instead (_boundary_fields_voxel_parallel), so
        every BoundaryReport field is bit-identical for every jobs value
        — same guarantee and argument as analyze_curvature's jobs.
    """
    if csl and len(sym_quats) != _CUBIC_N_PROPER:
        raise ConfigError(
            f"analysis.csl is restricted to cubic point groups "
            f"(found {len(sym_quats)} proper rotations, expected "
            f"{_CUBIC_N_PROPER}). Set analysis.csl: false."
        )

    L = np.asarray(box_lengths, dtype=np.float64)
    seeds = tess.seeds
    pairs = tess.adjacency()

    from grainsmith.tessellation.flat import FlatTessellation
    is_flat = isinstance(tess, FlatTessellation)

    multiphase = phase_of is not None
    if phase_of is not None:
        assert (phase_names is not None and sym_quats_list is not None
                and A_list is not None)
        # mypy-narrowed locals for the loop below.
        _pof, _names, _syms, _As = phase_of, phase_names, sym_quats_list, A_list
    else:
        _pof = np.empty(0, dtype=np.int32)
        _names, _syms, _As = [], [], []

    frames = [
        _pair_frames(i, j, multiphase, _pof, _syms, _As, sym_quats, A)
        for (i, j) in pairs
    ]

    if not is_flat and jobs > 1 and len(pairs) > 1:
        # Per-phase symmetry/cell tables (length 1 for a single-phase run,
        # one entry per phase for multiphase) — built once here so the
        # parallel driver can ship them to each worker ONCE, via the pool
        # initializer, instead of re-pickling a pair's sym_pair/A_i/A_j
        # (which repeat across every same-phase pair) per task.
        syms_table = _syms if multiphase else [sym_quats]
        As_table = _As if multiphase else [A]
        field_results = _boundary_fields_voxel_parallel(
            tess, L, periodic, quats, pairs, frames,
            syms_table, As_table, csl, character, jobs)
    else:
        geometry: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
        if is_flat:
            geometry = _pair_geometry_flat(tess, pairs)
        else:
            geometry = _pair_geometry_voxel(tess, L, periodic)
        field_results = [
            _pair_report_fields(
                *geometry.get((i, j), _EMPTY_NORMALS_AREAS),
                quats[i], quats[j], frame[3], frame[4], frame[5],
                frame[2], csl, character)
            for (i, j), frame in zip(pairs, frames, strict=True)
        ]

    reports: list[BoundaryReport] = []
    for (i, j), (p_i, p_j, *_rest), fields in zip(
            pairs, frames, field_results, strict=True):
        # Seed separation (min-image on periodic axes)
        dr = seeds[j] - seeds[i]
        for ax in range(3):
            if periodic[ax]:
                dr[ax] -= np.round(dr[ax] / L[ax]) * L[ax]
        seed_dist = float(np.linalg.norm(dr))

        n_deleted = 0
        if ledger is not None:
            n_deleted = int(ledger.deletions_by_pair.get((i, j), 0))

        reports.append(_build_report(
            i, j, seed_dist, fields, n_deleted,
            _names[p_i] if multiphase else "",
            _names[p_j] if multiphase else "",
        ))
    return reports
