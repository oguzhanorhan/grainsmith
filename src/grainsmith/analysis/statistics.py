"""Microstructure statistics for the publication outputs.

``compute_statistics`` assembles the statistics.csv content as
an ordered ``{section: {key: value}}`` dict (serialized by
io/publication.py and embedded into microstructure.json and
RunResult.statistics).  Everything is RE-MEASURED from the final
tessellation/reports — never taken from an optimizer's claim.

Estimator honesty (echoed per-section in an ``estimator`` key):

equivalent diameters / log-normal fit
    d_eq = (6V/π)^(1/3) from the per-grain volumes (exact for flat/power
    cells and the single crystal; voxel-counted for curved backends).
    The fit is the mean / unbiased-sample-std (ddof=1) plug-in for LogN on ln d
    (μ̂ = mean, σ̂ = std with ddof=1) — close to, but not exactly, the MLE,
    whose σ̂ uses ddof=0;
    the KS statistic/p-value are computed against the FITTED distribution,
    so the p-value is optimistic (the classical Lilliefors caveat) — it is
    a descriptive goodness-of-fit number, not a calibrated test.

sphericity Ψ = π^(1/3)·(6V)^(2/3) / A
    flat/power: exact polyhedral face areas.  curved/import: voxel-face
    (staircase) areas, whose area/true ratio equals ‖n̂‖₁ for a face with
    unit normal n̂ — exactly 1 for an axis-aligned face, √2 for a single-axis
    45° tilt e.g. (110), and a maximum of √3 ≈ 1.732 at the body-diagonal
    (111) orientation (NOT a single 45° tilt).  This
    OVERESTIMATES oblique boundaries and hence UNDERESTIMATES Ψ — the bias is
    systematic and documented, not corrected.  Single crystal: the box
    surface (the grain IS the box).

S_V (GB area per volume)
    Σ of the per-pair boundary areas over the box volume; same geometry
    sources as boundaries.csv.  Same-grain periodic self-image sheets are
    not inter-grain boundaries and are excluded (they are the G14 story
    for single crystals).  Multiphase runs split S_V into same-phase GB
    and interphase area.

L_V (triple-junction line length per volume)
    flat/power: EXACT — cell-face edges are deduplicated on a quantized
    wrapped-coordinate key and an edge counts when ≥ 3 distinct grains
    meet there.  curved/import: voxel-edge estimator (a grid edge counts
    when its 4 surrounding voxels carry ≥ 3 distinct labels), which
    overestimates oblique lines exactly like the face-count area
    estimator.  Junction lines of ≥ 4 grains count as triple lines
    (they are junction lines); in very small periodic systems a junction
    formed by < 3 DISTINCT grains (periodic self-images) is not counted.
"""
from __future__ import annotations

import numpy as np

from grainsmith.constants import (
    EULER_TOL,
    LAGB_MAX_DEG,
    LOGNORMAL_SIGMA_MIN,
)
from grainsmith.tessellation.base import Tessellation

__all__ = [
    "compute_statistics",
    "equivalent_diameters",
    "grain_surface_areas_flat",
    "grain_surface_areas_voxel",
    "lognormal_fit",
    "sphericity",
    "triple_line_length_flat",
    "triple_line_length_voxel",
]


# ---------------------------------------------------------------------------
# Grain size
# ---------------------------------------------------------------------------


def equivalent_diameters(volumes: np.ndarray) -> np.ndarray:
    """Equivalent-sphere diameters d = (6V/π)^(1/3) in Å."""
    v = np.asarray(volumes, dtype=np.float64)
    return np.cbrt(6.0 * v / np.pi)


def lognormal_fit(diameters: np.ndarray) -> dict[str, float]:
    """Log-normal plug-in fit of the diameters + KS goodness-of-fit.

    Returns ``{mu_hat, sigma_hat, ks_statistic, ks_p}``; μ̂/σ̂ are the
    mean/unbiased-std (ddof=1) of ln(d/Å) — the near-MLE plug-in (the exact
    MLE σ̂ uses ddof=0).  The KS pair is NaN when the fit is
    degenerate (n < 2 or σ̂ < LOGNORMAL_SIGMA_MIN, e.g. equal-volume
    targets) — see the module docstring for the fitted-parameter
    (Lilliefors) caveat.
    """
    d = np.asarray(diameters, dtype=np.float64)
    n = len(d)
    if n == 0 or np.any(d <= 0.0):
        return {"mu_hat": float("nan"), "sigma_hat": float("nan"),
                "ks_statistic": float("nan"), "ks_p": float("nan")}
    ln_d = np.log(d)
    mu = float(np.mean(ln_d))
    sigma = float(np.std(ln_d, ddof=1)) if n >= 2 else float("nan")
    ks_stat = ks_p = float("nan")
    if n >= 2 and np.isfinite(sigma) and sigma > LOGNORMAL_SIGMA_MIN:
        from scipy import stats
        res = stats.kstest(d, "lognorm", args=(sigma, 0.0, float(np.exp(mu))))
        ks_stat = float(res.statistic)
        ks_p = float(res.pvalue)
    return {"mu_hat": mu, "sigma_hat": sigma,
            "ks_statistic": ks_stat, "ks_p": ks_p}


# ---------------------------------------------------------------------------
# Sphericity
# ---------------------------------------------------------------------------


def sphericity(volumes: np.ndarray, surface_areas: np.ndarray) -> np.ndarray:
    """Wadell sphericity Ψ = π^(1/3)·(6V)^(2/3) / A per grain (Ψ = 1 for a
    sphere, (π/6)^(1/3) ≈ 0.806 for a cube)."""
    v = np.asarray(volumes, dtype=np.float64)
    a = np.asarray(surface_areas, dtype=np.float64)
    return np.pi ** (1.0 / 3.0) * np.cbrt((6.0 * v) ** 2) / a


def grain_surface_areas_flat(tess) -> np.ndarray:
    """Exact per-grain surface areas (Å²) from the polyhedral cell faces
    (GB faces, periodic self-image faces AND box-wall free surfaces — the
    grain's total surface, as sphericity requires)."""
    return np.array(
        [sum(f.area for f in cell.faces) for cell in tess.cells],
        dtype=np.float64)


def grain_surface_areas_voxel(vg, periodic: list[bool]) -> np.ndarray:
    """Voxel-face per-grain surface areas (Å²): label-change faces plus
    free-axis box-wall faces.  Overestimates oblique boundaries by up to
    √3 (§6.7) — documented, not corrected."""
    lab = vg.labels
    vox_vol = float(np.prod(vg.h_vec))
    areas = np.zeros(vg.n_grains, dtype=np.float64)
    for ax in range(3):
        face_area = vox_vol / float(vg.h_vec[ax])
        nb = np.roll(lab, -1, axis=ax)
        diff = lab != nb
        if not periodic[ax]:
            # The roll's wrap pair is not a physical contact; the two box
            # walls are free surfaces of the touching grains instead.
            sl = [slice(None)] * 3
            sl[ax] = slice(-1, None)
            diff[tuple(sl)] = False
            lo = [slice(None)] * 3
            lo[ax] = slice(0, 1)
            hi = [slice(None)] * 3
            hi[ax] = slice(-1, None)
            for wall in (lo, hi):
                counts = np.bincount(lab[tuple(wall)].ravel(),
                                     minlength=vg.n_grains)
                areas += counts[:vg.n_grains] * face_area
        counts_i = np.bincount(lab[diff].ravel(), minlength=vg.n_grains)
        counts_j = np.bincount(nb[diff].ravel(), minlength=vg.n_grains)
        areas += (counts_i[:vg.n_grains]
                  + counts_j[:vg.n_grains]) * face_area
    return areas


# ---------------------------------------------------------------------------
# Triple-junction line length
# ---------------------------------------------------------------------------


def triple_line_length_flat(tess, box_lengths: np.ndarray,
                            periodic: list[bool]) -> float:
    """EXACT total triple-junction line length (Å) of a flat/power
    tessellation: cell-face edges where ≥ 3 distinct grains meet.

    Edges are deduplicated on a quantized key of their wrapped endpoint
    coordinates (quantum EULER_TOL × max L — the G4 vertex tolerance);
    every cell touching an edge contributes its full grain triple {i, j, k}
    through its two faces at the edge, so each unique edge accumulates the
    complete set of distinct grains.  Wall faces (free surfaces) define no
    grain-boundary edges; an edge where two GB faces meet a wall has only
    2 distinct grains and correctly does not count.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    tol = EULER_TOL * float(np.max(L))
    n_quant = [max(1, int(round(L[ax] / tol))) for ax in range(3)]

    def key(v: np.ndarray) -> tuple[int, int, int]:
        out = []
        for ax in range(3):
            if periodic[ax]:
                q = int(round((v[ax] % L[ax]) / tol)) % n_quant[ax]
            else:
                q = int(round(v[ax] / tol))
            out.append(q)
        return (out[0], out[1], out[2])

    edges: dict[tuple, tuple[float, set[int]]] = {}
    for cell in tess.cells:
        gid = cell.grain_id
        for face in cell.faces:
            if face.neighbor_id < 0:        # box wall — no GB edge
                continue
            verts = face.vertices
            m = len(verts)
            for k in range(m):
                v_a = verts[k]
                v_b = verts[(k + 1) % m]
                length = float(np.linalg.norm(v_b - v_a))
                if length < tol:            # degenerate sliver edge
                    continue
                ka, kb = key(v_a), key(v_b)
                ekey = (ka, kb) if ka <= kb else (kb, ka)
                rec = edges.get(ekey)
                if rec is None:
                    rec = (length, set())
                    edges[ekey] = rec
                rec[1].add(gid)
                rec[1].add(int(face.neighbor_id))
    return float(sum(length for length, grains in edges.values()
                     if len(grains) >= 3))


def triple_line_length_voxel(vg, periodic: list[bool]) -> float:
    """Voxel-edge estimator of the triple-junction line length (Å): a grid
    edge counts (with its axis spacing) when the 4 voxels around it carry
    ≥ 3 distinct labels.  Distinctness via the pair-equality count of the
    4 values: ≥ 3 distinct ⇔ at most one of the 6 pairs is equal."""
    lab = vg.labels
    total = 0.0
    for ax in range(3):
        p1, p2 = [k for k in range(3) if k != ax]
        a = lab
        b = np.roll(lab, -1, axis=p1)
        c = np.roll(lab, -1, axis=p2)
        d = np.roll(b, -1, axis=p2)
        eq = ((a == b).astype(np.uint8) + (a == c) + (a == d)
              + (b == c) + (b == d) + (c == d))
        valid = eq <= 1
        # The roll wrap rows are physical edges only on periodic axes.
        if not periodic[p1]:
            sl = [slice(None)] * 3
            sl[p1] = slice(-1, None)
            valid[tuple(sl)] = False
        if not periodic[p2]:
            sl = [slice(None)] * 3
            sl[p2] = slice(-1, None)
            valid[tuple(sl)] = False
        total += float(np.count_nonzero(valid)) * float(vg.h_vec[ax])
    return total


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------


def _size_section(d: np.ndarray) -> dict[str, object]:
    fit = lognormal_fit(d)
    n = len(d)
    return {
        "n_grains": int(n),
        "d_eq_mean_A": float(np.mean(d)) if n else float("nan"),
        "d_eq_std_A": float(np.std(d, ddof=1)) if n >= 2 else float("nan"),
        "d_eq_min_A": float(np.min(d)) if n else float("nan"),
        "d_eq_max_A": float(np.max(d)) if n else float("nan"),
        "lognormal_mu_hat": fit["mu_hat"],
        "lognormal_sigma_hat": fit["sigma_hat"],
        "lognormal_ks_statistic": fit["ks_statistic"],
        "lognormal_ks_p": fit["ks_p"],
    }


def _spread_section(psi: np.ndarray, estimator: str) -> dict[str, object]:
    n = len(psi)
    return {
        "estimator": estimator,
        "mean": float(np.mean(psi)) if n else float("nan"),
        "std": float(np.std(psi, ddof=1)) if n >= 2 else float("nan"),
        "min": float(np.min(psi)) if n else float("nan"),
        "max": float(np.max(psi)) if n else float("nan"),
    }


def _mdf_section(reports: list) -> dict[str, object]:
    """Area-weighted misorientation scalars over SAME-PHASE boundaries."""
    ang = np.array([r.misorientation_deg for r in reports
                    if np.isfinite(r.misorientation_deg)],
                   dtype=np.float64)
    ar = np.array([r.area_A2 for r in reports
                   if np.isfinite(r.misorientation_deg)], dtype=np.float64)
    a_tot = float(np.sum(ar))
    if len(ang) == 0 or a_tot <= 0.0:
        return {"n_boundaries": int(len(ang)),
                "mean_misorientation_deg": float("nan"),
                "lagb_area_fraction": float("nan")}
    return {
        "n_boundaries": int(len(ang)),
        "mean_misorientation_deg": float(np.sum(ar * ang) / a_tot),
        "lagb_area_fraction": float(np.sum(ar[ang < LAGB_MAX_DEG]) / a_tot),
    }


def compute_statistics(
    tess: Tessellation,
    volumes: np.ndarray,
    grain_reports: list,
    boundary_reports: list,
    box_lengths: np.ndarray,
    periodic: list[bool],
    voxel_grid=None,
    phase_of: np.ndarray | None = None,
    phase_names: list[str] | None = None,
    character: bool = True,
    csl: bool = False,
    hurst_target: float | None = None,
    hurst_estimated: float | None = None,
) -> dict[str, dict[str, object]]:
    """Assemble the statistics.csv sections — phase-aware.

    *voxel_grid* is required for curved/imported backends (the voxel
    estimators); flat/power and single-crystal runs are exact without it.
    When phases are active the grain-size and sphericity sections
    repeat per phase and S_V splits into same-phase GB vs interphase area.
    """
    from grainsmith.tessellation.flat import FlatTessellation
    from grainsmith.tessellation.single import SingleCrystalTessellation

    L = np.asarray(box_lengths, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    v_total = float(np.sum(volumes))
    multiphase = phase_of is not None
    stats: dict[str, dict[str, object]] = {}

    # --- grain size (per phase when active) ---
    d = equivalent_diameters(volumes)
    stats["grain_size"] = _size_section(d)
    if multiphase:
        assert phase_names is not None
        for p, name in enumerate(phase_names):
            stats[f"grain_size:{name}"] = _size_section(d[phase_of == p])

    # --- sphericity ---
    if isinstance(tess, SingleCrystalTessellation):
        # General parallelepiped surface area 2*(|a×b| + |b×c| + |a×c|) —
        # reduces exactly to 2*(Lx*Ly + Ly*Lz + Lx*Lz) when the box matrix
        # is diagonal (orthogonal box); the triclinic (box.cells) box uses
        # its own tilted cell matrix.
        H = tess.cell_matrix
        a, b, c = H[:, 0], H[:, 1], H[:, 2]
        surf = np.array([2.0 * (
            float(np.linalg.norm(np.cross(a, b)))
            + float(np.linalg.norm(np.cross(b, c)))
            + float(np.linalg.norm(np.cross(a, c)))
        )])
        estimator = "exact"
    elif isinstance(tess, FlatTessellation):
        surf = grain_surface_areas_flat(tess)
        estimator = "exact"
    else:
        assert voxel_grid is not None, \
            "curved/imported backends need the analysis voxel grid"
        surf = grain_surface_areas_voxel(voxel_grid, periodic)
        estimator = "voxel"
    psi = sphericity(volumes, surf)
    stats["sphericity"] = _spread_section(psi, estimator)
    if multiphase:
        assert phase_names is not None
        for p, name in enumerate(phase_names):
            stats[f"sphericity:{name}"] = _spread_section(
                psi[phase_of == p], estimator)

    # --- topology: faces (= adjacent neighbors) per grain ---
    faces = np.array([r.n_neighbors for r in grain_reports], dtype=np.int64)
    topo: dict[str, object] = {
        "faces_per_grain_mean": float(np.mean(faces)) if len(faces)
                                else float("nan"),
        "faces_per_grain_min": int(np.min(faces)) if len(faces) else 0,
        "faces_per_grain_max": int(np.max(faces)) if len(faces) else 0,
    }
    for k in np.unique(faces):
        topo[f"n_grains_with_{int(k)}_faces"] = int(np.sum(faces == k))
    stats["topology"] = topo

    # --- boundary area / S_V ---
    areas = np.array([r.area_A2 for r in boundary_reports], dtype=np.float64)
    a_total = float(np.sum(areas))
    bsec: dict[str, object] = {
        "n_boundaries": int(len(boundary_reports)),
        "estimator": estimator,
        "gb_area_total_A2": a_total,
        "S_V_per_A": a_total / v_total,
    }
    if multiphase:
        same = np.array([r.phase_i == r.phase_j for r in boundary_reports],
                        dtype=bool)
        a_same = float(np.sum(areas[same]))
        a_inter = float(np.sum(areas[~same]))
        bsec["gb_same_phase_area_A2"] = a_same
        bsec["gb_interphase_area_A2"] = a_inter
        bsec["S_V_same_phase_per_A"] = a_same / v_total
        bsec["S_V_interphase_per_A"] = a_inter / v_total
    stats["boundaries"] = bsec

    # --- triple junctions ---
    if isinstance(tess, SingleCrystalTessellation):
        tj_len, tj_est = 0.0, "exact"
    elif isinstance(tess, FlatTessellation):
        tj_len, tj_est = triple_line_length_flat(tess, L, periodic), "exact"
    else:
        assert voxel_grid is not None
        tj_len, tj_est = triple_line_length_voxel(voxel_grid,
                                                  periodic), "voxel"
    stats["triple_junctions"] = {
        "estimator": tj_est,
        "length_total_A": tj_len,
        "L_V_per_A2": tj_len / v_total,
    }

    # --- GB character area fractions ---
    if character:
        cats = ["twist", "tilt", "mixed", "undefined"]
        if multiphase:
            cats.append("interphase")
        csec: dict[str, object] = {}
        for cat in cats:
            a_cat = float(np.sum(areas[[r.character == cat
                                        for r in boundary_reports]])) \
                if len(boundary_reports) else 0.0
            csec[f"area_fraction_{cat}"] = (a_cat / a_total if a_total > 0.0
                                            else float("nan"))
        stats["gb_character"] = csec

    # --- CSL area fractions ---
    if csl:
        labels = [r.csl_sigma for r in boundary_reports]
        ssec: dict[str, object] = {}
        a_csl = 0.0
        present = sorted(
            {lb for lb in labels if lb},
            key=lambda s: (int("".join(ch for ch in s if ch.isdigit())), s))
        for lb in present:
            a_lb = float(np.sum(areas[[x == lb for x in labels]]))
            a_csl += a_lb
            ssec[f"area_fraction_sigma{lb}"] = (a_lb / a_total
                                                if a_total > 0.0
                                                else float("nan"))
        ssec["area_fraction_non_csl"] = ((a_total - a_csl) / a_total
                                         if a_total > 0.0 else float("nan"))
        stats["csl"] = ssec

    # --- MDF scalars (within one point group ⇒ per phase when active) ---
    if multiphase:
        assert phase_names is not None
        for name in phase_names:
            stats[f"mdf:{name}"] = _mdf_section(
                [r for r in boundary_reports
                 if r.phase_i == name and r.phase_j == name])
    else:
        stats["mdf"] = _mdf_section(boundary_reports)

    # --- self-affine roughness ---
    if hurst_target is not None:
        stats["roughness"] = {
            "hurst_target": float(hurst_target),
            "hurst_estimated": (float(hurst_estimated)
                                if hurst_estimated is not None
                                else float("nan")),
        }

    return stats
