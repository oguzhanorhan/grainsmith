"""CSV report writers (§8.3) — exact column contracts.

The column lists below are the §8.3 specification verbatim; tests pin them
(test_reports).  Serialization uses the stdlib ``csv`` module (RFC 4180
quoting; no pandas, §12).  CSV carries no provenance comment line — RFC 4180
has no comment syntax and §8.3 pins the header row as the first line; the
provenance data lives in summary.csv rows instead (see io/common.py).

Floats are written with shortest-round-trip repr (io.common.fmt) so reports
are deterministic and lossless.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from grainsmith.analysis.boundaries import BoundaryReport
from grainsmith.analysis.curvature import CurvatureResult
from grainsmith.analysis.grains import GrainReport
from grainsmith.atoms.doping import DopingReportRow
from grainsmith.constants import EULER_TOL
from grainsmith.io.common import atomic_writer, fmt
from grainsmith.tessellation.flat import FlatTessellation

GRAINS_COLUMNS: list[str] = [
    "grain_id", "seed_x", "seed_y", "seed_z", "volume_A3", "volume_fraction",
    "n_atoms", "n_neighbors", "q_w", "q_x", "q_y", "q_z", "euler_phi1_deg",
    "euler_Phi_deg", "euler_phi2_deg", "axis_x", "axis_y", "axis_z",
    "angle_deg", "z_plane_hkl", "z_plane_dev_deg", "x_dir_uvw",
    "x_dir_dev_deg",
]
"""grains.csv columns — §8.3 verbatim (== GrainReport field order)."""

BOUNDARIES_COLUMNS: list[str] = [
    "grain_i", "grain_j", "seed_distance_A", "misorientation_deg",
    "axis_u", "axis_v", "axis_w", "axis_dev_deg", "area_A2",
    "mean_normal_x", "mean_normal_y", "mean_normal_z", "normal_spread_deg",
    "plane_i_hkl", "plane_i_dev_deg", "plane_j_hkl", "plane_j_dev_deg",
    "character", "character_angle_deg", "csl_sigma", "n_overlap_deleted",
]
"""boundaries.csv columns — §8.3 verbatim (BoundaryReport CSV fields; the
raw axis_crystal/mean_normal vectors are intentionally not serialized)."""

GRAINS_COLUMNS_PHASES: list[str] = [*GRAINS_COLUMNS, "phase"]
"""grains.csv columns for MULTIPHASE runs: adds a trailing "phase" column
naming each grain's phase.  Single-phase runs still use GRAINS_COLUMNS, so
their output stays byte-identical to the pre-multiphase format."""

BOUNDARIES_COLUMNS_PHASES: list[str] = [*BOUNDARIES_COLUMNS,
                                        "phase_i", "phase_j"]
"""boundaries.csv columns for MULTIPHASE runs: the single-phase set plus
the two phase-name columns (``character`` is ``interphase`` and the
misorientation columns are NaN/(000) for cross-phase pairs)."""

CURVATURE_BOUNDARY_COLUMNS: list[str] = [
    "H_mean_invA", "H_std_invA", "H_abs_mean_invA",
    "K_mean_invA2", "K_std_invA2", "curv_n_samples",
]
"""Per-boundary GB-curvature columns (analysis.gb_curvature):
area-weighted stats over the boundary's local samples."""

BOUNDARIES_COLUMNS_CURVATURE: list[str] = [
    *BOUNDARIES_COLUMNS, *CURVATURE_BOUNDARY_COLUMNS]
"""boundaries.csv columns when analysis.gb_curvature is on."""

BOUNDARIES_COLUMNS_PHASES_CURVATURE: list[str] = [
    *BOUNDARIES_COLUMNS_PHASES, *CURVATURE_BOUNDARY_COLUMNS]
"""boundaries.csv columns for multiphase + gb_curvature runs (curvature
columns append after the phase columns)."""

GB_CURVATURE_COLUMNS: list[str] = [
    "grain_i", "grain_j", "x", "y", "z", "area_A2", "H_invA", "K_invA2",
]
"""gb_curvature.csv columns — one row per retained local sample (curved)
or per exact face polygon with H = K = 0 (flat).  Sign: H > 0 means
grain_i is locally convex; negate H for grain_j's perspective."""

VERTICES_COLUMNS: list[str] = [
    "grain_id", "v_index", "x", "y", "z", "adjacent_grains",
]
"""vertices.csv columns — §8.3 (flat geometry only, legacy dump.dat parity)."""

SUMMARY_COLUMNS: list[str] = ["section", "key", "value"]
"""summary.csv long-format columns — §8.3."""

DOPING_COLUMNS: list[str] = [
    "grain_id", "element", "mode", "n_candidate_sites",
    "n_rejected_min_distance", "n_dopant", "n_dopant_shell",
    "n_dopant_bulk", "fraction_shell", "fraction_bulk",
]
"""doping.csv columns — one row per (grain, dopant), rows ordered
config-dopant order then ascending grain id."""

DOPING_PROFILE_COLUMNS: list[str] = [
    "element", "mode", "bin_lo", "bin_hi", "bin_center",
    "n_dopant", "n_candidate", "local_fraction",
    "nominal_fraction", "shell_width", "enrichment",
]
"""doping_profile.csv columns — one row per (dopant, GB-distance bin);
local_fraction = n_dopant / n_candidate is the proxigram value."""


def _cell(value: Any) -> Any:
    """CSV cell formatting: numpy scalars coerced to Python scalars first, then
    floats via shortest repr; non-finite floats as the tokens inf/-inf/nan
    (never a numpy repr like ``np.float64(inf)`` / ``np.True_``)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isfinite(value):
            return fmt(value)
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _open_csv(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" per csv-module docs; lineterminator pinned to "\n" so the
    # bytes are platform-independent (byte-identical re-run test, §10).
    return atomic_writer(path, newline="")


def write_grains_csv(reports: list[GrainReport], path: Path,
                     columns: list[str] | None = None) -> None:
    """grains.csv (§8.3).  *columns* defaults to the base set; multiphase
    runs pass GRAINS_COLUMNS_PHASES."""
    cols = GRAINS_COLUMNS if columns is None else columns
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in reports:
            w.writerow([_cell(getattr(r, col)) for col in cols])


def write_boundaries_csv(reports: list[BoundaryReport], path: Path,
                         columns: list[str] | None = None) -> None:
    """boundaries.csv (§8.3).  *columns* defaults to the base set;
    multiphase runs pass BOUNDARIES_COLUMNS_PHASES."""
    cols = BOUNDARIES_COLUMNS if columns is None else columns
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in reports:
            w.writerow([_cell(getattr(r, col)) for col in cols])


def write_gb_curvature_csv(result: CurvatureResult, path: Path) -> None:
    """gb_curvature.csv — local GB curvature samples.

    Rows follow the tess.adjacency() pair order of *result* and the
    deterministic sample order within each pair.
    """
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(GB_CURVATURE_COLUMNS)
        for (i, j), pc in result.pairs.items():
            for m in range(len(pc.H)):
                w.writerow([i, j,
                            fmt(pc.points[m, 0]), fmt(pc.points[m, 1]),
                            fmt(pc.points[m, 2]), fmt(pc.areas[m]),
                            fmt(pc.H[m]), fmt(pc.K[m])])


def write_vertices_csv(
    tess: FlatTessellation,
    box_lengths: np.ndarray,
    path: Path,
) -> None:
    """vertices.csv (§8.3, flat geometry only — legacy dump.dat parity).

    One row per Voronoi vertex of each cell, in absolute lab coordinates.
    ``adjacent_grains`` lists the neighbor ids of every cell face meeting at
    the vertex (semicolon-separated, sorted; -1 marks a box-wall face in
    slab geometry).  Faces and cell vertices come from the same Qhull vertex
    array, so coincidence matching at EULER_TOL·max(L) is exact in practice.
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    tol = EULER_TOL * float(np.max(L))
    seeds = tess.seeds
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(VERTICES_COLUMNS)
        for cell in tess.cells:
            abs_verts = cell.vertices + seeds[cell.grain_id]
            for v_index, v in enumerate(abs_verts):
                adjacent: set[int] = set()
                for face in cell.faces:
                    d = np.linalg.norm(face.vertices - v, axis=1)
                    if np.any(d < tol):
                        adjacent.add(int(face.neighbor_id))
                adj_str = ";".join(str(g) for g in sorted(adjacent))
                w.writerow([cell.grain_id, v_index,
                            fmt(v[0]), fmt(v[1]), fmt(v[2]), adj_str])


def write_summary_csv(
    rows: list[tuple[str, str, Any]],
    path: Path,
) -> None:
    """summary.csv (§8.3): long-format ``section, key, value`` rows.

    The pipeline assembles the rows (box, seed, spglib verification,
    tessellation parameters, atom totals, composition, timings, QA gate
    results, package versions); this function only serializes them.
    Multi-word / comma-containing values are quoted by the csv module
    (RFC 4180).
    """
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(SUMMARY_COLUMNS)
        for section, key, value in rows:
            w.writerow([section, key, _cell(value)])


def summary_mapping_rows(section: str, values: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """Flatten a complete mapping into CSV keys, retaining null and empty values."""
    rows: list[tuple[str, str, Any]] = []

    def append(value: Any, key: str) -> None:
        if isinstance(value, dict) and value:
            for name, item in value.items():
                append(item, f"{key}.{name}" if key else str(name))
        elif isinstance(value, (list, tuple)) and value:
            for index, item in enumerate(value):
                append(item, f"{key}[{index}]")
        else:
            if value is None:
                value = "null"
            elif isinstance(value, dict):
                value = "{}"
            elif isinstance(value, (list, tuple)):
                value = "[]"
            rows.append((section, key, value))

    for name, value in values.items():
        append(value, name)
    return rows


def write_doping_csv(rows: list[DopingReportRow], path: Path) -> None:
    """doping.csv — per-grain dopant placement report."""
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(DOPING_COLUMNS)
        for r in rows:
            w.writerow([_cell(getattr(r, col)) for col in DOPING_COLUMNS])


def write_doping_profile_csv(profiles: list, path: Path,
                             bin_width: float | None = None,
                             n_bins_min: int = 20) -> None:
    """doping_profile.csv — dopant–GB distance profile (proxigram).

    One row per (dopant, distance bin).  Columns:

    element, mode, bin_lo, bin_hi, bin_center : bin geometry (Å,
        distance to the nearest grain boundary).
    n_dopant : placed dopant atoms of this element in the bin.
    n_candidate : candidate sites (interstitial: min-distance survivors;
        substitutional: eligible host atoms) in the bin.
    local_fraction : n_dopant / n_candidate — the local dopant fraction
        on the candidate sublattice at this GB distance.  This is the
        proxigram: for GB segregation with enrichment E it should step
        from ≈ E·p_bulk inside the shell to ≈ p_bulk outside; without
        segregation it is flat at the nominal fraction.  Empty bins
        write nan.
    nominal_fraction, shell_width, enrichment : config references
        (constant per element) so the plot needs no other input.

    Bin width: *bin_width* if given, else shell_width/3 when segregation
    is on (≥ 3 bins resolve the shell step), else max-margin/n_bins_min —
    recorded per element in the shell_width/bin edges themselves (the
    file is self-describing; no silent assumptions).
    """
    import numpy as np

    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(DOPING_PROFILE_COLUMNS)
        for p in profiles:
            cand = np.asarray(p.candidate_margins, dtype=np.float64)
            dop = np.asarray(p.dopant_margins, dtype=np.float64)
            if len(cand) == 0:
                continue
            d_max = float(cand.max())
            if bin_width is not None:
                bw = float(bin_width)
            elif p.shell_width > 0.0:
                bw = p.shell_width / 3.0
            else:
                bw = d_max / n_bins_min if d_max > 0 else 1.0
            n_bins = max(int(np.ceil(d_max / bw)), 1)
            edges = np.arange(n_bins + 1, dtype=np.float64) * bw
            h_dop, _ = np.histogram(dop, bins=edges)
            h_cand, _ = np.histogram(cand, bins=edges)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(h_cand > 0, h_dop / np.maximum(h_cand, 1),
                                float("nan"))
                frac[h_cand == 0] = float("nan")
            for k in range(n_bins):
                w.writerow([
                    p.element, p.mode,
                    _cell(float(edges[k])), _cell(float(edges[k + 1])),
                    _cell(float(0.5 * (edges[k] + edges[k + 1]))),
                    int(h_dop[k]), int(h_cand[k]),
                    _cell(float(frac[k])),
                    _cell(p.nominal_concentration),
                    _cell(p.shell_width), _cell(p.enrichment),
                ])
