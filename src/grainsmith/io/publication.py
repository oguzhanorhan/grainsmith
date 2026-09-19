"""Publication-grade output writers.

statistics.csv      — long-format ``section,key,value`` rows from the
                      analysis/statistics.py dict (same RFC 4180 / no
                      provenance-comment policy as every CSV, §8.3).
slice_<ax><pos>.csv — EBSD-like 2D section: in-plane coordinates, grain
                      id, Bunge Euler angles (+ phase when multiphase).
microstructure.json — the machine-readable record of the whole run:
                      config echo, provenance, versions, gates, per-grain
                      and per-boundary records (the pinned CSV column
                      sets), statistics — dataset-repository (Zenodo)
                      ready.  Strict JSON: NaN/±inf are serialized as
                      ``null`` (RFC 8259 has no NaN).  Wall-clock stage
                      timings are deliberately NOT included; the run's UTC
                      timestamp IS (``provenance.timestamp_utc``), so two
                      re-runs differ in that one field unless
                      ``SOURCE_DATE_EPOCH`` is set (which freezes it and makes
                      the file byte-identical).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from grainsmith.constants import SECTION_GRID
from grainsmith.io.common import Provenance, atomic_writer, fmt
from grainsmith.io.reports import (
    BOUNDARIES_COLUMNS,
    BOUNDARIES_COLUMNS_CURVATURE,
    BOUNDARIES_COLUMNS_PHASES,
    BOUNDARIES_COLUMNS_PHASES_CURVATURE,
    GRAINS_COLUMNS,
    GRAINS_COLUMNS_PHASES,
    _cell,
    _open_csv,
)

STATISTICS_COLUMNS: list[str] = ["section", "key", "value"]
"""statistics.csv long-format columns."""

SECTION_COLUMNS: list[str] = [
    "x", "y", "grain_id", "phi1_deg", "Phi_deg", "phi2_deg",
]
"""slice_<axis><position>.csv columns.  ``x``/``y`` are the
two IN-PLANE coordinates in box-axis order (a z-section maps them to lab
x/y; an x-section to lab y/z)."""

SECTION_COLUMNS_PHASES: list[str] = [*SECTION_COLUMNS, "phase"]
"""Section columns for multiphase runs: phase columns are appended only
when phases are active."""

MICROSTRUCTURE_SCHEMA: str = "grainsmith/microstructure/v1"
"""Schema identifier embedded in microstructure.json (top-level key
``schema``) so downstream tooling can dispatch on the layout."""


def write_statistics_csv(
    statistics: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """statistics.csv: flatten the ordered section dicts to long-format
    ``section,key,value`` rows (insertion order — deterministic)."""
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(STATISTICS_COLUMNS)
        for section, entries in statistics.items():
            for key, value in entries.items():
                w.writerow([section, key, _cell(value)])


def write_section_csv(
    path: Path,
    tess,
    grain_reports: list,
    box_lengths: np.ndarray,
    axis: str,
    position: float,
    multiphase: bool = False,
) -> None:
    """EBSD-like section: sample a SECTION_GRID-resolution
    pixel grid on the plane ``axis = position·L_axis`` and report, per
    pixel, the in-plane coordinates (pixel centers), the grain id and the
    grain's Bunge Euler angles from *grain_reports* (+ phase for multiphase
    runs).  Rows
    run row-major in in-plane axis order (second axis fastest)."""
    L = np.asarray(box_lengths, dtype=np.float64)
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    u_ax, v_ax = (k for k in range(3) if k != ax)

    h = float(min(L[u_ax], L[v_ax])) / SECTION_GRID
    n_u = int(np.ceil(L[u_ax] / h))
    n_v = int(np.ceil(L[v_ax] / h))
    u = (np.arange(n_u) + 0.5) * (L[u_ax] / n_u)
    v = (np.arange(n_v) + 0.5) * (L[v_ax] / n_v)

    pts = np.empty((n_u * n_v, 3), dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    pts[:, u_ax] = uu.ravel()
    pts[:, v_ax] = vv.ravel()
    pts[:, ax] = position * L[ax]
    gid = tess.grain_of(pts)

    phi1 = np.array([r.euler_phi1_deg for r in grain_reports])
    Phi = np.array([r.euler_Phi_deg for r in grain_reports])
    phi2 = np.array([r.euler_phi2_deg for r in grain_reports])
    phases = [r.phase for r in grain_reports]

    cols = SECTION_COLUMNS_PHASES if multiphase else SECTION_COLUMNS
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for k, g in enumerate(gid):
            g = int(g)
            row = [fmt(pts[k, u_ax]), fmt(pts[k, v_ax]), g,
                   fmt(float(phi1[g])), fmt(float(Phi[g])),
                   fmt(float(phi2[g]))]
            if multiphase:
                row.append(phases[g])
            w.writerow(row)


def _jsonable(value: Any) -> Any:
    """Recursive strict-JSON sanitizer: numpy scalars → Python, NaN/±inf →
    None (RFC 8259), containers handled element-wise."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        # A bare ndarray would survive to json.dump(allow_nan=False) and raise
        # "not JSON serializable"; recurse element-wise.
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_microstructure_json(
    path: Path,
    config,
    provenance: Provenance,
    gates,
    grain_reports: list,
    boundary_reports: list,
    statistics: dict[str, dict[str, Any]],
    multiphase: bool = False,
    curvature: bool = False,
    environment: dict[str, str] | None = None,
) -> None:
    """microstructure.json — see the module docstring.

    Per-grain/per-boundary records carry exactly the pinned CSV column
    sets (GRAINS_COLUMNS[_PHASES] / BOUNDARIES_COLUMNS[_PHASES][_CURVATURE]),
    so the JSON and the CSVs cannot drift apart.

    ``environment`` (§R3): the BLAS/OS/numba/jobs record behind this run's
    floating-point bytes. The pipeline builds it once and passes the same
    dict here and into summary.csv's ``environment`` section so the two
    files can never disagree; a direct caller that omits it gets
    ``environment_record()`` computed fresh (no jobs values -> "unknown"),
    so this function still produces a well-formed document on its own."""
    import scipy
    import spglib

    import grainsmith
    from grainsmith.provenance import PROVENANCE_PAYLOAD_SCHEMA, environment_record

    if environment is None:
        environment = environment_record()

    gcols = GRAINS_COLUMNS_PHASES if multiphase else GRAINS_COLUMNS
    if multiphase and curvature:
        bcols = BOUNDARIES_COLUMNS_PHASES_CURVATURE
    elif multiphase:
        bcols = BOUNDARIES_COLUMNS_PHASES
    elif curvature:
        bcols = BOUNDARIES_COLUMNS_CURVATURE
    else:
        bcols = BOUNDARIES_COLUMNS
    doc = {
        "schema": MICROSTRUCTURE_SCHEMA,
        "provenance": {
            "grainsmith_version": provenance.version,
            "timestamp_utc": provenance.timestamp_iso,
            "seed": provenance.seed,
            "config_sha256": provenance.config_sha256,
            "provenance_sha256": provenance.provenance_sha256,
            "provenance_payload_schema": PROVENANCE_PAYLOAD_SCHEMA,
            "title": provenance.title,
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "spglib": spglib.__version__,
            "grainsmith": grainsmith.__version__,
        },
        "environment": environment,
        "config": config.model_dump(mode="json"),
        "gates": [
            {"gate": r.gate, "passed": r.passed, "measured": r.measured,
             "message": r.message}
            for r in gates.results()
        ],
        "grains": [
            {col: getattr(r, col) for col in gcols} for r in grain_reports
        ],
        "boundaries": [
            {col: getattr(r, col) for col in bcols}
            for r in boundary_reports
        ],
        "statistics": statistics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as fh:
        json.dump(_jsonable(doc), fh, indent=2, allow_nan=False)
        fh.write("\n")
