"""Pure analysis-page logic for *grainsmith studio*.

This module reads a finished run's recorded artifacts into typed containers and
display tables for the studio's outputs API (:mod:`grainsmith.ui.outputs_json`,
which serialises them to JSON).  It imports **no streamlit and no matplotlib** —
the studio's figures are drawn client-side by the React viewer — so it is pure,
headless, and unit-testable.

SCIENTIFIC-INTEGRITY CONTRACT (binding): the UI is a *viewer*,
never a second computation path.  Everything displayed here is read verbatim
from the engine's recorded outputs — primarily ``microstructure.json``
(the machine-readable record written by ``io/publication.py``), supplemented
by ``statistics.csv`` and ``mdf.csv``.  No statistic is recomputed: the
log-normal fit (mu/sigma/KS), the GB-character fractions and the CSL fractions
are taken as recorded.  The one derived quantity is the equivalent-diameter
*histogram*: the recorded per-grain volumes (``grains[*].volume_A3``) are passed
through raw (see :mod:`grainsmith.ui.outputs_json`) and the browser bins them via
the engine's own published formula d = (6V/pi)^(1/3); the overlaid analytic
curve uses the recorded mu/sigma (it is NOT refit in the engine or the viewer).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical output filenames written by the engine (io/publication.py,
# io/texture.py).  Pinned here so the viewer and the writers cannot drift.
MICROSTRUCTURE_JSON = "microstructure.json"
STATISTICS_CSV = "statistics.csv"
MDF_CSV = "mdf.csv"
GRAINS_CSV = "grains.csv"
VERTICES_CSV = "vertices.csv"

GRAIN_HULL_MAX_GRAINS = 4000
"""Cap on the number of grain polyhedra shipped to the viewer (payload guard).
Beyond this the shapes are omitted and the viewer falls back to spheres."""

# Per-grain fields surfaced to the studio's 3D microstructure view.  A curated
# subset of GRAINS_COLUMNS (io/reports.py), read VERBATIM from the records:
# the seed POSITION, the recorded VOLUME (the view sizes each grain by the
# recorded equivalent diameter d = (6V/pi)^(1/3) — the engine's own published
# formula), and the recorded ORIENTATION (quaternion + Bunge Euler + the
# nearest-lab-axis Miller indices).  NOTHING is recomputed here; the browser
# only maps these to sphere positions / radii / colours.
GRAIN_VIEW_FLOAT_FIELDS: tuple[str, ...] = (
    "seed_x", "seed_y", "seed_z", "volume_A3",
    "q_w", "q_x", "q_y", "q_z",
    "euler_phi1_deg", "euler_Phi_deg", "euler_phi2_deg",
)
GRAIN_VIEW_STR_FIELDS: tuple[str, ...] = ("z_plane_hkl", "x_dir_uvw")


# ---------------------------------------------------------------------------
# Loaded-outputs container
# ---------------------------------------------------------------------------


@dataclass
class GateRow:
    """One QA-gate result, exactly as recorded in microstructure.json."""

    gate: str
    passed: bool
    measured: Any
    message: str


@dataclass
class LoadedOutputs:
    """Everything the analysis page needs, read from one output directory.

    All fields are read verbatim from the recorded artifacts; nothing here
    is recomputed.  Optional artifacts that are absent leave their field at
    the empty/``None`` default (``has_*`` flags say which were found).
    """

    outdir: Path
    title: str = ""
    grainsmith_version: str = ""
    gates: list[GateRow] = field(default_factory=list)
    statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    grain_volumes_A3: list[float] = field(default_factory=list)
    mdf_rows: list[dict[str, float]] = field(default_factory=list)
    # Per-grain records for the 3D microstructure view (recorded subset; see
    # GRAIN_VIEW_*_FIELDS) and the recorded box edge lengths [Lx, Ly, Lz] in Å
    # (from the archived config; None when unavailable — the viewer then frames
    # the scene from the seed extents).
    grains: list[dict[str, Any]] = field(default_factory=list)
    box_size: list[float] | None = None
    # Per-grain cell polyhedra for the 3D view: the convex hull of each grain's
    # RECORDED vertices (vertices.csv).  Only written for the flat / power
    # (Laguerre) tessellations, whose cells are convex BY CONSTRUCTION, so the
    # hull reproduces the recorded cell exactly — no tessellation is recomputed
    # and no shape is invented.  Empty for warped / voxel-imported runs (which
    # do not record vertices and whose cells are not convex); the viewer then
    # falls back to equivalent-volume spheres.
    grain_hulls: list[dict[str, Any]] = field(default_factory=list)
    has_microstructure_json: bool = False
    has_statistics_csv: bool = False
    has_mdf_csv: bool = False
    has_grains_csv: bool = False
    has_vertices_csv: bool = False

    # -- convenience accessors (read-only views of recorded numbers) --

    def stat(self, section: str, key: str) -> Any:
        """Recorded statistics value, or ``None`` when section/key absent."""
        return self.statistics.get(section, {}).get(key)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def _read_statistics_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the long-format ``section,key,value`` statistics.csv back into the
    nested ``{section: {key: value}}`` dict (values parsed to float when they
    look numeric — only used as a *fallback* for microstructure.json)."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            section = row["section"]
            key = row["key"]
            out.setdefault(section, {})[key] = _maybe_float(row["value"])
    return out


def _read_mdf_csv(path: Path) -> list[dict[str, float]]:
    """Parse mdf.csv rows (already-recorded per-degree densities) to floats.

    ``target_density`` is blank when no MDF target was configured; such cells
    are recorded as ``nan`` so callers can detect "column absent".
    """
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: _maybe_float(v) for k, v in row.items()})
    return rows


def _read_grain_volumes(path: Path) -> list[float]:
    """Recorded per-grain ``volume_A3`` column from grains.csv (fallback when
    microstructure.json is absent)."""
    vols: list[float] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = _maybe_float(row.get("volume_A3", ""))
            if isinstance(v, float) and math.isfinite(v):
                vols.append(v)
    return vols


def _maybe_float(value: str) -> Any:
    """Parse a CSV cell to float when possible; '' -> nan; else the raw str."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def _as_float(value: Any) -> float:
    """Coerce a recorded value (JSON number / CSV string / None) to float;
    absent or unparseable -> nan (the 'non-finite means absent' convention)."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _grain_view_record(src: dict[str, Any]) -> dict[str, Any]:
    """Project one recorded grain (microstructure.json grain dict or grains.csv
    row) onto the curated 3D-view subset — verbatim, no recomputation."""
    gid = src.get("grain_id")
    rec: dict[str, Any] = {
        "grain_id": None if gid is None or gid == "" else int(float(gid))
    }
    for key in GRAIN_VIEW_FLOAT_FIELDS:
        rec[key] = _as_float(src.get(key))
    for key in GRAIN_VIEW_STR_FIELDS:
        val = src.get(key)
        rec[key] = None if val is None or val == "" else str(val)
    return rec


def _read_grain_view_csv(path: Path) -> list[dict[str, Any]]:
    """Per-grain 3D-view records from grains.csv (fallback when
    microstructure.json is absent)."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [_grain_view_record(row) for row in csv.DictReader(fh)]


def _read_grain_hulls(
    path: Path, max_grains: int = GRAIN_HULL_MAX_GRAINS
) -> list[dict[str, Any]]:
    """Per-grain cell polyhedra from the RECORDED ``vertices.csv``.

    Each row of ``vertices.csv`` is one cell vertex in absolute coordinates
    (``grain_id,v_index,x,y,z,adjacent_grains``).  The engine writes this file
    **only** for the flat and power (Laguerre) tessellations, whose Voronoi /
    Laguerre cells are convex by construction — so the convex hull of a grain's
    recorded vertices IS that grain's cell, exactly.  We therefore triangulate
    the hull for rendering rather than reconstructing any geometry: no
    tessellation, face or vertex is recomputed.

    Returns ``[{grain_id, vertices: [[x,y,z], ...], faces: [[i,j,k], ...]}]``
    with face indices into that grain's own ``vertices`` list.  Grains with
    fewer than four vertices, or whose points are degenerate (coplanar), are
    skipped — the viewer falls back to a sphere for those.
    """
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    by_grain: dict[int, list[tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            gid_raw = row.get("grain_id", "")
            if gid_raw in (None, ""):
                continue
            try:
                gid = int(float(gid_raw))
                pt = (float(row["x"]), float(row["y"]), float(row["z"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(c) for c in pt):
                continue
            by_grain.setdefault(gid, []).append(pt)

    hulls: list[dict[str, Any]] = []
    for gid in sorted(by_grain)[:max_grains]:
        pts = np.asarray(by_grain[gid], dtype=float)
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except (QhullError, ValueError):
            continue  # degenerate / coplanar cell → sphere fallback
        keep = hull.vertices
        remap = {int(orig): i for i, orig in enumerate(keep)}
        hulls.append(
            {
                "grain_id": gid,
                "vertices": pts[keep].tolist(),
                "faces": [[remap[int(i)] for i in tri] for tri in hull.simplices],
            }
        )
    return hulls


def load_outputs(outdir: Path) -> LoadedOutputs:
    """Read every recorded artifact present in *outdir* into a LoadedOutputs.

    ``microstructure.json`` is the primary source (gates + statistics + the
    per-grain volume records).  ``statistics.csv`` and ``grains.csv`` are read
    only to *supplement* a run whose JSON is missing.  ``mdf.csv`` is optional
    and handled gracefully when absent.

    Raises ``FileNotFoundError`` when *outdir* contains none of the recognised
    artifacts (so the page can show a clear "not a grainsmith output dir"
    message rather than a blank panel).
    """
    outdir = Path(outdir)
    loaded = LoadedOutputs(outdir=outdir)

    micro = outdir / MICROSTRUCTURE_JSON
    if micro.is_file():
        doc = _read_json(micro)
        loaded.has_microstructure_json = True
        prov = doc.get("provenance", {})
        loaded.title = str(prov.get("title", ""))
        loaded.grainsmith_version = str(prov.get("grainsmith_version", ""))
        loaded.gates = [
            GateRow(
                gate=str(g.get("gate", "")),
                passed=bool(g.get("passed", False)),
                measured=g.get("measured"),
                message=str(g.get("message", "")),
            )
            for g in doc.get("gates", [])
        ]
        loaded.statistics = doc.get("statistics", {}) or {}
        loaded.grain_volumes_A3 = [
            float(g["volume_A3"])
            for g in doc.get("grains", [])
            if g.get("volume_A3") is not None
        ]
        loaded.grains = [_grain_view_record(g) for g in doc.get("grains", [])]
        # Recorded box edge lengths [Lx, Ly, Lz] from the archived config.
        box = (doc.get("config") or {}).get("box") or {}
        lengths = box.get("lengths")
        if isinstance(lengths, list | tuple) and len(lengths) == 3:
            loaded.box_size = [float(x) for x in lengths]

    stats_csv = outdir / STATISTICS_CSV
    if stats_csv.is_file():
        loaded.has_statistics_csv = True
        if not loaded.statistics:  # supplement only — JSON wins
            loaded.statistics = _read_statistics_csv(stats_csv)

    grains_csv = outdir / GRAINS_CSV
    if grains_csv.is_file():
        loaded.has_grains_csv = True
        if not loaded.grain_volumes_A3:  # supplement only
            loaded.grain_volumes_A3 = _read_grain_volumes(grains_csv)
        if not loaded.grains:  # supplement only — JSON wins
            loaded.grains = _read_grain_view_csv(grains_csv)

    # Cell polyhedra (flat / power runs only — see LoadedOutputs.grain_hulls).
    vertices_csv = outdir / VERTICES_CSV
    if vertices_csv.is_file():
        loaded.has_vertices_csv = True
        try:
            loaded.grain_hulls = _read_grain_hulls(vertices_csv)
        except (OSError, ValueError, ImportError):
            loaded.grain_hulls = []  # shapes are optional; spheres still render

    mdf_csv = outdir / MDF_CSV
    if mdf_csv.is_file():
        loaded.has_mdf_csv = True
        loaded.mdf_rows = _read_mdf_csv(mdf_csv)

    if not (
        loaded.has_microstructure_json
        or loaded.has_statistics_csv
        or loaded.has_grains_csv
    ):
        raise FileNotFoundError(
            f"No grainsmith outputs found in {outdir} "
            f"(looked for {MICROSTRUCTURE_JSON}, {STATISTICS_CSV}, "
            f"{GRAINS_CSV})."
        )
    return loaded


# ---------------------------------------------------------------------------
# Tables for the page (recorded values only)
# ---------------------------------------------------------------------------


def gate_table(loaded: LoadedOutputs) -> list[dict[str, Any]]:
    """Recorded QA-gate results as plain rows (one dict per gate)."""
    return [
        {
            "gate": g.gate,
            "status": "PASS" if g.passed else "FAIL",
            "measured": g.measured,
            "message": g.message,
        }
        for g in loaded.gates
    ]


def all_gates_passed(loaded: LoadedOutputs) -> bool:
    """True iff every recorded gate passed (and at least one was recorded)."""
    return bool(loaded.gates) and all(g.passed for g in loaded.gates)


# Sections surfaced as the "key statistics" table: scalar entries only.
KEY_STAT_SECTIONS: tuple[str, ...] = (
    "grain_size",
    "sphericity",
    "boundaries",
    "triple_junctions",
    "mdf",
)


def key_stats_table(loaded: LoadedOutputs) -> list[dict[str, Any]]:
    """Flatten the headline recorded statistics sections to display rows.

    Every value is read verbatim from ``loaded.statistics`` (which came from
    microstructure.json / statistics.csv) — nothing is recomputed here.
    """
    rows: list[dict[str, Any]] = []
    for section in KEY_STAT_SECTIONS:
        entries = loaded.statistics.get(section)
        if not entries:
            continue
        for key, value in entries.items():
            rows.append({"section": section, "key": key, "value": value})
    return rows
