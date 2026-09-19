"""JSON serialization of a finished run's recorded outputs.

Pure logic — **no FastAPI, no Streamlit, no recomputation**.  This module turns
the :class:`grainsmith.ui.analysis.LoadedOutputs` container (read verbatim from
a run's recorded artifacts by :func:`grainsmith.ui.analysis.load_outputs`) into
a plain, JSON-safe ``dict`` so a non-Python client — the React viewer of
"grainsmith studio" — can render the analysis page without re-implementing any
engine knowledge.

It is the **serialization seam** for the outputs side: this is the
``GET /api/outputs`` body (served by :mod:`grainsmith.ui.server`), and the
server layer only has to ``json.dumps`` it.

SCIENTIFIC-INTEGRITY CONTRACT (binding — see also the
:mod:`grainsmith.ui.analysis` docstring): the UI is a *viewer*, never a second
computation path.  Every value emitted here is read verbatim from ``loaded``
(which came from ``microstructure.json`` / ``statistics.csv`` / ``mdf.csv`` /
``grains.csv``) by reusing the existing table builders — ``gate_table``,
``key_stats_table``, ``all_gates_passed`` — and the recorded fields directly.
**No statistic is recomputed here.**  The ONE derived-from-records quantity
downstream — the grain equivalent-diameter *histogram* — is intentionally NOT
built in this module: we pass the recorded per-grain ``grain_volumes_A3``
through raw, and the React viewer bins them client-side.  The only
transformation applied is shape, not value: a recursive sanitizer maps
non-finite floats (NaN / Inf), which JSON cannot represent, to ``null`` —
the "non-finite means absent" convention for blank recorded cells.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from grainsmith.ui import analysis
from grainsmith.ui.analysis import (
    LoadedOutputs,
    all_gates_passed,
    gate_table,
    key_stats_table,
)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce *obj* into a value ``json.dumps(..., allow_nan=False)``
    accepts.

    The only substantive change is on floats: ``NaN`` / ``Inf`` / ``-Inf`` have
    no JSON representation (stdlib ``json`` would otherwise emit the invalid
    tokens ``NaN`` / ``Infinity``), so they degrade to ``None`` — exactly the
    "non-finite means absent" convention for blank recorded cells
    (``_maybe_float`` yields ``float('nan')`` for an empty target column).
    Dicts and lists/tuples are walked; every other value is
    passed through unchanged so recorded scalars are never altered.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(value) for value in obj]
    return obj


def outputs_to_json(loaded: LoadedOutputs) -> dict[str, Any]:
    """Serialize *loaded* recorded outputs to a JSON-safe ``dict``.

    The returned mapping is the future ``GET /api/outputs`` payload::

        {
            "outdir": <str>,
            "title": <str>,
            "grainsmith_version": <str>,
            "gates": [ {"gate", "status", "measured", "message"}, ... ],
            "all_gates_passed": <bool>,
            "key_stats": [ {"section", "key", "value"}, ... ],
            "statistics": { <section>: { <key>: <value> }, ... },
            "grain_volumes_A3": [ <float>, ... ],
            "mdf_rows": [ { <column>: <float | None> }, ... ],
            "grains": [ { "grain_id", "seed_x/y/z", "volume_A3",
                          "q_w/x/y/z", "euler_*", "z_plane_hkl",
                          "x_dir_uvw" }, ... ],
            "box_size": [ <float>, <float>, <float> ] | None,
            "present": {
                "microstructure_json": <bool>,
                "statistics_csv": <bool>,
                "mdf_csv": <bool>,
                "grains_csv": <bool>,
            },
        }

    Every value is read verbatim from *loaded* via the existing
    :mod:`grainsmith.ui.analysis` table builders and recorded fields — nothing
    is recomputed.  ``grain_volumes_A3`` is passed through raw so the browser
    can derive the equivalent-diameter histogram (the engine's own
    ``d = (6V/pi)^(1/3)`` formula) from the records.  The whole dict is run
    through :func:`_json_safe`, so ``json.dumps(..., allow_nan=False)`` on the
    result is guaranteed to succeed.
    """
    payload: dict[str, Any] = {
        "outdir": str(loaded.outdir),
        "title": loaded.title,
        "grainsmith_version": loaded.grainsmith_version,
        "gates": gate_table(loaded),
        "all_gates_passed": all_gates_passed(loaded),
        "key_stats": key_stats_table(loaded),
        "statistics": loaded.statistics,
        "grain_volumes_A3": loaded.grain_volumes_A3,
        "mdf_rows": loaded.mdf_rows,
        # Per-grain records + recorded box for the 3D microstructure view. Like
        # grain_volumes_A3, these are recorded values passed through raw (the
        # browser maps them to sphere positions / radii / orientation colours);
        # nothing is recomputed. _json_safe turns any non-finite into null.
        "grains": loaded.grains,
        "box_size": loaded.box_size,
        # Cell polyhedra: the convex hull of each grain's RECORDED vertices
        # (vertices.csv). Written only for flat / power tessellations, whose
        # cells are convex by construction, so the hull IS the recorded cell —
        # nothing is reconstructed. Empty for warped / voxel runs; the viewer
        # then draws equivalent-volume spheres instead.
        "grain_hulls": loaded.grain_hulls,
        "present": {
            "microstructure_json": loaded.has_microstructure_json,
            "statistics_csv": loaded.has_statistics_csv,
            "mdf_csv": loaded.has_mdf_csv,
            "grains_csv": loaded.has_grains_csv,
            "vertices_csv": loaded.has_vertices_csv,
        },
    }
    return _json_safe(payload)


def load_outputs_json(outdir: str | Path) -> dict[str, Any]:
    """Load the recorded outputs from *outdir* and serialize them to JSON.

    Convenience wrapper around
    ``outputs_to_json(analysis.load_outputs(Path(outdir)))``; raises the same
    ``FileNotFoundError`` as :func:`grainsmith.ui.analysis.load_outputs` when
    *outdir* holds none of the recognised artifacts.
    """
    return outputs_to_json(analysis.load_outputs(Path(outdir)))
