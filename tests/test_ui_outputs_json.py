"""Tests for the outputs serialization seam (grainsmith.ui.outputs_json).

Strategy mirrors tests/test_ui_analysis.py: generate a TINY real run into
``tmp_path`` (the §10 minimal-config FCC Cu pattern, csl=True so the CSL section
is recorded), serialize its recorded outputs via the pure ``outputs_json``
module, and prove the serializer changes *shape* but NEVER a *value* — what the
future ``GET /api/outputs`` body carries equals what the engine RECORDED in
microstructure.json.  The UI never recomputes.

These tests are pure logic — ``outputs_json`` and ``analysis`` import no
streamlit and no matplotlib (the React viewer draws figures client-side), so
no optional UI dependency is required to run them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.pipeline import run
from grainsmith.ui import analysis as core
from grainsmith.ui.analysis import GateRow, LoadedOutputs
from grainsmith.ui.outputs_json import (
    load_outputs_json,
    outputs_to_json,
)

SEED = 20260614


def _tiny_config(outdir: Path):
    """Smallest fast polycrystal: 4 grains, 40^3 A, FCC Cu, CSL on (cubic).

    A small polycrystal (not a single crystal) so the GB-character, CSL and
    MDF artifacts are all populated.
    """
    raw = {
        "meta": {"title": "ui-outputs-json-fixture", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "analysis": {"gb_character": True, "csl": True},
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    return resolve_config(raw)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    outdir = tmp_path_factory.mktemp("ui_outputs_json")
    res = run(_tiny_config(outdir))
    assert res.gates.all_passed()
    return Path(res.outdir)


@pytest.fixture(scope="module")
def recorded(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "microstructure.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Serializability: the whole payload survives strict json.dumps
# ---------------------------------------------------------------------------


def test_outputs_json_serializable(run_dir: Path):
    """The payload round-trips through strict (no-NaN) JSON without raising."""
    payload = load_outputs_json(run_dir)
    text = json.dumps(payload, allow_nan=False)
    # round-trips back to an equal structure
    assert json.loads(text) == payload
    # top-level shape is the documented GET /api/outputs body
    assert set(payload) == {
        "outdir", "title", "grainsmith_version", "gates",
        "all_gates_passed", "key_stats", "statistics", "grain_volumes_A3",
        "mdf_rows", "grains", "box_size", "grain_hulls", "present",
    }


# ---------------------------------------------------------------------------
# THE PIN: serialized == recorded (shape changes, values never do)
# ---------------------------------------------------------------------------


def test_outputs_json_equals_recorded(run_dir: Path, recorded: dict):
    loaded = core.load_outputs(run_dir)
    payload = outputs_to_json(loaded)

    # provenance scalars verbatim
    assert payload["outdir"] == str(loaded.outdir)
    assert payload["title"] == recorded["provenance"]["title"]
    assert payload["grainsmith_version"] == \
        recorded["provenance"]["grainsmith_version"]

    # gates: the serialized table mirrors the existing gate_table builder,
    # and its PASS/FAIL status mirrors each recorded gate's passed flag.
    assert payload["gates"] == core.gate_table(loaded)
    rec_gates = recorded["gates"]
    assert [r["gate"] for r in payload["gates"]] == \
        [g["gate"] for g in rec_gates]
    assert all(
        (r["status"] == "PASS") == g["passed"]
        for r, g in zip(payload["gates"], rec_gates, strict=True)
    )
    assert payload["all_gates_passed"] == core.all_gates_passed(loaded)
    assert payload["all_gates_passed"] is True  # fixture run passes

    # statistics section: recorded verbatim (finite values are untouched)
    assert payload["statistics"] == loaded.statistics
    for section, entries in recorded["statistics"].items():
        for key, value in entries.items():
            if isinstance(value, float) and not math.isfinite(value):
                continue  # non-finite -> null is asserted elsewhere
            assert payload["statistics"][section][key] == value

    # key_stats table is the existing builder, verbatim
    assert payload["key_stats"] == core.key_stats_table(loaded)

    # raw per-grain volumes pass through unchanged (browser bins these)
    rec_vols = [g["volume_A3"] for g in recorded["grains"]]
    assert payload["grain_volumes_A3"] == rec_vols
    assert payload["grain_volumes_A3"] == loaded.grain_volumes_A3

    # per-grain 3D-view records: recorded subset, verbatim (one per grain)
    assert len(payload["grains"]) == len(recorded["grains"])
    for shown, rec in zip(payload["grains"], recorded["grains"], strict=True):
        assert shown["grain_id"] == rec["grain_id"]
        for key in ("seed_x", "seed_y", "seed_z", "volume_A3",
                    "q_w", "q_x", "q_y", "q_z"):
            assert shown[key] == rec[key]
        # Miller-index strings (e.g. "(0 0 1)") pass through unchanged
        assert shown["z_plane_hkl"] == rec["z_plane_hkl"]
        assert shown["x_dir_uvw"] == rec["x_dir_uvw"]

    # recorded box edge lengths, verbatim from the archived config
    assert payload["box_size"] == recorded["config"]["box"]["lengths"]
    assert payload["box_size"] == [40.0, 40.0, 40.0]

    # mdf rows pass through (finite values unchanged)
    assert len(payload["mdf_rows"]) == len(loaded.mdf_rows)
    for shown, rec in zip(payload["mdf_rows"], loaded.mdf_rows, strict=True):
        for key, value in rec.items():
            if isinstance(value, float) and not math.isfinite(value):
                assert shown[key] is None
            else:
                assert shown[key] == value


# ---------------------------------------------------------------------------
# NaN -> null (JSON has no NaN; the "non-finite means absent" convention)
# ---------------------------------------------------------------------------


def test_outputs_json_nan_to_null():
    """A blank MDF target cell (nan) serializes to null/None, and the result
    is strict-JSON serializable."""
    loaded = LoadedOutputs(
        outdir=Path("/tmp/does-not-exist"),
        title="nan-fixture",
        grainsmith_version="1.1.0",
        gates=[GateRow(gate="g", passed=True, measured=1.0, message="ok")],
        statistics={"grain_size": {"mean_d_eq_A": float("nan"),
                                   "count": 4.0}},
        grain_volumes_A3=[10.0, 20.0],
        mdf_rows=[
            {"bin_center_deg": 5.0, "area_weighted_density": 0.1,
             "target_density": float("nan")},
        ],
        has_microstructure_json=True,
    )
    payload = outputs_to_json(loaded)

    # the blank target cell became None
    assert payload["mdf_rows"][0]["target_density"] is None
    # a finite neighbour in the same row is untouched
    assert payload["mdf_rows"][0]["area_weighted_density"] == 0.1
    # a nan inside a recorded statistics section also became None
    assert payload["statistics"]["grain_size"]["mean_d_eq_A"] is None
    assert payload["statistics"]["grain_size"]["count"] == 4.0
    # strict JSON now succeeds (would raise on a raw NaN)
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# present flags reflect which artifacts exist
# ---------------------------------------------------------------------------


def test_present_flags(run_dir: Path, tmp_path: Path):
    """The full fixture has every artifact; a copy without mdf.csv flips that
    one flag (the panel-hiding signal) while the rest stay true."""
    payload = load_outputs_json(run_dir)
    assert payload["present"] == {
        "microstructure_json": True,
        "statistics_csv": True,
        "mdf_csv": True,
        "grains_csv": True,
        # vertices.csv is written only by the flat/power tessellations; it gates
        # the recorded cell polyhedra (`grain_hulls`) in the 3D view.
        "vertices_csv": (run_dir / "vertices.csv").is_file(),
    }

    import shutil

    dst = tmp_path / "no_mdf"
    dst.mkdir()
    for src in run_dir.iterdir():
        if src.is_file() and src.name != "mdf.csv":
            shutil.copy2(src, dst / src.name)
    assert not (dst / "mdf.csv").exists()

    payload_no_mdf = load_outputs_json(dst)
    assert payload_no_mdf["present"]["mdf_csv"] is False
    assert payload_no_mdf["present"]["microstructure_json"] is True
    assert payload_no_mdf["present"]["statistics_csv"] is True
    assert payload_no_mdf["present"]["grains_csv"] is True
    assert payload_no_mdf["mdf_rows"] == []


# ---------------------------------------------------------------------------
# Cell polyhedra: shipped shapes ARE the recorded cells (never a reconstruction)
# ---------------------------------------------------------------------------


def test_grain_hulls_reproduce_the_recorded_cell_volumes(run_dir: Path):
    """The 3D view's grain shapes are the RECORDED cells, not a re-tessellation.

    ``vertices.csv`` is written only for the flat / power (Laguerre) backends,
    whose cells are convex BY CONSTRUCTION.  The convex hull of a grain's
    recorded vertices is therefore that exact cell — proven here by checking the
    hull volume against the independently recorded ``volume_A3``.
    """
    numpy = pytest.importorskip("numpy")
    spatial = pytest.importorskip("scipy.spatial")

    payload = load_outputs_json(run_dir)
    assert payload["present"]["vertices_csv"] is True
    hulls = {h["grain_id"]: h for h in payload["grain_hulls"]}
    assert hulls, "a flat run records cell vertices, so it must ship polyhedra"

    recorded = {g["grain_id"]: g["volume_A3"] for g in payload["grains"]}
    for gid, hull in hulls.items():
        verts = numpy.asarray(hull["vertices"], dtype=float)
        assert verts.shape[1] == 3
        vol = spatial.ConvexHull(verts).volume
        assert vol == pytest.approx(recorded[gid], rel=1e-9)
        # Faces are triangles indexing into that grain's OWN vertex list.
        n = len(hull["vertices"])
        for face in hull["faces"]:
            assert len(face) == 3
            assert all(0 <= i < n for i in face)


def test_no_grain_hulls_when_vertices_were_not_recorded(run_dir: Path, tmp_path: Path):
    """Warped / voxel-imported runs record no vertices (and their cells are not
    convex), so no shape is invented — the viewer falls back to spheres."""
    import shutil

    dst = tmp_path / "no_vertices"
    dst.mkdir()
    for src in run_dir.iterdir():
        if src.is_file() and src.name != "vertices.csv":
            shutil.copy2(src, dst / src.name)

    payload = load_outputs_json(dst)
    assert payload["present"]["vertices_csv"] is False
    assert payload["grain_hulls"] == []
