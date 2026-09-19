"""Tests for the analysis viewer (grainsmith.ui.analysis).

Strategy: generate a TINY real run into ``tmp_path`` (the §10 minimal-config
FCC Cu pattern, with csl=True so the CSL section is recorded), then load it via
the pure viewer and prove that what the viewer would DISPLAY equals what the
engine RECORDED in microstructure.json — the UI never recomputes.

These tests are pure logic (no streamlit, no matplotlib): the viewer reads the
recorded artifacts and we prove what it would DISPLAY equals what the engine
RECORDED.  The studio's figures are drawn client-side by the React viewer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.pipeline import run
from grainsmith.ui import analysis as core

SEED = 20260614


def _tiny_config(outdir: Path):
    """Smallest fast polycrystal: 4 grains, 40^3 A, FCC Cu, CSL on (cubic).

    A small polycrystal (not a single crystal) so the GB-character, CSL and
    MDF artifacts are all populated.
    """
    raw = {
        "meta": {"title": "ui-analysis-fixture", "verbose": 0},
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
    outdir = tmp_path_factory.mktemp("ui_out")
    res = run(_tiny_config(outdir))
    assert res.gates.all_passed()
    return Path(res.outdir)


@pytest.fixture(scope="module")
def recorded(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "microstructure.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# load_outputs: structure + presence
# ---------------------------------------------------------------------------


def test_load_outputs_finds_artifacts(run_dir: Path):
    loaded = core.load_outputs(run_dir)
    assert loaded.has_microstructure_json
    assert loaded.has_statistics_csv
    assert loaded.has_grains_csv
    assert loaded.has_mdf_csv
    assert loaded.gates                      # gates were recorded
    assert loaded.grain_volumes_A3           # per-grain volumes were recorded


def test_load_outputs_raises_on_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        core.load_outputs(tmp_path)


# ---------------------------------------------------------------------------
# DISPLAYED == RECORDED (no recomputation / no divergence)
# ---------------------------------------------------------------------------


def test_gates_equal_recorded(run_dir: Path, recorded: dict):
    loaded = core.load_outputs(run_dir)
    rec_gates = recorded["gates"]
    assert len(loaded.gates) == len(rec_gates)
    for shown, rec in zip(loaded.gates, rec_gates, strict=True):
        assert shown.gate == rec["gate"]
        assert shown.passed == rec["passed"]
        assert shown.measured == rec["measured"]
        assert shown.message == rec["message"]
    # the gate_table the page renders mirrors the recorded pass/fail too
    table = core.gate_table(loaded)
    assert [r["gate"] for r in table] == [g["gate"] for g in rec_gates]
    assert all(
        (r["status"] == "PASS") == g["passed"]
        for r, g in zip(table, rec_gates, strict=True)
    )


def test_lognormal_fit_displayed_equals_recorded(run_dir: Path,
                                                 recorded: dict):
    """The mu/sigma/KS shown for the size histogram are taken verbatim from
    the recorded grain_size section — never refit."""
    loaded = core.load_outputs(run_dir)
    rec = recorded["statistics"]["grain_size"]
    assert loaded.stat("grain_size", "lognormal_mu_hat") == \
        rec["lognormal_mu_hat"]
    assert loaded.stat("grain_size", "lognormal_sigma_hat") == \
        rec["lognormal_sigma_hat"]
    assert loaded.stat("grain_size", "lognormal_ks_statistic") == \
        rec["lognormal_ks_statistic"]


def test_csl_fraction_displayed_equals_recorded(run_dir: Path,
                                                recorded: dict):
    """At least one CSL area fraction is recorded; the viewer surfaces the
    exact recorded value (proving no re-derivation of CSL fractions)."""
    loaded = core.load_outputs(run_dir)
    rec_csl = recorded["statistics"]["csl"]
    assert rec_csl, "fixture should record a csl section (csl=True)"
    for key, value in rec_csl.items():
        assert loaded.stat("csl", key) == value


def test_gb_character_fraction_displayed_equals_recorded(run_dir: Path,
                                                         recorded: dict):
    loaded = core.load_outputs(run_dir)
    rec_char = recorded["statistics"]["gb_character"]
    for key, value in rec_char.items():
        assert loaded.stat("gb_character", key) == value


def test_key_stats_table_values_are_recorded(run_dir: Path, recorded: dict):
    """Every row of the page's key-stats table equals the recorded value."""
    loaded = core.load_outputs(run_dir)
    rec_stats = recorded["statistics"]
    for row in core.key_stats_table(loaded):
        assert rec_stats[row["section"]][row["key"]] == row["value"]


def test_grain_volumes_equal_recorded(run_dir: Path, recorded: dict):
    """The histogram's source volumes are the recorded per-grain records."""
    loaded = core.load_outputs(run_dir)
    rec_vols = [g["volume_A3"] for g in recorded["grains"]]
    assert loaded.grain_volumes_A3 == rec_vols


# ---------------------------------------------------------------------------
# Graceful handling of a MISSING optional artifact (mdf.csv)
# ---------------------------------------------------------------------------


def test_missing_mdf_csv_handled(run_dir: Path, tmp_path: Path):
    """Copy the run without mdf.csv: load still succeeds and flags it absent
    (the viewer/serializer skip the MDF panel gracefully) while the other
    recorded artifacts still load."""
    import shutil

    dst = tmp_path / "no_mdf"
    dst.mkdir()
    for src in run_dir.iterdir():
        if src.is_file() and src.name != "mdf.csv":
            shutil.copy2(src, dst / src.name)
    assert not (dst / "mdf.csv").exists()

    loaded = core.load_outputs(dst)
    assert not loaded.has_mdf_csv
    assert not loaded.mdf_rows                       # MDF panel skipped
    assert loaded.has_microstructure_json            # the rest still loaded
    assert loaded.grain_volumes_A3
