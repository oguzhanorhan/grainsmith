"""End-to-end pipeline tests (§10): small FCC Cu configs, flat + curved,
all QA gates green, byte-identical re-run (timestamps stripped)."""
import csv
import json
import re

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.pipeline import _available_cpus, run
from grainsmith.provenance import ENVIRONMENT_KEYS

SEED = 20260611


def _config(outdir, **overrides):
    """NP=4, 40³ Å, SG225 Cu — the §10 end-to-end base config."""
    raw = {
        "meta": {"title": "e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu",
                               "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            for k2, v2 in sub.items():
                if isinstance(v2, dict) and isinstance(raw[key].get(k2),
                                                       dict):
                    raw[key][k2].update(v2)
                else:
                    raw[key][k2] = v2
        else:
            raw[key] = sub
    return resolve_config(raw)


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Flat
# ---------------------------------------------------------------------------


def test_flat_all_gates_and_outputs(tmp_path):
    cfg = _config(tmp_path / "out",
                  analysis={"per_atom_margin": True})
    res = run(cfg)

    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    # G23 (report-only) fires for every single-phase run that reaches the
    # MDF-histogram block, independent of orientation.mdf_target. G26
    # (report-only) fires whenever per-grain atom counts exist.
    assert gate_ids == {f"G{k}" for k in range(1, 11)} | {"G23", "G26"}

    out = res.outdir
    for name in ("polycrystal.data", "polycrystal.extxyz", "grains.csv",
                 "boundaries.csv", "vertices.csv", "summary.csv",
                 "seeds.dat", "edges.dat", "box.dat", "view.plt",
                 "resolved_config.yaml", "run.log", "MANIFEST.txt"):
        assert (out / name).exists(), f"missing output: {name}"

    # grains.csv: one row per grain; volume fractions sum to 1
    grows = _csv_rows(out / "grains.csv")
    assert len(grows) == 4
    assert sum(float(r["volume_fraction"]) for r in grows) == \
        pytest.approx(1.0, abs=1e-9)
    assert sum(int(r["n_atoms"]) for r in grows) == len(res.atoms)

    # boundaries.csv rows == adjacency pairs
    brows = _csv_rows(out / "boundaries.csv")
    assert len(brows) == len(res.tess.adjacency()) == \
        len(res.boundary_reports)

    # atoms sorted by (grain, generation order); ids implicit 1..N
    assert np.all(np.diff(res.atoms.grain) >= 0)

    # per_atom_margin reached the XYZ writer
    header = (out / "polycrystal.extxyz").read_text(
        encoding="utf-8").splitlines()[1]
    assert "gb_margin:R:1" in header

    # summary contains every gate row marked PASS (G1-G10 plus G23, which
    # fires for every single-phase run reaching the MDF-histogram block,
    # and G26, which fires whenever per-grain atom counts exist)
    srows = _csv_rows(out / "summary.csv")
    gate_rows = [r for r in srows if r["section"] == "gates"]
    assert len(gate_rows) == 12
    assert all(r["value"].startswith("PASS") for r in gate_rows)


@pytest.mark.parametrize("epoch", [None, "1700000000"])
def test_summary_contains_all_completed_timings(tmp_path, monkeypatch, epoch):
    if epoch is None:
        monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    else:
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
    result = run(_config(tmp_path / "out"))
    reported = {
        row["key"]: float(row["value"])
        for row in _csv_rows(result.outdir / "summary.csv")
        if row["section"] == "timings"
    }
    assert set(reported) == {f"{stage}_s" for stage in result.timings}
    for stage, elapsed in result.timings.items():
        assert reported[f"{stage}_s"] == elapsed
        assert np.isfinite(elapsed) and elapsed >= 0.0
    log_text = (result.outdir / "run.log").read_text(encoding="utf-8")
    for stage, elapsed in result.timings.items():
        assert f"{stage}={elapsed!r}" in log_text


def test_summary_composition_and_export_density(tmp_path):
    from grainsmith.constants import AMU_PER_A3_TO_G_PER_CM3

    result = run(_config(
        tmp_path / "out",
        output={"lammps": {"masses": {"Cu": 70.0}}},
        box={"periodic": [True, True, False], "vacuum": 5.0},
    ))
    summary = {(row["section"], row["key"]): row["value"]
               for row in _csv_rows(result.outdir / "summary.csv")}
    count = len(result.atoms)
    assert int(summary["composition", "Cu_n_final"]) == count
    assert float(summary["composition", "Cu_final_fraction"]) == 1.0
    assert float(summary["composition", "Cu_final_mass_fraction"]) == 1.0
    assert float(summary["lammps_masses", "Cu"]) == 70.0
    assert float(summary["atoms", "mass_total_amu"]) == 70.0 * count
    assert float(summary["atoms", "density_material_g_cm3"]) == pytest.approx(
        count * 70.0 / 40.0**3 * AMU_PER_A3_TO_G_PER_CM3)
    assert float(summary["atoms", "density_export_g_cm3"]) == pytest.approx(
        count * 70.0 / (40.0 * 40.0 * 45.0) * AMU_PER_A3_TO_G_PER_CM3)
    assert summary["composition", "nominal_basis"] == "pre_doping_host_atom_fraction"
    assert summary["composition", "final_basis"] == "post_doping_atom_fraction"
    assert summary["texture", "scheme"] == result.config.orientation.scheme
    assert summary["analysis", "csl"] == str(result.config.analysis.csl)
    assert summary["overlap", "enabled"] == "True"
    assert summary["sigma3", "angle_window_status"] == "measured"


def test_summary_contains_complete_config_and_measured_statistics(tmp_path):
    result = run(_config(tmp_path / "out", analysis={"statistics": True}))
    summary = {(row["section"], row["key"]): row["value"]
               for row in _csv_rows(result.outdir / "summary.csv")}

    def assert_config_entry(value, path):
        if isinstance(value, dict) and value:
            for name, item in value.items():
                assert_config_entry(item, f"{path}.{name}" if path else name)
        elif isinstance(value, list) and value:
            for index, item in enumerate(value):
                assert_config_entry(item, f"{path}[{index}]")
        else:
            expected = ("null" if value is None else
                        json.dumps(value) if isinstance(value, (dict, list)) else
                        str(value))
            assert summary["config", path] == expected

    assert_config_entry(result.config.model_dump(mode="json"), "")
    for section, values in result.statistics.items():
        for key, value in values.items():
            reported = summary[f"statistics:{section}", key]
            if isinstance(value, (float, int)):
                assert float(reported) == pytest.approx(value, nan_ok=True)
            else:
                assert reported == str(value)
    assert summary["memory", "allocation_limit_configured_gb"] == \
        str(result.config.runtime.memory_limit_gb)
    assert float(summary["memory", "allocation_limit_effective_gb"]) > 0.0
    assert summary["memory", "rss_limit_kind"] == "warning_only"


def test_flat_slab_with_vacuum(tmp_path):
    from grainsmith.io import parse_lammps

    cfg = _config(tmp_path / "out",
                  box={"periodic": [True, True, False], "vacuum": 5.0},
                  grains={"number": 3})
    res = run(cfg)
    assert res.gates.all_passed()

    parsed = parse_lammps(res.outdir / "polycrystal.data")
    assert parsed["bounds"][2, 1] == 45.0          # L + vacuum on free z
    assert parsed["bounds"][0, 1] == 40.0
    # all atoms inside the padded bounds, none in the outer vacuum halves
    assert parsed["pos"][:, 2].min() >= 2.5 - 1e-9
    assert parsed["pos"][:, 2].max() <= 42.5 + 1e-9

    # slab vertices.csv carries wall (-1) adjacency
    vtext = (res.outdir / "vertices.csv").read_text(encoding="utf-8")
    assert "-1" in vtext

    summary = {(row["section"], row["key"]): row["value"]
               for row in _csv_rows(res.outdir / "summary.csv")}
    np.testing.assert_array_equal(
        np.fromstring(summary["box", "export_lengths_A"], sep=" "),
        parsed["bounds"][:, 1] - parsed["bounds"][:, 0])
    assert summary["box", "vacuum_per_axis_A"] == "0.0 0.0 5.0"
    assert summary["box", "export_shift_A"] == "0.0 0.0 2.5"
    assert float(summary["box", "material_volume_A3"]) == 64000.0
    assert float(summary["box", "export_volume_A3"]) == 72000.0
    assert float(summary["box", "vacuum_volume_A3"]) == 8000.0
    assert float(summary["box", "vacuum_fraction"]) == pytest.approx(1.0 / 9.0)


# ---------------------------------------------------------------------------
# Curved (warp)
# ---------------------------------------------------------------------------


def test_curved_warp_all_gates_and_outputs(tmp_path):
    cfg = _config(
        tmp_path / "out",
        boundaries={"geometry": "curved",
                    # amplitude retuned 2.0 -> 0.6 (warp.py DC-mode-exclusion
                    # fix: removing the k=0 mode from the gaussian spectrum
                    # removes a rigid-translation contribution that used to
                    # inflate the RMS calibration denominator, so the same
                    # amplitude now delivers a larger max‖∇u‖).
                    "curved": {"method": "warp", "amplitude": 0.6,
                               "correlation_length": 10.0}},
    )
    res = run(cfg)
    assert res.gates.all_passed()

    out = res.outdir
    assert (out / "gb_points.dat").exists()        # curved gnuplot bundle
    assert not (out / "edges.dat").exists()
    assert not (out / "vertices.csv").exists()     # flat-only report
    assert len(res.boundary_reports) > 0

    # G6 row carries the measured max‖∇u‖
    g6 = [r for r in res.gates.results() if r.gate == "G6"][0]
    assert g6.measured is not None and 0.0 < g6.measured < 0.5


# ---------------------------------------------------------------------------
# Determinism: byte-identical re-run (timestamps stripped, §10)
# ---------------------------------------------------------------------------

_TS = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_SKIP = {"run.log", "MANIFEST.txt"}   # wall-clock content by design


def _normalized(path) -> bytes:
    data = _TS.sub(b"<TS>", path.read_bytes())
    if path.name == "summary.csv":
        # wall times are timestamps in disguise
        data = b"\n".join(line for line in data.split(b"\n")
                          if not line.startswith((b"timings,", b"memory,")))
    return data


def test_byte_identical_rerun(tmp_path, monkeypatch):
    # The SAME config (incl. the relative output.directory — it is hashed
    # into the provenance line) run from two working directories.
    root1 = tmp_path / "run1"
    root2 = tmp_path / "run2"
    root1.mkdir()
    root2.mkdir()
    monkeypatch.chdir(root1)
    res1 = run(_config("./out"))
    monkeypatch.chdir(root2)
    res2 = run(_config("./out"))
    assert res1.gates.all_passed() and res2.gates.all_passed()

    out1 = root1 / "out"
    out2 = root2 / "out"
    names1 = {p.name for p in out1.iterdir() if p.is_file()}
    names2 = {p.name for p in out2.iterdir() if p.is_file()}
    assert names1 == names2

    for name in sorted(names1 - _SKIP):
        b1 = _normalized(out1 / name)
        b2 = _normalized(out2 / name)
        assert b1 == b2, f"{name} differs between identical runs"


def test_byte_identical_rerun_source_date_epoch(tmp_path, monkeypatch):
    """Fixed timestamps preserve scientific bytes, not real measured timings.

    The summary's timing rows and its genuine manifest checksum may vary;
    every other manifest entry and every other summary row must match.
    """
    import hashlib

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    root1 = tmp_path / "run1"
    root2 = tmp_path / "run2"
    root1.mkdir()
    root2.mkdir()
    monkeypatch.chdir(root1)
    res1 = run(_config("./out"))
    monkeypatch.chdir(root2)
    res2 = run(_config("./out"))
    assert res1.gates.all_passed() and res2.gates.all_passed()

    out1 = root1 / "out"
    out2 = root2 / "out"
    names1 = {p.name for p in out1.iterdir() if p.is_file()}
    names2 = {p.name for p in out2.iterdir() if p.is_file()}
    assert names1 == names2

    for name in sorted(names1 - {"run.log", "summary.csv", "MANIFEST.txt"}):
        b1 = (out1 / name).read_bytes()
        b2 = (out2 / name).read_bytes()
        assert b1 == b2, f"{name} differs between identical runs"

    assert _normalized(out1 / "summary.csv") == _normalized(out2 / "summary.csv")
    manifest_stable = []
    for output in (out1, out2):
        data = (output / "summary.csv").read_bytes()
        lines = (output / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
        assert f"{hashlib.sha256(data).hexdigest()}  {len(data)}  summary.csv" in lines
        manifest_stable.append([line for line in lines
                                if not line.endswith("  summary.csv")])
    assert manifest_stable[0] == manifest_stable[1]
    summary = (out1 / "summary.csv").read_text(encoding="utf-8")
    assert "timings,total_s," in summary
    assert "meta,source_date_epoch,1700000000" in summary
    assert "meta,timings_status,measured" in summary
    assert "meta,timings_log,run.log" in summary
    log_text = (out1 / "run.log").read_text(encoding="utf-8")
    for stage, elapsed in res1.timings.items():
        assert f"{stage}={elapsed!r}" in log_text


def test_entropy_seed_recorded(tmp_path):
    cfg = _config(tmp_path / "out", seed={"mode": "entropy", "value": None},
                  grains={"number": 2})
    res = run(cfg)
    assert res.gates.all_passed()
    assert isinstance(res.seed, int)
    header = (res.outdir / "resolved_config.yaml").read_text(
        encoding="utf-8").splitlines()[0]
    assert f"seed={res.seed}" in header


def test_jobs_parallel_pipeline_bit_identical(tmp_path):
    """run(config, jobs=2) must produce the same atoms as jobs=1 (§13:
    --jobs is an execution detail, not physics)."""
    import numpy as np

    res1 = run(_config(tmp_path / "o1"), jobs=1)
    res2 = run(_config(tmp_path / "o2"), jobs=2)
    assert res1.gates.all_passed() and res2.gates.all_passed()
    np.testing.assert_array_equal(res1.atoms.pos, res2.atoms.pos)
    assert np.array_equal(res1.atoms.species, res2.atoms.species)
    assert np.array_equal(res1.atoms.grain, res2.atoms.grain)


def test_size_distribution_flat_e2e(tmp_path):
    """R1 e2e: lognormal volume targeting → 11 gates incl. G11 PASS."""
    cfg = _config(tmp_path / "out",
                  grains={"number": 6,
                          "size_distribution": {"type": "lognormal",
                                                "sigma_log": 0.3}})
    res = run(cfg)
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G11" in gate_ids
    g11 = [r for r in res.gates.results() if r.gate == "G11"][0]
    assert g11.measured <= 1e-3
    # summary carries the SDOT rows
    srows = _csv_rows(res.outdir / "summary.csv")
    sections = {r["section"] for r in srows}
    assert "size_distribution" in sections


def test_size_distribution_warp_base_e2e(tmp_path):
    """Warp on a volume-fitted power base — G11 applies to the
    unwarped base."""
    cfg = _config(
        tmp_path / "out",
        grains={"number": 4,
                "size_distribution": {"type": "equal"}},
        boundaries={"geometry": "curved",
                    # amplitude retuned 2.0 -> 0.6 (warp.py DC-mode-exclusion
                    # fix: removing the k=0 mode from the gaussian spectrum
                    # removes a rigid-translation contribution that used to
                    # inflate the RMS calibration denominator, so the same
                    # amplitude now delivers a larger max‖∇u‖).
                    "curved": {"method": "warp", "amplitude": 0.6,
                               "correlation_length": 10.0}},
    )
    res = run(cfg)
    assert res.gates.all_passed()
    assert "G11" in {r.gate for r in res.gates.results()}


def test_manifest_lists_only_this_runs_outputs(tmp_path):
    """MANIFEST.txt provenance must cover only the files THIS run produced, so a
    stale artefact left in a reused output directory is not falsely attributed to
    the run (its version/seed/config-sha provenance line)."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    stale = outdir / "stale_from_a_previous_run.txt"
    stale.write_text("leftover\n")

    res = run(_config(outdir))
    assert res.gates.all_passed()
    manifest = (res.outdir / "MANIFEST.txt").read_text(encoding="utf-8")

    assert "stale_from_a_previous_run.txt" not in manifest
    assert "grains.csv" in manifest          # a real output IS listed
    assert "run.log" in manifest             # run.log listed (without a hash)


# ---------------------------------------------------------------------------
# E6-E7: environment provenance (§R3) reaches summary.csv AND
# microstructure.json, and the two can never disagree (built once in run()
# and handed to both writers).
# ---------------------------------------------------------------------------


def test_environment_rows_reach_summary_and_microstructure_json(tmp_path):
    """E6: the `environment` section of summary.csv has exactly
    len(ENVIRONMENT_KEYS) rows, in ENVIRONMENT_KEYS order, and agrees
    byte-for-byte (as a dict) with microstructure.json's "environment" key."""
    cfg = _config(tmp_path / "out", analysis={"statistics": True})
    res = run(cfg, jobs=1)
    assert res.gates.all_passed()

    srows = _csv_rows(res.outdir / "summary.csv")
    env_rows = [r for r in srows if r["section"] == "environment"]
    assert len(env_rows) == len(ENVIRONMENT_KEYS)
    assert [r["key"] for r in env_rows] == list(ENVIRONMENT_KEYS)
    summary_env = {r["key"]: r["value"] for r in env_rows}

    doc = json.loads((res.outdir / "microstructure.json").read_text(
        encoding="utf-8"))
    assert doc["environment"] == summary_env
    assert summary_env["jobs_effective"] == "1"


def test_jobs_zero_records_both_requested_and_effective_forms(tmp_path):
    """E7: `--jobs 0` (run(cfg, jobs=0)) records the REQUESTED form (0, what
    reproduces the invocation) and the EFFECTIVE form (the expanded worker
    count, what actually ran) as two separate summary.csv rows."""
    cfg = _config(tmp_path / "out")
    res = run(cfg, jobs=0)
    assert res.gates.all_passed()

    srows = _csv_rows(res.outdir / "summary.csv")
    env = {(r["section"], r["key"]): r["value"] for r in srows
          if r["section"] == "environment"}
    assert env[("environment", "jobs_requested")] == "0"
    assert env[("environment", "jobs_effective")] == str(_available_cpus())
