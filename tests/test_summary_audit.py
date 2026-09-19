"""tools/audit_summary_csv.py reports on archived summaries and writes nothing.

Replaces tests/test_summary_completion.py, which pinned the opposite contract
(rewrite summary.csv in place, re-sign its MANIFEST.txt entry, leave a
``.before-summary-completion`` sidecar). That behaviour was retired: the
sidecars made ``grainsmith verify --strict`` fail on every directory the tool
touched, and re-signing the manifest entry made ``grainsmith verify`` report
VERIFIED for a summary.csv no grainsmith run ever wrote.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

from grainsmith.io.reports import write_summary_csv


def _tool():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import tools.audit_summary_csv as mod
    return mod


def _archive(directory, *, summary_rows=None):
    config = {"box": {"lengths": [40.0, 40.0, 40.0]},
              "boundaries": {"geometry": "curved"},
              "crystal": {"lattice": {"a": 3.615}},
              "orientation": {"components": [{"weight": 0.7}, {"weight": 0.3}]},
              "runtime": {"memory_limit_gb": 25.0},
              "analysis": {"statistics": True}}
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8")
    write_summary_csv(summary_rows or [("atoms", "n_final", 42)],
                      directory / "summary.csv")
    write_summary_csv([("grain_size", "d_eq_mean_A", 20.0)],
                      directory / "statistics.csv")
    (directory / "run.log").write_text(
        "Memory guard (13): 25 GB per allocation (runtime.memory_limit_gb).\n"
        "stage timings (s): fill=1.23456789, total=2.5\n"
        "Voxel grid (20, 20, 20): 8,000 voxels\n"
        "ok: peak driver RSS 12.76 GB vs system-RAM auto-ceiling 30.00 GB\n",
        encoding="utf-8")
    (directory / "polycrystal.data").write_bytes(b"untouched scientific artifact")
    entries = []
    for name in ("resolved_config.yaml", "summary.csv",
                 "statistics.csv", "polycrystal.data"):
        content = (directory / name).read_bytes()
        entries.append(
            f"{hashlib.sha256(content).hexdigest()}  {len(content)}  {name}")
    (directory / "MANIFEST.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8")


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_audit_writes_nothing(tmp_path):
    """The whole point of the rewrite: every byte in the directory survives,
    and no new file appears (least of all a sidecar)."""
    _archive(tmp_path)
    before = _snapshot(tmp_path)

    report = _tool().audit(tmp_path)

    assert _snapshot(tmp_path) == before, "audit modified the archive"
    assert not list(tmp_path.glob("*.before-summary-completion"))
    assert report.n_present == 1          # only atoms,n_final was recorded


def test_audit_reports_derivable_rows_as_missing(tmp_path):
    """A sparse archive is reported, not repaired."""
    _archive(tmp_path)
    report = _tool().audit(tmp_path)
    missing = set(report.missing)

    assert ("timings", "fill_s") in missing
    assert ("config", "orientation.components[1].weight") in missing
    assert ("statistics:grain_size", "d_eq_mean_A") in missing
    assert ("memory", "peak_driver_rss_gb") in missing
    assert ("analysis", "voxel_grid_shape") in missing
    assert not report.clean

    text = _tool().render(report)
    assert "re-run the config" in text     # the honest remedy
    assert "will not forge one" in text


def test_audit_flags_sidecars_left_by_the_retired_tool(tmp_path):
    """The un-manifested sidecars are why `verify --strict` fails; say so."""
    _archive(tmp_path)
    (tmp_path / "summary.csv.before-summary-completion").write_bytes(b"old")
    (tmp_path / "MANIFEST.txt.before-summary-completion").write_bytes(b"old")

    report = _tool().audit(tmp_path)

    assert report.sidecars == ["MANIFEST.txt.before-summary-completion",
                               "summary.csv.before-summary-completion"]
    assert "verify --strict" in _tool().render(report)


def test_audit_checks_every_manifest_entry_not_a_sample(tmp_path):
    """The retired tool verified 3 of ~17 entries before rewriting; this one
    reports on all of them and still refuses to change anything."""
    _archive(tmp_path)
    (tmp_path / "polycrystal.data").write_bytes(b"tampered")
    before = _snapshot(tmp_path)

    report = _tool().audit(tmp_path)

    assert any("polycrystal.data" in item for item in report.integrity)
    assert _snapshot(tmp_path) == before


def test_audit_reports_duplicate_keys_instead_of_raising(tmp_path):
    """A duplicate (section,key) is a defect to surface, not a reason to
    abandon the report."""
    _archive(tmp_path, summary_rows=[("doping", "C_nominal_fraction", 0.02),
                                     ("doping", "C_nominal_fraction", 0.005)])
    report = _tool().audit(tmp_path)

    assert ("doping", "C_nominal_fraction") in report.duplicates
    assert "silently keeps only the last" in _tool().render(report)


def test_audit_main_exit_code(tmp_path, capsys):
    _archive(tmp_path)
    assert _tool().main([str(tmp_path)]) == 1     # gaps -> non-zero
    assert str(tmp_path) in capsys.readouterr().out
