"""Tests for tools/golden_hashes.py (R4): the double-run determinism check,
the golden-digest collector, and the --check exit-code matrix ("CI must not
go permanently red") -- all exercised directly through the tool's functions,
with no subprocess, so this is the same coverage the CI jobs get (§4.2's
"determinism" job and §4.3's "golden-hash" job) but runs locally.

golden_hashes.json itself is produced by CI, never by a developer laptop (see
tools/golden_hashes.py's module docstring, "Never run --write on a developer
laptop") -- so R4 below skips when it is not yet committed, and no test here
writes it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN_HASHES = REPO / "tests" / "data" / "golden" / "golden_hashes.json"


def _load_tool():
    """Import tools/golden_hashes.py (not an installed package) -- same
    sys.path.insert(0, str(REPO)) pattern as tests/test_docs.py."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import tools.golden_hashes as gh
    return gh


# ---------------------------------------------------------------------------
# R1: double-run is byte-identical
# ---------------------------------------------------------------------------


def test_double_run_is_byte_identical(tmp_path, monkeypatch):
    gh = _load_tool()
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    assert gh.double_run(tmp_path) == []


def test_main_refuses_to_run_without_source_date_epoch(monkeypatch, capsys):
    """main() must refuse (parser.error -> SystemExit(2)) rather than produce
    a timing-dependent false result: every mode compares bytes that embed the
    run's UTC timestamp, so an unset SOURCE_DATE_EPOCH makes --double-run
    spuriously report "differing files" whenever the two runs straddle a
    second boundary, and makes --check spuriously report REGRESSION against
    a golden recorded with the clock frozen. This must exit before any golden
    run is triggered, so the test is instant."""
    import pytest

    gh = _load_tool()
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        gh.main(["--double-run"])
    assert exc_info.value.code == 2
    assert "SOURCE_DATE_EPOCH" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# R2: collect() excludes exactly the four documented names
# ---------------------------------------------------------------------------


def test_collect_excludes_documented_names_only(tmp_path):
    gh = _load_tool()
    outdir = gh.run_golden(tmp_path)
    files = gh.collect(outdir)

    assert gh.GOLDEN_EXCLUDE.isdisjoint(files)
    for expected in ("polycrystal.data", "grains.csv", "resolved_config.yaml"):
        assert expected in files, f"{expected} missing from collect() output"


# ---------------------------------------------------------------------------
# R3: compare() matrix -- all four branches, no subprocess
#
# This is the test that pins the "CI must not go permanently red" property:
# an absent golden file, or one recorded on a different machine, must never
# turn into a hard failure (exit 1) -- only an actual digest regression
# inside the SAME fingerprint does.
# ---------------------------------------------------------------------------


def _record(**overrides) -> dict:
    base = {
        "schema": "grainsmith/golden-hashes/v1",
        "grainsmith_version": "1.1.0",
        "source_date_epoch": 1700000000,
        "config": "tests/data/golden/golden_config.yaml",
        "config_sha256": "a" * 64,
        "provenance_sha256": "b" * 64,
        "fingerprint": {
            "python_implementation": "CPython",
            "python_version": "3.11.15",
            "os": "Linux",
            "machine": "x86_64",
            "blas_name": "blas",
            "blas_version": "3.9.0",
            "blas_detection": "pkgconfig",
            "numpy": "2.4.6",
            "scipy": "1.17.1",
            "spglib": "2.7.0",
        },
        "environment": {},
        "files": {
            "boundaries.csv": "1" * 64,
            "grains.csv": "2" * 64,
        },
    }
    base.update(overrides)
    return base


def test_compare_match():
    gh = _load_tool()
    golden = _record()
    fresh = _record()
    status, lines = gh.compare(golden, fresh)
    assert status == "MATCH"
    assert lines == []


def test_compare_bootstrap_when_golden_absent():
    gh = _load_tool()
    status, lines = gh.compare({}, _record())
    assert status == "BOOTSTRAP"


def test_compare_stale_env_on_fingerprint_change():
    gh = _load_tool()
    golden = _record()
    fresh = _record(fingerprint={**golden["fingerprint"], "blas_name": "openblas"})
    status, lines = gh.compare(golden, fresh)
    assert status == "STALE-ENV"
    assert any("blas_name" in line for line in lines)


def test_compare_regression_on_file_digest_change():
    gh = _load_tool()
    golden = _record()
    fresh = _record(files={**golden["files"], "grains.csv": "9" * 64})
    status, lines = gh.compare(golden, fresh)
    assert status == "REGRESSION"
    assert any("grains.csv" in line for line in lines)


# ---------------------------------------------------------------------------
# R4: golden JSON is well-formed if it exists
# ---------------------------------------------------------------------------


def test_golden_json_well_formed_if_present():
    import pytest

    if not GOLDEN_HASHES.is_file():
        pytest.skip("golden set not bootstrapped yet — see "
                    "tools/golden_hashes.py")
    doc = json.loads(GOLDEN_HASHES.read_text(encoding="utf-8"))
    assert doc["schema"] == "grainsmith/golden-hashes/v1"
    assert doc["files"]
