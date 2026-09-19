"""Tests for grainsmith.verify (§5, R2): recompute every MANIFEST.txt digest,
re-derive the config hash from the shipped resolved_config.yaml, and check
that the recorded grainsmith version is cryptographically bound to the run —
all from an output directory's shipped files alone."""
from __future__ import annotations

from pathlib import Path

import pytest

from grainsmith.errors import ManifestError
from grainsmith.pipeline import run
from grainsmith.verify import render_report, verify_outdir

SEED = 20260611


def _config(outdir):
    """NP=4, 40^3 A, SG225 Cu -- same base config as test_end_to_end.py, kept
    independent so this module has no import-time dependency on it."""
    from grainsmith.config.resolve import resolve_config

    raw = {
        "meta": {"title": "verify test", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    return resolve_config(raw)


def _run(tmp_path) -> Path:
    res = run(_config(tmp_path / "out"))
    assert res.gates.all_passed()
    return res.outdir


def _rewrite(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not found in {path}"
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


# ---------------------------------------------------------------------------
# V1: clean run verifies
# ---------------------------------------------------------------------------


def test_clean_run_verifies(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir)
    assert report.exit_code == 0
    assert report.n_fail == 0
    for c in report.checks:
        assert c.status in ("OK", "SKIP"), (c.name, c.status, c.summary)


# ---------------------------------------------------------------------------
# V2: tampered data file is caught
# ---------------------------------------------------------------------------


def test_tampered_data_file_is_caught(tmp_path):
    outdir = _run(tmp_path)
    with (outdir / "grains.csv").open("ab") as fh:
        fh.write(b"\n")
    report = verify_outdir(outdir)
    assert report.exit_code == 1
    fd = _check(report, "file-digests")
    assert fd.status == "FAIL"
    assert any("grains.csv" in d for d in fd.details)


# ---------------------------------------------------------------------------
# V3: deleted file is caught
# ---------------------------------------------------------------------------


def test_deleted_file_is_caught(tmp_path):
    outdir = _run(tmp_path)
    (outdir / "boundaries.csv").unlink()
    report = verify_outdir(outdir)
    assert report.exit_code == 1
    fd = _check(report, "file-digests")
    assert fd.status == "FAIL"
    assert any("boundaries.csv" in d for d in fd.details)


# ---------------------------------------------------------------------------
# V4: tampered resolved_config.yaml body is caught in THREE places
# ---------------------------------------------------------------------------


def test_tampered_config_body_fails_three_checks(tmp_path):
    outdir = _run(tmp_path)
    path = outdir / "resolved_config.yaml"
    lines = path.read_text(encoding="utf-8").split("\n")
    body_idx = next(i for i, line in enumerate(lines)
                    if not line.startswith("#") and line.strip())
    lines[body_idx] = lines[body_idx] + "  # tampered"
    path.write_text("\n".join(lines), encoding="utf-8")

    report = verify_outdir(outdir)
    assert report.exit_code == 1
    assert _check(report, "config-roundtrip").status == "FAIL"
    # comparison (b) uses the freshly recomputed (now-wrong) digest, so
    # version-binding independently detects the same tamper.
    assert _check(report, "version-binding").status == "FAIL"
    assert _check(report, "file-digests").status == "FAIL"


# ---------------------------------------------------------------------------
# V5: edited version string is caught -- the R1 test
# ---------------------------------------------------------------------------


def test_edited_version_string_invalidates_the_digest(tmp_path):
    """R1: the recorded version cannot be changed without invalidating the
    provenance digest. Only MANIFEST.txt's header prefix is rewritten here —
    the full-length digest comment lines are left untouched — so
    ``version-binding`` must fail purely from the version/digest mismatch,
    not from any hash-of-file discrepancy."""
    outdir = _run(tmp_path)
    manifest_path = outdir / "MANIFEST.txt"
    _rewrite(manifest_path, "# grainsmith 1.1.0 |", "# grainsmith 9.9.9 |")

    report = verify_outdir(outdir)
    assert report.exit_code == 1
    vb = _check(report, "version-binding")
    assert vb.status == "FAIL"
    assert _check(report, "version-consistency").status == "FAIL"


# ---------------------------------------------------------------------------
# V6: stale extra file is a WARN, not a failure -- unless --strict
# ---------------------------------------------------------------------------


def test_stale_extra_file_is_warn_not_failure(tmp_path):
    outdir = _run(tmp_path)
    (outdir / "stale.txt").write_text("leftover\n", encoding="utf-8")

    report = verify_outdir(outdir)
    assert report.exit_code == 0
    ef = _check(report, "extra-files")
    assert ef.status == "WARN"
    assert "stale.txt" in ef.details

    # --strict promotes ALL warnings to failures at the report level
    # (VerifyReport.exit_code), not by rewriting individual check statuses --
    # extra-files still reports "WARN" in the rendered output either way.
    strict_report = verify_outdir(outdir, strict=True)
    assert strict_report.exit_code == 1
    assert _check(strict_report, "extra-files").status == "WARN"


def test_render_report_strict_promoted_wording_is_self_consistent(tmp_path):
    """Under --strict with warnings only (n_fail == 0), the final line must
    not read "FAILED: ... 0 failed" -- the exit code says failure but a
    literal "0 failed" would contradict it. It must instead say the failure
    is because warnings were promoted under --strict."""
    outdir = _run(tmp_path)
    (outdir / "stale.txt").write_text("leftover\n", encoding="utf-8")

    report = verify_outdir(outdir, strict=True)
    assert report.exit_code == 1
    assert report.n_fail == 0

    final_line = render_report(report).strip().splitlines()[-1]
    assert final_line.startswith("FAILED:")
    assert "0 failed" not in final_line
    assert "--strict" in final_line


# ---------------------------------------------------------------------------
# --strict promotes every WARN-capable check, not just extra-files (C3)
# ---------------------------------------------------------------------------


def test_strict_promotes_version_binding_warning_too(tmp_path):
    """version-binding (C5) can WARN independently of extra-files (C3) --
    when resolved_config.yaml is present but its MANIFEST.txt entry is
    removed, config-roundtrip (C4) can no longer recompute a digest, so C5
    can only check comparison (a) and WARNs. --strict must promote THAT
    warning too, not just C3's."""
    outdir = _run(tmp_path)
    manifest_path = outdir / "MANIFEST.txt"
    lines = manifest_path.read_text(encoding="utf-8").split("\n")
    kept = [line for line in lines
           if not line.endswith("  resolved_config.yaml")]
    assert len(kept) == len(lines) - 1, "expected exactly one line removed"
    manifest_path.write_text("\n".join(kept), encoding="utf-8")

    report = verify_outdir(outdir)
    assert report.n_fail == 0
    assert _check(report, "config-roundtrip").status == "WARN"
    assert _check(report, "version-binding").status == "WARN"
    assert report.exit_code == 0

    strict_report = verify_outdir(outdir, strict=True)
    assert strict_report.n_fail == 0
    # Status text is unchanged under strict -- only the aggregate exit code.
    assert _check(strict_report, "version-binding").status == "WARN"
    assert strict_report.exit_code == 1


# ---------------------------------------------------------------------------
# V7: run.log is skipped cleanly
# ---------------------------------------------------------------------------


def test_run_log_is_skipped_cleanly(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir)
    fd = _check(report, "file-digests")
    assert fd.status == "OK"
    assert "run.log" in fd.summary

    (outdir / "run.log").write_text("mutated\n" * 10, encoding="utf-8")
    report2 = verify_outdir(outdir)
    assert report2.exit_code == 0
    assert _check(report2, "file-digests").status == "OK"


# ---------------------------------------------------------------------------
# V8: missing / malformed MANIFEST.txt -> exit 2 (via the CLI handler)
# ---------------------------------------------------------------------------


def test_missing_manifest_is_exit_2_via_cli(tmp_path, capsys):
    from grainsmith.cli import _cmd_verify
    import argparse

    outdir = _run(tmp_path)
    (outdir / "MANIFEST.txt").unlink()
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(outdir), strict=False, expect_version=None, quiet=False))
    assert rc == 2
    assert "cannot verify" in capsys.readouterr().err


def test_missing_manifest_raises_manifest_error_directly(tmp_path):
    outdir = _run(tmp_path)
    (outdir / "MANIFEST.txt").unlink()
    with pytest.raises(ManifestError):
        verify_outdir(outdir)


def test_garbage_manifest_raises_manifest_error(tmp_path):
    outdir = _run(tmp_path)
    (outdir / "MANIFEST.txt").write_text("not a manifest at all\n",
                                         encoding="utf-8")
    with pytest.raises(ManifestError):
        verify_outdir(outdir)


# ---------------------------------------------------------------------------
# V9: --expect-version
# ---------------------------------------------------------------------------


def test_expect_version_matching(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir, expect_version="1.1.0")
    assert report.exit_code == 0
    assert _check(report, "expected-version").status == "OK"


def test_expect_version_mismatch(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir, expect_version="9.9.9")
    assert report.exit_code == 1
    assert _check(report, "expected-version").status == "FAIL"


# ---------------------------------------------------------------------------
# V10: report rendering is deterministic
# ---------------------------------------------------------------------------


def test_render_report_is_deterministic(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir)
    r1 = render_report(report)
    r2 = render_report(report)
    assert r1 == r2

    # No wall-clock/elapsed-time digits beyond the one recorded timestamp:
    # strip that single occurrence and make sure no other ISO-8601-shaped
    # timestamp remains.
    import re
    stripped = r1.replace(report.manifest.timestamp_iso, "", 1)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stripped)


def test_render_report_quiet_is_one_line(tmp_path):
    outdir = _run(tmp_path)
    report = verify_outdir(outdir)
    rendered = render_report(report, quiet=True)
    assert "\n" not in rendered
    assert rendered.startswith("VERIFIED:")
