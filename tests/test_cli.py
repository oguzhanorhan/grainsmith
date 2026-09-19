"""Tests for the grainsmith CLI (S2 — cli.py was 0% covered).

Drives the subcommand handlers in-process via argparse.Namespace; covers the
happy and error paths of info/validate/generate and pins the gate-count summary
(S2 audit #D1: a failed warn-gate must report WARN + n_passed < n_gates)."""
from __future__ import annotations

import argparse
import types
from pathlib import Path

from grainsmith.cli import (
    _build_parser,
    _cmd_generate,
    _cmd_info,
    _cmd_ui,
    _cmd_validate,
    _cmd_verify,
)


def _write_min_config(tmp_path, value: int = 7) -> str:
    out = (tmp_path / "out").as_posix()
    cfg = tmp_path / "c.yaml"
    template = (
        "seed: {mode: fixed, value: __VAL__}\n"
        "box: {lengths: [12.0, 12.0, 12.0]}\n"
        "grains: {number: 2}\n"
        "crystal:\n"
        "  space_group: {number: 225}\n"
        "  lattice: {a: 3.615}\n"
        "  wyckoff_sites: [{element: Cu, coords: [0.0, 0.0, 0.0]}]\n"
        "orientation: {scheme: random_uniform}\n"
        "analysis: {statistics: false}\n"
        'output: {directory: "__OUT__", methods_snippet: false}\n'
    )
    cfg.write_text(
        template.replace("__VAL__", str(value)).replace("__OUT__", out),
        encoding="utf-8",
    )
    return str(cfg)


# --- info ---------------------------------------------------------------

def test_info_happy(capsys):
    rc = _cmd_info(argparse.Namespace(sg=221, setting=None, site=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Space group : 221" in out and "Point group" in out


def test_info_site_orbit(capsys):
    rc = _cmd_info(argparse.Namespace(sg=225, setting=None, site="0.0,0.0,0.0"))
    out = capsys.readouterr().out
    assert rc == 0 and "positions" in out


def test_info_bad_sg(capsys):
    rc = _cmd_info(argparse.Namespace(sg=999, setting=None, site=None))
    assert rc == 1 and "Error" in capsys.readouterr().err


def test_info_bad_site(capsys):
    rc = _cmd_info(argparse.Namespace(sg=221, setting=None, site="0,0"))
    assert rc == 1 and "site" in capsys.readouterr().err


# --- validate -----------------------------------------------------------

def test_validate_happy(tmp_path, capsys):
    rc = _cmd_validate(argparse.Namespace(config=_write_min_config(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 0 and "G1 OK" in out and "G2 OK" in out
    assert out.strip().endswith("OK")


def test_validate_missing_file(tmp_path, capsys):
    rc = _cmd_validate(argparse.Namespace(config=str(tmp_path / "nope.yaml")))
    assert rc == 1 and "G1 FAIL" in capsys.readouterr().err


# --- generate -----------------------------------------------------------

def test_generate_happy(tmp_path, capsys):
    rc = _cmd_generate(
        argparse.Namespace(config=_write_min_config(tmp_path), jobs=1,
                           max_rss=None))
    out = capsys.readouterr().out
    assert rc == 0 and out.startswith("OK:") and "QA gates passed" in out


def test_generate_summary_warn_on_failed_gate(tmp_path, monkeypatch, capsys):
    """S2 #D1: a recorded (warn) gate failure must surface as WARN with
    n_passed < n_gates, not a false 'N/N passed'."""
    import grainsmith.pipeline as pipeline
    g_ok = types.SimpleNamespace(gate="G1", passed=True)
    g_bad = types.SimpleNamespace(gate="G9", passed=False)
    fake = types.SimpleNamespace(
        gates=types.SimpleNamespace(results=lambda: [g_ok, g_bad]),
        atoms=[0, 1, 2],
        tess=types.SimpleNamespace(n_grains=2),
        boundary_reports=[0],
        outdir=str(tmp_path),
        timings={"total": 1.0},
        peak_rss_bytes=None,
    )
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: fake)
    rc = _cmd_generate(
        argparse.Namespace(config=_write_min_config(tmp_path), jobs=1,
                           max_rss=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("WARN:") and "1/2 QA gates passed" in out and "G9" in out


# --- ui -------------------------------------------------------------------

def test_ui_not_included_this_release(capsys):
    """The interactive UI is not part of this release: `_cmd_ui` must not
    import grainsmith.ui, must return 1, and must say so on stderr."""
    rc = _cmd_ui(argparse.Namespace(ui_args=[]))
    err = capsys.readouterr().err
    assert rc == 1
    assert "not included in this release" in err
    assert "future version" in err


# --- parser -------------------------------------------------------------

def test_parser_requires_subcommand():
    import pytest
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_generate_directory_as_config_is_clean_error(tmp_path, capsys):
    """Passing a directory (or otherwise unreadable path) as the config must
    surface the CLI's clean one-line diagnostic, not a raw IsADirectoryError
    traceback."""
    rc = _cmd_generate(argparse.Namespace(
        config=str(tmp_path), jobs=1, max_rss=None))
    assert rc == 1
    assert "Configuration error:" in capsys.readouterr().err


def test_validate_directory_as_config_is_clean_error(tmp_path, capsys):
    rc = _cmd_validate(argparse.Namespace(config=str(tmp_path)))
    assert rc == 1
    assert "G1 FAIL:" in capsys.readouterr().err


def test_jobs_zero_respects_cpu_affinity(monkeypatch):
    """--jobs 0 must size the pool from the cores available to THIS process
    (the cgroup/affinity mask — e.g. a SLURM --cpus-per-task allocation),
    not the node's total core count."""
    from grainsmith import pipeline

    monkeypatch.setattr(pipeline.os, "sched_getaffinity",
                        lambda pid: {0, 1, 2}, raising=False)
    assert pipeline._available_cpus() == 3


def test_jobs_zero_falls_back_without_affinity(monkeypatch):
    """Platforms without sched_getaffinity (macOS/Windows) fall back to
    os.cpu_count()."""
    from grainsmith import pipeline

    monkeypatch.delattr(pipeline.os, "sched_getaffinity", raising=False)
    assert pipeline._available_cpus() == (pipeline.os.cpu_count() or 1)


# --- --version ------------------------------------------------------------

def test_version_flag_prints_version_and_exits_0(capsys):
    import pytest

    from grainsmith import __version__

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"grainsmith {__version__}"


# --- verify -----------------------------------------------------------

def _run_min_config(tmp_path):
    """A real run() so `grainsmith verify` has a MANIFEST.txt to check."""
    from grainsmith.config.resolve import load_config
    from grainsmith.pipeline import run

    config = load_config(Path(_write_min_config(tmp_path)))
    return run(config)


def test_verify_happy_path(tmp_path, capsys):
    result = _run_min_config(tmp_path)
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(result.outdir), strict=False, expect_version=None,
        quiet=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERIFIED:" in out


def test_verify_non_directory_is_exit_2(tmp_path, capsys):
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(tmp_path / "does_not_exist"), strict=False,
        expect_version=None, quiet=False))
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_verify_quiet_prints_exactly_one_line(tmp_path, capsys):
    result = _run_min_config(tmp_path)
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(result.outdir), strict=False, expect_version=None,
        quiet=True))
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("VERIFIED:")


def test_verify_tampered_run_is_exit_1(tmp_path, capsys):
    result = _run_min_config(tmp_path)
    with (result.outdir / "grains.csv").open("ab") as fh:
        fh.write(b"\n")
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(result.outdir), strict=False, expect_version=None,
        quiet=False))
    out = capsys.readouterr()
    assert rc == 1
    assert "FAILED:" in out.out
    assert "check(s) failed" in out.err


def test_verify_strict_warning_only_stderr_wording_is_self_consistent(
        tmp_path, capsys):
    """Under --strict with a stale extra file (a WARN, not a FAIL), the run
    exits 1 with zero actual FAILs -- the stderr line must not say "0
    check(s) failed" while exiting 1 (self-contradictory); it must instead
    say the failure is warnings promoted under --strict."""
    result = _run_min_config(tmp_path)
    (result.outdir / "stale.txt").write_text("leftover\n", encoding="utf-8")
    rc = _cmd_verify(argparse.Namespace(
        outdir=str(result.outdir), strict=True, expect_version=None,
        quiet=False))
    out = capsys.readouterr()
    assert rc == 1
    assert "FAILED:" in out.out
    assert "0 check(s) failed" not in out.err
    assert "--strict" in out.err


def test_verify_parser_accepts_flags():
    parser = _build_parser()
    ns = parser.parse_args(
        ["verify", "somedir", "--strict", "--expect-version", "1.1.0"])
    assert ns.command == "verify"
    assert ns.outdir == "somedir"
    assert ns.strict is True
    assert ns.expect_version == "1.1.0"


def test_verify_parser_defaults():
    parser = _build_parser()
    ns = parser.parse_args(["verify", "somedir"])
    assert ns.strict is False
    assert ns.expect_version is None
    assert ns.quiet is False
