"""Tests for benchmarks/_cell_runner.py — the disk-clean, resumable cell
harness the local benchmark suite builds on.

Deliberately covers only the PURE logic (config construction, scratch-path
safety, CSV append/resume, progress math) with no ``grainsmith generate``
subprocess — those are exercised for real by a manual, not-CI, single-cell
smoke run and by the user's own sweeps. Keeping this file subprocess-free
is deliberate: it must stay fast enough to run on every ``pytest tests/``
invocation.

The ``benchmarks/`` directory is local, optional tooling and is not part
of a public checkout of this repository; this module skips itself
entirely when that directory isn't present, so plain ``pytest`` collection
never fails because of it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"


def _load_cell_runner():
    """Import benchmarks/_cell_runner.py (not an installed package) — same
    sys.path-injection pattern tests/test_docs.py uses for tools/*.py."""
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    spec = importlib.util.find_spec("_cell_runner")
    if spec is None:
        pytest.skip("benchmarks/ directory not present in this checkout",
                    allow_module_level=True)
    import _cell_runner as cr
    return cr


cr = _load_cell_runner()


# --- FCC Cu config (every N1..N5 cell is FCC Cu, SG225) --------------------


def test_fcc_cu_config_pins_material():
    cfg = cr.fcc_cu_config(100.0, 12, seed=7, output_directory="/tmp/x")
    assert cfg["crystal"]["space_group"]["number"] == 225
    assert cfg["crystal"]["lattice"]["a"] == pytest.approx(3.615)
    sites = cfg["crystal"]["wyckoff_sites"]
    assert len(sites) == 1
    assert sites[0]["element"] == "Cu"
    assert sites[0]["letter"] == "a"
    assert cfg["box"]["lengths"] == [100.0, 100.0, 100.0]
    assert cfg["grains"]["number"] == 12
    assert cfg["seed"] == {"mode": "fixed", "value": 7}


def test_fcc_cu_config_resolves_against_real_schema():
    """The dict round-trips through the actual RunConfig validator (gate
    G1) — catches a schema-field typo in fcc_cu_config() itself, not just
    a hand-checked assertion above."""
    from grainsmith.config.resolve import resolve_config

    cfg = cr.fcc_cu_config(80.0, 6, seed=1, output_directory="/tmp/x")
    resolved = resolve_config(cfg)
    assert resolved.grains.number == 6


def test_fcc_cu_config_accepts_overrides():
    cfg = cr.fcc_cu_config(
        450.0, 10, seed=1, output_directory="/tmp/x",
        boundaries={
            "geometry": "curved",
            "curved": {"method": "anisotropic",
                       "aspect_ratio_range": [1.0, 2.0]},
            "overlap_removal": {"enabled": True, "cutoff": "0.85*d_nn",
                                "policy": "delete_shallower"},
        },
    )
    assert cfg["boundaries"]["curved"]["method"] == "anisotropic"
    from grainsmith.config.resolve import resolve_config
    resolve_config(cfg)  # must still validate (gate G1)


def test_estimate_fcc_cu_atom_count_matches_lattice_geometry():
    """4 atoms per (3.615 Å)^3 conventional cell — checked against the
    closed-form volume ratio directly (independent of the helper's own
    arithmetic)."""
    for L in (50.0, 150.0, 300.0, 600.0):
        expected = round(L ** 3 / (3.615 ** 3) * 4)
        assert cr.estimate_fcc_cu_atom_count(L) == expected
    # Monotonic in box size.
    counts = [cr.estimate_fcc_cu_atom_count(L)
              for L in (50.0, 150.0, 300.0, 600.0)]
    assert counts == sorted(counts)


# --- scratch_root: must NEVER resolve into the repo tree -------------------


def test_scratch_root_default_is_absolute_and_outside_repo():
    root = cr.scratch_root()
    assert root.is_absolute(), (
        f"scratch_root() returned a relative path ({root!r}) — a cwd of "
        "the repo root would put transient benchmark output INSIDE the "
        "repo tree, violating the disk-clean harness contract.")
    assert not str(root).startswith(str(REPO)), (
        f"scratch_root() ({root!r}) resolved under the repo tree ({REPO!r})")


def test_scratch_root_repo_opt_in(monkeypatch):
    monkeypatch.setenv("GRAINSMITH_BENCH_SCRATCH", "repo")
    root = cr.scratch_root()
    assert root == BENCH_DIR / "results-dev" / "_scratch"
    assert root.is_dir()  # created idempotently


def test_absolute_system_tmp_rejects_relative_tmpdir(monkeypatch):
    """A relative $TMPDIR must NOT be trusted verbatim (it would resolve
    against the CALLING PROCESS's cwd at mkdtemp-time, not a fixed
    location) — this is the exact failure mode _absolute_system_tmp()
    exists to avoid; see its docstring."""
    monkeypatch.setenv("TMPDIR", "./relative_tmp_should_be_rejected")
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    result = cr._absolute_system_tmp()
    assert result.is_absolute()
    assert "relative_tmp_should_be_rejected" not in str(result)


def test_absolute_system_tmp_accepts_absolute_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert cr._absolute_system_tmp() == tmp_path


# --- CellCsvWriter: append + resume + schema guard --------------------------


HEADER = ["cell_id", "box_l", "wall_s", "ok"]


def test_csv_writer_append_and_resume(tmp_path):
    path = tmp_path / "cells.csv"
    with cr.CellCsvWriter(path, HEADER) as w:
        assert w.completed_ids() == set()
        w.append_row({"cell_id": "a", "box_l": 50.0, "wall_s": 1.0,
                     "ok": True})
        w.append_row({"cell_id": "b", "box_l": 100.0, "wall_s": 2.0,
                     "ok": True})

    # Re-open (simulating a resumed process) — must see prior rows without
    # re-running them, and append cleanly after.
    with cr.CellCsvWriter(path, HEADER) as w2:
        assert w2.completed_ids() == {"a", "b"}
        w2.append_row({"cell_id": "c", "box_l": 150.0, "wall_s": 3.0,
                      "ok": False})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == ",".join(HEADER)
    assert len(lines) == 4  # header + 3 rows
    assert lines[-1].startswith("c,150")


def test_csv_writer_rejects_schema_drift(tmp_path):
    path = tmp_path / "cells.csv"
    with cr.CellCsvWriter(path, HEADER):
        pass
    with pytest.raises(ValueError, match="schema changed"):
        cr.CellCsvWriter(path, ["cell_id", "box_l"])  # dropped columns


def test_csv_writer_requires_cell_id_column(tmp_path):
    with pytest.raises(ValueError, match="cell_id"):
        cr.CellCsvWriter(tmp_path / "x.csv", ["box_l", "wall_s"])


def test_csv_writer_flushes_durably(tmp_path):
    """append_row must be visible to an INDEPENDENT reader immediately
    (flush+fsync, not buffered) — the durability half of the disk-clean
    contract (a killed sweep keeps every completed row)."""
    path = tmp_path / "cells.csv"
    w = cr.CellCsvWriter(path, HEADER)
    w.append_row({"cell_id": "only", "box_l": 1.0, "wall_s": 0.1,
                 "ok": True})
    # Do NOT close w — read the file from a second, independent handle.
    content = path.read_text(encoding="utf-8")
    assert "only" in content
    w.close()


# --- repeat_stats / suggest_repeats -----------------------------------------


def _fake_result(wall_s: float, ok: bool = True) -> cr.CellResult:
    return cr.CellResult(
        cell_id="x", ok=ok, returncode=0, timed_out=False, wall_s=wall_s,
        cpu_s=wall_s, peak_rss_gb=0.1, pipeline_total_s=wall_s,
        n_atoms=100, gates_passed=5, gates_total=5, transient_disk_mb=1.0,
        error=None, stdout_tail="",
    )


def test_repeat_stats_basic():
    results = [_fake_result(1.0), _fake_result(2.0), _fake_result(3.0)]
    stats = cr.repeat_stats(results)
    assert stats["n"] == 3
    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == pytest.approx(2.0)


def test_repeat_stats_excludes_failed_runs():
    results = [_fake_result(1.0), _fake_result(100.0, ok=False)]
    stats = cr.repeat_stats(results)
    assert stats["n"] == 1
    assert stats["mean"] == 1.0


def test_suggest_repeats_low_variance_needs_one():
    results = [_fake_result(1.00), _fake_result(1.01), _fake_result(0.99)]
    assert cr.suggest_repeats(results) == 1


def test_suggest_repeats_high_variance_needs_more():
    results = [_fake_result(1.0), _fake_result(3.0), _fake_result(0.2)]
    assert cr.suggest_repeats(results) >= 3


def test_suggest_repeats_single_result_defaults_to_one():
    assert cr.suggest_repeats([_fake_result(1.0)]) == 1


# --- Progress ETA math -------------------------------------------------------


def test_progress_tick_reports_monotonic_done_count(capsys):
    prog = cr.Progress(total=3, label="test")
    for i in range(3):
        prog.tick(_fake_result(1.0))
        assert prog.done == i + 1
    out = capsys.readouterr().out
    assert "3/3" in out
