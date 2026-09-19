"""Tests for the optional best-effort total-RSS monitor (memory.py).

Covers:
- peak tracking + WARN when a low ceiling is crossed,
- "ok" when under the ceiling, auto-ceiling from physical RAM,
- inert behaviour when psutil is unavailable,
- the monitor never perturbs the byte-identical pipeline output.
"""
import sys

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.memory import MemoryMonitor
from grainsmith.pipeline import run

psutil = pytest.importorskip("psutil")


def _config(outdir, **overrides):
    raw = {
        "meta": {"title": "mem", "verbose": 0},
        "seed": {"mode": "fixed", "value": 20260623},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    }
    for key, sub in overrides.items():
        raw[key] = {**raw.get(key, {}), **sub} if isinstance(sub, dict) else sub
    return resolve_config(raw)


# --- unit: MemoryMonitor ----------------------------------------------------


def test_monitor_tracks_peak_and_warns_on_low_ceiling():
    """A ~1 kB ceiling is always exceeded → tripped, WARN summary, peak>0."""
    mon = MemoryMonitor(max_rss_gb=1e-6).start()  # 1e-6 GB = 1 kB
    mon.stop()
    assert mon.available
    assert mon.peak_rss_bytes is not None and mon.peak_rss_bytes > 0
    assert mon.tripped
    assert not mon.auto_ceiling
    assert mon.summary().startswith("WARN")


def test_monitor_ok_when_under_ceiling():
    """A 10 PB ceiling is never crossed → not tripped, 'ok' summary."""
    mon = MemoryMonitor(max_rss_gb=1e7).start()  # 1e7 GB = 10 PB
    mon.stop()
    assert not mon.tripped
    assert mon.summary().startswith("ok")


def test_monitor_auto_ceiling_from_physical_ram():
    """No explicit ceiling → auto-ceiling derived from physical RAM."""
    mon = MemoryMonitor(max_rss_gb=None).start()
    mon.stop()
    assert mon.auto_ceiling
    assert mon.max_rss_bytes is not None
    assert mon.max_rss_bytes <= psutil.virtual_memory().total


def test_monitor_inert_without_psutil(monkeypatch):
    """psutil unimportable → monitor is inert and reports nothing."""
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> error
    mon = MemoryMonitor(max_rss_gb=1e-6).start()
    mon.stop()
    assert not mon.available
    assert mon.peak_rss_bytes is None
    assert mon.summary() is None
    mon.log_summary()  # must not raise


def test_monitor_idempotent_stop():
    mon = MemoryMonitor().start()
    mon.stop()
    mon.stop()  # second stop is a no-op, must not raise


# --- integration: run() -----------------------------------------------------


def test_run_records_peak_rss(tmp_path):
    res = run(_config(tmp_path / "out"))
    assert res.peak_rss_bytes is not None
    assert res.peak_rss_bytes > 0


def test_max_rss_does_not_change_output(tmp_path):
    """Determinism: a tripping ceiling must not alter a single output atom."""
    res_off = run(_config(tmp_path / "off"), max_rss_gb=None)
    res_trip = run(_config(tmp_path / "trip"), max_rss_gb=1e-6)
    assert res_trip.gates.all_passed()
    np.testing.assert_array_equal(res_off.atoms.pos, res_trip.atoms.pos)
    assert np.array_equal(res_off.atoms.species, res_trip.atoms.species)
    assert np.array_equal(res_off.atoms.grain, res_trip.atoms.grain)


def test_monitor_warns_in_real_time_at_first_crossing(caplog):
    """The soft-ceiling crossing must be WARNed the moment the sampler detects
    it — not only from log_summary() — so an impending OOM-kill is announced in
    run.log even when the run dies before reaching the wrap-up."""
    import logging
    with caplog.at_level(logging.WARNING, logger="grainsmith.memory"):
        mon = MemoryMonitor(max_rss_gb=1e-6).start()   # 1 kB ceiling → crosses at once
        mon.stop()
    assert mon.tripped
    assert any(rec.levelno == logging.WARNING for rec in caplog.records), (
        "no real-time WARNING emitted at the ceiling crossing"
    )
