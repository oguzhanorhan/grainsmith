"""Tests for STEP 3b: pipeline wiring of the volume-weighted ODF trust
region into ``anneal_assignment`` (grain_volumes / odf_drift_max) and the
three new QA gates it enables (G22 odf fidelity, G23 sigma3 consistency,
G24 final post-warp volume fidelity).

Follows the style of tests/test_odf_mdf.py (`_e2e_config`-shaped resolved
configs, `_sym_cubic`/`_rng` fixtures) and tests/test_end_to_end.py
(`_csv_rows` summary.csv reader).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.crystal.cell import cell_matrix
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import hall_from_international, symmetry_ops
from grainsmith.orientation import odf
from grainsmith.orientation.mdf import anneal_assignment, build_bins_and_target
from grainsmith.orientation.samplers import random_uniform
from grainsmith.qa import gate_g22_odf_fidelity


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _sym_cubic() -> np.ndarray:
    A = cell_matrix(3.0, 3.0, 3.0, 90.0, 90.0, 90.0)
    rots, _ = symmetry_ops(hall_from_international(221))
    return proper_rotation_quaternions(A, rots)


def _ring_pairs(n: int, k: int = 3) -> list[tuple[int, int]]:
    pairs = set()
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


class _MdfCfg:
    """Duck-typed stand-in for MdfTargetConfig (matches test_odf_mdf.py)."""
    def __init__(self, **kw):
        self.type = kw.get("type", "mackenzie")
        self.sigma3_fraction = kw.get("sigma3_fraction", 0.3)
        self.bin_edges = kw.get("bin_edges")
        self.densities = kw.get("densities")


def _csv_rows(path):
    import csv
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Resolved-config helpers
# ---------------------------------------------------------------------------


def _mdf_config(outdir, *, n_grains: int = 12, annealing_steps: int = 4000,
                odf_drift_max: float | None = None, csl: bool = False,
                seed: int = 4242):
    """Single-phase cubic Cu, orientation.mdf_target active."""
    mdf_target: dict = {
        "type": "sigma3_angle_enriched",
        "sigma3_fraction": 0.4,
        "annealing_steps": annealing_steps,
        "chi2_max": 2.0,
    }
    if odf_drift_max is not None:
        mdf_target["odf_drift_max"] = odf_drift_max
    return resolve_config({
        "meta": {"title": "g22/g23 e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [36.0, 36.0, 36.0]},
        "grains": {"number": n_grains},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "random_uniform", "mdf_target": mdf_target},
        "analysis": {"csl": csl, "mdf_bins": 30},
        "output": {"directory": str(outdir)},
    })


def _no_mdf_target_config(outdir, seed: int = 4242, n_grains: int = 12):
    """Same shape as _mdf_config but WITHOUT orientation.mdf_target -- the
    golden-config style case (no `mdf_res`, no G22 row)."""
    return resolve_config({
        "meta": {"title": "no-mdf-target e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [36.0, 36.0, 36.0]},
        "grains": {"number": n_grains},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir)},
    })


def _warp_size_distribution_config(outdir, n_grains: int = 4, seed: int = 1):
    """Warp on a volume-fitted power base -- the calibrated G24 case
    (60 A box, equal targets, amplitude 0.6, correlation_length 10)."""
    return resolve_config({
        "meta": {"title": "g24 e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [60.0, 60.0, 60.0]},
        "grains": {"number": n_grains,
                   "size_distribution": {"type": "equal"}},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": {"geometry": "curved",
                       "curved": {"method": "warp", "amplitude": 0.6,
                                  "correlation_length": 10.0}},
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    })


def _flat_size_distribution_config(outdir, n_grains: int = 6, seed: int = 1):
    """flat + size_distribution: sdot_res.tess IS tess -- G24 must not
    fire (it would double-report G11)."""
    return resolve_config({
        "meta": {"title": "g24 flat e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": n_grains,
                   "size_distribution": {"type": "lognormal",
                                         "sigma_log": 0.3}},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir),
                   "lammps": {"atom_style": "molecular"}},
    })


# ---------------------------------------------------------------------------
# H.1 -- _stage_mdf's volumes equal the analysis stage's volumes
# ---------------------------------------------------------------------------


def test_stage_mdf_volumes_match_analysis_flat(tmp_path, monkeypatch):
    """The exact array passed as `grain_volumes` to anneal_assignment
    inside _stage_mdf equals grain_volumes(tess, L) computed AFTER the
    full run (same tess, unmutated in between) -- flat geometry."""
    import grainsmith.orientation.mdf as mdf_mod
    from grainsmith.analysis.grains import grain_volumes
    from grainsmith.pipeline import run

    captured = {}
    orig = mdf_mod.anneal_assignment

    def spy(*args, **kwargs):
        captured["grain_volumes"] = kwargs.get("grain_volumes")
        return orig(*args, **kwargs)

    monkeypatch.setattr(mdf_mod, "anneal_assignment", spy)

    cfg = _mdf_config(tmp_path / "out")
    res = run(cfg)
    assert res.gates.all_passed()
    assert captured["grain_volumes"] is not None

    L = np.asarray(cfg.box.lengths, dtype=np.float64)
    post_volumes = grain_volumes(res.tess, L)
    np.testing.assert_array_equal(captured["grain_volumes"], post_volumes)


def test_stage_mdf_volumes_match_analysis_warp(tmp_path, monkeypatch):
    """Same invariant as above, curved (warp) geometry, INCLUDING an
    explicit analysis.voxel_grid override -- the case _stage_mdf's own
    comment calls out (`run()` primes _analysis_voxel_grid before this
    stage precisely so the two consumers share tess._voxel)."""
    import grainsmith.orientation.mdf as mdf_mod
    from grainsmith.pipeline import _analysis_voxel_grid, run

    captured = {}
    orig = mdf_mod.anneal_assignment

    def spy(*args, **kwargs):
        captured["grain_volumes"] = kwargs.get("grain_volumes")
        return orig(*args, **kwargs)

    monkeypatch.setattr(mdf_mod, "anneal_assignment", spy)

    mdf_target = {
        "type": "haar_random",
        "annealing_steps": 500,
    }
    cfg = resolve_config({
        "meta": {"title": "g22 warp e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": 7},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 10},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "random_uniform", "mdf_target": mdf_target},
        "boundaries": {"geometry": "curved",
                       "curved": {"method": "warp", "amplitude": 0.3,
                                  "correlation_length": 8.0}},
        "analysis": {"voxel_grid": 24},
        "output": {"directory": str(tmp_path / "out")},
    })
    res = run(cfg)
    assert res.gates.all_passed()
    assert captured["grain_volumes"] is not None

    post_volumes = _analysis_voxel_grid(res.config, res.tess).volumes()
    np.testing.assert_array_equal(captured["grain_volumes"], post_volumes)


def test_stage_orientation_volumes_match_analysis_warp_odf_components(
        tmp_path, monkeypatch):
    """Extends the test_stage_mdf_volumes_match_analysis_warp pattern to the
    ORIENTATION stage: with orientation.scheme 'odf_components' and
    component_weight_basis 'volume' on curved (warp) geometry with an
    explicit analysis.voxel_grid, the `grain_volumes` array `odf_components`
    receives (spied via grainsmith.orientation.odf_components, the package
    binding `_stage_orientation`'s local import resolves against) must equal
    the analysis stage's volumes -- i.e. the priming hoisted in front of the
    orientation stage (contract 5a) actually reaches the sampler with the
    SAME array every later consumer uses, not a separately-built one."""
    import grainsmith.orientation as orientation_pkg
    from grainsmith.pipeline import _analysis_voxel_grid, run

    captured = {}
    orig = orientation_pkg.odf_components

    def spy(*args, **kwargs):
        captured["grain_volumes"] = kwargs.get("grain_volumes")
        captured["weight_basis"] = kwargs.get("weight_basis")
        return orig(*args, **kwargs)

    monkeypatch.setattr(orientation_pkg, "odf_components", spy)

    cfg = resolve_config({
        "meta": {"title": "g25 warp orientation-volumes e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": 7},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 10},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {
            "scheme": "odf_components",
            "components": [
                {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 0.5,
                 "spread_deg": 4.0},
                {"euler_bunge_deg": [45.0, 70.528779365509308, 45.0],
                 "weight": 0.5, "spread_deg": 4.0},
            ],
            "component_weight_basis": "volume",
        },
        "boundaries": {"geometry": "curved",
                       "curved": {"method": "warp", "amplitude": 0.3,
                                  "correlation_length": 8.0}},
        "analysis": {"voxel_grid": 24},
        "output": {"directory": str(tmp_path / "out")},
    })
    res = run(cfg)
    assert res.gates.all_passed()
    assert captured["weight_basis"] == "volume"
    assert captured["grain_volumes"] is not None

    post_volumes = _analysis_voxel_grid(res.config, res.tess).volumes()
    np.testing.assert_array_equal(captured["grain_volumes"], post_volumes)


# ---------------------------------------------------------------------------
# H.2/H.3 -- e2e G22 presence, message content, drift-cap + summary rows
# ---------------------------------------------------------------------------


def test_e2e_g22_present_and_ok(tmp_path):
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out"))
    assert res.gates.all_passed()
    g22 = [r for r in res.gates.results() if r.gate == "G22"]
    assert len(g22) == 1
    assert g22[0].passed is True
    msg = g22[0].message
    assert "n_eff=" in msg
    assert "drift=" in msg


def test_e2e_g22_drift_cap_and_summary_rows(tmp_path):
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out", odf_drift_max=0.02,
                          annealing_steps=6000))
    assert res.gates.all_passed()
    g22 = [r for r in res.gates.results() if r.gate == "G22"][0]
    assert g22.passed is True

    m = re.search(r"drift=([0-9.eE+-]+) \(cap ([0-9.eE+-]+)\)", g22.message)
    assert m is not None, g22.message
    drift_val = float(m.group(1))
    cap_val = float(m.group(2))
    assert cap_val == pytest.approx(0.02)
    assert drift_val <= 0.02 + 1e-6, g22.message

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert ("mdf", "odf_drift_final") in by_key
    assert float(by_key[("mdf", "odf_drift_final")]) <= 0.02 + 1e-6
    assert by_key[("mdf", "odf_drift_max")] == "0.02"
    assert ("mdf", "n_drift_vetoed") in by_key
    assert int(by_key[("mdf", "n_drift_vetoed")]) >= 0
    assert by_key["mdf", "objective"] == "area_weighted_disorientation_angle"
    assert by_key["mdf", "odf_constraint"] == "tv_cap"
    assert float(by_key["mdf", "odf_kernel_halfwidth_requested_deg"]) == 10.0
    assert float(by_key["mdf", "odf_kernel_degree"]) == 91.0
    effective = float(by_key["mdf", "odf_kernel_halfwidth_effective_deg"])
    assert 9.99 < effective < 10.0
    assert "requested_halfwidth=10 deg" in g22.message
    assert "kappa=91" in g22.message


def test_e2e_g22_target_type_canonical_row(tmp_path):
    """summary.csv records the CANONICAL target type even though the
    config wrote the same (already-canonical) name here -- a direct pin
    of the row's presence/format; the alias-normalization itself is
    covered by tests/test_mdf_target_naming.py."""
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out"))
    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert by_key[("mdf", "target_type_canonical")] == "sigma3_angle_enriched"


# ---------------------------------------------------------------------------
# H.4 -- G22's re-measured drift agrees with MDFResult.odf_drift_final
# ---------------------------------------------------------------------------


def test_g22_measured_drift_matches_mdf_result_to_1e9():
    """Unit-level (qa.py called directly): with the kernel block forced to
    skip via a zero memory budget, gate_g22_odf_fidelity's `measured` IS
    the re-measured atomic drift -- pin it against MDFResult.
    odf_drift_final (the annealer's own claim) to 1e-9, and independently
    against the raw orientation.odf primitives."""
    sym = _sym_cubic()
    n = 40
    quats = random_uniform(n, _rng(5))
    pairs = _ring_pairs(n, k=3)
    pair_areas = _rng(6).lognormal(0.0, 0.5, len(pairs))
    volumes = _rng(7).lognormal(0.0, 0.7, n)
    cfg = _MdfCfg(type="csl_enriched", sigma3_fraction=0.4)
    edges, target, ref = build_bins_and_target(cfg, sym, 30)

    mdf_res = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym,
        n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(8),
        grain_volumes=volumes, odf_drift_max=None,
    )
    assert mdf_res.odf_drift_final is not None
    quats_final = quats[mdf_res.permutation]

    result = gate_g22_odf_fidelity(
        volumes, mdf_res.permutation, quats_final, sym,
        mdf_res.odf_drift_final, None, 10.0, 32,
        memory_limit_bytes=0.0, memory_limit_source="test",
    )
    assert result.passed is True
    assert "NOT evaluated" in result.message   # kernel block was skipped
    assert result.measured == pytest.approx(mdf_res.odf_drift_final, abs=1e-9)

    # Independent cross-check via the raw primitives (bypassing the gate
    # entirely), mirroring test_odf_mdf.py's own drift-final pin.
    w = volumes / float(np.sum(volumes))
    class_of, n_classes = odf.orientation_classes(quats_final)
    v_initial = w[mdf_res.permutation]
    independent = odf.atomic_drift(w, v_initial, class_of, n_classes)
    assert result.measured == pytest.approx(independent, abs=1e-12)


def test_g22_memory_guard_skips_kernel_block_when_over_budget():
    """The §13 memory pre-check must SKIP the kernel block (never call
    `symmetrized_gram`, never raise) whenever the estimated Gram-matrix
    size exceeds the modest share of the resolved budget this diagnostic
    is allowed -- the whole point of a WARN-only gate never aborting an
    otherwise-successful run.

    This branch could not be exercised END-TO-END through a real
    pipeline.run(): lowering `runtime.memory_limit_gb` far enough trips
    the FILL-stage guard (`fill_grain`'s lattice grid) long before the
    analysis stage's G22 call is ever reached, and reaching this specific
    guard with a real grain count would need on the order of 7000 grains.
    So this is a direct unit test on `gate_g22_odf_fidelity` -- the only
    way to reach the branch at a sane test size -- covering every part of
    the "never abort a run" contract explicitly:

    1. the call returns normally (no exception reaches the caller, the
       point of wrapping this whole function in its own try/except);
    2. `passed` is True (a diagnostic, never a hard gate, even skipped);
    3. the message SAYS the kernel block was skipped and WHY, naming both
       N (the orientation count) and the estimated size in GB, plus the
       memory_limit_source/budget it was compared against; and
    4. the exact quantities that do NOT depend on the kernel block --
       n_eff, the count-vs-volume gap, and the re-measured drift -- are
       STILL computed and reported, unaffected by the skip.
    """
    sym = _sym_cubic()
    n = 30
    quats = random_uniform(n, _rng(1))
    volumes = _rng(2).lognormal(0.0, 0.5, n)
    perm0 = np.arange(n, dtype=np.intp)

    # 1 byte: the true Gram-matrix estimate (8*n*n bytes) exceeds this by
    # many orders of magnitude regardless of n, so the skip is forced
    # deterministically without needing a huge n to reach it "for real".
    result = gate_g22_odf_fidelity(
        volumes, perm0, quats, sym, 0.0, None, 10.0, 32,
        memory_limit_bytes=1.0,
        memory_limit_source="test-budget",
    )

    # (1) returned normally (reaching this line at all proves no
    #     exception escaped) and (2) never hard-fails.
    assert result.passed is True

    # (3) the skip reason names N, the estimated size, and the budget.
    assert "kernel discrepancy NOT evaluated" in result.message
    assert f"N={n}" in result.message
    assert re.search(r"estimated at [0-9.]+ GB", result.message)
    assert "test-budget" in result.message
    assert "memory budget" in result.message
    # the kernel numbers must be ABSENT -- they were never computed
    assert "mmd_vs_count" not in result.message

    # (4) n_eff / gap / drift are still reported, with real, finite values.
    m_n_eff = re.search(r"n_eff=([0-9.]+) \(of 30 grains\)", result.message)
    m_gap = re.search(r"count-vs-volume gap=([0-9.]+)", result.message)
    m_drift = re.search(r"drift=([0-9.]+) \(no cap\)", result.message)
    assert m_n_eff is not None, result.message
    assert m_gap is not None, result.message
    assert m_drift is not None, result.message
    n_eff_val = float(m_n_eff.group(1))
    gap_val = float(m_gap.group(1))
    drift_val = float(m_drift.group(1))
    assert 1.0 <= n_eff_val <= n
    assert 0.0 <= gap_val < 1.0
    assert drift_val == 0.0   # identity perm0, no cap -> drift is exact 0

    # measured falls back to `drift` (the kernel value never existed).
    assert result.measured == pytest.approx(drift_val, abs=1e-9)


# ---------------------------------------------------------------------------
# H.7 -- a run WITHOUT mdf_target has no G22 row
# ---------------------------------------------------------------------------


def test_e2e_no_mdf_target_no_g22_row(tmp_path):
    from grainsmith.pipeline import run

    res = run(_no_mdf_target_config(tmp_path / "out"))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G22" not in gate_ids
    summary = (res.outdir / "summary.csv").read_text(encoding="utf-8")
    assert "G22" not in summary
    assert "mdf,odf_drift_final" not in summary


# ---------------------------------------------------------------------------
# H.5 -- G23 for single-phase runs, CSL on/off
# ---------------------------------------------------------------------------


def test_e2e_g23_csl_off_reports_not_evaluated(tmp_path):
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out", csl=False))
    g23 = [r for r in res.gates.results() if r.gate == "G23"]
    assert len(g23) == 1
    assert g23[0].passed is True
    assert "NOT EVALUATED" in g23[0].message
    # never a false "0.0" standing in for "not evaluated"
    assert "CSL sigma3 area fraction 0.0000" not in g23[0].message

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert ("sigma3", "angle_window_area_fraction") in by_key
    assert by_key[("sigma3", "angle_window_area_fraction")] != ""
    assert by_key[("sigma3", "csl_area_fraction")] == ""


def test_e2e_g23_csl_on_both_present_window_ge_csl(tmp_path):
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out", csl=True))
    g23 = [r for r in res.gates.results() if r.gate == "G23"][0]
    assert g23.passed is True
    assert "NOT EVALUATED" not in g23.message

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    window = float(by_key[("sigma3", "angle_window_area_fraction")])
    csl_cell = by_key[("sigma3", "csl_area_fraction")]
    assert csl_cell != ""
    csl = float(csl_cell)
    assert 0.0 <= csl <= window + 1e-9   # the angular window is a superset


def test_g23_never_trips_regardless_of_content():
    """passed is always True and the message never says WARN, even for a
    deliberately extreme case (all boundaries in the window, none truly
    Sigma3)."""
    from grainsmith.qa import gate_g23_sigma3_consistency

    ang = np.full(20, 60.0)
    area = np.ones(20)
    csl_sigma = ["" for _ in range(20)]   # angle-only match, never CSL
    result = gate_g23_sigma3_consistency(ang, area, csl_sigma)
    assert result.passed is True
    assert "WARN" not in result.message
    window_frac, csl_frac = result.measured
    assert window_frac == pytest.approx(1.0)
    assert csl_frac == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H.6 -- G24 fires only for warp + size_distribution
# ---------------------------------------------------------------------------


def test_g24_absent_for_flat_size_distribution(tmp_path):
    """sdot_res.tess IS tess for flat + size_distribution -- G24 must not
    fire (it would double-report G11 on identical volumes)."""
    from grainsmith.pipeline import run

    res = run(_flat_size_distribution_config(tmp_path / "out"))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    assert "G11" in gate_ids
    assert "G24" not in gate_ids


def test_g24_present_for_warp_size_distribution_plausible_range(tmp_path):
    """warp + size_distribution: sdot_res.tess is the unwarped base,
    distinct from the final (warped) tess -- G24 fires, with a measured
    value in the calibrated 4-grain amplitude-0.6 range (constants.
    WARP_VOLUME_G24_TOL's own docstring: ~0.02 measured, well under the
    0.10 tolerance)."""
    from grainsmith.constants import WARP_VOLUME_G24_TOL
    from grainsmith.pipeline import run

    res = run(_warp_size_distribution_config(tmp_path / "out"))
    assert res.gates.all_passed()
    g24 = [r for r in res.gates.results() if r.gate == "G24"]
    assert len(g24) == 1
    assert g24[0].passed is True
    assert 0.0 < g24[0].measured < WARP_VOLUME_G24_TOL
    assert "UNWARPED-base" in g24[0].message

    srows = _csv_rows(res.outdir / "summary.csv")
    g24_rows = [r for r in srows if r["section"] == "gates" and r["key"] == "G24"]
    assert len(g24_rows) == 1


# ---------------------------------------------------------------------------
# G22's null test must be TWO-SIDED (regression: an earlier one-sided
# `mmd_vs_count > null_p95` check read an anomalously LOW quantile as "ok",
# exactly backwards -- a real reference-scenario run (24 grains, lognormal
# sigma_log=0.5, two-component twin ODF, sigma3_angle_enriched) landed the
# UNCAPPED anneal's mmd_vs_count at a 2.7% quantile while a tightly-capped
# rerun of the SAME config landed at 82.4% -- both extremes are evidence of
# an assignment-induced orientation-size correlation, and only the upper
# one would have tripped the old one-sided condition.
# ---------------------------------------------------------------------------


def _worked_example_setup():
    """The exact 60-grain lognormal setup pinned in gate_g22_odf_fidelity's
    own docstring worked example: SAME orientation set / adjacency / grain
    volumes for both the uncapped and odf_drift_max=0.02 anneals below."""
    sym = _sym_cubic()
    n = 60
    quats = random_uniform(n, _rng(101))
    pairs = _ring_pairs(n, k=3)
    pair_areas = _rng(201).lognormal(0.0, 0.5, len(pairs))
    cfg = _MdfCfg(type="csl_enriched", sigma3_fraction=0.5)
    edges, target, ref = build_bins_and_target(cfg, sym, 30)
    volumes = _rng(301).lognormal(0.0, 0.8, n)
    return sym, quats, pairs, pair_areas, edges, target, ref, volumes


def _measure_g22(sym, quats, pairs, pair_areas, edges, target, ref, volumes,
                 odf_drift_max):
    mdf_res = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym,
        n_steps=3000, t0=0.05, cooling=0.995, rng=_rng(401),
        grain_volumes=volumes, odf_drift_max=odf_drift_max,
    )
    quats_final = quats[mdf_res.permutation]
    result = gate_g22_odf_fidelity(
        volumes, mdf_res.permutation, quats_final, sym,
        mdf_res.odf_drift_final, odf_drift_max, 10.0, 256,
        memory_limit_bytes=1e12, memory_limit_source="test",
    )
    return mdf_res, result


def _parse_kernel_line(message: str):
    """(mmd_vs_count, p2_5, p97_5, empirical_quantile_pct) parsed straight
    off gate_g22_odf_fidelity's own message -- lets a test assert the
    VERDICT was decided on the VALUE against the SAME interval the message
    prints, the exact framing item under regression here (a real run once
    printed a value below its own stated lower bound while asserting
    "inside the interval" -- see qa.py's "VALUE VS. QUANTILE" docstring
    section)."""
    m = re.search(
        r"mmd_vs_count=([0-9.eE+-]+) vs\. central 95% interval "
        r"\[([0-9.eE+-]+), ([0-9.eE+-]+)\].*empirical quantile "
        r"([0-9.]+)%", message)
    assert m is not None, message
    return tuple(float(g) for g in m.groups())


def test_g22_worked_example_uncapped_lands_in_low_tail_and_warns():
    """The docstring's own worked example, pinned: the UNCAPPED anneal has
    the LARGER atomic drift (~0.40) yet its mmd_vs_count (0.6329) lands
    BELOW the null's own 2.5th percentile (0.6389) -- a one-sided
    `mmd_vs_count > null_p97_5` check would have missed this entirely and
    reported "ok"; the two-sided, VALUE-based check must WARN."""
    setup = _worked_example_setup()
    mdf_res, result = _measure_g22(*setup, odf_drift_max=None)

    assert "NOT evaluated" not in result.message   # kernel block ran
    assert result.passed is True                   # G22 never hard-fails
    assert "WARN" in result.message
    m = re.search(r"drift=([0-9.eE+-]+) \(no cap\)", result.message)
    assert m is not None, result.message
    assert float(m.group(1)) == pytest.approx(0.4024, abs=5e-4)

    mmd_vs_count, p2_5, p97_5, quantile_pct = _parse_kernel_line(result.message)
    assert mmd_vs_count == pytest.approx(0.6329, abs=5e-4)
    assert p2_5 == pytest.approx(0.6389, abs=5e-4)
    assert p97_5 == pytest.approx(0.7555, abs=5e-4)
    # the verdict is decided on THIS comparison, not on the quantile below
    assert mmd_vs_count < p2_5
    assert quantile_pct < 2.5, (
        f"expected a low-tail empirical quantile too (< 2.5%), got "
        f"{quantile_pct}%: {result.message}")
    assert "OUTSIDE the interval" in result.message


def test_g22_worked_example_capped_lands_mid_range_and_ok():
    """The SAME setup, capped at odf_drift_max=0.02: drift collapses
    (~0.02) and mmd_vs_count (0.6907) lands comfortably inside
    [p2.5, p97.5] -- the boring, expected "nothing resolvable" outcome,
    correctly "ok"."""
    setup = _worked_example_setup()
    mdf_res, result = _measure_g22(*setup, odf_drift_max=0.02)

    assert "NOT evaluated" not in result.message
    assert result.passed is True
    assert "WARN" not in result.message
    m = re.search(r"drift=([0-9.eE+-]+) \(cap 0.02\)", result.message)
    assert m is not None, result.message
    assert float(m.group(1)) <= 0.02 + 1e-6

    mmd_vs_count, p2_5, p97_5, quantile_pct = _parse_kernel_line(result.message)
    assert mmd_vs_count == pytest.approx(0.6907, abs=5e-4)
    assert p2_5 == pytest.approx(0.6423, abs=5e-4)
    assert p97_5 == pytest.approx(0.7683, abs=5e-4)
    assert p2_5 <= mmd_vs_count <= p97_5
    assert 2.5 <= quantile_pct <= 97.5, (
        f"expected a mid-range empirical quantile too, got "
        f"{quantile_pct}%: {result.message}")
    assert "inside the interval" in result.message


def test_g22_two_sided_via_monkeypatched_null(monkeypatch):
    """A more surgical unit test on the two-sided BRANCH LOGIC itself,
    decoupled from needing a lucky real low-tail coincidence (unlike the
    two worked-example tests above, which rely on one): monkeypatch
    `orientation.odf.null_mmd` to return a FIXED synthetic null array, so
    the REAL (not mocked) `mmd_vs_count` -- computed from a real Gram
    matrix -- deterministically lands outside it on both sides.

    Patches the module attribute rather than the gate's return value
    directly: `gate_g22_odf_fidelity` does
    ``from grainsmith.orientation.odf import (..., null_mmd, ...)`` as a
    LOCAL import inside its own body, which resolves the current value of
    `orientation.odf.null_mmd` at CALL time -- so patching the module
    attribute before each call is exactly what is needed, and confirms the
    gate does not cache an earlier import.
    """
    import grainsmith.orientation.odf as odf_mod

    sym = _sym_cubic()
    n = 30
    quats = random_uniform(n, _rng(3))
    volumes = _rng(4).lognormal(0.0, 0.5, n)
    perm0 = np.arange(n, dtype=np.intp)

    # A null entirely ABOVE any realistic mmd_vs_count for this setup
    # (real kernel-space distances here are O(1), not O(50)) drives the
    # quantile to 0% -- the LOW-tail case a one-sided `> null_p95` check
    # could never flag, because closeness to the null's lower reaches
    # makes mmd_vs_count SMALL, never large.
    monkeypatch.setattr(odf_mod, "null_mmd",
                        lambda *a, **k: np.linspace(50.0, 100.0, 256))
    low = gate_g22_odf_fidelity(
        volumes, perm0, quats, sym, 0.0, None, 10.0, 256,
        memory_limit_bytes=1e12, memory_limit_source="test",
    )
    assert "NOT evaluated" not in low.message
    assert low.passed is True
    assert "WARN" in low.message
    m = re.search(r"empirical quantile ([0-9.]+)%", low.message)
    assert m is not None, low.message
    assert float(m.group(1)) < 2.5
    assert "OUTSIDE the interval" in low.message

    # Mirror: a null entirely BELOW any realistic mmd_vs_count drives the
    # quantile to 100% -- the upper tail a one-sided check DOES catch, so
    # this side is the sanity check that the two-sided logic did not break
    # the direction the old code already got right.
    monkeypatch.setattr(odf_mod, "null_mmd",
                        lambda *a, **k: np.linspace(0.0, 1e-6, 256))
    high = gate_g22_odf_fidelity(
        volumes, perm0, quats, sym, 0.0, None, 10.0, 256,
        memory_limit_bytes=1e12, memory_limit_source="test",
    )
    assert "WARN" in high.message
    m2 = re.search(r"empirical quantile ([0-9.]+)%", high.message)
    assert m2 is not None, high.message
    assert float(m2.group(1)) > 97.5
    assert "OUTSIDE the interval" in high.message

    # And a null that safely straddles the real mmd_vs_count (~0.84 for
    # this setup, per the "low"-null call above) on both sides must NOT
    # warn (mid-range quantile, the "nothing resolvable" case).
    monkeypatch.setattr(odf_mod, "null_mmd",
                        lambda *a, **k: np.linspace(0.0, 2.0, 4096))
    mid = gate_g22_odf_fidelity(
        volumes, perm0, quats, sym, 0.0, None, 10.0, 256,
        memory_limit_bytes=1e12, memory_limit_source="test",
    )
    assert "WARN" not in mid.message
    assert "inside the interval" in mid.message


def test_g22_message_and_verdict_cannot_disagree(monkeypatch):
    """Regression for the exact contradiction a real run once produced:
    'mmd_vs_count=0.3439 (... central 95% interval [0.3494, ...] ...
    inside the interval)' -- a value BELOW its own printed lower bound
    while the message claimed "inside". That happened because the OLD
    code decided WARN from the empirical quantile while printing a
    value-vs-interval sentence (two framings that can disagree by one
    null sample right at a boundary) and separately mislabeled p95 as a
    central-95%-interval upper bound (should be p97.5).

    This test parses the ACTUAL printed value and interval bounds back
    out of the message and asserts, across several different null shapes
    (forcing the real value to land inside, below, and above), that the
    WARN/ok verdict and the OUTSIDE/inside wording are ALWAYS both
    equivalent to the SAME direct comparison of the parsed numbers --
    i.e. the message can never assert "inside" for a value it also prints
    as outside its own stated bounds, because both are now derived from
    one shared boolean rather than two independently-computed framings."""
    import grainsmith.orientation.odf as odf_mod

    sym = _sym_cubic()
    n = 25
    quats = random_uniform(n, _rng(42))
    volumes = _rng(43).lognormal(0.0, 0.5, n)
    perm0 = np.arange(n, dtype=np.intp)

    null_shapes = [
        np.linspace(50.0, 100.0, 256),    # real value far below -> low tail
        np.linspace(0.0, 1e-6, 256),      # real value far above -> high tail
        np.linspace(0.0, 2.0, 4096),      # real value comfortably inside
        np.full(256, 0.8433),             # degenerate (zero-width) null
    ]
    for null in null_shapes:
        monkeypatch.setattr(odf_mod, "null_mmd", lambda *a, _n=null, **k: _n)
        result = gate_g22_odf_fidelity(
            volumes, perm0, quats, sym, 0.0, None, 10.0, 256,
            memory_limit_bytes=1e12, memory_limit_source="test",
        )
        assert "NOT evaluated" not in result.message
        mmd_vs_count, p2_5, p97_5, _ = _parse_kernel_line(result.message)
        outside = mmd_vs_count < p2_5 or mmd_vs_count > p97_5
        assert ("WARN" in result.message) == outside, result.message
        assert ("OUTSIDE the interval" in result.message) == outside, \
            result.message
        assert ("inside the interval" in result.message) == (not outside), \
            result.message


# ---------------------------------------------------------------------------
# H.6 -- canonical-name sigma3_fraction summary row (regression pin)
# ---------------------------------------------------------------------------


def test_e2e_sigma3_fraction_row_for_canonical_name(tmp_path):
    """Regression pin: the mdf,sigma3_fraction summary row was once written
    only for the DEPRECATED alias 'csl_enriched'.  _mdf_config uses the
    CANONICAL 'sigma3_angle_enriched', so the row must be present."""
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out"))
    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert by_key[("mdf", "target_type")] == "sigma3_angle_enriched"
    assert by_key[("mdf", "target_type_canonical")] == "sigma3_angle_enriched"
    assert float(by_key[("mdf", "sigma3_fraction")]) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# H.7 -- G25 texture-component fidelity (odf_components scheme)
# ---------------------------------------------------------------------------


def _odf_components_config(outdir, seed: int = 5150, n_grains: int = 12):
    """Single-phase cubic Cu, odf_components scheme (two twin-related
    components, equal weights)."""
    return resolve_config({
        "meta": {"title": "g25 e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [36.0, 36.0, 36.0]},
        "grains": {"number": n_grains},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {
            "scheme": "odf_components",
            "components": [
                {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 0.5,
                 "spread_deg": 4.0},
                {"euler_bunge_deg": [45.0, 70.528779365509308, 45.0],
                 "weight": 0.5, "spread_deg": 4.0},
            ],
        },
        "output": {"directory": str(outdir)},
    })


def test_e2e_g25_present_with_summary_rows(tmp_path):
    from grainsmith.pipeline import run

    res = run(_odf_components_config(tmp_path / "out"))
    assert res.gates.all_passed()
    g25 = [r for r in res.gates.results() if r.gate == "G25"]
    assert len(g25) == 1
    assert g25[0].passed is True
    assert "basis=volume" in g25[0].message

    measured = g25[0].measured
    assert measured["basis"] == "volume"  # OrientationConfig default
    per = measured["per_component"]
    assert len(per) == 2  # two components
    for w_cfg, f_count, f_vol_pre, f_vol_post, mean_v in per:
        assert w_cfg == pytest.approx(0.5)
        assert 0.0 <= f_count <= 1.0
        assert 0.0 <= f_vol_pre <= 1.0
        assert 0.0 <= f_vol_post <= 1.0
        assert mean_v > 0.0
    # the per-component fractions partition the grains / the material
    assert sum(row[1] for row in per) == pytest.approx(1.0)
    assert sum(row[2] for row in per) == pytest.approx(1.0)
    assert sum(row[3] for row in per) == pytest.approx(1.0)
    for key in ("tv_cfg_vs_pre", "tv_pre_vs_post", "tv_cfg_vs_post"):
        assert measured[key] >= 0.0

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert by_key[("texture", "component_weight_basis")] == "volume"
    for k in range(2):
        assert float(by_key[("texture", f"component_{k}_weight_configured")]) \
            == pytest.approx(0.5)
        assert ("texture", f"component_{k}_count_fraction") in by_key
        assert ("texture", f"component_{k}_volume_fraction_pre") in by_key
        # the FINAL (post-anneal) number keeps the exact pre-existing key
        # so it stays the headline volume-weighted-ODF figure.
        assert ("texture", f"component_{k}_volume_fraction") in by_key
        # `_pre`: gate_g25 measures this on component_of_pre on purpose, so
        # the key says so rather than reading as a post-anneal mean.
        assert ("texture", f"component_{k}_mean_volume_pre_A3") in by_key
        assert ("texture", f"component_{k}_mean_volume_A3") not in by_key
    for key in ("component_tv_cfg_vs_pre", "component_tv_pre_vs_post",
                "component_tv_cfg_vs_post"):
        assert ("texture", key) in by_key


def test_e2e_g25_absent_for_other_schemes(tmp_path):
    """G25 is an odf_components-only gate (random_uniform here)."""
    from grainsmith.pipeline import run

    res = run(_no_mdf_target_config(tmp_path / "out"))
    assert all(r.gate != "G25" for r in res.gates.results())


def test_g25_unit_unequal_volumes_expose_count_vs_volume_gap():
    """Unit-level, no annealing (component_of_pre == component_of_final):
    equal weights and equal counts, but component 1's grains are larger --
    f_count matches the labels while f_vol_pre/f_vol_post expose the
    volume-weighted share (identical to each other since there is no
    permutation here). Report-only: passed is always True."""
    from grainsmith.qa import gate_g25_component_fidelity

    weights = np.array([1.0, 1.0])
    component_of = np.array([0, 0, 1, 1])
    volumes = np.array([1.0, 1.0, 3.0, 3.0])
    r = gate_g25_component_fidelity(
        weights, component_of, component_of, volumes, "count")
    assert r.passed is True
    m = r.measured
    assert m["basis"] == "count"
    (w0, fn0, fvp0, fvf0, mv0), (w1, fn1, fvp1, fvf1, mv1) = m["per_component"]
    assert w0 == w1 == pytest.approx(0.5)
    assert fn0 == fn1 == pytest.approx(0.5)
    assert fvp0 == fvf0 == pytest.approx(0.25)
    assert fvp1 == fvf1 == pytest.approx(0.75)
    assert mv0 == pytest.approx(1.0)
    assert mv1 == pytest.approx(3.0)
    assert m["tv_cfg_vs_pre"] == pytest.approx(0.25)
    assert m["tv_pre_vs_post"] == pytest.approx(0.0)  # no permutation
    assert m["tv_cfg_vs_post"] == pytest.approx(0.25)
    assert "0.7500" in r.message


def test_g25_unit_permutation_moves_pre_to_post():
    """Unit-level: component_of_final is a genuine permutation of
    component_of_pre (grains 1 and 2 swap orientations) -- f_count is
    unchanged (a permutation can never change a bincount) while f_vol_pre
    and f_vol_post differ, and tv_pre_vs_post captures exactly that
    movement."""
    from grainsmith.qa import gate_g25_component_fidelity

    weights = np.array([1.0, 1.0])
    pre = np.array([0, 0, 1, 1])
    perm = np.array([0, 2, 1, 3])   # grains 1 and 2 swap
    final = pre[perm]
    volumes = np.array([1.0, 1.0, 3.0, 3.0])
    r = gate_g25_component_fidelity(weights, pre, final, volumes, "volume")
    assert r.passed is True
    m = r.measured
    per = m["per_component"]
    # f_count identical for both components regardless of pre vs post.
    assert per[0][1] == pytest.approx(0.5)
    assert per[1][1] == pytest.approx(0.5)
    # pre: {0,1}->comp0 (vol 1+1=2), {2,3}->comp1 (vol 3+3=6), total 8
    assert per[0][2] == pytest.approx(2.0 / 8.0)
    assert per[1][2] == pytest.approx(6.0 / 8.0)
    # post: comp0 gets grains 0,2 (vol 1+3=4), comp1 gets grains 1,3
    # (vol 1+3=4), total 8
    assert per[0][3] == pytest.approx(4.0 / 8.0)
    assert per[1][3] == pytest.approx(4.0 / 8.0)
    assert m["tv_pre_vs_post"] > 0.0
    assert m["tv_pre_vs_post"] == pytest.approx(
        0.5 * (abs(0.5 - 0.25) + abs(0.5 - 0.75)))


def test_g25_never_aborts_on_out_of_range_component_index():
    from grainsmith.qa import gate_g25_component_fidelity

    r = gate_g25_component_fidelity(
        np.array([1.0]), np.array([0, 7]), np.array([0, 7]),
        np.array([1.0, 1.0]), "count")
    assert r.passed is True
    assert r.measured is None
    assert "could not be evaluated" in r.message


def test_g25_never_aborts_on_mismatched_lengths():
    """G25 never warns and never aborts: a deliberately broken input
    (component_of_pre/component_of_final one element shorter than
    volumes) still returns passed True with measured None -- the run
    itself must never be aborted by a diagnostic gate."""
    from grainsmith.qa import gate_g25_component_fidelity

    r = gate_g25_component_fidelity(
        np.array([1.0, 1.0]), np.array([0, 1]), np.array([0, 1]),
        np.array([1.0, 1.0, 1.0]), "volume")
    assert r.passed is True
    assert r.measured is None
    assert "could not be evaluated" in r.message


# ---------------------------------------------------------------------------
# H.7b -- component_weight_basis "volume" pins the counterexample
# ---------------------------------------------------------------------------


def _odf_components_size_dist_config(outdir, basis, seed: int = 13,
                                     n_grains: int = 24,
                                     drift_max: float | None = None):
    """odf_components on a lognormal size_distribution -- the
    counter-example shape: 24 grains, sigma_log 0.5, two equal-weight
    (0.5/0.5) components. The categorical (grain-count) draw realises
    'weight' as a grain-COUNT fraction, so the VOLUME fraction each
    component ends up representing drifts from the configured 0.5/0.5 by
    an amount set by the size distribution -- exactly the gap
    component_weight_basis 'volume' closes. seed=13 is pinned because it
    is where the natural per-seed variance of the 'count' draw exceeds the
    (loose) max(volume)/total bound -- see the module-level comment on
    test_e2e_g25_volume_basis_pins_counterexample."""
    orient: dict = {
        "scheme": "odf_components",
        "components": [
            {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 0.5,
             "spread_deg": 4.0},
            {"euler_bunge_deg": [45.0, 70.528779365509308, 45.0],
             "weight": 0.5, "spread_deg": 4.0},
        ],
        "component_weight_basis": basis,
    }
    if drift_max is not None:
        orient["mdf_target"] = {
            "type": "haar_random",
            "annealing_steps": 4000,
            "odf_drift_max": drift_max,
        }
    return resolve_config({
        "meta": {"title": "g25 size-distribution e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": seed},
        "box": {"lengths": [50.0, 50.0, 50.0]},
        "grains": {"number": n_grains,
                   "size_distribution": {"type": "lognormal",
                                         "sigma_log": 0.5}},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": orient,
        "output": {"directory": str(outdir)},
    })


def test_e2e_g25_volume_basis_pins_counterexample(tmp_path):
    """PINS THE DEFECT so it cannot silently return: 24 grains,
    size_distribution lognormal sigma_log 0.5, two odf_components at
    weight 0.5 each -- exactly the shape measured (without this fix) as
    configured 0.500/0.500 realising 0.745/0.255 by volume (TV 0.245).

    With component_weight_basis 'volume' the sampler-realised (pre-anneal)
    volume fractions must land within max(volume)/total of 0.5 each -- an
    easy, LOOSE corollary of volume_balanced_partition's proven invariant
    ("no component ever overshoots its volume target by more than the
    SMALLEST grain it received", which is <= the LARGEST grain in the
    whole set, hence <= max(volume)/total once normalised).

    With component_weight_basis 'count' (the pre-1.2 behaviour) that same
    invariant carries no guarantee at all -- it is a robust per-seed
    comparison against 'volume' (asserted below via tv_cfg_vs_pre) that
    stands in for it, since the categorical draw's bound violation itself
    is only a distributional claim; see
    tests/test_odf_guard.py::test_measured_number_count_vs_volume_basis
    (400 draws) for that distributional claim about the 'count' basis."""
    from grainsmith.analysis.grains import grain_volumes
    from grainsmith.pipeline import run

    max_frac = None
    tv_cfg_vs_pre = {}
    for basis in ("volume", "count"):
        res = run(_odf_components_size_dist_config(
            tmp_path / f"out_{basis}", basis))
        assert res.gates.all_passed()

        L = np.asarray(res.config.box.lengths, dtype=np.float64)
        vols = grain_volumes(res.tess, L)
        frac = float(np.max(vols) / np.sum(vols))
        if max_frac is None:
            max_frac = frac
        else:
            # Same seed/size_distribution -> same realised grain geometry
            # regardless of component_weight_basis: the basis only ever
            # touches the ORIENTATION assignment, never grain volumes.
            assert frac == pytest.approx(max_frac)

        srows = _csv_rows(res.outdir / "summary.csv")
        by_key = {(r["section"], r["key"]): r["value"] for r in srows}
        assert by_key[("texture", "component_weight_basis")] == basis
        deviations = [
            abs(float(by_key[
                ("texture", f"component_{k}_volume_fraction_pre")]) - 0.5)
            for k in range(2)
        ]
        if basis == "volume":
            # THEOREM (volume_balanced_partition's proven overshoot
            # invariant), so a single-seed assertion is legitimate here.
            assert all(d <= max_frac for d in deviations), (
                basis, deviations, max_frac)
        tv_cfg_vs_pre[basis] = float(
            by_key[("texture", "component_tv_cfg_vs_pre")])

    # Robust per-seed comparison (not a per-seed bound violation, which is
    # only true in distribution -- see the docstring above): at the SAME
    # seed and SAME grain volumes, 'count' realises a strictly larger
    # configured-vs-pre volume TV than 'volume', by a wide margin (observed
    # on this reference scenario: two to three orders of magnitude); factor
    # 20 leaves enormous headroom while still being a real comparison.
    assert tv_cfg_vs_pre["count"] >= 20 * tv_cfg_vs_pre["volume"], \
        tv_cfg_vs_pre


def test_e2e_g25_volume_basis_plus_drift_cap_bounds_final_odf_error(
        tmp_path):
    """THE HEADLINE SCIENTIFIC RESULT of this fix: component_weight_basis
    'volume' alone only bounds the SAMPLER's (pre-anneal) volume
    fractions -- an MDF-targeting anneal running afterward can still move
    volume between components (tv_pre_vs_post is NOT generally bounded by
    odf_drift_max, see gate_g25_component_fidelity's docstring). Pairing
    'volume' with a TIGHT odf_drift_max cap keeps that further movement
    small in practice, so basis 'volume' PLUS a drift cap together bound
    the FINAL, fully-realised volume-weighted ODF error (tv_cfg_vs_post)
    -- the actual quantity at issue, not just the
    sampler's output.

    The assertion is intentionally tight: without this fix (a bare
    categorical draw, no cap) the identical 24-grain/sigma_log-0.5 shape
    measures a component-volume TV around 0.4 (see the counterexample
    test above) -- roughly an order of magnitude looser than what is
    asserted here."""
    from grainsmith.pipeline import run

    res = run(_odf_components_size_dist_config(
        tmp_path / "out", "volume", drift_max=0.02))
    assert res.gates.all_passed()
    g25 = [r for r in res.gates.results() if r.gate == "G25"][0]
    m = g25.measured
    assert m["basis"] == "volume"
    # Tight: the sampler alone (tv_cfg_vs_pre) is already far under 0.02
    # by the volume_balanced_partition invariant, and the drift cap keeps
    # the annealer's own contribution (tv_pre_vs_post) of a similar
    # order -- 0.05 leaves a healthy margin over the measured ~0.01-0.03
    # while staying nowhere near the ~0.4 an uncapped 'count' run reaches.
    assert m["tv_cfg_vs_post"] < 0.05

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert float(by_key[("texture", "component_tv_cfg_vs_post")]) \
        == pytest.approx(m["tv_cfg_vs_post"])


# ---------------------------------------------------------------------------
# H.8 -- G22 reports the quotient-space (symmetry-merged) drift
# ---------------------------------------------------------------------------


def test_e2e_g22_reports_quotient_space_drift(tmp_path):
    """G22's kernel block also reports the quotient-space (symmetry-merged)
    drift, guaranteed <= the atomic one (TV never grows under a
    pushforward)."""
    from grainsmith.pipeline import run

    res = run(_mdf_config(tmp_path / "out"))
    g22 = [r for r in res.gates.results() if r.gate == "G22"][0]
    m_q = re.search(r"quotient-space drift[^=]*= ([0-9.eE+-]+)", g22.message)
    assert m_q is not None, g22.message
    drift_q = float(m_q.group(1))
    m_a = re.search(r"drift=([0-9.eE+-]+) \(", g22.message)
    assert m_a is not None, g22.message
    drift_atomic = float(m_a.group(1))
    assert drift_q <= drift_atomic + 1e-12


# ---------------------------------------------------------------------------
# H.9 -- G26 atomistic ODF-weighting discretisation floor
# ---------------------------------------------------------------------------


def test_g26_unit_proportional_atoms_zero_tv():
    """Volumes and atom counts exactly proportional (same shape, up to a
    constant) -> w_vol == w_atom elementwise -> tv == 0, max_dw == 0.
    Report-only: passed is always True."""
    from grainsmith.qa import gate_g26_odf_weighting_floor

    volumes = np.array([1.0, 2.0, 3.0, 4.0])
    n_atoms = np.array([10.0, 20.0, 30.0, 40.0])  # exactly 10x volumes
    r = gate_g26_odf_weighting_floor(volumes, n_atoms)
    assert r.gate == "G26"
    assert r.passed is True
    assert r.measured is not None
    tv, max_dw, mean_atoms = r.measured
    assert tv == pytest.approx(0.0, abs=1e-12)
    assert max_dw == pytest.approx(0.0, abs=1e-12)
    assert mean_atoms == pytest.approx(25.0)
    assert "WARN" not in r.message


def test_g26_unit_not_evaluated_when_atoms_absent():
    """All n_atoms zero (analyze_grains' own convention when the atom
    block is omitted) -> NOT EVALUATED, never a false 0.0: passed True,
    measured None, message says NOT EVALUATED and never claims 0.0."""
    from grainsmith.qa import gate_g26_odf_weighting_floor

    volumes = np.array([1.0, 2.0, 3.0])
    n_atoms = np.array([0.0, 0.0, 0.0])
    r = gate_g26_odf_weighting_floor(volumes, n_atoms)
    assert r.passed is True
    assert r.measured is None
    assert "NOT EVALUATED" in r.message
    assert "0.0" not in r.message
    assert "WARN" not in r.message


def test_g26_never_aborts_on_broken_input():
    """A deliberately broken input (n_atoms one element shorter than
    volumes) still returns passed True with measured None rather than
    raising -- a diagnostic gate must never abort the run."""
    from grainsmith.qa import gate_g26_odf_weighting_floor

    r = gate_g26_odf_weighting_floor(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    assert r.passed is True
    assert r.measured is None
    assert "could not be evaluated" in r.message
    assert "WARN" not in r.message


def test_g26_unit_multiphase_reports_per_phase():
    """phase_of/phase_names given -> `measured` is a tuple of one
    (tv, max_dw, mean_atoms) triple per phase, computed WITHIN each
    phase's own grain subset (never a global mix): phase "alpha" here is
    exactly proportional (tv == 0), phase "beta" is not."""
    from grainsmith.qa import gate_g26_odf_weighting_floor

    volumes = np.array([1.0, 1.0, 2.0, 2.0])
    n_atoms = np.array([5.0, 5.0, 3.0, 9.0])
    phase_of = np.array([0, 0, 1, 1])
    phase_names = ["alpha", "beta"]
    r = gate_g26_odf_weighting_floor(
        volumes, n_atoms, phase_of=phase_of, phase_names=phase_names)
    assert r.passed is True
    assert r.measured is not None
    assert len(r.measured) == 2

    tv0, max_dw0, mean0 = r.measured[0]
    assert tv0 == pytest.approx(0.0, abs=1e-12)
    assert max_dw0 == pytest.approx(0.0, abs=1e-12)
    assert mean0 == pytest.approx(5.0)

    tv1, max_dw1, mean1 = r.measured[1]
    assert tv1 == pytest.approx(0.25)
    assert max_dw1 == pytest.approx(0.25)
    assert mean1 == pytest.approx(6.0)

    assert "alpha" in r.message
    assert "beta" in r.message


def test_g26_unit_multiphase_partial_not_evaluated():
    """One phase's atoms are all zero, the other's are not -- that
    phase's entry is None (NOT EVALUATED), the other still reports a
    real triple; the gate itself is never fully un-evaluated just
    because one phase's atom block is empty."""
    from grainsmith.qa import gate_g26_odf_weighting_floor

    volumes = np.array([1.0, 1.0, 2.0, 2.0])
    n_atoms = np.array([0.0, 0.0, 3.0, 9.0])
    phase_of = np.array([0, 0, 1, 1])
    phase_names = ["alpha", "beta"]
    r = gate_g26_odf_weighting_floor(
        volumes, n_atoms, phase_of=phase_of, phase_names=phase_names)
    assert r.passed is True
    assert r.measured[0] is None
    assert r.measured[1] is not None
    assert "NOT EVALUATED" in r.message


def test_e2e_g26_present_finite_positive_and_matches_gate_measured(tmp_path):
    """A small real run with atoms enabled: the odf,weighting_tv_volume_
    vs_atoms summary.csv row exists, is finite and > 0 (a genuine
    discretisation gap on a small, real grain structure), and equals the
    gate's own `measured` value -- read straight off the field, not
    re-derived from the message."""
    from grainsmith.pipeline import run

    res = run(_odf_components_config(tmp_path / "out"))
    assert res.gates.all_passed()
    g26 = [r for r in res.gates.results() if r.gate == "G26"]
    assert len(g26) == 1
    assert g26[0].passed is True
    assert g26[0].measured is not None
    tv, max_dw, mean_atoms = g26[0].measured
    assert np.isfinite(tv) and tv > 0.0
    assert np.isfinite(max_dw) and max_dw >= 0.0
    assert mean_atoms > 0.0

    srows = _csv_rows(res.outdir / "summary.csv")
    by_key = {(r["section"], r["key"]): r["value"] for r in srows}
    assert ("odf", "weighting_tv_volume_vs_atoms") in by_key
    assert ("odf", "weighting_max_dw") in by_key
    assert ("odf", "atoms_per_grain_mean") in by_key

    row_tv = float(by_key[("odf", "weighting_tv_volume_vs_atoms")])
    assert np.isfinite(row_tv)
    assert row_tv > 0.0
    assert row_tv == pytest.approx(tv)
    assert float(by_key[("odf", "weighting_max_dw")]) == pytest.approx(max_dw)
    assert float(by_key[("odf", "atoms_per_grain_mean")]) == \
        pytest.approx(mean_atoms)


def test_e2e_g26_absent_when_message_never_warns(tmp_path):
    """Report-only sanity: G26 is present on an ordinary run and its
    message never carries a WARN token, matching G23/G25's contract."""
    from grainsmith.pipeline import run

    res = run(_no_mdf_target_config(tmp_path / "out"))
    g26 = [r for r in res.gates.results() if r.gate == "G26"]
    assert len(g26) == 1
    assert "WARN" not in g26[0].message
