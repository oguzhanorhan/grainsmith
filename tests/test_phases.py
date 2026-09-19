"""Tests for multiphase polycrystals + gate G15."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError
from grainsmith.phases import achieved_fractions, assign_phases
from grainsmith.qa import gate_g15_phase_fractions

ALPHA = {
    "name": "alpha", "fraction": 0.6, "crystal": {
        "space_group": {"number": 194},
        "lattice": {"a": 2.951, "c": 4.684},
        "wyckoff_sites": [{"element": "Ti",
                           "coords": [1.0 / 3.0, 2.0 / 3.0, 0.25],
                           "letter": "c"}],
    },
}
BETA = {
    "name": "beta", "fraction": 0.4, "crystal": {
        "space_group": {"number": 229},
        "lattice": {"a": 3.32},
        "wyckoff_sites": [{"element": "Ti", "coords": [0.0, 0.0, 0.0]}],
    },
}


# ---------------------------------------------------------------------------
# Greedy assignment (deterministic, exact pins)
# ---------------------------------------------------------------------------


def test_assign_exact_split():
    """Volumes [5, 3, 2], fractions 50/50: largest grain alone balances
    the other two — both fractions land EXACTLY on target."""
    phase_of = assign_phases(np.array([5.0, 3.0, 2.0]),
                             np.array([0.5, 0.5]))
    np.testing.assert_array_equal(phase_of, [0, 1, 1])
    ach = achieved_fractions(np.array([5.0, 3.0, 2.0]), phase_of, 2)
    np.testing.assert_allclose(ach, [0.5, 0.5])


def test_assign_descending_deficit_order():
    """Equal volumes, fractions 0.75/0.25 over 4 grains → 3:1 split;
    deficit ties go to the LOWEST phase index, the last grain is forced
    to the still-empty phase."""
    phase_of = assign_phases(np.ones(4), np.array([0.75, 0.25]))
    np.testing.assert_array_equal(phase_of, [0, 0, 0, 1])


def test_assign_every_phase_gets_a_grain():
    """Extreme fractions cannot starve a phase (forced final assignment;
    resolve Rule 23 guarantees n >= P)."""
    phase_of = assign_phases(np.array([0.5, 0.5]),
                             np.array([0.99, 0.01]))
    assert set(phase_of.tolist()) == {0, 1}


def test_assign_deterministic_under_ties():
    v = np.array([1.0, 1.0, 1.0, 1.0])
    a = assign_phases(v, np.array([0.5, 0.5]))
    b = assign_phases(v, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(a, b)


def test_assign_too_few_grains_raises():
    with pytest.raises(ValueError, match="Rule 23"):
        assign_phases(np.array([1.0]), np.array([0.5, 0.5]))


def test_g15_force_feed_can_exceed_granularity_bound():
    """SCIENCE_AUDIT #11: the V_max/V_box bound holds for the greedy LPT
    assignment, but with ≥ 3 phases at extreme fractions and few grains the
    every-phase-≥1-grain guard force-feeds a phase against the greedy choice,
    so the deviation can exceed the bound — G15 still only WARNs."""
    vols = np.ones(4)                                  # 4 equal grains
    targets = np.array([0.85, 0.05, 0.05, 0.05])       # extreme, 4 phases
    phase_of = assign_phases(vols, targets)
    assert set(phase_of.tolist()) == {0, 1, 2, 3}      # every phase force-fed
    ach = achieved_fractions(vols, phase_of, 4)
    max_dev = float(np.max(np.abs(ach - targets)))
    granularity = float(vols.max() / vols.sum())       # V_max/V_box = 0.25
    assert max_dev > granularity                       # bound is exceeded here
    g15 = gate_g15_phase_fractions(ach, targets,
                                   ["a", "b", "c", "d"], granularity)
    assert g15.passed and "WARN" in g15.message        # never hard-fails


def test_g15_levels():
    ok = gate_g15_phase_fractions(np.array([0.58, 0.42]),
                                  np.array([0.6, 0.4]),
                                  ["a", "b"], granularity=0.05)
    assert ok.passed and "ok" in ok.message and "WARN" not in ok.message
    warn = gate_g15_phase_fractions(np.array([0.5, 0.5]),
                                    np.array([0.6, 0.4]),
                                    ["a", "b"], granularity=0.05)
    assert warn.passed and "WARN" in warn.message   # warn-only
    assert warn.measured == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Schema + resolve rules (21–23)
# ---------------------------------------------------------------------------


def _raw(**over):
    raw = {
        "seed": {"mode": "fixed", "value": 11},
        "box": {"lengths": [50.0, 50.0, 50.0]},
        "grains": {"number": 8},
        "phases": [dict(ALPHA), dict(BETA)],
        "orientation": {"scheme": "random_uniform"},
    }
    raw.update(over)
    return raw


def test_rule21_crystal_and_phases_exclusive():
    raw = _raw(crystal=dict(ALPHA["crystal"]))
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve_config(raw)
    raw = _raw()
    del raw["phases"]
    with pytest.raises(ConfigError, match="Exactly one of"):
        resolve_config(raw)


def test_rule21_needs_two_phases():
    raw = _raw(phases=[dict(ALPHA)])
    with pytest.raises(ConfigError, match=">= 2 entries"):
        resolve_config(raw)


def test_rule21_unique_names_and_fraction_sum():
    bad = dict(BETA)
    bad["name"] = "alpha"
    with pytest.raises(ConfigError, match="unique"):
        resolve_config(_raw(phases=[dict(ALPHA), bad]))
    bad = dict(BETA)
    bad["fraction"] = 0.5
    with pytest.raises(ConfigError, match="sum to 1"):
        resolve_config(_raw(phases=[dict(ALPHA), bad]))


def test_phase_name_pattern_enforced():
    bad = dict(BETA)
    bad["name"] = "beta phase"   # space → schema pattern violation (G1)
    with pytest.raises(ConfigError, match="Schema validation"):
        resolve_config(_raw(phases=[dict(ALPHA), bad]))


def test_rule22_csl_and_mdf_target_rejected():
    raw = _raw(analysis={"csl": True})
    with pytest.raises(ConfigError, match="csl is undefined"):
        resolve_config(raw)
    raw = _raw()
    raw["orientation"]["mdf_target"] = {"type": "mackenzie"}
    with pytest.raises(ConfigError, match="mdf_target cannot be combined"):
        resolve_config(raw)


def test_rule22_scheme_restricted():
    raw = _raw(orientation={"scheme": "fiber",
                            "fiber": {"crystal_axis": [1, 1, 1]}})
    with pytest.raises(ConfigError, match="random_uniform"):
        resolve_config(raw)


def test_rule22_from_list_quaternions_allowed_hkl_rejected():
    n = 8
    raw = _raw(orientation={
        "scheme": "from_list",
        "from_list": [{"quaternion": [1.0, 0.0, 0.0, 0.0]}] * n})
    resolve_config(raw)   # passes G1
    raw = _raw(orientation={
        "scheme": "from_list",
        "from_list": [{"hkl_uvw": {"plane": [1, 1, 1],
                                   "direction": [1, -1, 0]}}] * n})
    with pytest.raises(ConfigError, match="hkl_uvw"):
        resolve_config(raw)


def test_rule23_enough_grains():
    raw = _raw(grains={"number": 1})
    with pytest.raises(ConfigError, match="at least one grain"):
        resolve_config(raw)


def test_rule5b_per_phase_occupancy_checked():
    bad = dict(BETA)
    bad = {**bad, "crystal": {**bad["crystal"], "wyckoff_sites": [
        {"element": {"Ti": 0.7, "V": 0.2}, "coords": [0.0, 0.0, 0.0]}]}}
    with pytest.raises(ConfigError, match=r"phases\[beta\]"):
        resolve_config(_raw(phases=[dict(ALPHA), bad]))


# ---------------------------------------------------------------------------
# End-to-end: Ti alpha+beta (same element, two structures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ti_run(tmp_path_factory):
    from grainsmith.pipeline import run
    out = tmp_path_factory.mktemp("ti") / "out"
    return run(resolve_config(_raw(output={"directory": str(out)})))


def test_e2e_ti_gates_and_fractions(ti_run):
    assert ti_run.gates.all_passed()
    g15 = [r for r in ti_run.gates.results() if r.gate == "G15"][0]
    assert g15.passed
    assert ti_run.phase_of is not None
    assert set(ti_run.phase_of.tolist()) == {0, 1}
    # Achieved fractions reproduce the gate's measured deviation.
    vols = np.array([g.volume_A3 for g in ti_run.grain_reports])
    ach = achieved_fractions(vols, ti_run.phase_of, 2)
    assert abs(ach[0] - 0.6) == pytest.approx(g15.measured) or \
        abs(ach[1] - 0.4) == pytest.approx(g15.measured)


def test_e2e_ti_phase_columns(ti_run):
    with (ti_run.outdir / "grains.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-1] == "phase"
    names = [r[-1] for r in rows[1:]]
    expect = ["alpha" if p == 0 else "beta" for p in ti_run.phase_of]
    assert names == expect

    with (ti_run.outdir / "boundaries.csv").open() as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    assert hdr[-2:] == ["phase_i", "phase_j"]
    i_mis = hdr.index("misorientation_deg")
    i_chr = hdr.index("character")
    inter = [r for r in rows[1:] if r[-2] != r[-1]]
    same = [r for r in rows[1:] if r[-2] == r[-1]]
    assert inter and same
    for r in inter:
        assert r[i_chr] == "interphase"
        assert r[i_mis] == "nan"
        assert r[hdr.index("axis_u")] == "0"
        # Habit planes stay defined for interphase pairs.
        assert r[hdr.index("plane_i_hkl")] not in ("",)
    for r in same:
        assert r[i_chr] in ("twist", "tilt", "mixed", "undefined")
        if r[i_chr] != "undefined":
            assert float(r[i_mis]) > 0.0


def test_e2e_ti_per_phase_texture_files(ti_run):
    names = {p.name for p in ti_run.files}
    assert {"mdf_alpha.csv", "mdf_beta.csv",
            "odf_mtex_alpha.txt", "odf_mtex_beta.txt"} <= names
    assert "mdf.csv" not in names and "odf_mtex.txt" not in names
    # Per-phase ODF weights are within-phase volume fractions (sum 1);
    # row counts match the phase populations.
    for p, name in enumerate(("alpha", "beta")):
        lines = [ln for ln in
                 (ti_run.outdir / f"odf_mtex_{name}.txt").read_text(
                     encoding="utf-8").splitlines()
                 if ln and not ln.startswith("%")]
        assert len(lines) == int(np.sum(ti_run.phase_of == p))
        weights = [float(ln.split()[3]) for ln in lines]
        assert sum(weights) == pytest.approx(1.0)
        # mdf_<name>.csv has mdf_bins data rows.
        with (ti_run.outdir / f"mdf_{name}.csv").open() as fh:
            n_rows = sum(1 for _ in fh) - 1
        assert n_rows == ti_run.config.analysis.mdf_bins


def test_e2e_ti_summary_sections(ti_run):
    txt = (ti_run.outdir / "summary.csv").read_text(encoding="utf-8")
    assert "phase:alpha" in txt and "phase:beta" in txt
    assert "fraction_target" in txt and "fraction_achieved" in txt
    assert "G15" in txt
    # No single-phase 'crystal' section rows.
    assert "\ncrystal," not in txt


# ---------------------------------------------------------------------------
# End-to-end: Cu+Fe composite (different elements)
# ---------------------------------------------------------------------------


def _cufe_raw(outdir, **over):
    raw = _raw(output={"directory": str(outdir)}, **over)
    raw["grains"] = {"number": 6}
    raw["box"] = {"lengths": [40.0, 40.0, 40.0]}
    raw["phases"] = [
        {"name": "Cu_fcc", "fraction": 0.5, "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu",
                               "coords": [0.0, 0.0, 0.0]}]}},
        {"name": "Fe_bcc", "fraction": 0.5, "crystal": {
            "space_group": {"number": 229},
            "lattice": {"a": 2.866},
            "wyckoff_sites": [{"element": "Fe",
                               "coords": [0.0, 0.0, 0.0]}]}},
    ]
    return raw


def test_e2e_cufe_species_follow_phases(tmp_path):
    from grainsmith.pipeline import run
    res = run(resolve_config(_cufe_raw(tmp_path / "out")))
    assert res.gates.all_passed()
    # Every atom of a grain carries its phase's element — exact pin.
    for g, p in enumerate(res.phase_of):
        sp = set(np.unique(res.atoms.species[res.atoms.grain == g]).tolist())
        assert sp == ({"Cu"} if p == 0 else {"Fe"})
    # Two LAMMPS types, alphabetical.
    txt = (res.outdir / "polycrystal.data").read_text(encoding="utf-8")
    assert "2 atom types" in txt
    assert "# type 1 = Cu" in txt and "# type 2 = Fe" in txt
    # G9 nominal is the ρ·V-weighted mix: final fractions within drift.
    g9 = [r for r in res.gates.results() if r.gate == "G9"][0]
    assert g9.measured < 0.05

    from grainsmith.constants import ATOMIC_MASSES

    with (res.outdir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = {(row["section"], row["key"]): row["value"]
                   for row in csv.DictReader(stream)}
    species, counts = np.unique(res.atoms.species, return_counts=True)
    total_mass = sum(int(count) * ATOMIC_MASSES[element]
                     for element, count in zip(species, counts, strict=True))
    for element, count in zip(species, counts, strict=True):
        assert int(summary["composition", f"{element}_n_final"]) == int(count)
        assert float(summary["composition", f"{element}_final_fraction"]) == \
            pytest.approx(int(count) / len(res.atoms))
        assert float(summary["composition", f"{element}_final_mass_fraction"]) == \
            pytest.approx(int(count) * ATOMIC_MASSES[element] / total_mass)
    phase_volumes = np.bincount(
        res.phase_of, weights=[grain.volume_A3 for grain in res.grain_reports],
        minlength=2)
    expected_atoms = phase_volumes * np.array([4 / 3.615**3, 2 / 2.866**3])
    assert float(summary["composition", "Cu_nominal_fraction"]) == \
        pytest.approx(expected_atoms[0] / expected_atoms.sum())
    assert float(summary["composition", "Fe_nominal_fraction"]) == \
        pytest.approx(expected_atoms[1] / expected_atoms.sum())
    assert ("phase:Cu_fcc", "hall_number") in summary
    assert ("phase:Fe_bcc", "hall_number") in summary


def test_e2e_cufe_jobs_invariant(tmp_path):
    from grainsmith.pipeline import run
    r1 = run(resolve_config(_cufe_raw(tmp_path / "a")), jobs=1)
    r2 = run(resolve_config(_cufe_raw(tmp_path / "b")), jobs=2)
    np.testing.assert_array_equal(r1.atoms.pos, r2.atoms.pos)
    np.testing.assert_array_equal(r1.atoms.species, r2.atoms.species)
    np.testing.assert_array_equal(r1.atoms.grain, r2.atoms.grain)


def test_single_phase_columns_unchanged():
    """v1 column sets are untouched by R4b (byte-identity guarantee)."""
    from grainsmith.io import (
        BOUNDARIES_COLUMNS,
        BOUNDARIES_COLUMNS_PHASES,
        GRAINS_COLUMNS,
        GRAINS_COLUMNS_PHASES,
    )
    assert GRAINS_COLUMNS[-1] == "x_dir_dev_deg"
    assert "phase" not in GRAINS_COLUMNS
    assert GRAINS_COLUMNS_PHASES == [*GRAINS_COLUMNS, "phase"]
    assert BOUNDARIES_COLUMNS[-1] == "n_overlap_deleted"
    assert BOUNDARIES_COLUMNS_PHASES == [*BOUNDARIES_COLUMNS,
                                         "phase_i", "phase_j"]
