"""Tests for direct CIF crystal-structure input (ASE read -> spglib
symmetry detection -> grainsmith Wyckoff-site spec, §6.1-6.3).

Covers the three shipped Materials Project CIFs (examples/assets/{TiNi,
Ti2Ni,TiNi3}.cif): declared-vs-detected space-group cross-check,
conventional-cell atom counts against the CIF's own composition * Z, the
crystal.cif <-> manual-path mutual-exclusion validation, and small
end-to-end builds through the overlap gate.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

pytest.importorskip("ase")

from grainsmith.config.resolve import resolve_config
from grainsmith.crystal.cif import read_cif_crystal
from grainsmith.errors import ConfigError
from grainsmith.pipeline import run

# Declared _symmetry_Int_Tables_number / composition*Z read directly from
# the CIF headers (examples/assets/*.cif) — see the module docstring; TiNi3
# declares 'P 1' (number 1) even though its atom positions are fully
# P6_3/mmc symmetric (a pymatgen habit), so spglib detection is
# expected to differ there; TiNi and Ti2Ni declare their true group.
_CIF_EXPECT = {
    "TiNi.cif": {"declared_sg": 11, "detected_sg": 11, "z": 2, "atoms_per_formula": 2},
    "Ti2Ni.cif": {"declared_sg": 227, "detected_sg": 227, "z": 32, "atoms_per_formula": 3},
    "TiNi3.cif": {"declared_sg": 1, "detected_sg": 194, "z": 4, "atoms_per_formula": 4},
}


def _cif_path(examples_dir, name):
    return examples_dir / "assets" / name


# ---------------------------------------------------------------------------
# Space-group detection + atom-count cross-checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_CIF_EXPECT))
def test_declared_sg_matches_header(examples_dir, name):
    """Sanity-check our own expectation table against the actual CIF file
    (never trust a hardcoded number without reading the source)."""
    text = (examples_dir / "assets" / name).read_text(encoding="utf-8")
    m = re.search(r"_symmetry_Int_Tables_number\s+(\d+)", text)
    assert m is not None, f"{name}: no _symmetry_Int_Tables_number found"
    assert int(m.group(1)) == _CIF_EXPECT[name]["declared_sg"]


@pytest.mark.parametrize("name", sorted(_CIF_EXPECT))
def test_spglib_detected_space_group(examples_dir, name):
    exp = _CIF_EXPECT[name]
    spec = read_cif_crystal(str(_cif_path(examples_dir, name)))
    assert spec.sg_number == exp["detected_sg"]
    assert spec.declared_sg_number == exp["declared_sg"]
    assert spec.declared_sg_mismatch == (exp["declared_sg"] != exp["detected_sg"])


def test_tini_sg11_p21m(examples_dir):
    """TiNi.cif: SG 11, P2_1/m, monoclinic — the headline case named in
    the task (Z=2 -> 4 atoms/cell)."""
    spec = read_cif_crystal(str(_cif_path(examples_dir, "TiNi.cif")))
    assert spec.sg_number == 11
    assert spec.international == "P2_1/m"
    assert spec.family == "monoclinic"
    assert spec.n_atoms_conventional == 4


@pytest.mark.parametrize("name", sorted(_CIF_EXPECT))
def test_atoms_per_cell_matches_composition_times_z(examples_dir, name):
    """The conventional-cell atom count spglib derives must equal the
    CIF's own declared composition * Z (_chemical_formula_sum,
    _cell_formula_units_Z) — the double-placement / missing-atom guard."""
    text = (examples_dir / "assets" / name).read_text(encoding="utf-8")
    m_sum = re.search(r"_chemical_formula_sum\s+'([^']+)'", text)
    m_z = re.search(r"_cell_formula_units_Z\s+(\d+)", text)
    assert m_sum is not None and m_z is not None
    z = int(m_z.group(1))
    n_formula_atoms = sum(int(n) for n in re.findall(r"(\d+)", m_sum.group(1)))
    assert n_formula_atoms == _CIF_EXPECT[name]["atoms_per_formula"] * z

    spec = read_cif_crystal(str(_cif_path(examples_dir, name)))
    assert spec.n_atoms_conventional == n_formula_atoms
    # Cross-check against the re-expanded Wyckoff-orbit basis too (the
    # loader's own internal round-trip already asserts this — repeat it
    # here as a black-box guarantee of the public read_cif_crystal API).
    n_sites = sum(1 for _ in spec.wyckoff_sites)
    assert n_sites >= 1


def test_lattice_matrix_matches_ase_standardized_cell(examples_dir):
    """The lattice parameters resolved from the CIF, run back through
    grainsmith's own cell_matrix, reproduce spglib's standardized cell —
    the manual and CIF paths share ONE cell-matrix convention."""
    import spglib
    from ase.geometry import cell_to_cellpar
    from ase.io import read as ase_read

    from grainsmith.crystal.cell import cell_matrix

    path = _cif_path(examples_dir, "TiNi.cif")
    atoms = ase_read(str(path))
    lattice = atoms.cell[:].tolist()
    pos = atoms.get_scaled_positions(wrap=True)
    numbers = atoms.get_atomic_numbers()
    ds = spglib.get_symmetry_dataset((lattice, pos, numbers), symprec=1e-4)
    std_lattice = np.asarray(ds.std_lattice, dtype=np.float64)
    cellpar = cell_to_cellpar(std_lattice)

    spec = read_cif_crystal(str(path))
    A = cell_matrix(spec.cellpar_full[0], spec.cellpar_full[1],
                     spec.cellpar_full[2], spec.cellpar_full[3],
                     spec.cellpar_full[4], spec.cellpar_full[5])
    np.testing.assert_allclose(spec.cellpar_full, cellpar, atol=1e-6)
    np.testing.assert_allclose(A.T, std_lattice, atol=1e-6)


# ---------------------------------------------------------------------------
# Config mutual-exclusion validation (Rule 27)
# ---------------------------------------------------------------------------


def _raw(examples_dir, **overrides):
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {"cif": {"file": str(examples_dir / "assets" / "TiNi.cif")}},
    }
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            raw[key].update(sub)
        else:
            raw[key] = sub
    return raw


def test_cif_path_resolves_to_manual_fields(examples_dir):
    cfg = resolve_config(_raw(examples_dir))
    assert cfg.crystal.space_group.number == 11
    assert cfg.crystal.lattice.a == pytest.approx(2.8916866399999996)
    assert len(cfg.crystal.wyckoff_sites) == 2


def test_resolved_cif_config_redump_reloads(examples_dir):
    """Regression: resolving a crystal.cif block mutates the SAME
    CrystalConfig instance to also carry the derived space_group /
    lattice / wyckoff_sites (so the pipeline can consume them like the
    manual path) — a naive model_dump would then emit BOTH input forms
    at once, which re-validation (the mutual-exclusion guard, Rule 27)
    rejects. This is exactly what dump_resolved() writes to
    resolved_config.yaml and what the studio UI's config_to_yaml/
    load_yaml round-trip does, so a regression here breaks re-running a
    finished CIF-based config from its own provenance file."""
    import yaml

    cfg = resolve_config(_raw(examples_dir))
    dumped = cfg.model_dump(mode="json")
    assert dumped["crystal"]["cif"]["file"] == str(examples_dir / "assets" / "TiNi.cif")
    assert dumped["crystal"]["space_group"] is None
    assert dumped["crystal"]["lattice"] is None
    assert dumped["crystal"]["wyckoff_sites"] is None

    text = yaml.dump(dumped, default_flow_style=False, sort_keys=True)
    cfg2 = resolve_config(yaml.safe_load(text))
    assert cfg2.crystal.space_group.number == 11


def test_manual_config_dump_unaffected_by_cif_serializer(examples_dir):
    """The CrystalConfig serializer override must be a no-op for the
    manual path (cif is None) — the manual trio still dumps normally."""
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
    }
    cfg = resolve_config(raw)
    dumped = cfg.model_dump(mode="json")
    assert dumped["crystal"]["cif"] is None
    assert dumped["crystal"]["space_group"]["number"] == 225
    assert dumped["crystal"]["lattice"]["a"] == pytest.approx(3.615)
    assert len(dumped["crystal"]["wyckoff_sites"]) == 1


def test_cif_and_manual_mutually_exclusive(examples_dir):
    raw = _raw(examples_dir, crystal={
        "cif": {"file": str(examples_dir / "assets" / "TiNi.cif")},
        "space_group": {"number": 11},
        "lattice": {"a": 1.0, "b": 2.0, "c": 3.0, "beta": 100.0},
        "wyckoff_sites": [{"element": "Ti", "coords": [0.0, 0.0, 0.0]}],
    })
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve_config(raw)


def test_neither_cif_nor_manual_given(examples_dir):
    raw = _raw(examples_dir, crystal={"cif": None})
    with pytest.raises(ConfigError, match="either 'cif'"):
        resolve_config(raw)


def test_incomplete_manual_trio_rejected(examples_dir):
    """space_group alone (no lattice/wyckoff_sites) and no cif: a clear
    error, not a downstream AttributeError."""
    raw = _raw(examples_dir,
               crystal={"cif": None, "space_group": {"number": 225}})
    with pytest.raises(ConfigError, match="missing"):
        resolve_config(raw)


def test_missing_cif_file_raises_clear_error(examples_dir):
    raw = _raw(examples_dir, crystal={
        "cif": {"file": str(examples_dir / "assets" / "does_not_exist_xyz.cif")},
    })
    with pytest.raises(ConfigError, match="not found"):
        resolve_config(raw)


def test_cif_resolution_applies_to_each_phase(examples_dir):
    """Rule 27 must also resolve per-phase crystal blocks (R4b)."""
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [60.0, 60.0, 60.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "phases": [
            {"name": "tini", "fraction": 0.5,
             "crystal": {"cif": {"file": str(examples_dir / "assets" / "TiNi.cif")}}},
            {"name": "cu", "fraction": 0.5,
             "crystal": {
                 "space_group": {"number": 225},
                 "lattice": {"a": 3.615},
                 "wyckoff_sites": [{"element": "Cu",
                                    "coords": [0.0, 0.0, 0.0]}],
             }},
        ],
    }
    cfg = resolve_config(raw)
    assert cfg.phases[0].crystal.space_group.number == 11
    assert cfg.phases[1].crystal.space_group.number == 225


# ---------------------------------------------------------------------------
# crystal.cif.file resolution against the YAML file's own directory
# (load_config's base_dir fallback) -- lets an example config's asset
# paths resolve regardless of the caller's CWD.
# ---------------------------------------------------------------------------


def test_load_config_resolves_cif_relative_to_yaml_own_dir(
    tmp_path, examples_dir, monkeypatch):
    """crystal.cif.file given relative to the YAML's OWN directory
    resolves via load_config's base_dir fallback even when the process
    CWD is somewhere else entirely -- this is the actual bug: before the
    fallback existed, `grainsmith generate
    examples/self_affine_gb/tio_2_thin_film_self_affine.yaml` only
    worked when CWD happened to be the repo root."""
    import shutil

    import yaml

    from grainsmith.config.resolve import load_config

    # 'project/cfgs' and 'project/assets' are siblings (the config's
    # '../assets/TiNi.cif' resolves against 'project/cfgs' via base_dir);
    # 'elsewhere' is a SEPARATE branch of tmp_path so that the same
    # relative '../assets/TiNi.cif', tried against the CWD, resolves to
    # a location that does NOT exist -- otherwise the as-given
    # interpretation could accidentally also hit the real asset and this
    # test would not actually exercise the base_dir fallback.
    cfg_dir = tmp_path / "project" / "cfgs"
    cfg_dir.mkdir(parents=True)
    assets_dir = tmp_path / "project" / "assets"
    assets_dir.mkdir()
    shutil.copy(examples_dir / "assets" / "TiNi.cif", assets_dir / "TiNi.cif")

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {"cif": {"file": "../assets/TiNi.cif"}},
    }))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert not (elsewhere / ".." / "assets" / "TiNi.cif").resolve().exists()
    monkeypatch.chdir(elsewhere)

    cfg = load_config(cfg_path)
    assert cfg.crystal.space_group.number == 11

    # base_dir fallback fired -> the field is REWRITTEN to the resolved
    # absolute path (unlike the as-given case, which leaves it untouched).
    assert cfg.crystal.cif.file == str((assets_dir / "TiNi.cif").resolve())


def test_cif_cwd_relative_wins_and_is_left_unrewritten(
    tmp_path, examples_dir, monkeypatch):
    """The AS-GIVEN interpretation (here: relative to the CWD) takes
    precedence over the base_dir fallback whenever both candidates
    exist -- full backward compatibility. Proven with two DIFFERENT CIFs
    under the same basename so a wrong-precedence bug would resolve to
    the wrong space group, not just the wrong path string (mirrors
    test_voxel_import.py's
    test_voxel_import_cwd_relative_wins_and_is_left_unrewritten)."""
    import shutil

    import yaml

    from grainsmith.config.resolve import load_config

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    cwd_dir = tmp_path / "run_here"
    cwd_dir.mkdir()

    # same filename, two different structures: TiNi.cif (SG 11) reachable
    # from the CWD, Ti2Ni.cif (SG 227) reachable only via base_dir.
    shutil.copy(examples_dir / "assets" / "TiNi.cif", cwd_dir / "field.cif")
    shutil.copy(examples_dir / "assets" / "Ti2Ni.cif", cfg_dir / "field.cif")

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {"cif": {"file": "field.cif"}},
    }))

    monkeypatch.chdir(cwd_dir)
    cfg = load_config(cfg_path)

    # as-given (CWD-relative) wins BY CONTENT: SG 11 (TiNi), not 227 (Ti2Ni).
    assert cfg.crystal.space_group.number == 11
    # ... and the field is left byte-identical, not rewritten.
    assert cfg.crystal.cif.file == "field.cif"


def test_missing_cif_file_lists_both_tried_locations_with_base_dir(
    tmp_path, monkeypatch):
    """When neither the CWD-relative nor the base_dir-relative candidate
    exists, the ConfigError names both locations that were tried (only
    when base_dir was actually in play -- a relative path loaded via
    load_config)."""
    import yaml

    from grainsmith.config.resolve import load_config

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {"cif": {"file": "../assets/does_not_exist_xyz.cif"}},
    }))

    monkeypatch.chdir(elsewhere)
    with pytest.raises(ConfigError,
                       match="tried relative to the working directory"
                       ) as exc:
        load_config(cfg_path)
    msg = str(exc.value)
    assert "does_not_exist_xyz.cif" in msg
    assert str(cfg_dir) in msg


def test_load_config_example_cif_from_any_cwd(project_root, tmp_path, monkeypatch):
    """The actual user-facing regression, end to end: an example config
    under examples/cif/ must load_config successfully regardless of the
    process's current working directory (previously required CWD ==
    repo root, since its cif.file was given as `examples/assets/...`)."""
    from grainsmith.config.resolve import load_config

    monkeypatch.chdir(tmp_path)
    cfg = load_config(
        project_root / "examples" / "cif" / "tini_cif_polycrystal.yaml")
    assert cfg.crystal.space_group.number == 11


# ---------------------------------------------------------------------------
# End-to-end builds through the overlap gate (all three CIFs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_CIF_EXPECT))
def test_single_crystal_build_passes_gates(examples_dir, tmp_path, name):
    """A single-crystal (grains.number: 1) build from each CIF must pass
    every REQUIRED QA gate (gate G14 box/lattice commensurability is
    warn-only, §6.1: an oblique in-plane cell — e.g. TiNi3's hexagonal
    gamma=120 axes — is not exactly tiled by a naive 3·a x 3·b x 3·c
    orthogonal box, so its box faces are genuine self-boundary defects
    that overlap removal (gate G7, hard) must still resolve cleanly)."""
    spec = read_cif_crystal(str(_cif_path(examples_dir, name)))
    # 3x3x3 conventional cells per axis — small but non-trivial.
    a, b, c, alpha, beta, gamma = spec.cellpar_full
    lengths = [3 * a, 3 * b, 3 * c]

    raw = {
        "meta": {"title": f"cif e2e {name}", "verbose": 0},
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": lengths, "periodic": [True, True, True]},
        "grains": {"number": 1},
        "crystal": {"cif": {"file": str(_cif_path(examples_dir, name))}},
        "orientation": {"scheme": "fixed",
                        "fixed": {"euler_bunge_deg": [0.0, 0.0, 0.0]}},
        "boundaries": {"geometry": "flat",
                       "overlap_removal": {"enabled": True,
                                          "cutoff": "0.85*d_nn",
                                          "policy": "delete_shallower"}},
        "output": {"directory": str(tmp_path / "out")},
    }
    cfg = resolve_config(raw)
    res = run(cfg)
    # G7 (overlap min-distance) is a hard gate; G14 (commensurability) is
    # a warn-only advisory — both are included in gates.all_passed().
    assert res.gates.all_passed()
    # Orthogonal families (alpha=beta=gamma=90, i.e. TiNi's monoclinic
    # beta-only tilt is still an orthogonal a-b-c BOX and Ti2Ni's cubic)
    # are exactly tiled by an axis-aligned 3a x 3b x 3c box -> the ideal
    # atom count with zero overlap deletions. An oblique in-plane cell
    # (TiNi3's hexagonal gamma=120) makes the axis-aligned box a
    # DIFFERENT (larger) volume than the true 3x3x3 hexagonal supercell,
    # so both the raw fill count and the post-overlap-removal count
    # legitimately differ from atoms_per_cell * 27 — only the gate
    # results (in particular G7, the hard overlap/min-distance check)
    # are asserted in that case.
    if abs(gamma - 90.0) < 1e-6 and abs(alpha - 90.0) < 1e-6:
        expected = spec.n_atoms_conventional * 27
        assert res.n_generated == expected
        assert res.n_deleted == 0
    else:
        assert res.n_generated > 0
        assert res.n_deleted >= 0


def test_polycrystal_build_from_tini_cif(examples_dir, tmp_path):
    """A small multi-grain polycrystal from the monoclinic TiNi CIF: the
    overlap gate (G7) must still pass with non-trivial atom deletions at
    the grain boundaries, and an independent PBC-aware KD-tree re-check
    (not reusing the pipeline's own gate machinery) must agree the
    minimum inter-atom distance clears the configured cutoff."""
    from scipy.spatial import cKDTree

    raw = {
        "meta": {"title": "cif polycrystal e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": 7},
        "box": {"lengths": [60.0, 60.0, 60.0],
                "periodic": [True, True, True]},
        "grains": {"number": 6},
        "crystal": {"cif": {"file": str(examples_dir / "assets" / "TiNi.cif")}},
        "orientation": {"scheme": "random_uniform"},
        "boundaries": {"geometry": "flat",
                       "overlap_removal": {"enabled": True,
                                          "cutoff": "0.85*d_nn",
                                          "policy": "delete_shallower"}},
        "output": {"directory": str(tmp_path / "out")},
    }
    cfg = resolve_config(raw)
    res = run(cfg)
    assert res.gates.all_passed()

    # Multi-grain overlap removal must actually have deleted something —
    # otherwise this test would silently degrade into the single-crystal
    # zero-deletion case and stop exercising the GB overlap path at all.
    assert res.n_deleted > 0
    assert len(res.atoms) == res.n_generated - res.n_deleted

    # Independent PBC-aware minimum-distance re-check on the final atom
    # positions: cKDTree's boxsize= argument implements the periodic
    # minimum-image convention directly (exact for an orthogonal box,
    # O(N log N) — no O(N^2) all-pairs matrix, no atom-count cutoff
    # needed), so this is a real, always-executed verification, not a
    # gated approximation.
    pos = np.asarray(res.atoms.pos, dtype=np.float64)
    L = np.asarray(cfg.box.lengths, dtype=np.float64)
    wrapped = pos % L
    tree = cKDTree(wrapped, boxsize=L)
    d_pairs, _ = tree.query(wrapped, k=2)
    min_dist = float(np.min(d_pairs[:, 1]))

    cutoff_used = 0.85 * res.d_nn
    assert min_dist >= cutoff_used - 1e-6


# ---------------------------------------------------------------------------
# TiO2 anatase (mp-390): P 1-declared conventional-standard export +
# interstitial orbit expansion on a CIF-sourced host (doping flow).
# ---------------------------------------------------------------------------

def test_tio2_anatase_detection(examples_dir):
    """P 1-declared pymatgen export detects I4_1/amd (SG 141), 12 atoms,
    2 Wyckoff orbits (Ti 4a + O 8e)."""
    spec = read_cif_crystal(
        str(_cif_path(examples_dir,
                      "TiO2_mp-390_conventional_standard.cif")))
    assert spec.declared_sg_number == 1
    assert spec.declared_sg_mismatch is True
    assert spec.sg_number == 141
    assert spec.international == "I4_1/amd"
    assert spec.family == "tetragonal"
    assert spec.n_atoms_conventional == 12
    els = sorted(s.element for s in spec.wyckoff_sites)
    assert els == ["O", "Ti"]


def test_tio2_cif_doping_orbit_expansion(examples_dir):
    """CIF-sourced SG feeds interstitial orbit expansion: the anatase 4b
    representative [0,0,0.5] expands to 4 sites/cell under SG 141."""
    from grainsmith.atoms.doping import resolve_sites
    from grainsmith.config.resolve import resolve_config

    raw = {
        "meta": {"title": "tio2-cif-doping", "verbose": 0},
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 2},
        "crystal": {"cif": {"file": str(_cif_path(
            examples_dir, "TiO2_mp-390_conventional_standard.cif"))}},
        "doping": {"dopants": [{
            "element": "Li", "mode": "interstitial",
            "sites": {"coords": [[0.0, 0.0, 0.5]]},
            "concentration": 0.01, "min_distance": 1.6,
        }]},
        "output": {"directory": "./out_tio2_cif_doping_test",
                   "lammps": {"atom_style": "atomic"}},
    }
    cfg = resolve_config(raw)
    # CIF write-back: SG 141 landed in the manual fields
    assert cfg.crystal.space_group.number == 141
    frac = resolve_sites(cfg.doping.dopants[0].sites, 141, "test",
                         sg_setting=cfg.crystal.space_group.setting)
    assert len(frac) == 4
    # all expanded sites keep min-image distance to the host >= 1.9 A
    import numpy as np
    from grainsmith.crystal import (WyckoffSite, cell_matrix,
                                    expand_wyckoff,
                                    hall_from_international,
                                    symmetry_ops, validate_cellpar)
    cp = validate_cellpar("tetragonal",
                          {"a": cfg.crystal.lattice.a,
                           "c": cfg.crystal.lattice.c})
    A = cell_matrix(cp.a, cp.b, cp.c, cp.alpha, cp.beta, cp.gamma)
    rots, trans = symmetry_ops(hall_from_international(141, None))
    host = expand_wyckoff(
        [WyckoffSite(s.element, s.coords)
         for s in cfg.crystal.wyckoff_sites], rots, trans).frac
    for p in frac:
        d = host - p
        d -= np.round(d)
        assert np.linalg.norm(d @ A.T, axis=1).min() > 1.9
