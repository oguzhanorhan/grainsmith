"""Scientific debug round-trip tests for io/*.py and cli.py (io+cli track).

Focus: LAMMPS/extXYZ writer<->parser fidelity (orthogonal + restricted-
triclinic, atomic + molecular atom_style), edge cases (empty block, single
atom, non-periodic vacuum axis), G10 gate agreement across mode, atomic-mass
sourcing, PLY mesh self-consistency, and the METHODS.md G14 single-crystal
commensurability wording (regression for the defect fixed in this pass:
the generated text used to claim "self-boundary defects" unconditionally,
even when the measured misfit was below COMMENSURATE_TOL).
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pytest

from grainsmith.atoms.fill import AtomBlock
from grainsmith.config.resolve import resolve_config
from grainsmith.constants import ATOMIC_MASSES, COMMENSURATE_TOL
from grainsmith.errors import ConfigError
from grainsmith.io.common import Provenance, fmt
from grainsmith.io.lammps import (
    gate_g10_lammps,
    parse_lammps,
    parse_lammps_header,
    write_lammps,
)
from grainsmith.io.xyz import write_extxyz
from grainsmith.pipeline import run

PROV = Provenance(version="test", timestamp_iso="2026-01-01T00:00:00Z",
                  seed=1, config_sha256="abcdef012345" + "0" * 52,
                  provenance_sha256="fedcba987654" + "0" * 52)


def _atoms(n=10):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10, size=(n, 3))
    species = np.array(["Fe"] * (n // 2) + ["Cu"] * (n - n // 2))
    grain = np.array([i * 3 // n for i in range(n)])
    return AtomBlock(pos=pos, species=species, grain=grain)


TRICLINIC_H = np.array([[10.0, 2.0, 1.0],
                        [0.0, 9.0, 1.5],
                        [0.0, 0.0, 8.0]])


# ---------------------------------------------------------------------------
# LAMMPS round-trip: orthogonal + restricted-triclinic, atomic + molecular
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atom_style", ["atomic", "molecular"])
@pytest.mark.parametrize("cell_matrix", [None, TRICLINIC_H])
def test_lammps_write_parse_roundtrip(tmp_path, atom_style, cell_matrix):
    atoms = _atoms()
    L = (np.diag(cell_matrix) if cell_matrix is not None
         else np.array([10.0, 10.0, 10.0]))
    p = tmp_path / "t.data"
    type_map = write_lammps(p, atoms, L, [True, True, True], atom_style,
                            PROV, cell_matrix=cell_matrix)
    parsed = parse_lammps(p)
    assert parsed["natoms"] == len(atoms)
    assert parsed["ntypes"] == len(type_map)
    assert parsed["atom_style"] == atom_style
    np.testing.assert_allclose(
        parsed["bounds"], [[0.0, L[0]], [0.0, L[1]], [0.0, L[2]]],
        atol=1e-12)
    if cell_matrix is not None:
        H = cell_matrix
        assert parsed["tilt"] is not None
        np.testing.assert_allclose(
            parsed["tilt"], [H[0, 1], H[0, 2], H[1, 2]], atol=1e-12)
        # Every parsed position lies in the [0,1) fractional triclinic cell.
        frac = (np.linalg.inv(H) @ parsed["pos"].T).T
        assert np.all(frac >= -1e-9) and np.all(frac < 1.0 + 1e-9)
    else:
        assert parsed["tilt"] is None
    if atom_style == "molecular":
        np.testing.assert_array_equal(
            parsed["mol"], atoms.grain.astype(np.int64) + 1)
    # ids contiguous 1..N
    np.testing.assert_array_equal(
        np.sort(parsed["ids"]), np.arange(1, len(atoms) + 1))
    # G10 full and sampled modes agree and pass.
    for mode in ("full", "sampled", "off"):
        res = gate_g10_lammps(p, atoms, L, [True, True, True], 0.0,
                              atom_style, mode=mode, cell_matrix=cell_matrix)
        assert res.passed, res.message


def test_lammps_header_matches_full_parse(tmp_path):
    atoms = _atoms()
    p = tmp_path / "t.data"
    write_lammps(p, atoms, np.array([10.0, 10.0, 10.0]), [True, True, True],
                "atomic", PROV, cell_matrix=TRICLINIC_H)
    header = parse_lammps_header(p)
    full = parse_lammps(p)
    assert header["natoms"] == full["natoms"]
    assert header["ntypes"] == full["ntypes"]
    np.testing.assert_allclose(header["bounds"], full["bounds"])
    np.testing.assert_allclose(header["tilt"], full["tilt"])


def test_lammps_tilt_bound_violation_raises(tmp_path):
    """A tilt factor beyond the LAMMPS restricted-triclinic bound
    (|xy| > ax/2) must raise, not silently write an invalid data file —
    reduce_triclinic_tilts is supposed to guarantee this never happens for
    pipeline-generated boxes; this is the defense-in-depth check."""
    H_bad = np.array([[10.0, 8.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 8.0]])
    atoms = _atoms()
    with pytest.raises(ConfigError, match="exceeds the LAMMPS"):
        write_lammps(tmp_path / "bad.data", atoms, np.diag(H_bad),
                    [True, True, True], "atomic", PROV, cell_matrix=H_bad)


# ---------------------------------------------------------------------------
# Edge cases: empty block, single atom, non-periodic vacuum axis
# ---------------------------------------------------------------------------


def test_lammps_empty_atom_block(tmp_path):
    empty = AtomBlock(pos=np.empty((0, 3)), species=np.array([], dtype=str),
                      grain=np.array([], dtype=np.int64))
    p = tmp_path / "empty.data"
    type_map = write_lammps(p, empty, np.array([10.0, 10.0, 10.0]),
                            [True, True, True], "atomic", PROV)
    assert type_map == {}
    parsed = parse_lammps(p)
    assert parsed["natoms"] == 0 and parsed["ntypes"] == 0
    for mode in ("full", "sampled", "off"):
        res = gate_g10_lammps(p, empty, np.array([10.0, 10.0, 10.0]),
                              [True, True, True], 0.0, "atomic", mode=mode)
        assert res.passed


def test_lammps_single_atom(tmp_path):
    one = AtomBlock(pos=np.array([[1.0, 2.0, 3.0]]), species=np.array(["Fe"]),
                    grain=np.array([0]))
    p = tmp_path / "one.data"
    write_lammps(p, one, np.array([10.0, 10.0, 10.0]), [True, True, True],
                "atomic", PROV)
    for mode in ("full", "sampled"):
        res = gate_g10_lammps(p, one, np.array([10.0, 10.0, 10.0]),
                              [True, True, True], 0.0, "atomic", mode=mode,
                              sample_k=1000)
        assert res.passed and res.measured == 1


def test_lammps_nonperiodic_axis_vacuum_bounds(tmp_path):
    """A free (non-periodic) axis gets bounds [0, L+vacuum] and atoms are
    shifted by +vacuum/2 (§8.1); a periodic axis keeps [0, L]."""
    atoms = _atoms(6)
    p = tmp_path / "vac.data"
    write_lammps(p, atoms, np.array([10.0, 10.0, 10.0]), [True, True, False],
                "atomic", PROV, vacuum=5.0)
    parsed = parse_lammps(p)
    np.testing.assert_allclose(
        parsed["bounds"], [[0.0, 10.0], [0.0, 10.0], [0.0, 15.0]])
    res = gate_g10_lammps(p, atoms, np.array([10.0, 10.0, 10.0]),
                          [True, True, False], 5.0, "atomic", mode="full")
    assert res.passed


# ---------------------------------------------------------------------------
# extXYZ round-trip against ASE (spec compliance, not just self-parse)
# ---------------------------------------------------------------------------


def test_extxyz_ase_readback_orthogonal(tmp_path):
    ase = pytest.importorskip("ase.io")
    atoms = _atoms()
    p = tmp_path / "t.extxyz"
    write_extxyz(p, atoms, np.array([10.0, 10.0, 10.0]), [True, True, True],
                PROV)
    a = ase.read(p)
    assert len(a) == len(atoms)
    np.testing.assert_allclose(np.diag(a.cell[:]), [10.0, 10.0, 10.0])
    assert list(a.get_pbc()) == [True, True, True]
    np.testing.assert_array_equal(a.arrays["grain"], atoms.grain)


def test_extxyz_ase_readback_triclinic(tmp_path):
    """Lattice="..." is ROW-major (rows = box vectors) per the extxyz/ASE
    convention — the TRANSPOSE of grainsmith's own column-vector H."""
    ase = pytest.importorskip("ase.io")
    atoms = _atoms()
    p = tmp_path / "t.extxyz"
    write_extxyz(p, atoms, np.diag(TRICLINIC_H), [True, True, True], PROV,
                cell_matrix=TRICLINIC_H)
    a = ase.read(p)
    np.testing.assert_allclose(a.cell[:], TRICLINIC_H.T, atol=1e-9)


# ---------------------------------------------------------------------------
# fmt() shortest-round-trip guarantee (used by every writer for coordinates)
# ---------------------------------------------------------------------------


def test_fmt_is_exact_round_trip():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1e6, 1e6, size=2000)
    for x in xs:
        assert float(fmt(x)) == x


# ---------------------------------------------------------------------------
# Atomic masses: spot-check a handful against IUPAC standard atomic weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("element,expected", [
    ("Fe", 55.845), ("Cu", 63.546), ("Al", 26.982), ("Ti", 47.867),
    ("W", 183.84), ("Au", 196.967), ("Ni", 58.693), ("Zn", 65.38),
])
def test_atomic_masses_match_iupac(element, expected):
    assert ATOMIC_MASSES[element] == pytest.approx(expected, abs=5e-3)


# ---------------------------------------------------------------------------
# METHODS.md — G14 single-crystal commensurability wording must track the
# ACTUAL measured misfit, not assert defects unconditionally (regression).
# ---------------------------------------------------------------------------


def _single_crystal_config(tmp_path, angle_deg):
    raw = {
        "meta": {"title": "methods-g14-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [36.15, 36.15, 36.15],
               "periodic": [True, True, True]},
        "grains": {"number": 1},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "fixed",
                        "fixed": {"euler_bunge_deg": [angle_deg, 0.0, 0.0]}},
        "boundaries": {"geometry": "flat",
                       "overlap_removal": {"enabled": True,
                                           "cutoff": "0.85*d_nn",
                                           "policy": "delete_shallower"}},
        "analysis": {"statistics": False},
        "output": {"directory": str(tmp_path), "methods_snippet": True},
    }
    return resolve_config(raw)


def test_methods_md_commensurate_single_crystal_no_defect_claim(tmp_path):
    config = _single_crystal_config(tmp_path, angle_deg=0.0)
    res = run(config, jobs=1)
    g14 = next(r for r in res.gates.results() if r.gate == "G14")
    assert g14.measured <= COMMENSURATE_TOL
    text = (tmp_path / "METHODS.md").read_text(encoding="utf-8")
    assert "commensurate with the box" in text
    assert "introduce no self-boundary defect" in text
    assert "self-boundary defects" not in text.split(
        "commensurate with the box")[1].split(".")[0]


def test_methods_md_incommensurate_single_crystal_states_defect(tmp_path):
    config = _single_crystal_config(tmp_path, angle_deg=15.0)
    res = run(config, jobs=1)
    g14 = next(r for r in res.gates.results() if r.gate == "G14")
    assert g14.measured > COMMENSURATE_TOL
    text = (tmp_path / "METHODS.md").read_text(encoding="utf-8")
    assert "INCOMMENSURATE with the box" in text
    assert "self-boundary defects" in text


# ---------------------------------------------------------------------------
# METHODS.md — no numbered reference list, self-contained author-year
# citations instead (regression for the reference-list removal).
# ---------------------------------------------------------------------------


def test_methods_md_no_reference_list_and_author_year_citations(tmp_path):
    config = _single_crystal_config(tmp_path, angle_deg=0.0)
    run(config, jobs=1)
    text = (tmp_path / "METHODS.md").read_text(encoding="utf-8")
    assert "## References" not in text
    assert "(Togo & Tanaka 2018)" in text          # spglib round-trip (gate G2)
    assert re.search(r"\[\d+(,\d+)*\]", text) is None


# ---------------------------------------------------------------------------
# METHODS.md — warp pre-warp base must be named (drift fix): the opening
# tessellation sentence used to always say "periodic Voronoi tessellation"
# even when curved.base was 'additive_weights' or 'anisotropic'.
# ---------------------------------------------------------------------------


def _warp_base_config(tmp_path, base, **curved_extra):
    curved = {"method": "warp", "base": base, "amplitude": 1.0,
             "correlation_length": 15.0}
    curved.update(curved_extra)
    raw = {
        "meta": {"title": "methods-warp-base-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 7},
        "box": {"lengths": [50.0, 50.0, 50.0], "periodic": [True, True, True]},
        "grains": {"number": 6},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": {"geometry": "curved", "curved": curved},
        "analysis": {"statistics": False},
        "output": {"directory": str(tmp_path / "out")},
    }
    return resolve_config(raw)


def test_methods_md_warp_base_additive_weights_named(tmp_path):
    config = _warp_base_config(tmp_path, "additive_weights", weight_sigma=1.0)
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "Johnson–Mehl) diagram of the seed points" in text
    assert "pre-warp base tessellation" in text
    assert "periodic Voronoi tessellation" not in text


def test_methods_md_warp_base_anisotropic_named(tmp_path):
    config = _warp_base_config(tmp_path, "anisotropic",
                               aspect_ratio_range=[0.7, 1.4])
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "ellipsoidal-metric (anisotropic) diagram of the seed points" in text
    assert "pre-warp base tessellation" in text
    assert "periodic Voronoi tessellation" not in text


def test_methods_md_warp_base_flat_still_names_voronoi(tmp_path):
    config = _warp_base_config(tmp_path, "flat")
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "periodic Voronoi tessellation" in text
    assert "pre-warp base tessellation" not in text


# ---------------------------------------------------------------------------
# METHODS.md — odf_components must describe the ACTUAL mix of Euler/fiber/
# random sub-components instead of hardcoding "Euler-angle centres with
# isotropic spread" regardless of what is configured (drift fix).
# ---------------------------------------------------------------------------


def _odf_components_config(tmp_path, components):
    raw = {
        "meta": {"title": "methods-odf-mix-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 3},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 6},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "odf_components", "components": components},
        "analysis": {"statistics": False},
        "output": {"directory": str(tmp_path / "out")},
    }
    return resolve_config(raw)


def test_methods_md_odf_components_describes_mixed_component_types(tmp_path):
    config = _odf_components_config(tmp_path, [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "spread_deg": 5.0, "weight": 1.0},
        {"fiber": {"crystal_axis": [1, 1, 1], "sample_direction": "z",
                   "spread_deg": 8.0}, "weight": 1.0},
        {"random": True, "weight": 0.5},
    ])
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "3 weighted texture components" in text
    assert "1 Euler-angle centre " in text
    assert "1 fiber component" in text
    assert "1 Haar-uniform random component" in text


def test_methods_md_odf_components_euler_only_omits_other_kinds(tmp_path):
    config = _odf_components_config(tmp_path, [
        {"euler_bunge_deg": [0.0, 0.0, 0.0], "weight": 1.0},
        {"euler_bunge_deg": [45.0, 10.0, 0.0], "weight": 1.0},
    ])
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "2 Euler-angle centres" in text
    assert "fiber component" not in text
    assert "Haar-uniform random component" not in text


# ---------------------------------------------------------------------------
# METHODS.md — analysis.gb_character (tilt/twist/mixed classification +
# boundary-plane Miller indices) had no branch at all (drift fix).
# ---------------------------------------------------------------------------


def _gb_character_config(tmp_path, gb_character):
    raw = {
        "meta": {"title": "methods-gb-character-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 5},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "analysis": {"gb_character": gb_character, "statistics": False},
        "output": {"directory": str(tmp_path / "out")},
    }
    return resolve_config(raw)


def test_methods_md_gb_character_sentence_present_when_enabled(tmp_path):
    config = _gb_character_config(tmp_path, True)
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "twist" in text and "tilt" in text and "mixed" in text
    assert "boundary-plane Miller indices" in text


def test_methods_md_gb_character_sentence_absent_when_disabled(tmp_path):
    config = _gb_character_config(tmp_path, False)
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "boundary-plane Miller indices" not in text


# ---------------------------------------------------------------------------
# PLY mesh writer: face indices in-bounds, grain pairs consistent with
# boundaries.csv (requires scikit-image; skip if the [mesh] extra absent).
# ---------------------------------------------------------------------------


def test_ply_mesh_self_consistent(tmp_path):
    pytest.importorskip("skimage")
    plyfile = pytest.importorskip("plyfile")
    raw = {
        "meta": {"title": "mesh-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 42},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4, "min_seed_distance": "auto"},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": {
            "geometry": "curved",
            # amplitude retuned 2.0 -> 0.6 (warp.py DC-mode-exclusion fix:
            # removing the k=0 mode from the gaussian spectrum removes a
            # rigid-translation contribution that used to inflate the RMS
            # calibration denominator, so the same amplitude now delivers a
            # larger max‖∇u‖ — see tessellation/warp.py docstring).
            "curved": {"method": "warp", "base": "flat", "amplitude": 0.6,
                      "correlation_length": 10.0},
            "overlap_removal": {"enabled": True, "cutoff": "0.85*d_nn",
                                "policy": "delete_shallower"},
        },
        "analysis": {"voxel_grid": "auto", "gb_character": True,
                    "csl": False},
        "output": {"directory": str(tmp_path),
                  "lammps": {"atom_style": "molecular"},
                  "gnuplot": {"enabled": False},
                  "mesh": {"enabled": True}},
    }
    config = resolve_config(raw)
    run(config, jobs=1)
    ply = plyfile.PlyData.read(str(tmp_path / "boundaries.ply"))
    n_v = len(ply["vertex"].data)
    faces = ply["face"].data
    if len(faces):
        max_idx = max(int(np.max(row[0])) for row in faces)
        assert max_idx < n_v
        ply_pairs = {(int(r[1]), int(r[2])) for r in faces}
        with open(tmp_path / "boundaries.csv") as fh:
            csv_pairs = {(int(r["grain_i"]), int(r["grain_j"]))
                        for r in csv.DictReader(fh)}
        assert ply_pairs.issubset(csv_pairs)


# ---------------------------------------------------------------------------
# CLI: --jobs 0 resolves to all cores but never changes bytes; error surfaces
# ---------------------------------------------------------------------------


def test_cli_generate_jobs_variants_byte_identical(tmp_path, monkeypatch):
    """--jobs 1 vs --jobs 2 must produce byte-identical polycrystal.data
    (P3.2: chunked/parallel formatting is byte-identical to serial) — same
    config (and hence same provenance line down to the config sha256),
    run from two working directories so only the atom-row FORMATTING
    parallelism differs, not the hashed config path."""
    from grainsmith.cli import _cmd_generate
    import argparse

    root1 = tmp_path / "run1"
    root2 = tmp_path / "run2"
    root1.mkdir()
    root2.mkdir()
    template = (
        "seed: {mode: fixed, value: 5}\n"
        "box: {lengths: [20.0, 20.0, 20.0]}\n"
        "grains: {number: 3}\n"
        "crystal:\n"
        "  space_group: {number: 225}\n"
        "  lattice: {a: 3.615}\n"
        "  wyckoff_sites: [{element: Cu, coords: [0.0, 0.0, 0.0]}]\n"
        "orientation: {scheme: random_uniform}\n"
        "analysis: {statistics: false}\n"
        'output: {directory: "./out", methods_snippet: false}\n'
    )
    monkeypatch.chdir(root1)
    (root1 / "c.yaml").write_text(template, encoding="utf-8")
    rc1 = _cmd_generate(argparse.Namespace(config="c.yaml", jobs=1,
                                          max_rss=None))
    monkeypatch.chdir(root2)
    (root2 / "c.yaml").write_text(template, encoding="utf-8")
    rc2 = _cmd_generate(argparse.Namespace(config="c.yaml", jobs=2,
                                          max_rss=None))
    assert rc1 == 0 and rc2 == 0
    _ts = __import__("re").compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    b1 = _ts.sub(b"<TS>", (root1 / "out" / "polycrystal.data").read_bytes())
    b2 = _ts.sub(b"<TS>", (root2 / "out" / "polycrystal.data").read_bytes())
    assert b1 == b2


def test_cli_generate_missing_config_exit_code(tmp_path, capsys):
    from grainsmith.cli import _cmd_generate
    import argparse

    rc = _cmd_generate(argparse.Namespace(
        config=str(tmp_path / "nope.yaml"), jobs=1, max_rss=None))
    assert rc == 1
    assert "Configuration error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# METHODS.md perturbed_distance amplitude_convention wording: curved.amplitude
# means a DIFFERENT physical quantity depending on amplitude_convention, and
# the generated paragraph must say which rather than labeling both
# identically.
# ---------------------------------------------------------------------------


def _perturbed_config(tmp_path, amplitude_convention="total_rms",
                      amplitude=0.3, reference_wavelength=None):
    curved: dict = {
        "method": "perturbed_distance", "spectrum": "self_affine",
        "amplitude": amplitude, "hurst": 0.7, "l_min": 8.0, "l_max": 30.0,
        "amplitude_convention": amplitude_convention,
    }
    if reference_wavelength is not None:
        curved["reference_wavelength"] = reference_wavelength
    raw = {
        "meta": {"title": "methods-perturbed-test", "verbose": 0},
        "seed": {"mode": "fixed", "value": 11},
        "box": {"lengths": [70.0, 70.0, 70.0], "periodic": [True, True, True]},
        "grains": {"number": 8, "min_seed_distance": "auto"},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": {"geometry": "curved", "curved": curved},
        "output": {"directory": str(tmp_path / "out")},
    }
    return resolve_config(raw)


def test_methods_md_total_rms_convention_reports_plain_rms_amplitude(tmp_path):
    """Default amplitude_convention: total_rms keeps the pre-existing,
    unqualified 'RMS amplitude <A> Å' phrasing -- no kappa/reference
    wavelength text leaks into a methods paragraph that does not need it."""
    config = _perturbed_config(tmp_path, amplitude_convention="total_rms",
                               amplitude=0.3)
    res = run(config, jobs=1)
    assert res.gates.all_passed()
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "RMS amplitude 0.3 Å" in text
    assert "reference wavelength" not in text
    assert "equivalent whole-band RMS amplitude" not in text


def test_methods_md_reference_wavelength_convention_states_both_amplitudes(tmp_path):
    """amplitude_convention: reference_wavelength must report BOTH the
    config's own reference-octave amplitude AND its equivalent total-RMS
    amplitude (plus kappa) -- reporting only the raw `amplitude` value
    under this convention would silently misrepresent it as a total-band
    RMS, which it is not."""
    config = _perturbed_config(tmp_path, amplitude_convention="reference_wavelength",
                               amplitude=0.3)
    res = run(config, jobs=1)
    assert res.gates.all_passed()
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "RMS amplitude 0.3 Å at the 30 Å reference wavelength" in text
    assert "equivalent whole-band RMS amplitude" in text
    assert "κ = " in text


# ---------------------------------------------------------------------------
# METHODS.md D_b (self_affine box-counting) wording must use the gate
# G20's own "roughness index, not a converged fractal dimension" framing,
# and cite the box-counting TARGET paper (braun2020fractal, Braun et al.
# 2020) distinctly from the general phenomenology paper (fractal_gb,
# Braun et al. 2018) — both a drift fix and a citation-attribution fix.
# ---------------------------------------------------------------------------


def test_methods_md_db_sentence_uses_roughness_index_framing_and_braun2020(tmp_path):
    config = _perturbed_config(tmp_path, amplitude_convention="total_rms",
                               amplitude=0.3)
    run(config, jobs=1)
    text = (tmp_path / "out" / "METHODS.md").read_text(encoding="utf-8")
    assert "(Braun et al. 2020)" in text
    assert "Braun et al. 2018" in text            # general phenomenology, kept
    assert "box-counting roughness index" in text
    # old overclaim ("...produces a genuinely self-affine boundary — the
    # box-counting dimension estimated from 2D cross-sections is D_b =")
    # is gone; the softened "can produce" phrasing replaces it.
    assert "this level-set rule can produce a genuinely self-affine boundary" in text
    assert "the box-counting dimension estimated from 2D cross-sections is" \
        not in text
    assert "not a converged, large-effect-size fractal-dimension" in text
    assert "flat-Voronoi limit D_b = 1.000" in text
