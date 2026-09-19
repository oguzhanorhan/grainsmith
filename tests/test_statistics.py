"""Tests for publication statistics, microstructure.json, METHODS.md,
EBSD-like section.

Pins: analytic sphericity of a cube; EXACT S_V / L_V on constructed flat
tessellations and synthetic voxel fields; log-normal fit parameter
recovery; METHODS.md mentions exactly the methods used; JSON round-trip
with strict (NaN-free) serialization; column-set constants.
"""
import csv
import json
import re

import numpy as np
import pytest

from grainsmith.analysis.statistics import (
    compute_statistics,
    equivalent_diameters,
    grain_surface_areas_flat,
    grain_surface_areas_voxel,
    lognormal_fit,
    sphericity,
    triple_line_length_flat,
    triple_line_length_voxel,
)
from grainsmith.config.resolve import resolve_config
from grainsmith.io import (
    MICROSTRUCTURE_SCHEMA,
    SECTION_COLUMNS,
    SECTION_COLUMNS_PHASES,
    STATISTICS_COLUMNS,
)
from grainsmith.pipeline import run
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.voxel import VoxelGrid

SEED = 20260612
L40 = np.array([40.0, 40.0, 40.0])
PER = [True, True, True]


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _stat(rows, section, key):
    for r in rows:
        if r["section"] == section and r["key"] == key:
            return r["value"]
    raise KeyError((section, key))


# ---------------------------------------------------------------------------
# Unit pins: estimators
# ---------------------------------------------------------------------------


def test_equivalent_diameter_pin():
    """V = π/6 ⇒ d_eq = 1 (definition pin)."""
    d = equivalent_diameters(np.array([np.pi / 6.0]))
    assert d[0] == pytest.approx(1.0, rel=1e-14)


def test_sphericity_cube_pin():
    """Analytic cube: Ψ = (π/6)^(1/3) for V = a³, A = 6a²."""
    a = 3.7
    psi = sphericity(np.array([a**3]), np.array([6.0 * a**2]))
    assert psi[0] == pytest.approx((np.pi / 6.0) ** (1.0 / 3.0), rel=1e-12)


def test_sphericity_sphere_is_one():
    psi = sphericity(np.array([4.0 * np.pi / 3.0]), np.array([4.0 * np.pi]))
    assert psi[0] == pytest.approx(1.0, rel=1e-12)


def test_lognormal_fit_recovery():
    """MLE recovers (μ, σ) of a true log-normal sample; KS does not
    reject."""
    rng = np.random.default_rng(42)
    d = rng.lognormal(mean=1.2, sigma=0.3, size=600)
    fit = lognormal_fit(d)
    assert fit["mu_hat"] == pytest.approx(1.2, abs=0.05)
    assert fit["sigma_hat"] == pytest.approx(0.3, abs=0.03)
    assert fit["ks_p"] > 0.01
    assert 0.0 < fit["ks_statistic"] < 0.1


def test_lognormal_fit_degenerate():
    """Equal diameters (σ̂ at roundoff): the KS pair is NaN, not a
    numerically meaningless near-delta test (LOGNORMAL_SIGMA_MIN)."""
    fit = lognormal_fit(np.full(10, 2.5))
    assert fit["sigma_hat"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fit["ks_statistic"]) and np.isnan(fit["ks_p"])
    one = lognormal_fit(np.array([2.5]))
    assert np.isnan(one["sigma_hat"])


# ---------------------------------------------------------------------------
# Unit pins: exact flat S_V / L_V / surfaces
# ---------------------------------------------------------------------------


def test_flat_bicrystal_exact_pins():
    """Fully periodic bicrystal split along z: pair area = 2·Lx·Ly
    (two sheets), per-grain surface = 2·Lx·Ly + 4·Lx·(Lz/2) (z sheets +
    periodic self-image x/y faces), and NO triple junctions."""
    seeds = np.array([[20.0, 20.0, 10.0], [20.0, 20.0, 30.0]])
    tess = FlatTessellation(seeds, L40, PER)
    surf = grain_surface_areas_flat(tess)
    np.testing.assert_allclose(surf, [6400.0, 6400.0], rtol=1e-12)
    assert triple_line_length_flat(tess, L40, PER) == 0.0


def test_flat_tricrystal_triple_lines_exact():
    """3 columnar grains (equal z): the periodic 2D Voronoi of N seeds has
    exactly 2N triple points (torus Euler formula), so the triple-line
    length is EXACTLY 6·Lz — for ANY generic seed positions."""
    seeds = np.array([[5.0, 7.0, 20.0],
                      [25.0, 13.0, 20.0],
                      [14.0, 31.0, 20.0]])
    tess = FlatTessellation(seeds, L40, PER)
    assert triple_line_length_flat(tess, L40, PER) == \
        pytest.approx(6.0 * 40.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Unit pins: voxel estimators
# ---------------------------------------------------------------------------


def _quadrant_labels(n=8):
    lab = np.zeros((n, n, n), dtype=np.int32)
    lab[n // 2:, :n // 2, :] = 1
    lab[:n // 2, n // 2:, :] = 2
    lab[n // 2:, n // 2:, :] = 3
    return lab


def test_voxel_triple_lines_quadrant_pin():
    """2×2 columnar quadrants: 4 junction lines along z (each with 4
    distinct labels around it) ⇒ L = 4·Lz exactly — the voxel-edge
    estimator is exact for axis-aligned junctions."""
    vg = VoxelGrid(_quadrant_labels(), L40, (8, 8, 8), 4)
    assert triple_line_length_voxel(vg, PER) == pytest.approx(160.0)


def test_voxel_triple_lines_bicrystal_zero():
    lab = np.zeros((8, 8, 8), dtype=np.int32)
    lab[:, :, 4:] = 1
    vg = VoxelGrid(lab, L40, (8, 8, 8), 2)
    assert triple_line_length_voxel(vg, PER) == 0.0


def test_voxel_surface_two_slab_pin():
    """Two z-slabs, fully periodic: each grain sees 2 GB sheets of
    Lx·Ly ⇒ 3200 Å² — the face-count estimator is exact for axis-aligned
    boundaries."""
    lab = np.zeros((8, 8, 8), dtype=np.int32)
    lab[:, :, 4:] = 1
    vg = VoxelGrid(lab, L40, (8, 8, 8), 2)
    np.testing.assert_allclose(grain_surface_areas_voxel(vg, PER),
                               [3200.0, 3200.0])


def test_voxel_surface_free_axis_walls():
    """Slab geometry (free z): the wrap pair is not a contact, but each
    grain gains a box-wall free surface: 1 GB sheet + 1 wall = 3200 Å²."""
    lab = np.zeros((8, 8, 8), dtype=np.int32)
    lab[:, :, 4:] = 1
    vg = VoxelGrid(lab, L40, (8, 8, 8), 2)
    np.testing.assert_allclose(
        grain_surface_areas_voxel(vg, [True, True, False]),
        [3200.0, 3200.0])


# ---------------------------------------------------------------------------
# Unit: compute_statistics on a constructed tessellation
# ---------------------------------------------------------------------------


class _Rep:
    """Minimal report stubs for compute_statistics."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_compute_statistics_bicrystal_sections():
    """S_V = 2/Lz exactly; L_V = 0; equal volumes ⇒ degenerate fit."""
    seeds = np.array([[20.0, 20.0, 10.0], [20.0, 20.0, 30.0]])
    tess = FlatTessellation(seeds, L40, PER)
    volumes = np.array([c.volume for c in tess.cells])
    greps = [_Rep(n_neighbors=1), _Rep(n_neighbors=1)]
    breps = [_Rep(area_A2=3200.0, misorientation_deg=45.0,
                  character="mixed", csl_sigma="", phase_i="", phase_j="")]
    stats = compute_statistics(tess, volumes, greps, breps, L40, PER)
    assert stats["boundaries"]["S_V_per_A"] == pytest.approx(2.0 / 40.0)
    assert stats["boundaries"]["estimator"] == "exact"
    assert stats["triple_junctions"]["L_V_per_A2"] == 0.0
    assert stats["grain_size"]["n_grains"] == 2
    assert stats["grain_size"]["d_eq_std_A"] == pytest.approx(0.0)
    assert np.isnan(stats["grain_size"]["lognormal_ks_p"])
    assert stats["mdf"]["mean_misorientation_deg"] == pytest.approx(45.0)
    assert stats["mdf"]["lagb_area_fraction"] == 0.0
    assert stats["gb_character"]["area_fraction_mixed"] == 1.0
    assert stats["topology"]["n_grains_with_1_faces"] == 2


def test_compute_statistics_roughness_section():
    seeds = np.array([[20.0, 20.0, 10.0], [20.0, 20.0, 30.0]])
    tess = FlatTessellation(seeds, L40, PER)
    volumes = np.array([c.volume for c in tess.cells])
    stats = compute_statistics(
        tess, volumes, [_Rep(n_neighbors=1)] * 2, [], L40, PER,
        hurst_target=0.8, hurst_estimated=0.74)
    assert stats["roughness"] == {"hurst_target": 0.8,
                                  "hurst_estimated": 0.74}


# ---------------------------------------------------------------------------
# Column pins
# ---------------------------------------------------------------------------


def test_column_constants_pinned():
    assert STATISTICS_COLUMNS == ["section", "key", "value"]
    assert SECTION_COLUMNS == ["x", "y", "grain_id",
                               "phi1_deg", "Phi_deg", "phi2_deg"]
    assert SECTION_COLUMNS_PHASES == [*SECTION_COLUMNS, "phase"]
    assert MICROSTRUCTURE_SCHEMA == "grainsmith/microstructure/v1"


# ---------------------------------------------------------------------------
# End-to-end: single-phase Cu (module-scoped run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cu_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("r5_cu") / "out"
    cfg = resolve_config({
        "meta": {"title": "r5 e2e", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 4},
        "crystal": {"space_group": {"number": 225},
                    "lattice": {"a": 3.615},
                    "wyckoff_sites": [{"element": "Cu",
                                       "coords": [0.0, 0.0, 0.0]}]},
        "analysis": {"csl": True,
                     "section": {"axis": "z", "position": 0.5}},
        "output": {"directory": str(outdir)},
    })
    return run(cfg)


def test_e2e_statistics_csv(cu_run):
    res = cu_run
    assert res.gates.all_passed()
    path = res.outdir / "statistics.csv"
    assert path.exists()
    with path.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == STATISTICS_COLUMNS
    rows = _csv_rows(path)
    sections = {r["section"] for r in rows}
    assert {"grain_size", "sphericity", "topology", "boundaries",
            "triple_junctions", "gb_character", "csl",
            "mdf"} <= sections

    # S_V consistency: statistics row == Σ boundary areas / V_box (exact)
    a_total = sum(r.area_A2 for r in res.boundary_reports)
    assert float(_stat(rows, "boundaries", "S_V_per_A")) == \
        pytest.approx(a_total / 40.0**3, rel=1e-12)
    assert _stat(rows, "boundaries", "estimator") == "exact"
    assert _stat(rows, "triple_junctions", "estimator") == "exact"
    assert float(_stat(rows, "triple_junctions", "length_total_A")) > 0.0

    # grain size: n + mean pin against the exact volumes
    d = equivalent_diameters(np.array([r.volume_A3
                                       for r in res.grain_reports]))
    assert int(_stat(rows, "grain_size", "n_grains")) == 4
    assert float(_stat(rows, "grain_size", "d_eq_mean_A")) == \
        pytest.approx(float(np.mean(d)), rel=1e-12)

    # character + CSL fractions partition the GB area
    char_sum = sum(float(r["value"]) for r in rows
                   if r["section"] == "gb_character")
    assert char_sum == pytest.approx(1.0, rel=1e-9)
    csl_sum = sum(float(r["value"]) for r in rows if r["section"] == "csl")
    assert csl_sum == pytest.approx(1.0, rel=1e-9)

    # in-memory dict matches the file
    assert res.statistics is not None
    assert res.statistics["boundaries"]["n_boundaries"] == \
        int(_stat(rows, "boundaries", "n_boundaries"))


def test_e2e_microstructure_json(cu_run):
    res = cu_run
    doc = json.loads((res.outdir / "microstructure.json")
                     .read_text(encoding="utf-8"))
    assert doc["schema"] == MICROSTRUCTURE_SCHEMA
    assert doc["provenance"]["seed"] == SEED
    assert doc["config"]["grains"]["number"] == 4
    assert len(doc["grains"]) == 4
    assert len(doc["boundaries"]) == len(res.boundary_reports)
    from grainsmith.io import BOUNDARIES_COLUMNS, GRAINS_COLUMNS
    assert list(doc["grains"][0]) == GRAINS_COLUMNS
    assert list(doc["boundaries"][0]) == BOUNDARIES_COLUMNS
    gates = {g["gate"]: g["passed"] for g in doc["gates"]}
    # G23 (report-only) fires for every single-phase run that reaches the
    # MDF-histogram block, independent of orientation.mdf_target. G26
    # (report-only) fires whenever per-grain atom counts exist.
    assert gates == {f"G{k}": True for k in range(1, 11)} | \
        {"G23": True, "G26": True}
    # statistics embedded == RunResult.statistics (NaN → null)
    assert set(doc["statistics"]) == set(res.statistics)
    assert doc["statistics"]["boundaries"]["S_V_per_A"] == \
        pytest.approx(res.statistics["boundaries"]["S_V_per_A"])
    # strict JSON: no NaN tokens anywhere
    raw = (res.outdir / "microstructure.json").read_text(encoding="utf-8")
    assert "NaN" not in raw


def test_e2e_methods_md(cu_run):
    res = cu_run
    text = (res.outdir / "METHODS.md").read_text(encoding="utf-8")
    # mentions exactly the methods used (string pins)
    assert "random sequential adsorption" in text
    assert "Haar-uniformly" in text
    assert "Voronoi tessellation" in text
    assert "spglib" in text
    assert "Brandon criterion" in text           # csl: true
    assert "Mackenzie" in text
    # methods NOT used must not appear
    assert "optimal transport" not in text
    assert "self-affine" not in text
    assert "Lloyd" not in text
    # no numbered reference list: citations are self-contained inline
    # author-year labels read straight off constants.METHODS_BIBLIOGRAPHY
    assert "## References" not in text
    assert "(Togo & Tanaka 2018)" in text
    assert "(Mackenzie 1958)" in text
    # Brandon criterion sentence cites both the criterion itself and the
    # CSL table it's applied against (STEP 3a: closes a previously
    # undocumented attribution gap, constants.CSL_TABLE).
    assert "(Brandon 1966; Randle & Engler 2000)" in text
    assert not re.search(r"\[\d+(,\d+)*\]", text)  # no bare [1]-style markers


def test_e2e_section_slice(cu_run):
    from grainsmith.constants import SECTION_GRID

    res = cu_run
    path = res.outdir / "slice_z0.5.csv"
    assert path.exists()
    rows = _csv_rows(path)
    assert list(rows[0]) == SECTION_COLUMNS
    assert len(rows) == SECTION_GRID * SECTION_GRID
    gids = {int(r["grain_id"]) for r in rows}
    assert gids <= set(range(4)) and len(gids) > 1
    # Euler angles match the grain report of that grain
    r0 = rows[0]
    g = int(r0["grain_id"])
    assert float(r0["phi1_deg"]) == \
        pytest.approx(res.grain_reports[g].euler_phi1_deg, rel=1e-12)
    # pixel-center coordinates inside the box
    assert 0.0 < float(r0["x"]) < 40.0 and 0.0 < float(r0["y"]) < 40.0


def test_statistics_disabled(tmp_path):
    cfg = resolve_config({
        "meta": {"verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [30.0, 30.0, 30.0]},
        "grains": {"number": 2},
        "crystal": {"space_group": {"number": 225},
                    "lattice": {"a": 3.615},
                    "wyckoff_sites": [{"element": "Cu",
                                       "coords": [0.0, 0.0, 0.0]}]},
        "analysis": {"statistics": False},
        "output": {"directory": str(tmp_path / "out"),
                   "methods_snippet": False},
    })
    res = run(cfg)
    assert res.gates.all_passed()
    assert res.statistics is None
    assert not (res.outdir / "statistics.csv").exists()
    assert not (res.outdir / "microstructure.json").exists()
    assert not (res.outdir / "METHODS.md").exists()


# ---------------------------------------------------------------------------
# End-to-end: single crystal (degenerate distributions stay well-defined)
# ---------------------------------------------------------------------------


def test_single_crystal_statistics(tmp_path):
    a = 3.615
    box = 10 * a
    cfg = resolve_config({
        "meta": {"verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [box, box, box]},
        "grains": {"number": 1},
        "crystal": {"space_group": {"number": 225},
                    "lattice": {"a": a},
                    "wyckoff_sites": [{"element": "Cu",
                                       "coords": [0.0, 0.0, 0.0]}]},
        "orientation": {"scheme": "fixed",
                        "fixed": {"quaternion": [1.0, 0.0, 0.0, 0.0]}},
        "output": {"directory": str(tmp_path / "out")},
    })
    res = run(cfg)
    assert res.gates.all_passed()
    s = res.statistics
    assert s["grain_size"]["n_grains"] == 1
    assert s["grain_size"]["d_eq_mean_A"] == \
        pytest.approx((6.0 * box**3 / np.pi) ** (1.0 / 3.0), rel=1e-12)
    # one grain == the cubic box: sphericity is the cube pin
    assert s["sphericity"]["mean"] == \
        pytest.approx((np.pi / 6.0) ** (1.0 / 3.0), rel=1e-12)
    assert s["boundaries"]["S_V_per_A"] == 0.0
    assert s["triple_junctions"]["L_V_per_A2"] == 0.0
    assert s["mdf"]["n_boundaries"] == 0
    # NaN scalars survive the CSV ("nan") and the JSON (null)
    rows = _csv_rows(res.outdir / "statistics.csv")
    assert _stat(rows, "grain_size", "lognormal_sigma_hat") == "nan"
    doc = json.loads((res.outdir / "microstructure.json")
                     .read_text(encoding="utf-8"))
    assert doc["statistics"]["grain_size"]["lognormal_sigma_hat"] is None
    text = (res.outdir / "METHODS.md").read_text(encoding="utf-8")
    assert "single crystal" in text and "G14" in text


# ---------------------------------------------------------------------------
# End-to-end: SDOT citations + lognormal fit on an R1-fitted structure
# ---------------------------------------------------------------------------


def test_sdot_methods_and_fit(tmp_path):
    """The log-normal fit recovers the R1 target shape and
    METHODS.md cites the SDOT chain."""
    sigma_log = 0.3
    cfg = resolve_config({
        "meta": {"verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [40.0, 40.0, 40.0]},
        "grains": {"number": 6,
                   "size_distribution": {"type": "lognormal",
                                         "sigma_log": sigma_log}},
        "crystal": {"space_group": {"number": 225},
                    "lattice": {"a": 3.615},
                    "wyckoff_sites": [{"element": "Cu",
                                       "coords": [0.0, 0.0, 0.0]}]},
        "output": {"directory": str(tmp_path / "out")},
    })
    res = run(cfg)
    assert res.gates.all_passed()
    # the fitted σ̂ is the sample std of ln d of the 6 ACHIEVED cells —
    # compare against the same statistic of the achieved volumes (exact),
    # not the population value (n = 6).
    d = equivalent_diameters(np.array([r.volume_A3
                                       for r in res.grain_reports]))
    assert res.statistics["grain_size"]["lognormal_sigma_hat"] == \
        pytest.approx(float(np.std(np.log(d), ddof=1)), rel=1e-12)
    text = (res.outdir / "METHODS.md").read_text(encoding="utf-8")
    assert "semi-discrete optimal transport" in text
    assert "Laguerre (power) tessellation" in text
    assert "gate G11" in text
    assert "Kitagawa" in text


# ---------------------------------------------------------------------------
# End-to-end: multiphase (R4b × R5 — phase-aware statistics)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ti_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("r5_ti") / "out"
    cfg = resolve_config({
        "meta": {"title": "r5 ti", "verbose": 0},
        "seed": {"mode": "fixed", "value": SEED},
        "box": {"lengths": [50.0, 50.0, 50.0]},
        "grains": {"number": 8},
        "phases": [
            {"name": "alpha", "fraction": 0.6,
             "crystal": {"space_group": {"number": 194},
                         "lattice": {"a": 2.951, "c": 4.684},
                         "wyckoff_sites": [
                             {"element": "Ti",
                              "coords": [1.0 / 3.0, 2.0 / 3.0, 0.25],
                              "letter": "c"}]}},
            {"name": "beta", "fraction": 0.4,
             "crystal": {"space_group": {"number": 229},
                         "lattice": {"a": 3.32},
                         "wyckoff_sites": [{"element": "Ti",
                                            "coords": [0.0, 0.0, 0.0]}]}},
        ],
        "analysis": {"section": {"axis": "z", "position": 0.5}},
        "output": {"directory": str(outdir)},
    })
    return run(cfg)


def test_multiphase_statistics_sections(ti_run):
    res = ti_run
    assert res.gates.all_passed()
    s = res.statistics
    # grain-size + sphericity repeat per phase
    for name in ("alpha", "beta"):
        assert f"grain_size:{name}" in s
        assert f"sphericity:{name}" in s
        assert f"mdf:{name}" in s
    assert "mdf" not in s                       # no global point group
    n_a = s["grain_size:alpha"]["n_grains"]
    n_b = s["grain_size:beta"]["n_grains"]
    assert n_a + n_b == 8 == s["grain_size"]["n_grains"]
    # S_V splits exactly into same-phase + interphase
    b = s["boundaries"]
    assert b["S_V_same_phase_per_A"] + b["S_V_interphase_per_A"] == \
        pytest.approx(b["S_V_per_A"], rel=1e-12)
    assert b["gb_interphase_area_A2"] > 0.0
    assert "area_fraction_interphase" in s["gb_character"]
    assert s["gb_character"]["area_fraction_interphase"] > 0.0
    # interphase pairs are EXCLUDED from the per-phase MDF scalars
    n_pairs_same = sum(
        1 for r in res.boundary_reports if r.phase_i == r.phase_j)
    assert (s["mdf:alpha"]["n_boundaries"]
            + s["mdf:beta"]["n_boundaries"]) == n_pairs_same


def test_multiphase_json_and_section(ti_run):
    res = ti_run
    doc = json.loads((res.outdir / "microstructure.json")
                     .read_text(encoding="utf-8"))
    from grainsmith.io import (
        BOUNDARIES_COLUMNS_PHASES,
        GRAINS_COLUMNS_PHASES,
    )
    assert list(doc["grains"][0]) == GRAINS_COLUMNS_PHASES
    assert list(doc["boundaries"][0]) == BOUNDARIES_COLUMNS_PHASES
    assert {g["phase"] for g in doc["grains"]} == {"alpha", "beta"}
    # interphase misorientation is null (strict JSON), habit planes kept
    inter = [b for b in doc["boundaries"]
             if b["character"] == "interphase"]
    assert inter and inter[0]["misorientation_deg"] is None
    assert inter[0]["plane_i_hkl"] != ""
    # the EBSD-like slice carries the phase column
    rows = _csv_rows(res.outdir / "slice_z0.5.csv")
    assert list(rows[0]) == SECTION_COLUMNS_PHASES
    assert {r["phase"] for r in rows} <= {"alpha", "beta"}


def test_multiphase_methods(ti_run):
    text = (ti_run.outdir / "METHODS.md").read_text(encoding="utf-8")
    assert "2 phases" in text
    assert "alpha" in text and "beta" in text
    assert "gate G15" in text
    assert "partitioned among the phases" in text
    assert "per phase" in text and "habit-plane" in text


# ---------------------------------------------------------------------------
# Determinism: the new files are byte-identical on a re-run
# ---------------------------------------------------------------------------


def test_r5_outputs_byte_identical(tmp_path, monkeypatch):
    import re

    def _run(root):
        # The SAME config (incl. the relative output.directory — hashed
        # into the provenance and echoed in microstructure.json) from two
        # working directories, as in test_end_to_end.
        root.mkdir()
        monkeypatch.chdir(root)
        return run(resolve_config({
            "meta": {"verbose": 0},
            "seed": {"mode": "fixed", "value": SEED},
            "box": {"lengths": [30.0, 30.0, 30.0]},
            "grains": {"number": 3},
            "crystal": {"space_group": {"number": 225},
                        "lattice": {"a": 3.615},
                        "wyckoff_sites": [{"element": "Cu",
                                           "coords": [0.0, 0.0, 0.0]}]},
            "analysis": {"section": {"axis": "y", "position": 0.25}},
            "output": {"directory": "./out"},
        }))

    r1 = _run(tmp_path / "run1")
    r2 = _run(tmp_path / "run2")
    ts = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    for name in ("statistics.csv", "microstructure.json", "METHODS.md",
                 "slice_y0.25.csv"):
        b1 = ts.sub(b"<TS>", (r1.outdir / name).read_bytes())
        b2 = ts.sub(b"<TS>", (r2.outdir / name).read_bytes())
        assert b1 == b2, f"{name} differs between identical runs"
