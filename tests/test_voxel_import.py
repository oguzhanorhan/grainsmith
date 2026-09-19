"""Tests for the imported voxel-label tessellation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError, TessellationError
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.voxel import build_voxel_grid
from grainsmith.tessellation.voxel_import import (
    VoxelTessellation,
    imported_orientations,
    load_label_field,
)

L20 = np.array([20.0, 20.0, 20.0])
PER = [True, True, True]


def _bicrystal_labels() -> np.ndarray:
    """x < 10 → grain 0, x ≥ 10 → grain 1 on a 20³ grid (h = 1 Å)."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[10:] = 1
    return labels


def _wrap_labels() -> np.ndarray:
    """Grain 0 spans the x = 0 face (x < 5 or x ≥ 15); grain 1 between."""
    labels = np.ones((20, 20, 20), dtype=np.int32)
    labels[:5] = 0
    labels[15:] = 0
    return labels


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_validation_errors():
    with pytest.raises(ConfigError, match="3D"):
        VoxelTessellation(np.zeros((4, 4), dtype=np.int32), L20, PER)
    with pytest.raises(ConfigError, match="integer"):
        VoxelTessellation(np.zeros((4, 4, 4)), L20, PER)
    with pytest.raises(ConfigError, match="non-negative"):
        VoxelTessellation(np.full((4, 4, 4), -1, dtype=np.int32), L20, PER)
    gappy = np.zeros((4, 4, 4), dtype=np.int32)
    gappy[2:] = 2                          # id 1 missing
    with pytest.raises(ConfigError, match="gaps"):
        VoxelTessellation(gappy, L20, PER)


def test_bicrystal_exact_geometry():
    """Axis-aligned bicrystal: exact volumes, areas, margins, adjacency."""
    t = VoxelTessellation(_bicrystal_labels(), L20, PER)
    assert t.n_grains == 2
    np.testing.assert_allclose(t.voxel_grid.volumes(), [4000.0, 4000.0])
    assert t.adjacency() == [(0, 1)]
    assert abs(t.voxel_grid.gb_areas(PER)[(0, 1)] - 800.0) < 1e-9
    # centroids: x exact; y/z fall back to the plain mean (axis-spanning
    # grain ⇒ degenerate circular mean) = L/2
    np.testing.assert_allclose(t.seeds[:, 0], [5.0, 15.0], atol=1e-9)
    np.testing.assert_allclose(t.seeds[:, 1:], 10.0, atol=1e-9)

    X = np.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]])
    np.testing.assert_array_equal(t.grain_of(X), [0, 1])
    # signed EDT margins (center-to-center): ±5 at the grain centers
    np.testing.assert_allclose(t.margin(X, 0), [5.0, -5.0], atol=1e-9)
    np.testing.assert_allclose(t.margin(X, 1), [-5.0, 5.0], atol=1e-9)


def test_margin_uses_periodic_wrap():
    """A narrow grain at the box face: the nearest boundary of an inside
    point is ACROSS the wrap — unpadded EDT would overestimate."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[18:] = 1                        # grain 1: x ∈ [18, 20)
    t = VoxelTessellation(labels, L20, PER)
    # voxel center x = 19.5: in-array nearest outside center is 17.5
    # (distance 2), across the wrap it is 0.5 (distance 1) — exact: 1.0
    m = t.margin(np.array([[19.5, 10.0, 10.0]]), 1)
    assert m[0] == pytest.approx(1.0, abs=1e-9)


def test_tiling_invariant_wrap_spanning():
    """Σ over (27 lifts × all grains) of owns == 1 for every torus point —
    fill's tiling invariant, including a wrap-spanning grain."""
    t = VoxelTessellation(_wrap_labels(), L20, PER)
    # the wrap-spanning grain's centroid sits at the box face (x ≈ 0 ≡ 20)
    assert min(t.seeds[0, 0] % 20.0, 20.0 - t.seeds[0, 0] % 20.0) < 0.5
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(5)))
    P = rng.random((400, 3)) * L20
    total = np.zeros(len(P))
    for shift in np.ndindex(3, 3, 3):
        lift = P + (np.array(shift, dtype=np.float64) - 1.0) * L20
        for g in range(t.n_grains):
            total += t.owns(lift, g)
    np.testing.assert_array_equal(total, 1.0)


# ---------------------------------------------------------------------------
# Fill counts (exact pins, simple-cubic a = 2 Å, identity orientation)
# ---------------------------------------------------------------------------


def _fill_counts(tess) -> list[int]:
    from grainsmith.atoms.fill import fill_grain

    A = 2.0 * np.eye(3)
    frac = np.array([[0.0, 0.0, 0.0]])
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    rng = np.random.Generator(np.random.PCG64(1))
    return [
        len(fill_grain(g, tess, frac, ["Fe"], [None], A, q_id, L20, PER,
                       rng))
        for g in range(tess.n_grains)
    ]


def test_fill_counts_exact_bicrystal():
    """SC a=2 on the 20³ box: 10×10×10 = 1000 sites split 500/500."""
    t = VoxelTessellation(_bicrystal_labels(), L20, PER)
    counts = _fill_counts(t)
    assert counts == [500, 500]


def test_fill_counts_exact_wrap_spanning():
    """The wrap-spanning grain fills exactly its 500 sites — atoms are
    generated in the grain's compact (unwrapped) frame around x ≈ 0."""
    t = VoxelTessellation(_wrap_labels(), L20, PER)
    counts = _fill_counts(t)
    assert counts == [500, 500]


# ---------------------------------------------------------------------------
# Round-trip: voxelized FlatTessellation → import → identical voxel ops
# ---------------------------------------------------------------------------


def test_roundtrip_voxelized_voronoi():
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(11)))
    seeds = rng.random((8, 3)) * L20
    flat = FlatTessellation(seeds, L20, PER)
    vg = build_voxel_grid(flat, L20, grid_size=24)

    t = VoxelTessellation(vg.labels, L20, PER)
    np.testing.assert_allclose(t.voxel_grid.volumes(), vg.volumes())
    assert t.adjacency() == vg.adjacency(PER)
    # membership agrees exactly at voxel centers
    from grainsmith.tessellation.voxel import _voxel_centers
    centers = _voxel_centers(L20, vg.shape)
    np.testing.assert_array_equal(t.grain_of(centers),
                                  vg.labels.ravel())


# ---------------------------------------------------------------------------
# Loaders: .npy, relabel, DREAM.3D HDF5, missing h5py
# ---------------------------------------------------------------------------


def test_npy_relabel_map(tmp_path):
    labels = np.zeros((6, 6, 6), dtype=np.int64)
    labels[2:] = 7
    labels[4:] = 12                         # ids {0, 7, 12}
    path = tmp_path / "labels.npy"
    np.save(path, labels)

    out, relabel_map, eulers = load_label_field(path, relabel=True)
    assert eulers is None
    assert relabel_map == {0: 0, 7: 1, 12: 2}
    assert out.dtype == np.int32
    np.testing.assert_array_equal(np.unique(out), [0, 1, 2])

    with pytest.raises(ConfigError, match="relabel"):
        load_label_field(path, relabel=False)


def test_npy_rejects_euler_dataset(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.zeros((4, 4, 4), dtype=np.int32))
    with pytest.raises(ConfigError, match="euler_dataset"):
        load_label_field(path, euler_dataset="whatever")


def test_dream3d_roundtrip(tmp_path):
    """HDF5 cell data (Nz, Ny, Nx, 1) transposes to internal (Nx, Ny, Nz);
    per-feature Euler rows are re-indexed by the relabel map."""
    h5py = pytest.importorskip("h5py")
    # asymmetric field: label = 1 + (x >= 2), shapes Nx=4, Ny=3, Nz=2
    nx, ny, nz = 4, 3, 2
    internal = np.ones((nx, ny, nz), dtype=np.int32)
    internal[2:] = 5                        # ids {1, 5} → relabel {0, 1}
    d3d = internal.transpose(2, 1, 0)[..., None]   # (Nz, Ny, Nx, 1)
    eulers = np.zeros((6, 3))               # rows indexed by ORIGINAL ids
    eulers[1] = [0.1, 0.2, 0.3]
    eulers[5] = [1.0, 0.5, 0.25]
    path = tmp_path / "synthetic.dream3d"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("DataContainers/SV/CellData/FeatureIds", data=d3d)
        fh.create_dataset("DataContainers/SV/Grain Data/AvgEulerAngles",
                          data=eulers)

    labels, relabel_map, eul = load_label_field(
        path, dataset="DataContainers/SV/CellData/FeatureIds",
        relabel=True,
        euler_dataset="DataContainers/SV/Grain Data/AvgEulerAngles")
    np.testing.assert_array_equal(labels, internal - (internal == 1) -
                                  (internal == 5) * 4)
    assert relabel_map == {1: 0, 5: 1}
    np.testing.assert_allclose(eul, [[0.1, 0.2, 0.3], [1.0, 0.5, 0.25]])

    quats = imported_orientations(eul, degrees=False)
    assert quats.shape == (2, 4)
    # convention pin: same active rotation as orientation.fixed euler
    from grainsmith.orientation.samplers import fixed_orientation
    want = fixed_orientation(
        1, {"euler_bunge_deg": list(np.degrees(eul[1]))})[0]
    np.testing.assert_allclose(quats[1], want, atol=1e-12)


def test_missing_h5py_clean_error(tmp_path, monkeypatch):
    import sys
    path = tmp_path / "field.dream3d"
    path.write_bytes(b"not really hdf5")
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ConfigError, match=r'pip install "\.\[import\]"'):
        load_label_field(path, dataset="x")


# ---------------------------------------------------------------------------
# Config rules (resolve Rules 16–19)
# ---------------------------------------------------------------------------


def _raw(tmp_path, **over):
    labels = _bicrystal_labels()
    path = tmp_path / "labels.npy"
    np.save(path, labels)
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [20.0, 20.0, 20.0]},
        "grains": {},
        "crystal": {
            "space_group": {"number": 229},
            "lattice": {"a": 2.866},
            "wyckoff_sites": [{"element": "Fe", "coords": [0, 0, 0]}],
        },
        "boundaries": {"geometry": "voxel_import",
                       "voxel_import": {"file": str(path)}},
    }
    raw.update(over)
    return raw


def test_rule16_number_required_without_import(tmp_path):
    raw = _raw(tmp_path, boundaries={"geometry": "flat"})
    with pytest.raises(ConfigError, match="grains.number is required"):
        resolve_config(raw)


def test_rule17_block_and_file(tmp_path):
    raw = _raw(tmp_path)
    raw["boundaries"] = {"geometry": "voxel_import"}
    with pytest.raises(ConfigError, match="voxel_import"):
        resolve_config(raw)
    raw = _raw(tmp_path)
    raw["boundaries"]["voxel_import"]["file"] = str(tmp_path / "nope.npy")
    with pytest.raises(ConfigError, match="not found"):
        resolve_config(raw)


def test_rule19_imported_orientation_needs_eulers(tmp_path):
    raw = _raw(tmp_path, orientation={"scheme": "imported"})
    with pytest.raises(ConfigError, match="euler_dataset"):
        resolve_config(raw)


def test_size_distribution_forbidden(tmp_path):
    raw = _raw(tmp_path)
    raw["grains"] = {"size_distribution": {"type": "equal"}}
    with pytest.raises(ConfigError, match="size_distribution"):
        resolve_config(raw)


# ---------------------------------------------------------------------------
# boundaries.voxel_import.file resolution against the YAML file's own
# directory (load_config's base_dir fallback, mirrors crystal.cif.file
# in config/resolve.py) -- lets an example config's asset paths resolve
# regardless of the caller's CWD.
# ---------------------------------------------------------------------------


def _voxel_raw(file_value: str) -> dict:
    """A minimal voxel_import raw config with a caller-chosen file value
    (deliberately NOT resolved to an absolute path here, unlike the
    module's own `_raw` helper -- these tests need to control exactly
    what string appears in the YAML)."""
    return {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [20.0, 20.0, 20.0], "periodic": [True, True, True]},
        "grains": {},
        "crystal": {
            "space_group": {"number": 229},
            "lattice": {"a": 2.866},
            "wyckoff_sites": [{"element": "Fe", "coords": [0, 0, 0]}],
        },
        "boundaries": {"geometry": "voxel_import",
                       "voxel_import": {"file": file_value}},
    }


def test_voxel_import_cwd_relative_wins_and_is_left_unrewritten(
    tmp_path, monkeypatch):
    """The AS-GIVEN interpretation (here: relative to the CWD) takes
    precedence over the base_dir fallback whenever both candidates
    exist -- full backward compatibility. Proven with two DIFFERENT
    label fields under the same filename so a wrong-precedence bug
    would load the wrong grain count, not just the wrong path string."""
    import yaml

    from grainsmith.config.resolve import load_config

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    cwd_dir = tmp_path / "run_here"
    cwd_dir.mkdir()

    np.save(cwd_dir / "field.npy", _bicrystal_labels())          # 2 grains
    three_grain = np.zeros((20, 20, 20), dtype=np.int32)
    three_grain[7:14] = 1
    three_grain[14:] = 2
    np.save(cfg_dir / "field.npy", three_grain)                  # 3 grains

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump(_voxel_raw("field.npy")))

    monkeypatch.chdir(cwd_dir)
    cfg = load_config(cfg_path)

    # as-given (CWD-relative) form left byte-identical -- not rewritten.
    assert cfg.boundaries.voxel_import.file == "field.npy"

    loaded, _, _ = load_label_field(cfg.boundaries.voxel_import.file)
    assert len(np.unique(loaded)) == 2   # the CWD field, not the cfg-dir one


def test_voxel_import_not_found_lists_both_tried_locations(
    tmp_path, monkeypatch):
    import yaml

    from grainsmith.config.resolve import load_config

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump(_voxel_raw("missing_field.npy")))

    monkeypatch.chdir(elsewhere)
    with pytest.raises(ConfigError,
                       match="tried relative to the working directory"
                       ) as exc:
        load_config(cfg_path)
    msg = str(exc.value)
    assert "missing_field.npy" in msg
    assert str(cfg_dir) in msg


def test_voxel_import_base_dir_fallback_rewrites_absolute_and_loads(
    tmp_path, monkeypatch):
    """When the base_dir fallback fires, boundaries.voxel_import.file is
    REWRITTEN to the resolved absolute path (unlike crystal.cif.file,
    which is consumed immediately inside resolve_config, vi.file is read
    again later at tessellation-build time -- pipeline._stage_voxel_import
    -- by code with no base_dir of its own, so the rewrite is what keeps
    that later read working from any CWD)."""
    import yaml

    from grainsmith.config.resolve import load_config
    from grainsmith.pipeline import _stage_voxel_import

    # See the analogous CIF test's comment (test_cif.py) for why 'cfgs'
    # and 'assets' sit under a shared 'project' branch while 'elsewhere'
    # is a separate branch of tmp_path: it keeps the CWD-relative
    # '../assets/field.npy' from accidentally also resolving, which
    # would defeat the point of this test.
    cfg_dir = tmp_path / "project" / "cfgs"
    cfg_dir.mkdir(parents=True)
    assets_dir = tmp_path / "project" / "assets"
    assets_dir.mkdir()
    np.save(assets_dir / "field.npy", _bicrystal_labels())

    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(yaml.dump(_voxel_raw("../assets/field.npy")))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert not (elsewhere / ".." / "assets" / "field.npy").resolve().exists()
    monkeypatch.chdir(elsewhere)

    cfg = load_config(cfg_path)

    resolved = Path(cfg.boundaries.voxel_import.file)
    assert resolved.is_absolute()
    assert resolved == (assets_dir / "field.npy").resolve()

    # the tessellation-build stage reads vi.file itself, from a CWD
    # (still 'elsewhere' here) that has nothing to do with either the
    # config file or the asset -- only the rewrite makes this work.
    tess, _, _ = _stage_voxel_import(cfg)
    assert tess.n_grains == 2


# ---------------------------------------------------------------------------
# End-to-end (pipeline wiring, derived n, G5 demotion, determinism)
# ---------------------------------------------------------------------------


def _e2e_config(tmp_path, outdir, labels: np.ndarray, **over):
    path = tmp_path / "field.npy"
    np.save(path, labels)
    raw = _raw(tmp_path, output={"directory": str(outdir)})
    raw["boundaries"]["voxel_import"]["file"] = str(path)
    raw["meta"] = {"verbose": 0}
    raw.update(over)
    return resolve_config(raw)


def test_e2e_import_all_gates(tmp_path):
    from grainsmith.pipeline import run

    res = run(_e2e_config(tmp_path, tmp_path / "out", _bicrystal_labels()))
    assert res.gates.all_passed()
    gate_ids = {r.gate for r in res.gates.results()}
    # G23 (report-only) fires for every single-phase run that reaches the
    # MDF-histogram block, independent of orientation.mdf_target. G26
    # (report-only) fires whenever per-grain atom counts exist.
    assert gate_ids == {f"G{k}" for k in range(1, 11)} | {"G23", "G26"}
    assert res.tess.n_grains == 2
    # derived count reaches summary.csv; backend recorded
    summary = (res.outdir / "summary.csv").read_text(encoding="utf-8")
    assert "grains,number,2" in summary
    assert "tessellation,backend,VoxelTessellation" in summary
    assert "voxel_import,n_grains_derived,2" in summary
    # G3 measured on the voxel volumes (exact: labels tile the box)
    g3 = [r for r in res.gates.results() if r.gate == "G3"][0]
    assert g3.measured == pytest.approx(0.0, abs=1e-12)


def test_e2e_number_mismatch(tmp_path):
    from grainsmith.pipeline import run

    cfg = _e2e_config(tmp_path, tmp_path / "out", _bicrystal_labels(),
                      grains={"number": 5})
    with pytest.raises(ConfigError, match="does not match"):
        run(cfg)


def test_e2e_g5_warn_demotion(tmp_path):
    """A deliberately fragmented imported grain: WARN by default,
    hard failure with strict_connectivity: true."""
    from grainsmith.pipeline import run

    # grain 1 = two macroscopic fragments [4,6) and [10,16) that do NOT
    # touch through the wrap (grain 0 fills [16,20) ∪ [0,4) and [6,10))
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[4:6] = 1
    labels[10:16] = 1

    res = run(_e2e_config(tmp_path, tmp_path / "out_warn", labels))
    assert res.gates.all_passed()
    g5 = [r for r in res.gates.results() if r.gate == "G5"][0]
    assert "WARN" in g5.message

    cfg = _e2e_config(tmp_path, tmp_path / "out_strict", labels)
    cfg.boundaries.voxel_import.strict_connectivity = True
    with pytest.raises(Exception, match="G5"):
        run(cfg)


def test_e2e_deterministic_rerun(tmp_path, monkeypatch):
    """Byte-identical re-run of an imported field (§10)."""
    import re

    from grainsmith.pipeline import run

    root1 = tmp_path / "r1"
    root2 = tmp_path / "r2"
    root1.mkdir()
    root2.mkdir()
    path = tmp_path / "field.npy"
    np.save(path, _wrap_labels())

    def cfg():
        raw = _raw(tmp_path, output={"directory": "./out"})
        raw["boundaries"]["voxel_import"]["file"] = str(path)
        raw["meta"] = {"verbose": 0}
        return resolve_config(raw)

    monkeypatch.chdir(root1)
    run(cfg())
    monkeypatch.chdir(root2)
    run(cfg())
    ts = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    for name in ("polycrystal.data", "grains.csv", "boundaries.csv",
                 "mdf.csv"):
        b1 = ts.sub(b"<TS>", (root1 / "out" / name).read_bytes())
        b2 = ts.sub(b"<TS>", (root2 / "out" / name).read_bytes())
        assert b1 == b2, f"{name} differs between identical runs"


def test_g5_connectivity_unit():
    """check_connectivity itself: a genuinely fragmented grain raises (the
    WARN demotion lives in the pipeline, not in the gate)."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[4:6] = 1
    labels[10:16] = 1     # two fragments NOT touching through the wrap
    t = VoxelTessellation(labels, L20, PER)
    with pytest.raises(TessellationError):
        t.voxel_grid.check_connectivity(PER)
    # the wrap-spanning grain alone is CONNECTED under periodic stitching
    t2 = VoxelTessellation(_wrap_labels(), L20, PER)
    assert t2.voxel_grid.check_connectivity(PER) == {0: True, 1: True}
