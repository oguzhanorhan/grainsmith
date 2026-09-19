"""Tests for tessellation/voxel.py (§6.7): volumes, GB areas, local normals,
slab masking and the optional marching-cubes mesh.

The axis-aligned bicrystal is the exactness fixture: the voxel-face area
estimator and the central-difference normals are EXACT for boundaries that
coincide with voxel faces, so these tests pin equalities, not bands.
"""
from __future__ import annotations

import numpy as np
import pytest

from grainsmith.errors import TessellationError
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.voxel import VoxelGrid, build_voxel_grid, mesh_area


def _bicrystal_x():
    """Two grains split by planes x = 10 and x = 0 (≡ 20), fully periodic."""
    L = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    return build_voxel_grid(tess, L, grid_size=20), L


def test_volumes_exact_bicrystal():
    vg, L = _bicrystal_x()
    vols = vg.volumes()
    np.testing.assert_allclose(vols, [4000.0, 4000.0], atol=1e-9)
    assert abs(float(vols.sum()) - float(np.prod(L))) < 1e-9


def test_gb_areas_exact_axis_aligned():
    """Two planar boundaries (x = 10 and x = 0), each 20×20 Å² — the face
    estimator is exact for axis-aligned boundaries."""
    vg, _ = _bicrystal_x()
    areas = vg.gb_areas()
    assert set(areas) == {(0, 1)}
    assert abs(areas[(0, 1)] - 800.0) < 1e-9


def test_gb_normals_planar():
    """All local normals on the planar boundaries are ±x̂ exactly, oriented
    from grain 0 toward grain 1."""
    vg, _ = _bicrystal_x()
    normals = vg.gb_normals()
    pts, n, w = normals[(0, 1)]
    assert len(pts) == 800  # 2 boundaries × 20×20 faces
    np.testing.assert_allclose(w, 1.0, atol=1e-12)  # 1×1 Å voxel faces
    np.testing.assert_allclose(np.abs(n[:, 0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(n[:, 1:], 0.0, atol=1e-12)
    # Orientation 0→1: +x̂ on the x=10 sheet, -x̂ on the x=0 (≡20) sheet
    on_mid = np.isclose(pts[:, 0], 10.0)
    assert np.all(n[on_mid, 0] > 0)
    assert np.all(n[~on_mid, 0] < 0)


def test_gb_faces_cache_repeat_calls_match_and_are_reused():
    """_gb_faces is memoized per instance keyed on the periodic tuple:
    two consecutive calls (via the gb_normals / gb_face_samples public
    accessors, mirroring the real callers) must return equal arrays, the
    second call must be the SAME cached object (not recomputed), and the
    values must be unchanged vs a fresh, uncached instance."""
    vg, _ = _bicrystal_x()
    faces_a = vg.gb_face_samples()
    faces_b = vg.gb_face_samples()
    assert set(faces_a) == set(faces_b) == {(0, 1)}
    pts_a, areas_a = faces_a[(0, 1)]
    pts_b, areas_b = faces_b[(0, 1)]
    assert pts_a is pts_b and areas_a is areas_b  # same cached arrays
    np.testing.assert_array_equal(pts_a, pts_b)
    np.testing.assert_array_equal(areas_a, areas_b)

    # gb_normals shares the same cache key/instance and must see identical
    # underlying face data.
    normals = vg.gb_normals()
    pts_n, _, areas_n = normals[(0, 1)]
    assert len(pts_n) <= len(pts_a)  # normals filters near-zero gradients

    # Fresh, uncached instance must produce numerically identical output.
    vg_fresh, _ = _bicrystal_x()
    faces_fresh = vg_fresh.gb_face_samples()
    pts_f, areas_f = faces_fresh[(0, 1)]
    np.testing.assert_array_equal(pts_f, pts_a)
    np.testing.assert_array_equal(areas_f, areas_a)


def test_gb_areas_slab_masks_wrap():
    """Slab (free z): a z-stacked bicrystal has ONE boundary plane; the
    np.roll wrap pair at z = 0 ≡ L must not be counted."""
    L = np.array([20.0, 20.0, 30.0])
    seeds = np.array([[10.0, 10.0, 7.5], [10.0, 10.0, 22.5]])
    tess = FlatTessellation(seeds, L, [True, True, False])
    vg = build_voxel_grid(tess, L, grid_size=20)

    areas_slab = vg.gb_areas(periodic=[True, True, False])
    assert abs(areas_slab[(0, 1)] - 400.0) < 1e-9

    # Without the mask the wrap face would double the area — regression pin
    areas_wrong = vg.gb_areas(periodic=[True, True, True])
    assert abs(areas_wrong[(0, 1)] - 800.0) < 1e-9


def test_boundary_voxels_nonempty_and_in_box():
    vg, L = _bicrystal_x()
    pts = vg.boundary_voxels()
    assert len(pts) > 0
    assert np.all(pts >= 0.0) and np.all(pts <= L)


def test_g5_passes_connected_and_filters_slivers():
    """A connected grain with a 1-voxel discretization satellite passes G5
    (sliver filter), but the gate still verifies real connectivity."""
    L = np.array([20.0, 20.0, 20.0])
    shape = (20, 20, 20)
    labels = np.ones(shape, dtype=np.int32)
    labels[2:12, 2:12, 2:12] = 0          # 1000-voxel connected grain 0
    labels[17, 17, 17] = 0                # 1-voxel satellite (sliver)
    vg = VoxelGrid(labels, L, shape, 2)
    results = vg.check_connectivity([True, True, True])
    assert results == {0: True, 1: True}


def test_g5_trips_on_macroscopic_fragmentation():
    """Two macroscopic fragments of one grain fail G5 — the sliver filter
    must not mask genuine fragmentation."""
    L = np.array([20.0, 20.0, 20.0])
    shape = (20, 20, 20)
    labels = np.ones(shape, dtype=np.int32)
    labels[2:7, 2:7, 2:7] = 0             # fragment A: 125 voxels
    labels[12:17, 12:17, 12:17] = 0       # fragment B: 125 voxels, detached
    vg = VoxelGrid(labels, L, shape, 2)
    with pytest.raises(TessellationError):
        vg.check_connectivity([True, True, True])


def test_g5_trips_on_empty_grain():
    """A grain with zero voxels is the §3.2 empty-cell failure."""
    L = np.array([20.0, 20.0, 20.0])
    shape = (10, 10, 10)
    labels = np.ones(shape, dtype=np.int32)   # grain 0 owns nothing
    vg = VoxelGrid(labels, L, shape, 2)
    with pytest.raises(TessellationError):
        vg.check_connectivity([True, True, True])


def test_gb_mesh_planar_area():
    """Marching-cubes mesh of the interior boundary sheet (x = 10) has the
    exact planar area of the documented estimator; requires the [mesh]
    extra."""
    pytest.importorskip("skimage")
    vg, _ = _bicrystal_x()
    verts, faces = vg.gb_mesh((0, 1))
    area = mesh_area(verts, faces)
    # gb_mesh is computed on the non-periodic voxel-CENTER grid
    # (documented): the x = 0 sheet on the box face is not captured, and
    # the interior sheet spans the center-grid extent (L − h)² per axis —
    # 19 × 19 = 361 Å² exactly for the axis-aligned plane (visualization
    # mesh; gb_areas provides the PBC-correct 800 Å² estimate).
    h = 1.0
    assert area == pytest.approx((20.0 - h) ** 2, abs=1e-9)


def test_explicit_grid_size_clamp_is_warned(caplog):
    """An explicit grid_size whose per-axis count exceeds VOXEL_GRID_MAX on the
    long axis of an anisotropic box must WARN — a silent clamp would coarsen the
    analysis resolution below what the user asked for with no diagnostic."""
    import logging
    from grainsmith.constants import VOXEL_GRID_MAX
    L = np.array([10.0, 10.0, 800.0])            # long z axis
    seeds = np.array([[5.0, 5.0, 400.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    with caplog.at_level(logging.WARNING, logger="grainsmith.tessellation.voxel"):
        vg = build_voxel_grid(tess, L, grid_size=60)
    # z would need 4800 voxels at h = 10/60; it is clamped to VOXEL_GRID_MAX.
    assert vg.shape[2] == VOXEL_GRID_MAX
    assert any(rec.levelno == logging.WARNING for rec in caplog.records), (
        "no WARNING emitted for the silent per-axis voxel-grid clamp"
    )
