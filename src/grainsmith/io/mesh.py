"""PLY grain-boundary mesh writer (§8.5, curved geometry, optional).

Marching-cubes triangles per adjacent grain pair (VoxelGrid.gb_mesh, §6.7)
are merged into a single ASCII PLY with per-face ``grain_i``/``grain_j``
integer properties.  Vertex coordinates are double precision, Å, lab frame.
Requires the ``mesh`` optional extra (scikit-image); gb_mesh raises
GrainsmithError with installation guidance when it is missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from grainsmith.io.common import Provenance, atomic_writer, fmt
from grainsmith.tessellation.voxel import VoxelGrid


def write_ply(
    path: Path,
    voxel_grid: VoxelGrid,
    pairs: list[tuple[int, int]],
    provenance: Provenance,
) -> None:
    """Write the boundary mesh of every grain pair into one PLY file."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    face_pairs: list[tuple[int, int]] = []
    offset = 0
    for (i, j) in sorted(pairs):
        verts, faces = voxel_grid.gb_mesh((i, j))
        if len(faces) == 0:
            continue
        all_verts.append(verts)
        all_faces.append(np.asarray(faces, dtype=np.int64) + offset)
        face_pairs.extend([(i, j)] * len(faces))
        offset += len(verts)

    n_v = offset
    faces_arr = (np.concatenate(all_faces, axis=0) if all_faces
                 else np.empty((0, 3), dtype=np.int64))
    verts_arr = (np.concatenate(all_verts, axis=0) if all_verts
                 else np.empty((0, 3), dtype=np.float64))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"comment {provenance.line()}\n")
        fh.write("comment grainsmith grain-boundary mesh; coordinates in "
                 "Angstrom, lab frame\n")
        fh.write(f"element vertex {n_v}\n")
        fh.write("property double x\n")
        fh.write("property double y\n")
        fh.write("property double z\n")
        fh.write(f"element face {len(faces_arr)}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("property int grain_i\n")
        fh.write("property int grain_j\n")
        fh.write("end_header\n")
        for v in verts_arr:
            fh.write(f"{fmt(v[0])} {fmt(v[1])} {fmt(v[2])}\n")
        for f, (i, j) in zip(faces_arr, face_pairs, strict=True):
            fh.write(f"3 {f[0]} {f[1]} {f[2]} {i} {j}\n")
