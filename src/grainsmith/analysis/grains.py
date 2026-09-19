"""Per-grain analysis: volumes, atom counts, orientation report (§6.10).

Produces exactly the grains.csv column set of §8.3; the CSV serialization
itself lives in io/reports.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grainsmith.orientation.descriptors import grain_descriptors
from grainsmith.tessellation.base import Tessellation


@dataclass
class GrainReport:
    """One grains.csv row (§8.3)."""
    grain_id: int
    seed_x: float
    seed_y: float
    seed_z: float
    volume_A3: float
    volume_fraction: float
    n_atoms: int
    n_neighbors: int
    q_w: float
    q_x: float
    q_y: float
    q_z: float
    euler_phi1_deg: float
    euler_Phi_deg: float
    euler_phi2_deg: float
    axis_x: float
    axis_y: float
    axis_z: float
    angle_deg: float
    z_plane_hkl: str
    z_plane_dev_deg: float
    x_dir_uvw: str
    x_dir_dev_deg: float
    phase: str = ""              # phase name (multiphase runs only)


def get_voxel_grid(tess: Tessellation, box_lengths: np.ndarray):
    """Return the tessellation's voxel grid, reusing the one built for the
    G5 check when available (warp property / weighted lazy method), else
    building a fresh one."""
    vg = getattr(tess, "voxel_grid", None)
    if callable(vg):           # weighted backends: lazy method
        vg = vg()
    if vg is None:             # warp with connectivity_check=False, or other
        from grainsmith.tessellation.voxel import build_voxel_grid
        vg = build_voxel_grid(tess, box_lengths, "auto")
    return vg


def grain_volumes(tess: Tessellation, box_lengths: np.ndarray) -> np.ndarray:
    """Per-grain volumes (Å³): exact polyhedral volumes for the flat
    backend, the exact box volume for a single crystal, voxel
    volumes for curved backends (§6.7)."""
    from grainsmith.tessellation.flat import FlatTessellation
    from grainsmith.tessellation.single import SingleCrystalTessellation
    if isinstance(tess, FlatTessellation):
        return np.array([c.volume for c in tess.cells], dtype=np.float64)
    if isinstance(tess, SingleCrystalTessellation):
        return np.array([tess.total_volume()], dtype=np.float64)
    return get_voxel_grid(tess, box_lengths).volumes()


def analyze_grains(
    tess: Tessellation,
    quats: np.ndarray,
    A: np.ndarray,
    box_lengths: np.ndarray,
    atoms=None,
    volumes: np.ndarray | None = None,
    phase_of: np.ndarray | None = None,
    A_list: list[np.ndarray] | None = None,
    phase_names: list[str] | None = None,
) -> list[GrainReport]:
    """Build the per-grain report (§6.10 / grains.csv §8.3).

    Parameters
    ----------
    tess : Tessellation
    quats : (N, 4) float64 scalar-first grain orientations
    A : (3, 3) conventional cell matrix
    box_lengths : (3,) float64
    atoms : AtomBlock | None
        Final atom block; ``n_atoms`` is 0 per grain when omitted.
    volumes : optional override; default per :func:`grain_volumes`.
    phase_of, A_list, phase_names : multiphase runs — the
        orientation descriptors (hkl/uvw) use each grain's OWN cell
        matrix ``A_list[phase_of[i]]`` and the ``phase`` column carries
        the phase name; *A* is ignored when these are given.
    """
    n = tess.n_grains
    seeds = tess.seeds
    if volumes is None:
        volumes = grain_volumes(tess, np.asarray(box_lengths, dtype=np.float64))
    total_vol = float(np.sum(volumes))

    n_neighbors = np.zeros(n, dtype=int)
    for i, j in tess.adjacency():
        n_neighbors[i] += 1
        n_neighbors[j] += 1

    if atoms is not None and len(atoms) > 0:
        atom_counts = np.bincount(atoms.grain, minlength=n)
    else:
        atom_counts = np.zeros(n, dtype=int)

    reports: list[GrainReport] = []
    for i in range(n):
        if phase_of is not None:
            assert A_list is not None and phase_names is not None
            p = int(phase_of[i])
            desc = grain_descriptors(quats[i], A_list[p])
            desc["phase"] = phase_names[p]
        else:
            desc = grain_descriptors(quats[i], A)
        reports.append(GrainReport(
            grain_id=i,
            seed_x=float(seeds[i, 0]),
            seed_y=float(seeds[i, 1]),
            seed_z=float(seeds[i, 2]),
            volume_A3=float(volumes[i]),
            volume_fraction=float(volumes[i]) / total_vol,
            n_atoms=int(atom_counts[i]),
            n_neighbors=int(n_neighbors[i]),
            **desc,
        ))
    return reports
