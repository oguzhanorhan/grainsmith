"""Imported voxel-label tessellation backend (method M7).

Wraps an externally produced voxel label field (Potts / phase-field grain
growth, DREAM.3D synthetic or 3D-EBSD reconstructions) as a full
:class:`~grainsmith.tessellation.base.Tessellation`, so the unchanged
fill / overlap / analysis stack can atomize it:

- ``grain_of``  — wrapped voxel-label lookup.
- ``owns``      — label match restricted to the per-axis fundamental
  window ``[c_i − L/2, c_i + L/2)`` around the grain's centroid c_i.  The
  labels tile the torus exactly once and every torus point has exactly
  one lift inside each grain's window, so exactly one (lift, grain) pair
  owns each torus point — fill's tiling invariant is preserved for
  arbitrary (wrap-spanning, even disconnected) imported grains.
- ``margin``    — signed Euclidean distance transform per grain (two
  ``distance_transform_edt`` passes, h_vec-scaled, wrap padding of half
  the axis extent on periodic axes ⇒ exact min-image distances).  Margins
  are voxel-center estimates: nearest-voxel lookup with bias ≲ h
  (documented; the gradient-based gb_normals consume DIFFERENCES, where
  the center-vs-face offset largely cancels).
- ``bounding_radius`` — max min-image voxel distance from the centroid
  plus one voxel diagonal.

Centroids use the circular mean on periodic axes (the standard wrap-safe
estimator); ANY center keeps the owns-window construction correct — the
choice only tightens ``bounding_radius``.

IO helpers load ``.npy`` integer fields (shape from the file, box from
the config) and DREAM.3D HDF5 datasets (optional extra ``[import]`` /
h5py; arrays are stored (Nz, Ny, Nx[, 1]) per the XDMF convention and are
transposed to the internal (Nx, Ny, Nz) layout), with optional per-grain
Bunge Euler angles for ``orientation.scheme: imported``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grainsmith.errors import ConfigError, TessellationError
from grainsmith.memory import build_memory_guard_message
from grainsmith.tessellation.base import Tessellation
from grainsmith.tessellation.voxel import VoxelGrid

log = logging.getLogger(__name__)

_MARGIN_CACHE_FIELDS: int = 4
"""Per-grain signed-distance fields kept in memory (FIFO).

gb_normals probes margins of the two grains of a pair in alternation, so a
cache of ≥ 2 makes each per-grain EDT run once per pair loop; 4 covers the
i/j alternation with slack.  Each field is one float64 grid (e.g. 134 MB at
256³) — the cap bounds steady-state memory."""


class VoxelTessellation(Tessellation):
    """Tessellation defined by an imported integer voxel label field.

    Parameters
    ----------
    labels : (Nx, Ny, Nz) integer array
        Grain id per voxel; ids must be exactly 0 … N−1 (use
        :func:`load_label_field` / ``relabel: true`` to compact arbitrary
        ids).
    box_lengths : (3,) float64 — box edge lengths in Å (from the config;
        the grid shape comes from the file).
    periodic : per-axis periodicity flags.
    """

    def __init__(
        self,
        labels: np.ndarray,
        box_lengths: np.ndarray,
        periodic: list[bool],
    ) -> None:
        labels = np.asarray(labels)
        if labels.ndim != 3:
            raise ConfigError(
                f"voxel_import labels must be a 3D array, got shape "
                f"{labels.shape}.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ConfigError(
                f"voxel_import labels must be integers, got dtype "
                f"{labels.dtype}.")
        lab_min = int(labels.min())
        if lab_min < 0:
            raise ConfigError(
                f"voxel_import labels must be non-negative, found {lab_min}.")
        n = int(labels.max()) + 1
        present = np.bincount(labels.ravel(), minlength=n) > 0
        if not np.all(present):
            missing = np.flatnonzero(~present)[:8].tolist()
            raise ConfigError(
                f"voxel_import labels must cover 0 … {n - 1} without gaps; "
                f"ids {missing} have zero voxels. Set voxel_import.relabel: "
                "true to compact the ids.")

        self._labels = np.ascontiguousarray(labels, dtype=np.int32)
        self._L = np.asarray(box_lengths, dtype=np.float64)
        self._periodic = list(periodic)
        self._n = n
        self._shape = self._labels.shape
        self._h_vec = np.array(
            [self._L[k] / self._shape[k] for k in range(3)], dtype=np.float64)
        # Reused by analysis (get_voxel_grid finds this attribute).
        self.voxel_grid = VoxelGrid(self._labels, self._L, self._shape, n,
                                    tess=self)
        self._seeds_arr = self._centroids()
        self._radii = self._bounding_radii()
        self._margin_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _axis_coords(self, axis: int) -> np.ndarray:
        """Per-voxel center coordinate along *axis*, flattened (n_vox,)."""
        n_ax = self._shape[axis]
        coord = (np.arange(n_ax, dtype=np.float64) + 0.5) * self._h_vec[axis]
        shape = [1, 1, 1]
        shape[axis] = n_ax
        return np.broadcast_to(coord.reshape(shape), self._shape).ravel()

    def _centroids(self) -> np.ndarray:
        """(N, 3) grain centroids: circular mean on periodic axes (wrap-safe
        for grains spanning the box face), arithmetic mean on free axes."""
        flat = self._labels.ravel()
        counts = np.bincount(flat, minlength=self._n).astype(np.float64)
        c = np.empty((self._n, 3), dtype=np.float64)
        for ax in range(3):
            x = self._axis_coords(ax)
            L_ax = self._L[ax]
            mean_plain = np.bincount(flat, weights=x,
                                     minlength=self._n) / counts
            if self._periodic[ax]:
                theta = (2.0 * np.pi / L_ax) * x
                s = np.bincount(flat, weights=np.sin(theta),
                                minlength=self._n)
                co = np.bincount(flat, weights=np.cos(theta),
                                 minlength=self._n)
                circ = (L_ax / (2.0 * np.pi)) * np.arctan2(s, co) % L_ax
                # Degenerate resultant ⇒ the grain (nearly) spans the axis
                # uniformly and the circular mean is arbitrary; the plain
                # mean of the wrapped coordinates (≈ L/2) is the saner
                # reported center.  ANY center keeps owns() correct.
                resultant = np.hypot(s, co) / counts
                c[:, ax] = np.where(resultant > 1e-9, circ, mean_plain)
            else:
                c[:, ax] = mean_plain
        return c

    def _bounding_radii(self) -> np.ndarray:
        """(N,) max min-image voxel-center distance from the centroid plus
        one voxel diagonal (covers voxel corners with slack)."""
        flat = self._labels.ravel()
        r2 = np.zeros(flat.shape, dtype=np.float64)
        for ax in range(3):
            d = self._axis_coords(ax) - self._seeds_arr[flat, ax]
            if self._periodic[ax]:
                L_ax = self._L[ax]
                d -= L_ax * np.floor(d / L_ax + 0.5)   # min-image
            r2 += d * d
        order = np.argsort(flat, kind="stable")
        starts = np.searchsorted(flat[order], np.arange(self._n))
        r_max = np.sqrt(np.maximum.reduceat(r2[order], starts))
        return r_max + float(np.linalg.norm(self._h_vec))

    # ------------------------------------------------------------------
    # Tessellation ABC
    # ------------------------------------------------------------------

    def _voxel_index(self, X: np.ndarray) -> tuple[np.ndarray, ...]:
        """Voxel index of each point: wrap on periodic axes, clamp on free
        axes (nearest-edge extension for out-of-box probes)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        idx = []
        for ax in range(3):
            x = X[:, ax]
            if self._periodic[ax]:
                x = np.mod(x, self._L[ax])
            i_ax = np.floor(x / self._h_vec[ax]).astype(np.intp)
            idx.append(np.clip(i_ax, 0, self._shape[ax] - 1))
        return tuple(idx)

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        return self._labels[self._voxel_index(X)].astype(np.int32)

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        keep = np.ones(len(X), dtype=bool)
        c = self._seeds_arr[i]
        for ax in range(3):
            d = X[:, ax] - c[ax]
            if self._periodic[ax]:
                half = 0.5 * self._L[ax]
                keep &= (d >= -half) & (d < half)   # fundamental window
            else:
                keep &= (X[:, ax] >= 0.0) & (X[:, ax] <= self._L[ax])
        if not np.any(keep):
            return keep
        keep[keep] = self._labels[self._voxel_index(X[keep])] == i
        return keep

    def _margin_field(self, i: int) -> np.ndarray:
        """Signed distance field of grain i at voxel centers (Å); cached
        FIFO up to _MARGIN_CACHE_FIELDS grains.

        Wrap padding is capped at ``bounding_radius(i)`` per periodic axis
        (cheaper than the exact-everywhere half-extent pad): every margin
        consumer probes within the grain's own extent — inside atoms,
        gb_normals' ±h stencils at the boundary — where the values are
        EXACT; for far-outside points beyond the pad the magnitude may be
        overestimated (the nearest wrapped source can fall outside the
        padded window), which no consumer reads (documented honesty)."""
        cached = self._margin_cache.get(i)
        if cached is not None:
            return cached
        from scipy.ndimage import distance_transform_edt

        pad = []
        for ax in range(3):
            if not self._periodic[ax]:
                pad.append((0, 0))
                continue
            w = min(self._shape[ax] // 2 + 1,
                    int(np.ceil(self._radii[i] / self._h_vec[ax])) + 2)
            pad.append((w, w))
        padded_shape = [self._shape[ax] + 2 * pad[ax][0] for ax in range(3)]
        # §13: two float64 EDT outputs of the padded grid dominate the peak.
        # Read the limit off THIS instance (self.memory_limit_bytes,
        # VoxelTessellation IS a Tessellation subclass) rather than the
        # constants.py module default -- see tessellation/base.py's
        # Tessellation.memory_limit_bytes docstring for why an instance
        # attribute (spawn-safety) rather than a module global is used.
        est = int(np.prod(padded_shape)) * 8 * 2
        limit_bytes = self.memory_limit_bytes
        if est > limit_bytes:
            raise TessellationError(build_memory_guard_message(
                f"margin EDT on the padded grid {tuple(padded_shape)}",
                est, limit_bytes, self.memory_limit_source,
                extra_advice=(
                    "Import a coarser label field, or disable "
                    "analysis.gb_character / analysis.per_atom_margin and "
                    "the delete_shallower overlap policy, which consume "
                    "margins."),
            ))
        mask = np.pad(self._labels == i, pad, mode="wrap")
        crop = tuple(slice(p[0], p[0] + self._shape[ax])
                     for ax, p in enumerate(pad))
        d_in = distance_transform_edt(mask, sampling=self._h_vec)[crop]
        d_out = distance_transform_edt(~mask, sampling=self._h_vec)[crop]
        field = d_in - d_out
        if len(self._margin_cache) >= _MARGIN_CACHE_FIELDS:
            self._margin_cache.pop(next(iter(self._margin_cache)))
        self._margin_cache[i] = field
        return field

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed distance to the boundary of grain i (positive inside).

        Voxel-center estimate (nearest-voxel lookup of the per-grain signed
        EDT field): the value is the center-to-center distance, so it
        carries a bias of up to ~h relative to the true face distance —
        adequate for the overlap policies and gb_margin reporting, and the
        bias cancels in the central differences of gb_normals."""
        return self._margin_field(i)[self._voxel_index(X)]

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> None:
        """No certified GB-shell bound: ``margin()`` here (its own
        docstring above) is a voxel-center EDT ESTIMATE with a documented
        bias of up to ~h (the voxel spacing) relative to the true face
        distance -- adequate for the overlap policies and reporting, but
        not an exact or provably-1-Lipschitz distance in the continuous
        atom coordinate, so it cannot certify a superset of the true pair
        endpoints. Always returns ``None`` (unchanged full-N overlap
        behaviour)."""
        return None

    def adjacency(self) -> list[tuple[int, int]]:
        return self.voxel_grid.adjacency(self._periodic)

    def bounding_radius(self, i: int) -> float:
        return float(self._radii[i])

    @property
    def seeds(self) -> np.ndarray:
        """(N, 3) grain centroids (the 'seeds' of an imported field)."""
        return self._seeds_arr

    @property
    def n_grains(self) -> int:
        return self._n

    def total_volume(self) -> float:
        return float(np.prod(self._L))

    # Margin fields are large transients — never ship them to fill workers.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_margin_cache"] = {}
        return state


# ---------------------------------------------------------------------------
# IO: .npy and DREAM.3D HDF5 label fields
# ---------------------------------------------------------------------------


def load_label_field(
    file: str | Path,
    dataset: str | None = None,
    relabel: bool = True,
    euler_dataset: str | None = None,
) -> tuple[np.ndarray, dict[int, int] | None, np.ndarray | None]:
    """Load an integer label field from ``.npy`` or DREAM.3D HDF5.

    Returns ``(labels (Nx,Ny,Nz) int32, relabel_map | None,
    eulers (N,3) | None)``: *relabel_map* maps original → compact id when
    ``relabel`` compacted anything (recorded in summary.csv); *eulers* are
    per-grain Bunge Euler angles re-indexed to the compact ids when
    *euler_dataset* is given (HDF5 only).
    """
    path = Path(file)
    if not path.exists():
        raise ConfigError(f"voxel_import.file not found: {path}")

    eulers: np.ndarray | None = None
    if path.suffix.lower() == ".npy":
        if euler_dataset is not None:
            raise ConfigError(
                "voxel_import.euler_dataset requires a DREAM.3D HDF5 file "
                "(.dream3d/.h5/.hdf5) — a .npy label field carries no "
                "orientation data.")
        labels = np.load(path)
        if labels.ndim != 3:
            raise ConfigError(
                f"{path}: expected a 3D label array, got shape "
                f"{labels.shape}.")
    elif path.suffix.lower() in (".dream3d", ".h5", ".hdf5"):
        labels, eulers = _load_hdf5(path, dataset, euler_dataset)
    else:
        raise ConfigError(
            f"voxel_import.file must be .npy or DREAM.3D HDF5 "
            f"(.dream3d/.h5/.hdf5), got {path.suffix!r}.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ConfigError(
            f"{path}: label field must be integer, got dtype "
            f"{labels.dtype}.")

    unique, inverse = np.unique(labels, return_inverse=True)
    relabel_map: dict[int, int] | None = None
    identity = (int(unique[0]) == 0 and len(unique) == int(unique[-1]) + 1)
    if relabel:
        if not identity:
            relabel_map = {int(orig): new for new, orig in enumerate(unique)}
            log.info("voxel_import: relabeled %d ids (%d … %d → 0 … %d)",
                     len(unique), int(unique[0]), int(unique[-1]),
                     len(unique) - 1)
        labels = inverse.reshape(labels.shape).astype(np.int32)
    elif not identity:
        raise ConfigError(
            f"voxel_import labels are not compact 0 … N−1 (found "
            f"{len(unique)} ids in [{int(unique[0])}, {int(unique[-1])}]). "
            "Set voxel_import.relabel: true.")
    else:
        labels = labels.astype(np.int32)

    if eulers is not None:
        if eulers.ndim != 2 or eulers.shape[1] != 3:
            raise ConfigError(
                f"euler_dataset must be an (N, 3) per-grain Bunge Euler "
                f"array, got shape {eulers.shape}.")
        if int(unique[-1]) >= len(eulers):
            raise ConfigError(
                f"euler_dataset has {len(eulers)} rows but the label field "
                f"contains id {int(unique[-1])} — per-grain rows must be "
                "indexed by the ORIGINAL feature ids.")
        eulers = np.asarray(eulers, dtype=np.float64)[unique]

    return labels, relabel_map, eulers


def _load_hdf5(
    path: Path,
    dataset: str | None,
    euler_dataset: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """DREAM.3D HDF5 reader: cell dataset (Nz, Ny, Nx[, 1]) per the XDMF
    convention → internal (Nx, Ny, Nz); optional per-feature Euler rows."""
    try:
        import h5py
    except ImportError as exc:
        raise ConfigError(
            'DREAM.3D import requires h5py: pip install ".[import]"'
        ) from exc
    if dataset is None:
        raise ConfigError(
            "voxel_import.dataset (HDF5 path to the FeatureIds cell array) "
            "is required for DREAM.3D files, e.g. "
            "'DataContainers/SyntheticVolumeDataContainer/CellData/"
            "FeatureIds'.")
    with h5py.File(path, "r") as fh:
        if dataset not in fh:
            raise ConfigError(
                f"{path}: dataset {dataset!r} not found in the HDF5 file.")
        arr = np.asarray(fh[dataset][...])
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ConfigError(
                f"{path}:{dataset}: expected (Nz, Ny, Nx[, 1]) cell data, "
                f"got shape {arr.shape}.")
        labels = np.ascontiguousarray(arr.transpose(2, 1, 0))
        eulers = None
        if euler_dataset is not None:
            if euler_dataset not in fh:
                raise ConfigError(
                    f"{path}: euler_dataset {euler_dataset!r} not found in "
                    "the HDF5 file.")
            eulers = np.asarray(fh[euler_dataset][...], dtype=np.float64)
    return labels, eulers


def imported_orientations(
    eulers: np.ndarray,
    degrees: bool = False,
) -> np.ndarray:
    """Per-grain quaternions from imported Bunge Euler angles.

    DREAM.3D stores Euler angles in RADIANS (standard Bunge ZXZ); the same
    active-rotation convention as orientation.fixed applies:
    R(q) = from_euler('ZXZ', …) so the Bunge matrix g = R(q)ᵀ matches the
    MTEX/EBSD convention (g maps sample → crystal, the transpose of R(q)'s
    crystal → sample rotation — a convention choice, not an inverse
    correction).
    """
    from scipy.spatial.transform import Rotation as _Rotation

    from grainsmith.orientation.quaternion import from_scipy

    rot = _Rotation.from_euler("ZXZ", np.asarray(eulers, dtype=np.float64),
                               degrees=degrees)
    return from_scipy(rot).reshape(-1, 4)
