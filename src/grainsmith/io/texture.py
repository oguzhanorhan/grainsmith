"""Texture outputs: mdf.csv and odf_mtex.txt.

mdf.csv — misorientation-angle histogram of the generated boundary
network: per bin the area-weighted density, the number density, the
random-pair reference of the actual point group (the Mackenzie law for
cubic), and — when MDF targeting is active — the target density.
Densities are per degree, so each column integrates to 1 over the bins.
Like all CSVs (§8.3), no provenance comment line; provenance lives in
summary.csv.

odf_mtex.txt — per-grain Bunge Euler angles + volume-fraction weights,
directly loadable in MTEX via ``loadOrientation_generic`` (the exact
one-liner is embedded as a ``%`` comment header) — the experimental
comparison path of docs/physics.md §4. When atom counts are available
(non-``None`` and not all zero), a fifth column ``atom_fraction`` is added:
the fraction of ATOMS per grain in the exported atomistic structure, as
opposed to ``weight``'s tessellation-volume fraction. The two differ
because overlap deletion strips a boundary shell from every grain (gate
G26, docs/physics.md §4); with no atom counts the file is the original
four columns, byte for byte.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from grainsmith.io.common import Provenance, fmt
from grainsmith.io.reports import _cell, _open_csv
from grainsmith.orientation.quaternion import quat_to_bunge

MDF_COLUMNS: list[str] = [
    "bin_center_deg", "area_weighted_density", "number_density",
    "haar_random_reference", "target_density",
]
"""mdf.csv columns — docs/physics.md §4 (densities per degree; target_density
is empty when no orientation.mdf_target is configured).

``haar_random_reference`` is the deterministic random-pair
disorientation-angle reference of the ACTUAL crystal point group
(``orientation.mdf.reference_angles``): it equals the Mackenzie 1958 law
only for the cubic proper point group, so the column is named for what the
curve IS for every point group rather than for the cubic special case
(renamed from the legacy ``mackenzie_reference``)."""


def write_mdf_csv(
    path: Path,
    bin_edges: np.ndarray,
    area_masses: np.ndarray,
    number_masses: np.ndarray,
    reference_masses: np.ndarray,
    target_masses: np.ndarray | None,
) -> None:
    """mdf.csv: one row per misorientation-angle bin (docs/physics.md §4).

    All ``*_masses`` arguments are bin-mass vectors summing to 1 (or 0 for
    a structure without boundaries); they are converted to per-degree
    densities with the bin widths.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def dens(masses: np.ndarray) -> np.ndarray:
        return np.asarray(masses, dtype=np.float64) / widths

    area_d = dens(area_masses)
    num_d = dens(number_masses)
    ref_d = dens(reference_masses)
    tgt_d = dens(target_masses) if target_masses is not None else None
    with _open_csv(path) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(MDF_COLUMNS)
        for b in range(len(centers)):
            w.writerow([
                _cell(float(centers[b])),
                _cell(float(area_d[b])),
                _cell(float(num_d[b])),
                _cell(float(ref_d[b])),
                _cell(float(tgt_d[b])) if tgt_d is not None else "",
            ])


def write_odf_mtex(
    path: Path,
    quats: np.ndarray,
    volume_weights: np.ndarray,
    provenance: Provenance,
    atom_counts: np.ndarray | None = None,
) -> None:
    """odf_mtex.txt: per-grain ``phi1 Phi phi2 weight [atom_fraction]`` rows
    (degrees, Bunge) with an MTEX import one-liner in the ``%`` comment
    header (docs/physics.md §4).

    ``weight`` is the TESSELLATION-volume fraction -- the quantity
    grainsmith controls (gate G22, ``orientation.mdf_target.odf_drift_max``)
    and the standard definition of the volume-weighted ODF.

    ``atom_counts``, when given, is the per-grain atom count of the
    exported atomistic structure, aligned with ``quats``/``volume_weights``
    (``GrainReport.n_atoms``; 0 per grain when the run has no atom block).
    When its sum is nonzero a fifth column, ``atom_fraction = n_atoms /
    sum(n_atoms)``, is written -- the ODF a downstream MD simulation
    actually experiences, since the shipped structure is atomistic, not
    the tessellation. ``weight`` and ``atom_fraction`` differ because
    overlap deletion removes a boundary shell from every grain, so the
    relative loss scales as surface/volume and large grains are
    systematically over-represented in ``atom_fraction`` (gate G26 reports
    the total-variation distance between the two).

    When ``atom_counts`` is ``None``, or its sum is zero, the file is
    exactly the four-column ``phi1 Phi phi2 weight`` form -- byte for byte
    what this function wrote before ``atom_fraction`` existed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(volume_weights, dtype=np.float64)
    weights = weights / float(np.sum(weights))

    atom_fractions: np.ndarray | None = None
    if atom_counts is not None:
        counts = np.asarray(atom_counts, dtype=np.float64)
        total = float(np.sum(counts))
        if total != 0.0:
            atom_fractions = counts / total

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"% {provenance.line()}\n")
        fh.write("% per-grain Bunge Euler angles (degrees) + "
                 "volume-fraction weights\n")
        if atom_fractions is None:
            fh.write("% MTEX import:\n")
            fh.write("%   ori = loadOrientation_generic('odf_mtex.txt', "
                     "'CS', cs, 'ColumnNames', "
                     "{'phi1' 'Phi' 'phi2' 'weight'}, 'Bunge', 'degree');\n")
            fh.write("% phi1 Phi phi2 weight\n")
            for q, wgt in zip(quats, weights, strict=True):
                phi1, Phi, phi2 = quat_to_bunge(q)
                fh.write(f"{fmt(phi1)} {fmt(Phi)} {fmt(phi2)} {fmt(wgt)}\n")
        else:
            fh.write("% weight = tessellation-volume fraction, the "
                     "quantity grainsmith CONTROLS (gate G22, "
                     "orientation.mdf_target.odf_drift_max).\n")
            fh.write("% atom_fraction = fraction of ATOMS in the exported "
                     "structure, the ODF a downstream MD simulation "
                     "actually experiences.\n")
            fh.write("% They differ because overlap deletion strips a "
                     "boundary shell from every grain (loss ~ "
                     "surface/volume): large grains are over-represented "
                     "in atom_fraction. Gate G26 reports the "
                     "total-variation distance between the two.\n")
            fh.write("% MTEX import:\n")
            fh.write("%   ori = loadOrientation_generic('odf_mtex.txt', "
                     "'CS', cs, 'ColumnNames', "
                     "{'phi1' 'Phi' 'phi2' 'weight' 'atom_fraction'}, "
                     "'Bunge', 'degree');\n")
            fh.write("% phi1 Phi phi2 weight atom_fraction\n")
            for q, wgt, af in zip(quats, weights, atom_fractions,
                                   strict=True):
                phi1, Phi, phi2 = quat_to_bunge(q)
                fh.write(f"{fmt(phi1)} {fmt(Phi)} {fmt(phi2)} {fmt(wgt)} "
                         f"{fmt(af)}\n")
