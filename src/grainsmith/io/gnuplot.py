"""gnuplot visualization bundle (§8.4).

Files (all in the generation frame [0, L] — vacuum padding is a write-time
convention of the atomic outputs and does not move the tessellation):

- ``seeds.dat``      : ``x y z grain_id`` rows.
- ``edges.dat``      : flat geometry — per cell-face edge two ``x y z grain_id``
                       vertex rows + blank line; grains separated by a
                       ``# grain N`` comment and a DOUBLE blank line (gnuplot
                       ``index`` addressing).  The ``grain_id`` 4th column lets
                       ``view.plt`` colour each cell distinctly.  Each edge of a
                       cell is emitted once (shared by two faces of the cell).
- ``gb_points.dat``  : curved geometry — boundary voxel-face midpoints
                       ``x y z grain_i grain_j``.
- ``box.dat``        : box wireframe, 12 edges as 2-point segment blocks.
- ``view.plt``       : ready-to-run headless script (pngcairo; X11 line
                       included as a comment).
- ``curvature_hist.plt`` : histogram of local mean curvature H from
                       ``gb_curvature.csv`` (``analysis.gb_curvature`` only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from grainsmith.io.common import Provenance, atomic_writer, fmt
from grainsmith.tessellation.base import Tessellation
from grainsmith.tessellation.flat import FlatTessellation

_EDGE_KEY_DECIMALS = 9
"""Rounding for the per-cell edge dedupe key (Å).  Vertices of the two
faces sharing an edge come from the same Qhull vertex array, so they are
bitwise equal; rounding only guards against printf noise."""


def write_gnuplot_bundle(
    outdir: Path,
    tess: Tessellation,
    box_lengths: np.ndarray,
    periodic: list[bool],
    provenance: Provenance,
    curvature: bool = False,
    doping_elements: list[str] | None = None,
) -> list[Path]:
    """Write the §8.4 bundle into *outdir*.  Returns the written paths.

    *doping_elements* (dopant symbols, config order) enables
    doping_profile.plt — the dopant–GB distance histogram + proxigram
    companion of doping_profile.csv.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    L = np.asarray(box_lengths, dtype=np.float64)
    written: list[Path] = []
    flat = isinstance(tess, FlatTessellation)

    written.append(_write_seeds(outdir / "seeds.dat", tess, provenance))
    if isinstance(tess, FlatTessellation):
        written.append(_write_edges(outdir / "edges.dat", tess, provenance))
    else:
        written.append(_write_gb_points(outdir / "gb_points.dat", tess, L,
                                        periodic, provenance))
    from grainsmith.tessellation.single import SingleCrystalTessellation

    box_cell = (tess.cell_matrix
                if isinstance(tess, SingleCrystalTessellation)
                   and tess.is_triclinic
                else None)
    written.append(_write_box(outdir / "box.dat", L, provenance,
                              cell_matrix=box_cell))
    written.append(_write_view_plt(outdir / "view.plt", flat,
                                   tess.n_grains, provenance))
    if curvature:
        written.append(_write_curvature_hist_plt(
            outdir / "curvature_hist.plt", provenance))
    if doping_elements:
        written.append(_write_doping_profile_plt(
            outdir / "doping_profile.plt", doping_elements, provenance))
    return written


def _write_seeds(path: Path, tess: Tessellation,
                 provenance: Provenance) -> Path:
    seeds = tess.seeds
    with atomic_writer(path) as fh:
        fh.write(f"# {provenance.line()}\n")
        fh.write("# x y z grain_id\n")
        for i in range(tess.n_grains):
            fh.write(f"{fmt(seeds[i, 0])} {fmt(seeds[i, 1])} "
                     f"{fmt(seeds[i, 2])} {i}\n")
    return path


def _write_edges(path: Path, tess: FlatTessellation,
                 provenance: Provenance) -> Path:
    with atomic_writer(path) as fh:
        fh.write(f"# {provenance.line()}\n")
        fh.write("# x y z grain_id  (grain_id column colours each cell in view.plt)\n")
        for cell in tess.cells:
            gid = cell.grain_id
            fh.write(f"# grain {gid}\n")
            seen: set[tuple] = set()
            for face in cell.faces:
                verts = face.vertices
                m = len(verts)
                for t in range(m):
                    v1 = verts[t]
                    v2 = verts[(t + 1) % m]
                    key = tuple(sorted((
                        tuple(np.round(v1, _EDGE_KEY_DECIMALS)),
                        tuple(np.round(v2, _EDGE_KEY_DECIMALS)),
                    )))
                    if key in seen:
                        continue
                    seen.add(key)
                    fh.write(f"{fmt(v1[0])} {fmt(v1[1])} {fmt(v1[2])} {gid}\n")
                    fh.write(f"{fmt(v2[0])} {fmt(v2[1])} {fmt(v2[2])} {gid}\n\n")
            # double blank line → next gnuplot index (one per grain)
            fh.write("\n")
    return path


def _write_gb_points(path: Path, tess: Tessellation, L: np.ndarray,
                     periodic: list[bool], provenance: Provenance) -> Path:
    from grainsmith.analysis.grains import get_voxel_grid
    vg = get_voxel_grid(tess, L)
    faces = vg._gb_faces(periodic)
    with atomic_writer(path) as fh:
        fh.write(f"# {provenance.line()}\n")
        fh.write("# x y z grain_i grain_j\n")
        for (i, j) in sorted(faces):
            pts, _ = faces[(i, j)]
            for p in pts:
                fh.write(f"{fmt(p[0])} {fmt(p[1])} {fmt(p[2])} {i} {j}\n")
    return path


def _write_box(path: Path, L: np.ndarray, provenance: Provenance,
               cell_matrix: np.ndarray | None = None) -> Path:
    """12 box edges as 2-point segment blocks.

    *cell_matrix* (columns = box vectors a, b, c) draws the true
    parallelepiped wireframe for a triclinic (``box.cells``) single
    crystal; ``None`` (default) draws the orthogonal box ``diag(L)`` —
    identical output to the non-triclinic writer.  Corners are the 8
    combinations of fractional {0,1} coefficients on the box vectors
    (reduces exactly to the axis-aligned corners when the matrix is
    diagonal); edges connect corners differing in exactly one
    fractional coefficient.
    """
    H = cell_matrix if cell_matrix is not None else np.diag(L)
    H = np.asarray(H, dtype=np.float64)
    fracs = np.array([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0)
                      for k in (0.0, 1.0)])
    corners = fracs @ H.T
    # Edges connect corners whose fractional coefficients differ in
    # exactly one component (works for both diagonal and triclinic H).
    edges = [(a, b)
             for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(fracs[a] != fracs[b]) == 1]
    with atomic_writer(path) as fh:
        fh.write(f"# {provenance.line()}\n")
        for a, b in edges:
            for v in (corners[a], corners[b]):
                fh.write(f"{fmt(v[0])} {fmt(v[1])} {fmt(v[2])}\n")
            fh.write("\n")
    return path


def _write_view_plt(path: Path, flat: bool, n_grains: int,
                    provenance: Provenance) -> Path:
    """Ready-to-run headless view, **coloured per grain** so a reviewer can pick
    out each grain's shape rather than a single-colour tangle / point cloud.

    Flat geometry draws each Voronoi cell's wireframe in the grain's colour
    (``edges.dat`` column 4); curved geometry colours the boundary-voxel surface
    points by their owning grain.  The grain-id palette is a full-hue sweep with
    the colour bar pinned to ``[-0.5, n_grains-0.5]`` so the ids are used as
    discrete categories (without this ``cbrange`` the palette auto-scales and the
    ids all collapse into one end of the ramp)."""
    geometry = ("splot 'edges.dat' using 1:2:3:4 with lines lw 1.5 "
                "palette notitle, \\"
                if flat else
                "splot 'gb_points.dat' using 1:2:3:4 with points "
                "pt 7 ps 0.4 palette notitle, \\")
    cb_hi = n_grains - 0.5
    lines = [
        f"# {provenance.line()}",
        "set term pngcairo size 1200,900",
        "# uncomment for X11: set term x11",
        "set output 'view.png'",
        "set view equal xyz",
        "set xlabel 'x (A)'",
        "set ylabel 'y (A)'",
        "set zlabel 'z (A)'",
        "set ticslevel 0",
        "# per-grain colour (full-hue sweep over the grain-id range)",
        "set palette model HSV",
        "set palette defined (0 0 1 1, 1 0.85 1 1)",
        f"set cbrange [-0.5:{fmt(cb_hi)}]",
        "set cblabel 'grain id'",
        geometry,
        "      'box.dat' with lines lc rgb 'black' lw 2 notitle, \\",
        "      'seeds.dat' using 1:2:3 with points pt 7 ps 1.0 "
        "lc rgb 'black' notitle, \\",
        "      'seeds.dat' using 1:2:3:(sprintf('%d', $4)) with labels "
        "offset 1,1 font ',8' tc rgb 'black' notitle",
    ]
    with atomic_writer(path) as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _write_doping_profile_plt(path: Path, elements: list[str],
                              provenance: Provenance) -> Path:
    """Two-panel dopant–GB distance figure from doping_profile.csv.

    Top: dopant-count histogram vs distance to the nearest GB (how the
    placed dopants are distributed in space).  Bottom: the proxigram
    local_fraction(d) = n_dopant / n_candidate against the nominal
    fraction (dashed) and, when GB segregation is on, the shell edge
    (vertical line) and the ideal shell/bulk step implied by the
    enrichment.  A correctly segregated model shows the fraction
    stepping down at the shell edge to ≈ nominal-level bulk; a flat
    profile at nominal means no segregation was requested/achieved.
    """
    lines = [
        f"# {provenance.line()}",
        "set term pngcairo size 1000,800",
        "# uncomment for X11: set term x11",
        "set output 'doping_profile.png'",
        "set datafile separator ','",
        "# element = 'all' plots every dopant; set to one symbol",
        "# (e.g. element = 'Li') to filter",
        "element = 'all'",
        "set multiplot layout 2,1",
        "set xlabel 'distance to nearest GB (Angstrom)'",
        "set ylabel 'dopant atoms per bin'",
        "set style fill solid 0.6",
        "set key top right",
        "plot for [el in \"" + " ".join(elements) + "\"] \\",
        "     'doping_profile.csv' skip 1 using \\",
        "     (stringcolumn(1) eq el && (element eq 'all' || " +
        "element eq el) ? $5 : 1/0):6 \\",
        "     with boxes title el.' count'",
        "set ylabel 'local dopant fraction c(d)'",
        "# dashed: nominal fraction; vertical: shell edge (col 10)",
        "plot for [el in \"" + " ".join(elements) + "\"] \\",
        "     'doping_profile.csv' skip 1 using \\",
        "     (stringcolumn(1) eq el && (element eq 'all' || " +
        "element eq el) ? $5 : 1/0):8 \\",
        "     with linespoints title el.' c(d)', \\",
        "     for [el in \"" + " ".join(elements) + "\"] \\",
        "     'doping_profile.csv' skip 1 using \\",
        "     (stringcolumn(1) eq el && (element eq 'all' || " +
        "element eq el) ? $5 : 1/0):9 \\",
        "     with lines dashtype 2 title el.' nominal', \\",
        "     for [el in \"" + " ".join(elements) + "\"] \\",
        "     'doping_profile.csv' skip 1 using \\",
        "     (stringcolumn(1) eq el && (element eq 'all' || " +
        "element eq el) && $10 > 0 ? $10 : 1/0):8 \\",
        "     with impulses dashtype 3 title el.' shell edge'",
        "unset multiplot",
    ]
    with atomic_writer(path) as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _write_curvature_hist_plt(path: Path,
                              provenance: Provenance) -> Path:
    """Histogram of local mean curvature H from gb_curvature.csv.

    `grain` filters to one grain's boundaries with grain-relative sign
    (H is stored oriented i→j, so it is negated where the selected
    grain is grain_j); grain = -1 plots all samples as stored.
    """
    lines = [
        f"# {provenance.line()}",
        "set term pngcairo size 900,600",
        "# uncomment for X11: set term x11",
        "set output 'curvature_hist.png'",
        "set datafile separator ','",
        "# grain = -1: all samples; grain = <id>: that grain's",
        "# boundaries, sign flipped where the grain is grain_j",
        "grain = -1",
        "binw = 0.002",
        "bin(x) = binw * (floor(x / binw) + 0.5)",
        "set xlabel 'H (1/Angstrom)'",
        "set ylabel 'samples'",
        "set boxwidth binw",
        "set style fill solid 0.6",
        "plot 'gb_curvature.csv' skip 1 using \\",
        "     (grain < 0 ? bin($7) : \\",
        "      ($1 == grain ? bin($7) : \\",
        "       ($2 == grain ? bin(-$7) : 1/0))):(1.0) \\",
        "     smooth freq with boxes notitle",
    ]
    with atomic_writer(path) as fh:
        fh.write("\n".join(lines) + "\n")
    return path
