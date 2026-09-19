"""Extended XYZ writer (§8.2).

Comment line carries ``Lattice``, ``Properties``, per-axis ``pbc`` flags and
the §8 provenance string as a quoted key — OVITO reads the grain column for
per-grain coloring out of the box.

Conventions:
- Lattice is the box matrix: the orthogonal ``diag(L)`` box (free axes get
  ``L + vacuum`` with atom coordinates shifted by ``+vacuum/2``, same
  convention as the LAMMPS writer, §8.1) — or, for a ``box.cells``
  triclinic single crystal, the FULL 3x3 restricted-triclinic box matrix,
  row-major per the ASE ``Lattice="ax ay az bx by bz cx cy cz"`` convention
  (rows = box vectors a, b, c; note this is the TRANSPOSE of grainsmith's
  own column-vector cell-matrix convention used internally).
- Positions wrapped to the periodic cell at write time (§1): per-axis
  Cartesian wrap for the orthogonal box, fractional-coordinate wrap for a
  triclinic ``cell_matrix``.
- ``grain`` column is the 0-based grain id (matches grains.csv; the LAMMPS
  molecule id is the same id + 1).
- Optional ``gb_margin`` column (Å, signed distance to the nearest GB) when
  ``analysis.per_atom_margin`` is set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from grainsmith.atoms.fill import AtomBlock
from grainsmith.atoms.overlap import _wrap_positions, _wrap_positions_general
from grainsmith.errors import ConfigError
from grainsmith.io.common import (
    WRITE_CHUNK,
    Provenance,
    atomic_writer,
    drain_in_order,
    fmt,
)


def _format_extxyz_chunk(
    pos_chunk: np.ndarray,
    species_chunk: np.ndarray,
    grain_chunk: np.ndarray,
    margin_chunk: np.ndarray | None,
    periodic: list[bool],
    L: np.ndarray,
    vacuum: float,
    cell_matrix: np.ndarray | None = None,
) -> str:
    """Format the extended-XYZ rows for one contiguous atom chunk.

    Byte-identical to the per-row serial loop (same wrap + unconditional
    free-axis ``+0.5*vacuum`` shift + :func:`fmt`).  *cell_matrix*
    (``box.cells``), when given, wraps through the general triclinic
    fractional-coordinate path instead of the per-axis Cartesian wrap."""
    if cell_matrix is not None:
        H = np.asarray(cell_matrix, dtype=np.float64)
        w = _wrap_positions_general(pos_chunk, periodic, H, np.linalg.inv(H))
    else:
        w = _wrap_positions(pos_chunk, periodic, L)
    for ax in range(3):
        if not periodic[ax]:
            w[:, ax] += 0.5 * vacuum
    out: list[str] = []
    if margin_chunk is not None:
        for j in range(len(pos_chunk)):
            out.append(f"{species_chunk[j]} "
                       f"{fmt(w[j, 0])} {fmt(w[j, 1])} {fmt(w[j, 2])} "
                       f"{int(grain_chunk[j])} {fmt(margin_chunk[j])}")
    else:
        for j in range(len(pos_chunk)):
            out.append(f"{species_chunk[j]} "
                       f"{fmt(w[j, 0])} {fmt(w[j, 1])} {fmt(w[j, 2])} "
                       f"{int(grain_chunk[j])}")
    return "\n".join(out) + "\n" if out else ""


_EXTXYZ_CTX: dict | None = None
"""Per-worker-process extxyz context, set once by the pool initializer."""


def _init_extxyz_worker(ctx: dict) -> None:
    global _EXTXYZ_CTX
    _EXTXYZ_CTX = ctx


def _extxyz_chunk_worker(pos_chunk: np.ndarray, species_chunk: np.ndarray,
                         grain_chunk: np.ndarray,
                         margin_chunk: np.ndarray | None) -> str:
    c = _EXTXYZ_CTX
    assert c is not None
    return _format_extxyz_chunk(pos_chunk, species_chunk, grain_chunk,
                                margin_chunk, c["periodic"], c["L"],
                                c["vacuum"], cell_matrix=c.get("cell_matrix"))


def write_extxyz(
    path: Path,
    atoms: AtomBlock,
    box_lengths: np.ndarray,
    periodic: list[bool],
    provenance: Provenance,
    vacuum: float = 0.0,
    per_atom_margin: bool = False,
    jobs: int = 1,
    cell_matrix: np.ndarray | None = None,
) -> None:
    """Write an extended XYZ file (§8.2).

    Atom rows are streamed in row-chunks; *jobs* > 1 formats them
    across worker processes with byte-identical output for every *jobs*.

    Parameters
    ----------
    cell_matrix : (3,3), optional
        Restricted-triclinic box matrix (columns = box vectors) for a
        ``box.cells`` single-crystal build.  ``None`` (default)
        writes the orthogonal ``diag(box_lengths)`` Lattice exactly as
        before.
    """
    margins: np.ndarray | None = atoms.gb_margin if per_atom_margin else None
    if per_atom_margin and margins is None:
        raise ConfigError(
            "analysis.per_atom_margin is set but the atom block carries no "
            "gb_margin data (internal pipeline wiring error)."
        )

    L = np.asarray(box_lengths, dtype=np.float64)

    if cell_matrix is not None:
        H = np.asarray(cell_matrix, dtype=np.float64)
        # ASE Lattice="ax ay az bx by bz cx cy cz" is ROW-major (row i =
        # box vector i); grainsmith's H has box vectors as COLUMNS, hence
        # the transpose.
        Ht = H.T
        lattice = " ".join(fmt(v) for v in Ht.ravel())
    else:
        lat = L.copy()
        for ax in range(3):
            if not periodic[ax]:
                lat[ax] += vacuum
        lattice = (f"{fmt(lat[0])} 0.0 0.0 "
                   f"0.0 {fmt(lat[1])} 0.0 "
                   f"0.0 0.0 {fmt(lat[2])}")
    props = "species:S:1:pos:R:3:grain:I:1"
    if per_atom_margin:
        props += ":gb_margin:R:1"
    pbc = " ".join("T" if p else "F" for p in periodic)

    n = len(atoms)
    pos, species, grain = atoms.pos, atoms.species, atoms.grain

    def serial(a: int, b: int) -> str:
        m = margins[a:b] if margins is not None else None
        return _format_extxyz_chunk(pos[a:b], species[a:b], grain[a:b], m,
                                    periodic, L, vacuum,
                                    cell_matrix=cell_matrix)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as fh:
        fh.write(f"{n}\n")
        fh.write(f'Lattice="{lattice}" Properties={props} pbc="{pbc}" '
                 f'provenance="{provenance.line()}"\n')
        if n == 0:
            return
        nchunks = (n + WRITE_CHUNK - 1) // WRITE_CHUNK
        if jobs <= 1 or nchunks <= 1:
            for a in range(0, n, WRITE_CHUNK):
                fh.write(serial(a, min(a + WRITE_CHUNK, n)))
            return
        from concurrent.futures import ProcessPoolExecutor

        ctx = {"periodic": periodic, "L": L, "vacuum": vacuum,
               "cell_matrix": cell_matrix}
        with ProcessPoolExecutor(
            max_workers=min(jobs, nchunks),
            initializer=_init_extxyz_worker,
            initargs=(ctx,),
        ) as pool:
            def submit(a: int, b: int):
                m = margins[a:b] if margins is not None else None
                return pool.submit(_extxyz_chunk_worker, pos[a:b],
                                   species[a:b], grain[a:b], m)
            drain_in_order(fh, n, WRITE_CHUNK, max(2 * jobs, 2), submit)
