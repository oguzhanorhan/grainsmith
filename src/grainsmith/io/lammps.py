"""LAMMPS data-file writer + minimal re-parser for gate G10 (§8.1).

Conventions (§8.1):
- ``units metal`` semantics (Å, amu), orthogonal box.
- Bounds: periodic axes ``0.0 .. L``; free axes ``0.0 .. L + vacuum`` with
  atoms shifted by ``+vacuum/2``.
- ``atom_style atomic``: ``id type x y z``; ``molecular``:
  ``id mol type x y z`` with ``mol = grain_id + 1`` (1-based, OVITO-friendly).
- Coordinates wrapped to [0, L) on periodic axes at write time (§1 — atom
  storage stays unwrapped).
- Atom ids contiguous 1..N in the order of the input block (the pipeline
  sorts atoms by (grain, generation order) before writing — deterministic).
- Type map: species → 1..T, alphabetical; echoed as comment lines and
  returned for summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from grainsmith.atoms.fill import AtomBlock
from grainsmith.atoms.overlap import _wrap_positions, _wrap_positions_general
from grainsmith.constants import ATOMIC_MASSES, TRICLINIC_TILT_BOUND_TOL
from grainsmith.errors import ConfigError
from grainsmith.io.common import (
    WRITE_CHUNK,
    Provenance,
    atomic_writer,
    drain_in_order,
    fmt,
)
from grainsmith.qa import GateResult


def species_type_map(species: np.ndarray) -> dict[str, int]:
    """Alphabetical species → LAMMPS type id (1..T) mapping (§8.1)."""
    return {sp: t + 1 for t, sp in enumerate(sorted(np.unique(species)))}


def _resolve_masses(
    type_map: dict[str, int],
    masses_override: dict[str, float] | None,
) -> dict[str, float]:
    """Per-species masses (amu) from constants.ATOMIC_MASSES, overridable
    via YAML ``output.lammps.masses`` (§8.1)."""
    masses: dict[str, float] = {}
    for sp in type_map:
        if masses_override and sp in masses_override:
            masses[sp] = float(masses_override[sp])
        elif sp in ATOMIC_MASSES:
            masses[sp] = ATOMIC_MASSES[sp]
        else:
            raise ConfigError(
                f"No atomic mass known for species {sp!r}. Add it to "
                "output.lammps.masses in the YAML config, e.g.\n"
                "  output:\n    lammps:\n      masses: {" + sp + ": 55.0}"
            )
    return masses


def _format_lammps_chunk(
    a: int,
    pos_chunk: np.ndarray,
    species_chunk: np.ndarray,
    grain_chunk: np.ndarray,
    type_map: dict[str, int],
    atom_style: str,
    periodic: list[bool],
    L: np.ndarray,
    vacuum: float,
    cell_matrix: np.ndarray | None = None,
) -> str:
    """Format the LAMMPS Atoms rows for atoms ``[a, a+len(chunk))``.

    Byte-identical to the per-row serial loop: ``_wrap_positions`` is applied
    to the chunk (elementwise, so a slice of the wrap equals the wrap of the
    slice), the free-axis ``+0.5*vacuum`` shift is applied unconditionally
    (matching the original, including its ``-0.0 -> 0.0`` normalisation), and
    every value still goes through the same :func:`fmt` = ``repr(float)``.

    *cell_matrix* (``box.cells``), when given, wraps through the
    general triclinic fractional-coordinate path (``_wrap_positions_general``)
    instead of the per-axis Cartesian wrap — required because a restricted-
    triclinic box's periodic images are not axis-aligned wraps.
    """
    if cell_matrix is not None:
        H = np.asarray(cell_matrix, dtype=np.float64)
        w = _wrap_positions_general(pos_chunk, periodic, H, np.linalg.inv(H))
    else:
        w = _wrap_positions(pos_chunk, periodic, L)
    for ax in range(3):
        if not periodic[ax]:
            w[:, ax] += 0.5 * vacuum
    types = [type_map[sp] for sp in species_chunk]
    out: list[str] = []
    if atom_style == "molecular":
        for j in range(len(pos_chunk)):
            k = a + j
            out.append(f"{k + 1} {int(grain_chunk[j]) + 1} {types[j]} "
                       f"{fmt(w[j, 0])} {fmt(w[j, 1])} {fmt(w[j, 2])}")
    else:
        for j in range(len(pos_chunk)):
            k = a + j
            out.append(f"{k + 1} {types[j]} "
                       f"{fmt(w[j, 0])} {fmt(w[j, 1])} {fmt(w[j, 2])}")
    return "\n".join(out) + "\n" if out else ""


_LAMMPS_CTX: dict | None = None
"""Per-worker-process write context, set once by the pool initializer
(mirrors fill._FILL_CTX — process-local plumbing, not shared module state)."""


def _init_lammps_worker(ctx: dict) -> None:
    global _LAMMPS_CTX
    _LAMMPS_CTX = ctx


def _lammps_chunk_worker(a: int, pos_chunk: np.ndarray,
                         species_chunk: np.ndarray,
                         grain_chunk: np.ndarray) -> str:
    c = _LAMMPS_CTX
    assert c is not None  # initializer ran before any task
    return _format_lammps_chunk(
        a, pos_chunk, species_chunk, grain_chunk, c["type_map"],
        c["atom_style"], c["periodic"], c["L"], c["vacuum"],
        cell_matrix=c.get("cell_matrix"))


def _stream_lammps_atoms(fh, atoms: AtomBlock, periodic: list[bool],
                         L: np.ndarray, vacuum: float,
                         type_map: dict[str, int], atom_style: str,
                         jobs: int, cell_matrix: np.ndarray | None = None) -> None:
    """Write the Atoms section in row-chunks: serial streaming for
    ``jobs<=1`` / small files, else a bounded sliding window of worker
    processes formatting chunks in parallel and written in order."""
    n = len(atoms)
    if n == 0:
        return
    pos, species, grain = atoms.pos, atoms.species, atoms.grain

    def serial(a: int, b: int) -> str:
        return _format_lammps_chunk(a, pos[a:b], species[a:b], grain[a:b],
                                    type_map, atom_style, periodic, L, vacuum,
                                    cell_matrix=cell_matrix)

    nchunks = (n + WRITE_CHUNK - 1) // WRITE_CHUNK
    if jobs <= 1 or nchunks <= 1:
        for a in range(0, n, WRITE_CHUNK):
            fh.write(serial(a, min(a + WRITE_CHUNK, n)))
        return

    from concurrent.futures import ProcessPoolExecutor

    ctx = {"type_map": type_map, "atom_style": atom_style,
           "periodic": periodic, "L": L, "vacuum": vacuum,
           "cell_matrix": cell_matrix}
    with ProcessPoolExecutor(
        max_workers=min(jobs, nchunks),
        initializer=_init_lammps_worker,
        initargs=(ctx,),
    ) as pool:
        def submit(a: int, b: int):
            return pool.submit(_lammps_chunk_worker, a, pos[a:b],
                               species[a:b], grain[a:b])
        drain_in_order(fh, n, WRITE_CHUNK, max(2 * jobs, 2), submit)


def write_lammps(
    path: Path,
    atoms: AtomBlock,
    box_lengths: np.ndarray,
    periodic: list[bool],
    atom_style: str,
    provenance: Provenance,
    vacuum: float = 0.0,
    masses: dict[str, float] | None = None,
    jobs: int = 1,
    cell_matrix: np.ndarray | None = None,
) -> dict[str, int]:
    """Write a LAMMPS data file (§8.1).  Returns the species → type map.

    The Atoms section is streamed in row-chunks: *jobs* > 1 formats
    chunks across worker processes; the bytes are identical for every *jobs*.

    Parameters
    ----------
    cell_matrix : (3,3), optional
        Restricted-triclinic box matrix (columns = box vectors; already
        LAMMPS-tilt-reduced — see ``crystal.cell.reduce_triclinic_tilts``)
        for a ``box.cells`` single-crystal build.  ``None`` (default)
        writes a plain orthogonal box, with no tilt-factor line.  When given, the
        ``xlo xhi``/``ylo yhi``/``zlo zhi`` lines are followed by an
        ``xy xz yz`` tilt-factor line per the LAMMPS restricted-triclinic
        data-file convention (Howto_triclinic): ``xhi-xlo = H[0,0]``,
        ``yhi-ylo = H[1,1]``, ``zhi-zlo = H[2,2]``, ``xy = H[0,1]``,
        ``xz = H[0,2]``, ``yz = H[1,2]``.  box.cells REQUIRES all-periodic,
        zero-vacuum (config/resolve.py Rule 28), so the free-axis/vacuum
        branch below is never exercised in that mode.
    """
    if atom_style not in ("atomic", "molecular"):
        raise ConfigError(f"Unsupported LAMMPS atom_style: {atom_style!r}")

    L = np.asarray(box_lengths, dtype=np.float64)
    type_map = species_type_map(atoms.species)
    mass_map = _resolve_masses(type_map, masses)

    triclinic = cell_matrix is not None
    tilt = None
    if triclinic:
        H = np.asarray(cell_matrix, dtype=np.float64)
        # Restricted-triclinic convention (a along +x, b in xy-plane, c
        # with positive z): bounds are [0, diag(H)] and the tilt factors
        # are read directly off the upper-triangle-transpose entries.
        bounds = np.array([[0.0, H[0, 0]], [0.0, H[1, 1]], [0.0, H[2, 2]]])
        xy, xz, yz = H[0, 1], H[0, 2], H[1, 2]
        tilt = (xy, xz, yz)
        bound_ax, bound_by = H[0, 0], H[1, 1]
        for name, val, edge in (("xy", xy, bound_ax), ("xz", xz, bound_ax),
                                ("yz", yz, bound_by)):
            if abs(val) > edge / 2.0 + TRICLINIC_TILT_BOUND_TOL:
                raise ConfigError(
                    f"write_lammps: tilt factor {name}={val:.6g} Å exceeds "
                    f"the LAMMPS restricted-triclinic bound {edge / 2.0:.6g} "
                    "Å — cell_matrix must be pre-reduced via "
                    "reduce_triclinic_tilts (internal error)."
                )
    else:
        bounds = np.zeros((3, 2), dtype=np.float64)
        for ax in range(3):
            if periodic[ax]:
                bounds[ax] = (0.0, L[ax])
            else:
                bounds[ax] = (0.0, L[ax] + vacuum)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as fh:
        title = f" | {provenance.title}" if provenance.title else ""
        fh.write(f"# {provenance.line()}{title}\n")
        box_desc = ("restricted-triclinic box (box.cells: exact "
                    "lattice-multiple single crystal)" if triclinic
                    else "orthogonal box")
        fh.write(f"# units metal; {box_desc}; "
                 f"atom_style {atom_style}")
        if atom_style == "molecular":
            fh.write(" (molecule id = grain id, 1-based)")
        fh.write("\n")
        for sp, t in type_map.items():
            fh.write(f"# type {t} = {sp}\n")
        fh.write("\n")
        fh.write(f"{len(atoms)} atoms\n")
        fh.write(f"{len(type_map)} atom types\n")
        fh.write("\n")
        for ax, tag in enumerate(("x", "y", "z")):
            fh.write(f"{fmt(bounds[ax, 0])} {fmt(bounds[ax, 1])} "
                     f"{tag}lo {tag}hi\n")
        if tilt is not None:
            xy, xz, yz = tilt
            fh.write(f"{fmt(xy)} {fmt(xz)} {fmt(yz)} xy xz yz\n")
        fh.write("\nMasses\n\n")
        for sp, t in type_map.items():
            fh.write(f"{t} {fmt(mass_map[sp])}  # {sp}\n")
        fh.write(f"\nAtoms  # {atom_style}\n\n")
        _stream_lammps_atoms(fh, atoms, periodic, L, vacuum, type_map,
                             atom_style, jobs, cell_matrix=cell_matrix)
    return type_map


def parse_lammps(path: Path) -> dict:
    """Minimal re-parser for gate G10 (§8.1): header counts, bounds,
    masses, and atom rows.

    Returns dict with keys: ``natoms``, ``ntypes``, ``bounds`` (3,2),
    ``tilt`` ((3,) [xy, xz, yz] or None for an orthogonal box), ``masses``
    {type: amu}, ``ids``, ``types``, ``mol`` (None for atomic), ``pos``
    (N,3), ``atom_style``.
    """
    natoms = ntypes = None
    bounds = np.full((3, 2), np.nan)
    tilt: np.ndarray | None = None
    masses: dict[int, float] = {}
    atom_style = None
    ids: list[int] = []
    types: list[int] = []
    mols: list[int] = []
    pos: list[list[float]] = []

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    section = None
    axis_of = {"xlo": 0, "ylo": 1, "zlo": 2}
    for raw in lines[1:]:  # first line is the free comment line
        line = raw.split("#", 1)[0].strip()
        if raw.strip().startswith("Atoms"):
            section = "Atoms"
            if "#" in raw:
                atom_style = raw.split("#", 1)[1].strip()
            continue
        if line == "Masses":
            section = "Masses"
            continue
        if not line:
            continue
        if section is None:
            tok = line.split()
            if line.endswith(" atoms"):
                natoms = int(tok[0])
            elif line.endswith(" atom types"):
                ntypes = int(tok[0])
            elif len(tok) == 4 and tok[2] in axis_of:
                ax = axis_of[tok[2]]
                bounds[ax] = (float(tok[0]), float(tok[1]))
            elif len(tok) == 6 and tok[3:6] == ["xy", "xz", "yz"]:
                tilt = np.array([float(tok[0]), float(tok[1]),
                                 float(tok[2])], dtype=np.float64)
        elif section == "Masses":
            tok = line.split()
            masses[int(tok[0])] = float(tok[1])
        elif section == "Atoms":
            tok = line.split()
            ids.append(int(tok[0]))
            if atom_style == "molecular":
                mols.append(int(tok[1]))
                types.append(int(tok[2]))
                pos.append([float(v) for v in tok[3:6]])
            else:
                types.append(int(tok[1]))
                pos.append([float(v) for v in tok[2:5]])

    return {
        "natoms": natoms,
        "ntypes": ntypes,
        "bounds": bounds,
        "tilt": tilt,
        "masses": masses,
        "ids": np.array(ids, dtype=np.int64),
        "types": np.array(types, dtype=np.int64),
        "mol": np.array(mols, dtype=np.int64) if mols else None,
        "pos": np.array(pos, dtype=np.float64).reshape(-1, 3),
        "atom_style": atom_style,
    }


def parse_lammps_header(path: Path) -> dict:
    """Parse only the LAMMPS data-file header — counts, bounds, masses, and
    ``atom_style`` — stopping at the ``Atoms`` section line (no atom rows read).

    Unlike :func:`parse_lammps` this is O(header), not O(N): it is the cheap
    half of gate G10 in ``sampled``/``off`` mode. Returns a dict with
    keys ``natoms``, ``ntypes``, ``bounds`` (3,2), ``tilt`` ((3,) [xy, xz,
    yz] or None), ``masses`` {type: amu}, ``atom_style``.
    """
    natoms = ntypes = None
    bounds = np.full((3, 2), np.nan)
    tilt: np.ndarray | None = None
    masses: dict[int, float] = {}
    atom_style = None
    axis_of = {"xlo": 0, "ylo": 1, "zlo": 2}
    section = None
    with Path(path).open(encoding="utf-8") as fh:
        first = True
        for raw in fh:
            if first:  # first line is the free comment line
                first = False
                continue
            if raw.strip().startswith("Atoms"):
                if "#" in raw:
                    atom_style = raw.split("#", 1)[1].strip()
                break  # header fully consumed — do not read atom rows
            line = raw.split("#", 1)[0].strip()
            if line == "Masses":
                section = "Masses"
                continue
            if not line:
                continue
            tok = line.split()
            if section == "Masses":
                masses[int(tok[0])] = float(tok[1])
            elif line.endswith(" atoms"):
                natoms = int(tok[0])
            elif line.endswith(" atom types"):
                ntypes = int(tok[0])
            elif len(tok) == 4 and tok[2] in axis_of:
                bounds[axis_of[tok[2]]] = (float(tok[0]), float(tok[1]))
            elif len(tok) == 6 and tok[3:6] == ["xy", "xz", "yz"]:
                tilt = np.array([float(tok[0]), float(tok[1]),
                                 float(tok[2])], dtype=np.float64)
    return {
        "natoms": natoms,
        "ntypes": ntypes,
        "bounds": bounds,
        "tilt": tilt,
        "masses": masses,
        "atom_style": atom_style,
    }


def _read_first_atom_rows(path: Path, k: int) -> list[str]:
    """Return the first *k* non-empty rows of the ``Atoms`` section."""
    rows: list[str] = []
    if k <= 0:
        return rows
    with Path(path).open(encoding="utf-8") as fh:
        in_atoms = False
        for raw in fh:
            if not in_atoms:
                if raw.strip().startswith("Atoms"):
                    in_atoms = True
                continue
            line = raw.strip()
            if not line:
                continue  # the blank line after "Atoms  # <style>"
            rows.append(line)
            if len(rows) >= k:
                break
    return rows


def _read_last_lines(path: Path, k: int) -> list[str]:
    """Return the last *k* non-empty lines without reading the whole file.

    Reads a trailing byte window (growing it until it holds *k* full lines),
    so the cost is O(k) regardless of file size. The final *k* non-empty
    lines of a LAMMPS data file are exactly atom rows (nothing is written
    after them), provided ``k <= natoms``.
    """
    if k <= 0:
        return []
    size = Path(path).stat().st_size
    if size == 0:
        return []
    chunk = min(size, max(64 * 1024, k * 160))
    while True:
        with Path(path).open("rb") as fh:
            fh.seek(size - chunk)
            data = fh.read(chunk)
        text = data.decode("utf-8", errors="replace")
        if chunk < size:  # drop the (possibly partial) first line
            nl = text.find("\n")
            text = text[nl + 1:] if nl >= 0 else ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= k or chunk >= size:
            return lines[-k:]
        chunk = min(size, chunk * 2)


def _expected_bounds(box_lengths: np.ndarray, periodic: list[bool],
                     vacuum: float,
                     cell_matrix: np.ndarray | None = None) -> np.ndarray:
    if cell_matrix is not None:
        H = np.asarray(cell_matrix, dtype=np.float64)
        return np.array([[0.0, H[0, 0]], [0.0, H[1, 1]], [0.0, H[2, 2]]])
    L = np.asarray(box_lengths, dtype=np.float64)
    expected = np.zeros((3, 2))
    for ax in range(3):
        expected[ax, 1] = L[ax] if periodic[ax] else L[ax] + vacuum
    return expected


def _expected_tilt(cell_matrix: np.ndarray | None) -> np.ndarray | None:
    if cell_matrix is None:
        return None
    H = np.asarray(cell_matrix, dtype=np.float64)
    return np.array([H[0, 1], H[0, 2], H[1, 2]], dtype=np.float64)


def gate_g10_lammps(
    path: Path,
    atoms: AtomBlock,
    box_lengths: np.ndarray,
    periodic: list[bool],
    vacuum: float,
    atom_style: str,
    *,
    mode: str = "sampled",
    sample_k: int = 1000,
    cell_matrix: np.ndarray | None = None,
) -> GateResult:
    """G10: verify the written LAMMPS data file against what the writer
    produced — counts, types, bounds, id contiguity, and the
    molecule-id == grain-id contract (hard, §9).

    Coordinates are written with shortest-round-trip float repr, so the
    bounds comparison is exact (atol covers pathological libc printf only).

    *cell_matrix* (``box.cells``), when given, additionally verifies
    the ``xy xz yz`` tilt-factor line matches the restricted-triclinic box
    matrix and re-checks every sampled atom row lies within the tilted
    parallelepiped (not just the axis-aligned bounding box) via a
    fractional-coordinate containment test.

    *mode* selects how much of the file is re-read:

    - ``"full"`` — re-parse every atom row (today's behaviour, O(N)).
    - ``"sampled"`` (default) — parse the header plus the first and last
      *sample_k* atom rows; O(1) in N. Catches catastrophic writer /
      encoding / truncation breakage without the full re-read.
    - ``"off"`` — skip the on-disk read-back entirely; the counts and bounds
      are still asserted by construction (the writer knows them).
    """
    if mode == "full":
        return _gate_g10_full(path, atoms, box_lengths, periodic, vacuum,
                              atom_style, cell_matrix=cell_matrix)
    if mode == "off":
        return GateResult(
            gate="G10",
            passed=True,
            measured=len(atoms),
            message=(f"read-back disabled (output.lammps.g10_readback=off); "
                     f"{len(atoms)} atoms written by construction"),
        )
    if mode != "sampled":
        raise ConfigError(
            f"Unknown g10_readback mode {mode!r} (full|sampled|off)")
    return _gate_g10_sampled(path, atoms, box_lengths, periodic, vacuum,
                             atom_style, sample_k, cell_matrix=cell_matrix)


def _check_triclinic_containment(pos: np.ndarray,
                                 cell_matrix: np.ndarray) -> list[str]:
    """Verify every row of *pos* lies within [0, 1) fractional coordinates
    of the restricted-triclinic box (a proper superset check of the
    axis-aligned bounds comparison, which cannot detect an atom sitting
    inside the bounding box but outside the tilted parallelepiped)."""
    if len(pos) == 0:
        return []
    H = np.asarray(cell_matrix, dtype=np.float64)
    frac = (np.linalg.inv(H) @ pos.T).T
    tol = 1e-9
    bad = np.where((frac < -tol) | (frac >= 1.0 + tol))[0]
    if len(bad) == 0:
        return []
    return [f"{len(bad)} atom(s) outside the [0,1) fractional triclinic "
            f"cell (e.g. row {int(bad[0])}: frac={frac[bad[0]].tolist()})"]


def _gate_g10_full(
    path: Path,
    atoms: AtomBlock,
    box_lengths: np.ndarray,
    periodic: list[bool],
    vacuum: float,
    atom_style: str,
    cell_matrix: np.ndarray | None = None,
) -> GateResult:
    """G10, ``full`` mode: re-parse every atom row (O(N))."""
    parsed = parse_lammps(path)
    problems: list[str] = []

    if parsed["natoms"] != len(atoms):
        problems.append(
            f"atom count {parsed['natoms']} != written {len(atoms)}")
    if len(parsed["ids"]) != len(atoms):
        problems.append(
            f"Atoms section has {len(parsed['ids'])} rows, "
            f"expected {len(atoms)}")
    n_types_expected = len(np.unique(atoms.species))
    if parsed["ntypes"] != n_types_expected:
        problems.append(
            f"atom types {parsed['ntypes']} != expected {n_types_expected}")

    expected = _expected_bounds(box_lengths, periodic, vacuum,
                                cell_matrix=cell_matrix)
    if not np.allclose(parsed["bounds"], expected, rtol=0.0, atol=1e-12):
        problems.append(
            f"bounds {parsed['bounds'].tolist()} != "
            f"expected {expected.tolist()}")

    expected_tilt = _expected_tilt(cell_matrix)
    if expected_tilt is not None:
        # _expected_tilt returns None for a None cell_matrix, so a non-None
        # tilt proves the matrix is present.
        assert cell_matrix is not None
        if parsed["tilt"] is None:
            problems.append("expected xy xz yz tilt line, none found")
        elif not np.allclose(parsed["tilt"], expected_tilt, rtol=0.0,
                             atol=1e-12):
            problems.append(
                f"tilt {parsed['tilt'].tolist()} != "
                f"expected {expected_tilt.tolist()}")
        problems.extend(_check_triclinic_containment(
            parsed["pos"], cell_matrix))

    if len(parsed["ids"]) and not np.array_equal(
            np.sort(parsed["ids"]), np.arange(1, len(atoms) + 1)):
        problems.append("atom ids are not contiguous 1..N")

    if atom_style == "molecular":
        if parsed["mol"] is None:
            problems.append("molecular file has no molecule-id column")
        elif not np.array_equal(parsed["mol"],
                                atoms.grain.astype(np.int64) + 1):
            problems.append("molecule ids do not equal grain ids + 1")

    return GateResult(
        gate="G10",
        passed=not problems,
        measured=parsed["natoms"],
        message="; ".join(problems) if problems
                else f"re-parse round-trip OK ({parsed['natoms']} atoms, "
                     f"{parsed['ntypes']} types)",
    )


def _gate_g10_sampled(
    path: Path,
    atoms: AtomBlock,
    box_lengths: np.ndarray,
    periodic: list[bool],
    vacuum: float,
    atom_style: str,
    sample_k: int,
    cell_matrix: np.ndarray | None = None,
) -> GateResult:
    """G10, ``sampled`` mode: header + first/last *sample_k* atom rows."""
    n = len(atoms)
    n_types_expected = len(np.unique(atoms.species))
    type_of = species_type_map(atoms.species)
    grain = atoms.grain.astype(np.int64)
    ncol = 6 if atom_style == "molecular" else 5
    problems: list[str] = []
    sampled_rows_pos: list[list[float]] = []

    header = parse_lammps_header(path)
    if header["natoms"] != n:
        problems.append(f"header atom count {header['natoms']} != written {n}")
    if header["ntypes"] != n_types_expected:
        problems.append(
            f"header atom types {header['ntypes']} != "
            f"expected {n_types_expected}")
    if header["atom_style"] != atom_style:
        problems.append(
            f"header atom_style {header['atom_style']!r} != "
            f"requested {atom_style!r}")
    expected = _expected_bounds(box_lengths, periodic, vacuum,
                                cell_matrix=cell_matrix)
    if not np.allclose(header["bounds"], expected, rtol=0.0, atol=1e-12):
        problems.append(
            f"bounds {header['bounds'].tolist()} != "
            f"expected {expected.tolist()}")
    expected_tilt = _expected_tilt(cell_matrix)
    if expected_tilt is not None:
        if header["tilt"] is None:
            problems.append("expected xy xz yz tilt line, none found")
        elif not np.allclose(header["tilt"], expected_tilt, rtol=0.0,
                             atol=1e-12):
            problems.append(
                f"tilt {header['tilt'].tolist()} != "
                f"expected {expected_tilt.tolist()}")

    def _check_rows(rows: list[str], start: int) -> None:
        for j, line in enumerate(rows):
            gi = start + j  # 0-based global atom index
            tok = line.split()
            if len(tok) != ncol:
                problems.append(
                    f"atom row {gi + 1} has {len(tok)} columns, "
                    f"expected {ncol}")
                continue
            try:
                if int(tok[0]) != gi + 1:
                    problems.append(
                        f"atom row {gi + 1}: id {tok[0]} != {gi + 1}")
                if atom_style == "molecular":
                    if int(tok[1]) != grain[gi] + 1:
                        problems.append(
                            f"atom {gi + 1}: mol id {tok[1]} != "
                            f"grain+1 {grain[gi] + 1}")
                    type_tok = int(tok[2])
                else:
                    type_tok = int(tok[1])
                if type_tok != type_of[atoms.species[gi]]:
                    problems.append(
                        f"atom {gi + 1}: type {type_tok} != "
                        f"{type_of[atoms.species[gi]]} for "
                        f"species {atoms.species[gi]!r}")
                row_pos = [float(v) for v in tok[-3:]]  # must parse
                if cell_matrix is not None:
                    sampled_rows_pos.append(row_pos)
            except ValueError:
                problems.append(f"atom row {gi + 1} is malformed: {line!r}")

    if not problems and n:
        k = min(sample_k, n)
        first_rows = _read_first_atom_rows(path, k)
        if len(first_rows) < k:
            problems.append(
                f"Atoms section has only {len(first_rows)} rows, "
                f"expected at least {k}")
        else:
            _check_rows(first_rows, 0)
        last_rows = _read_last_lines(path, k)
        _check_rows(last_rows, n - len(last_rows))
        # Truncation guard: the final atom row's id must be exactly N.
        if last_rows:
            last_id = last_rows[-1].split()[0]
            if last_id.isdigit() and int(last_id) != n:
                problems.append(
                    f"last atom id {last_id} != {n} (file truncated?)")
        if cell_matrix is not None and sampled_rows_pos:
            problems.extend(_check_triclinic_containment(
                np.array(sampled_rows_pos, dtype=np.float64), cell_matrix))

    sampled = min(sample_k, n)
    return GateResult(
        gate="G10",
        passed=not problems,
        measured=header["natoms"],
        message="; ".join(problems) if problems
                else (f"sampled round-trip OK ({header['natoms']} atoms, "
                      f"{header['ntypes']} types; header + first/last "
                      f"{sampled} rows)"),
    )
