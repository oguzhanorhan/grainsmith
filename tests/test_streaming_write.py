"""Byte-identity of the streaming / parallel LAMMPS + extxyz writers (P3.2).

The Atoms section is now formatted in contiguous row-chunks, optionally across
worker processes.  The hard invariant (§1) is that the output bytes are
identical to the original per-row serial loop for ANY chunk size and ANY
``jobs``.  These tests:

  1. force many chunks (and the parallel sliding window) by shrinking
     ``WRITE_CHUNK`` to a few rows;
  2. compare the produced atom rows against an INDEPENDENT, obviously-correct
     per-row reference (catches formatting regressions vs the original);
  3. compare ``jobs=1`` vs ``jobs=4`` full files byte-for-byte (the
     jobs-independence clause — catches any ordering / boundary bug).
"""
import numpy as np
import pytest

import grainsmith.io.lammps as lammps_mod
import grainsmith.io.xyz as xyz_mod
from grainsmith.atoms.fill import AtomBlock
from grainsmith.atoms.overlap import _wrap_positions
from grainsmith.io import (
    Provenance, species_type_map, write_extxyz, write_lammps,
)
from grainsmith.io.common import fmt

PROV = Provenance(version="0.1.0", timestamp_iso="2026-06-20T00:00:00Z",
                  seed=7, config_sha256="0123456789ab" + "0" * 52,
                  provenance_sha256="ba9876543210" + "0" * 52,
                  title="stream test")
L = np.array([20.0, 20.0, 20.0])


@pytest.fixture(autouse=True)
def _tiny_chunk(monkeypatch):
    # Five rows per chunk forces multi-chunk joins and the parallel window
    # to engage on small blocks (no need for 250k-atom fixtures).
    monkeypatch.setattr(lammps_mod, "WRITE_CHUNK", 5)
    monkeypatch.setattr(xyz_mod, "WRITE_CHUNK", 5)


def _block(n: int, species=("Cu",), margins: bool = False) -> AtomBlock:
    rng = np.random.default_rng(123)
    pos = rng.uniform(-2.0, 22.0, size=(n, 3))   # some atoms wrap into the box
    sp = np.array([species[i % len(species)] for i in range(n)], dtype="U2")
    grain = np.sort(rng.integers(0, 4, size=n)).astype(np.int32)
    blk = AtomBlock(pos=pos, species=sp, grain=grain)
    if margins:
        blk.gb_margin = rng.uniform(-3.0, 3.0, size=n)
    return blk


# --- independent per-row references ----------------------------------------

def _ref_lammps_rows(atoms, periodic, vacuum, atom_style) -> str:
    tm = species_type_map(atoms.species)
    pos = _wrap_positions(atoms.pos, periodic, L)
    for ax in range(3):
        if not periodic[ax]:
            pos[:, ax] += 0.5 * vacuum
    rows = []
    for k in range(len(atoms)):
        if atom_style == "molecular":
            rows.append(f"{k + 1} {int(atoms.grain[k]) + 1} "
                        f"{tm[atoms.species[k]]} "
                        f"{fmt(pos[k, 0])} {fmt(pos[k, 1])} {fmt(pos[k, 2])}")
        else:
            rows.append(f"{k + 1} {tm[atoms.species[k]]} "
                        f"{fmt(pos[k, 0])} {fmt(pos[k, 1])} {fmt(pos[k, 2])}")
    return "".join(r + "\n" for r in rows)


def _ref_extxyz_rows(atoms, periodic, vacuum, margins) -> str:
    pos = _wrap_positions(atoms.pos, periodic, L)
    for ax in range(3):
        if not periodic[ax]:
            pos[:, ax] += 0.5 * vacuum
    rows = []
    for k in range(len(atoms)):
        line = (f"{atoms.species[k]} "
                f"{fmt(pos[k, 0])} {fmt(pos[k, 1])} {fmt(pos[k, 2])} "
                f"{int(atoms.grain[k])}")
        if margins is not None:
            line += f" {fmt(margins[k])}"
        rows.append(line)
    return "".join(r + "\n" for r in rows)


def _lammps_atom_block(path) -> str:
    lines = path.read_text(encoding="utf-8").split("\n")
    i = next(idx for idx, ln in enumerate(lines) if ln.startswith("Atoms"))
    rows = lines[i + 2:]                       # skip "Atoms  # ..." + blank
    while rows and rows[-1] == "":
        rows = rows[:-1]
    return "".join(r + "\n" for r in rows)


def _extxyz_atom_block(path) -> str:
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = lines[2:]                           # skip count + comment line
    while rows and rows[-1] == "":
        rows = rows[:-1]
    return "".join(r + "\n" for r in rows)


# --- LAMMPS ----------------------------------------------------------------

@pytest.mark.parametrize("atom_style", ["atomic", "molecular"])
@pytest.mark.parametrize("periodic,vacuum", [
    ([True, True, True], 0.0),
    ([True, True, False], 12.0),     # vacuum shift on the free axis
])
def test_lammps_chunked_serial_matches_reference(tmp_path, atom_style,
                                                 periodic, vacuum):
    atoms = _block(23, species=("Cu", "Al", "Ni"))   # multi-type, 5 chunks
    path = tmp_path / "p.data"
    write_lammps(path, atoms, L, periodic, atom_style, PROV, vacuum=vacuum,
                 jobs=1)
    assert _lammps_atom_block(path) == _ref_lammps_rows(
        atoms, periodic, vacuum, atom_style)


@pytest.mark.parametrize("atom_style", ["atomic", "molecular"])
def test_lammps_jobs_independent_bytes(tmp_path, atom_style):
    atoms = _block(23, species=("Cu", "Al"))
    p1 = tmp_path / "j1.data"
    p4 = tmp_path / "j4.data"
    write_lammps(p1, atoms, L, [True, True, True], atom_style, PROV, jobs=1)
    write_lammps(p4, atoms, L, [True, True, True], atom_style, PROV, jobs=4)
    assert p1.read_bytes() == p4.read_bytes()
    # …and still equal to the independent per-row reference.
    assert _lammps_atom_block(p4) == _ref_lammps_rows(
        atoms, [True, True, True], 0.0, atom_style)


def test_lammps_empty_block(tmp_path):
    # Zero atoms: the writer still emits a well-formed, atom-row-free file.
    atoms = AtomBlock(pos=np.zeros((0, 3)), species=np.array([], dtype="U2"),
                      grain=np.array([], dtype=np.int32))
    path = tmp_path / "empty.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    assert _lammps_atom_block(path) == ""


# --- extended XYZ ----------------------------------------------------------

@pytest.mark.parametrize("periodic,vacuum", [
    ([True, True, True], 0.0),
    ([True, True, False], 8.0),
])
def test_extxyz_chunked_serial_matches_reference(tmp_path, periodic, vacuum):
    atoms = _block(23, species=("Cu", "Al"))
    path = tmp_path / "p.extxyz"
    write_extxyz(path, atoms, L, periodic, PROV, vacuum=vacuum, jobs=1)
    assert _extxyz_atom_block(path) == _ref_extxyz_rows(
        atoms, periodic, vacuum, None)


def test_extxyz_jobs_independent_with_margins(tmp_path):
    atoms = _block(23, species=("Cu", "Al"), margins=True)
    p1 = tmp_path / "j1.extxyz"
    p4 = tmp_path / "j4.extxyz"
    write_extxyz(p1, atoms, L, [True, True, True], PROV,
                 per_atom_margin=True, jobs=1)
    write_extxyz(p4, atoms, L, [True, True, True], PROV,
                 per_atom_margin=True, jobs=4)
    assert p1.read_bytes() == p4.read_bytes()
    assert _extxyz_atom_block(p4) == _ref_extxyz_rows(
        atoms, [True, True, True], 0.0, atoms.gb_margin)


def test_extxyz_jobs_independent_no_margins(tmp_path):
    # The plain (no gb_margin column) parallel extxyz path must also be
    # byte-identical between jobs=1 and jobs=4 — distinct from the margin path.
    atoms = _block(23, species=("Cu", "Al"))
    p1 = tmp_path / "j1.extxyz"
    p4 = tmp_path / "j4.extxyz"
    write_extxyz(p1, atoms, L, [True, True, True], PROV, jobs=1)
    write_extxyz(p4, atoms, L, [True, True, True], PROV, jobs=4)
    assert p1.read_bytes() == p4.read_bytes()
    assert _extxyz_atom_block(p4) == _ref_extxyz_rows(
        atoms, [True, True, True], 0.0, None)
