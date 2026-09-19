"""LAMMPS writer ↔ re-parser round-trip + extended XYZ (§8.1, §8.2, §10)."""
import numpy as np
import pytest

from grainsmith.atoms.fill import AtomBlock
from grainsmith.constants import ATOMIC_MASSES
from grainsmith.errors import ConfigError
from grainsmith.io import (
    Provenance, gate_g10_lammps, parse_lammps, parse_lammps_header,
    species_type_map, write_extxyz, write_lammps,
)

PROV = Provenance(version="0.1.0", timestamp_iso="2026-06-11T00:00:00Z",
                  seed=42, config_sha256="abcdef012345" + "0" * 52,
                  provenance_sha256="fedcba987654" + "0" * 52, title="io test")
L = np.array([20.0, 20.0, 20.0])


def _block() -> AtomBlock:
    """4 atoms, 2 species, 2 grains; one atom outside the box (wrap test)."""
    return AtomBlock(
        pos=np.array([
            [1.0, 2.0, 3.0],
            [-0.3, 5.0, 1.0],     # x < 0 → wraps to 19.7 on a periodic axis
            [10.0, 10.0, 10.0],
            [19.5, 0.2, 7.0],
        ]),
        species=np.array(["Cu", "Cu", "Al", "Al"], dtype="U2"),
        grain=np.array([0, 0, 1, 1], dtype=np.int32),
    )


def test_type_map_alphabetical():
    tm = species_type_map(_block().species)
    assert tm == {"Al": 1, "Cu": 2}


def test_write_parse_roundtrip_molecular(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    tm = write_lammps(path, atoms, L, [True, True, True], "molecular", PROV)
    parsed = parse_lammps(path)

    assert parsed["natoms"] == 4
    assert parsed["ntypes"] == 2
    assert parsed["atom_style"] == "molecular"
    assert np.array_equal(parsed["ids"], [1, 2, 3, 4])
    # mol = grain id + 1 (1-based)
    assert np.array_equal(parsed["mol"], [1, 1, 2, 2])
    # types follow the alphabetical map
    assert np.array_equal(parsed["types"], [tm["Cu"], tm["Cu"],
                                            tm["Al"], tm["Al"]])
    # bounds exact (shortest-repr round-trip)
    assert np.array_equal(parsed["bounds"],
                          np.array([[0.0, 20.0]] * 3))
    # masses from constants, keyed by type
    assert parsed["masses"][tm["Al"]] == ATOMIC_MASSES["Al"]
    assert parsed["masses"][tm["Cu"]] == ATOMIC_MASSES["Cu"]


def test_atomic_style_has_no_mol_column(tmp_path):
    path = tmp_path / "poly.data"
    write_lammps(path, _block(), L, [True, True, True], "atomic", PROV)
    parsed = parse_lammps(path)
    assert parsed["atom_style"] == "atomic"
    assert parsed["mol"] is None
    assert len(parsed["pos"]) == 4


def test_positions_wrapped_at_write_storage_untouched(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    parsed = parse_lammps(path)
    # written: wrapped (exact — repr round-trip of 19.7)
    assert parsed["pos"][1, 0] == pytest.approx(-0.3 % 20.0, abs=0.0)
    # storage: unwrapped (§1)
    assert atoms.pos[1, 0] == -0.3


def test_vacuum_on_free_axis(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, False], "atomic", PROV,
                 vacuum=10.0)
    parsed = parse_lammps(path)
    assert np.array_equal(parsed["bounds"],
                          np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 30.0]]))
    # atoms shifted by +vacuum/2 on the free axis only
    assert parsed["pos"][0, 2] == 3.0 + 5.0
    assert parsed["pos"][0, 1] == 2.0


def test_unknown_species_raises_and_override_works(tmp_path):
    atoms = _block()
    atoms.species = np.array(["Xq", "Xq", "Al", "Al"], dtype="U2")
    path = tmp_path / "poly.data"
    with pytest.raises(ConfigError):
        write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV,
                 masses={"Xq": 99.5})
    parsed = parse_lammps(path)
    tm = species_type_map(atoms.species)
    assert parsed["masses"][tm["Xq"]] == 99.5


def test_gate_g10_pass_and_mismatch(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "molecular", PROV)
    ok = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0,
                         "molecular")
    assert ok.passed, ok.message

    truncated = AtomBlock(pos=atoms.pos[:3], species=atoms.species[:3],
                          grain=atoms.grain[:3])
    bad = gate_g10_lammps(path, truncated, L, [True, True, True], 0.0,
                          "molecular")
    assert not bad.passed
    assert "atom count" in bad.message


def test_provenance_header_first_line(tmp_path):
    path = tmp_path / "poly.data"
    write_lammps(path, _block(), L, [True, True, True], "atomic", PROV)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# grainsmith 0.1.0 | 2026-06-11T00:00:00Z | "
                            "seed=42 | config sha256=abcdef012345 | "
                            "provenance sha256=fedcba987654")


# ---------------------------------------------------------------------------
# Gate G10 read-back modes (P3.1): full | sampled | off
# ---------------------------------------------------------------------------


def _big_block(n: int = 12) -> AtomBlock:
    """An n-atom single-species block with ids that will be 1..n in order."""
    rng = np.random.default_rng(0)
    return AtomBlock(
        pos=rng.uniform(0.5, 19.5, size=(n, 3)),
        species=np.array(["Cu"] * n, dtype="U2"),
        grain=(np.arange(n, dtype=np.int32) % 3),
    )


def test_parse_lammps_header_is_cheap_and_correct(tmp_path):
    path = tmp_path / "poly.data"
    write_lammps(path, _block(), L, [True, True, True], "molecular", PROV)
    h = parse_lammps_header(path)
    assert h["natoms"] == 4
    assert h["ntypes"] == 2
    assert h["atom_style"] == "molecular"
    assert np.array_equal(h["bounds"], np.array([[0.0, 20.0]] * 3))


@pytest.mark.parametrize("style", ["atomic", "molecular"])
def test_g10_sampled_and_full_agree_on_good_file(tmp_path, style):
    atoms = _big_block(12)
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], style, PROV)
    full = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, style,
                           mode="full")
    sampled = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, style,
                              mode="sampled", sample_k=3)
    assert full.passed, full.message
    assert sampled.passed, sampled.message


def test_g10_default_mode_is_sampled(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "molecular", PROV)
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "molecular")
    assert res.passed
    assert "sampled" in res.message


def test_g10_sampled_rejects_corrupt_header(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    text = path.read_text(encoding="utf-8").replace("4 atoms", "5 atoms")
    path.write_text(text, encoding="utf-8", newline="\n")
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                          mode="sampled")
    assert not res.passed
    assert "atom count" in res.message


def test_g10_sampled_rejects_truncated_file(tmp_path):
    atoms = _big_block(8)
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    # Drop the final atom row but leave the header claiming 8 atoms.
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8",
                    newline="\n")
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                          mode="sampled", sample_k=2)
    assert not res.passed
    assert "truncated" in res.message


def test_g10_sampled_rejects_corrupt_type_atomic(tmp_path):
    atoms = _big_block(8)
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    # Corrupt the first atom row's type field (atomic: id type x y z).
    lines = path.read_text(encoding="utf-8").splitlines()
    ai = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    first_row = ai + 2
    tok = lines[first_row].split()
    tok[1] = "99"                     # bogus type id (only type 1 = Cu exists)
    lines[first_row] = " ".join(tok)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                          mode="sampled", sample_k=2)
    assert not res.passed
    assert "type" in res.message


def test_g10_sampled_rejects_corrupt_sampled_row(tmp_path):
    atoms = _big_block(8)
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "molecular", PROV)
    # Corrupt the first atom row's molecule id (1 -> 999).
    lines = path.read_text(encoding="utf-8").splitlines()
    ai = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    first_row = ai + 2  # "Atoms  # ...", blank, first row
    tok = lines[first_row].split()
    tok[1] = "999"
    lines[first_row] = " ".join(tok)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "molecular",
                          mode="sampled", sample_k=2)
    assert not res.passed
    assert "mol id" in res.message


def test_g10_sampled_is_a_spot_check_misses_middle(tmp_path):
    """Documented trade-off: a corruption outside the first/last window is
    intentionally NOT caught by 'sampled' — 'full' catches it."""
    atoms = _big_block(12)
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    lines = path.read_text(encoding="utf-8").splitlines()
    ai = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    mid = ai + 2 + 5  # a middle atom row (global id 6)
    tok = lines[mid].split()
    tok[0] = "999"     # break id contiguity in the middle
    lines[mid] = " ".join(tok)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    sampled = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                              mode="sampled", sample_k=2)
    full = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                           mode="full")
    assert sampled.passed          # window misses the middle
    assert not full.passed         # full re-parse catches it


def test_g10_off_skips_readback(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.data"
    write_lammps(path, atoms, L, [True, True, True], "atomic", PROV)
    # Corrupt the file thoroughly — 'off' must not read it, so it still passes.
    path.write_text("garbage\n", encoding="utf-8", newline="\n")
    res = gate_g10_lammps(path, atoms, L, [True, True, True], 0.0, "atomic",
                          mode="off")
    assert res.passed
    assert "disabled" in res.message


# ---------------------------------------------------------------------------
# Extended XYZ (§8.2)
# ---------------------------------------------------------------------------


def test_extxyz_layout_and_pbc_flags(tmp_path):
    atoms = _block()
    path = tmp_path / "poly.extxyz"
    write_extxyz(path, atoms, L, [True, True, False], PROV, vacuum=10.0)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "4"
    header = lines[1]
    assert 'Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 30.0"' in header
    assert 'pbc="T T F"' in header
    assert "Properties=species:S:1:pos:R:3:grain:I:1" in header
    assert "gb_margin" not in header
    assert len(lines) == 6
    tok = lines[2].split()
    assert tok[0] == "Cu" and len(tok) == 5
    assert float(tok[3]) == 3.0 + 5.0    # free-axis vacuum shift
    assert tok[4] == "0"                 # 0-based grain id


def test_extxyz_margin_column(tmp_path):
    atoms = _block()
    atoms.gb_margin = np.array([1.5, -0.25, 3.0, 0.0])
    path = tmp_path / "poly.extxyz"
    write_extxyz(path, atoms, L, [True, True, True], PROV,
                 per_atom_margin=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "gb_margin:R:1" in lines[1]
    tok = lines[3].split()
    assert len(tok) == 6
    assert float(tok[5]) == -0.25


def test_extxyz_margin_requested_but_missing_raises(tmp_path):
    with pytest.raises(ConfigError):
        write_extxyz(tmp_path / "x.extxyz", _block(), L,
                     [True, True, True], PROV, per_atom_margin=True)


def test_atomic_writer_leaves_no_partial_on_failure(tmp_path):
    """A mid-write crash must leave the target untouched (previous good file
    preserved) and no stray .tmp — output writers must never publish a truncated
    file at a user-facing path."""
    from grainsmith.io.common import atomic_writer

    target = tmp_path / "out.data"
    target.write_text("PREVIOUS GOOD RUN\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_writer(target) as fh:
            fh.write("half a file")
            raise RuntimeError("worker crashed mid-write")

    assert target.read_text(encoding="utf-8") == "PREVIOUS GOOD RUN\n"
    assert not (tmp_path / "out.data.tmp").exists()


def test_atomic_writer_commits_on_success(tmp_path):
    """On clean exit the temp file is atomically renamed onto the target."""
    from grainsmith.io.common import atomic_writer

    target = tmp_path / "out.data"
    with atomic_writer(target) as fh:
        fh.write("committed content\n")
    assert target.read_text(encoding="utf-8") == "committed content\n"
    assert not (tmp_path / "out.data.tmp").exists()
