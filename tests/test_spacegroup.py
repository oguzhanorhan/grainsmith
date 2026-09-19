"""Tests for crystal/spacegroup.py — §10. Orbit sizes, round-trip detection."""
import pytest
from grainsmith.crystal.spacegroup import (
    hall_from_international, symmetry_ops, expand_wyckoff, WyckoffSite
)


def _expand(sg_number, sites, setting=None):
    hall = hall_from_international(sg_number, setting)
    rots, trans = symmetry_ops(hall)
    return expand_wyckoff(sites, rots, trans)


def test_b2_nial_sg221():
    """SG 221 B2 NiAl: 2 atoms per cell (1a + 1b)."""
    sites = [
        WyckoffSite("Ni", [0.0, 0.0, 0.0], letter="a"),
        WyckoffSite("Al", [0.5, 0.5, 0.5], letter="b"),
    ]
    basis = _expand(221, sites)
    assert basis.atoms_per_cell == 2


def test_fcc_sg225():
    """SG 225 FCC: 4 atoms per cell (4a site)."""
    sites = [WyckoffSite("Cu", [0.0, 0.0, 0.0])]
    basis = _expand(225, sites)
    assert basis.atoms_per_cell == 4


def test_hcp_sg194():
    """SG 194 HCP: 2 atoms per cell (2c site at 1/3,2/3,1/4)."""
    sites = [WyckoffSite("Ti", [1/3, 2/3, 0.25])]
    basis = _expand(194, sites)
    assert basis.atoms_per_cell == 2


def test_diamond_sg227_origin2():
    """SG 227 diamond origin choice 2: 8 atoms per cell."""
    sites = [WyckoffSite("C", [0.125, 0.125, 0.125])]
    basis = _expand(227, sites, setting="2")
    assert basis.atoms_per_cell == 8


def test_rutile_sg136():
    """SG 136 rutile: 6 atoms (2 Ti at 2a + 4 O at 4f with x=0.305)."""
    sites = [
        WyckoffSite("Ti", [0.0, 0.0, 0.0]),
        WyckoffSite("O", [0.305, 0.305, 0.0]),
    ]
    basis = _expand(136, sites)
    assert basis.atoms_per_cell == 6


def test_verify_spacegroup_wrong_number_raises():
    """SCIENCE_AUDIT #2: the G2 spglib round-trip detects a wrong space-group
    NUMBER — an FCC (Fm-3m, 225) basis declared as Pm-3m (221) must raise."""
    from grainsmith.crystal.cell import validate_cellpar
    from grainsmith.crystal.verify import verify_spacegroup
    from grainsmith.errors import CrystalError

    basis = _expand(225, [WyckoffSite("Cu", [0.0, 0.0, 0.0])])   # FCC, 4 atoms
    cellpar = validate_cellpar("cubic", {"a": 3.615})
    with pytest.raises(CrystalError, match="225"):
        verify_spacegroup(cellpar, basis, 221)


def test_hall_unknown():
    from grainsmith.errors import ConfigError
    with pytest.raises(ConfigError):
        hall_from_international(999)


def test_verify_wyckoff_letters_b2():
    """G2 extension: user-supplied Wyckoff letters checked against spglib."""
    from grainsmith.crystal.cell import validate_cellpar
    from grainsmith.crystal.verify import verify_spacegroup, verify_wyckoff_letters
    from grainsmith.errors import CrystalError

    sites = [
        WyckoffSite("Ni", [0.0, 0.0, 0.0], letter="a"),
        WyckoffSite("Al", [0.5, 0.5, 0.5], letter="b"),
    ]
    basis = _expand(221, sites)
    cellpar = validate_cellpar("cubic", {"a": 2.88})
    dataset = verify_spacegroup(cellpar, basis, 221)

    # correct letters (with and without multiplicity prefix) pass
    verify_wyckoff_letters([([0.0, 0.0, 0.0], "a"), ([0.5, 0.5, 0.5], "1b")],
                           basis, dataset["wyckoffs"])
    # None entries are skipped
    verify_wyckoff_letters([([0.0, 0.0, 0.0], None)],
                           basis, dataset["wyckoffs"])
    # a wrong letter raises with both letters in the message
    with pytest.raises(CrystalError, match="mismatch"):
        verify_wyckoff_letters([([0.5, 0.5, 0.5], "a")],
                               basis, dataset["wyckoffs"])


def test_verify_wyckoff_letters_nonstrict_warns_only():
    """Non-default settings: spglib letters refer to its standardized
    description, so mismatches must only warn (diamond origin choice 2:
    ITA 8a at (1/8,1/8,1/8), spglib labels the orbit 'b')."""
    from grainsmith.crystal.cell import validate_cellpar
    from grainsmith.crystal.verify import verify_spacegroup, verify_wyckoff_letters

    sites = [WyckoffSite("Si", [0.125, 0.125, 0.125], letter="a")]
    basis = _expand(227, sites, setting="2")
    cellpar = validate_cellpar("cubic", {"a": 5.431})
    dataset = verify_spacegroup(cellpar, basis, 227)

    # strict=False → no raise, regardless of how spglib labels the orbit
    verify_wyckoff_letters([([0.125, 0.125, 0.125], "a")],
                           basis, dataset["wyckoffs"], strict=False)
