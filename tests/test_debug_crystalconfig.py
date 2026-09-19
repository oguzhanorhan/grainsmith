"""Scientific-debug property/spot-check tests for the crystal and config
modules (defense-in-depth review).

These tests are ADDITIVE to test_cif.py / test_config.py / test_triclinic.py
/ test_single_crystal.py — they target the specific claims re-verified in
this review pass:

* space-group orbit expansion (crystal/spacegroup.py) reproduces the
  requested space group under an independent spglib round-trip, for a
  spread of Bravais families including two long-orbit cubic groups
  (225 Fm-3m, 227 Fd-3m — the ones named in the review brief);
* Wyckoff site multiplicities match published values for a few textbook
  positions;
* proper_rotation_quaternions (crystal/pointgroup.py) returns the correct
  ORDER of the proper rotation subgroup for a spread of point groups;
* the cell_matrix six-parameter convention (crystal/cell.py) is a faithful
  round-trip (lengths/angles recovered) and matches the documented
  "a along +x, b in the xy-plane" (upper-triangular in the row sense /
  lower-triangular in the column sense used here) shape;
* reduce_triclinic_tilts (crystal/cell.py) is a genuine GL(3,Z) change of
  basis (|det| == 1, integer combination) that restores the LAMMPS tilt
  bound without changing the volume or the represented lattice;
* the CIF path (crystal/cif.py) correctly flags primitive-rhombohedral-
  axes input for the seven R-centered space groups with an ACTIONABLE
  error (this review's fix — see the defect note in cif.py), instead of
  the generic "adjust symprec" message that cannot fix that case.
"""
from __future__ import annotations

import numpy as np
import pytest
import spglib

from grainsmith.crystal.cell import (
    _FAMILY_RULES,
    cell_matrix,
    family_from_sg,
    reduce_triclinic_tilts,
    validate_cellpar,
)
from grainsmith.crystal.pointgroup import proper_rotation_quaternions
from grainsmith.crystal.spacegroup import (
    WyckoffSite,
    expand_wyckoff,
    hall_from_international,
    symmetry_ops,
)
from grainsmith.crystal.verify import verify_spacegroup
from grainsmith.errors import CrystalError

pytest.importorskip("ase")

# Generic (non-degenerate) lattice lengths/angles used for every family
# below, so no accidental higher symmetry sneaks in for the free
# parameters (a "square" 90/90/90 orthorhombic cell would silently test
# tetragonal/cubic supergroup collapse instead of the requested group).
_GENERIC_LENGTHS = {"a": 4.11, "b": 5.37, "c": 6.83}
_GENERIC_ANGLES = {"alpha": 83.0, "beta": 96.0, "gamma": 71.0}


def _generic_cellpar(family: str):
    free = _FAMILY_RULES[family]["free"]
    params = {
        p: (_GENERIC_ANGLES[p] if p in ("alpha", "beta", "gamma")
            else _GENERIC_LENGTHS[p])
        for p in free
    }
    return validate_cellpar(family, params)


# ---------------------------------------------------------------------------
# Space-group orbit expansion vs. independent spglib round-trip
# ---------------------------------------------------------------------------

# One space group per Bravais family, plus the two long-orbit cubic groups
# named explicitly in the review brief (225, 227).
_SPACEGROUP_SPOTCHECK = [1, 2, 4, 11, 19, 47, 62, 65, 74, 123, 141,
                         146, 148, 160, 167, 173, 176, 186, 194,
                         195, 200, 205, 217, 221, 225, 227, 229, 230]


@pytest.mark.parametrize("sg", _SPACEGROUP_SPOTCHECK)
def test_expand_wyckoff_matches_spglib_roundtrip(sg):
    """A single generic-position Wyckoff site, expanded by
    expand_wyckoff, must be detected as EXACTLY the requested space
    group by an independent spglib call on the resulting basis (the
    same round-trip verify_spacegroup performs, reproduced here as a
    property test with an assertion instead of a pipeline call).

    A single generic decoration can accidentally land on a supergroup
    (e.g. SG 4 -> 11, SG 33 -> 62) purely from the coincidental extra
    symmetry of one species at one generic point — that is a property
    of the *test fixture*, not a defect, so those known cases are
    decorated with two species instead (see
    test_expand_wyckoff_two_species_breaks_accidental_supergroup).
    """
    hall = hall_from_international(sg)
    rots, trans = symmetry_ops(hall)
    family = family_from_sg(sg)
    cellpar = _generic_cellpar(family)
    A = cell_matrix(cellpar.a, cellpar.b, cellpar.c,
                     cellpar.alpha, cellpar.beta, cellpar.gamma)

    rng = np.random.default_rng(1000 + sg)
    x0 = rng.uniform(0.05, 0.45, 3)
    site = WyckoffSite(element="Fe", coords=list(x0))
    basis = expand_wyckoff([site], rots, trans)

    lattice = A.T.tolist()
    numbers = [1] * basis.atoms_per_cell
    ds = spglib.get_symmetry_dataset(
        (lattice, basis.frac.tolist(), numbers), symprec=1e-4)
    assert ds is not None
    if ds.number != sg:
        # Only an ACCIDENTAL SUPERGROUP (more symmetry ops than requested)
        # is a benign fixture property; detecting FEWER ops than requested
        # would mean expand_wyckoff genuinely broke the symmetry and must
        # fail loudly, not xfail.
        rots_det, _ = symmetry_ops(hall_from_international(int(ds.number)))
        assert len(rots_det) > len(rots), (
            f"expand_wyckoff produced SG {ds.number} ({ds.international}) "
            f"with {len(rots_det)} ops from requested SG {sg} "
            f"({len(rots)} ops) — a genuine symmetry-breaking defect"
        )
        pytest.xfail(
            f"single-species generic decoration accidentally has the "
            f"higher symmetry of SG {ds.number} ({ds.international}) — "
            "a fixture property, see test_expand_wyckoff_two_species_*"
        )


@pytest.mark.parametrize("sg", [4, 19, 33, 65, 173, 186])
def test_expand_wyckoff_two_species_breaks_accidental_supergroup(sg):
    """The same accidental-supergroup space groups, decorated with TWO
    distinct species at two independent generic points: this removes
    the coincidental extra symmetry, and verify_spacegroup (the actual
    gate-G2 code path used by pipeline._build_crystal / crystal.cif)
    must accept the requested group exactly.
    """
    hall = hall_from_international(sg)
    rots, trans = symmetry_ops(hall)
    family = family_from_sg(sg)
    cellpar = _generic_cellpar(family)

    rng = np.random.default_rng(2000 + sg)
    x0 = rng.uniform(0.05, 0.45, 3)
    x1 = rng.uniform(0.55, 0.95, 3)
    sites = [WyckoffSite(element="Fe", coords=list(x0)),
             WyckoffSite(element="O", coords=list(x1))]
    basis = expand_wyckoff(sites, rots, trans)
    dataset = verify_spacegroup(cellpar, basis, sg)
    assert dataset["number"] == sg


@pytest.mark.parametrize(
    "sg,coords,expected_multiplicity,label",
    [
        (225, [0.0, 0.0, 0.0], 4, "4a"),
        (225, [0.5, 0.5, 0.5], 4, "4b"),
        (225, [0.25, 0.25, 0.25], 8, "8c"),
        (227, [0.0, 0.0, 0.0], 8, "8a (diamond, origin choice 1)"),
        (229, [0.0, 0.0, 0.0], 2, "2a (W)"),
        (221, [0.0, 0.0, 0.0], 1, "1a (Pm-3m)"),
        (194, [0.0, 0.0, 0.0], 2, "2a (HCP c)"),
        (194, [1 / 3, 2 / 3, 0.25], 2, "2c (HCP basal)"),
    ],
)
def test_wyckoff_multiplicity_matches_ita(
    sg, coords, expected_multiplicity, label,
):
    """Orbit sizes for a few textbook special positions must match the
    published ITA Wyckoff multiplicities exactly."""
    hall = hall_from_international(sg)
    rots, trans = symmetry_ops(hall)
    site = WyckoffSite(element="Fe", coords=coords)
    basis = expand_wyckoff([site], rots, trans)
    assert basis.atoms_per_cell == expected_multiplicity, (
        f"SG {sg} {label}: expected multiplicity {expected_multiplicity}, "
        f"got {basis.atoms_per_cell}"
    )


def test_expand_wyckoff_rejects_coincident_distinct_species():
    """Two DIFFERENT species placed on the same fractional position (a
    config authoring mistake, not genuine mixed occupancy) must raise —
    silently accepting it would produce zero-separation atoms that
    disable the overlap-removal cutoff (see spacegroup.py's "Cross-site
    coincidence guard" comment)."""
    hall = hall_from_international(225)
    rots, trans = symmetry_ops(hall)
    sites = [WyckoffSite(element="Cu", coords=[0.0, 0.0, 0.0]),
             WyckoffSite(element="Ni", coords=[0.0, 0.0, 0.0])]
    with pytest.raises(CrystalError, match="coincident atoms"):
        expand_wyckoff(sites, rots, trans)


# ---------------------------------------------------------------------------
# Point-group proper-rotation counts (crystal/pointgroup.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sg,expected_proper_order",
    [
        (1, 1),      # C1
        (2, 1),      # Ci
        (11, 2),     # C2h
        (19, 4),     # D2 (Sohncke, no improper ops)
        (47, 4),     # D2h -> proper D2
        (65, 4),     # Cmmm -> proper D2
        (123, 8),    # D4h -> proper D4
        (167, 6),    # D3d -> proper D3
        (194, 12),   # D6h -> proper D6
        (221, 24),   # Oh -> proper O
        (225, 24),   # Oh (Fm-3m) -> proper O
        (227, 24),   # Oh (Fd-3m) -> proper O
    ],
)
def test_proper_rotation_quaternion_count(sg, expected_proper_order):
    """The number of DISTINCT proper-rotation quaternions returned must
    equal the order of the proper rotation subgroup of the crystallo-
    graphic point group — i.e. exactly half the full point-group order
    for every centrosymmetric group tested here, and the full order for
    the Sohncke group (19, which has no improper operations to discard).
    """
    hall = hall_from_international(sg)
    rots, trans = symmetry_ops(hall)
    family = family_from_sg(sg)
    cellpar = _generic_cellpar(family)
    A = cell_matrix(cellpar.a, cellpar.b, cellpar.c,
                     cellpar.alpha, cellpar.beta, cellpar.gamma)
    quats = proper_rotation_quaternions(A, rots)
    assert len(quats) == expected_proper_order
    # Every returned quaternion must be unit-norm (a real rotation).
    norms = np.linalg.norm(quats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# cell_matrix six-parameter convention (crystal/cell.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_cell_matrix_roundtrips_lattice_parameters(seed):
    """cell_matrix's documented formula must reproduce (a, b, c, alpha,
    beta, gamma) from the resulting column vectors' own lengths/angles,
    and must produce the documented shape: a1 along +x, a2 in the
    xy-plane (H[1,0] == H[2,0] == H[2,1] == 0 — the "a along x, b in
    xy-plane, c free" restricted-triclinic convention used throughout
    box.cells / reduce_triclinic_tilts).

    Some random (alpha, beta, gamma) triples fail the spherical triangle
    inequality (v² <= 0) and are physically impossible for ANY (a, b,
    c) — cell_matrix correctly raises ConfigError for those (see
    test_cell_matrix_rejects_unphysical_parameters); skip them here.
    """
    from grainsmith.errors import ConfigError

    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(2, 8, 3)
    alpha, beta, gamma = rng.uniform(50, 130, 3)
    try:
        A = cell_matrix(a, b, c, alpha, beta, gamma)
    except ConfigError:
        pytest.skip("unphysical (alpha, beta, gamma) triple for this seed")

    assert abs(A[1, 0]) < 1e-10 and abs(A[2, 0]) < 1e-10 and abs(A[2, 1]) < 1e-10

    a1, a2, a3 = A[:, 0], A[:, 1], A[:, 2]

    def angle(u, v):
        return np.degrees(
            np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    assert np.isclose(np.linalg.norm(a1), a, atol=1e-9)
    assert np.isclose(np.linalg.norm(a2), b, atol=1e-9)
    assert np.isclose(np.linalg.norm(a3), c, atol=1e-9)
    assert np.isclose(angle(a2, a3), alpha, atol=1e-7)
    assert np.isclose(angle(a1, a3), beta, atol=1e-7)
    assert np.isclose(angle(a1, a2), gamma, atol=1e-7)


def test_cell_matrix_rejects_unphysical_parameters():
    """v^2 <= 0 (violates the spherical triangle inequality on the three
    interaxial angles) must raise ConfigError, not silently produce a
    complex/NaN cell."""
    from grainsmith.errors import ConfigError
    with pytest.raises(ConfigError):
        cell_matrix(3.0, 3.0, 3.0, 10.0, 10.0, 170.0)


# ---------------------------------------------------------------------------
# reduce_triclinic_tilts: GL(3,Z) validity + LAMMPS bound (crystal/cell.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(15))
def test_reduce_triclinic_tilts_is_unimodular_and_bounded(seed):
    """reduce_triclinic_tilts(H) must equal H @ M for an INTEGER M with
    |det(M)| == 1 (a genuine GL(3,Z) change of box-vector basis — volume
    and represented lattice are exactly preserved), and the result must
    satisfy the LAMMPS restricted-triclinic tilt bound
    |xy|,|xz| <= ax/2, |yz| <= by/2 (Howto_triclinic)."""
    from grainsmith.errors import ConfigError

    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(2, 6, 3)
    alpha, beta, gamma = rng.uniform(50, 130, 3)
    try:
        A = cell_matrix(a, b, c, alpha, beta, gamma)
    except ConfigError:
        pytest.skip("unphysical (alpha, beta, gamma) triple for this seed")
    n1, n2, n3 = rng.integers(1, 30, 3)
    H = A @ np.diag([float(n1), float(n2), float(n3)])
    Hr = reduce_triclinic_tilts(H)

    # Same represented lattice: Hr = H @ M, M integer, |det M| == 1.
    M = np.linalg.solve(H, Hr)
    assert np.allclose(M, np.round(M), atol=1e-8)
    assert np.isclose(abs(np.linalg.det(M)), 1.0, atol=1e-8)

    # Volume and diagonal (box edge lengths) unchanged.
    assert np.isclose(abs(np.linalg.det(H)), abs(np.linalg.det(Hr)),
                       rtol=1e-10)
    assert np.allclose(np.diag(H), np.diag(Hr), atol=1e-9)

    # LAMMPS restricted-triclinic tilt bound.
    ax, by = Hr[0, 0], Hr[1, 1]
    xy, xz, yz = Hr[0, 1], Hr[0, 2], Hr[1, 2]
    assert abs(xy) <= ax / 2.0 + 1e-9
    assert abs(xz) <= ax / 2.0 + 1e-9
    assert abs(yz) <= by / 2.0 + 1e-9


def test_reduce_triclinic_tilts_preserves_restricted_triclinic_shape():
    """The lower-triangular zero pattern (row 1 col 0, row 2 col 0, row
    2 col 1) that cell_matrix produces must survive reduction — the
    subtractions only touch columns b and c using multiples of a and b,
    which cannot introduce a nonzero in those positions."""
    A = cell_matrix(3.0, 4.0, 5.0, 75.0, 95.0, 60.0)
    H = A @ np.diag([7.0, 11.0, 13.0])
    Hr = reduce_triclinic_tilts(H)
    assert abs(Hr[1, 0]) < 1e-9
    assert abs(Hr[2, 0]) < 1e-9
    assert abs(Hr[2, 1]) < 1e-9


# ---------------------------------------------------------------------------
# family_from_sg / lattice_params_for_family boundary + rhombohedral override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "number,expected_family",
    [
        (1, "triclinic"), (2, "triclinic"),
        (3, "monoclinic"), (15, "monoclinic"),
        (16, "orthorhombic"), (74, "orthorhombic"),
        (75, "tetragonal"), (142, "tetragonal"),
        (143, "trigonal"), (167, "trigonal"),
        (168, "hexagonal"), (194, "hexagonal"),
        (195, "cubic"), (230, "cubic"),
    ],
)
def test_family_from_sg_boundaries(number, expected_family):
    """Every family boundary (both the low and high space-group number
    of each ITA range) must map correctly — an off-by-one here would
    silently route a space group to the wrong cell-matrix construction."""
    assert family_from_sg(number) == expected_family


def test_family_from_sg_rhombohedral_override_is_setting_specific():
    """setting='R' selects the rhombohedral family ONLY for a trigonal
    (R-centered) space-group number; the hexagonal-axis default (None,
    or explicit 'H') must keep the trigonal family."""
    assert family_from_sg(146, "R") == "rhombohedral"
    assert family_from_sg(146, "H") == "trigonal"
    assert family_from_sg(146, None) == "trigonal"
    assert family_from_sg(146, "r") == "rhombohedral"  # case-insensitive


def test_family_from_sg_rejects_out_of_range():
    from grainsmith.errors import ConfigError
    with pytest.raises(ConfigError):
        family_from_sg(0)
    with pytest.raises(ConfigError):
        family_from_sg(231)


# ---------------------------------------------------------------------------
# CIF path: R-centered primitive-rhombohedral-axes input (this review's fix)
# ---------------------------------------------------------------------------

def _write_rhombohedral_hexaxes_cif(path, sg_number=160):
    """Write a CIF for an R-centered space group in the CORRECT
    (hexagonal-axes) setting crystal.cif supports.

    The basis point is chosen off every mirror/rotation special line of
    these point groups (not e.g. y == x/2, y == x) so the generic
    Wyckoff orbit is the FULL orbit and no accidental extra symmetry
    (a different, higher R-centered space group) appears — a property
    of a careless test fixture, not of the code under test.
    """
    from ase.spacegroup import crystal
    import ase.io

    atoms = crystal(
        "Fe", basis=[[0.213, 0.087, 0.311]], spacegroup=sg_number, setting=1,
        cellpar=[6.0, 6.0, 15.0, 90.0, 90.0, 120.0],
    )
    ase.io.write(str(path), atoms)


def _write_rhombohedral_primitive_axes_cif(path, sg_number=160):
    """Write a CIF for the SAME structure in PRIMITIVE RHOMBOHEDRAL axes
    (the ICSD/COD/VESTA convention many real CIFs use) — this is the
    input crystal.cif must reject with an ACTIONABLE message rather
    than the generic 'adjust symprec' hint (see this review's fix in
    cif.py's atom-count-mismatch branch)."""
    import ase.io
    from ase import Atoms
    from ase.spacegroup import crystal

    hex_atoms = crystal(
        "Fe", basis=[[0.213, 0.087, 0.311]], spacegroup=sg_number, setting=1,
        cellpar=[6.0, 6.0, 15.0, 90.0, 90.0, 120.0],
    )
    lattice = hex_atoms.cell[:].tolist()
    positions = hex_atoms.get_scaled_positions(wrap=True).tolist()
    numbers = hex_atoms.get_atomic_numbers().tolist()
    plat, ppos, pnum = spglib.find_primitive(
        (lattice, positions, numbers), symprec=1e-4)
    prim_atoms = Atoms(numbers=pnum, scaled_positions=ppos, cell=plat,
                        pbc=True)
    ase.io.write(str(path), prim_atoms)


@pytest.mark.parametrize("sg_number", [146, 148, 160, 167])
def test_cif_rhombohedral_primitive_axes_gives_actionable_error(
    tmp_path, sg_number,
):
    """A CIF describing one of the seven R-centered space groups in
    PRIMITIVE rhombohedral axes must fail with an error that (a) names
    the real cause (R-centered space group in the wrong axis setting)
    and (b) does NOT suggest adjusting symprec, since no tolerance can
    fix a 3x primitive/conventional atom-count ratio."""
    cif_path = tmp_path / f"rhomb_prim_{sg_number}.cif"
    _write_rhombohedral_primitive_axes_cif(cif_path, sg_number)

    from grainsmith.crystal.cif import read_cif_crystal

    with pytest.raises(CrystalError) as exc_info:
        read_cif_crystal(str(cif_path))
    msg = str(exc_info.value)
    assert "R-centered" in msg
    assert "symprec cannot fix this" in msg


@pytest.mark.parametrize("sg_number", [146, 148, 160, 167])
def test_cif_rhombohedral_hexagonal_axes_succeeds(tmp_path, sg_number):
    """The SAME structure in the hexagonal-axes setting (the one
    crystal.cif supports) must resolve cleanly, with no regression from
    the primitive-axes guard added above."""
    cif_path = tmp_path / f"rhomb_hex_{sg_number}.cif"
    _write_rhombohedral_hexaxes_cif(cif_path, sg_number)

    from grainsmith.crystal.cif import read_cif_crystal

    spec = read_cif_crystal(str(cif_path))
    assert spec.sg_number == sg_number
    assert spec.family in ("trigonal", "rhombohedral")


# ---------------------------------------------------------------------------
# CIF path: occupancy guard
# ---------------------------------------------------------------------------

def test_cif_rejects_partial_occupancy(tmp_path):
    """A CIF with a vacancy (single species, fraction < 1) must be
    rejected — crystal.cif assumes full occupancy; the manual
    wyckoff_sites path with an explicit occupancy dict is the documented
    escape hatch."""
    cif_text = (
        "data_test\n"
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "_atom_site_occupancy\nTi1 Ti 0.0 0.0 0.0 0.85\n"
    )
    cif_path = tmp_path / "vacancy.cif"
    cif_path.write_text(cif_text)

    from grainsmith.crystal.cif import read_cif_crystal
    from grainsmith.errors import ConfigError

    with pytest.raises(ConfigError, match="partial/mixed site occupancy"):
        read_cif_crystal(str(cif_path))


def test_cif_rejects_mixed_occupancy(tmp_path):
    """A CIF with a genuine solid-solution site (two species, fractions
    summing to 1) must also be rejected by the same guard — the manual
    path's occupancy-dict mechanism is the supported route for this."""
    cif_text = (
        "data_test\n"
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Ti1 Ti 0.0 0.0 0.0 0.9\nAl1 Al 0.0 0.0 0.0 0.1\n"
    )
    cif_path = tmp_path / "mixed_occ.cif"
    cif_path.write_text(cif_text)

    from grainsmith.crystal.cif import read_cif_crystal
    from grainsmith.errors import ConfigError

    with pytest.raises(ConfigError, match="partial/mixed site occupancy"):
        read_cif_crystal(str(cif_path))


# ---------------------------------------------------------------------------
# config/resolve.py cross-field rules: box.cells (Rule 28) preconditions
# ---------------------------------------------------------------------------

def _minimal_box_cells_config(**overrides):
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"cells": [2, 2, 2]},
        "grains": {"number": 1},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "fixed",
                         "fixed": {"euler_bunge_deg": [0.0, 0.0, 0.0]}},
    }
    for key, val in overrides.items():
        raw[key] = val
    return raw


def test_box_cells_and_lengths_are_mutually_exclusive():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    raw = _minimal_box_cells_config(
        box={"cells": [2, 2, 2], "lengths": [10.0, 10.0, 10.0]})
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve_config(raw)


def test_box_cells_requires_single_grain():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    raw = _minimal_box_cells_config(grains={"number": 4})
    with pytest.raises(ConfigError, match="grains.number: 1"):
        resolve_config(raw)


def test_box_cells_requires_identity_orientation():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    raw = _minimal_box_cells_config(
        orientation={"scheme": "fixed",
                     "fixed": {"euler_bunge_deg": [10.0, 0.0, 0.0]}})
    with pytest.raises(ConfigError, match="IDENTITY"):
        resolve_config(raw)


def test_box_cells_rejects_non_flat_geometry():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    # grains.number == 1 + curved geometry is already rejected by Rule 24
    # before Rule 28 gets a chance to run its own flat-geometry check —
    # both rules must independently refuse this combination.
    raw = _minimal_box_cells_config(boundaries={"geometry": "curved"})
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_box_cells_produces_bit_identical_resolved_h_for_same_seed():
    """Reproducibility floor: resolving the same config twice must give
    byte-identical resolved_h (no hidden randomness in Rule 28)."""
    from grainsmith.config.resolve import resolve_config

    raw1 = _minimal_box_cells_config()
    raw2 = _minimal_box_cells_config()
    cfg1 = resolve_config(raw1)
    cfg2 = resolve_config(raw2)
    assert cfg1.box.resolved_h == cfg2.box.resolved_h


# ---------------------------------------------------------------------------
# config/resolve.py cross-field rules: crystal.cif (Rule 27)
# ---------------------------------------------------------------------------

def test_crystal_cif_and_manual_trio_are_mutually_exclusive():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [20.0, 20.0, 20.0]},
        "grains": {"number": 2},
        "crystal": {
            "cif": {"file": "does_not_matter.cif"},
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "orientation": {"scheme": "random_uniform"},
    }
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve_config(raw)


def test_crystal_cif_requires_one_input_form():
    from grainsmith.config.resolve import resolve_config
    from grainsmith.errors import ConfigError

    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [20.0, 20.0, 20.0]},
        "grains": {"number": 2},
        "crystal": {},
        "orientation": {"scheme": "random_uniform"},
    }
    with pytest.raises(ConfigError, match="cif"):
        resolve_config(raw)
