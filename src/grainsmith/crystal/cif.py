"""Direct CIF crystal-structure input (ASE read -> spglib symmetry
detection -> grainsmith Wyckoff-site spec, §6.1-6.3).

Converts an arbitrary CIF file into EXACTLY the same internal
representation the manual ``crystal.space_group`` / ``crystal.lattice`` /
``crystal.wyckoff_sites`` path produces (space-group number + optional
setting, family-reduced free lattice parameters, one representative
Wyckoff site per symmetry orbit) so every downstream stage — orientation,
tessellation, fill, LAMMPS triclinic handling — runs completely unchanged.

Pipeline
--------
1. ``ase.io.read`` parses the CIF into an ``Atoms`` object (lattice +
   fractional coordinates + species + any declared symmetry/occupancy).
2. ``spglib.get_symmetry_dataset`` on the (lattice, scaled_positions,
   atomic_numbers) cell detects the TRUE space group at ``symprec`` —
   independent of whatever the CIF header claims (a CIF can legally list
   every atom under ``P 1`` even though the true symmetry is much higher;
   grainsmith always uses the spglib-detected group).
3. The dataset's own STANDARDIZED cell (``std_lattice`` / ``std_positions``
   / ``std_types``) is re-submitted to spglib so the reported Wyckoff
   letters and equivalent-atom grouping are index-aligned to the exact
   cell grainsmith will build (the first pass's assignment is relative to
   the original, non-standardized cell and may use a different Hall
   setting).
4. One representative atom per symmetry orbit (``equivalent_atoms``)
   becomes one :class:`~grainsmith.crystal.spacegroup.WyckoffSite`; the
   full lattice parameters are reduced to the crystal family's free
   parameters (:func:`grainsmith.crystal.cell.lattice_params_for_family`).
5. The whole spec is round-tripped through the SAME machinery the manual
   path uses (``validate_cellpar`` -> ``hall_from_international`` ->
   ``symmetry_ops`` -> ``expand_wyckoff`` -> ``verify_spacegroup``) here,
   at config-resolution time, so a mismatched setting or tolerance
   surfaces immediately with a CIF-specific error instead of deep inside
   the pipeline.

Traceability (project rule: no silent assumptions): the returned
:class:`CifCrystalSpec` carries the detected space-group number + H-M
symbol, the symprec used, the reduced lattice parameters, and the
per-site Wyckoff letters — the caller (config/resolve.py) surfaces these
via logging and the pipeline threads them into summary.csv / METHODS.md
exactly like the manual path's spglib dataset.

Limitation — R-centered space groups (146, 148, 155, 160, 161, 166, 167):
spglib's standardized cell for these is always the centered HEXAGONAL
axes convention (a=b, alpha=beta=90 deg, gamma=120 deg), so a CIF
describing the structure in PRIMITIVE RHOMBOHEDRAL axes (the common
ICSD/COD/VESTA export convention) reads 1/3 the conventional-cell atom
count and is rejected with a specific, actionable error (step 4 above,
the n_atoms_conventional cross-check) rather than the generic
tolerance-mismatch message. Re-export in hexagonal axes, or use the
manual ``crystal.space_group`` (``setting: "R"``) / ``crystal.lattice``
/ ``crystal.wyckoff_sites`` path, which supports both settings directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from grainsmith.errors import ConfigError, CrystalError

if TYPE_CHECKING:
    from ase import Atoms

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CifWyckoffSite:
    """One symmetry-inequivalent site derived from the CIF (full
    occupancy only — see CifCrystalSpec docstring)."""
    element: str
    coords: list[float]
    letter: str | None


@dataclass(frozen=True)
class CifCrystalSpec:
    """Everything derived from a CIF file, in grainsmith's internal
    representation (space group + family-reduced lattice parameters +
    representative Wyckoff sites).

    ``lattice_params`` is a dict of ONLY the free parameters for
    ``family`` (e.g. ``{"a": ..., "c": ...}`` for hexagonal) — the exact
    shape :func:`grainsmith.crystal.cell.validate_cellpar` expects, so it
    can be handed straight to the manual-path machinery.
    """
    source_file: str
    symprec: float
    sg_number: int
    setting: str | None
    international: str
    hall_number: int
    family: str
    lattice_params: dict[str, float]
    cellpar_full: tuple[float, float, float, float, float, float]
    wyckoff_sites: list[CifWyckoffSite]
    n_atoms_conventional: int
    declared_sg_number: int | None
    declared_sg_mismatch: bool


def _check_full_occupancy(atoms) -> None:
    """Raise ConfigError on any partial or mixed-species occupancy.

    ASE's CIF reader records ``atoms.info["occupancy"]`` as
    ``{site_label: {element: fraction, ...}}``. Full occupancy is exactly
    one element with fraction 1 (within float roundoff); anything else
    (a vacancy fraction < 1, or a solid-solution mix of >= 2 elements) is
    not representable by the CIF path (which builds plain-string
    WyckoffSiteConfig.element entries) — the manual
    ``crystal.wyckoff_sites`` path with an explicit occupancy dict is the
    documented escape hatch.
    """
    occ = atoms.info.get("occupancy")
    if not occ:
        return
    bad: list[str] = []
    for label, frac_map in occ.items():
        if len(frac_map) != 1:
            bad.append(f"site {label}: mixed occupancy {frac_map}")
            continue
        (elem, frac), = frac_map.items()
        if abs(float(frac) - 1.0) > 1e-6:
            bad.append(f"site {label}: partial occupancy {elem}={frac}")
    if bad:
        raise ConfigError(
            "crystal.cif does not support partial/mixed site occupancy "
            "(full occupancy is assumed): " + "; ".join(bad) + ". Use the "
            "manual crystal.wyckoff_sites path instead, with an explicit "
            "occupancy dict, e.g. "
            "{element: {Ti: 0.90, Al: 0.10}, coords: [...]}."
        )


def read_cif_crystal(file: str, symprec: float = 1e-4) -> CifCrystalSpec:
    """Read *file* with ASE, detect symmetry with spglib, and return a
    :class:`CifCrystalSpec` built entirely from the file's own content.

    Raises
    ------
    ConfigError
        Missing/unreadable file, missing ASE/spglib dependency, or
        partial/mixed site occupancy (unsupported by this path).
    CrystalError
        spglib fails to detect any symmetry, or the internal round-trip
        (expand Wyckoff orbit -> re-verify with spglib) does not
        reproduce the CIF's own atom count — the atom-doubling / setting
        mismatch guard.
    """
    path = Path(file)
    if not path.exists():
        raise ConfigError(f"crystal.cif.file not found: {path}")

    try:
        import ase.io
    except ImportError as exc:
        raise ConfigError(
            "crystal.cif requires the 'ase' package (ase.io.read): "
            "pip install ase"
        ) from exc
    try:
        import spglib
    except ImportError as exc:
        raise ConfigError(
            "crystal.cif requires the 'spglib' package: pip install spglib"
        ) from exc

    try:
        # ase.io.read is typed `Atoms | list[Atoms]` because it returns a
        # list when `index=` selects multiple images; called without
        # `index=` (as here) it always returns a single Atoms.
        atoms = cast("Atoms", ase.io.read(str(path)))
    except Exception as exc:
        raise ConfigError(f"Failed to read CIF file {path}: {exc}") from exc

    _check_full_occupancy(atoms)

    from ase.data import chemical_symbols
    from ase.geometry import cell_to_cellpar

    from grainsmith.crystal.cell import family_from_sg, lattice_params_for_family
    from grainsmith.crystal.spacegroup import _build_hall_map

    lattice = atoms.cell[:].tolist()
    positions = atoms.get_scaled_positions(wrap=True)
    numbers = atoms.get_atomic_numbers()
    n_read = len(atoms)

    ds = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                      symprec=symprec)
    if ds is None:
        raise CrystalError(
            f"spglib could not detect any symmetry for {path} at "
            f"symprec={symprec:g}. Check the CIF's cell/coordinates, or "
            "loosen crystal.cif.symprec."
        )

    # Re-run on spglib's own STANDARDIZED cell so the Wyckoff letters and
    # equivalent-atom grouping are index-aligned to the exact cell
    # grainsmith will build below (§ module docstring, step 3).
    std_lattice = np.asarray(ds.std_lattice, dtype=np.float64)
    std_positions = np.asarray(ds.std_positions, dtype=np.float64)
    std_types = np.asarray(ds.std_types, dtype=np.int64)
    ds2 = spglib.get_symmetry_dataset(
        (std_lattice.tolist(), std_positions.tolist(), std_types.tolist()),
        symprec=symprec,
    )
    if ds2 is None:
        raise CrystalError(
            f"spglib failed to re-detect symmetry on its own standardized "
            f"cell for {path} at symprec={symprec:g} (internal "
            "inconsistency) — try adjusting crystal.cif.symprec."
        )

    number = int(ds2.number)
    international = str(ds2.international)
    hall_number = int(ds2.hall_number)
    n_atoms_conventional = len(std_types)

    if n_atoms_conventional != n_read:
        # A ratio of exactly 3 for one of the seven R-centered space
        # groups (146, 148, 155, 160, 161, 166, 167) is NOT a tolerance
        # artifact: it is the expected primitive/conventional atom-count
        # ratio when the CIF gives the PRIMITIVE rhombohedral cell (the
        # common ICSD/COD/VESTA export convention for R-groups) while
        # spglib's ``std_lattice`` is always the centered HEXAGONAL
        # conventional cell (3x the primitive-cell volume/atom count).
        # No amount of symprec tuning changes that ratio, so the generic
        # "adjust crystal.cif.symprec" hint below would be actively
        # misleading here — name the real cause instead.
        _R_CENTERED_SG = {146, 148, 155, 160, 161, 166, 167}
        if (number in _R_CENTERED_SG
                and n_atoms_conventional == 3 * n_read):
            raise CrystalError(
                f"CIF atom-count mismatch for {path}: {n_read} atoms "
                f"were read but the standardized (hexagonal-axes) "
                f"conventional cell for space group {number} "
                f"({international}) has {n_atoms_conventional} — this "
                "is the expected 3x primitive/conventional ratio for an "
                "R-centered space group given in PRIMITIVE RHOMBOHEDRAL "
                "axes. crystal.cif only supports CIFs already in the "
                "centered hexagonal-axes setting (a=b, alpha=beta=90 deg, "
                "gamma=120 deg) — re-export the structure in that "
                "setting (most crystallographic software offers a "
                "hexagonal/rhombohedral axis-choice toggle), or use the "
                "manual crystal.space_group (setting: 'R') / "
                "crystal.lattice / crystal.wyckoff_sites path instead. "
                "Adjusting crystal.cif.symprec cannot fix this."
            )
        raise CrystalError(
            f"CIF atom-count mismatch for {path}: {n_read} atoms were "
            f"read from the file but spglib's standardized conventional "
            f"cell has {n_atoms_conventional} atoms — the file may "
            "describe a non-conventional (e.g. primitive) cell that "
            "spglib is expanding/reducing; inspect the CIF and adjust "
            "crystal.cif.symprec if this is a tolerance artifact."
        )

    # Resolve the setting string (mirrors grainsmith's own Hall-number
    # convention, crystal.spacegroup.hall_from_international): None when
    # spglib's detected Hall setting IS grainsmith's default for this
    # space-group number, otherwise the spglib "choice" label.
    hall_map = _build_hall_map()
    default_hall = hall_map.get((number, None))
    setting: str | None = None
    if hall_number != default_hall:
        info = spglib.get_spacegroup_type(hall_number)
        choice = getattr(info, "choice", None) or None
        setting = str(choice) if choice else None

    family = family_from_sg(number, setting)
    # ase.geometry.cell_to_cellpar always returns exactly (a, b, c, α, β, γ);
    # a genexp cannot express that fixed arity to mypy.
    cellpar_full = cast("tuple[float, float, float, float, float, float]",
                        tuple(float(x) for x in cell_to_cellpar(std_lattice)))
    lattice_params = lattice_params_for_family(family, *cellpar_full)

    equiv = np.asarray(ds2.equivalent_atoms, dtype=np.int64)
    wyckoffs = list(ds2.wyckoffs)
    # First (lowest-index) atom of each symmetry orbit is the
    # representative — deterministic given a fixed spglib version/symprec.
    rep_of_orbit: dict[int, int] = {}
    for i, orbit in enumerate(equiv.tolist()):
        rep_of_orbit.setdefault(orbit, i)
    reps = [rep_of_orbit[k] for k in sorted(rep_of_orbit)]

    wyckoff_sites = [
        CifWyckoffSite(
            element=chemical_symbols[int(std_types[r])],
            coords=[float(x) for x in std_positions[r]],
            letter=wyckoffs[r],
        )
        for r in reps
    ]

    # Declared-vs-detected space-group cross-check (traceability rule:
    # never silently accept a CIF's own claim). ASE parses
    # _symmetry_Int_Tables_number into atoms.info["spacegroup"].no when
    # present; a CIF is free to declare P1 (number 1) while listing the
    # full symmetric atom set (as e.g. pymatgen sometimes does) — that is
    # NOT an error, only a WARN, because spglib's detection is always
    # authoritative here.
    declared_sg_number: int | None = None
    spg_info = atoms.info.get("spacegroup")
    if spg_info is not None and hasattr(spg_info, "no"):
        declared_sg_number = int(spg_info.no)
    declared_sg_mismatch = (
        declared_sg_number is not None and declared_sg_number != number
    )
    if declared_sg_mismatch:
        log.warning(
            "%s declares space group %d but spglib detects %d (%s) at "
            "symprec=%g; the spglib-detected group is used.",
            path, declared_sg_number, number, international, symprec,
        )

    spec = CifCrystalSpec(
        source_file=str(path),
        symprec=symprec,
        sg_number=number,
        setting=setting,
        international=international,
        hall_number=hall_number,
        family=family,
        lattice_params=lattice_params,
        cellpar_full=cellpar_full,
        wyckoff_sites=wyckoff_sites,
        n_atoms_conventional=n_atoms_conventional,
        declared_sg_number=declared_sg_number,
        declared_sg_mismatch=declared_sg_mismatch,
    )

    _verify_cif_spec(spec)

    log.info(
        "crystal.cif: %s -> SG %d (%s)%s, family=%s, symprec=%g, "
        "%d atoms/cell, %d Wyckoff sites",
        path, number, international,
        f" setting={setting!r}" if setting else "",
        family, symprec, n_atoms_conventional, len(wyckoff_sites),
    )
    return spec


def _verify_cif_spec(spec: CifCrystalSpec) -> None:
    """Round-trip *spec* through the manual-path machinery right away
    (validate_cellpar -> hall_from_international -> symmetry_ops ->
    expand_wyckoff -> verify_spacegroup) so a setting/tolerance mismatch
    is reported against ``crystal.cif`` at config-resolution time, not
    deep inside the pipeline (fixes CRYSTAL/BASIS/SG defects the same
    guards the manual path already has — no double-placed atoms).
    """
    from grainsmith.crystal.cell import validate_cellpar
    from grainsmith.crystal.spacegroup import (
        WyckoffSite,
        expand_wyckoff,
        hall_from_international,
        symmetry_ops,
    )
    from grainsmith.crystal.verify import verify_spacegroup

    cellpar = validate_cellpar(spec.family, spec.lattice_params)
    hall = hall_from_international(spec.sg_number, spec.setting)
    rots, trans = symmetry_ops(hall)
    sites = [
        WyckoffSite(element=s.element, coords=s.coords, letter=s.letter)
        for s in spec.wyckoff_sites
    ]
    basis = expand_wyckoff(sites, rots, trans)
    if basis.atoms_per_cell != spec.n_atoms_conventional:
        raise CrystalError(
            f"crystal.cif ({spec.source_file}): the Wyckoff-orbit "
            f"expansion produced {basis.atoms_per_cell} atoms/cell but "
            f"the CIF's conventional cell has {spec.n_atoms_conventional} "
            "— this indicates a mismatched space-group setting or "
            "symmetry-detection tolerance (atoms would be double-placed "
            "or missing). Try adjusting crystal.cif.symprec."
        )
    # Gate G2 round-trip, exactly as the manual path does in
    # pipeline._build_crystal — surfaces here with CIF context instead.
    verify_spacegroup(cellpar, basis, spec.sg_number)
