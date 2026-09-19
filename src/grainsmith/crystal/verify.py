"""spglib round-trip verification — gate G2."""
from __future__ import annotations

import logging

import numpy as np
import spglib

from grainsmith.constants import VERIFY_SYMPREC
from grainsmith.crystal.cell import CellPar, cell_matrix
from grainsmith.crystal.spacegroup import Basis
from grainsmith.errors import CrystalError

log = logging.getLogger(__name__)


def verify_spacegroup(
    cellpar: CellPar,
    basis: Basis,
    requested_number: int,
    symprec: float = VERIFY_SYMPREC,
) -> dict:
    """
    Build the conventional cell and pass to spglib.get_symmetry_dataset().
    Require detected international number == requested_number.
    Returns the dataset dict for logging (Wyckoff letters, site symmetries).
    Raises CrystalError if verification fails.

    IMPORTANT: Uses ATTRIBUTE access on the spglib dataset (not dict-style — deprecated in ≥2.x).
    """
    A = cell_matrix(cellpar.a, cellpar.b, cellpar.c,
                    cellpar.alpha, cellpar.beta, cellpar.gamma)
    # spglib wants lattice as (3,3) with ROW vectors = transposed A
    lattice = A.T.tolist()
    positions = basis.frac.tolist()
    # Map species to integers (spglib needs atomic numbers or arbitrary ints per species)
    unique_species = sorted(set(basis.species))
    species_map = {s: i + 1 for i, s in enumerate(unique_species)}
    numbers = [species_map[s] for s in basis.species]

    cell = (lattice, positions, numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        raise CrystalError(
            f"spglib.get_symmetry_dataset returned None for SG {requested_number}. "
            "Check lattice parameters and Wyckoff site coordinates."
        )

    # Attribute-style access (spglib ≥ 2.x)
    try:
        detected_number = dataset.number
        hall_number_detected = dataset.hall_number
        international = dataset.international
        wyckoff_letters = dataset.wyckoffs
        site_symmetry_symbols = dataset.site_symmetry_symbols
        pointgroup = dataset.pointgroup
    except AttributeError:
        # Fallback to dict-style if old spglib
        detected_number = dataset["number"]
        hall_number_detected = dataset["hall_number"]
        international = dataset["international"]
        wyckoff_letters = list(dataset["wyckoffs"])
        site_symmetry_symbols = list(dataset["site_symmetry_symbols"])
        pointgroup = dataset["pointgroup"]

    if detected_number != requested_number:
        raise CrystalError(
            f"Space-group verification failed: requested SG {requested_number}, "
            f"but spglib detected SG {detected_number} ({international}). "
            "Check lattice parameters and Wyckoff site coordinates."
        )

    return {
        "number": detected_number,
        "international": international,
        "hall_number": hall_number_detected,
        "pointgroup": pointgroup,
        "wyckoffs": list(wyckoff_letters),
        "site_symmetry_symbols": list(site_symmetry_symbols),
    }


def verify_wyckoff_letters(
    sites: list[tuple[list[float], str | None]],
    basis: Basis,
    wyckoff_letters: list[str],
    tol: float = 1e-5,
    strict: bool = True,
) -> None:
    """Verify user-supplied Wyckoff letters against spglib's assignment
    (gate G2 extension, §7: "letter optional → verified").

    Parameters
    ----------
    sites : list of (coords, letter)
        Representative fractional coordinates and the optional letter from
        the config (e.g. "a" or "4c" — a leading multiplicity is ignored).
    basis : Basis
        Expanded orbit; row order matches *wyckoff_letters*.
    wyckoff_letters : list of str
        Per-atom letters from the spglib dataset of the SAME cell.
    tol : float
        Fractional min-image tolerance for locating the representative
        atom in the expanded basis (FRAC_DEDUPE_TOL scale).
    strict : bool
        spglib assigns letters in ITS OWN standardized description of the
        group.  For the default setting these coincide with the ITA
        letters of the user's coordinates; for a non-default
        ``space_group.setting`` (origin choice 2, R axes, ...) they may
        legitimately differ — e.g. Fd-3m origin choice 2 puts 8a at
        (1/8,1/8,1/8) while spglib reports 'b' for that orbit.  Pass
        ``strict=False`` in that case: mismatches log a warning instead
        of raising.

    Raises
    ------
    CrystalError
        If a representative cannot be located (internal error) or the
        letters disagree (``strict=True`` only).
    """
    frac = np.asarray(basis.frac, dtype=np.float64)
    for coords, letter in sites:
        if letter is None:
            continue
        expected = "".join(ch for ch in letter if ch.isalpha()).lower()
        if not expected:
            raise CrystalError(
                f"Invalid Wyckoff letter {letter!r} for site {coords}: "
                "expected a letter such as 'a' or '4c'."
            )
        rep = np.asarray(coords, dtype=np.float64) % 1.0
        d = frac - rep
        d -= np.round(d)
        dist = np.linalg.norm(d, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] > tol:
            raise CrystalError(
                f"Representative site {coords} not found in the expanded "
                f"orbit (min fractional distance {dist[idx]:.2e}) — "
                "internal orbit-expansion error."
            )
        detected = wyckoff_letters[idx].lower()
        if detected != expected:
            msg = (
                f"Wyckoff letter mismatch for site {coords}: config says "
                f"{letter!r} but spglib assigns '{detected}'. Fix the "
                "letter (or drop it — it is optional) or check the "
                "coordinates."
            )
            if strict:
                raise CrystalError(msg)
            log.warning(
                "%s (warning only: a non-default space-group setting is in "
                "use, and spglib letters refer to its standardized "
                "description, which may label this orbit differently.)",
                msg,
            )
