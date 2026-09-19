"""Space-group orbit expansion via spglib (Hall-number lookup, symmetry
operations, Wyckoff-site orbit expansion — §6.2).

Not a legacy-defect fix — the legacy Fortran code had no symmetry-driven
atom placement at all, so this module is new capability rather than a
correction of prior wrong behavior.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import spglib

from grainsmith.errors import ConfigError, CrystalError


@dataclass(frozen=True)
class WyckoffSite:
    element: str | dict[str, float]  # species or occupancy dict
    coords: list[float]              # [x, y, z] fractional, free params already substituted
    letter: str | None = None        # optional Wyckoff letter for verification


@dataclass
class Basis:
    """Expanded basis: fractional positions + species list + optional occupancy."""
    frac: np.ndarray          # (N, 3) float64, fractional
    species: list[str]        # (N,) element symbols (sampled if occupancy)
    occupancy: list[dict[str, float] | None]  # per-site occupancy (None if fixed)
    atoms_per_cell: int       # = N


@functools.lru_cache(maxsize=1)
def _build_hall_map() -> dict[tuple[int, str | None], int]:
    """Scan spglib halls 1..530, map (international_number, choice) -> first matching hall."""
    result: dict[tuple[int, str | None], int] = {}
    for hall in range(1, 531):
        try:
            info = spglib.get_spacegroup_type(hall)
        except Exception:
            continue
        if info is None:
            continue
        # Use attribute access (spglib ≥ 2.x); fall back to dict for older versions
        try:
            num = info.number
            choice = getattr(info, "choice", None) or None
        except AttributeError:
            num = info["number"]
            choice = info.get("choice", None) or None
        key_default = (num, None)
        key_choice = (num, choice) if choice else None
        if key_default not in result:
            result[key_default] = hall
        if key_choice and key_choice not in result:
            result[key_choice] = hall
    return result


def hall_from_international(number: int, setting: str | None = None) -> int:
    """Return the Hall number for an ITA space-group number (1–230) + optional setting."""
    hall_map = _build_hall_map()
    key = (number, setting)
    if key in hall_map:
        return hall_map[key]
    key_default = (number, None)
    if key_default in hall_map:
        return hall_map[key_default]
    raise ConfigError(
        f"Space group number {number} (setting={setting!r}) not found in spglib database. "
        "Check the number (1–230) and setting string."
    )


def symmetry_ops(hall: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotations (Nops,3,3) int, translations (Nops,3) float64) from spglib."""
    dataset = spglib.get_symmetry_from_database(hall)
    if dataset is None:
        raise CrystalError(f"spglib returned None for hall number {hall}.")
    rots = np.asarray(dataset["rotations"], dtype=np.int32)
    trans = np.asarray(dataset["translations"], dtype=np.float64)
    return rots, trans


def _frac_minimage(x: np.ndarray) -> np.ndarray:
    """Bring fractional coords to [0,1)."""
    return x - np.floor(x)


def _dedupe_frac(pts: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Remove duplicate fractional coords using minimum-image distance."""
    if len(pts) == 0:
        return pts
    unique = [pts[0]]
    for p in pts[1:]:
        dp = p - np.array(unique)
        dp -= np.round(dp)
        if np.all(np.linalg.norm(dp, axis=1) > tol):
            unique.append(p)
    return np.array(unique)


def expand_wyckoff(sites: list[WyckoffSite], rots: np.ndarray, trans: np.ndarray) -> Basis:
    """
    Expand Wyckoff sites into the full basis of the conventional cell.
    Deduplicates using min-image tolerance FRAC_DEDUPE_TOL (§6.2).
    """
    from grainsmith.constants import FRAC_DEDUPE_TOL
    all_frac: list[np.ndarray] = []
    all_species: list[str] = []
    all_occ: list[dict[str, float] | None] = []

    for site in sites:
        x0 = np.array(site.coords, dtype=np.float64)
        orbit: list[np.ndarray] = []
        for R, t in zip(rots, trans, strict=True):
            xp = _frac_minimage(R @ x0 + t)
            orbit.append(xp)
        unique_orbit = _dedupe_frac(np.array(orbit), tol=FRAC_DEDUPE_TOL)
        for pos in unique_orbit:
            all_frac.append(pos)
            if isinstance(site.element, dict):
                # occupancy site — choose primary species for symbol (actual sampling at fill time)
                primary = max(site.element, key=site.element.__getitem__)
                all_species.append(primary)
                all_occ.append(dict(site.element))
            else:
                all_species.append(site.element)
                all_occ.append(None)

    frac_arr = np.array(all_frac, dtype=np.float64)
    # Cross-site coincidence guard: each site's orbit is deduped
    # above, so any remaining coincident pair comes from two DIFFERENT sites.
    # Distinct atoms at the same point give zero separation -> d_nn = 0, which
    # silently disables overlap removal while still passing the spglib G2
    # round-trip. Reject it (use a single site with an occupancy dict for
    # genuinely mixed occupancy).
    n_at = len(frac_arr)
    for i in range(n_at):
        for j in range(i + 1, n_at):
            d = frac_arr[i] - frac_arr[j]
            d -= np.round(d)                       # min-image (fractional)
            if float(np.linalg.norm(d)) < FRAC_DEDUPE_TOL:
                raise CrystalError(
                    f"Wyckoff sites produce coincident atoms: {all_species[i]} "
                    f"and {all_species[j]} at fractional {frac_arr[i].tolist()} "
                    f"(separation < {FRAC_DEDUPE_TOL}). Distinct sites must not "
                    "share a position; use one site with an occupancy dict for "
                    "mixed occupancy.")
    return Basis(frac=frac_arr, species=all_species, occupancy=all_occ,
                 atoms_per_cell=len(all_species))
