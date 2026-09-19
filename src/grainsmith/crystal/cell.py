"""Cell matrix from lattice parameters."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from grainsmith.errors import ConfigError


@dataclass(frozen=True)
class CellPar:
    a: float
    b: float
    c: float
    alpha: float  # degrees
    beta: float   # degrees
    gamma: float  # degrees
    # cubic|tetragonal|orthorhombic|hexagonal|trigonal|rhombohedral|
    # monoclinic|triclinic — "rhombohedral" is the primitive R-axes
    # setting of a trigonal (R-centered) space group (setting: "R"),
    # not a distinct Bravais-lattice family; see family_from_sg below.
    family: str


def cell_matrix(a: float, b: float, c: float,
                alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Return the (3,3) cell matrix A whose COLUMNS are the lattice vectors a1, a2, a3 (Å).

    Formula (valid for all 7 crystal families):
        a1 = a * (1, 0, 0)
        a2 = b * (cos γ, sin γ, 0)
        a3 = c * (cos β, (cos α - cos β cos γ)/sin γ, v/sin γ)
        v  = sqrt(1 - cos²α - cos²β - cos²γ + 2 cos α cos β cos γ)

    All angles in degrees. Returns float64 array.
    Raises ConfigError if the parameter combination is unphysical (v² ≤ 0 or sin γ = 0).
    """
    ar = np.radians(alpha)
    br = np.radians(beta)
    gr = np.radians(gamma)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sg = np.sin(gr)
    if abs(sg) < 1e-12:
        raise ConfigError(f"sin(gamma) ≈ 0 (gamma={gamma}°): unphysical cell parameters.")
    v2 = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    if v2 <= 0.0:
        raise ConfigError(
            f"Unphysical cell: v² = {v2:.6g} ≤ 0 "
            f"(a={a}, b={b}, c={c}, α={alpha}°, β={beta}°, γ={gamma}°). "
            "Check lattice parameters for triangle inequality."
        )
    v = np.sqrt(v2)
    A = np.zeros((3, 3), dtype=np.float64)
    A[0, 0] = a
    A[0, 1] = b * cg
    A[1, 1] = b * sg
    A[0, 2] = c * cb
    A[1, 2] = c * (ca - cb * cg) / sg
    A[2, 2] = c * v / sg
    return A


# Family validation rules: maps family → {free params, fixed params}
# "trigonal" uses the HEXAGONAL axis setting (the spglib default and the
# standard for most R-groups); "rhombohedral" is the R (primitive
# rhombohedral) axis setting selected via space_group.setting: "R" —
# one edge length a and one inter-axial angle α shared by all three pairs.
_FAMILY_RULES: dict[str, dict] = {
    "cubic":        {"free": ["a"],               "fixed": {"b": "a", "c": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "tetragonal":   {"free": ["a", "c"],           "fixed": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "orthorhombic": {"free": ["a", "b", "c"],      "fixed": {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "hexagonal":    {"free": ["a", "c"],           "fixed": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0}},
    "trigonal":     {"free": ["a", "c"],           "fixed": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0}},  # hexagonal setting
    "rhombohedral": {"free": ["a", "alpha"],       "fixed": {"b": "a", "c": "a", "beta": "alpha", "gamma": "alpha"}},  # R setting
    "monoclinic":   {"free": ["a", "b", "c", "beta"], "fixed": {"alpha": 90.0, "gamma": 90.0}},
    "triclinic":    {"free": ["a", "b", "c", "alpha", "beta", "gamma"], "fixed": {}},
}


def family_from_sg(number: int, setting: str | None = None) -> str:
    """Map an ITA space-group number (1-230) + optional setting to a
    grainsmith crystal-family name (§6.1).

    Mirrors the family boundaries used throughout the crystal machinery
    (pipeline._build_crystal, the manual crystal.space_group/lattice path,
    and the CIF loader — crystal/cif.py — share this single definition so
    the two input paths can never silently disagree on which family a
    space group belongs to).  The trigonal -> rhombohedral override
    triggers only for ``setting == "R"`` (primitive rhombohedral axes);
    the hexagonal-axis setting ("H", or unset) keeps the trigonal family.
    """
    if 1 <= number <= 2:
        family = "triclinic"
    elif 3 <= number <= 15:
        family = "monoclinic"
    elif 16 <= number <= 74:
        family = "orthorhombic"
    elif 75 <= number <= 142:
        family = "tetragonal"
    elif 143 <= number <= 167:
        family = "trigonal"
    elif 168 <= number <= 194:
        family = "hexagonal"
    elif 195 <= number <= 230:
        family = "cubic"
    else:
        raise ConfigError(
            f"Invalid space-group number: {number}. Must be 1-230."
        )
    if family == "trigonal" and setting is not None and setting.upper() == "R":
        family = "rhombohedral"
    return family


def lattice_params_for_family(
    family: str, a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> dict[str, float]:
    """Reduce a full (a, b, c, alpha, beta, gamma) tuple to the FREE
    parameters of *family* — the form :func:`validate_cellpar` expects.

    Used by the CIF loader (crystal/cif.py) to convert spglib's
    standardized-cell parameters into the same ``params`` dict the manual
    ``crystal.lattice`` path supplies, so both paths flow through the
    identical validation and cell-matrix construction.
    """
    family = family.lower()
    if family not in _FAMILY_RULES:
        raise ConfigError(
            f"Unknown crystal family '{family}'. Choose from: {list(_FAMILY_RULES)}"
        )
    full = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    return {k: full[k] for k in _FAMILY_RULES[family]["free"]}


def reduce_triclinic_tilts(H: np.ndarray) -> np.ndarray:
    """Reduce a restricted-triclinic box matrix to the LAMMPS canonical
    tilt range (box.cells).

    ``H`` has COLUMNS a, b, c (same convention as :func:`cell_matrix`)
    with the restricted-triclinic structure already in place — ``a``
    along +x, ``b`` in the xy-plane with positive y, ``c`` with positive
    z (``H[1,0] = H[2,0] = H[2,1] = 0``).  LAMMPS additionally requires
    the tilt factors ``xy = H[0,1]``, ``xz = H[0,2]``, ``yz = H[1,2]``
    to each satisfy ``|tilt| <= half of the parallel box length``
    (Howto_triclinic): ``|xy|, |xz| <= H[0,0]/2`` and ``|yz| <=
    H[1,1]/2``.

    A box built as ``H = A @ diag(n1, n2, n3)`` (integer lattice
    multiples) generally violates this bound once the off-diagonal
    lattice-parameter contributions (O(a), fixed) are compared against
    a small ``n_i`` diagonal (O(n_i * a)).  The bound is restored by an
    integer GL(3,Z) change of box-vector basis — subtracting an integer
    multiple of one box vector from another — which leaves the
    represented sublattice (and hence every physical quantity: volume,
    atom count, commensurability) EXACTLY unchanged, only re-expressing
    the same periodic cell with a differently-shaped fundamental domain.

    Reduction order (each step leaves the previous step's bound intact,
    since the subtracted vector's zero components do not touch the
    already-reduced tilt):

    1. ``c -= round(yz/by) * b``   (reduces yz; also shifts xz)
    2. ``c -= round(xz/ax) * a``   (reduces the now-updated xz only)
    3. ``b -= round(xy/ax) * a``   (reduces xy only)

    Returns the reduced ``H`` (still restricted-triclinic: same
    diagonal, same volume, same represented lattice).
    """
    H = np.array(H, dtype=np.float64, copy=True)
    ax, by = H[0, 0], H[1, 1]
    k = round(float(H[1, 2] / by))
    H[:, 2] -= k * H[:, 1]
    k = round(float(H[0, 2] / ax))
    H[:, 2] -= k * H[:, 0]
    k = round(float(H[0, 1] / ax))
    H[:, 1] -= k * H[:, 0]
    return H


def validate_cellpar(family: str, params: dict[str, Any]) -> CellPar:
    """
    Validate lattice parameters for a given crystal family and return a CellPar.
    Raises ConfigError if wrong parameters are specified (over- or under-specified).
    'params' should contain only the FREE parameters of the family (§6.1).
    """
    family = family.lower()
    if family not in _FAMILY_RULES:
        raise ConfigError(f"Unknown crystal family '{family}'. Choose from: {list(_FAMILY_RULES)}")
    rules = _FAMILY_RULES[family]
    free = rules["free"]
    fixed = rules["fixed"]
    # Check no extra keys beyond the free set
    extra = set(params) - set(free)
    if extra:
        raise ConfigError(
            f"Crystal family '{family}' has free parameters {free}. "
            f"Unexpected parameters provided: {sorted(extra)}. Remove them."
        )
    missing = set(free) - set(params)
    if missing:
        raise ConfigError(
            f"Crystal family '{family}' requires free parameters {free}. "
            f"Missing: {sorted(missing)}."
        )
    # Build full parameter set
    full: dict[str, Any] = dict(params)
    for k, v in fixed.items():
        if isinstance(v, str):  # e.g. b = "a" means b = a
            full[k] = full[v]
        else:
            full[k] = v
    return CellPar(
        a=float(full["a"]), b=float(full["b"]), c=float(full["c"]),
        alpha=float(full["alpha"]), beta=float(full["beta"]), gamma=float(full["gamma"]),
        family=family,
    )
