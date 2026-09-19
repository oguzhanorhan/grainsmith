"""Single-crystal "tessellation" (also supports triclinic lattice-multiple
boxes, ``box.cells``).

``grains.number: 1`` selects this trivial backend: one grain owns the whole
box.  No seeding happens (the seed sits at the box centre, deterministically,
consuming NO rng) and no Voronoi construction runs.

Physics: in a periodic box the single grain meets its own periodic images at
the box faces.  When every periodic box vector is a lattice vector of the
rotated cell ``R·A`` the structure is a perfect crystal; otherwise the box
faces are genuine self-boundary defects.  Overlap removal already treats
same-grain contacts across a periodic self-image as legitimate GB pairs, so
an incommensurate single crystal gets its face atoms cleaned exactly like a
real boundary — and gate G14 (warn) measures the per-axis misfit so the user
knows which structure they got (:func:`commensurability_misfit`).

Two box representations
------------------------
- ``box.lengths`` (orthogonal, default): the box matrix is ``diag(Lx, Ly,
  Lz)``.  Generically incommensurate with a rotated lattice unless the user
  picks lengths and orientation that happen to align.
- ``box.cells`` (triclinic lattice-multiple): the box matrix ``H`` is
  built directly from the crystal's own conventional-cell matrix, ``H = A @
  diag(n1, n2, n3)`` (LAMMPS-tilt-reduced, see
  ``crystal.cell.reduce_triclinic_tilts``), with an identity crystal
  orientation enforced at config time (resolve.py Rule 28).  Every box
  vector is then an EXACT integer combination of the crystal's own lattice
  vectors by construction — commensurability holds exactly (gate G14 == 0),
  not just approximately.

Everything below is expressed through the general (3,3) box matrix ``H``
(columns = box vectors ``a, b, c``); the orthogonal case is simply
``H = diag(box_lengths)`` and every formula reduces algebraically to the
per-axis Cartesian form (fractional coordinates ``frac = H⁻¹ X``
collapse to ``X / L`` when ``H`` is diagonal — up to float64 rounding at
the ulp level, since the implementation multiplies by the matrix inverse
rather than dividing directly; the difference is many orders of magnitude
below every tolerance used here, so it never changes a decision, but it
is not literal bit-identity).

Tiling invariant: ``owns(X, 0)`` is the FRACTIONAL fundamental window
``[−tol, 1 − tol)³`` (with ``tol = SINGLE_WINDOW_TOL``, dimensionless — the
same shift that, for the diagonal case, is exactly the old
``SINGLE_WINDOW_TOL·L`` Cartesian shift, since dividing by ``L`` cancels the
multiplication).  The shift keeps a commensurate atom plane sitting exactly
on the 0≡1 fractional edge inside one lift, not zero or two.  Every torus
point has exactly one lift inside the window, so the fill stage generates
each torus point exactly once — the §6.8 contract, trivially, for ANY box
matrix (diagonal or triclinic).
"""
from __future__ import annotations

import numpy as np

from grainsmith.tessellation.base import Tessellation


class SingleCrystalTessellation(Tessellation):
    """The whole box as one grain (also supports triclinic ``box.cells``).

    Parameters
    ----------
    box_lengths : (3,) float64, Å
        Orthogonal box edge lengths. Ignored (may be a diagonal proxy of
        ``cell_matrix``) when ``cell_matrix`` is given.
    periodic : per-axis periodicity flags
    cell_matrix : (3,3) float64, optional
        General box matrix (columns = box vectors a, b, c) for a
        lattice-multiple triclinic box.  ``None`` (default) selects the
        orthogonal box ``diag(box_lengths)``.
    """

    def __init__(self, box_lengths: np.ndarray, periodic: list[bool],
                 cell_matrix: np.ndarray | None = None):
        self._periodic = list(periodic)
        if cell_matrix is not None:
            self._H = np.asarray(cell_matrix, dtype=np.float64).copy()
            self._triclinic = True
        else:
            L = np.asarray(box_lengths, dtype=np.float64)
            self._H = np.diag(L).astype(np.float64)
            self._triclinic = False
        self._Hinv = np.linalg.inv(self._H)
        # Distance between opposite faces along axis k (perpendicular
        # width of the parallelepiped): 1/‖row_k(H⁻¹)‖ — reduces to L_k
        # exactly when H is diagonal (row k of H⁻¹ is e_k/L_k).
        self._face_height = 1.0 / np.linalg.norm(self._Hinv, axis=1)
        # Deterministic seed at the box centre (sum of half box vectors) —
        # no rng consumption.  Reduces to L/2 exactly for diagonal H.
        self._seeds = (0.5 * self._H @ np.ones(3)).reshape(1, 3)

    @property
    def cell_matrix(self) -> np.ndarray:
        """(3,3) box matrix (columns = box vectors); ``diag(L)`` when the
        box is orthogonal, the lattice-multiple triclinic matrix otherwise.
        """
        return self._H.copy()

    @property
    def is_triclinic(self) -> bool:
        """Whether this box carries off-diagonal tilt (``box.cells``)."""
        return self._triclinic

    # -- Tessellation ABC ---------------------------------------------------

    @property
    def seeds(self) -> np.ndarray:
        return self._seeds

    @property
    def n_grains(self) -> int:
        return 1

    def grain_of(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        return np.zeros(len(X), dtype=np.int32)

    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Per-axis fundamental window in FRACTIONAL coordinates, branched
        on periodicity.

        ``frac = H⁻¹ X``.  Periodic axis: the half-open window
        [−tol, 1 − tol).  The tol shift (SINGLE_WINDOW_TOL, constants.py,
        dimensionless in fractional space) keeps the window length exactly
        1 — still one lift per torus point — while handling the
        commensurate case where whole atom planes sit EXACTLY on the 0 ≡ 1
        fractional edge: a raw [0, 1) test can exclude both float lifts of
        such a plane (one ~−2e−16, one == 1) and drop the plane entirely.

        Free (non-periodic) axis: frac=0 and frac=1 are DISTINCT positions,
        not a single torus point, so the closed window [−tol, 1 + tol]
        keeps BOTH wall atom layers; the one-sided [−tol, 1−tol) dedup
        would silently drop the high (frac=1) wall layer of a commensurate
        slab.

        For a diagonal H this reproduces the Cartesian window (frac =
        X/L, tol_frac = tol_cartesian/L = SINGLE_WINDOW_TOL) up to float64
        rounding: ``frac = X @ Hinv`` uses the matrix inverse (a
        reciprocal multiplication), which is not bit-identical to a direct
        division ``X/L`` for every input — the discrepancy is at the ulp
        level, many orders of magnitude below SINGLE_WINDOW_TOL, so it
        cannot flip a window-membership decision in practice, but it is
        not a literal bit-for-bit equivalence.
        """
        from grainsmith.constants import SINGLE_WINDOW_TOL

        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        frac = (self._Hinv @ X.T).T
        tol = SINGLE_WINDOW_TOL
        keep = np.ones(len(X), dtype=bool)
        for ax in range(3):
            if self._periodic[ax]:
                keep &= (frac[:, ax] >= -tol) & (frac[:, ax] < 1.0 - tol)
            else:
                keep &= (frac[:, ax] >= -tol) & (frac[:, ax] <= 1.0 + tol)
        return keep

    def _corner_radius(self) -> float:
        """Covering-sphere radius from the central seed: the parallelepiped
        window's 8 corners relative to the seed are ``(±a ±b ±c)/2`` over
        all sign combinations — for a diagonal H these all have equal norm
        (the familiar half box diagonal), but for a tilted H they do NOT,
        so the true bounding radius is the MAX over all 8 combinations, not
        just ``(a+b+c)/2``."""
        signs = np.array([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0)
                          for k in (-1.0, 1.0)])
        corners = 0.5 * (signs @ self._H.T)
        return float(np.max(np.linalg.norm(corners, axis=1)))

    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed distance to the self-image boundary at the box faces.

        Per periodic axis k, the perpendicular Cartesian distance to the
        nearest of the two faces normal to the k-th reciprocal direction is
        ``min(frac_k, 1 − frac_k) · height_k`` (``height_k`` = distance
        between opposite faces along that reciprocal direction) — negative
        outside the window (frac_k < 0 or > 1), by construction of the
        min().  The overall margin is the min over periodic axes.  With no
        periodic axis there is no boundary at all: the margin is the
        covering-sphere radius from :meth:`_corner_radius` (finite
        stand-in for +inf, safely larger than any overlap cutoff — this is
        the same max-over-8-corners radius :meth:`bounding_radius` returns,
        not merely the half box diagonal, since the two coincide only for
        a diagonal H).

        For a diagonal H, height_k == L_k and frac_k == X_k/L_k up to the
        usual float64 division-vs-reciprocal-multiplication rounding (both
        well under SINGLE_WINDOW_TOL / the overlap cutoff), so this
        reduces to the formula ``min(X_k, L_k − X_k)`` for all
        practical purposes.
        """
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        frac = (self._Hinv @ X.T).T
        out = np.full(len(X), self._corner_radius(), dtype=np.float64)
        for ax in range(3):
            if self._periodic[ax]:
                d = np.minimum(frac[:, ax], 1.0 - frac[:, ax]) * self._face_height[ax]
                out = np.minimum(out, d)
        return out

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> None:
        """No certified GB-shell bound: on any axis with NO periodicity,
        ``margin()`` above returns ``_corner_radius()`` -- a constant,
        position-independent finite stand-in for +inf, not the (also
        infinite/undefined, since there is no boundary at all) true
        distance -- so the exact-interior-distance argument used for
        ``FlatTessellation`` does not carry over without a separate proof.
        A single grain also has no OTHER-grain boundary to overlap
        against, so this bound would only ever matter for a periodic
        self-image GB, a case not separately re-derived here. Always
        returns ``None`` (unchanged full-N overlap behaviour)."""
        return None

    def adjacency(self) -> list[tuple[int, int]]:
        return []

    def bounding_radius(self, i: int) -> float:
        """Covering-sphere radius from the central seed: max over the 8
        parallelepiped-corner sign combinations ``‖(±a ±b ±c)/2‖``
        (reduces to the familiar half Cartesian box diagonal when H is
        diagonal, since all 8 combinations then have equal norm)."""
        return self._corner_radius()

    # -- extras used by the pipeline -----------------------------------------

    def total_volume(self) -> float:
        return float(abs(np.linalg.det(self._H)))


def commensurability_misfit(
    A_rot: np.ndarray,
    box_lengths: np.ndarray,
    periodic: list[bool],
    box_matrix: np.ndarray | None = None,
) -> dict[int, tuple[float, float]]:
    """Per-periodic-axis box/lattice misfit of a single crystal (gate G14).

    Axis i is commensurate when the i-th box vector (``L_i e_i`` for an
    orthogonal box, or ``box_matrix[:, i]`` for a general triclinic
    box) is a lattice vector of the rotated cell ``A_rot = R(q)·A``: the
    fractional solution ``m = A_rot⁻¹ v_i`` must be integer.  The misfit is
    the residual displacement ``δ_i = ‖A_rot (m − round m)‖`` (Å) — the
    dislocation-free strain needed to close the gap is ``ε_i = δ_i / ‖v_i‖``.

    Parameters
    ----------
    box_matrix : (3,3), optional
        General box matrix (columns = box vectors); ``None`` (default)
        uses the orthogonal box ``diag(box_lengths)`` — the same
        construction used for boxes without triclinic tilt.  For a
        ``box.cells`` triclinic box, the caller passes
        the actual (lattice-multiple) box matrix, so a ``box.cells`` build
        with the required identity orientation measures a misfit of 0 to
        float64 roundoff (H is built from A itself, so the fractional
        solution m is integer algebraically; only floating-point rounding
        keeps the measured residual from being a literal exact zero).

    Returns ``{axis: (delta_A, strain)}`` for periodic axes only (free
    axes have no images, hence no misfit).
    """
    A_rot = np.asarray(A_rot, dtype=np.float64)
    if box_matrix is not None:
        H = np.asarray(box_matrix, dtype=np.float64)
    else:
        H = np.diag(np.asarray(box_lengths, dtype=np.float64))
    Ainv = np.linalg.inv(A_rot)
    out: dict[int, tuple[float, float]] = {}
    for ax in range(3):
        if not periodic[ax]:
            continue
        vec = H[:, ax]
        vec_norm = float(np.linalg.norm(vec))
        m = Ainv @ vec
        delta = float(np.linalg.norm(A_rot @ (m - np.round(m))))
        out[ax] = (delta, delta / vec_norm)
    return out
