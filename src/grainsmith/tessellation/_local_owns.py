"""Shared local-set ownership kernel for Flat / Power tessellations.

``owns(X, i)`` decides, for each query point, whether grain *i*'s HOME replica
is the nearest competitor — Euclidean in 3D for the flat backend, in the 4D
power lifting for the power backend — with the lowest-global-replica-index
tie-break.  Once the query and competitor points are expressed in the right
dimension the hot inner computation is identical for both backends: the
distance to each of the ~deg local competitors, the nearest distance, and the
lowest global index among the within-tolerance ties.

This module provides one dimension-agnostic driver, :func:`local_owns`, with:

* a pure-NumPy vectorised path (chunked, the proven reference), and
* an OPTIONAL Numba kernel that fuses the two reductions and avoids the
  ``(chunk × deg × dim)`` temporaries.

Numba is an optional dependency.  The kernel mirrors the NumPy arithmetic
exactly — same per-coordinate accumulation order, same ``sqrt``, same
``<= d1 + tol`` tie test, same lowest-index reduction, and **no** ``fastmath``
(IEEE-compliant) — so its boolean mask is bit-identical to the NumPy path and
to the ``_owns_global`` reference.  ``_USE_NUMBA`` selects the path and is
monkeypatched in the equality tests to prove the two agree on every scenario.
If Numba is absent the driver silently uses NumPy.
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:                       # pragma: no cover - env without numba
    _HAVE_NUMBA = False

# Default: use the Numba kernel when it is importable.  The equality tests flip
# this to compare the two backends; nothing in the engine mutates it.
_USE_NUMBA: bool = _HAVE_NUMBA


if _HAVE_NUMBA:
    import math

    @numba.njit(cache=True)
    def _owns_kernel(Q, P, nbr, home_idx, tol, large, is_mir):  # pragma: no cover
        """Per row: nearest of P, then lowest ``nbr`` index among within-tol ties.

        Mirrors the NumPy branch elementwise (same accumulation order over the
        ``dim`` coordinates, same ``sqrt``) so the mask is bit-identical.  Wall
        MIRROR replicas (``is_mir[k]``) still contribute to the nearest distance
        (they clip the home cell at the box wall) but are barred from winning the
        ownership tie-break — a mirror is a fictitious clipping image that no real
        grain claims, so it must never beat the home replica on a wall bisector.
        """
        rows = Q.shape[0]
        deg = P.shape[0]
        dim = Q.shape[1]
        out = np.empty(rows, dtype=np.bool_)
        for r in range(rows):
            d1 = np.inf
            for k in range(deg):
                s = 0.0
                for c in range(dim):
                    diff = Q[r, c] - P[k, c]
                    s += diff * diff
                d = math.sqrt(s)
                if d < d1:
                    d1 = d
            thr = d1 + tol
            min_idx = large
            for k in range(deg):
                s = 0.0
                for c in range(dim):
                    diff = Q[r, c] - P[k, c]
                    s += diff * diff
                d = math.sqrt(s)
                if d <= thr and not is_mir[k] and nbr[k] < min_idx:
                    min_idx = nbr[k]
            out[r] = min_idx == home_idx
        return out
else:                                      # pragma: no cover - env without numba
    _owns_kernel = None


def local_owns(Q, P, nbr, home_idx, tol, n_replicas, chunk, use_numba=None,
               is_mirror=None):
    """Boolean mask: does grain *i*'s home replica own each query point?

    Parameters
    ----------
    Q : (rows, dim) query points (3D for Flat; 4D lifted for Power).
    P : (deg, dim) local competitor points (home + face-neighbour replicas).
    nbr : (deg,) int64 global replica indices matching the rows of ``P``.
    home_idx : global replica index of grain *i*'s home.
    tol : tie tolerance (``OWNS_TIE_TOL · max L``).
    n_replicas : total replica count; the masking sentinel is ``n_replicas + 1``.
    chunk : NumPy-path row block size (bounds the ``(chunk × deg)`` temporary).
    use_numba : override the module default (``None`` → :data:`_USE_NUMBA`).
    is_mirror : (deg,) bool flags marking wall-MIRROR competitors, aligned with
        ``nbr``.  Mirrors still set the nearest distance (they clip the home cell
        at the box wall) but are excluded from the ownership tie-break so a
        fictitious clipping image never beats the home replica on a wall
        bisector.  ``None`` → no mirrors (all competitors eligible).
    """
    Q = np.asarray(Q, dtype=np.float64)
    rows = Q.shape[0]
    if rows == 0:
        return np.zeros(0, dtype=bool)
    large = int(n_replicas) + 1
    if use_numba is None:
        use_numba = _USE_NUMBA
    deg = P.shape[0]
    if is_mirror is None:
        is_mir = np.zeros(deg, dtype=np.bool_)
    else:
        is_mir = np.ascontiguousarray(is_mirror, dtype=np.bool_)

    if use_numba and _owns_kernel is not None:
        mask = _owns_kernel(
            np.ascontiguousarray(Q, dtype=np.float64),
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(nbr, dtype=np.int64),
            np.int64(home_idx), np.float64(tol), np.int64(large), is_mir)
        return np.asarray(mask, dtype=bool)

    eligible = ~is_mir                                           # (deg,)
    out = np.empty(rows, dtype=bool)
    for start in range(0, rows, chunk):
        Qc = Q[start:start + chunk]                              # (C, dim)
        D = np.linalg.norm(Qc[:, None, :] - P[None, :, :], axis=2)  # (C, deg)
        d1 = D.min(axis=1)
        within = (D <= d1[:, None] + tol) & eligible[None, :]
        nbr_bc = np.where(within, nbr[None, :], large)
        out[start:start + chunk] = nbr_bc.min(axis=1) == home_idx
    return out
