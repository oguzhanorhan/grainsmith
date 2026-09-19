"""Shared certified k-NN GB-shell bound.

Both the additive-weighted backend (``weighted.WeightedTessellation``) and
the fully-periodic flat/Voronoi backend (``flat.FlatTessellation``, weight
identically zero) own a generalized distance ``d(x, s) = |x - s| - w`` that
is 1-Lipschitz in ``x`` -- a plain Euclidean distance shifted by a per-seed
constant.  Flat's zero-weight case is not a separate proof, it is the same
one with ``w == 0`` everywhere, so the certified k-NN construction below is
shared verbatim rather than re-derived per backend.

The replica-aware margin ``m_rep(x) = (second - best) / 2`` -- best/second
the two smallest ``d - w`` values over ALL periodic replica seeds, home =
the replica attaining ``best`` -- certifies the GB shell (see
``Tessellation.gb_shell_lower_bound``'s module-level proof).  Evaluating
that densely over every replica is O(N * n_replicas); this instead runs ONE
cKDTree k-NN query (k=8) against the replica seeds and CERTIFIES that no
UNSEEN replica could beat the runner-up: with ``d_K`` the k-th (largest)
Euclidean distance returned for a point and ``w_max`` the global maximum
replica weight, any unseen replica has generalized distance
``>= d_K - w_max`` (worst case); if that exceeds the runner-up generalized
distance among the k seen replicas, no unseen replica can unseat either the
best or the runner-up, so the seen best/second ARE the true global
best/second and ``m_lb = (second - best) / 2`` equals the exact margin (not
merely a bound).  Points this fails to certify stay in the returned shell
UNCONDITIONALLY (mask True) rather than risk an uncertified exclusion, so
soundness never depends on k or on the certificate firing.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

SHELL_KNN_K = 8
"""k for the certified k-NN bound: measured 99.96% certified, 20.7% shell
fraction at cutoff 2.432 Å on a 7.3M-atom / 140-grain / 3780-replica
system."""

_CHUNK = 50_000


def certified_knn_shell_mask(
    pos: np.ndarray,
    tree: KDTree,
    rep_weights: np.ndarray,
    cutoff: float,
    workers: int,
    k: int = SHELL_KNN_K,
) -> np.ndarray:
    """Certified GB-shell mask via a k-NN query against *tree* (built over
    every periodic replica seed) and *rep_weights* (the matching per-replica
    additive weight, all zero for a plain Euclidean/flat diagram).

    See the module docstring for the certificate's derivation. ``tree`` and
    ``rep_weights`` are supplied by the caller so the (expensive to build)
    tree can be cached on the owning Tessellation instance across calls.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=bool)
    n_rep = tree.n
    k_eff = min(k, n_rep)
    if k_eff < 2:
        # No runner-up replica exists at all (degenerate: a single grain
        # with no periodic image) -- nothing can be certified.
        return np.ones(n, dtype=bool)
    w_max = float(np.max(rep_weights))
    shell = np.empty(n, dtype=bool)
    for start in range(0, n, _CHUNK):
        chunk = pos[start:start + _CHUNK]
        d, idx = tree.query(chunk, k=k_eff, workers=workers)
        if k_eff == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = rep_weights[idx]
        g = d - w
        rows = np.arange(len(chunk))
        best_col = np.argmin(g, axis=1)
        best = g[rows, best_col]
        g_masked = g.copy()
        g_masked[rows, best_col] = np.inf
        second = np.min(g_masked, axis=1)
        d_K = d[:, -1]
        certified = (d_K - w_max) > second
        m_lb = (second - best) / 2.0
        shell[start:start + _CHUNK] = (~certified) | (m_lb <= cutoff)
    return shell
