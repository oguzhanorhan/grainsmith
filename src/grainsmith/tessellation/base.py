"""Tessellation abstract base class (§4.2, §5)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from grainsmith.constants import MEMORY_HARD_LIMIT_BYTES


class Tessellation(ABC):
    """Abstract base for all tessellation backends.

    The entire atom-fill, overlap-removal, and analysis stack depends only
    on the methods defined here — backend-agnostic by design (§3.3, §4.1).
    """

    memory_limit_bytes: float = MEMORY_HARD_LIMIT_BYTES
    """§13 per-allocation memory budget (bytes) for THIS run, consumed by the
    guard checks in atoms/fill.py, tessellation/voxel.py and
    tessellation/voxel_import.py.

    This is a CLASS attribute default (== the constants.py DEFAULT) so every
    direct construction — tests, analysis/grains.py — keeps working
    unchanged. ``pipeline.run()`` overwrites it with a per-instance,
    RAM-clamped value (``memory.resolve_memory_budget``) once a run's
    config is known -- for the four CURVED backends (WeightedTessellation,
    AnisotropicTessellation, WarpTessellation,
    PerturbedDistanceTessellation) this happens as CONSTRUCTOR keyword
    arguments from ``pipeline._stage_tessellation`` (their own guarded
    voxel-grid allocation runs inside ``__init__``, before the pipeline
    gets another chance at the instance); for every other backend
    ``pipeline._apply_runtime_limits`` stamps it as a post-construction
    safety net.

    Why an INSTANCE attribute and not a module global in constants.py:
    ``fill_grain`` runs inside a ``ProcessPoolExecutor`` (atoms/fill.py's
    ``fill_grains``, ``jobs > 1``). Under the ``spawn`` start method
    (Windows, macOS) a worker process re-imports grainsmith modules from
    scratch and would read the UNPATCHED module-level default, silently
    ignoring any run-specific override; an instance attribute, by contrast,
    lives in the tessellation object's ``__dict__`` and travels with it
    through ``pickle`` when the object is sent to the worker. This repo has
    already hit exactly this class of bug once — see the fork-only skipif
    in tests/test_curvature.py (a monkeypatched MODULE global that a spawned
    worker cannot see).
    """

    memory_limit_source: str = "config"
    """Provenance tag for :attr:`memory_limit_bytes`: ``"config"`` when the
    user's own ``runtime.memory_limit_gb`` was the binding value (RAM had
    headroom, or was undetectable), ``"ram"`` when detected physical RAM was
    the smaller, binding bound (see ``memory.resolve_memory_budget``, whose
    return this attribute mirrors). It selects the advice branch in
    ``memory.build_memory_guard_message`` -- "raise runtime.memory_limit_gb"
    is actionable advice only in the ``"config"`` case; in the ``"ram"``
    case the message instead says the machine itself lacks the RAM, since
    raising a config value the RAM clamp would just re-clamp is not
    actionable.

    Same CLASS-attribute-default / spawn-safety rationale as
    :attr:`memory_limit_bytes` above: it must travel with the tessellation
    instance through ``pickle`` to a ``ProcessPoolExecutor`` 'spawn' worker,
    so it cannot live as a module global either.
    """

    @abstractmethod
    def grain_of(self, X: np.ndarray) -> np.ndarray:
        """Return grain index (0-based) for each point in X.

        Parameters
        ----------
        X : (N, 3) float64
            Points in lab/box coordinates.

        Returns
        -------
        (N,) int32
        """

    @abstractmethod
    def owns(self, X: np.ndarray, i: int) -> np.ndarray:
        """Compact-cell ownership test used by the atom fill (§6.8).

        Returns True for points that belong to the *home* (identity-image)
        cell of grain i, evaluated at UNWRAPPED lab coordinates.  Unlike
        :meth:`grain_of` — which folds periodic images of a grain back onto
        its home id — this method distinguishes the home cell from its
        periodic/mirror images, so the home cells of all grains tile the
        simulation domain exactly once.  Filling with ``owns`` therefore
        generates each torus point exactly once, for any (incommensurate)
        lattice orientation, with no wrap-and-deduplicate step.

        Ties on cell boundaries are broken deterministically (lowest replica
        index), so a boundary point is owned by exactly one image.

        Parameters
        ----------
        X : (N, 3) float64
            Points in UNWRAPPED lab coordinates.
        i : int
            Grain index (0-based).

        Returns
        -------
        (N,) bool
        """

    @abstractmethod
    def margin(self, X: np.ndarray, i: int) -> np.ndarray:
        """Signed distance to the boundary of grain i (positive = inside).

        Parameters
        ----------
        X : (N, 3) float64
        i : int
            Grain index (0-based).

        Returns
        -------
        (N,) float64, in Å.
        """

    @abstractmethod
    def adjacency(self) -> list[tuple[int, int]]:
        """Return list of (i, j) adjacent grain pairs, i < j."""

    @abstractmethod
    def bounding_radius(self, i: int) -> float:
        """Guaranteed bounding sphere radius (Å) centered at seeds[i].

        Fill algorithm uses this to bound the lattice enumeration (§6.8).
        """

    @property
    @abstractmethod
    def seeds(self) -> np.ndarray:
        """(N, 3) float64 seed positions."""

    @property
    @abstractmethod
    def n_grains(self) -> int:
        """Number of grains."""

    def cell_vertices_rel(self, i: int) -> np.ndarray | None:
        """Cell vertices of grain i, RELATIVE to seeds[i], in lab/Cartesian
        coordinates — or None if this backend has no explicit polyhedral cell
        (curved/warp/voxel backends).  The fill uses these for a tight oriented
        bounding box; None makes it fall back to the bounding_radius sphere."""
        return None

    def gb_shell_lower_bound(
        self, pos: np.ndarray, grain: np.ndarray, cutoff: float,
        workers: int = 1,
    ) -> np.ndarray | None:
        """Certified GB-shell pre-filter mask for overlap pair discovery
        (see ``atoms.overlap.remove_overlaps``).

        A pair ``(x, y)`` with ``|x - y| <= cutoff`` must have different
        OWNER REPLICAS (same-replica atoms of a perfect lattice are
        ``>= d_nn > cutoff`` apart), and for any generalized "distance"
        that is 1-Lipschitz in ``x`` the replica-aware margin
        ``m_rep(x) = (second - best) / 2`` (best/second the two smallest
        generalized distances to the ``B*N`` periodic replica seeds, home
        = the replica attaining ``best``) satisfies ``m_rep(x) <= |x - y|``.
        So ``shell := {x : m_rep(x) <= cutoff}`` contains every pair
        endpoint, and the pair set found by searching only the shell
        equals the full-population pair set.

        A backend overrides this ONLY where that Lipschitz-1 property
        holds exactly (or the bound can be computed and certified cheaply
        for a Lipschitz-1 generalized distance) -- see the per-backend
        docstrings for which ones do and why the others do not.  This base
        implementation always returns ``None`` ("no certified bound; the
        caller must use the full atom population").

        Parameters
        ----------
        pos : (N, 3) float64
            Candidate atom positions -- SAME coordinate convention
            (wrapped/canonical) that will be handed to the pair-discovery
            neighbor search.
        grain : (N,) int
            Each atom's OWN (owning) grain id.  Not every backend's bound
            needs this (see per-backend docstrings) -- home ownership for
            the margin itself is always determined by the argmin replica,
            never by this array.
        cutoff : float
            The overlap cutoff, in Å.
        workers : int, default 1
            Worker/thread count for any internal tree query this backend's
            bound performs (mirrors the ``jobs``/``workers`` plumbing in
            ``atoms.overlap``).  Must affect only how fast the mask is
            computed, never its value (scipy ``cKDTree.query`` computes
            each query row independently, so its result is worker-count
            invariant).

        Returns
        -------
        (N,) bool, or None.
            ``True`` marks an atom that must stay in the neighbor search;
            ``None`` means no certified bound is available for this
            backend, so the caller falls back to the full atom population.
        """
        return None
