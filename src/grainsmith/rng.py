"""RNG bundle: deterministic child Generator streams derived from one seed.

Each pipeline stage receives its own independent PCG64 stream produced by
``numpy.random.SeedSequence.spawn()``.  The spawn hierarchy guarantees
that streams from different stages are statistically uncorrelated even when
the master seed is small or pathological.

IMPORTANT: ``np.random.*`` module-level calls are strictly forbidden in
this codebase (guardrail §15).  All randomness goes through an
``RNGBundle`` obtained via ``make_rng(seed)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stage names in spawn order — changing this order changes child streams,
# breaking reproducibility.  Add new stages at the end only.
# "sizes" (target-volume sampling) and "mdf" (orientation-assignment
# annealing) were appended after the original four stages, which stayed
# unchanged so earlier configs reproduce their earlier outputs exactly.
# "doping" (per-grain dopant-insertion sampling) was appended last on the
# same basis: the first six children are unchanged by its addition.
STAGE_NAMES: tuple[str, ...] = (
    "seeding", "orientation", "fields", "occupancy", "sizes", "mdf",
    "doping",
)


@dataclass
class RNGBundle:
    """A bundle of named Generator streams derived from a single master seed.

    Attributes
    ----------
    seed : int
        The master seed used to create this bundle (recorded in outputs).
    seeding : np.random.Generator
        PCG64 stream for grain-seed placement (RSA + Lloyd centroidal).
    orientation : np.random.Generator
        PCG64 stream for grain-orientation assignment (all schemes).
    fields : np.random.Generator
        PCG64 stream for GRF synthesis (domain-warp curved boundaries).
    occupancy : np.random.Generator
        PCG64 stream for stochastic site-occupancy sampling (solid solutions).
    sizes : np.random.Generator
        PCG64 stream for log-normal grain-size targeting.
    mdf : np.random.Generator
        PCG64 stream for MDF (misorientation distribution function)
        annealing (orientation-assignment refinement).
    occupancy_ss : np.random.SeedSequence
        Parent SeedSequence for per-grain occupancy streams (see
        ``occupancy_streams``).
    doping : np.random.Generator
        PCG64 stream for the dopant-insertion stage.
    doping_ss : np.random.SeedSequence
        Parent SeedSequence for per-grain doping streams (see
        ``doping_streams``).
    """

    seed: int
    seeding: np.random.Generator
    orientation: np.random.Generator
    fields: np.random.Generator
    occupancy: np.random.Generator
    sizes: np.random.Generator = None      # type: ignore[assignment]
    mdf: np.random.Generator = None        # type: ignore[assignment]
    occupancy_ss: np.random.SeedSequence = None  # type: ignore[assignment]
    doping: np.random.Generator = None       # type: ignore[assignment]
    doping_ss: np.random.SeedSequence = None  # type: ignore[assignment]

    def occupancy_streams(self, n: int) -> list[np.random.Generator]:
        """Per-grain independent occupancy streams (parallel fill, §6.8/§13).

        Grain *i* gets the stream derived from the occupancy child
        SeedSequence by extending its spawn key with ``(i,)`` — a STATELESS
        construction (``SeedSequence.spawn()`` itself counts children, so
        repeated calls would not be reproducible).  The streams therefore
        depend only on (master seed, grain id): identical for any
        ``--jobs`` value, any execution order, and any number of calls.
        """
        return [
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                entropy=self.occupancy_ss.entropy,
                spawn_key=self.occupancy_ss.spawn_key + (i,),
            )))
            for i in range(n)
        ]

    def doping_streams(self, n: int) -> list[np.random.Generator]:
        """One deterministic per-grain stream for the doping stage
        (same stateless spawn_key-extension derivation as
        occupancy_streams; independent of every other stage family)."""
        return [
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                entropy=self.doping_ss.entropy,
                spawn_key=self.doping_ss.spawn_key + (i,),
            )))
            for i in range(n)
        ]


def make_rng(seed: int) -> RNGBundle:
    """Create deterministic child RNG streams from a single seed.

    Uses ``numpy.random.SeedSequence.spawn()`` to derive one independent
    PCG64 stream per pipeline stage.  Given the same *seed*, this function
    always returns byte-identical generators, satisfying the determinism
    requirement (§1).

    Parameters
    ----------
    seed : int
        Master seed — typically ``config.seed.value`` for ``mode: fixed``,
        or a harvested entropy integer for ``mode: entropy``.

    Returns
    -------
    RNGBundle
        Named ``numpy.random.Generator`` instances for each pipeline stage.

    Examples
    --------
    >>> rng = make_rng(42)
    >>> rng.seed
    42
    >>> rng.orientation.random()  # reproducible
    0.4674907799518424
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(STAGE_NAMES))
    return RNGBundle(
        seed=seed,
        occupancy_ss=children[STAGE_NAMES.index("occupancy")],
        doping_ss=children[STAGE_NAMES.index("doping")],
        **{
            name: np.random.Generator(np.random.PCG64(child))
            for name, child in zip(STAGE_NAMES, children, strict=True)
        },
    )
