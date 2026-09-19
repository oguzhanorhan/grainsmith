"""Grain → phase assignment for multiphase polycrystals.

The user prescribes per-phase VOLUME fractions (the metallographic
convention); grains are partitioned so the achieved fractions match the
targets as closely as the grain-count granularity allows.  The assignment
is DETERMINISTIC — a greedy partition of the measured pre-fill grain
volumes, no rng stream involved:

    grains in (−volume, index) order; each goes to the phase with the
    largest remaining volume deficit (tie → lowest phase index).

This is the LPT (longest-processing-time) heuristic for balanced
partitioning: each phase's final deviation is bounded by the last (hence
smallest) grain it received, so deviations beyond ONE largest cell signal
a genuinely improvable assignment — that bound is gate G15's threshold
(qa.gate_g15_phase_fractions).  A guard pass ensures every phase receives
at least one grain even for extreme fractions (resolve Rule 23 guarantees
n_grains ≥ n_phases).
"""
from __future__ import annotations

import numpy as np


def assign_phases(
    volumes: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """Deterministic greedy grain → phase partition.

    Parameters
    ----------
    volumes : (N,) float64
        Measured pre-fill grain volumes (Å³): exact cell volumes for
        flat/power backends, voxel volumes for curved/imported ones.
    fractions : (P,) float64
        Target volume fractions (resolve Rule 21: positive, sum to 1).

    Returns
    -------
    (N,) int32 — phase index per grain; every phase appears at least once.
    """
    volumes = np.asarray(volumes, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    n = len(volumes)
    n_phases = len(fractions)
    if n < n_phases:
        raise ValueError(
            f"{n} grains cannot cover {n_phases} phases (resolve Rule 23).")

    deficit = fractions * float(np.sum(volumes))
    # argsort of -volumes is stable → equal volumes keep grain-index order.
    order = np.argsort(-volumes, kind="stable")
    phase_of = np.empty(n, dtype=np.int32)
    empty = set(range(n_phases))
    for k, g in enumerate(order):
        if len(empty) == n - k:
            # Exactly one grain left per empty phase: force-feed them
            # (largest remaining deficit first, tie → lowest index).
            p = max(empty, key=lambda q: (deficit[q], -q))
        else:
            p = int(np.argmax(deficit))   # tie → lowest phase index
        phase_of[g] = p
        deficit[p] -= volumes[g]
        empty.discard(p)
    return phase_of


def achieved_fractions(
    volumes: np.ndarray,
    phase_of: np.ndarray,
    n_phases: int,
) -> np.ndarray:
    """(P,) achieved volume fraction per phase (gate G15 input)."""
    volumes = np.asarray(volumes, dtype=np.float64)
    per_phase = np.bincount(np.asarray(phase_of), weights=volumes,
                            minlength=n_phases)
    return per_phase / float(np.sum(volumes))
