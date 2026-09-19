"""Mandatory validation for the numba short-circuit owns() kernel
(tessellation/weighted.py's "Numba acceleration" docstring section).

Two tiers:

1. Fast, always-run unit batteries — random point clouds, near-tie points,
   the exact tie rule, and empty input — for BOTH weighted backends
   (AnisotropicTessellation / WeightedTessellation), all three periodicity
   geometries. These duplicate (deliberately — belt-and-suspenders) the
   equivalent cases already added to tests/test_weighted.py.

2. REAL-SYSTEM mask equality: build the EXACT Fig. 2 validation system of
   the accompanying paper (360 Å box, 10 grains, seed 282930,
   aspect_ratio_range [1.0, 2.5]), replay fill_grain's OWN candidate-grid
   construction for every grain, and assert the numba mask equals the
   numpy mask on EVERY candidate — not a sample. This is expensive
   (≈4.2e8 candidates, ≈30 minutes on the numpy side — the entire point of
   the kernel is that the numba side is negligible next to it) and is
   SKIPPED by default; opt in with:

       GRAINSMITH_RUN_FIG2_VALIDATION=1 \\
       PYTHONPATH=src pytest tests/test_owns_kernel.py -v -s -m fig2_validation

   A mismatch here means the kernel is not safe to ship against that
   reference system and is treated as a stop-ship defect — it protects
   the "bit-identically reproducible from shipped config" claim for the
   paper's own Fig. 2 panels.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

import grainsmith.tessellation.weighted as weighted_mod
from grainsmith.seeding import seed_grains, wigner_seitz_radius
from grainsmith.tessellation.weighted import (
    AnisotropicTessellation,
    WeightedTessellation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# paper/ is local, optional tooling outside a public checkout of this
# repository (same status as benchmarks/ in test_bench_cell_runner.py);
# the one test below that reads this path is opt-in
# (GRAINSMITH_RUN_FIG2_VALIDATION) and skipped by default, so a plain
# `pytest` run in a public checkout never touches it.
FIG2_CURVED_YAML = REPO_ROOT / "paper" / "paper_yaml" / "fig_2" / "2_cu_curved.yaml"


def _rng(seed=42):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


@pytest.fixture(params=["numpy", "numba"])
def owns_backend(request, monkeypatch):
    """NOT autouse (deliberately — see module docstring): the Tier 2
    fig2_validation test manages the numpy/numba comparison internally
    on every owns() call and does not request this fixture. Making it
    autouse would silently run that expensive test TWICE (once per
    fixture param) for no benefit. Tier-1 tests below take
    ``owns_backend`` as an explicit parameter instead."""
    if request.param == "numba" and not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setattr(weighted_mod, "_USE_NUMBA", request.param == "numba")
    return request.param


# ---------------------------------------------------------------------------
# Tier 1: fast unit batteries (always run)
# ---------------------------------------------------------------------------

_GEOMETRIES = [
    ("periodic", 6, np.array([40.0, 40.0, 40.0]), [True, True, True]),
    ("slab_z", 5, np.array([40.0, 40.0, 28.0]), [True, True, False]),
    ("two_free", 4, np.array([35.0, 28.0, 28.0]), [True, False, False]),
]


def _build_aniso(n, Lg, periodic, seed=7, aspect=(1.0, 2.5)):
    seeds = seed_grains(n, Lg, periodic, _rng(seed))
    return AnisotropicTessellation(seeds, Lg, periodic,
                                   aspect_ratio_range=aspect, rng=_rng(seed + 1),
                                   connectivity_check=True), seeds


def _build_weighted(n, Lg, periodic, seed=7, sigma=None):
    seeds = seed_grains(n, Lg, periodic, _rng(seed))
    msd = wigner_seitz_radius(float(np.prod(Lg)), n)
    sigma_use = sigma if sigma is not None else msd / 8.0
    return WeightedTessellation(seeds, Lg, periodic, sigma_w=sigma_use,
                                rng=_rng(seed + 1), min_seed_distance=msd,
                                connectivity_check=True), seeds


@pytest.mark.parametrize("geom", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
@pytest.mark.parametrize("backend", ["aniso", "weighted"])
def test_owns_kernel_random_points(backend, geom, owns_backend):
    """(a) uniform random in [-0.2L, 1.2L]^3, both backends, 3 geometries
    — the module's owns_backend fixture already runs this under both the
    numpy and numba dispatch; a physical/invariant-free direct check
    against the tessellation's OWN numpy reference (captured before the
    fixture flips the dispatch) proves the active path is correct."""
    label, n, Lg, periodic = geom
    if backend == "aniso":
        tess, _ = _build_aniso(n, Lg, periodic)
    else:
        tess, _ = _build_weighted(n, Lg, periodic)
    rng = _rng(101)
    X = rng.uniform(-0.2, 1.2, size=(500, 3)) * Lg

    # Reference: force numpy regardless of the fixture's current setting,
    # capture, then restore and compare against whatever the fixture set.
    saved = weighted_mod._USE_NUMBA
    weighted_mod._USE_NUMBA = False
    ref = {i: tess.owns(X, i) for i in range(tess.n_grains)}
    weighted_mod._USE_NUMBA = saved
    got = {i: tess.owns(X, i) for i in range(tess.n_grains)}
    for i in range(tess.n_grains):
        assert np.array_equal(ref[i], got[i]), (
            f"{backend}/{label} grain {i} mismatch under _USE_NUMBA={saved}")


@pytest.mark.parametrize("backend", ["aniso", "weighted"])
def test_owns_kernel_tie_rule(backend, owns_backend):
    """Engineered exact-tie case: the lowest replica (grain) index must
    win, identically under numpy and numba (parametrized via
    owns_backend -> both routes exercised)."""
    Lg = np.array([20.0, 20.0, 20.0])
    seeds = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
    periodic = [True, True, True]
    if backend == "aniso":
        tess = AnisotropicTessellation(seeds, Lg, periodic,
                                       metrics=[np.eye(3), np.eye(3)],
                                       connectivity_check=True)
    else:
        tess = WeightedTessellation(seeds, Lg, periodic, weights=np.zeros(2),
                                    connectivity_check=True)
    mid = np.array([[10.0, 10.0, 10.0]])
    assert bool(tess.owns(mid, 0)[0]) is True
    assert bool(tess.owns(mid, 1)[0]) is False


@pytest.mark.parametrize("backend", ["aniso", "weighted"])
def test_owns_kernel_empty_input(backend, owns_backend):
    Lg = np.array([30.0, 30.0, 30.0])
    seeds = seed_grains(4, Lg, [True, True, True], _rng(11))
    if backend == "aniso":
        tess = AnisotropicTessellation(seeds, Lg, [True, True, True],
                                       aspect_ratio_range=(1.0, 2.0),
                                       rng=_rng(12), connectivity_check=True)
    else:
        tess = WeightedTessellation(seeds, Lg, [True, True, True], sigma_w=1.0,
                                    rng=_rng(12), connectivity_check=True)
    out = tess.owns(np.empty((0, 3)), 0)
    assert out.dtype == bool
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Tier 2: REAL-SYSTEM mask equality against the Fig. 2 validation system
# of the accompanying paper — expensive, opt-in only.
# ---------------------------------------------------------------------------


def _fig2_validation_enabled() -> bool:
    return os.environ.get("GRAINSMITH_RUN_FIG2_VALIDATION", "").strip() in (
        "1", "true", "yes", "on")


@pytest.mark.fig2_validation
@pytest.mark.skipif(not _fig2_validation_enabled(),
                    reason="expensive (~30 min); opt in with "
                          "GRAINSMITH_RUN_FIG2_VALIDATION=1")
def test_fig2_curved_real_system_mask_equality():
    """Build the EXACT Fig. 2 validation system's tessellation,
    replay fill_grain's own candidate-grid construction for every grain,
    and assert the numba mask equals the numpy mask on every single
    candidate point (no sampling). A mismatch here means the kernel is
    NOT safe to ship against the paper's own reference system and must
    be treated as a stop-ship defect.
    """
    if not weighted_mod._HAVE_NUMBA:
        pytest.skip("numba not installed")
    assert FIG2_CURVED_YAML.is_file(), f"missing {FIG2_CURVED_YAML}"

    from grainsmith.atoms.fill import fill_grain
    from grainsmith.config.resolve import load_config
    from grainsmith.pipeline import (
        _build_crystal,
        _resolve_min_seed_distance,
        _stage_orientation,
        _stage_seeding,
        _stage_tessellation,
    )
    from grainsmith.rng import make_rng

    config = load_config(FIG2_CURVED_YAML)
    assert config.boundaries.geometry == "curved"
    assert config.boundaries.curved.method == "anisotropic"

    seed_val = int(config.seed.value)
    rng_bundle = make_rng(seed_val)
    msd = _resolve_min_seed_distance(config)
    seeds = _stage_seeding(config, msd, rng_bundle.seeding)
    tess, _ = _stage_tessellation(config, seeds, msd, rng_bundle.fields,
                                  rng_sizes=rng_bundle.sizes)
    assert isinstance(tess, AnisotropicTessellation)
    crystal = _build_crystal(config.crystal, config.output.lammps.masses)
    quats = _stage_orientation(config, tess.n_grains, crystal.A,
                               rng_bundle.orientation)
    occ_streams = rng_bundle.occupancy_streams(tess.n_grains)

    L = np.asarray(config.box.lengths, dtype=np.float64)
    periodic = config.box.periodic

    total_candidates = 0
    total_mismatches = 0
    mismatch_examples: list[str] = []

    class _CompareBothPaths:
        """Drop-in for ``tess`` inside fill_grain: on every owns() call
        (one per fill.py chunk — see atoms/fill.py's ``chunk_t`` loop),
        compute BOTH the numpy and the numba mask on the SAME candidate
        chunk, tally any mismatch, and return the numba mask so the
        actual fill result matches what production code will emit."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def cell_vertices_rel(self, i):
            cvr = getattr(self._inner, "cell_vertices_rel", None)
            return cvr(i) if cvr is not None else None

        def owns(self, X, i):
            nonlocal total_candidates, total_mismatches
            weighted_mod._USE_NUMBA = False
            numpy_mask = self._inner.owns(X, i)
            weighted_mod._USE_NUMBA = True
            numba_mask = self._inner.owns(X, i)
            diff = numpy_mask != numba_mask
            n_diff = int(diff.sum())
            total_candidates += len(X)
            if n_diff:
                total_mismatches += n_diff
                idx = np.where(diff)[0][:3]
                mismatch_examples.append(
                    f"grain {i}: {n_diff} mismatches, e.g. X[{idx}]="
                    f"{X[idx]}")
            return numba_mask

    wrapped = _CompareBothPaths(tess)
    t0 = time.perf_counter()
    for gi in range(tess.n_grains):
        tg0 = time.perf_counter()
        fill_grain(gi, wrapped, crystal.basis.frac, crystal.basis.species,
                  crystal.basis.occupancy, crystal.A, quats[gi], L, periodic,
                  occ_streams[gi], store_margin=False, a_clip=0.0)
        print(f"[fig2_validation] grain {gi}/{tess.n_grains - 1} done in "
             f"{time.perf_counter() - tg0:.1f} s; running totals: "
             f"{total_candidates:,} candidates, {total_mismatches} "
             f"mismatches so far.", flush=True)
    elapsed = time.perf_counter() - t0

    print(f"[fig2_validation] {total_candidates:,} candidates checked "
         f"across {tess.n_grains} grains in {elapsed:.1f} s; "
         f"{total_mismatches} mismatches.")
    assert total_mismatches == 0, (
        f"{total_mismatches} / {total_candidates} candidates mismatch "
        f"between numba and numpy on the shipped Fig. 2 curved system:\n"
        + "\n".join(mismatch_examples))
def test_sqrt_rounding_collision_tie():
    """Regression: squared-space comparison is NOT exact in float64.

    Engineered pair: qf_home = u*u and qf_other = v*v + w*w are ONE ULP
    apart (qf_home > qf_other strictly), yet np.sqrt maps both to the
    same double -- so the numpy reference (sqrt-space argmin,
    first-minimum tie rule) sees a TIE and awards the point to the
    LOWEST replica index, while a squared-space comparison would see a
    strict inequality and award it to the other grain. The kernel must
    agree with the numpy reference (i.e., compare in sqrt space).

    Constants were found by direct search; the assertions on their ulp
    relationship make the fixture self-verifying on any platform.
    """
    u = float.fromhex("0x1.8000000000001p+1")
    v = float.fromhex("0x1.6a09e667f3e90p+1")
    w = float.fromhex("0x1.fffffffffe0c6p-1")
    qf_home = u * u
    qf_other = v * v + w * w
    # self-verify the engineered relationship
    assert qf_home > qf_other, "fixture broken: expected strict qf inequality"
    assert np.nextafter(qf_other, np.inf) == qf_home, "fixture broken: not 1 ulp apart"
    assert np.sqrt(qf_home) == np.sqrt(qf_other), "fixture broken: sqrt must collide"

    rep_seeds = np.array([[u, 0.0, 0.0], [v, w, 0.0]])
    rep_mid = np.array([0, 1], dtype=np.int64)
    M = np.stack([np.eye(3), np.eye(3)])
    X = np.zeros((1, 3))

    # numpy reference: sqrt-space distances, first-minimum argmin
    d = np.empty((1, 2))
    for j in range(2):
        dr = X - rep_seeds[j][None, :]
        d[:, j] = np.einsum("ni,ij,nj->n", dr, M[rep_mid[j]], dr)
    d = np.sqrt(d)
    assert d[0, 0] == d[0, 1], "fixture broken: reference must tie"
    ref_owner = int(np.argmin(d, axis=1)[0])
    assert ref_owner == 0  # first-minimum rule -> lowest index

    if weighted_mod._owns_kernel_aniso is None:
        pytest.skip("numba unavailable -- kernel not built")
    for home_idx in (0, 1):
        scan = np.array([home_idx, 1 - home_idx], dtype=np.int64)
        got = bool(weighted_mod._owns_kernel_aniso(
            X, rep_seeds, rep_mid, M, np.int64(home_idx), scan)[0])
        assert got == (ref_owner == home_idx), (
            f"kernel diverges from numpy reference at sqrt-rounding "
            f"collision: home_idx={home_idx}, kernel={got}, "
            f"reference_owner={ref_owner}")

