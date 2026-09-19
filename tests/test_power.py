"""Power (Laguerre) diagram + SDOT volume targeting."""
import numpy as np
import pytest
from unittest.mock import patch

from grainsmith.errors import ConfigError, TessellationError
from grainsmith.seeding import seed_grains
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.power import PowerTessellation
from grainsmith.tessellation.sdot import (
    fit_power_weights,
    sample_target_volumes,
)

L = np.array([40.0, 40.0, 40.0])
PER = [True, True, True]
BOX_VOL = float(np.prod(L))


def _rng(seed=7):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _seeds(n=8, seed=7, periodic=PER):
    return seed_grains(n, L, periodic, _rng(seed))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_zero_weights_equals_flat_voronoi():
    """w = 0 ⇒ the power diagram IS the Voronoi diagram (regression pin):
    volumes, adjacency, grain_of, and owns agree."""
    seeds = _seeds()
    flat = FlatTessellation(seeds, L, PER)
    power = PowerTessellation(seeds, L, PER)

    v_f = np.array([c.volume for c in flat.cells])
    v_p = np.array([c.volume for c in power.cells])
    np.testing.assert_allclose(v_p, v_f, rtol=0.0, atol=1e-8)
    assert power.adjacency() == flat.adjacency()

    X = _rng(11).uniform(0.0, 40.0, size=(2000, 3))
    assert np.array_equal(flat.grain_of(X), power.grain_of(X))
    for i in range(len(seeds)):
        assert np.array_equal(flat.owns(X, i), power.owns(X, i))


def test_two_seed_plane_shift_analytic():
    """The (i,j) interface plane sits at the Voronoi bisector shifted by
    (w_i − w_j)/(2 d) toward the lighter grain — including the periodic
    self-pair plane; volumes follow exactly."""
    seeds = np.array([[10.0, 20.0, 20.0], [30.0, 20.0, 20.0]])
    w = np.array([50.0, 0.0])           # shift = 50/(2·20) = 1.25 Å
    tess = PowerTessellation(seeds, L, PER, weights=w)

    xs = sorted(round(float(f.vertices[:, 0].mean()), 9)
                for f in tess.cells[0].faces if f.neighbor_id == 1)
    assert xs == [-1.25, 21.25]
    # slab widths: grain 0 spans [-1.25, 21.25] → V = 22.5·40·40
    assert tess.cells[0].volume == pytest.approx(36000.0, abs=1e-6)
    assert tess.cells[1].volume == pytest.approx(28000.0, abs=1e-6)
    assert sum(c.volume for c in tess.cells) == pytest.approx(BOX_VOL,
                                                              abs=1e-6)


def test_owns_tiles_torus_with_weights():
    """The compact home cells still tile the torus exactly once for
    nonzero weights (fill's correctness invariant)."""
    seeds = _seeds(6)
    w = _rng(3).uniform(0.0, 40.0, size=6)
    tess = PowerTessellation(seeds, L, PER, weights=w)
    X = _rng(13).uniform(-40.0, 80.0, size=(400, 3))   # includes lifts
    lifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
         for k in (-1, 0, 1)], dtype=np.float64) * L
    counts = np.zeros(len(X), dtype=int)
    base = X % L
    for lift in lifts:
        for g in range(6):
            counts += tess.owns(base + lift, g)
    assert np.all(counts == 1)


def test_margin_signed_with_weights():
    """margin is the exact signed plane distance: positive at interior
    points of the owning grain, negative across the boundary."""
    seeds = np.array([[10.0, 20.0, 20.0], [30.0, 20.0, 20.0]])
    tess = PowerTessellation(seeds, L, PER,
                             weights=np.array([50.0, 0.0]))
    # plane at x = 21.25: a point at x = 20.0 is INSIDE grain 0 by 1.25
    X = np.array([[20.0, 20.0, 20.0]])
    assert tess.margin(X, 0)[0] == pytest.approx(1.25, abs=1e-9)
    assert tess.margin(X, 1)[0] == pytest.approx(-1.25, abs=1e-9)


def test_slab_walls_with_weights():
    seeds = _seeds(4, periodic=[True, True, False])
    w = _rng(5).uniform(0.0, 20.0, size=4)
    tess = PowerTessellation(seeds, L, [True, True, False], weights=w)
    wall_faces = [f for c in tess.cells for f in c.faces
                  if f.neighbor_id == -1]
    assert wall_faces, "slab must produce wall faces"
    # exact wall clipping: every wall-face vertex sits on z=0 or z=L
    for f in wall_faces:
        z = f.vertices[:, 2]
        assert np.allclose(z, 0.0, atol=1e-8) or \
            np.allclose(z, 40.0, atol=1e-8)


# ---------------------------------------------------------------------------
# SDOT volume targeting
# ---------------------------------------------------------------------------


def test_sdot_equal_volumes():
    seeds = _seeds(8)
    targets = sample_target_volumes("equal", 8, BOX_VOL, _rng(1))
    res = fit_power_weights(seeds, L, PER, targets, vol_tol=1e-4)
    assert res.converged
    assert res.max_rel_error <= 1e-4
    v = np.array([c.volume for c in res.tess.cells])
    np.testing.assert_allclose(v, BOX_VOL / 8.0, rtol=1e-4)


def test_sdot_lognormal_matches_target_quantiles():
    seeds = _seeds(8)
    targets = sample_target_volumes("lognormal", 8, BOX_VOL, _rng(2),
                                    sigma_log=0.35)
    assert targets.sum() == pytest.approx(BOX_VOL, rel=1e-12)
    res = fit_power_weights(seeds, L, PER, targets, vol_tol=1e-3)
    assert res.converged
    np.testing.assert_allclose(np.sort(res.achieved_volumes),
                               np.sort(targets), rtol=1e-3)


def test_sdot_slab():
    per = [True, True, False]
    seeds = _seeds(6, seed=9, periodic=per)
    targets = sample_target_volumes("lognormal", 6, BOX_VOL, _rng(4), 0.3)
    res = fit_power_weights(seeds, L, per, targets, vol_tol=1e-3)
    assert res.converged and res.max_rel_error <= 1e-3


def test_sdot_centroidal_iterations_keep_volumes():
    seeds = _seeds(6, seed=21)
    targets = sample_target_volumes("lognormal", 6, BOX_VOL, _rng(6), 0.25)
    res = fit_power_weights(seeds, L, PER, targets, vol_tol=1e-3,
                            centroidal_iterations=2)
    assert res.converged and res.max_rel_error <= 1e-3


def test_sdot_deterministic():
    seeds = _seeds(6, seed=33)
    t1 = sample_target_volumes("lognormal", 6, BOX_VOL, _rng(8), 0.3)
    t2 = sample_target_volumes("lognormal", 6, BOX_VOL, _rng(8), 0.3)
    np.testing.assert_array_equal(t1, t2)
    r1 = fit_power_weights(seeds, L, PER, t1, vol_tol=1e-3)
    r2 = fit_power_weights(seeds, L, PER, t2, vol_tol=1e-3)
    np.testing.assert_array_equal(r1.weights, r2.weights)


def test_sample_targets_validation():
    with pytest.raises(ConfigError):
        sample_target_volumes("nope", 4, BOX_VOL, _rng(0))
    with pytest.raises(ConfigError):
        sample_target_volumes("lognormal", 4, BOX_VOL, _rng(0),
                              sigma_log=0.0)
    with pytest.raises(ConfigError):
        sample_target_volumes("volumes", 4, BOX_VOL, _rng(0),
                              volumes=[1.0, 2.0])
    with pytest.raises(ConfigError):
        sample_target_volumes("volumes", 2, BOX_VOL, _rng(0),
                              volumes=[1.0, -2.0])
    v = sample_target_volumes("volumes", 2, BOX_VOL, _rng(0),
                              volumes=[1.0, 3.0])
    np.testing.assert_allclose(v, [BOX_VOL / 4.0, 3.0 * BOX_VOL / 4.0])


# ---------------------------------------------------------------------------
# Fix #23 — QhullError is caught and re-raised as TessellationError
# ---------------------------------------------------------------------------


def test_qhull_error_reraises_as_tessellation_error():
    """Fix #23: a QhullError from HalfspaceIntersection must surface as a
    TessellationError so the SDOT backtracking loop (which catches only
    TessellationError) can halve the step instead of crashing.

    We trigger the path by monkeypatching HalfspaceIntersection to raise
    QhullError directly — this is the minimal approach that avoids needing
    a pathological geometry while still exercising the exact try/except
    branch added by the fix."""
    try:
        from scipy.spatial import QhullError
    except ImportError:
        from scipy.spatial.qhull import QhullError  # type: ignore[no-redef]

    seeds = np.array([[20.0, 20.0, 20.0], [10.0, 10.0, 10.0]])
    w = np.array([0.0, 0.0])

    with patch(
        "grainsmith.tessellation.power.HalfspaceIntersection",
        side_effect=QhullError("mock degenerate halfspace"),
    ):
        with pytest.raises(TessellationError, match="halfspace intersection degenerate"):
            PowerTessellation(seeds, L, PER, weights=w)
