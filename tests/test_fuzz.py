"""Property / fuzz tests.

Three invariants that must hold across randomized inputs:
  1. config resolve never crashes — it returns a config or raises ConfigError;
  2. owns() tiles the torus exactly once for random rotations × boxes;
  3. the LAMMPS writer round-trips (parse-back conserves count + coordinates).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from grainsmith.config.resolve import resolve_config
from grainsmith.errors import ConfigError

_BASE: dict = {
    "seed": {"mode": "fixed", "value": 1},
    "box": {"lengths": [16.0, 16.0, 16.0]},
    "grains": {"number": 3},
    "crystal": {
        "space_group": {"number": 225},
        "lattice": {"a": 3.615},
        "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
    },
    "orientation": {"scheme": "random_uniform"},
}


def test_config_resolve_fuzz_never_crashes():
    """1e3 seeded mutations of a valid config must each either resolve or raise
    a clean ConfigError — never a raw KeyError/TypeError/ValidationError."""
    rng = np.random.default_rng(20260613)
    weird_floats = [0.0, -1.0, float("nan"), float("inf"), 1e9, 1e-9, 16.0]
    weird_any = [None, "x", [], {}, -5, 0, 3.5, [1, 2], float("nan")]
    def pick_float():
        return weird_floats[int(rng.integers(len(weird_floats)))]

    def pick_any():
        return weird_any[int(rng.integers(len(weird_any)))]
    n_ok = n_cfgerr = 0
    for _ in range(1000):
        raw = copy.deepcopy(_BASE)
        # box lengths
        if rng.random() < 0.5:
            raw["box"]["lengths"] = [pick_float() for _ in range(3)]
        # grain count
        if rng.random() < 0.4:
            raw["grains"]["number"] = pick_any()
        # space group number
        if rng.random() < 0.4:
            raw["crystal"]["space_group"]["number"] = int(rng.integers(-5, 260))
        # lattice param
        if rng.random() < 0.3:
            raw["crystal"]["lattice"]["a"] = pick_float()
        # wyckoff coords
        if rng.random() < 0.3:
            raw["crystal"]["wyckoff_sites"][0]["coords"] = [pick_any()
                                                            for _ in range(3)]
        # inject an unknown top-level key (extra='forbid' must reject)
        if rng.random() < 0.2:
            raw[f"junk_{int(rng.integers(99))}"] = pick_any()
        # drop a required block
        if rng.random() < 0.15:
            raw.pop(["box", "grains", "crystal", "orientation"][int(rng.integers(4))])

        try:
            resolve_config(raw)
            n_ok += 1
        except ConfigError:
            n_cfgerr += 1
        # any OTHER exception propagates and fails the test
    assert n_cfgerr > 0 and n_ok > 0   # the fuzz actually exercised both paths


def test_owns_tiling_fuzz():
    """owns() over the 27 home-cell lifts × all grains must sum to exactly 1
    for every query point, across random seed sets and anisotropic boxes."""
    from grainsmith.tessellation.flat import FlatTessellation
    rng = np.random.default_rng(7)
    lifts = np.array([[sx, sy, sz]
                      for sx in (-1, 0, 1)
                      for sy in (-1, 0, 1)
                      for sz in (-1, 0, 1)], dtype=np.float64)
    for _ in range(15):
        L = rng.uniform(14.0, 28.0, size=3)
        n = int(rng.integers(2, 6))
        seeds = rng.uniform(0.0, 1.0, size=(n, 3)) * L
        tess = FlatTessellation(seeds, L, [True, True, True])
        pts = rng.uniform(0.0, 1.0, size=(40, 3)) * L
        total = np.zeros(len(pts), dtype=np.int64)
        for shift in lifts:
            shifted = pts + shift * L
            for g in range(n):
                total += tess.owns(shifted, g).astype(np.int64)
        assert np.all(total == 1), f"tiling broken: counts {np.unique(total)}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lammps_writer_roundtrip_fuzz(tmp_path, seed):
    """A small random polycrystal: parse_lammps must conserve atom count and
    coordinates (modulo the periodic wrap) — extends the G10 bounds pin to xyz."""
    from grainsmith.io.lammps import parse_lammps
    from grainsmith.pipeline import run

    raw = copy.deepcopy(_BASE)
    raw["seed"]["value"] = seed
    raw["box"]["lengths"] = [13.0, 14.0, 15.0]
    raw["analysis"] = {"statistics": False}
    raw["output"] = {"directory": str(tmp_path / "out"),
                     "methods_snippet": False}
    res = run(resolve_config(raw))
    L = np.asarray(raw["box"]["lengths"])

    d = parse_lammps(res.outdir / "polycrystal.data")
    assert d["natoms"] == len(res.atoms.pos)
    parsed = np.asarray(d["pos"], dtype=np.float64)
    # all atoms inside the box
    assert np.all(parsed >= -1e-6) and np.all(parsed <= L + 1e-6)
    # coordinate multiset matches the writer's periodic wrap (0 ≡ L safe)
    wrapped = res.atoms.pos % L
    a = parsed[np.lexsort(parsed.T)]
    b = wrapped[np.lexsort(wrapped.T)]
    delta = (a - b + L / 2.0) % L - L / 2.0
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)
