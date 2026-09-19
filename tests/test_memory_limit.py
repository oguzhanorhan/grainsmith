"""SPEC7 — user-configurable §13 per-allocation memory guard.

Covers the ``runtime.memory_limit_gb`` config knob (config/schema.py's
``RuntimeConfig``), the raised 8 -> 16 GB default (constants.py's
``MEMORY_HARD_LIMIT_BYTES``), and the instance-attribute plumbing
(``Tessellation.memory_limit_bytes``, tessellation/base.py) that lets the
pipeline stamp a per-run value onto every guard site (atoms/fill.py,
tessellation/voxel.py, tessellation/voxel_import.py) while staying
correct under a ``ProcessPoolExecutor`` 'spawn' worker.

Item 8 of the spec ("update any existing test that hard-codes 8 GB / 8e9
semantics") is handled IN PLACE, not here: tests/test_fill.py's
``_BigRadiusTess``, tests/test_curvature.py's ``_SphereTess``/``_ShellTess``,
and tests/test_doping.py's ``_OneGrainTess`` are duck-typed tessellation
stubs that don't subclass ``Tessellation`` and therefore need an explicit
``memory_limit_bytes`` class attribute now that the guard sites read it off
the instance instead of the constants.py module constant; each got a
one-line addition with a guardrail comment. tests/test_fill.py:350's own
"~2001^3 ~ 8e9" comment is about int32 GRID-POINT-COUNT overflow, not the
memory guard's byte budget, so it was deliberately left untouched.
"""
from __future__ import annotations

import contextlib
import logging
import pickle

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import grainsmith.memory as memory_mod
from grainsmith.config.resolve import load_config, resolve_config
from grainsmith.config.schema import RuntimeConfig
from grainsmith.constants import MEMORY_HARD_LIMIT_BYTES
from grainsmith.errors import TessellationError
from grainsmith.memory import (
    build_memory_guard_message,
    format_bytes_adaptive,
    resolve_memory_budget,
)
from grainsmith.pipeline import (
    _resolve_min_seed_distance,
    _stage_seeding,
    _stage_tessellation,
    run,
)
from grainsmith.rng import make_rng
from grainsmith.tessellation.flat import FlatTessellation
from grainsmith.tessellation.voxel import build_voxel_grid
from grainsmith.tessellation.voxel_import import VoxelTessellation


def _raw(**overrides):
    """Minimal valid RunConfig dict (FCC Cu, 4 grains, 40 A box) --
    mirrors tests/test_config.py's / tests/test_doping.py's ``_raw``."""
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0],
                "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
    }
    for key, sub in overrides.items():
        if isinstance(sub, dict) and isinstance(raw.get(key), dict):
            raw[key].update(sub)
        else:
            raw[key] = sub
    return raw


# ---------------------------------------------------------------------------
# 1. Single source of truth + pinned default (D3, D4)
# ---------------------------------------------------------------------------


def test_runtime_default_matches_constant_and_is_pinned_at_16gb():
    """RuntimeConfig.memory_limit_gb's default must be DERIVED from
    MEMORY_HARD_LIMIT_BYTES (single source of truth, D3), and that
    constant is now 16 GB (D4, raised from 8 GB) -- pinned explicitly so a
    silent future change to either one fails this test instead of being
    discovered downstream."""
    assert RuntimeConfig().memory_limit_gb * 1e9 == MEMORY_HARD_LIMIT_BYTES
    assert MEMORY_HARD_LIMIT_BYTES == 16e9


# ---------------------------------------------------------------------------
# 2. Schema bounds: extra='forbid', gt=0.0 (D3)
# ---------------------------------------------------------------------------


def test_runtime_config_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        RuntimeConfig(memory_limit_gb=16, bogus=1)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_runtime_config_memory_limit_gb_must_be_positive(bad_value):
    with pytest.raises(ValidationError):
        RuntimeConfig(memory_limit_gb=bad_value)


# ---------------------------------------------------------------------------
# 3. YAML round-trip through load_config / resolve_config (D3)
# ---------------------------------------------------------------------------


def test_runtime_memory_limit_gb_round_trips_through_load_config(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_raw(runtime={"memory_limit_gb": 24})),
        encoding="utf-8",
    )
    config = load_config(cfg_path)   # calls resolve_config internally
    assert config.runtime.memory_limit_gb == 24


# ---------------------------------------------------------------------------
# 4. The guard actually reaches fill.py / voxel.py / voxel_import.py (D2)
# ---------------------------------------------------------------------------


def test_fill_guard_reaches_via_tess_memory_limit_bytes():
    """A resolved config's runtime.memory_limit_gb, stamped onto a real
    tessellation exactly as pipeline._apply_runtime_limits does, must
    make fill_grain raise §13 -- and the message must name the knob."""
    from grainsmith.atoms.fill import fill_grain
    from grainsmith.crystal.cell import cell_matrix
    from grainsmith.crystal.spacegroup import (
        WyckoffSite,
        expand_wyckoff,
        hall_from_international,
        symmetry_ops,
    )

    config = resolve_config(_raw(runtime={"memory_limit_gb": 1e-6}))

    A_len = 3.615
    A = cell_matrix(A_len, A_len, A_len, 90.0, 90.0, 90.0)
    rots, trans = symmetry_ops(hall_from_international(225))
    basis = expand_wyckoff([WyckoffSite("Cu", [0.0, 0.0, 0.0])], rots, trans)

    L = np.array([4 * A_len] * 3)
    seeds = np.array([[0.5 * L[0], 0.5 * L[1], 0.5 * L[2]]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    tess.memory_limit_bytes = config.runtime.memory_limit_gb * 1e9

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(0)))
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TessellationError) as excinfo:
        fill_grain(0, tess, basis.frac, basis.species, basis.occupancy,
                   A, q_id, L, [True, True, True], rng)
    msg = str(excinfo.value)
    assert "runtime.memory_limit_gb" in msg
    assert "§13" in msg


def test_voxel_grid_guard_reaches_via_tess_memory_limit_bytes():
    config = resolve_config(_raw(runtime={"memory_limit_gb": 1e-6}))

    L = np.array([40.0, 40.0, 40.0])
    seeds = np.array([[10.0, 10.0, 10.0], [30.0, 30.0, 30.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    tess.memory_limit_bytes = config.runtime.memory_limit_gb * 1e9

    with pytest.raises(TessellationError) as excinfo:
        build_voxel_grid(tess, L, grid_size=20)
    msg = str(excinfo.value)
    assert "runtime.memory_limit_gb" in msg
    assert "§13" in msg


def test_voxel_import_margin_guard_reaches_via_self_memory_limit_bytes():
    """Reachable cheaply: VoxelTessellation takes an in-memory labels array,
    no file fixture needed (mirrors tests/test_voxel_import.py's
    ``_bicrystal_labels``)."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[10:] = 1
    tess = VoxelTessellation(labels, np.array([20.0, 20.0, 20.0]),
                             [True, True, True])
    tess.memory_limit_bytes = 1e-6 * 1e9   # absurdly small, decimal GB

    with pytest.raises(TessellationError) as excinfo:
        tess.margin(np.array([[1.0, 1.0, 1.0]]), 0)
    msg = str(excinfo.value)
    assert "runtime.memory_limit_gb" in msg
    assert "§13" in msg


# ---------------------------------------------------------------------------
# 5. Raising the limit admits what the default refuses (D1/D2)
# ---------------------------------------------------------------------------


def test_raising_limit_admits_what_smaller_limit_refuses():
    """Pick a voxel grid size whose byte estimate sits strictly between two
    limits: the guard must fire at the small one and pass at the large
    one, for the SAME tessellation and SAME grid request."""
    L = np.array([40.0, 40.0, 40.0])
    seeds = np.array([[10.0, 10.0, 10.0], [30.0, 30.0, 30.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])

    # grid_shape = (50, 50, 50) exactly (L_min=40, h=40/50=0.8, cubic box)
    # -> n_vox = 125_000 -> est_bytes = 125_000 * 56 = 7_000_000 (~7 MB).
    grid_size = 50

    tess.memory_limit_bytes = 1e6          # 1 MB -- below the ~7 MB estimate
    with pytest.raises(TessellationError, match="§13"):
        build_voxel_grid(tess, L, grid_size=grid_size)

    tess.memory_limit_bytes = 1e9          # 1 GB -- above the ~7 MB estimate
    vg = build_voxel_grid(tess, L, grid_size=grid_size)
    assert vg.labels.shape == (50, 50, 50)


def test_analysis_reuses_existing_grid_at_requested_resolution(monkeypatch):
    from types import SimpleNamespace

    import grainsmith.tessellation.voxel as voxel_mod
    from grainsmith.pipeline import _analysis_voxel_grid

    config = resolve_config(_raw(analysis={"voxel_grid": 20}))
    lengths = np.asarray(config.box.lengths, dtype=np.float64)
    existing = voxel_mod.VoxelGrid(
        np.zeros((20, 20, 20), dtype=np.int32), lengths, (20, 20, 20), 4)
    tess = SimpleNamespace(_voxel=existing)

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("Existing analysis grid must not be allocated again")

    monkeypatch.setattr(voxel_mod, "build_voxel_grid", unexpected_rebuild)
    assert _analysis_voxel_grid(config, tess) is existing
    assert tess._voxel_is_explicit


def test_hurst_diagnostic_does_not_copy_all_color_fields(monkeypatch):
    import grainsmith.tessellation.warp as warp_mod

    field = np.random.default_rng(7).normal(size=(6, 16, 16, 16)).astype(np.float32)
    lengths = np.full(3, 80.0)
    expected = warp_mod.estimate_hurst(field.astype(np.float64), lengths, 10.0, 40.0)
    original_asarray = np.asarray

    def reject_full_conversion(value, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if value is field and dtype is not None and np.dtype(dtype) == np.float64:
            pytest.fail("Hurst diagnostics must convert one color at a time")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(warp_mod.np, "asarray", reject_full_conversion)
    assert warp_mod.estimate_hurst(field, lengths, 10.0, 40.0) == expected


def test_hurst_diagnostic_checks_limit_before_large_allocations(monkeypatch):
    import grainsmith.tessellation.warp as warp_mod

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("The Hurst memory guard must run before allocating the k grid")

    monkeypatch.setattr(warp_mod, "_k_magnitude", unexpected_allocation)
    with pytest.raises(TessellationError, match="Hurst.*runtime.memory_limit_gb"):
        warp_mod.estimate_hurst(
            np.zeros((2, 16, 16, 16), dtype=np.float32), np.full(3, 80.0),
            10.0, 40.0, memory_limit_bytes=1.0,
        )


def test_memory_failure_records_traceback_and_partial_csv(tmp_path, monkeypatch):
    import csv

    import grainsmith.pipeline as pipeline_mod

    config = resolve_config(_raw(output={"directory": str(tmp_path)},
                                 meta={"verbose": 0}))
    previous = b"section,key,value\nmeta,status,previous_completed_run\n"
    (tmp_path / "summary.csv").write_bytes(previous)

    def fail_allocation(*args, **kwargs):
        raise MemoryError("synthetic diagnostic allocation failure")

    monkeypatch.setattr(pipeline_mod, "_stage_tessellation", fail_allocation)
    with pytest.raises(MemoryError, match="synthetic diagnostic allocation failure"):
        pipeline_mod.run(config)
    assert (tmp_path / "summary.csv").read_bytes() == previous
    with (tmp_path / "summary.failed.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(row["section"], row["key"]): row["value"] for row in csv.DictReader(stream)}
    assert rows["meta", "status"] == "failed"
    assert rows["failure", "exception_type"] == "MemoryError"
    assert rows["failure", "message"] == "synthetic diagnostic allocation failure"
    assert float(rows["timings", "elapsed_until_failure_s"]) > 0.0
    assert float(rows["timings", "crystal_s"]) > 0.0
    assert rows["config", "runtime.memory_limit_gb"] == "16.0"
    assert rows["memory", "rss_limit_kind"] == "warning_only"
    log_text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log_text
    assert "MemoryError: synthetic diagnostic allocation failure" in log_text


# ---------------------------------------------------------------------------
# 6. Spawn-safety: the budget travels through pickle (D1)
# ---------------------------------------------------------------------------


def test_memory_limit_bytes_survives_pickle_spawn_safety():
    """fill_grain runs inside ProcessPoolExecutor workers (atoms/fill.py,
    jobs > 1); under the 'spawn' start method (Windows, macOS -- and
    available, if not default, on Linux) a worker re-imports grainsmith
    modules from scratch rather than inheriting process state, so it would
    see the unpatched class/module default for anything stored as a
    module global. Because the run's budget is stamped as an INSTANCE
    attribute (D1), it lives in the tessellation's __dict__ and is
    serialized by pickle -- exactly the mechanism ProcessPoolExecutor uses
    to hand arguments to a spawned worker. This proves the VALUE (not just
    a reference to a shared object) makes that trip."""
    L = np.array([40.0, 40.0, 40.0])
    seeds = np.array([[20.0, 20.0, 20.0]])
    tess = FlatTessellation(seeds, L, [True, True, True])
    custom = 12.5e9
    tess.memory_limit_bytes = custom

    restored = pickle.loads(pickle.dumps(tess))
    assert restored.memory_limit_bytes == custom


# ---------------------------------------------------------------------------
# 7. End-to-end through pipeline.run() (D5)
# ---------------------------------------------------------------------------


def test_pipeline_e2e_tiny_limit_fails_default_succeeds(tmp_path):
    """pipeline.run() must stamp config.runtime.memory_limit_gb onto the
    tessellation before any stage that can trip the §13 guard -- an
    absurdly small configured limit fails the run, the (raised, 16 GB)
    default succeeds it, for the identical tiny 4-grain/40 A/FCC-Cu
    config."""
    raw_small = _raw(
        output={"directory": str(tmp_path / "out_small")},
        runtime={"memory_limit_gb": 1e-6},
    )
    with pytest.raises(TessellationError) as excinfo:
        run(resolve_config(raw_small))
    msg = str(excinfo.value)
    assert "runtime.memory_limit_gb" in msg
    assert "§13" in msg

    raw_default = _raw(output={"directory": str(tmp_path / "out_default")})
    result = run(resolve_config(raw_default))
    assert result.gates.all_passed()


# ---------------------------------------------------------------------------
# 8. The CURVED-BACKEND construction-time regression.
#
# PROVEN DEFECT (fixed by this section's changes): for every curved
# backend the dominant allocation -- the voxel grid -- is built INSIDE
# __init__ (the G5 connectivity check for additive_weights/anisotropic/
# warp; perturbed_distance's repair-and-build-override), which runs
# during pipeline._stage_tessellation, strictly BEFORE
# pipeline._apply_runtime_limits ever gets a chance to stamp the instance.
# Before the fix, that construction-time guard therefore always read the
# 16 GB CLASS default, silently ignoring runtime.memory_limit_gb --
# empirically confirmed: with runtime.memory_limit_gb: 7.0 configured,
# build_voxel_grid on a WeightedTessellation read 16.0 GB. The fix
# resolves the budget ONCE in _stage_tessellation and threads it to every
# curved constructor as memory_limit_bytes=/memory_limit_source= keyword
# arguments, so construction sees the real, RAM-clamped per-run value.
# ---------------------------------------------------------------------------


def _build_curved_tess(config):
    """Replicate pipeline.run()'s seeding + tessellation dispatch for a
    resolved curved config, returning the constructed tessellation (or
    raising, exactly as run() would at the tessellation stage)."""
    rng = make_rng(config.seed.value)
    msd = _resolve_min_seed_distance(config)
    seeds = _stage_seeding(config, msd, rng.seeding)
    tess, _sdot = _stage_tessellation(config, seeds, msd, rng.fields,
                                      rng_sizes=rng.sizes)
    return tess


# One config per curved method reachable from boundaries.curved.method,
# copied from tests/test_curvature.py's own e2e fixtures (same 40 A / 4
# grain / FCC-Cu box as _raw() above) so each is known to construct
# cleanly (G5/G6 guards pass) at a generous memory limit.
CURVED_METHOD_OVERRIDES = {
    "additive_weights": {"method": "additive_weights", "weight_sigma": 1.0},
    "anisotropic": {"method": "anisotropic",
                    "aspect_ratio_range": [1.0, 2.5]},
    "warp": {"method": "warp", "amplitude": 0.6, "correlation_length": 10.0},
    "perturbed_distance": {"method": "perturbed_distance", "amplitude": 1.0,
                           "spectrum": "self_affine", "hurst": 0.9,
                           "l_min": 5.0, "l_max": 13.0},
}


@pytest.mark.parametrize("method", sorted(CURVED_METHOD_OVERRIDES))
def test_curved_backend_construction_sees_configured_limit(method):
    """THE core regression test: a DISTINCTIVE configured limit (12.3 GB,
    chosen to differ from both the 16 GB class default and any
    conveniently-round number) must be visible on tess.memory_limit_bytes
    immediately after construction, for EVERY curved backend. This would
    have FAILED before the fix (tess.memory_limit_bytes would read
    16e9 == MEMORY_HARD_LIMIT_BYTES, not 12.3e9, because construction never
    received the run's budget)."""
    distinctive_gb = 12.3
    config = resolve_config(_raw(
        runtime={"memory_limit_gb": distinctive_gb},
        boundaries={"geometry": "curved",
                   "curved": CURVED_METHOD_OVERRIDES[method]},
    ))
    tess = _build_curved_tess(config)
    assert tess.memory_limit_bytes == pytest.approx(distinctive_gb * 1e9)
    assert tess.memory_limit_bytes != MEMORY_HARD_LIMIT_BYTES
    assert tess.memory_limit_source == "config"


def test_pipeline_e2e_limit_between_fill_and_voxel_trips_voxel_guard_only(
    tmp_path,
):
    """A limit strictly between the per-grain fill-grid estimate and the
    CONSTRUCTION-TIME voxel-grid estimate, for the identical curved config,
    must trip the voxel guard specifically -- and must NOT be reachable by
    fill at all, because tessellation (where curved construction happens)
    runs strictly before fill in the pipeline.

    Measured directly for this exact config (additive_weights,
    weight_sigma=1.0, the _raw() 4-grain/40 A/FCC-Cu box, seed=1): the
    fill-grid estimate for grain 0 is ~0.236 MB, and the construction-time
    voxel grid (auto resolution, (26, 26, 26)) is ~0.984 MB. 0.5 MB (5e-4
    GB) sits strictly between the two.

    Before the fix this test would NOT have failed the run at all: the
    construction-time guard read the 16 GB class default, far above
    either estimate, so tessellation (and the whole run) would have
    succeeded regardless of this configured limit.
    """
    curved = {"method": "additive_weights", "weight_sigma": 1.0}
    between_gb = 500_000 / 1e9   # 0.5 MB, strictly between ~0.236 and ~0.984 MB

    # Isolate: at this SAME limit, fill_grain alone (voxel construction
    # bypassed via connectivity_check=False) must NOT trip.
    from grainsmith.atoms.fill import fill_grain
    from grainsmith.crystal.cell import cell_matrix
    from grainsmith.crystal.spacegroup import (
        WyckoffSite,
        expand_wyckoff,
        hall_from_international,
        symmetry_ops,
    )
    from grainsmith.tessellation.weighted import WeightedTessellation

    config = resolve_config(_raw(
        boundaries={"geometry": "curved", "curved": curved},
        runtime={"memory_limit_gb": between_gb},
    ))
    L = np.asarray(config.box.lengths, dtype=np.float64)
    rng = make_rng(config.seed.value)
    msd = _resolve_min_seed_distance(config)
    seeds = _stage_seeding(config, msd, rng.seeding)
    isolated_tess = WeightedTessellation(
        seeds, L, config.box.periodic, sigma_w=1.0, rng=rng.fields,
        min_seed_distance=msd, connectivity_check=False,
        memory_limit_bytes=between_gb * 1e9, memory_limit_source="config",
    )
    A_len = 3.615
    A = cell_matrix(A_len, A_len, A_len, 90.0, 90.0, 90.0)
    rots, trans = symmetry_ops(hall_from_international(225))
    basis = expand_wyckoff([WyckoffSite("Cu", [0.0, 0.0, 0.0])], rots, trans)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    fill_grain(0, isolated_tess, basis.frac, basis.species, basis.occupancy,
              A, q_id, L, config.box.periodic,
              np.random.Generator(np.random.PCG64(np.random.SeedSequence(0))))
    # (no raise above => fill alone does not trip at this limit)

    # The real pipeline run: tessellation runs before fill, so the SAME
    # limit trips the voxel guard at construction, before fill ever runs.
    raw = _raw(
        output={"directory": str(tmp_path / "out")},
        boundaries={"geometry": "curved", "curved": curved},
        runtime={"memory_limit_gb": between_gb},
    )
    with pytest.raises(TessellationError) as excinfo:
        run(resolve_config(raw))
    msg = str(excinfo.value)
    assert "Voxel grid" in msg
    assert "fill_grain" not in msg
    assert "§13" in msg


# ---------------------------------------------------------------------------
# 9. RAM clamp: physical RAM, when smaller, binds -- and the message says
#    the MACHINE lacks RAM, not "raise the config value" (which the clamp
#    would just re-clamp).
# ---------------------------------------------------------------------------


def test_ram_clamp_binds_and_reports_source_ram(monkeypatch):
    small_ram_bytes = 2e9   # 2 GB, far below the 16 GB default
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: small_ram_bytes)

    config = resolve_config(_raw())   # default runtime.memory_limit_gb=16.0
    limit_bytes, source, ram_bytes = resolve_memory_budget(config)
    assert source == "ram"
    assert limit_bytes == small_ram_bytes
    assert ram_bytes == small_ram_bytes

    msg = build_memory_guard_message(
        "test allocation", 3e9, limit_bytes, source)
    assert "does not have enough RAM" in msg
    assert "§13" in msg
    # The "raise runtime.memory_limit_gb" advice is NOT actionable when
    # RAM itself is the binding constraint -- the clamp would just
    # re-clamp a higher config value straight back down.
    assert "Raise runtime.memory_limit_gb" not in msg


def test_ram_clamp_propagates_to_curved_construction(monkeypatch):
    """End-to-end: a RAM clamp small enough to bind must reach the curved
    backend's construction-time guard (memory_limit_source == 'ram') and
    produce the RAM-flavored message there too -- not just in the unit-
    level resolve_memory_budget/build_memory_guard_message check above."""
    small_ram_bytes = 100_000.0   # 100 kB -- below the ~0.984 MB voxel estimate
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: small_ram_bytes)

    curved = {"method": "additive_weights", "weight_sigma": 1.0}
    config = resolve_config(_raw(
        boundaries={"geometry": "curved", "curved": curved}))
    with pytest.raises(TessellationError) as excinfo:
        _build_curved_tess(config)
    msg = str(excinfo.value)
    assert "does not have enough RAM" in msg
    assert "§13" in msg


# ---------------------------------------------------------------------------
# 10. Undetectable RAM: falls back to configured, warns, never crashes.
# ---------------------------------------------------------------------------


def test_undetectable_ram_falls_back_to_configured_and_warns(monkeypatch,
                                                              caplog):
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: None)

    config = resolve_config(_raw(runtime={"memory_limit_gb": 5.0}))
    with caplog.at_level("WARNING", logger="grainsmith.memory"):
        limit_bytes, source, ram_bytes = resolve_memory_budget(config)

    assert limit_bytes == pytest.approx(5.0e9)
    assert source == "config"
    assert ram_bytes is None
    assert any("RAM" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 10b. resolve_memory_budget must run EXACTLY ONCE per pipeline.run() call --
#    not once in _stage_tessellation AND again in _apply_runtime_limits (the
#    original defect: both independently called
#    memory.resolve_memory_budget, so a single run logged its "RAM could not
#    be detected" warning TWICE). pipeline.run() now resolves the budget
#    once, up front, and threads the resolved tuple into both consumers,
#    which skip re-resolving (and re-logging) when it is given.
#
#    The trap a naive fix could fall into: silencing the warning inside
#    _apply_runtime_limits specifically. That would be WRONG, because
#    _apply_runtime_limits is the ONLY resolver on two of the three
#    tessellation-binding paths in run() -- grains.number == 1
#    (SingleCrystalTessellation) and boundaries.geometry == "voxel_import"
#    (_stage_voxel_import) -- neither of which ever calls
#    _stage_tessellation. Silencing _apply_runtime_limits's warning would
#    make those two paths lose the warning entirely rather than dedupe it.
#    The tests below cover a flat/curved run (goes through
#    _stage_tessellation) AND both of those _apply_runtime_limits-only
#    paths, so a regression to the naive fix is caught either way.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_memory_log():
    """Capture ``grainsmith.memory`` log records by attaching a handler
    directly to that logger, bypassing pytest's ``caplog`` fixture.

    ``pipeline._setup_logging`` calls ``logging.basicConfig(..., force=
    True)`` on every ``run()`` call -- ``force=True`` clears and replaces
    the ROOT logger's handlers, which silently discards whatever handler
    ``caplog`` had installed there and makes ``caplog.records`` come back
    empty for anything logged during ``run()``. A handler attached directly
    to the ``grainsmith.memory`` logger (not the root) is untouched by
    that reconfiguration, since ``basicConfig`` only ever touches the root
    logger's own handler list."""
    logger = logging.getLogger("grainsmith.memory")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _count(records, needle):
    return sum(1 for rec in records if needle in rec.getMessage())


def test_undetectable_ram_warns_exactly_once_per_run_flat_curved(
    tmp_path, monkeypatch,
):
    """The ordinary flat/curved path (grains.number > 1) goes through BOTH
    _stage_tessellation and _apply_runtime_limits -- exactly the two call
    sites that used to each resolve independently. One run must produce
    exactly ONE "RAM could not be detected" warning, not two."""
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: None)
    raw = _raw(output={"directory": str(tmp_path / "out")})
    with _capture_memory_log() as records:
        result = run(resolve_config(raw))
    assert result.gates.all_passed()
    assert _count(records, "could not be detected") == 1
    assert _count(records, "Memory guard (§13)") == 1


def test_undetectable_ram_warns_exactly_once_per_run_single_crystal(
    tmp_path, monkeypatch,
):
    """THE TRAP: grains.number == 1 binds via SingleCrystalTessellation
    (pipeline.py, run()) and NEVER calls _stage_tessellation --
    _apply_runtime_limits is its only resolver/stamp. A naive fix that
    simply silenced (or removed) the warning inside _apply_runtime_limits
    would make this path lose the warning ENTIRELY (0, not 1). It must
    still fire exactly once."""
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: None)
    raw = _raw(
        output={"directory": str(tmp_path / "out")},
        grains={"number": 1},
    )
    with _capture_memory_log() as records:
        result = run(resolve_config(raw))
    assert result.gates.all_passed()
    assert _count(records, "could not be detected") == 1
    assert _count(records, "Memory guard (§13)") == 1


def test_undetectable_ram_warns_exactly_once_per_run_voxel_import(
    tmp_path, monkeypatch,
):
    """THE TRAP, second instance: boundaries.geometry == 'voxel_import'
    binds via _stage_voxel_import (pipeline.py) and likewise never calls
    _stage_tessellation -- _apply_runtime_limits is again the only
    resolver/stamp. Must still warn exactly once.

    Config mirrors tests/test_voxel_import.py's own ``_raw``/``_e2e_config``
    fixtures: a plain ``.npy`` bicrystal label field (no optional h5py
    dependency needed), ``grains: {}`` (the grain count is DERIVED from the
    field, not configured -- Rule 17/19)."""
    monkeypatch.setattr(memory_mod, "detect_physical_ram_bytes",
                        lambda: None)

    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[10:] = 1   # bicrystal: x < 10 -> grain 0, x >= 10 -> grain 1
    field_path = tmp_path / "field.npy"
    np.save(field_path, labels)

    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [20.0, 20.0, 20.0],
                "periodic": [True, True, True]},
        "grains": {},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "boundaries": {"geometry": "voxel_import",
                       "voxel_import": {"file": str(field_path)}},
        "output": {"directory": str(tmp_path / "out")},
    }
    with _capture_memory_log() as records:
        result = run(resolve_config(raw))
    assert result.gates.all_passed()
    assert result.tess.n_grains == 2
    assert _count(records, "could not be detected") == 1
    assert _count(records, "Memory guard (§13)") == 1


def test_ram_detectable_emits_exactly_one_effective_budget_info_line(
    tmp_path,
):
    """Normal (RAM-detectable) path: no warning at all, but exactly ONE
    effective-budget INFO line -- covering the other half of the original
    double-resolution defect (the wasted second RAM detection + duplicated
    "Memory guard (§13)" INFO line whenever _stage_tessellation ran)."""
    raw = _raw(output={"directory": str(tmp_path / "out")})
    with _capture_memory_log() as records:
        result = run(resolve_config(raw))
    assert result.gates.all_passed()
    assert _count(records, "could not be detected") == 0
    assert _count(records, "Memory guard (§13)") == 1


# ---------------------------------------------------------------------------
# 11. Adaptive units: a sub-GB limit must never collapse to "0.0 GB".
# ---------------------------------------------------------------------------


def test_adaptive_units_sub_gb_never_reads_zero_point_zero_gb():
    small_bytes = 25_000_000.0   # 25 MB
    formatted = format_bytes_adaptive(small_bytes)
    assert formatted != "0.0 GB"
    assert "MB" in formatted

    msg = build_memory_guard_message(
        "test allocation", small_bytes * 2, small_bytes, "config")
    assert "0.0 GB" not in msg
    assert "MB" in msg


# ---------------------------------------------------------------------------
# 12. Byte-reproducibility: the (machine-dependent) effective budget must
#    NEVER reach a hashed output file -- only config.runtime.memory_limit_gb
#    (the user's own value) is ever serialized. Mirrors tests/
#    test_end_to_end.py::test_byte_identical_rerun_source_date_epoch's
#    "whole output directory, byte-for-byte" methodology.
# ---------------------------------------------------------------------------


def test_effective_budget_preserves_config_and_scientific_outputs(tmp_path, monkeypatch):
    """Machine measurements belong in the summary, never in resolved inputs.

    Real elapsed time and RSS observations can differ, as can the summary's
    genuine manifest checksum. Scientific artifacts and input bytes cannot.
    """
    from tests.test_end_to_end import _normalized

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    root1 = tmp_path / "run1"
    root2 = tmp_path / "run2"
    root1.mkdir()
    root2.mkdir()

    raw = _raw(output={"directory": "./out"},
              runtime={"memory_limit_gb": 9.5})
    monkeypatch.chdir(root1)
    res1 = run(resolve_config(raw))
    monkeypatch.chdir(root2)
    res2 = run(resolve_config(raw))
    assert res1.gates.all_passed() and res2.gates.all_passed()

    out1, out2 = root1 / "out", root2 / "out"
    names1 = {p.name for p in out1.iterdir() if p.is_file()}
    names2 = {p.name for p in out2.iterdir() if p.is_file()}
    assert names1 == names2

    for name in sorted(names1 - {"run.log", "summary.csv", "MANIFEST.txt"}):
        b1 = (out1 / name).read_bytes()
        b2 = (out2 / name).read_bytes()
        assert b1 == b2, f"{name} differs between identical runs"

    assert _normalized(out1 / "summary.csv") == _normalized(out2 / "summary.csv")
    manifests = [(out / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
                 for out in (out1, out2)]
    assert [line for line in manifests[0] if not line.endswith("  summary.csv")] == \
        [line for line in manifests[1] if not line.endswith("  summary.csv")]
    # The ONLY runtime.memory_limit_gb-shaped value anywhere in
    # resolved_config.yaml must be the user's own 9.5 -- never a clamped
    # or RAM-detected number (this run's machine RAM is almost certainly
    # neither exactly 9.5 GB nor absent, so any leak would show up here).
    rc_text = (out1 / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "memory_limit_gb: 9.5" in rc_text
