"""grainsmith pipeline orchestrator (§4.1).

``run(config)`` executes the full stage chain

    crystal → seeding → tessellation → orientation → fill → overlap →
    analysis → write

with optional stages spliced into that backbone when configured: phase
assignment between tessellation and
orientation; MDF-targeting annealing between
orientation and fill; dopant insertion between overlap
and analysis; GB curvature and publication statistics
inside the analysis stage.  ``run()`` returns a
:class:`RunResult`.  Every stage is a pure function of
(inputs, rng-stream); the orchestrator only assembles stages, collects wall
times, and records QA gate results (§9) into the :class:`~grainsmith.qa.QAGates`
registry — construction-enforced gates (G2–G6) raise inside their stage and
are recorded as passed afterwards; evaluator gates (G3-voxel, G7–G10) are
``require``-d here.

Logging (§1, §8.6): verbose 0→WARNING, 1→INFO, 2→DEBUG, 3→DEBUG +
diagnostic dumps into ``<outdir>/diagnostics/``.  A ``run.log`` FileHandler
is attached for the duration of the run.

Determinism (§1): one master seed → named child streams (rng.py); atoms are
sorted by (grain, generation order) before writing; all writers use
shortest-round-trip float formatting — re-running the same config produces
byte-identical outputs up to the embedded timestamps.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grainsmith.config.schema import CurvedBoundaryConfig, RunConfig
from grainsmith.errors import ConfigError
from grainsmith.provenance import (
    ENVIRONMENT_KEYS,
    PROVENANCE_PAYLOAD_SCHEMA,
    environment_record,
    make_provenance,
    run_timestamp_iso,
    source_date_epoch,
)

log = logging.getLogger(__name__)

_VERBOSE_TO_LEVEL: dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.DEBUG,  # same level; diagnostic dumps triggered by verbose >= 3
}

# Sample-direction strings for fiber specs live in orientation.samplers
# (SAMPLE_DIRECTIONS) — shared by the fiber scheme and odf_components.


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CrystalData:
    """Crystal-stage output: everything downstream stages need (§6.1–6.3)."""
    cellpar: Any                 # crystal.cell.CellPar
    A: np.ndarray                # (3,3) cell matrix, COLUMNS = a1,a2,a3 (Å)
    basis: Any                   # crystal.spacegroup.Basis
    sym_quats: np.ndarray        # (Nsym,4) proper point-group rotations
    dataset: dict                # spglib verification subset (gate G2)
    d_nn: float                  # ideal-crystal nearest-neighbor distance (Å)
    rho_atom: float              # ideal-crystal atom density (atoms/Å³)
    density_g_cm3: float         # mass density (NaN if a mass is unknown)


@dataclass
class RunResult:
    """Everything a caller might need after ``run()`` (§5)."""
    config: RunConfig
    seed: int
    outdir: Path
    atoms: Any                       # AtomBlock, sorted by (grain, gen order)
    tess: Any                        # Tessellation
    quats: np.ndarray                # (N,4) grain orientations
    grain_reports: list = field(default_factory=list)
    boundary_reports: list = field(default_factory=list)
    gates: Any = None                # qa.QAGates (all results, §9)
    timings: dict[str, float] = field(default_factory=dict)
    n_generated: int = 0
    n_deleted: int = 0
    d_nn: float = float("nan")
    files: list[Path] = field(default_factory=list)
    phase_of: np.ndarray | None = None   # grain → phase index
    statistics: dict[str, dict[str, Any]] | None = None  # §5 sections
    peak_rss_bytes: float | None = None  # optional total-RSS monitor (memory.py)
    provenance: Any = None           # io.common.Provenance for this run


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: int, outdir: Path | None = None):
    """Configure stdlib logging (§1 verbose map) + optional run.log handler.

    Returns the FileHandler (caller must remove/close it in ``finally``) or
    None when *outdir* is not given.
    """
    level = _VERBOSE_TO_LEVEL.get(verbose, logging.DEBUG)
    log_format = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    fmt = logging.Formatter(log_format, datefmt="%H:%M:%S")
    logging.basicConfig(level=level, format=log_format, datefmt="%H:%M:%S",
                        force=True)
    if outdir is None:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outdir / "run.log", mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)
    return fh


# ---------------------------------------------------------------------------
# Stage: crystal (gate G2)
# ---------------------------------------------------------------------------


def _available_cpus() -> int:
    """Cores available to THIS process, for resolving ``--jobs 0`` (§13).

    ``os.sched_getaffinity`` honours the process's cgroup/affinity mask —
    under a SLURM ``--cpus-per-task`` allocation it returns the allocated
    cores, where ``os.cpu_count()`` reports the node's full core count and
    would oversubscribe the allocation. Platforms without affinity masks
    (macOS, Windows) fall back to ``os.cpu_count()``; both fall back to 1
    when the count is unavailable.
    """
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def _family_from_sg(number: int) -> str:
    """Map ITA space-group number (1–230) to crystal family name.

    Thin wrapper kept for call-site stability; the mapping itself (shared
    with the CIF loader, crystal/cif.py) lives in
    ``crystal.cell.family_from_sg``.
    """
    from grainsmith.crystal.cell import family_from_sg
    return family_from_sg(number)


def _stage_crystal(config: RunConfig) -> CrystalData:
    """Single-phase crystal stage (resolve Rule 21 guarantees 'crystal')."""
    assert config.crystal is not None  # gate G1 (resolve Rule 21)
    return _build_crystal(config.crystal, config.output.lammps.masses)


def _build_crystal(crystal_cfg, masses_cfg) -> CrystalData:
    """Cell matrix → Wyckoff orbit → spglib round-trip (G2) → point-group
    quaternions → derived quantities d_nn, ρ_atom, mass density (§6.1–6.3).

    Takes ONE CrystalConfig so multiphase runs can build a
    CrystalData per phase."""
    from grainsmith.atoms.fill import _compute_d_nn
    from grainsmith.constants import AMU_PER_A3_TO_G_PER_CM3, ATOMIC_MASSES
    from grainsmith.crystal import (
        WyckoffSite,
        cell_matrix,
        expand_wyckoff,
        hall_from_international,
        proper_rotation_quaternions,
        symmetry_ops,
        validate_cellpar,
        verify_spacegroup,
    )
    from grainsmith.crystal.verify import verify_wyckoff_letters

    sg_number = crystal_cfg.space_group.number
    sg_setting = crystal_cfg.space_group.setting
    family = _family_from_sg(sg_number)
    # Trigonal R-groups in the rhombohedral (primitive) axis setting take
    # (a, α) instead of the hexagonal (a, c) — selected via setting: "R".
    if family == "trigonal" and sg_setting is not None \
            and sg_setting.upper() == "R":
        family = "rhombohedral"

    latt = crystal_cfg.lattice
    params = {
        k: v for k, v in {
            "a": latt.a, "b": latt.b, "c": latt.c,
            "alpha": latt.alpha, "beta": latt.beta, "gamma": latt.gamma,
        }.items()
        if v is not None
    }
    cellpar = validate_cellpar(family, params)
    hall = hall_from_international(sg_number, crystal_cfg.space_group.setting)
    rots, trans = symmetry_ops(hall)

    sites = [
        WyckoffSite(element=s.element, coords=s.coords, letter=s.letter)
        for s in crystal_cfg.wyckoff_sites
    ]
    basis = expand_wyckoff(sites, rots, trans)
    dataset = verify_spacegroup(cellpar, basis, sg_number)  # gate G2
    # G2 extension; non-default settings only warn — spglib letters refer
    # to its standardized description (see verify_wyckoff_letters).
    verify_wyckoff_letters(
        [(s.coords, s.letter) for s in crystal_cfg.wyckoff_sites],
        basis, dataset["wyckoffs"],
        strict=sg_setting is None,
    )

    A = cell_matrix(cellpar.a, cellpar.b, cellpar.c,
                    cellpar.alpha, cellpar.beta, cellpar.gamma)
    sym_quats = proper_rotation_quaternions(A, rots)
    d_nn = _compute_d_nn(np.asarray(basis.frac, dtype=np.float64), A)

    v_cell = abs(float(np.linalg.det(A)))
    rho_atom = basis.atoms_per_cell / v_cell

    masses_override = masses_cfg or {}
    mass_cell = 0.0
    density = float("nan")
    try:
        for sp, occ in zip(basis.species, basis.occupancy, strict=True):
            if occ is None:
                mass_cell += masses_override.get(sp) or ATOMIC_MASSES[sp]
            else:
                mass_cell += sum(
                    x * (masses_override.get(sym) or ATOMIC_MASSES[sym])
                    for sym, x in occ.items()
                )
        density = mass_cell * AMU_PER_A3_TO_G_PER_CM3 / v_cell
    except KeyError as exc:
        log.warning("No atomic mass for %s - density reported as NaN "
                    "(add output.lammps.masses).", exc)

    # ASCII-safe log text: Windows consoles often run a non-UTF-8 codepage.
    log.info("crystal: SG %d (%s), %d atoms/cell, d_nn=%.4f A, "
             "rho=%.4f g/cm3, %d proper point-group rotations",
             dataset["number"], dataset["international"],
             basis.atoms_per_cell, d_nn, density, len(sym_quats))
    return CrystalData(cellpar=cellpar, A=A, basis=basis,
                       sym_quats=sym_quats, dataset=dataset, d_nn=d_nn,
                       rho_atom=rho_atom, density_g_cm3=density)


def _validate_crystal(config: RunConfig) -> list[tuple[str, dict[str, Any]]]:
    """G2 dry build for ``grainsmith validate``: one (label, spglib subset)
    per crystal — a single entry, or one per phase."""
    if config.crystal is not None:
        return [("crystal", _stage_crystal(config).dataset)]
    assert config.phases is not None  # gate G1 (resolve Rule 21)
    return [
        (p.name, _build_crystal(p.crystal, config.output.lammps.masses).dataset)
        for p in config.phases
    ]


# ---------------------------------------------------------------------------
# Stage: seeding
# ---------------------------------------------------------------------------


def _resolve_min_seed_distance(config: RunConfig) -> float:
    """min_seed_distance with 'auto' resolved to 1.0·r_ws (§6.4)."""
    from grainsmith.seeding import wigner_seitz_radius

    assert config.grains.number is not None  # gate G1 (resolve Rule 16)
    msd = config.grains.min_seed_distance
    if msd == "auto":
        volume = float(np.prod(np.asarray(config.box.lengths,
                                          dtype=np.float64)))
        msd = wigner_seitz_radius(volume, config.grains.number)
    return float(msd)


def _stage_seeding(config: RunConfig, msd: float,
                   rng: np.random.Generator) -> np.ndarray:
    """RSA + optional Lloyd relaxation (§6.4)."""
    from grainsmith.seeding import seed_grains

    assert config.grains.number is not None  # gate G1 (resolve Rule 16)
    return seed_grains(
        config.grains.number,
        np.asarray(config.box.lengths, dtype=np.float64),
        config.box.periodic,
        rng,
        min_seed_distance=msd,
        lloyd_iterations=config.grains.lloyd_iterations,
    )


def _apply_runtime_limits(
    tess: Any, config: RunConfig,
    memory_budget: tuple[float, str, float | None] | None = None,
) -> Any:
    """Belt-and-braces POST-stamp of the run's §13 per-allocation budget
    onto the tessellation (and any secondary tessellation object it wraps
    or was fitted from) -- a safety net, NOT the primary mechanism.

    For the four CURVED backends (WeightedTessellation,
    AnisotropicTessellation, WarpTessellation,
    PerturbedDistanceTessellation) the dominant allocation -- the voxel
    grid -- is built INSIDE ``__init__``, which runs during
    ``_stage_tessellation``, strictly BEFORE this function ever runs (it is
    only called later in ``run()``). Those four constructors therefore
    receive ``memory_limit_bytes``/``memory_limit_source`` directly as
    keyword arguments from ``_stage_tessellation``, so their
    construction-time guard already sees the correct, RAM-clamped value --
    this function's stamp lands too late to affect a guard check that
    already ran during construction. What this function DOES cover: the
    backends that do NOT guard at construction (``FlatTessellation`` /
    ``PowerTessellation``, and ``VoxelTessellation``, whose guarded
    allocations -- ``fill_grain``'s lattice grid, ``voxel_import``'s margin
    EDT -- happen later, in the fill/analysis stages this call precedes)
    and re-stamping as a robustness net against future refactors of the
    tessellation-binding branches in ``run()``.

    ``memory_budget`` : the ``(limit_bytes, source, ram_bytes)`` tuple
    ``memory.resolve_memory_budget`` returns. ``run()`` resolves this
    EXACTLY ONCE per run, before any tessellation-binding branch
    (voxel_import / single-crystal / flat-or-curved) constructs anything,
    and passes it here -- this is in fact the ONLY resolution/stamp the
    voxel_import and single-crystal paths ever get, since neither goes
    through ``_stage_tessellation``. When omitted (``None``, the default),
    this function resolves the budget itself for a standalone caller
    outside ``run()`` (e.g. a direct test); ``run()`` itself never hits
    that branch, which is what keeps ``resolve_memory_budget`` -- and
    therefore its "RAM could not be detected" warning -- to exactly one
    call per run regardless of path.

    The limit lives on the INSTANCE (``Tessellation.memory_limit_bytes``/
    ``memory_limit_source``, see tessellation/base.py) rather than a module
    global specifically so it survives being pickled to a
    ProcessPoolExecutor 'spawn' worker (atoms/fill.py's ``fill_grains``,
    ``jobs > 1``) -- a worker re-imports grainsmith modules from scratch
    under spawn and would read the UNPATCHED module-level default.
    """
    if memory_budget is None:
        from grainsmith.memory import log_effective_budget, resolve_memory_budget

        memory_budget = resolve_memory_budget(config)
        log_effective_budget(*memory_budget, config.runtime.memory_limit_gb)
    limit_bytes, limit_source, _ram_bytes = memory_budget
    tess.memory_limit_bytes = limit_bytes
    tess.memory_limit_source = limit_source
    # WarpTessellation wraps a `_base` tessellation that builds its OWN
    # voxel grid (its `grain_of`/`owns`/`margin` delegate to it, but a
    # direct build_voxel_grid(base, ...) call -- e.g. from a future
    # analysis path -- would read the base's OWN memory_limit_bytes).
    base = getattr(tess, "_base", None)
    if base is not None:
        base.memory_limit_bytes = limit_bytes
        base.memory_limit_source = limit_source
    return tess


# ---------------------------------------------------------------------------
# Stage: tessellation (flat or curved backend; gates G3–G6 at construction)
# ---------------------------------------------------------------------------


def _curved_cfg(config: RunConfig) -> CurvedBoundaryConfig:
    return config.boundaries.curved or CurvedBoundaryConfig()


def _stage_tessellation(
    config: RunConfig,
    seeds: np.ndarray,
    msd: float,
    rng: np.random.Generator,
    rng_sizes: np.random.Generator | None = None,
    memory_budget: tuple[float, str, float | None] | None = None,
):
    """Backend dispatch per boundaries.geometry/method (§3.3, §6.5–6.7).

    A present ``grains.size_distribution`` replaces the flat
    backend (or the warp base) with a volume-fitted power diagram.
    Returns ``(tessellation, SDOTResult | None)``.

    §13: for every CURVED backend (WeightedTessellation,
    AnisotropicTessellation, WarpTessellation,
    PerturbedDistanceTessellation) the dominant allocation -- the voxel
    grid -- is built INSIDE ``__init__`` (the connectivity check, or
    perturbed_distance's repair-and-build-override), which runs HERE,
    strictly BEFORE ``pipeline._apply_runtime_limits`` gets a chance to
    stamp the instance post-construction. The budget is therefore needed
    up front and threaded through to every curved constructor (including
    the warp `base`) as keyword arguments, so their construction-time
    guard sees the real, RAM-clamped per-run value instead of the class
    default.

    ``memory_budget``: the ``(limit_bytes, source, ram_bytes)`` tuple
    ``memory.resolve_memory_budget`` returns. ``run()`` resolves this
    EXACTLY ONCE per run -- before calling this function -- and passes it
    here; this function no longer resolves it itself in that case, which
    is what keeps ``resolve_memory_budget`` (and therefore its "RAM could
    not be detected" warning, and the effective-budget INFO line) to
    exactly one call per run. When omitted (``None``, the default) this
    function resolves (and logs) the budget itself, for a standalone
    caller outside ``run()`` (e.g. a direct test) -- ``run()`` itself
    never hits that branch.
    """
    from grainsmith.memory import log_effective_budget, resolve_memory_budget
    from grainsmith.tessellation import (
        AnisotropicTessellation,
        FlatTessellation,
        WarpTessellation,
        WeightedTessellation,
        fit_power_weights,
        sample_target_volumes,
    )

    L = np.asarray(config.box.lengths, dtype=np.float64)
    periodic = config.box.periodic

    if memory_budget is None:
        memory_budget = resolve_memory_budget(config)
        log_effective_budget(*memory_budget, config.runtime.memory_limit_gb)
    limit_bytes, limit_source, _ram_bytes = memory_budget

    sdot = None
    size_dist = config.grains.size_distribution
    if size_dist is not None:
        assert rng_sizes is not None  # pipeline always passes the stream
        # number is required with size_distribution (resolve Rules 10/16)
        assert config.grains.number is not None
        targets = sample_target_volumes(
            size_dist.type, config.grains.number, float(np.prod(L)),
            rng_sizes, sigma_log=size_dist.sigma_log,
            volumes=size_dist.volumes,
        )
        sdot = fit_power_weights(
            seeds, L, periodic, targets,
            vol_tol=size_dist.vol_tol, max_iter=size_dist.max_iter,
            centroidal_iterations=size_dist.centroidal_iterations,
        )
        log.info("sdot: %d Newton iterations, max rel volume error %.3e",
                 sdot.iterations, sdot.max_rel_error)

    if config.boundaries.geometry == "flat":
        if sdot is not None:
            return sdot.tess, sdot
        flat_tess = FlatTessellation(seeds, L, periodic)
        # FlatTessellation never guards at construction (no voxel grid is
        # built in __init__), so this stamp is not fixing a defect here --
        # it's stamped anyway for consistency with the curved backends
        # below (_apply_runtime_limits would otherwise be the first to do
        # it, later, in run()).
        flat_tess.memory_limit_bytes = limit_bytes
        flat_tess.memory_limit_source = limit_source
        return flat_tess, None

    curved = _curved_cfg(config)

    def weighted(connectivity: bool) -> Any:
        return WeightedTessellation(
            seeds, L, periodic, sigma_w=curved.weight_sigma, rng=rng,
            min_seed_distance=msd, connectivity_check=connectivity,
            memory_limit_bytes=limit_bytes, memory_limit_source=limit_source,
        )

    def anisotropic(connectivity: bool) -> Any:
        lo, hi = curved.aspect_ratio_range
        return AnisotropicTessellation(
            seeds, L, periodic,
            aspect_ratio_range=(lo, hi), rng=rng,
            connectivity_check=connectivity,
            memory_limit_bytes=limit_bytes, memory_limit_source=limit_source,
        )

    if curved.method == "additive_weights":
        return weighted(True), None
    if curved.method == "anisotropic":
        return anisotropic(True), None
    if curved.method == "perturbed_distance":
        from grainsmith.tessellation.perturbed import PerturbedDistanceTessellation

        return PerturbedDistanceTessellation(
            seeds, L, periodic,
            amplitude=curved.amplitude,
            min_seed_distance=msd,
            rng=rng,
            connectivity_check=True,
            spectrum=curved.spectrum,
            hurst=curved.hurst,
            correlation_length=curved.correlation_length,
            l_min=curved.l_min,
            l_max=curved.l_max,
            amplitude_convention=curved.amplitude_convention,
            reference_wavelength=curved.reference_wavelength,
            memory_limit_bytes=limit_bytes, memory_limit_source=limit_source,
        ), None

    # warp: connectivity (G5) is checked on the WARPED geometry only — the
    # base is an intermediate object whose grain shapes never reach outputs.
    base: Any
    if sdot is not None:
        base = sdot.tess          # volume-fitted power base (never guards
        # at construction; stamped anyway, for consistency).
        base.memory_limit_bytes = limit_bytes
        base.memory_limit_source = limit_source
    elif curved.base == "flat":
        base = FlatTessellation(seeds, L, periodic)  # never guards either
        base.memory_limit_bytes = limit_bytes
        base.memory_limit_source = limit_source
    elif curved.base == "additive_weights":
        base = weighted(False)
    else:
        base = anisotropic(False)
    return WarpTessellation(
        base, L, periodic,
        amplitude=curved.amplitude,
        correlation_length=curved.correlation_length,
        min_seed_distance=msd,
        rng=rng,
        connectivity_check=True,
        spectrum=curved.spectrum,
        hurst=curved.hurst,
        l_min=curved.l_min,
        l_max=curved.l_max,
        memory_limit_bytes=limit_bytes, memory_limit_source=limit_source,
    ), sdot


# ---------------------------------------------------------------------------
# Stage: voxel import (replaces seeding + tessellation)
# ---------------------------------------------------------------------------


def _stage_voxel_import(config: RunConfig):
    """Load the imported label field and wrap it as a VoxelTessellation.
    Returns ``(tess, relabel_map | None,
    imported_quats | None)``; the grain count is DERIVED from the field
    and cross-checked against grains.number when that is provided."""
    from grainsmith.tessellation.voxel_import import (
        VoxelTessellation,
        imported_orientations,
        load_label_field,
    )

    vi = config.boundaries.voxel_import
    assert vi is not None  # gate G1 (resolve Rule 17)
    labels, relabel_map, eulers = load_label_field(
        vi.file, dataset=vi.dataset, relabel=vi.relabel,
        euler_dataset=vi.euler_dataset,
    )
    tess = VoxelTessellation(
        labels, np.asarray(config.box.lengths, dtype=np.float64),
        config.box.periodic,
    )
    if (config.grains.number is not None
            and config.grains.number != tess.n_grains):
        raise ConfigError(
            f"grains.number={config.grains.number} does not match the "
            f"{tess.n_grains} grains of the imported label field "
            f"{vi.file!r} - fix or omit grains.number (it is derived for "
            "voxel_import).")
    quats = None
    if eulers is not None and config.orientation.scheme == "imported":
        quats = imported_orientations(eulers, degrees=vi.euler_degrees)
    log.info("voxel_import: %s -> %s voxels, %d grains%s",
             vi.file, "x".join(str(s) for s in labels.shape),
             tess.n_grains,
             ", per-grain orientations loaded" if quats is not None else "")
    return tess, relabel_map, quats


# ---------------------------------------------------------------------------
# Stage: orientation
# ---------------------------------------------------------------------------


def _fixed_spec_dict(spec_model) -> dict:
    """OrientationFixedConfig → {key: value} with exactly one key (§7)."""
    spec = spec_model.model_dump(exclude_none=True)
    if len(spec) != 1:
        raise ConfigError(
            "orientation fixed spec must set exactly one of hkl_uvw, "
            f"euler_bunge_deg, axis_angle, quaternion (got {sorted(spec)})."
        )
    return spec


def _stage_orientation(config: RunConfig, n: int, A: np.ndarray,
                       rng: np.random.Generator,
                       imported_quats: np.ndarray | None = None,
                       grain_volumes: np.ndarray | None = None,
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-grain quaternion assignment per orientation.scheme (§6.10).

    Returns ``(quats, component_of)``: ``component_of[i]`` is the texture
    component index grain ``i`` drew, and is ``None`` for every scheme
    except ``odf_components`` (gate G25's input -- the sampler's EXACT
    assignment, returned by ``odf_components(return_component_index=True)``,
    never a post-hoc nearest-centre reconstruction).

    *grain_volumes*, when given, is forwarded (together with
    ``config.orientation.component_weight_basis``) into ``odf_components``
    -- required by that sampler when the basis is "volume"
    (``orientation.odf.volume_balanced_partition`` needs per-grain volumes
    to match the configured weights by VOLUME rather than by grain count);
    ignored by every other scheme. ``run()`` computes it ONCE, from the
    freshly-primed shared analysis voxel grid, before this stage runs --
    see the priming block's comment at the call site.
    """
    from grainsmith.orientation import (
        SAMPLE_DIRECTIONS,
        fiber_texture,
        fixed_orientation,
        from_list,
        odf_components,
        random_uniform,
    )

    o = config.orientation
    if o.scheme == "imported":
        # Per-grain Euler angles from the voxel_import file
        # (resolve Rule 19 guarantees the dataset was configured).
        if imported_quats is None:
            raise ConfigError(
                "orientation.scheme 'imported' but no orientations were "
                "loaded from the voxel_import file.")
        if len(imported_quats) != n:
            raise ConfigError(
                f"imported euler_dataset provides {len(imported_quats)} "
                f"orientations for {n} grains.")
        return imported_quats, None

    if o.scheme == "random_uniform":
        return random_uniform(n, rng), None

    if o.scheme == "fixed":
        if o.fixed is None:
            raise ConfigError(
                "orientation.scheme is 'fixed' but the 'fixed' block is "
                "missing (§7).")
        return fixed_orientation(n, _fixed_spec_dict(o.fixed), A), None

    if o.scheme == "fiber":
        if o.fiber is None:
            raise ConfigError(
                "orientation.scheme is 'fiber' but the 'fiber' block is "
                "missing (§7).")
        direction = SAMPLE_DIRECTIONS.get(o.fiber.sample_direction)
        if direction is None:
            raise ConfigError(
                f"orientation.fiber.sample_direction must be one of "
                f"{sorted(SAMPLE_DIRECTIONS)}, "
                f"got {o.fiber.sample_direction!r}.")
        return fiber_texture(
            n, np.asarray(o.fiber.crystal_axis, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            o.fiber.spread_deg, rng, A,
        ), None

    if o.scheme == "odf_components":
        assert o.components  # gate G1 (resolve Rule 13)
        sampled = odf_components(
            n, [c.model_dump() for c in o.components], rng, A,
            return_component_index=True,
            grain_volumes=grain_volumes,
            weight_basis=o.component_weight_basis)
        assert isinstance(sampled, tuple)
        return sampled

    # from_list (cross-field Rule 3: length check happens here)
    if o.from_list is None or len(o.from_list) != n:
        got = 0 if o.from_list is None else len(o.from_list)
        raise ConfigError(
            f"orientation.from_list must contain exactly grains.number="
            f"{n} entries, got {got}.")
    specs = [_fixed_spec_dict(entry) for entry in o.from_list]
    return from_list(specs, A), None


# ---------------------------------------------------------------------------
# Stage: MDF targeting (gate G12 re-measured in analysis)
# ---------------------------------------------------------------------------


def _stage_mdf(config: RunConfig, tess, quats: np.ndarray,
               sym_quats: np.ndarray, rng: np.random.Generator,
               grain_volumes: np.ndarray | None):
    """Orientation-assignment annealing toward orientation.mdf_target.
    Returns ``(quats, MDFResult | None)`` — only the grain → orientation
    assignment is permuted, which leaves the NUMBER-weighted discrete
    orientation distribution invariant by construction; the VOLUME-weighted
    ODF is NOT invariant in general (see orientation/mdf.py, orientation/
    odf.py).

    *grain_volumes* is the array ``run()`` computed ONCE, right after
    priming the shared analysis voxel grid and BEFORE the orientation
    stage (hoisted there so ``component_weight_basis == "volume"`` has
    volumes available too) -- this stage no longer computes its own copy.
    It is only read when ``orientation.mdf_target`` is configured (the
    early return below), and ``run()``'s ``need_vols`` condition guarantees
    a non-``None`` array whenever that is the case, so this stage never
    dereferences ``None``."""
    mdf_cfg = config.orientation.mdf_target
    if mdf_cfg is None:
        return quats, None
    # `compute_pair_areas` is an alias, not a rename of the imported
    # function itself: the LOCAL variable below is named `pair_areas` (see
    # comment at its assignment), and Python would otherwise treat that
    # name as local for the ENTIRE function body -- including this earlier
    # call site -- raising UnboundLocalError.
    from grainsmith.analysis.boundaries import pair_areas as compute_pair_areas
    from grainsmith.orientation.mdf import (
        anneal_assignment,
        build_bins_and_target,
    )

    L = np.asarray(config.box.lengths, dtype=np.float64)
    areas_map = compute_pair_areas(tess, L, config.box.periodic)
    pairs = sorted(areas_map)
    # Renamed from `areas` for the same reason `anneal_assignment`'s own
    # parameter was renamed to `pair_areas`: boundary AREA and grain
    # VOLUME are exactly the ambiguity this revision fixes, so the two
    # must never share a generic name in the same function.
    pair_areas = np.array([areas_map[p] for p in pairs], dtype=np.float64)
    edges, target, ref = build_bins_and_target(
        mdf_cfg, sym_quats, config.analysis.mdf_bins)

    # WHY THIS IS THE RIGHT ARRAY: `grain_volumes` (the parameter) was
    # computed by `run()` right after the shared analysis voxel grid was
    # primed for curved backends with an explicit `analysis.voxel_grid`,
    # and BEFORE the orientation stage -- strictly after the tessellation
    # became FINAL (warp and self-affine geometry are applied at
    # tessellation CONSTRUCTION time, `_stage_tessellation`, before either
    # priming or this stage ever run). It is therefore EXACTLY the array
    # the analysis stage will later compute (`pipeline.py`'s
    # `volumes = grain_volumes(tess, L)` for flat / `vg.volumes()` for
    # curved): both go through the SAME cached `tess._voxel` for curved
    # backends (`analysis.grains.get_voxel_grid` reads `tess.voxel_grid`,
    # which returns that cache) and through the exact polyhedral cells for
    # flat -- nothing between the priming call and the analysis stage's own
    # mutates the tessellation. Pinned by
    # tests/test_odf_gates.py::test_stage_mdf_volumes_match_analysis_*.
    assert grain_volumes is not None  # run()'s need_vols is True here

    res = anneal_assignment(
        quats, pairs, pair_areas, edges, target, ref, sym_quats,
        mdf_cfg.annealing_steps, mdf_cfg.t0, mdf_cfg.cooling, rng,
        grain_volumes=grain_volumes, odf_drift_max=mdf_cfg.odf_drift_max,
    )
    return quats[res.permutation], res


# ---------------------------------------------------------------------------
# Stage: fill + overlap (gates G7–G9 evaluated in run())
# ---------------------------------------------------------------------------


def _stage_fill(config: RunConfig, tess, crystals: list[CrystalData],
                quats: np.ndarray, rng_bundle, jobs: int = 1,
                phase_of: np.ndarray | None = None):
    """Per-grain compact-home-cell lattice fill (§6.8), optionally
    process-parallel (§13).  Each grain draws from its own deterministic
    occupancy stream, so the result is identical for every *jobs* value.
    Multiphase runs build each grain with ITS phase's crystal."""
    from grainsmith.atoms.fill import AtomBlock, fill_grains, fill_grains_phases

    # box.cells: L is only consulted for non-periodic-axis clipping in
    # fill_grain, and box.cells REQUIRES box.periodic all true (Rule 28),
    # so that clip is never exercised — diag(H) is a safe stand-in (it is
    # also exactly what np.prod(L) needs to equal det(H) elsewhere).
    L = (np.diag(np.asarray(config.box.resolved_h, dtype=np.float64))
         if config.box.cells is not None
         else np.asarray(config.box.lengths, dtype=np.float64))
    periodic = config.box.periodic
    n = tess.n_grains   # == grains.number; derived for voxel_import
    a_clip = 0.0
    if (config.boundaries.geometry == "curved"
            and _curved_cfg(config).method == "warp"):
        # owns() is evaluated on the warped geometry, but the lattice
        # enumeration radius must cover the pre-image displacement.
        a_clip = _resolve_min_seed_distance(config) / 4.0

    if phase_of is None:
        crystal = crystals[0]
        frac = np.asarray(crystal.basis.frac, dtype=np.float64)
        blocks = fill_grains(
            tess, frac, crystal.basis.species, crystal.basis.occupancy,
            crystal.A, quats, L, periodic,
            rngs=rng_bundle.occupancy_streams(n),
            store_margin=False, a_clip=a_clip, jobs=jobs,
        )
    else:
        blocks = fill_grains_phases(
            tess, phase_of,
            [np.asarray(c.basis.frac, dtype=np.float64) for c in crystals],
            [c.basis.species for c in crystals],
            [c.basis.occupancy for c in crystals],
            [c.A for c in crystals],
            quats, L, periodic,
            rngs=rng_bundle.occupancy_streams(n),
            store_margin=False, a_clip=a_clip, jobs=jobs,
        )
    for i, block in enumerate(blocks):
        if len(block) == 0:
            log.warning("grain %d received 0 atoms (very small cell?)", i)
    atoms = AtomBlock.concatenate(blocks)
    log.info("fill: %d atoms generated for %d grains%s",
             len(atoms), n,
             f" ({jobs} worker processes)" if jobs > 1 else "")
    return atoms


def _stage_overlap(config: RunConfig, atoms, tess, d_nn: float,
                   cell_matrix: np.ndarray | None = None, jobs: int = 1):
    """GB overlap removal (§6.9).  Returns (atoms, ledger|None, cutoff|None).

    *cell_matrix*, when given (``box.cells``), routes the periodic
    neighbor search through the general triclinic path (see
    ``atoms.overlap.remove_overlaps``).
    """
    from grainsmith.atoms.overlap import remove_overlaps, resolve_cutoff

    ov = config.boundaries.overlap_removal
    if not ov.enabled:
        log.info("overlap removal disabled")
        return atoms, None, None
    cutoff = resolve_cutoff(ov.cutoff, d_nn=d_nn)
    L = (np.diag(cell_matrix) if cell_matrix is not None
         else np.asarray(config.box.lengths, dtype=np.float64))
    atoms, ledger = remove_overlaps(
        atoms, tess, cutoff, ov.policy, config.box.periodic, L,
        cell_matrix=cell_matrix, jobs=jobs,
    )
    log.info("overlap: %d atoms deleted (cutoff %.4g A, policy %s)",
             ledger.total_deleted, cutoff, ov.policy)
    return atoms, ledger, cutoff


def _sort_by_grain(atoms):
    """Stable sort by grain id — (grain, generation order), §8.1."""
    from grainsmith.atoms.fill import AtomBlock

    order = np.argsort(atoms.grain, kind="stable")
    return AtomBlock(
        pos=atoms.pos[order],
        species=atoms.species[order],
        grain=atoms.grain[order],
        gb_margin=atoms.gb_margin[order] if atoms.gb_margin is not None
                  else None,
    )


def _attach_margins(atoms, tess, n_grains: int):
    """Recompute gb_margin on the FINAL atom set (analysis.per_atom_margin).

    Computed after overlap removal so midpoint-merged positions carry
    correct margins.

    Relies on atoms being sorted by grain id (guaranteed by _sort_by_grain
    immediately before this call in the pipeline).  Uses np.searchsorted so
    the total cost is O(n_atoms) rather than O(n_grains * n_atoms).
    """
    margins = np.empty(len(atoms), dtype=np.float64)
    bounds = np.searchsorted(atoms.grain, np.arange(n_grains + 1))
    for i in range(n_grains):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if e > s:
            margins[s:e] = tess.margin(atoms.pos[s:e], i)
    atoms.gb_margin = margins
    return atoms


# ---------------------------------------------------------------------------
# Stage: analysis (gates G3-voxel, G8, G9)
# ---------------------------------------------------------------------------


def _analysis_voxel_grid(config: RunConfig, tess):
    """Voxel grid for curved-geometry analysis (explicit size or reuse).

    For an explicit ``analysis.voxel_grid`` on a curved backend the grid is
    built once and cached onto the tessellation (``tess._voxel``), so EVERY
    downstream consumer — grain volumes, boundary geometry (analyze_boundaries'
    internal ``get_voxel_grid``), statistics, and the MDF area weights
    (``pair_areas``) — measures on the SAME resolution.  Without this,
    boundaries.csv would silently use the tessellation's own 'auto' grid while
    grains.csv/statistics.csv use the explicit one (a cross-file desync whenever
    the user overrides ``analysis.voxel_grid``).
    """
    from grainsmith.analysis.grains import get_voxel_grid
    from grainsmith.tessellation.voxel import build_voxel_grid, voxel_grid_shape

    L = np.asarray(config.box.lengths, dtype=np.float64)
    if isinstance(config.analysis.voxel_grid, int):
        # Only curved backends carry a ``_voxel`` slot; flat/single use exact
        # geometry and never share a grid, so build fresh for them.
        if hasattr(tess, "_voxel"):
            requested_shape = voxel_grid_shape(L, config.analysis.voxel_grid)
            if tess._voxel is None or tess._voxel.shape != requested_shape:
                tess._voxel = build_voxel_grid(
                    tess, L, config.analysis.voxel_grid)
            tess._voxel_is_explicit = True
            return tess._voxel
        return build_voxel_grid(tess, L, config.analysis.voxel_grid)
    return get_voxel_grid(tess, L)


# ---------------------------------------------------------------------------
# Stage: write (gate G10)
# ---------------------------------------------------------------------------


def _version_rows() -> list[tuple[str, str]]:
    """``(name, version)`` for the `versions` section of every summary.

    One helper, two call sites: ``_summary_rows`` (success) and ``run``'s
    failure handler. They used to build this list independently, which is how
    the failure summary ended up with NO versions section at all -- the single
    most useful thing to know when diagnosing a failed run.

    ``grainsmith`` itself is in the list. ``meta,grainsmith_version`` carries
    the same string, but a reader scanning the `versions` section for "which
    grainsmith produced this?" should find it there too -- and
    microstructure.json's own `versions` block has always included it, so
    summary.csv was the odd one out.
    """
    import scipy
    import spglib

    from grainsmith import __version__

    return [
        ("grainsmith", __version__),
        ("python", sys.version.split()[0]),
        ("numpy", np.__version__),
        ("scipy", scipy.__version__),
        ("spglib", spglib.__version__),
    ]


def _summary_rows(
    config: RunConfig, provenance, crystals: list[CrystalData], tess,
    msd: float, cutoff, type_map: dict[str, int],
    n_generated: int, n_deleted: int, atoms,
    timings: dict[str, float], gates, sdot_res=None, mdf_res=None,
    relabel_map: dict[int, int] | None = None,
    nominal: dict[str, float] | None = None,
    phase_of: np.ndarray | None = None,
    phase_achieved: np.ndarray | None = None,
    curvature_rows: list | None = None,
    doping_rows: list | None = None,
    environment: dict[str, str] | None = None,
    statistics: dict[str, dict[str, Any]] | None = None,
    memory_report: dict[str, Any] | None = None,
    analysis_grid=None,
    boundary_reports: list | None = None,
    mesh_status: str | None = None,
) -> list[tuple[str, str, Any]]:
    """Assemble the §8.3 summary.csv long-format rows."""
    import grainsmith
    from grainsmith.constants import AMU_PER_A3_TO_G_PER_CM3
    from grainsmith.io.lammps import _resolve_masses
    from grainsmith.io.reports import summary_mapping_rows
    from grainsmith.qa import nominal_composition
    from grainsmith.tessellation.flat import FlatTessellation
    from grainsmith.tessellation.warp import WarpTessellation

    crystal = crystals[0]

    rows: list[tuple[str, str, Any]] = []
    # Completeness as a POSITIVE assertion. The failure path writes
    # ("meta", "status", "failed") into summary.failed.csv, so "failed" was
    # stated but "complete" only ever inferred from the absence of that file
    # -- and a reader holding just summary.csv could not tell a finished run
    # from one whose writer was killed mid-directory.
    rows.append(("meta", "status", "complete"))
    rows.append(("meta", "title", config.meta.title))
    rows.append(("meta", "grainsmith_version", grainsmith.__version__))
    rows.append(("meta", "timestamp_utc", provenance.timestamp_iso))
    # Full 64-hex, not the 12-hex header display form: this row is the
    # machine-readable record (BREAKING vs 1.1.0, which wrote 12 hex here).
    rows.append(("meta", "config_sha256", provenance.config_sha256))
    rows.append(("meta", "provenance_sha256", provenance.provenance_sha256))
    rows.append(("meta", "provenance_payload_schema",
                 PROVENANCE_PAYLOAD_SCHEMA))
    _epoch = source_date_epoch()
    rows.append(("meta", "source_date_epoch",
                 "unset" if _epoch is None else _epoch))
    rows.append(("meta", "timings_status", "measured"))
    rows.append(("meta", "timings_log", "run.log"))
    rows.append(("meta", "timing_stages", " ".join(timings)))
    rows.append(("meta", "timings_scope",
                 "pipeline through primary outputs; excludes summary, "
                 "manifest hashing and shutdown; curvature is part of analysis. "
                 "timings,total_s is the WHOLE run and sits in the same "
                 "section as the per-stage rows, so summing timings,* "
                 "double-counts it; the stages also do not sum to total_s "
                 "(untimed work between them)"))

    if config.box.cells is not None:
        H = np.asarray(config.box.resolved_h, dtype=np.float64)
        material_lengths = np.diag(H)
        rows.append(("box", "mode", "cells (triclinic lattice-multiple)"))
        rows.append(("box", "cells", " ".join(str(v) for v in config.box.cells)))
        rows.append(("box", "cell_matrix_a_A",
                     " ".join(str(v) for v in H[:, 0])))
        rows.append(("box", "cell_matrix_b_A",
                     " ".join(str(v) for v in H[:, 1])))
        rows.append(("box", "cell_matrix_c_A",
                     " ".join(str(v) for v in H[:, 2])))
        rows.append(("box", "tilt_xy_xz_yz_A",
                     f"{H[0, 1]} {H[0, 2]} {H[1, 2]}"))
    else:
        assert config.box.lengths is not None  # Rule 28: lengths XOR cells
        L = config.box.lengths
        material_lengths = np.asarray(L, dtype=np.float64)
        rows.append(("box", "mode", "lengths (orthogonal)"))
        rows.append(("box", "lengths_A", " ".join(str(v) for v in L)))
    rows.append(("box", "periodic",
                 " ".join(str(p) for p in config.box.periodic)))
    rows.append(("box", "vacuum_A", config.box.vacuum))
    padding = np.where(config.box.periodic, 0.0, config.box.vacuum)
    export_lengths = material_lengths + padding
    material_volume = float(np.prod(material_lengths))
    export_volume = float(np.prod(export_lengths))
    rows.append(("box", "vacuum_per_axis_A",
                 " ".join(str(value) for value in padding)))
    rows.append(("box", "export_shift_A",
                 " ".join(str(value) for value in padding / 2.0)))
    rows.append(("box", "export_lengths_A",
                 " ".join(str(value) for value in export_lengths)))
    rows.append(("box", "material_volume_A3", material_volume))
    rows.append(("box", "export_volume_A3", export_volume))
    rows.append(("box", "vacuum_volume_A3", export_volume - material_volume))
    rows.append(("box", "vacuum_fraction",
                 (export_volume - material_volume) / export_volume))

    rows.append(("grains", "number", tess.n_grains))
    rows.append(("grains", "min_seed_distance_A", msd))
    rows.append(("grains", "lloyd_iterations", config.grains.lloyd_iterations))

    rows.append(("seed", "mode", config.seed.mode))
    rows.append(("seed", "value", provenance.seed))
    for setting in ("csl", "gb_character", "gb_curvature", "statistics", "voxel_grid"):
        rows.append(("analysis", setting, getattr(config.analysis, setting)))

    if config.crystal is not None:
        ds = crystal.dataset
        assert config.crystal.space_group is not None  # Rule 27 resolved this
        rows.append(("crystal", "requested_sg",
                     config.crystal.space_group.number))
        rows.append(("crystal", "detected_sg", ds["number"]))
        rows.append(("crystal", "international", ds["international"]))
        rows.append(("crystal", "hall_number", ds["hall_number"]))
        rows.append(("crystal", "pointgroup", ds["pointgroup"]))
        rows.append(("crystal", "wyckoff_letters", " ".join(ds["wyckoffs"])))
        rows.append(("crystal", "atoms_per_cell",
                     crystal.basis.atoms_per_cell))
        rows.append(("crystal", "density_g_cm3", crystal.density_g_cm3))
        rows.append(("crystal", "d_nn_A", crystal.d_nn))
        if config.crystal.cif is not None:
            rows.append(("crystal", "cif_file", config.crystal.cif.file))
            rows.append(("crystal", "cif_symprec", config.crystal.cif.symprec))
    else:
        # Multiphase: one summary section per phase.
        assert config.phases is not None  # gate G1 (resolve Rule 21)
        for p, (pc, c) in enumerate(zip(config.phases, crystals,
                                        strict=True)):
            assert pc.crystal.space_group is not None  # Rule 27 resolved this
            sec = f"phase:{pc.name}"
            ds = c.dataset
            rows.append((sec, "requested_sg", pc.crystal.space_group.number))
            rows.append((sec, "detected_sg", ds["number"]))
            rows.append((sec, "international", ds["international"]))
            rows.append((sec, "hall_number", ds["hall_number"]))
            rows.append((sec, "pointgroup", ds["pointgroup"]))
            rows.append((sec, "wyckoff_letters", " ".join(ds["wyckoffs"])))
            rows.append((sec, "atoms_per_cell", c.basis.atoms_per_cell))
            rows.append((sec, "density_g_cm3", c.density_g_cm3))
            rows.append((sec, "d_nn_A", c.d_nn))
            rows.append((sec, "fraction_target", pc.fraction))
            if pc.crystal.cif is not None:
                rows.append((sec, "cif_file", pc.crystal.cif.file))
                rows.append((sec, "cif_symprec", pc.crystal.cif.symprec))
            if phase_achieved is not None:
                rows.append((sec, "fraction_achieved",
                             float(phase_achieved[p])))
            if phase_of is not None:
                rows.append((sec, "n_grains",
                             int(np.sum(np.asarray(phase_of) == p))))

    rows.append(("tessellation", "backend", type(tess).__name__))
    if config.boundaries.geometry == "voxel_import":
        vi = config.boundaries.voxel_import
        assert vi is not None  # gate G1 (resolve Rule 17)
        rows.append(("voxel_import", "file", vi.file))
        if vi.dataset is not None:
            rows.append(("voxel_import", "dataset", vi.dataset))
        rows.append(("voxel_import", "grid_shape",
                     " ".join(str(s) for s in tess.voxel_grid.shape)))
        rows.append(("voxel_import", "n_grains_derived", tess.n_grains))
        rows.append(("voxel_import", "strict_connectivity",
                     vi.strict_connectivity))
        rows.append(("voxel_import", "relabeled", relabel_map is not None))
        if relabel_map is not None:
            rows.append(("voxel_import", "relabel_map",
                         " ".join(f"{orig}->{new}"
                                  for orig, new in relabel_map.items())))
    if config.boundaries.geometry == "curved":
        curved = _curved_cfg(config)
        rows.append(("tessellation", "method", curved.method))
        if curved.method == "warp":
            # warp accepts spectrum: gaussian only (Rule 8a in
            # config/resolve.py + the WarpTessellation.__init__ guard
            # reject self_affine before this point is ever reached), so
            # gaussian + correlation_length is the complete row set --
            # hurst/l_min/l_max/hurst_estimated apply only to the
            # self_affine spectrum, which this branch never sees.
            rows.append(("tessellation", "base", curved.base))
            rows.append(("tessellation", "amplitude_A", curved.amplitude))
            rows.append(("tessellation", "spectrum", curved.spectrum))
            rows.append(("tessellation", "correlation_length_A",
                         curved.correlation_length))
            rows.append(("tessellation", "a_clip_A", msd / 4.0))
            if isinstance(tess, WarpTessellation):
                rows.append(("tessellation", "max_grad_u", tess.grad_max))
        elif curved.method == "additive_weights":
            rows.append(("tessellation", "weight_sigma_A",
                         curved.weight_sigma))
        elif curved.method == "perturbed_distance":
            from grainsmith.constants import ETA_CLIP as _ETA_CLIP
            from grainsmith.constants import PERTURBED_DISTANCE_SAFETY as _PDS

            # amplitude_convention: report BOTH the
            # config-file convention/value and the resolved total-RMS-
            # equivalent amplitude the guard/field synthesis actually
            # used, so resolved_config.yaml/summary.csv never hide which
            # number is which -- see PerturbedDistanceTessellation's
            # amplitude_convention/amplitude_reference/amplitude_total_rms/
            # kappa properties (tessellation/perturbed.py).
            rows.append(("tessellation", "amplitude_convention",
                         curved.amplitude_convention))
            rows.append(("tessellation", "amplitude_A", curved.amplitude))
            if curved.amplitude_convention == "reference_wavelength":
                rows.append(("tessellation", "reference_wavelength_A",
                             tess.reference_wavelength))
                rows.append(("tessellation", "kappa", tess.kappa))
                rows.append(("tessellation", "amplitude_total_rms_A",
                             tess.amplitude_total_rms))
            rows.append(("tessellation", "spectrum", curved.spectrum))
            if curved.spectrum == "gaussian":
                rows.append(("tessellation", "correlation_length_A",
                             curved.correlation_length))
            else:
                rows.append(("tessellation", "hurst", curved.hurst))
                rows.append(("tessellation", "l_min_A", curved.l_min))
                rows.append(("tessellation", "l_max_A", curved.l_max))
                # G13 (Hurst back-estimate) lives here because
                # perturbed_distance is now the only self_affine consumer,
                # so it is also G13's only owner (alongside its own G20).
                g13 = [r for r in gates.results() if r.gate == "G13"]
                if g13:
                    rows.append(("tessellation", "hurst_estimated",
                                 g13[0].measured))
                g20 = [r for r in gates.results() if r.gate == "G20"]
                if g20:
                    rows.append(("tessellation", "d_b_estimated",
                                 g20[0].measured))
            rows.append(("tessellation", "a_max_A",
                         _PDS * msd / (2.0 * _ETA_CLIP)))
            rows.append(("tessellation", "n_colors", tess.n_colors))
            rows.append(("tessellation", "reassigned_fraction",
                         tess.reassigned_fraction))
        else:
            rows.append(("tessellation", "aspect_ratio_range",
                         " ".join(str(v)
                                  for v in curved.aspect_ratio_range)))
    rows.append(("tessellation", "is_flat",
                 isinstance(tess, FlatTessellation)))
    if sdot_res is not None:
        sd = config.grains.size_distribution
        assert sd is not None  # sdot_res exists only when configured
        rows.append(("size_distribution", "type", sd.type))
        if sd.type == "lognormal":
            rows.append(("size_distribution", "sigma_log", sd.sigma_log))
        rows.append(("size_distribution", "sdot_iterations",
                     sdot_res.iterations))
        rows.append(("size_distribution", "max_rel_volume_error",
                     sdot_res.max_rel_error))
        rows.append(("size_distribution", "weight_range_A2",
                     f"{float(np.min(sdot_res.weights)):.6g} .. "
                     f"{float(np.max(sdot_res.weights)):.6g}"))

    o_cfg = config.orientation
    rows.append(("texture", "scheme", o_cfg.scheme))
    if o_cfg.scheme == "odf_components" and o_cfg.components:
        w = np.array([c.weight for c in o_cfg.components], dtype=np.float64)
        w /= w.sum()
        rows.append(("texture", "n_components", len(o_cfg.components)))
        # Σ wᵢ² of the normalized component weights — discrete texture
        # concentration index (1 = single component, 1/n = equal weights).
        rows.append(("texture", "component_sum_w2", float(np.sum(w**2))))
    if mdf_res is not None:
        from grainsmith.orientation.mdf import canonical_target_type
        from grainsmith.orientation.odf import vp_halfwidth, vp_kappa

        mdf_cfg = o_cfg.mdf_target
        assert mdf_cfg is not None  # mdf_res exists only when configured
        rows.append(("mdf", "target_type", mdf_cfg.type))
        # Records which target CANONICALLY ran when the user wrote a
        # deprecated alias (mackenzie/csl_enriched) -- config/resolve.py
        # deliberately never rewrites the user's own `type` value in place
        # (config_sha256 stability), so this is the only place the
        # canonical name a deprecated alias resolves to is recorded.
        rows.append(("mdf", "target_type_canonical",
                     canonical_target_type(mdf_cfg.type)))
        if canonical_target_type(mdf_cfg.type) == "sigma3_angle_enriched":
            rows.append(("mdf", "sigma3_fraction", mdf_cfg.sigma3_fraction))
        rows.append(("mdf", "annealing_steps", mdf_res.n_steps))
        rows.append(("mdf", "swaps_accepted", mdf_res.n_accepted))
        rows.append(("mdf", "t0", mdf_cfg.t0))
        rows.append(("mdf", "cooling", mdf_cfg.cooling))
        rows.append(("mdf", "n_bins", len(mdf_res.bin_edges) - 1))
        rows.append(("mdf", "theta_max_deg", float(mdf_res.bin_edges[-1])))
        rows.append(("mdf", "chi2_initial", mdf_res.chi2_initial))
        rows.append(("mdf", "chi2_final_annealer", mdf_res.chi2_final))
        rows.append(("mdf", "objective", "area_weighted_disorientation_angle"))
        rows.append(("mdf", "odf_constraint",
                     "measure_only" if mdf_cfg.odf_drift_max is None else "tv_cap"))
        rows.append(("mdf", "odf_kernel_halfwidth_requested_deg",
                     mdf_cfg.odf_kernel_halfwidth_deg))
        kernel_degree: float | str
        kernel_halfwidth: float | str
        try:
            kernel_degree = vp_kappa(mdf_cfg.odf_kernel_halfwidth_deg)
            kernel_halfwidth = vp_halfwidth(kernel_degree)
        except ConfigError:
            kernel_degree = kernel_halfwidth = "not_evaluated"
        rows.append(("mdf", "odf_kernel_degree", kernel_degree))
        rows.append(("mdf", "odf_kernel_halfwidth_effective_deg", kernel_halfwidth))
        rows.append(("mdf", "odf_null_samples", mdf_cfg.odf_null_samples))
        # Volume-weighted-ODF trust region (populated whenever grain_volumes
        # was threaded through anneal_assignment -- i.e. always, since
        # _stage_mdf now always supplies it; see MDFResult's own docstring).
        if mdf_res.odf_drift_final is not None:
            rows.append(("mdf", "odf_drift_final", mdf_res.odf_drift_final))
            rows.append(("mdf", "odf_drift_max",
                         mdf_res.odf_drift_max
                         if mdf_res.odf_drift_max is not None else ""))
            rows.append(("mdf", "n_drift_vetoed", mdf_res.n_drift_vetoed))

    # G23 (report-only): angle-window vs true-CSL Sigma3 area fraction,
    # read straight off the gate's own `measured` pair -- see
    # gate_g23_sigma3_consistency's docstring for why `measured` is a pair
    # here rather than the usual single scalar. An empty cell (never 0.0)
    # for the CSL side when analysis.csl was off (NOT EVALUATED, not zero).
    g23 = [r for r in gates.results() if r.gate == "G23"]
    if g23:
        window_frac, csl_frac = g23[0].measured
        rows.append(("sigma3", "angle_window_area_fraction", window_frac))
        rows.append(("sigma3", "csl_area_fraction",
                     csl_frac if csl_frac is not None else ""))
        rows.append(("sigma3", "csl_status",
                 "not_evaluated" if csl_frac is None else
                 "measured" if np.isfinite(csl_frac) else "no_boundary_area"))
        rows.append(("sigma3", "angle_window_status",
                 "measured" if np.isfinite(window_frac) else "no_boundary_area"))

    # G25 (report-only): per-component configured weight vs realised
    # count/volume fractions (sampler pre-anneal AND final post-anneal),
    # straight off the gate's `measured` dict (same "None only if the gate
    # could not be evaluated, in which case no rows are written" pattern as
    # the G23 pair above; the shape changed from a tuple of triples to a
    # dict of 5-tuples plus three TV summary numbers -- see
    # gate_g25_component_fidelity). `component_{k}_volume_fraction` (no
    # `_pre`/`_post` suffix) is kept as the FINAL (post-anneal) number so
    # it stays the headline volume-weighted-ODF figure at its existing key.
    g25 = [r for r in gates.results() if r.gate == "G25"]
    if g25 and g25[0].measured is not None:
        m25 = g25[0].measured
        rows.append(("texture", "component_weight_basis", m25["basis"]))
        for k, (w_c, f_n, f_v_pre, f_v_post, mean_v) in enumerate(
                m25["per_component"]):
            rows.append(("texture", f"component_{k}_weight_configured", w_c))
            rows.append(("texture", f"component_{k}_count_fraction", f_n))
            rows.append(("texture", f"component_{k}_volume_fraction_pre",
                         f_v_pre))
            rows.append(("texture", f"component_{k}_volume_fraction",
                         f_v_post))
            # `_pre`, not the bare form: this IS the pre-anneal number (see
            # gate_g25's docstring -- it is measured on component_of_pre on
            # purpose, so the sampler's own size-component correlation stays
            # unmixed with the annealer's redistribution). Its neighbours
            # above use bare = POST (`..._volume_fraction` vs
            # `..._volume_fraction_pre`), so the bare name here read as a
            # post-anneal mean and quietly disagreed with grains.csv.
            rows.append(("texture", f"component_{k}_mean_volume_pre_A3",
                         mean_v))
        rows.append(("texture", "component_tv_cfg_vs_pre",
                     m25["tv_cfg_vs_pre"]))
        rows.append(("texture", "component_tv_pre_vs_post",
                     m25["tv_pre_vs_post"]))
        rows.append(("texture", "component_tv_cfg_vs_post",
                     m25["tv_cfg_vs_post"]))

    # G26 (report-only): atomistic ODF-weighting discretisation floor --
    # tessellation-volume weight (what odf_mtex.txt exports and
    # odf_drift_max controls) vs. atom-count weight (what the exported
    # atomistic structure actually realises) -- straight off the gate's
    # `measured` field (see gate_g26_odf_weighting_floor). Single-phase
    # `measured` is one ``(tv, max_dw, mean_atoms)`` triple; multiphase is
    # a tuple of that same shape, one entry per phase in `config.phases`
    # order (the same order `phase_names` was built in, so index p here
    # is phase p there), with a phase-local ``None`` wherever that one
    # phase's atom counts were absent. A completely un-evaluated gate
    # (single-phase, no atoms at all) writes NO rows at all -- same "no
    # partial rows" behaviour as G25 above when its own `measured` is
    # None; a phase-local ``None`` inside a multiphase tuple instead
    # writes an empty cell for just that phase's three keys (never a
    # false 0.0), the same per-item pattern G23's CSL side already uses.
    g26 = [r for r in gates.results() if r.gate == "G26"]
    if g26 and g26[0].measured is not None:
        m26 = g26[0].measured

        def _g26_rows(stat: tuple | None, suffix: str) -> None:
            if stat is None:
                rows.append(("odf", f"weighting_tv_volume_vs_atoms{suffix}",
                             ""))
                rows.append(("odf", f"weighting_max_dw{suffix}", ""))
                rows.append(("odf", f"atoms_per_grain_mean{suffix}", ""))
                return
            tv, max_dw, mean_atoms = stat
            rows.append(("odf", f"weighting_tv_volume_vs_atoms{suffix}", tv))
            rows.append(("odf", f"weighting_max_dw{suffix}", max_dw))
            rows.append(("odf", f"atoms_per_grain_mean{suffix}", mean_atoms))

        if config.crystal is None:
            # Multiphase: same section-per-phase split _summary_rows
            # already uses above (`for p, (pc, c) in
            # enumerate(zip(config.phases, crystals, ...))`).
            assert config.phases is not None  # gate G1 (resolve Rule 21)
            for p, pc in enumerate(config.phases):
                _g26_rows(m26[p], f"_{pc.name}")
        else:
            _g26_rows(m26, "")

    rows.append(("overlap", "enabled", cutoff is not None))
    if cutoff is not None:
        rows.append(("overlap", "cutoff_A", cutoff))
        rows.append(("overlap", "policy",
                     config.boundaries.overlap_removal.policy))

    rows.append(("atoms", "n_generated", n_generated))
    rows.append(("atoms", "n_deleted", n_deleted))
    rows.append(("atoms", "n_final", len(atoms)))

    if nominal is None:
        nominal = nominal_composition(crystal.basis.species,
                                      crystal.basis.occupancy)
    species, counts = np.unique(atoms.species, return_counts=True)
    final_counts = dict(zip(species.tolist(), counts.tolist(), strict=True))
    n_total = max(1, len(atoms))
    mass_map = _resolve_masses(type_map, config.output.lammps.masses)
    species_masses = {sp: count * mass_map[sp]
                      for sp, count in final_counts.items()}
    total_mass = sum(species_masses.values())
    rows.append(("composition", "nominal_basis", "pre_doping_host_atom_fraction"))
    rows.append(("composition", "final_basis", "post_doping_atom_fraction"))
    rows.append(("composition", "n_species_final", len(final_counts)))
    for sp in sorted(set(nominal) | set(final_counts)):
        rows.append(("composition", f"{sp}_nominal_fraction",
                     nominal.get(sp, 0.0)))
        rows.append(("composition", f"{sp}_final_fraction",
                     final_counts.get(sp, 0) / n_total))
        rows.append(("composition", f"{sp}_n_final", final_counts.get(sp, 0)))
        rows.append(("composition", f"{sp}_final_mass_fraction",
                     species_masses.get(sp, 0.0) / total_mass
                     if total_mass > 0.0 else float("nan")))

    rows.append(("atoms", "mass_total_amu", total_mass))
    rows.append(("atoms", "number_density_material_per_A3",
                 len(atoms) / material_volume))
    rows.append(("atoms", "number_density_export_per_A3",
                 len(atoms) / export_volume))
    rows.append(("atoms", "density_material_g_cm3",
                 total_mass / material_volume * AMU_PER_A3_TO_G_PER_CM3))
    rows.append(("atoms", "density_export_g_cm3",
                 total_mass / export_volume * AMU_PER_A3_TO_G_PER_CM3))

    for sp, t in type_map.items():
        rows.append(("lammps_types", sp, t))
        rows.append(("lammps_masses", sp, mass_map[sp]))

    for stage, elapsed in timings.items():
        rows.append(("timings", f"{stage}_s", elapsed))

    if curvature_rows:
        rows.extend(curvature_rows)

    if doping_rows:
        rows.extend(doping_rows)

    # Only gates that actually RAN get a row -- a deliberate contract, pinned
    # by tests/test_end_to_end.py::test_flat_all_gates_and_outputs (a flat run
    # writes exactly 12) and tests/test_odf_gates.py::
    # test_e2e_no_mdf_target_no_g22_row ("G22" must not appear at all without
    # an mdf_target). A gate that RUNS but has nothing to measure states that
    # in its own message instead ("n/a (flat geometry)", "NOT EVALUATED"),
    # which is where the not-applicable case is recorded.
    for r in gates.results():
        status = "PASS" if r.passed else "FAIL"
        rows.append(("gates", r.gate,
                     f"{status} | measured={r.measured} | {r.message}"))

    rows.extend(("versions", name, version) for name, version in _version_rows())

    # Execution environment (§R3): the versions section above pins the Python
    # libraries; these rows pin what those libraries were COMPILED and RUN
    # against — the BLAS behind numpy's `@`, the OS/CPU arch, whether the
    # optional numba kernel was active, and the worker count. None of them
    # change the physics; all of them can change the last floating-point bit,
    # and therefore the MANIFEST.txt digests, across machines.
    env = environment if environment is not None else environment_record()
    for key in ENVIRONMENT_KEYS:
        rows.append(("environment", key, env[key]))
    for phase_index, built in enumerate(crystals):
        section = ("crystal" if config.phases is None else
                   f"phase:{config.phases[phase_index].name}")
        for parameter in ("a", "b", "c", "alpha", "beta", "gamma"):
            unit = "A" if parameter in ("a", "b", "c") else "deg"
            rows.append((section, f"lattice_{parameter}_{unit}",
                         getattr(built.cellpar, parameter)))
        rows.append((section, "number_density_per_A3", built.rho_atom))
    rows.extend(summary_mapping_rows("config", config.model_dump(mode="json")))
    # The boundary NETWORK size, recorded whether or not analysis.statistics
    # ran. `analyze_boundaries` runs unconditionally and boundaries.csv is
    # always written, so with statistics off the summary used to say nothing
    # at all about a network it had fully computed -- a reader could not learn
    # the boundary count without opening another file.
    if boundary_reports is not None:
        rows.append(("boundaries", "number", len(boundary_reports)))
        rows.append(("boundaries", "gb_area_total_A2",
                     float(sum(b.area_A2 for b in boundary_reports))))
    if mesh_status is not None:
        # "written" | "skipped_flat_geometry" | "disabled". Without this a
        # config asking for a mesh on flat geometry produced no mesh file and
        # a summary still reading `config,output.mesh.enabled,True` -- the
        # skip was only ever a log line. Same idiom as sigma3,csl_status.
        rows.append(("mesh", "status", mesh_status))
    rows.append(("statistics", "status", "measured" if statistics is not None else
                 "disabled" if not config.analysis.statistics else "not_available"))
    if statistics is not None:
        for section, values in statistics.items():
            rows.extend(summary_mapping_rows(f"statistics:{section}", values))
    if memory_report is not None:
        rows.extend(summary_mapping_rows("memory", memory_report))
    if analysis_grid is not None:
        rows.append(("analysis", "voxel_grid_shape",
                     " ".join(str(value) for value in analysis_grid.shape)))
        rows.append(("analysis", "voxel_spacing_A",
                     " ".join(str(value) for value in analysis_grid.h_vec)))
        rows.append(("analysis", "n_voxels", int(np.prod(analysis_grid.shape))))
    else:
        rows.append(("analysis", "voxel_grid_status", "not_used"))
    return rows


def _write_diagnostics(outdir: Path, config: RunConfig, tess, atoms,
                       provenance) -> list[Path]:
    """verbose ≥ 3 diagnostic dumps (§8.6) into <outdir>/diagnostics/."""
    from grainsmith.io.common import fmt
    from grainsmith.tessellation.warp import WarpTessellation

    diag = outdir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    p = diag / "grain_atom_counts.dat"
    counts = np.bincount(atoms.grain, minlength=tess.n_grains)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {provenance.line()}\n# grain_id n_atoms\n")
        for i, c in enumerate(counts):
            fh.write(f"{i} {int(c)}\n")
    written.append(p)

    if isinstance(tess, WarpTessellation):
        p = diag / "warp_field_stats.dat"
        u = tess.field  # (3, Nx, Ny, Nz)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {provenance.line()}\n")
            fh.write(f"max_grad_u {fmt(tess.grad_max)}\n")
            for ax, name in enumerate("xyz"):
                fh.write(f"rms_u_{name} "
                         f"{fmt(float(np.sqrt(np.mean(u[ax] ** 2))))}\n")
            fh.write("max_norm_u "
                     f"{fmt(float(np.max(np.sqrt(np.sum(u ** 2, axis=0)))))}\n")
        written.append(p)

    if config.boundaries.geometry != "flat":
        vg = _analysis_voxel_grid(config, tess)
        p = diag / "voxel_histogram.dat"
        counts = np.bincount(vg.labels.ravel(),
                             minlength=tess.n_grains)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {provenance.line()}\n# grain_id n_voxels\n")
            for i, c in enumerate(counts):
                fh.write(f"{i} {int(c)}\n")
        written.append(p)
    return written


def _write_manifest(outdir: Path, provenance, files: list[Path]) -> Path:
    """MANIFEST.txt: sha256 + size of every output file THIS run produced (§4.1).

    Only the run's own outputs (``files``) plus ``run.log`` are listed, so a
    stale artefact left in a reused output directory is never attributed to this
    run's provenance line.  run.log is listed without a hash (its handler is
    still open) and MANIFEST.txt itself is excluded.

    The digest is streamed in fixed-size blocks rather than reading each file
    whole: the flagship LAMMPS dump is multi-GB, and a whole-file read here —
    after every science stage has succeeded — would transiently add the full
    file size to RSS and risk an OOM at the very last step.
    """
    manifest = outdir / "MANIFEST.txt"
    lines = [
        f"# {provenance.line()}",
        # Full-length digests: the header line above shows 12-hex display
        # forms, but `grainsmith verify` needs the untruncated values and must
        # find them in ONE self-contained anchor file. Parsed as
        # "# <key> = <value>" by grainsmith.verify.
        f"# config_sha256 = {provenance.config_sha256}",
        f"# provenance_sha256 = {provenance.provenance_sha256}",
        f"# provenance_payload_schema = {PROVENANCE_PAYLOAD_SCHEMA}",
        "# sha256  size_bytes  path (relative to this file)",
    ]
    entries = set(files)
    run_log = outdir / "run.log"
    if run_log.exists():
        entries.add(run_log)
    for f in sorted(entries):
        if not f.is_file() or f.name == "MANIFEST.txt":
            continue
        rel = f.relative_to(outdir).as_posix()
        if f.name == "run.log":
            lines.append(f"{'-' * 64}  -  {rel}")
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh_in:
            for block in iter(lambda: fh_in.read(1 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {f.stat().st_size}  {rel}")
    with manifest.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(config: RunConfig, jobs: int = 1,
        max_rss_gb: float | None = None) -> RunResult:
    """Run the full grainsmith pipeline (§4.1) and write all outputs (§8).

    Parameters
    ----------
    config : RunConfig
        Validated configuration from ``config.resolve.load_config`` (gate G1
        has already passed when this object exists).
    jobs : int
        Worker processes for the per-grain fill stage and the GB-curvature
        analysis stage (§13; CLI ``--jobs``). 1 = serial (default); 0 = all
        CPU cores. An execution detail only: outputs are bit-identical for
        every value (per-grain rng streams; per-boundary-pair curvature
        purity + ordered reassembly).
    max_rss_gb : float | None
        Optional soft total-RSS WARN ceiling in GB for the best-effort monitor
        (CLI ``--max-rss``; needs the optional ``psutil`` extra). ``None`` uses
        an auto-ceiling at a fraction of physical RAM. WARN only — never raises,
        never changes any output byte (see ``memory.MemoryMonitor``).

    Returns
    -------
    RunResult

    Raises
    ------
    grainsmith.errors.GrainsmithError
        Subclass per failing stage; QAGateError for evaluator-gate failures.
    """
    # Validate SOURCE_DATE_EPOCH before the lazy import block below: numpy's
    # f2py reads the same variable during scipy's import chain and would raise
    # a bare ValueError from inside numpy, burying our own clear diagnostic
    # under a 40-frame traceback.
    from grainsmith.provenance import source_date_epoch as _sde_check
    _sde_check()

    from grainsmith import __version__
    from grainsmith.analysis.boundaries import analyze_boundaries
    from grainsmith.analysis.curvature import (
        analyze_curvature,
        attach_curvature,
        global_curvature_rows,
        per_grain_gauss_bonnet,
    )
    from grainsmith.analysis.grains import analyze_grains, grain_volumes
    from grainsmith.atoms.doping import run_doping
    from grainsmith.config.resolve import config_sha256, dump_resolved
    from grainsmith.io import (
        fmt,
        gate_g10_lammps,
        write_boundaries_csv,
        write_doping_csv,
        write_doping_profile_csv,
        write_extxyz,
        write_gb_curvature_csv,
        write_gnuplot_bundle,
        write_grains_csv,
        write_lammps,
        write_mdf_csv,
        write_methods_md,
        write_microstructure_json,
        write_odf_mtex,
        write_ply,
        write_section_csv,
        write_statistics_csv,
        write_summary_csv,
        write_vertices_csv,
    )
    from grainsmith.qa import (
        GateResult,
        QAGates,
        gate_g3_voxel,
        gate_g7_min_distance,
        gate_g8_atom_count,
        gate_g9_composition,
        gate_g11_volume_targets,
        gate_g12_mdf_target,
        gate_g13_hurst,
        gate_g14_commensurate,
        gate_g15_phase_fractions,
        gate_g16_curvature,
        gate_g17_doping_composition,
        gate_g18_doping_geometry,
        gate_g19_reassigned_fraction,
        gate_g20_db_estimate,
        gate_g21_gauss_bonnet,
        gate_g22_odf_fidelity,
        gate_g23_sigma3_consistency,
        gate_g24_final_volume_targets,
        gate_g25_component_fidelity,
        gate_g26_odf_weighting_floor,
        nominal_composition,
        nominal_composition_phases,
    )
    from grainsmith.rng import make_rng
    from grainsmith.seeding import wigner_seitz_radius
    from grainsmith.tessellation.flat import FlatTessellation
    from grainsmith.tessellation.single import (
        SingleCrystalTessellation,
        commensurability_misfit,
    )
    from grainsmith.tessellation.warp import WarpTessellation

    if jobs < 0:
        raise ConfigError(f"jobs must be >= 0 (0 = all cores), got {jobs}.")
    # Keep the value the caller asked for: `--jobs 0` is a *request* for "all
    # cores available to this process" and the number it expands to is
    # machine-specific, so provenance records both (the requested form is what
    # reproduces the invocation; the effective form is what ran).
    jobs_requested = jobs
    if jobs == 0:
        jobs = _available_cpus()

    # Resolve once: the run may outlive the caller's working directory.
    outdir = Path(config.output.directory).resolve()
    fh = _setup_logging(config.meta.verbose, outdir)

    # NOTE: a previous run's summary.csv is deliberately NOT removed here.
    # A failed run leaves it in place and writes summary.failed.csv beside it
    # (see the except handler below), so a failure never destroys the record
    # of the last run that did succeed -- pinned by tests/test_memory_limit.py
    # ::test_memory_failure_records_traceback_and_partial_csv. The pairing is
    # what makes it readable: summary.csv carries meta,status=complete for the
    # run that wrote it, and summary.failed.csv's meta,status=failed marks the
    # later attempt. `grainsmith verify` also catches the mismatch outright,
    # because the failed attempt rewrites resolved_config.yaml with a fresh
    # timestamp and its manifest digest then no longer matches.

    # Best-effort total-RSS monitor: samples on a background thread,
    # WARNs (logging only) if the soft ceiling is crossed. Inert without psutil;
    # reads RSS only, so outputs stay byte-identical. Created before the try so
    # the finally can always join the sampler thread.
    from grainsmith.memory import MemoryMonitor
    mem = MemoryMonitor(max_rss_gb).start()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    memory_budget: tuple[float, str, float | None] | None = None
    try:
        log.info("grainsmith run: %s", config.meta.title)
        gates = QAGates()

        # §13: resolve the run's effective per-allocation memory budget
        # EXACTLY ONCE, here, before any of the three tessellation-binding
        # branches below (voxel_import / single-crystal / flat-or-curved)
        # constructs anything, and log the one effective-budget INFO line
        # for the whole run. The resolved tuple is threaded through to
        # _stage_tessellation (flat-or-curved backends need it AT
        # construction time -- see that function's docstring) and to every
        # _apply_runtime_limits call below (voxel_import's and
        # single-crystal's ONLY resolution/stamp); neither re-resolves when
        # given this tuple. Before this, each of those two call sites
        # resolved independently, which on the flat-or-curved path called
        # memory.resolve_memory_budget (and, when RAM was undetectable, its
        # warning) twice per run.
        from grainsmith.memory import log_effective_budget, resolve_memory_budget

        memory_budget = resolve_memory_budget(config)
        log_effective_budget(*memory_budget, config.runtime.memory_limit_gb)

        # Seed resolution (§7): 'entropy' harvests OS entropy via
        # SeedSequence (recorded for reproduction); module-level np.random
        # FUNCTIONS remain forbidden — this is a class constructor
        # with no global state.
        if config.seed.mode == "fixed":
            assert config.seed.value is not None  # gate G1 (resolve Rule 1)
            seed = int(config.seed.value)
        else:
            entropy = np.random.SeedSequence().entropy
            assert isinstance(entropy, int)  # default ctor harvests one int
            seed = entropy
            log.info("seed.mode=entropy -> harvested seed %d", seed)
        rng = make_rng(seed)

        timestamp = run_timestamp_iso()
        provenance = make_provenance(
            version=__version__, timestamp_iso=timestamp, seed=seed,
            config_sha256=config_sha256(config), title=config.meta.title,
        )
        # Built once and handed to both writers (summary.csv,
        # microstructure.json) below, so they can never disagree (§1.5).
        environment = environment_record(jobs_requested=jobs_requested,
                                         jobs_effective=jobs)
        dump_resolved(config, outdir / "resolved_config.yaml",
                      provenance=provenance)
        gates.record(GateResult(
            "G1", True, message="schema + cross-field validation passed "
            "(config.resolve.load_config)"))

        # box.cells: config/resolve.py Rule 28 already built and
        # LAMMPS-tilt-reduced the box matrix onto config.box.resolved_h.
        # H is lower-triangular (restricted-triclinic convention), so
        # det(H) == H[0,0]*H[1,1]*H[2,2] and L = diag(H) is EXACTLY the
        # box volume/edge-length proxy every orthogonal-box volume/wrap
        # formula below already uses (np.prod(L) == det(H)) — only the
        # stages that need the true off-diagonal tilt (tessellation
        # construction, overlap removal, G7/G14, LAMMPS/extxyz/gnuplot
        # writers) are additionally given cell_matrix.
        cell_matrix: np.ndarray | None = None
        if config.box.cells is not None:
            cell_matrix = np.asarray(config.box.resolved_h, dtype=np.float64)
            L = np.diag(cell_matrix).copy()
        else:
            L = np.asarray(config.box.lengths, dtype=np.float64)
        periodic = config.box.periodic
        flat = config.boundaries.geometry == "flat"
        vox_import = config.boundaries.geometry == "voxel_import"

        # --- crystal (G2; one per phase for multiphase runs) ---
        t = time.perf_counter()
        phase_names: list[str] | None = None
        if config.phases is not None:
            crystals = [_build_crystal(p.crystal, config.output.lammps.masses)
                        for p in config.phases]
            phase_names = [p.name for p in config.phases]
            # Requested SG numbers precomputed: a comprehension cannot host
            # the `assert ... is not None` that Rule 27 guarantees, and mypy
            # does not carry a narrowing from an enclosing loop into a genexp.
            _req_sgs = []
            for _p in config.phases:
                assert _p.crystal.space_group is not None  # Rule 27 resolved this
                _req_sgs.append(_p.crystal.space_group.number)
            detail = "; ".join(
                f"{p.name}: detected SG {c.dataset['number']} == requested "
                f"{sg_num}"
                for p, c, sg_num in zip(config.phases, crystals, _req_sgs,
                                        strict=True))
            gates.record(GateResult(
                "G2", True,
                measured=" ".join(str(c.dataset["number"])
                                  for c in crystals),
                message=f"spglib round-trip per phase: {detail}"))
        else:
            crystals = [_stage_crystal(config)]
            assert config.crystal is not None  # gate G1 (resolve Rule 21)
            assert config.crystal.space_group is not None  # Rule 27 resolved this
            gates.record(GateResult(
                "G2", True, measured=crystals[0].dataset["number"],
                message=f"spglib round-trip: detected SG "
                        f"{crystals[0].dataset['number']} == requested "
                        f"{config.crystal.space_group.number}"))
        crystal = crystals[0]   # representative for single-phase paths
        timings["crystal"] = time.perf_counter() - t

        relabel_map: dict[int, int] | None = None
        imported_quats: np.ndarray | None = None
        if vox_import:
            # --- voxel import: replaces seeding + tessellation; the
            #     grain count is DERIVED from the label field.  G5 is
            #     demoted to WARN unless strict_connectivity (external
            #     microstructures may legitimately contain disconnected
            #     or wrap-spanning grains). ---
            from grainsmith.errors import TessellationError

            t = time.perf_counter()
            tess, relabel_map, imported_quats = _stage_voxel_import(config)
            sdot_res = None
            n = tess.n_grains
            # Informative only (summary row): no seeding happened.
            msd = wigner_seitz_radius(float(np.prod(L)), n)
            timings["tessellation"] = time.perf_counter() - t
            gates.record(GateResult(
                "G4", True,
                message="n/a (imported voxel field — no polyhedral "
                        "cells)"))
            vi_cfg = config.boundaries.voxel_import
            assert vi_cfg is not None  # gate G1 (resolve Rule 17)
            try:
                tess.voxel_grid.check_connectivity(periodic)
                gates.record(GateResult(
                    "G5", True,
                    message="imported field: every grain is one periodic "
                            "connected component"))
            except TessellationError as exc:
                if vi_cfg.strict_connectivity:
                    gates.require(GateResult(
                        "G5", False,
                        message=f"{exc} (strict_connectivity=true)"))
                else:
                    gates.record(GateResult(
                        "G5", True,
                        message=f"WARN: {exc} — accepted (imported field, "
                                "strict_connectivity=false)"))
            gates.record(GateResult(
                "G6", True, message="n/a (no warp field)"))
        elif config.grains.number == 1:
            # --- single crystal: trivial backend,
            #     NO seeding (centre seed, no rng consumption), exact
            #     volume.  The only "boundary" is the box-face self-image
            #     (overlap removal handles it as a legitimate GB pair);
            #     gate G14 (commensurability) is recorded after the
            #     orientation stage — it needs R(q)·A. ---
            n = 1
            t = time.perf_counter()
            msd = wigner_seitz_radius(float(np.prod(L)), 1)  # informative
            tess = SingleCrystalTessellation(L, periodic,
                                             cell_matrix=cell_matrix)
            sdot_res = None
            timings["tessellation"] = time.perf_counter() - t
            gates.record(GateResult(
                "G3", True, measured=0.0,
                message="single crystal: cell volume == box volume "
                        "exactly"))
            gates.record(GateResult(
                "G4", True, message="n/a (single crystal)"))
            gates.record(GateResult(
                "G5", True, message="n/a (single crystal)"))
            gates.record(GateResult(
                "G6", True, message="n/a (single crystal)"))
        else:
            n = config.grains.number
            assert n is not None  # gate G1 (resolve Rule 16)

            # --- seeding ---
            t = time.perf_counter()
            msd = _resolve_min_seed_distance(config)
            seeds = _stage_seeding(config, msd, rng.seeding)
            timings["seeding"] = time.perf_counter() - t

            # --- tessellation (G3/G4 flat, G5/G6 curved — at
            #     construction; G11 volume targeting when
            #     size_distribution is set) ---
            t = time.perf_counter()
            tess, sdot_res = _stage_tessellation(config, seeds, msd,
                                                 rng.fields,
                                                 rng_sizes=rng.sizes,
                                                 memory_budget=memory_budget)
            timings["tessellation"] = time.perf_counter() - t
            if flat:
                box_vol = float(np.prod(L))
                rel = abs(tess.total_volume() - box_vol) / box_vol
                gates.record(GateResult(
                    "G3", True, measured=rel,
                    message="flat volume sum enforced at construction"))
                gates.record(GateResult(
                    "G4", True,
                    message="Euler V−E+F=2 enforced per cell at "
                            "construction"))
                gates.record(GateResult(
                    "G5", True, message="n/a (flat geometry)"))
                gates.record(GateResult(
                    "G6", True, message="n/a (flat geometry)"))
            else:
                gates.record(GateResult(
                    "G4", True,
                    message="n/a (curved geometry — no polyhedral cells)"))
                from grainsmith.tessellation.perturbed import (
                    PerturbedDistanceTessellation,
                )
                if isinstance(tess, PerturbedDistanceTessellation):
                    # G5 for perturbed_distance is repair-and-report, not
                    # a hard construction-time failure (warp/weighted's
                    # semantics) — the constructor already repaired any
                    # fragmentation; G19 reports the severity.
                    gates.record(GateResult(
                        "G5", True, measured=tess.reassigned_fraction,
                        message=(
                            f"repair-and-report: {tess.reassigned_fraction:.4%} "
                            "of voxels reassigned to restore periodic "
                            "connectivity (see G19)")))
                    gates.record(gate_g19_reassigned_fraction(
                        tess.reassigned_fraction))
                    gates.record(GateResult(
                        "G6", True,
                        message="n/a (perturbed_distance has no "
                                "diffeomorphism to check; seed-containment "
                                "+ exact seed-ownership guards apply "
                                "instead, enforced at construction)"))
                    if tess.spectrum == "self_affine":
                        # G13 (warn) lives here: warp does not accept
                        # spectrum: self_affine at all (it is a coordinate
                        # diffeomorphism and cannot produce a genuinely
                        # self-affine boundary regardless of Hurst; docs/
                        # physics.md §5b), so perturbed_distance is G13's
                        # only owner. Hurst back-estimation from the
                        # radially averaged PSD of the synthesized per-color
                        # scalar fields -- still meaningful evidence of the
                        # field's own spectral law, and completes G20 below
                        # (the resulting BOUNDARY's box-counting dimension).
                        curved_cfg = _curved_cfg(config)
                        gates.record(gate_g13_hurst(
                            tess.hurst_estimate(), curved_cfg.hurst))
                        # G20 (warn): box-counting dimension (D_b)
                        # estimate, the G13 sibling for this method.
                        gates.record(gate_g20_db_estimate(
                            tess.d_b_estimate()))
                else:
                    gates.record(GateResult(
                        "G5", True,
                        message="periodic-connectivity check enforced at "
                                "construction"))
                    if isinstance(tess, WarpTessellation):
                        gates.record(GateResult(
                            "G6", True, measured=tess.grad_max,
                            message=f"warp guards passed (max‖∇u‖="
                                    f"{tess.grad_max:.4f} < 0.5; ‖u‖ ≤ A_clip)"))
                        # warp never reaches spectrum == "self_affine" any
                        # more (Rule 8a in config/resolve.py + the matching
                        # API guard in WarpTessellation.__init__), so there
                        # is no G13 branch here -- G13 moved to
                        # perturbed_distance above.
                    else:
                        gates.record(GateResult(
                            "G6", True, message="n/a (no warp field)"))

        # §13: belt-and-braces (re)stamp of the run's per-allocation memory
        # budget onto `tess` -- this is the single point where all three
        # tessellation-binding branches above (voxel_import / single-crystal
        # / flat-or-curved) have converged. `memory_budget` was resolved
        # (and logged) exactly once, above, before any of those branches
        # ran, so this call never re-resolves. For the flat-or-curved
        # branch this is a no-op re-stamp: _stage_tessellation already
        # threaded the SAME tuple through to construction time (where it
        # actually matters for the four curved backends); voxel_import and
        # single-crystal never call _stage_tessellation, so this is their
        # only stamp, applied before the first stage (fill, analysis) that
        # can trigger a §13 guard check on THEM.
        _apply_runtime_limits(tess, config, memory_budget)

        if sdot_res is not None:
            # sdot_res.tess (a PowerTessellation) is the SAME object as
            # `tess` for the flat+size_distribution case (already stamped
            # above) but a DISTINCT wrapped `base` for warp+size_distribution
            # -- already covered by _apply_runtime_limits' `_base` handling
            # on `tess`, restamped here too so this stays correct even if a
            # future refactor breaks that aliasing.
            _apply_runtime_limits(sdot_res.tess, config, memory_budget)
            # G11 on the RE-MEASURED cells of the fitted power diagram
            # (for warp, targeting applies to the unwarped base).
            base_vols = np.array(
                [c.volume for c in sdot_res.tess.cells], dtype=np.float64)
            sd_cfg = config.grains.size_distribution
            assert sd_cfg is not None
            gates.require(gate_g11_volume_targets(
                base_vols, sdot_res.target_volumes, sd_cfg.vol_tol,
                warp_active=isinstance(tess, WarpTessellation)))

        # --- phase assignment: deterministic greedy partition of
        #     the measured pre-fill volumes — no rng stream; gate G15
        #     (warn) re-measures the achieved volume fractions ---
        phase_of: np.ndarray | None = None
        phase_achieved: np.ndarray | None = None
        phase_atom_weights: np.ndarray | None = None
        if config.phases is not None:
            from grainsmith.phases import achieved_fractions, assign_phases

            t = time.perf_counter()
            if flat:
                pre_vols = grain_volumes(tess, L)
            else:
                pre_vols = _analysis_voxel_grid(config, tess).volumes()
            fracs = np.array([p.fraction for p in config.phases],
                             dtype=np.float64)
            if tess.n_grains < len(fracs):
                # Rule 23 covers configured counts; a DERIVED voxel_import
                # count can still be too small.
                raise ConfigError(
                    f"{tess.n_grains} grains cannot cover "
                    f"{len(fracs)} phases - every phase needs at least "
                    "one grain.")
            phase_of = assign_phases(pre_vols, fracs)
            phase_achieved = achieved_fractions(pre_vols, phase_of,
                                                len(fracs))
            assert phase_names is not None
            gates.record(gate_g15_phase_fractions(
                phase_achieved, fracs, phase_names,
                float(np.max(pre_vols)) / float(np.sum(pre_vols))))
            # Expected atom share per phase: ρ_p · V_p^assigned — drives
            # the G8 count estimate and the G9 nominal-composition mix.
            v_assigned = np.bincount(phase_of, weights=pre_vols,
                                     minlength=len(fracs))
            phase_atom_weights = np.array(
                [c.rho_atom for c in crystals]) * v_assigned
            timings["phases"] = time.perf_counter() - t
            log.info("phases: %s", ", ".join(
                f"{nm}: {fa:.4f} (target {ft:g}, {int(np.sum(phase_of == p))} grains)"
                for p, (nm, fa, ft) in enumerate(
                    zip(phase_names, phase_achieved, fracs, strict=True))))

        # Prime the shared analysis voxel grid BEFORE the orientation stage
        # (HOISTED from just above the MDF stage, where it used to sit --
        # its own area-weight rationale is unchanged: the MDF stage's
        # pair_areas are still the first curved-geometry consumer, so they
        # still measure on the same resolution as boundaries.csv /
        # grains.csv when analysis.voxel_grid is overridden, see
        # _analysis_voxel_grid). The hoist is needed because
        # component_weight_basis == "volume" now ALSO needs per-grain
        # volumes, at orientation time -- i.e. before this priming used to
        # run. Nothing between the OLD position (just above the MDF stage)
        # and this NEW one touches the voxel grid or the tessellation: the
        # orientation stage (_stage_orientation) only samples/assigns
        # quaternions, and the G14 check right after it only reads
        # `quats[0]` and `crystal.A` -- neither mutates `tess`. The block
        # itself is RNG-free and a pure function of the FINAL tessellation
        # geometry (warp/self-affine geometry is applied at tessellation
        # CONSTRUCTION time, `_stage_tessellation`, strictly before this
        # point either way), and `_analysis_voxel_grid` caches the grid it
        # builds onto `tess._voxel` (idempotent -- a second call just
        # returns the cache), so moving WHEN this runs cannot change any
        # result, only which stage happens to trigger the (identical) grid
        # build first.
        if isinstance(config.analysis.voxel_grid, int) and hasattr(tess, "_voxel"):
            _analysis_voxel_grid(config, tess)

        # Per-grain volumes, computed ONCE right after the priming above so
        # every later consumer -- the orientation stage's
        # component_weight_basis == "volume" and the MDF stage's
        # odf_drift_max trust region -- reads the SAME array, computed on
        # the SAME (now-primed) grid the analysis stage will later use too
        # (see _stage_mdf's "WHY THIS IS THE RIGHT ARRAY" comment).
        need_vols = (config.orientation.mdf_target is not None
                     or (config.orientation.scheme == "odf_components"
                         and config.orientation.component_weight_basis
                             == "volume"))
        grain_vols = grain_volumes(tess, L) if need_vols else None

        # --- orientation ---
        t = time.perf_counter()
        quats, component_of = _stage_orientation(
            config, n, crystal.A, rng.orientation,
            imported_quats=imported_quats, grain_volumes=grain_vols)
        timings["orientation"] = time.perf_counter() - t

        # --- G14 (warn): single-crystal box/lattice commensurability
        #     — measured on the ROTATED lattice R(q)·A ---
        if isinstance(tess, SingleCrystalTessellation):
            from grainsmith.orientation.quaternion import quat_to_matrix
            gates.record(gate_g14_commensurate(commensurability_misfit(
                quat_to_matrix(quats[0]) @ crystal.A, L, periodic,
                box_matrix=cell_matrix)))

        # --- MDF targeting (assignment annealing on the `mdf` stream;
        #     timing row only when active so summaries without it are
        #     unchanged) ---
        t = time.perf_counter()
        quats, mdf_res = _stage_mdf(config, tess, quats, crystal.sym_quats,
                                    rng.mdf, grain_volumes=grain_vols)
        if mdf_res is not None:
            timings["mdf"] = time.perf_counter() - t
        # G25's component labels must follow the FINAL assignment: the
        # anneal moved sampled orientation j to whichever grain i has
        # perm[i] == j, so the final label of grain i is
        # component_of[perm[i]] (same index bookkeeping as gate G22's
        # KEY FACT derivation; without annealing the labels are unchanged).
        component_of_final = (
            component_of[mdf_res.permutation]
            if (component_of is not None and mdf_res is not None)
            else component_of
        )

        # --- fill ---
        t = time.perf_counter()
        atoms = _stage_fill(config, tess, crystals, quats, rng, jobs=jobs,
                            phase_of=phase_of)
        n_generated = len(atoms)
        timings["fill"] = time.perf_counter() - t

        # --- overlap (G7); "0.85*d_nn" resolves against the SMALLEST
        #     phase d_nn for multiphase runs (conservative) ---
        d_nn_min = min(c.d_nn for c in crystals)
        t = time.perf_counter()
        atoms, ledger, cutoff = _stage_overlap(config, atoms, tess,
                                               d_nn_min,
                                               cell_matrix=cell_matrix,
                                               jobs=jobs)
        n_deleted = ledger.total_deleted if ledger is not None else 0
        timings["overlap"] = time.perf_counter() - t
        if cutoff is not None:
            gates.require(gate_g7_min_distance(atoms, cutoff, periodic, L,
                                               cell_matrix=cell_matrix,
                                               jobs=jobs))
        else:
            gates.record(GateResult(
                "G7", True, message="n/a (overlap removal disabled)"))

        atoms = _sort_by_grain(atoms)

        # --- doping (G17/G18) ---
        # G8/G9 measure the HOST lattice: snapshot before dopants exist.
        n_host_final = len(atoms)
        host_species_u, host_counts_u = np.unique(atoms.species,
                                                  return_counts=True)
        doping_res = None
        if config.doping is not None:
            t = time.perf_counter()
            assert config.crystal is not None  # Rule 26: doping is single-phase only
            assert config.crystal.space_group is not None  # Rule 27 resolved this
            atoms, doping_res = run_doping(
                config.doping, atoms, tess, rng, crystal, quats, n, L,
                periodic,
                sg_setting=config.crystal.space_group.setting)
            if any(d.mode == "interstitial"
                   for d in config.doping.dopants):
                atoms = _sort_by_grain(atoms)   # substitutional never
                                                # reorders or appends
            timings["doping"] = time.perf_counter() - t
            gates.require(gate_g18_doping_geometry(
                doping_res.n_violations, doping_res.violation_detail))
            gates.record(gate_g17_doping_composition(
                doping_res.gate_rows))

        if config.analysis.per_atom_margin:
            atoms = _attach_margins(atoms, tess, n)

        # --- analysis (G3-voxel, G8, G9) ---
        t = time.perf_counter()
        if flat:
            volumes = grain_volumes(tess, L)
            vg = None
        else:
            vg = _analysis_voxel_grid(config, tess)
            volumes = vg.volumes()
            gates.require(gate_g3_voxel(volumes, L, vg.h_vec))
        # G24 (warn): the geometry changed AFTER the volume fit -- i.e. the
        # warp displaced the SDOT-fitted cells G11 already checked on the
        # unwarped base. For flat + size_distribution `sdot_res.tess is
        # tess` (the SAME object, see the G11 call site comment above), so
        # this never double-reports G11 on identical volumes.
        if sdot_res is not None and sdot_res.tess is not tess:
            gates.record(gate_g24_final_volume_targets(
                volumes, sdot_res.target_volumes))
        # G25 (report-only): configured texture-component weights vs the
        # REALISED count/volume fractions, sampler (pre-anneal) AND final
        # (post-anneal) -- the count-vs-volume gap behind the volume-
        # weighted-ODF criticism, reported directly (odf_components scheme
        # only). `component_of` (pre-anneal) and `component_of_final`
        # (post-anneal) are the SAME labelling before/after the MDF
        # permutation -- see the derivation comment above
        # `component_of_final`'s assignment.
        if component_of_final is not None:
            assert component_of is not None
            assert config.orientation.components is not None
            gates.record(gate_g25_component_fidelity(
                np.array([c.weight for c in config.orientation.components],
                         dtype=np.float64),
                component_of, component_of_final, volumes,
                config.orientation.component_weight_basis))
        multiphase = phase_of is not None
        grain_reports = analyze_grains(
            tess, quats, crystal.A, L, atoms=atoms, volumes=volumes,
            phase_of=phase_of,
            A_list=[c.A for c in crystals] if multiphase else None,
            phase_names=phase_names if multiphase else None)
        # G26 (report-only, never trips): the atomistic ODF-weighting
        # discretisation floor -- odf_mtex.txt/G22/odf_drift_max all use
        # TESSELLATION-volume grain weights (grain_reports[i].volume_A3),
        # but the exported artefact is ATOM-count weighted
        # (grain_reports[i].n_atoms); see gate_g26_odf_weighting_floor's
        # docstring for the measured (atoms/grain)^(-1/3) scaling. Right
        # after grain_reports so both fields are in hand; multiphase
        # reports per phase, never a global mix (differing atom number
        # densities across phases would make a global number meaningless).
        gates.record(gate_g26_odf_weighting_floor(
            np.array([r.volume_A3 for r in grain_reports], dtype=np.float64),
            np.array([r.n_atoms for r in grain_reports], dtype=np.float64),
            phase_of=phase_of,
            phase_names=phase_names if multiphase else None,
        ))
        boundary_reports = analyze_boundaries(
            tess, quats, crystal.sym_quats, crystal.A, L, periodic,
            ledger=ledger, csl=config.analysis.csl,
            character=config.analysis.gb_character,
            phase_of=phase_of,
            phase_names=phase_names if multiphase else None,
            sym_quats_list=[c.sym_quats for c in crystals] if multiphase
                           else None,
            A_list=[c.A for c in crystals] if multiphase else None,
            jobs=jobs,
        )
        # G8/G9: multiphase runs expect Σ_p ρ_p·V_p^assigned atoms and a
        # volume-weighted nominal composition mix.
        if phase_atom_weights is not None:
            rho_eff = float(np.sum(phase_atom_weights)) / float(np.prod(L))
            nominal = nominal_composition_phases(
                [nominal_composition(c.basis.species, c.basis.occupancy)
                 for c in crystals],
                phase_atom_weights)
        else:
            rho_eff = crystal.rho_atom
            nominal = nominal_composition(crystal.basis.species,
                                          crystal.basis.occupancy)
        gates.require(gate_g8_atom_count(
            n_host_final, n_deleted, rho_eff, float(np.prod(L))))
        gates.record(gate_g9_composition(
            dict(zip(host_species_u.tolist(), host_counts_u.tolist(),
                     strict=True)),
            nominal))

        # --- GB curvature (G16 + G21, warn-only; analysis.gb_curvature) ---
        curvature_res = None
        if config.analysis.gb_curvature:
            t_curv = time.perf_counter()
            curvature_res = analyze_curvature(tess, L, periodic, vg=vg,
                                              jobs=jobs)
            attach_curvature(boundary_reports, curvature_res)
            gates.record(gate_g16_curvature(curvature_res.n_dropped,
                                            curvature_res.n_raw))
            # G21: independent read of the SAME analysis — G16 asks "did
            # we keep enough samples" (drop fraction), G21 asks "how much
            # of the excluded edge/vertex curvature is leaking into the
            # kept set" (per-grain Gauss-Bonnet face-interior residual vs
            # the 4π closed-surface unit). Warn-only, coarse tool
            # diagnostic (routine WARNs on curved runs are expected with
            # the current estimator — calibration in qa.py
            # gate_g21_gauss_bonnet / constants.CURV_G21_GAUSS_BONNET_TOL
            # docstrings), never a model-validity judgment.
            gb_totals = per_grain_gauss_bonnet(curvature_res, tess.n_grains)
            gates.record(gate_g21_gauss_bonnet(gb_totals))
            timings["curvature"] = time.perf_counter() - t_curv

        # MDF histogram of the FINAL boundary network (mdf.csv input; G12
        # re-measures here rather than trusting the annealer's claim).
        # Multiphase runs defer to PER-PHASE histograms in the
        # write stage — there is no global point group to bin against.
        from grainsmith.orientation.mdf import (
            chi2_distance,
            histogram_masses,
            reference_angles,
            theta_max_deg,
        )
        if not multiphase:
            b_angles = np.array([r.misorientation_deg
                                 for r in boundary_reports],
                                dtype=np.float64)
            b_areas = np.array([r.area_A2 for r in boundary_reports],
                               dtype=np.float64)
            # G23 (report-only, never trips): angle-window vs true-CSL
            # Sigma3 area fraction, from these SAME per-boundary arrays —
            # independent of whether mdf_target is configured (informative
            # either way). csl_sigma is only meaningful when analysis.csl
            # ran; otherwise every BoundaryReport.csl_sigma is "" and the
            # CSL side must be reported NOT EVALUATED, never a false 0.0.
            b_csl_sigma = ([r.csl_sigma for r in boundary_reports]
                           if config.analysis.csl else None)
            gates.record(gate_g23_sigma3_consistency(
                b_angles, b_areas, b_csl_sigma))
            if mdf_res is not None:
                mdf_edges = mdf_res.bin_edges
                mdf_ref = mdf_res.reference_masses
                mdf_target = mdf_res.target_masses
            else:
                ref_ang = reference_angles(crystal.sym_quats)
                mdf_edges = np.linspace(0.0, theta_max_deg(ref_ang),
                                        config.analysis.mdf_bins + 1)
                mdf_ref = histogram_masses(ref_ang, np.ones_like(ref_ang),
                                           mdf_edges)
                mdf_target = None
            mdf_area_masses = histogram_masses(b_angles, b_areas, mdf_edges)
            mdf_num_masses = histogram_masses(b_angles,
                                              np.ones_like(b_angles),
                                              mdf_edges)
            if mdf_res is not None:
                assert config.orientation.mdf_target is not None
                mdf_cfg = config.orientation.mdf_target
                chi2_measured = chi2_distance(mdf_area_masses, mdf_target)
                gates.record(gate_g12_mdf_target(
                    chi2_measured, mdf_cfg.chi2_max))
                # G22 (warn-only, never fails): re-measures the assignment
                # annealer's own volume-weighted-ODF drift claim on the
                # FINAL structure (analysis stage, not inside the
                # annealer — see gate_g22_odf_fidelity's docstring for the
                # "orientation index == grain index" derivation this
                # relies on) plus, memory permitting, its kernel-smoothed
                # bias signature against a null. `tess.memory_limit_bytes`/
                # `memory_limit_source` were stamped by
                # `_apply_runtime_limits` earlier in this run — the same
                # §13 budget every other guard in this pipeline uses.
                gates.record(gate_g22_odf_fidelity(
                    volumes, mdf_res.permutation, quats, crystal.sym_quats,
                    mdf_res.odf_drift_final, mdf_res.odf_drift_max,
                    mdf_cfg.odf_kernel_halfwidth_deg,
                    mdf_cfg.odf_null_samples,
                    tess.memory_limit_bytes, tess.memory_limit_source,
                ))

        # --- publication statistics: RE-MEASURED from
        #     the final tessellation/reports; phase-aware by design ---
        statistics: dict[str, dict[str, Any]] | None = None
        if config.analysis.statistics:
            from grainsmith.analysis.statistics import compute_statistics

            # hurst_target/hurst_estimated live under "perturbed_distance"
            # only: warp does not accept spectrum: self_affine at all
            # (docs/physics.md §5b), so perturbed_distance is the only
            # geometry that populates statistics.csv's "roughness" section.
            hurst_target = hurst_estimated = None
            if (config.boundaries.geometry == "curved"
                    and _curved_cfg(config).method == "perturbed_distance"
                    and _curved_cfg(config).spectrum == "self_affine"):
                hurst_target = _curved_cfg(config).hurst
                g13 = [r for r in gates.results() if r.gate == "G13"]
                if g13:
                    hurst_estimated = g13[0].measured
            statistics = compute_statistics(
                tess, volumes, grain_reports, boundary_reports, L, periodic,
                voxel_grid=vg,
                phase_of=phase_of,
                phase_names=phase_names if multiphase else None,
                character=config.analysis.gb_character,
                csl=config.analysis.csl,
                hurst_target=hurst_target,
                hurst_estimated=hurst_estimated,
            )
        timings["analysis"] = time.perf_counter() - t

        # --- write (G10) ---
        t = time.perf_counter()
        files: list[Path] = [outdir / "resolved_config.yaml"]
        out = config.output

        lammps_path = outdir / out.lammps.filename
        type_map = write_lammps(
            lammps_path, atoms, L, periodic, out.lammps.atom_style,
            provenance, vacuum=config.box.vacuum, masses=out.lammps.masses,
            jobs=jobs, cell_matrix=cell_matrix,
        )
        files.append(lammps_path)
        gates.require(gate_g10_lammps(
            lammps_path, atoms, L, periodic, config.box.vacuum,
            out.lammps.atom_style,
            mode=out.lammps.g10_readback,
            sample_k=out.lammps.g10_readback_sample,
            cell_matrix=cell_matrix))

        if out.xyz.enabled:
            xyz_path = outdir / out.xyz.filename
            write_extxyz(xyz_path, atoms, L, periodic, provenance,
                         vacuum=config.box.vacuum,
                         per_atom_margin=config.analysis.per_atom_margin,
                         jobs=jobs, cell_matrix=cell_matrix)
            files.append(xyz_path)

        from grainsmith.io import (
            BOUNDARIES_COLUMNS_CURVATURE,
            BOUNDARIES_COLUMNS_PHASES,
            BOUNDARIES_COLUMNS_PHASES_CURVATURE,
            GRAINS_COLUMNS_PHASES,
        )
        write_grains_csv(grain_reports, outdir / out.csv.grains,
                         columns=GRAINS_COLUMNS_PHASES if multiphase
                                 else None)
        files.append(outdir / out.csv.grains)
        curv_on = curvature_res is not None
        if multiphase and curv_on:
            bcols = BOUNDARIES_COLUMNS_PHASES_CURVATURE
        elif multiphase:
            bcols = BOUNDARIES_COLUMNS_PHASES
        elif curv_on:
            bcols = BOUNDARIES_COLUMNS_CURVATURE
        else:
            bcols = None
        write_boundaries_csv(boundary_reports, outdir / out.csv.boundaries,
                             columns=bcols)
        files.append(outdir / out.csv.boundaries)

        # Texture outputs: MDF histogram + MTEX-ready ODF.
        # Multiphase runs write PER-PHASE files (mdf_<name>.csv,
        # odf_mtex_<name>.txt): the misorientation reference and the ODF
        # are defined within one point group.
        # atom_counts (grain_reports[i].n_atoms, same order as quats/
        # volumes) lets write_odf_mtex add the atom_fraction column
        # (G26) alongside the tessellation-volume weight column.
        n_atoms_arr = np.array([r.n_atoms for r in grain_reports],
                               dtype=np.float64)
        if not multiphase:
            write_mdf_csv(outdir / "mdf.csv", mdf_edges, mdf_area_masses,
                          mdf_num_masses, mdf_ref, mdf_target)
            files.append(outdir / "mdf.csv")
            write_odf_mtex(outdir / "odf_mtex.txt", quats,
                           volumes / float(np.sum(volumes)), provenance,
                           atom_counts=n_atoms_arr)
            files.append(outdir / "odf_mtex.txt")
        else:
            assert phase_names is not None and phase_of is not None
            for p, name in enumerate(phase_names):
                same = [r for r in boundary_reports
                        if r.phase_i == name and r.phase_j == name]
                ang = np.array([r.misorientation_deg for r in same],
                               dtype=np.float64)
                ar = np.array([r.area_A2 for r in same], dtype=np.float64)
                ref_ang_p = reference_angles(crystals[p].sym_quats)
                edges_p = np.linspace(0.0, theta_max_deg(ref_ang_p),
                                      config.analysis.mdf_bins + 1)
                ref_p = histogram_masses(ref_ang_p,
                                         np.ones_like(ref_ang_p), edges_p)
                mdf_path = outdir / f"mdf_{name}.csv"
                write_mdf_csv(mdf_path, edges_p,
                              histogram_masses(ang, ar, edges_p),
                              histogram_masses(ang, np.ones_like(ang),
                                               edges_p),
                              ref_p, None)
                files.append(mdf_path)
                mask = phase_of == p
                w_p = volumes[mask]
                odf_path = outdir / f"odf_mtex_{name}.txt"
                write_odf_mtex(odf_path, quats[mask],
                               w_p / float(np.sum(w_p)), provenance,
                               atom_counts=n_atoms_arr[mask])
                files.append(odf_path)
        if isinstance(tess, FlatTessellation):
            write_vertices_csv(tess, L, outdir / out.csv.vertices)
            files.append(outdir / out.csv.vertices)

        if curvature_res is not None:
            curv_path = outdir / "gb_curvature.csv"
            write_gb_curvature_csv(curvature_res, curv_path)
            files.append(curv_path)

        if doping_res is not None:
            doping_path = outdir / "doping.csv"
            write_doping_csv(doping_res.rows, doping_path)
            files.append(doping_path)
            profile_path = outdir / "doping_profile.csv"
            write_doping_profile_csv(doping_res.profiles, profile_path)
            files.append(profile_path)

        # Publication outputs: statistics.csv +
        # microstructure.json (analysis.statistics), the EBSD-like
        # section (analysis.section), METHODS.md (output.methods_snippet).
        if statistics is not None:
            stats_path = outdir / "statistics.csv"
            write_statistics_csv(statistics, stats_path)
            files.append(stats_path)
            micro_path = outdir / "microstructure.json"
            write_microstructure_json(
                micro_path, config, provenance, gates, grain_reports,
                boundary_reports, statistics, multiphase=multiphase,
                curvature=curv_on, environment=environment)
            files.append(micro_path)
        if config.analysis.section is not None:
            sec = config.analysis.section
            sec_path = outdir / f"slice_{sec.axis}{fmt(sec.position)}.csv"
            write_section_csv(sec_path, tess, grain_reports, L, sec.axis,
                              sec.position, multiphase=multiphase)
            files.append(sec_path)
        if out.methods_snippet:
            files.append(write_methods_md(
                outdir / "METHODS.md", config=config, provenance=provenance,
                crystals=crystals, phase_names=phase_names, tess=tess,
                msd=msd, cutoff=cutoff, n_deleted=n_deleted,
                sdot_res=sdot_res, mdf_res=mdf_res, gates=gates,
                phase_achieved=phase_achieved))

        if out.gnuplot.enabled:
            files.extend(write_gnuplot_bundle(
                outdir, tess, L, periodic, provenance,
                curvature=curvature_res is not None,
                doping_elements=(
                    [p.element for p in doping_res.profiles]
                    if doping_res is not None else None)))

        mesh_status = "disabled"
        if out.mesh.enabled:
            if flat:
                mesh_status = "skipped_flat_geometry"
                log.warning("output.mesh.enabled is set but geometry is "
                            "flat — mesh output is curved-only (§8.5), "
                            "skipping.")
            else:
                mesh_status = "written"
                mesh_path = outdir / "boundaries.ply"
                write_ply(mesh_path, _analysis_voxel_grid(config, tess),
                          tess.adjacency(), provenance)
                files.append(mesh_path)

        if config.meta.verbose >= 3:
            files.extend(_write_diagnostics(outdir, config, tess, atoms,
                                            provenance))
        timings["write"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        if fh is not None:
            fh.handle(log.makeRecord(
                log.name, logging.INFO, __file__, 0,
                "stage timings (s): %s",
                (", ".join(f"{stage}={elapsed!r}"
                           for stage, elapsed in timings.items()),), None))

        # summary.csv last: it contains every gate result (incl. G10) and
        # the complete stage timings (its own serialization is excluded).
        rows = _summary_rows(config, provenance, crystals, tess, msd, cutoff,
                             type_map, n_generated, n_deleted, atoms,
                             timings, gates, sdot_res=sdot_res,
                             mdf_res=mdf_res, relabel_map=relabel_map,
                             nominal=nominal, phase_of=phase_of,
                             phase_achieved=phase_achieved,
                             curvature_rows=(global_curvature_rows(curvature_res)
                                             if curvature_res is not None
                                             else None),
                             doping_rows=(doping_res.summary_rows
                                          if doping_res is not None
                                          else None),
                             environment=environment,
                             statistics=statistics,
                             analysis_grid=vg,
                             boundary_reports=boundary_reports,
                             mesh_status=mesh_status,
                             memory_report={
                                 "allocation_limit_configured_gb": config.runtime.memory_limit_gb,
                                 "allocation_limit_effective_gb": memory_budget[0] / 1e9,
                                 "allocation_limit_source": memory_budget[1],
                                 "physical_ram_gb": (None if memory_budget[2] is None else
                                                     memory_budget[2] / 1e9),
                                 **mem.report(),
                             })
        write_summary_csv(rows, outdir / out.csv.summary)
        files.append(outdir / out.csv.summary)
        files.append(_write_manifest(outdir, provenance, files))

        # Total-RSS monitor wrap-up AFTER the write stage (incl. manifest
        # hashing), so the whole run is monitored — the manifest's own file
        # reads are the last allocation and must not escape the soft ceiling.
        # RSS is non-deterministic, so it is reported via the log + RunResult
        # only — never written to a reproducible output. The peak is logged in
        # the finally block so it is reported on the exception path too.
        mem.stop()

        log.info("done: %d atoms, %d grains, %d boundaries -> %s "
                 "(total %.2f s)", len(atoms), n, len(boundary_reports),
                 outdir, timings["total"])
        return RunResult(
            config=config, seed=seed, outdir=outdir, atoms=atoms, tess=tess,
            quats=quats, grain_reports=grain_reports,
            boundary_reports=boundary_reports, gates=gates, timings=timings,
            n_generated=n_generated, n_deleted=n_deleted, d_nn=d_nn_min,
            files=files, phase_of=phase_of, statistics=statistics,
            peak_rss_bytes=mem.peak_rss_bytes, provenance=provenance,
        )
    except Exception as exc:
        log.exception("grainsmith generation failed: %s", exc)
        try:
            from grainsmith.io.reports import summary_mapping_rows

            failure_rows = [
                ("meta", "status", "failed"),
                ("meta", "title", config.meta.title),
                ("meta", "grainsmith_version", __version__),
                ("meta", "config_sha256", config_sha256(config)),
                ("meta", "completed_timing_stages", " ".join(timings)),
                ("failure", "exception_type", type(exc).__name__),
                ("failure", "message", str(exc)),
                ("timings", "elapsed_until_failure_s", time.perf_counter() - t0),
            ]
            failure_rows.extend(("timings", f"{stage}_s", elapsed)
                                for stage, elapsed in timings.items())
            failure_rows.extend(summary_mapping_rows("config", config.model_dump(mode="json")))
            # versions + environment are the two sections a reader needs MOST
            # when diagnosing a failure (which BLAS, which numpy, how many
            # threads) and both are available here for free -- the success
            # summary has carried them all along, and their absence from the
            # failure summary was an oversight, not a decision. Wrapped
            # separately so a broken environment probe reports itself instead
            # of masking the original exception.
            try:
                failure_rows.extend(("versions", name, version)
                                    for name, version in _version_rows())
                env = environment_record(jobs_requested=jobs_requested,
                                         jobs_effective=jobs)
                failure_rows.extend(("environment", key, env[key])
                                    for key in ENVIRONMENT_KEYS)
            except Exception as env_error:      # pragma: no cover - defensive
                failure_rows.append(
                    ("failure", "environment_probe_error", str(env_error)))
            failure_memory = {
                **mem.report(),
                "rss_scope": "driver_only; sampled at failure",
                "allocation_limit_configured_gb": config.runtime.memory_limit_gb,
            }
            if memory_budget is not None:
                failure_memory.update({
                    "allocation_limit_effective_gb": memory_budget[0] / 1e9,
                    "allocation_limit_source": memory_budget[1],
                    "physical_ram_gb": None if memory_budget[2] is None else memory_budget[2] / 1e9,
                })
            failure_rows.extend(summary_mapping_rows("memory", failure_memory))
            summary_path = outdir / config.output.csv.summary
            failure_path = summary_path.with_name(
                f"{summary_path.stem}.failed{summary_path.suffix}")
            write_summary_csv(failure_rows, failure_path)
        except Exception as report_error:
            log.error("Could not write failure summary: %s", report_error)
        raise
    finally:
        mem.stop()  # idempotent: guarantees the sampler thread is joined
        mem.log_summary()  # report the observed peak on success AND failure
        if fh is not None:
            logging.getLogger().removeHandler(fh)
            fh.close()
