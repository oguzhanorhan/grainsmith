"""Config loading, cross-field validation, and resolved-config echo.

Public API
----------
load_config(path)                    — load a YAML file and return a validated RunConfig
resolve_config(raw, base_dir=None)   — validate a raw dict and check cross-field rules
dump_resolved(config, path)          — write resolved config as YAML with provenance header
"""
from __future__ import annotations

import hashlib
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import yaml
from pydantic import BaseModel, ValidationError

from grainsmith.config.schema import RunConfig
from grainsmith.constants import ATOMIC_MASSES
from grainsmith.errors import ConfigError

if TYPE_CHECKING:
    from grainsmith.io.common import Provenance

log = logging.getLogger(__name__)


def _model_at(loc: tuple[Any, ...]) -> type[BaseModel] | None:
    """The nested config model at schema path *loc* (best effort).

    Walks RunConfig's field annotations, unwrapping Optional/list/dict
    wrappers to the contained BaseModel; integer parts (list indices)
    are skipped because the list wrapper was already unwrapped."""
    model: type[BaseModel] = RunConfig
    for part in loc:
        if isinstance(part, int):
            continue
        field = model.model_fields.get(part)
        if field is None:
            return None
        nxt, queue = None, [field.annotation]
        while queue and nxt is None:
            tp = queue.pop()
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                nxt = tp
            else:
                queue.extend(get_args(tp))
        if nxt is None:
            return None
        model = nxt
    return model


def _schema_error_message(exc: ValidationError) -> str:
    """Pydantic's message plus, for each misplaced key (extra_forbidden),
    the valid field names of the model it landed on — pydantic itself
    never enumerates them, which leaves slips like ``grains.type`` (for
    ``grains.size_distribution.type``) with no in-message way out."""
    msg = f"Schema validation failed: {exc}"
    hints = []
    for err in exc.errors():
        if err.get("type") != "extra_forbidden":
            continue
        loc = tuple(err.get("loc", ()))
        model = _model_at(loc[:-1])
        if model is None:
            continue
        parent = ".".join(str(p) for p in loc[:-1]) or "the top level"
        hints.append(
            f"Hint: '{'.'.join(str(p) for p in loc)}' is not a field; "
            f"valid fields under '{parent}' are: "
            + ", ".join(model.model_fields) + ".")
    if hints:
        msg += "\n" + "\n".join(hints)
    return msg


def resolve_config(
    raw: dict[str, Any], *, base_dir: Path | None = None,
) -> RunConfig:
    """Parse a raw config dict and enforce cross-field validation rules (§7).

    This function is the gate-G1 implementation: schema validation via
    Pydantic, followed by cross-field rules that require context spanning
    multiple sub-models.

    Parameters
    ----------
    raw : dict
        Mapping loaded from a YAML file (e.g. via ``yaml.safe_load``).
    base_dir : Path, optional
        Directory to fall back to when resolving ``crystal.cif.file`` /
        ``boundaries.voxel_import.file``: each is tried AS GIVEN first
        (absolute, or relative to the process's current working
        directory — unchanged, full backward compatibility), and only
        when that does not exist is a relative path retried as
        ``base_dir / path``. ``load_config`` passes the YAML file's own
        parent directory here so an example config's asset paths resolve
        regardless of the caller's CWD. A raw dict has no file location of
        its own to anchor a fallback against, so every dict-based
        ``resolve_config`` call site (tests, the studio UI's yaml_builder's
        ``str`` branch) omits it, and path fields are then resolved
        against the CWD only.

    Returns
    -------
    RunConfig
        Fully-validated Pydantic model ready for use by the pipeline.

    Raises
    ------
    ConfigError
        On Pydantic schema errors (unknown keys, wrong types, failed
        constraints) or cross-field constraint violations.
    """
    try:
        config = RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_schema_error_message(exc)) from exc
    except Exception as exc:
        raise ConfigError(f"Schema validation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Cross-field rules (§7)
    # ------------------------------------------------------------------

    # Rule 1: seed.mode == "fixed" requires seed.value
    if config.seed.mode == "fixed" and config.seed.value is None:
        raise ConfigError(
            "seed.mode is 'fixed' but seed.value is not set. "
            "Provide an integer, for example:\n"
            "  seed:\n"
            "    mode: fixed\n"
            "    value: 42"
        )

    # Rule 16: grains.number is required for every GENERATED
    # tessellation; only voxel_import derives it from the label field
    # (a provided value is cross-checked against the derived count at
    # run time).  Checked early — later rules compare against it.
    if (config.grains.number is None
            and config.boundaries.geometry != "voxel_import"):
        raise ConfigError(
            "grains.number is required (it is derived from the label "
            "field only for boundaries.geometry 'voxel_import')."
        )

    # Rule 21: exactly one of 'crystal' / 'phases'.  Checked
    # early — later rules touch the crystal definition(s).
    if (config.crystal is None) == (config.phases is None):
        raise ConfigError(
            "Exactly one of 'crystal' (single phase) or 'phases' "
            "(multiphase) must be set."
            if config.crystal is None else
            "'crystal' and 'phases' are mutually exclusive — for a "
            "multiphase run, move the crystal definition(s) into the "
            "per-phase 'crystal' blocks."
        )
    if config.phases is not None:
        if len(config.phases) < 2:
            raise ConfigError(
                "phases needs >= 2 entries — for a single phase use the "
                "'crystal' shortcut."
            )
        names = [p.name for p in config.phases]
        if len(set(names)) != len(names):
            raise ConfigError(
                f"phase names must be unique, got {names}."
            )
        frac_sum = sum(p.fraction for p in config.phases)
        if abs(frac_sum - 1.0) > 1e-6:
            raise ConfigError(
                "phase volume fractions must sum to 1 ± 1e-6, got "
                f"{frac_sum!r} for "
                f"{ {p.name: p.fraction for p in config.phases} }."
            )

    # Rule 2: vacuum > 0 requires at least one non-periodic axis
    if config.box.vacuum > 0.0 and all(config.box.periodic):
        raise ConfigError(
            f"box.vacuum={config.box.vacuum} Å but all axes are periodic. "
            "Set at least one entry in box.periodic to false for a "
            "slab / thin-film setup (e.g. periodic: [true, true, false])."
        )

    # Rule 3: from_list length == grains.number — enforced in the
    # orientation stage (pipeline._stage_orientation) with a clear message.

    # Rule 27: crystal.cif is mutually exclusive with the manual
    # space_group/lattice/wyckoff_sites trio; when given, the CIF is read
    # (ASE) and its symmetry detected (spglib) HERE so it fills in the
    # SAME space_group/lattice/wyckoff_sites fields the manual path uses
    # — every downstream stage (orientation, tessellation, fill, LAMMPS
    # writer) then runs completely unchanged. Applied to the single-phase
    # 'crystal' block AND to every phase's 'crystal' block.
    if config.crystal is not None:
        _resolve_crystal_cif(config.crystal, "crystal", base_dir=base_dir)
    if config.phases is not None:
        for p in config.phases:
            _resolve_crystal_cif(p.crystal, f"phases[{p.name}].crystal",
                                 base_dir=base_dir)

    # Rule 28: box.lengths / box.cells mutual exclusivity and the
    # box.cells (triclinic lattice-multiple single-crystal box)
    # preconditions.  Runs AFTER Rule 27 so config.crystal.lattice is
    # resolved for both the CIF and manual input paths.
    _resolve_box_cells(config)

    # Rule 4: csl=true requires a cubic point group (§6.10).  The
    # cubic crystal FAMILY is checkable here (SG 195–230); the precise
    # 24-proper-rotation requirement (rules out T/Th/Td classes) is
    # re-checked at analysis time against the actual point group.
    # With phases, CSL is undefined across phase pairs → Rule 22.
    _sg_csl = config.crystal.space_group if config.crystal is not None else None
    if config.analysis.csl and _sg_csl is not None and not (
            195 <= _sg_csl.number <= 230):
        raise ConfigError(
            f"analysis.csl is restricted to cubic point groups, but "
            f"space group {_sg_csl.number} is not cubic "
            "(195–230). Set analysis.csl: false."
        )

    # Rule 25: gb_curvature requires a differentiable margin — the
    # voxel_import backend's margin is a piecewise-constant nearest-voxel
    # lookup, so level-set derivatives are meaningless.
    if (config.analysis.gb_curvature
            and config.boundaries.geometry == "voxel_import"):
        raise ConfigError(
            "analysis.gb_curvature is not supported with "
            "boundaries.geometry 'voxel_import': the imported-label "
            "margin is piecewise-constant, so curvature derivatives "
            "are meaningless. Disable gb_curvature or use a warp "
            "backend."
        )

    # Rule 5: midpoint_merge + multi-species — deferred to overlap stage.
    # The overlap module inspects the assembled AtomBlock for species count.

    # Rule 5b: per-site occupancies must sum to 1 ± 1e-6 with positive
    # fractions (§7) — applied to the single crystal AND to every phase.
    if config.crystal is not None:
        crystal_blocks = [("crystal", config.crystal)]
    else:
        assert config.phases is not None  # Rule 21
        crystal_blocks = [(f"phases[{p.name}].crystal", p.crystal)
                          for p in config.phases]
    for label, crys in crystal_blocks:
        assert crys.wyckoff_sites is not None  # Rule 27 resolved this
        for k, site in enumerate(crys.wyckoff_sites):
            if not isinstance(site.element, dict):
                continue
            if any((not math.isfinite(x)) or x <= 0.0
                   for x in site.element.values()):
                raise ConfigError(
                    f"{label}.wyckoff_sites[{k}]: occupancy fractions must "
                    f"be finite and positive, got {site.element}."
                )
            total = sum(site.element.values())
            if abs(total - 1.0) > 1e-6:
                raise ConfigError(
                    f"{label}.wyckoff_sites[{k}]: occupancies must sum to "
                    f"1 ± 1e-6, got {total!r} for {site.element}."
                )

    # Rule 6: geometry=flat but curved block present → hard error. The
    # block would be silently ignored by the flat tessellation branch and
    # the run would come out flat while resolved_config.yaml still echoes
    # the curved parameters — honest refusal beats silent ignoring.
    if (
        config.boundaries.geometry == "flat"
        and config.boundaries.curved is not None
    ):
        raise ConfigError(
            "boundaries.geometry is 'flat' but a 'curved' block is "
            "present; the block would be silently ignored. Set "
            "boundaries.geometry: 'curved' to use it, or remove the "
            "curved block."
        )

    # Rules 7–9: curved-boundary parameter constraints (§6.6, §7).
    # min_seed_distance resolves to 1.0·r_ws when "auto".
    if config.boundaries.geometry == "curved" and config.boundaries.curved is not None:
        curved = config.boundaries.curved
        from grainsmith.seeding import wigner_seitz_radius

        assert config.grains.number is not None  # Rule 16 (geometry != voxel_import)
        volume = float(np.prod(np.asarray(config.box.lengths, dtype=np.float64)))
        msd = config.grains.min_seed_distance
        if msd == "auto":
            msd = wigner_seitz_radius(volume, config.grains.number)

        if curved.method == "warp":
            # Rule 8a (hard): 'warp' + spectrum 'self_affine' is rejected
            # outright, not merely deprecated — warp is a coordinate
            # diffeomorphism (grain_of(x) = base.grain_of(x + u(x)),
            # bijective under the G6 guard); a bijective map cannot change
            # the box-counting dimension of the surface it
            # displaces, so giving u a self-affine PSD only recolors
            # the waviness of a boundary whose box-counting dimension it
            # cannot change (see docs/physics.md
            # §5b for the full argument and the measured field-correlation/
            # cross-section numbers). Checked FIRST, ahead of Rules 7/8
            # below, so an invalid method+spectrum combination fails on
            # its own terms rather than on an unrelated amplitude
            # tolerance.
            if curved.spectrum == "self_affine":
                raise ConfigError(
                    "boundaries.curved.method 'warp' does not accept spectrum "
                    "'self_affine' (removed: warp is a coordinate "
                    "diffeomorphism and cannot produce a genuinely self-affine "
                    "boundary regardless of Hurst exponent — see "
                    "docs/physics.md §5b). For a self-affine grain boundary, use "
                    "boundaries.curved.method: perturbed_distance (spectrum: "
                    "self_affine is supported there). For warp's own boundary "
                    "waviness, use spectrum: gaussian + correlation_length "
                    "instead."
                )

            a_clip = msd / 4.0
            # Rule 7 (hard): seed-containment proof requires A_rms ≤ A_clip.
            if curved.amplitude > a_clip:
                raise ConfigError(
                    f"boundaries.curved.amplitude={curved.amplitude:.4g} Å exceeds "
                    f"min_seed_distance/4 = {a_clip:.4g} Å (seed-containment guard, "
                    "§6.6 G6). Reduce amplitude, increase min_seed_distance, or "
                    "use fewer grains."
                )
            # Rule 8 (warn): bijectivity guard likely to trip. (Rule 8a above
            # already excludes self_affine for warp, so spectrum is guaranteed
            # 'gaussian' here — correlation_length always applies.)
            if curved.amplitude > 0.3 * curved.correlation_length:
                log.warning(
                    f"boundaries.curved.amplitude={curved.amplitude:.4g} Å > "
                    f"0.3·correlation_length={0.3 * curved.correlation_length:.4g} Å; "
                    "the warp bijectivity guard (G6) may trip at run time."
                )

        if curved.method == "additive_weights":
            # Rule 9 (hard): empty-cell guard (§6.6).
            limit = msd / 6.0
            if curved.weight_sigma > limit:
                raise ConfigError(
                    f"boundaries.curved.weight_sigma={curved.weight_sigma:.4g} Å "
                    f"exceeds min_seed_distance/6 = {limit:.4g} Å (empty-cell "
                    "guard, §6.6). Reduce weight_sigma."
                )

        if curved.method == "perturbed_distance":
            # Rule 7b (hard): seed-containment guard, perturbed_distance's
            # analogue of Rule 7 above — NOT the same formula as warp's
            # A_clip=msd/4, because this method has no bijectivity guard
            # (G6) to fall back on; the exact seed-ownership check is
            # done at CONSTRUCTION time (perturbed.py), where it can
            # actually evaluate grain_of() — this is only the cheap
            # pre-flight sufficient condition.
            from grainsmith.constants import ETA_CLIP, PERTURBED_DISTANCE_SAFETY

            a_max = PERTURBED_DISTANCE_SAFETY * msd / (2.0 * ETA_CLIP)
            if curved.amplitude > a_max:
                raise ConfigError(
                    f"boundaries.curved.amplitude={curved.amplitude:.4g} Å "
                    f"exceeds {PERTURBED_DISTANCE_SAFETY:g}·min_seed_distance"
                    f"/(2·ETA_CLIP) = {a_max:.4g} Å (perturbed_distance "
                    "seed-containment guard). Reduce amplitude, increase "
                    "min_seed_distance, or use fewer grains."
                )

        # Rule 20 (restricted to perturbed_distance only
        # -- warp+self_affine is REJECTED earlier by Rule 8a above with its
        # own diffeomorphism-specific message, so by the time we reach here
        # curved.method == "warp" can never coexist with spectrum ==
        # "self_affine"; this check now only guards the OTHER two curved
        # methods, additive_weights/anisotropic, which never consume a
        # spectral field at all). self_affine band constraints (l_min <
        # l_max <= min(L)/2) still apply verbatim -- see perturbed.py's
        # module docstring for why perturbed_distance is the only method
        # whose as-built boundary responds to this spectrum.
        if curved.spectrum == "self_affine":
            if curved.method != "perturbed_distance":
                raise ConfigError(
                    "boundaries.curved.spectrum 'self_affine' applies to "
                    "the perturbed_distance method only (got method "
                    f"{curved.method!r}). 'warp' never accepts "
                    "spectrum 'self_affine' (see the boundaries.curved."
                    "method 'warp' error above / docs/physics.md §5b)."
                )
            if curved.l_min >= curved.l_max:
                raise ConfigError(
                    f"self_affine band requires l_min < l_max, got "
                    f"l_min={curved.l_min:g} ≥ l_max={curved.l_max:g} Å."
                )
            l_cap = float(np.min(np.asarray(config.box.lengths))) / 2.0
            if curved.l_max > l_cap:
                raise ConfigError(
                    f"self_affine l_max={curved.l_max:g} Å exceeds "
                    f"min(box.lengths)/2 = {l_cap:g} Å — wavelengths "
                    "longer than half the box are not representable."
                )
            # l_min ≥ 2·h_field is checked against the ACTUAL grid in
            # WarpTessellation / PerturbedDistanceTessellation (the grid
            # is resolved there).

        # Rule 29 (hard): amplitude_convention /
        # reference_wavelength cross-field checks + the seed-containment
        # guard re-check under the new convention. Placed AFTER the
        # self_affine block above so that, whenever spectrum ==
        # "self_affine" is required below, Rule 20 has already confirmed
        # l_min < l_max ≤ min(box.lengths)/2 — the (c) range check below
        # can therefore trust that interval.
        if (curved.amplitude_convention == "total_rms"
                and curved.reference_wavelength is not None):
            # (a) a silently-ignored field is a config error here, not a
            # no-op (project convention: extra='forbid' everywhere else).
            raise ConfigError(
                "boundaries.curved.reference_wavelength is set but "
                "amplitude_convention is 'total_rms' (reference_wavelength "
                "is only meaningful under amplitude_convention: "
                "'reference_wavelength'). Either drop reference_wavelength "
                "or set amplitude_convention: reference_wavelength."
            )
        if curved.amplitude_convention == "reference_wavelength":
            # (b) the reference-octave anchor only has meaning for a
            # band-limited power-law spectrum — checked before (c) so a
            # simultaneous spectrum+range mistake reports the more
            # fundamental problem first.
            if curved.spectrum != "self_affine":
                raise ConfigError(
                    "boundaries.curved.amplitude_convention "
                    "'reference_wavelength' requires spectrum: "
                    f"'self_affine' (got spectrum={curved.spectrum!r}); "
                    "the 'gaussian' spectrum has no octave/band structure "
                    "for a reference-octave anchor to be defined against."
                )
            # (c) explicit reference_wavelength must lie in [l_min, l_max]
            # (None means "use l_max", always in-range by construction).
            if curved.reference_wavelength is not None and not (
                curved.l_min <= curved.reference_wavelength <= curved.l_max
            ):
                raise ConfigError(
                    f"boundaries.curved.reference_wavelength="
                    f"{curved.reference_wavelength:g} Å must lie within "
                    f"[l_min, l_max] = [{curved.l_min:g}, "
                    f"{curved.l_max:g}] Å."
                )
            # (d) seed-containment guard, RE-CHECKED against the REALIZED
            # (total-RMS-equivalent) amplitude — the convention switch
            # must not weaken the guard. kappa ≤ 1 always (a reference
            # octave's own variance cannot exceed the whole band's), so
            # amplitude/kappa ≥ amplitude: this is a STRICTER re-check
            # than Rule 7b's plain `curved.amplitude > a_max` above (that
            # check alone is necessary but not sufficient once `amplitude`
            # means A0 rather than the total-RMS amplitude). Only
            # 'perturbed_distance' reaches here (self_affine ⇒
            # perturbed_distance is already enforced above), so `a_max`
            # (undefined in the 'warp'/'additive_weights' branches above)
            # is always in scope by the time this line runs.
            if curved.method == "perturbed_distance":
                from grainsmith.tessellation.warp import reference_shell_kappa

                kappa = reference_shell_kappa(
                    curved.hurst, curved.l_min, curved.l_max,
                    curved.reference_wavelength,
                )
                equivalent_amplitude = curved.amplitude / kappa
                if equivalent_amplitude > a_max:
                    raise ConfigError(
                        f"perturbed_distance amplitude={curved.amplitude:.4g} "
                        "Å under amplitude_convention: reference_wavelength "
                        f"is equivalent to a total-RMS amplitude of "
                        f"{equivalent_amplitude:.4g} Å (÷ κ={kappa:.4g}), "
                        f"which exceeds {PERTURBED_DISTANCE_SAFETY:g}·"
                        f"min_seed_distance/(2·ETA_CLIP) = {a_max:.4g} Å "
                        "(perturbed_distance seed-containment guard, "
                        "binding on the REALIZED field amplitude "
                        "regardless of convention). Reduce amplitude, "
                        "increase min_seed_distance, or use fewer grains."
                    )

    # Rules 10–12: grain-size distribution targeting.
    size_dist = config.grains.size_distribution
    if size_dist is not None:
        # Rule 10: volume control exists only for the power diagram —
        # flat geometry or the warp base; the other curved methods have no
        # volume handle (honest refusal beats silent ignoring).
        if config.boundaries.geometry == "voxel_import":
            raise ConfigError(
                "grains.size_distribution cannot be combined with "
                "boundaries.geometry 'voxel_import': the imported label "
                "field fixes the grain volumes."
            )
        if config.boundaries.geometry == "curved":
            curved_blk = config.boundaries.curved
            method = curved_blk.method if curved_blk is not None else "warp"
            if method != "warp":
                raise ConfigError(
                    "grains.size_distribution requires flat geometry or "
                    f"the warp method (got curved method {method!r}): only "
                    "the power/Laguerre diagram offers volume control."
                )
        # Rule 11: explicit volumes must match the grain count.
        if size_dist.type == "volumes":
            n_vol = 0 if size_dist.volumes is None else len(size_dist.volumes)
            if n_vol != config.grains.number:
                raise ConfigError(
                    "grains.size_distribution.volumes must list exactly "
                    f"grains.number={config.grains.number} values, "
                    f"got {n_vol}."
                )
            if size_dist.volumes is not None and any(
                    v <= 0.0 for v in size_dist.volumes):
                raise ConfigError(
                    "grains.size_distribution.volumes must all be positive."
                )
    # Rule 12: a 'power' warp base needs targets to fit.
    if (config.boundaries.geometry == "curved"
            and config.boundaries.curved is not None
            and config.boundaries.curved.base == "power"
            and size_dist is None):
        raise ConfigError(
            "boundaries.curved.base 'power' requires "
            "grains.size_distribution (the Laguerre weights are fitted to "
            "the target volumes)."
        )

    # Rules 13–15: texture / MDF targeting.
    # Rule 13: odf_components needs a well-formed component list.
    if config.orientation.scheme == "odf_components":
        comps = config.orientation.components
        if not comps:
            raise ConfigError(
                "orientation.scheme is 'odf_components' but the "
                "'components' list is missing or empty."
            )
        n_random = 0
        for k, comp in enumerate(comps):
            kinds = [name for name, set_ in (
                ("euler_bunge_deg", comp.euler_bunge_deg is not None),
                ("fiber", comp.fiber is not None),
                ("random", comp.random),
            ) if set_]
            if len(kinds) != 1:
                raise ConfigError(
                    f"orientation.components[{k}] must set exactly one of "
                    f"euler_bunge_deg, fiber, random — got {kinds or 'none'}."
                )
            if comp.random:
                n_random += 1
                if comp.spread_deg != 0.0:
                    raise ConfigError(
                        f"orientation.components[{k}]: spread_deg is "
                        "meaningless for a random (Haar-uniform) component."
                    )
        if n_random > 1:
            raise ConfigError(
                "orientation.components may contain at most one "
                "'random: true' component (a single uniform fraction)."
            )
    elif config.orientation.components is not None:
        log.warning(
            "orientation.components is set but scheme is %r — the "
            "components list will be ignored.", config.orientation.scheme,
        )

    mdf = config.orientation.mdf_target
    if mdf is not None:
        # Rule 14: histogram targets must be a valid density table.
        if mdf.type == "histogram":
            if mdf.bin_edges is None or mdf.densities is None:
                raise ConfigError(
                    "orientation.mdf_target type 'histogram' requires both "
                    "bin_edges and densities."
                )
            if len(mdf.bin_edges) != len(mdf.densities) + 1:
                raise ConfigError(
                    "orientation.mdf_target: len(bin_edges) must equal "
                    f"len(densities) + 1, got {len(mdf.bin_edges)} edges "
                    f"for {len(mdf.densities)} densities."
                )
            edges = np.asarray(mdf.bin_edges, dtype=np.float64)
            if edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
                raise ConfigError(
                    "orientation.mdf_target.bin_edges must be "
                    "non-negative and strictly increasing (degrees)."
                )
            dens = np.asarray(mdf.densities, dtype=np.float64)
            if np.any(dens < 0.0) or float(np.sum(dens)) <= 0.0:
                raise ConfigError(
                    "orientation.mdf_target.densities must be non-negative "
                    "with a positive sum."
                )
        # NOTE (config_sha256 stability): `mdf.type` is deliberately never
        # rewritten to its canonical name here — config_sha256 is the
        # sha256 of THIS dumped resolved config (`_config_yaml` below), so
        # normalising a deprecated alias in place would change the
        # recorded hash of every archived run that used the old name and
        # break its reproducibility. Warn only; the value the user wrote
        # is echoed back verbatim in resolved_config.yaml.
        _sg_mdf = (config.crystal.space_group
                   if config.crystal is not None else None)
        _sg_mdf_cubic = _sg_mdf is not None and 195 <= _sg_mdf.number <= 230
        if mdf.type == "csl_enriched":
            warnings.warn(
                "orientation.mdf_target type 'csl_enriched' is deprecated "
                "in favour of 'sigma3_angle_enriched'. The objective "
                "enforces the Σ3 Brandon ANGULAR window (60° ± 15°/√3) "
                "only — the CSL ⟨111⟩ axis condition is not part of the "
                "energy.",
                DeprecationWarning, stacklevel=2,
            )
        if mdf.type == "mackenzie":
            if _sg_mdf is not None and not _sg_mdf_cubic:
                warnings.warn(
                    "orientation.mdf_target type 'mackenzie' is deprecated "
                    "in favour of 'haar_random'. The configured point "
                    f"group is not cubic (space group {_sg_mdf.number}, "
                    "not in 195–230): the reference is the Haar-random "
                    "disorientation-angle curve of THAT point group, "
                    "which equals Mackenzie's 1958 cubic law only for the "
                    "cubic proper point group.",
                    DeprecationWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    "orientation.mdf_target type 'mackenzie' is deprecated "
                    "in favour of 'haar_random'.",
                    DeprecationWarning, stacklevel=2,
                )
        # Rule 15: Σ3 enrichment is meaningful for cubic crystals only
        # (same family check as analysis.csl, Rule 4).  With phases the
        # whole mdf_target block is rejected by Rule 22. Covers both the
        # canonical name and its deprecated 'csl_enriched' alias.
        if (mdf.type in ("csl_enriched", "sigma3_angle_enriched")
                and _sg_mdf is not None and not _sg_mdf_cubic):
            raise ConfigError(
                f"orientation.mdf_target type {mdf.type!r} targets the "
                "Σ3 CSL class, which is defined for cubic point groups "
                f"only — space group {_sg_mdf.number} "
                "is not cubic (195–230)."
            )

    # Rules 17–19: imported voxel fields.
    vi = config.boundaries.voxel_import
    if config.boundaries.geometry == "voxel_import":
        # Rule 17: the import block (with an existing file of a known
        # format) is required; HDF5 needs the dataset path.
        if vi is None:
            raise ConfigError(
                "boundaries.geometry is 'voxel_import' but the "
                "'voxel_import' block is missing."
            )
        resolved, used_fallback = _resolve_path_with_base_dir(
            vi.file, base_dir, "boundaries.voxel_import.file")
        if used_fallback:
            # Rewritten (not left as-given): unlike crystal.cif.file
            # (consumed here in resolve_config, via _resolve_crystal_cif,
            # well before the pipeline ever runs), vi.file is read again
            # later at tessellation-build time (pipeline._stage_voxel_import)
            # by code that has no base_dir of its own — the rewrite is
            # what keeps that later read (and a redump/reload of
            # resolved_config.yaml) working from any CWD.
            vi.file = resolved
        path = Path(vi.file)
        suffix = path.suffix.lower()
        if suffix not in (".npy", ".dream3d", ".h5", ".hdf5"):
            raise ConfigError(
                "boundaries.voxel_import.file must be .npy or DREAM.3D "
                f"HDF5 (.dream3d/.h5/.hdf5), got {suffix!r}."
            )
        if suffix != ".npy" and vi.dataset is None:
            raise ConfigError(
                "boundaries.voxel_import.dataset (HDF5 path to the "
                "FeatureIds cell array) is required for DREAM.3D files."
            )
        if suffix == ".npy" and vi.euler_dataset is not None:
            raise ConfigError(
                "boundaries.voxel_import.euler_dataset requires a "
                "DREAM.3D HDF5 file — .npy carries no orientation data."
            )
        # Rule 18: generation-only grain controls are ignored — warn.
        if config.grains.lloyd_iterations > 0:
            log.warning(
                "grains.lloyd_iterations is ignored for voxel_import "
                "(the imported field is not re-seeded).")
        if config.boundaries.curved is not None:
            log.warning(
                "boundaries.curved is ignored for voxel_import geometry.")
    elif vi is not None:
        log.warning(
            "boundaries.voxel_import is set but geometry is %r — the "
            "import block will be ignored.", config.boundaries.geometry,
        )

    # Rule 19: imported orientations need an imported field + Euler data.
    if config.orientation.scheme == "imported":
        if config.boundaries.geometry != "voxel_import" or vi is None:
            raise ConfigError(
                "orientation.scheme 'imported' requires "
                "boundaries.geometry 'voxel_import' (per-grain Euler "
                "angles come from the imported file)."
            )
        if vi.euler_dataset is None:
            raise ConfigError(
                "orientation.scheme 'imported' requires "
                "boundaries.voxel_import.euler_dataset (the per-grain "
                "Bunge Euler array in the DREAM.3D file)."
            )

    # Rule 22: multiphase restrictions — quantities defined for
    # ONE point group (CSL, MDF targets, frame-dependent orientation
    # specs) are rejected honestly; per-phase texture and interphase
    # orientation relationships are the documented follow-up.
    if config.phases is not None:
        if config.analysis.csl:
            raise ConfigError(
                "analysis.csl is undefined for multiphase runs (CSL Σ "
                "needs a single point group; interphase boundaries have "
                "none). Set analysis.csl: false."
            )
        if config.orientation.mdf_target is not None:
            raise ConfigError(
                "orientation.mdf_target cannot be combined with phases: "
                "the misorientation distribution is defined within one "
                "point group only (same-phase boundaries). Per-phase MDF "
                "targeting is a planned extension."
            )
        if config.orientation.scheme not in ("random_uniform", "from_list"):
            raise ConfigError(
                "With phases, orientation.scheme must be 'random_uniform' "
                f"or 'from_list' (got {config.orientation.scheme!r}): "
                "fixed/fiber/odf_components/imported need a single crystal "
                "frame. Per-phase texture is a planned extension."
            )
        if (config.orientation.scheme == "from_list"
                and config.orientation.from_list is not None):
            for k, entry in enumerate(config.orientation.from_list):
                if entry.hkl_uvw is not None:
                    raise ConfigError(
                        f"orientation.from_list[{k}]: hkl_uvw entries are "
                        "not allowed with phases (Miller indices need a "
                        "per-phase frame choice) — use euler_bunge_deg, "
                        "axis_angle, or quaternion."
                    )
        # Rule 23: enough grains to give every phase at least one.
        if (config.grains.number is not None
                and config.grains.number < len(config.phases)):
            raise ConfigError(
                f"grains.number={config.grains.number} is smaller than "
                f"the number of phases ({len(config.phases)}) — every "
                "phase needs at least one grain."
            )

    # Rule 24: grains.number == 1 selects the single-crystal
    # backend (flat geometry only); grain-statistics machinery that needs
    # >= 2 grains is rejected.  voxel_import keeps its own path (a
    # 1-grain imported field stays a VoxelTessellation).
    if config.grains.number == 1:
        if config.boundaries.geometry == "curved":
            raise ConfigError(
                "grains.number: 1 (single crystal) requires "
                "boundaries.geometry 'flat': curving the boundary of a "
                "single periodic cell is meaningless — the only "
                "'boundary' is the box-face self-image."
            )
        if config.grains.size_distribution is not None:
            raise ConfigError(
                "grains.size_distribution is meaningless for a single "
                "crystal (grains.number: 1)."
            )
        if config.orientation.mdf_target is not None:
            raise ConfigError(
                "orientation.mdf_target needs >= 2 grains "
                "(a single crystal has no grain boundaries)."
            )
        if config.grains.lloyd_iterations > 0:
            log.warning(
                "grains.lloyd_iterations is ignored for a single crystal "
                "(the seed sits at the box centre).")

    # Rule 26 (doping): single-phase only; per-dopant
    # mode coherence; preset ↔ space-group compatibility; dopant mass
    # known; substitutional host present in the host crystal.
    if config.doping is not None:
        if config.phases is not None:
            raise ConfigError(
                "doping is not supported together with 'phases': "
                "interstitial presets and concentration targets are "
                "defined against a single host crystal. Remove one of "
                "the two sections."
            )
        # Rule 21 guarantees exactly one of 'crystal'/'phases' is set;
        # phases was just rejected above, so crystal must be present.
        assert config.crystal is not None
        assert config.crystal.space_group is not None    # Rule 27 resolved this
        assert config.crystal.wyckoff_sites is not None  # Rule 27 resolved this
        sg = config.crystal.space_group.number
        host_species: set[str] = set()
        for site in config.crystal.wyckoff_sites:
            el = site.element
            host_species |= (set(el) if isinstance(el, dict) else {el})
        masses_override = config.output.lammps.masses or {}
        preset_sg = {"fcc": 225, "bcc": 229, "hcp": 194}
        for k, d in enumerate(config.doping.dopants):
            where = f"doping.dopants[{k}] ({d.element})"
            if (d.element not in ATOMIC_MASSES
                    and d.element not in masses_override):
                raise ConfigError(
                    f"{where}: no atomic mass known for {d.element!r}. "
                    "Add it to output.lammps.masses, e.g.\n"
                    "  output:\n    lammps:\n      masses: {"
                    + d.element + ": 55.0}"
                )
            if d.mode == "interstitial":
                if d.sites is None or d.min_distance is None:
                    raise ConfigError(
                        f"{where}: interstitial mode requires both "
                        "'sites' (preset or coords) and 'min_distance'."
                    )
                if isinstance(d.sites, str):
                    need = preset_sg[d.sites.split("_")[0]]
                    if sg != need:
                        raise ConfigError(
                            f"{where}: preset '{d.sites}' requires "
                            f"space group {need}, but the crystal uses "
                            f"{sg}. Use explicit sites.coords for other "
                            "structures."
                        )
                else:
                    for c in d.sites.coords:
                        if len(c) != 3 or not all(
                                0.0 <= float(x) < 1.0 for x in c):
                            raise ConfigError(
                                f"{where}: each sites.coords entry needs "
                                f"exactly 3 fractional components in "
                                f"[0, 1), got {c!r}."
                            )
                    # Run the orbit expansion NOW so a bad space-group/
                    # setting combination fails at config time with this
                    # dopant's label, not deep inside the doping stage;
                    # also reports the expanded sublattice size up front
                    # (traceability: no silent site-count changes).
                    from grainsmith.atoms.doping import resolve_sites
                    frac = resolve_sites(
                        d.sites, sg,
                        f"{where} (config-time check)",
                        sg_setting=config.crystal.space_group.setting)
                    if d.sites.expand_orbit:
                        log.info(
                            "%s: %d listed interstitial site(s) expand "
                            "to %d site(s)/cell under SG %d.",
                            where, len(d.sites.coords), len(frac), sg)
            else:  # substitutional
                if d.sites is not None or d.min_distance is not None:
                    raise ConfigError(
                        f"{where}: 'sites' and 'min_distance' apply to "
                        "interstitial mode only."
                    )
                if d.host is not None and d.host not in host_species:
                    raise ConfigError(
                        f"{where}: host {d.host!r} is not a species of "
                        f"the host crystal {sorted(host_species)}."
                    )

    return config


def _resolve_box_cells(config: RunConfig) -> None:
    """Rule 28: ``box.lengths`` / ``box.cells`` mutual exclusivity,
    and every precondition of the ``box.cells`` (triclinic
    lattice-multiple, single-crystal-only) box.

    Runs AFTER Rule 27 (``_resolve_crystal_cif``), so ``config.crystal``
    carries a resolved ``lattice`` block for both the CIF and manual input
    paths by the time this executes.

    On success for the ``cells`` path, this WRITES the resolved,
    LAMMPS-tilt-reduced (3,3) box matrix onto ``config.box.resolved_h``
    (row-major nested list — the same "derived cache written back onto the
    config" pattern Rule 27 uses for CIF-derived crystal fields) so the
    pipeline never has to recompute or re-derive it, and reruns/YAML
    round-trips are bit-identical.
    """
    box = config.box
    if (box.lengths is None) == (box.cells is None):
        raise ConfigError(
            "Exactly one of box.lengths (orthogonal box) or box.cells "
            "(triclinic lattice-multiple single-crystal box) must "
            "be set."
            if box.lengths is None else
            "box.lengths and box.cells are mutually exclusive: box.cells "
            "already determines every box vector from the crystal's own "
            "cell matrix — remove one of the two."
        )
    if box.cells is None:
        return  # orthogonal path (box.lengths) — nothing further here

    # --- box.cells preconditions ---
    if config.grains.number != 1:
        raise ConfigError(
            "box.cells requires grains.number: 1 — triclinic support "
            "is single-crystal only; the polycrystal tessellation core "
            "stays orthogonal (an arbitrarily oriented grain boundary "
            "network has no natural lattice-multiple box)."
        )
    if config.phases is not None:
        raise ConfigError(
            "box.cells is not supported with 'phases': it derives ONE box "
            "matrix from ONE crystal cell — use the single-phase "
            "'crystal' block."
        )
    if not all(box.periodic):
        raise ConfigError(
            "box.cells requires box.periodic: [true, true, true] — the "
            "lattice-multiple box is exactly periodic BY CONSTRUCTION; a "
            "free axis has no periodic image for the box vector to be "
            "commensurate with, so vacuum/free-axis slabs must use "
            "box.lengths instead."
        )
    if box.vacuum > 0.0:
        raise ConfigError(
            "box.cells is incompatible with box.vacuum > 0 (box.cells "
            "requires box.periodic all true — see above; vacuum needs a "
            "free axis)."
        )
    if config.doping is not None:
        raise ConfigError(
            "box.cells is not supported together with 'doping': "
            "the doping module's periodic neighbor search assumes an "
            "orthogonal box. Remove one of the two sections."
        )
    if config.boundaries.geometry != "flat":
        raise ConfigError(
            "box.cells requires boundaries.geometry: 'flat' (Rule 24 "
            "already requires this for any single crystal, grains.number: "
            "1)."
        )
    if config.analysis.gb_curvature:
        raise ConfigError(
            "box.cells is incompatible with analysis.gb_curvature: a "
            "single crystal has no grain boundary to curve (Rule 24 "
            "already excludes curved geometry)."
        )
    if config.analysis.section is not None:
        raise ConfigError(
            "box.cells is not supported with analysis.section: the "
            "EBSD-like slice sampler (io/publication.write_section_csv) "
            "assumes an axis-aligned orthogonal sampling grid over "
            "box.lengths, which is not the correct sampling domain for a "
            "tilted parallelepiped. Remove analysis.section or use "
            "box.lengths instead of box.cells."
        )

    assert config.crystal is not None  # Rule 21 (phases excluded above)
    assert config.crystal.lattice is not None  # Rule 27 resolved this

    # --- build the crystal's own conventional cell matrix A ---
    from grainsmith.constants import TRICLINIC_TILT_BOUND_TOL
    from grainsmith.crystal.cell import (
        cell_matrix,
        family_from_sg,
        reduce_triclinic_tilts,
        validate_cellpar,
    )

    latt = config.crystal.lattice
    sg = config.crystal.space_group
    assert sg is not None  # Rule 27 resolved this too

    family = family_from_sg(sg.number, sg.setting)
    params = {
        k: v for k, v in {
            "a": latt.a, "b": latt.b, "c": latt.c,
            "alpha": latt.alpha, "beta": latt.beta, "gamma": latt.gamma,
        }.items() if v is not None
    }
    cellpar = validate_cellpar(family, params)
    A = cell_matrix(cellpar.a, cellpar.b, cellpar.c,
                    cellpar.alpha, cellpar.beta, cellpar.gamma)

    # Orientation MUST be identity: box.cells builds the box vectors
    # directly from the crystal's OWN (unrotated) conventional cell, so
    # any orientation other than identity would misalign the lattice with
    # the box/lab frame the box vectors were built in — breaking exact
    # commensurability by construction rather than achieving it.
    o = config.orientation
    if o.scheme != "fixed" or o.fixed is None:
        raise ConfigError(
            "box.cells requires orientation.scheme: 'fixed' with an "
            "IDENTITY rotation (e.g. fixed.euler_bunge_deg: [0, 0, 0]) — "
            "the box vectors are built directly from the crystal's own "
            "unrotated cell matrix, so any other scheme or a non-identity "
            "fixed orientation would misalign the lattice with the box."
        )
    from grainsmith.orientation.quaternion import quat_to_matrix
    from grainsmith.orientation.samplers import fixed_orientation

    spec = {k: v for k, v in o.fixed.model_dump().items() if v is not None}
    if len(spec) != 1:
        raise ConfigError(
            "orientation.fixed must set exactly one of hkl_uvw, "
            f"euler_bunge_deg, axis_angle, quaternion (got {sorted(spec)})."
        )
    q = fixed_orientation(1, spec, A)[0]
    R = quat_to_matrix(q)
    if not np.allclose(R, np.eye(3), atol=1e-9):
        raise ConfigError(
            "box.cells requires an IDENTITY orientation.fixed spec (got a "
            f"rotation matrix deviating from identity by "
            f"{float(np.max(np.abs(R - np.eye(3)))):.3g}) — use "
            "euler_bunge_deg: [0, 0, 0], axis_angle angle_deg: 0, or "
            "quaternion: [1, 0, 0, 0]."
        )

    # --- build the reduced triclinic box matrix H = A @ diag(n1,n2,n3) ---
    n1, n2, n3 = box.cells
    H = A @ np.diag([float(n1), float(n2), float(n3)])
    H = reduce_triclinic_tilts(H)

    ax_edge, by_edge = H[0, 0], H[1, 1]
    xy, xz, yz = H[0, 1], H[0, 2], H[1, 2]
    tol = TRICLINIC_TILT_BOUND_TOL
    for name, val, bound in (("xy", xy, ax_edge / 2.0),
                             ("xz", xz, ax_edge / 2.0),
                             ("yz", yz, by_edge / 2.0)):
        if abs(val) > bound + tol:
            raise ConfigError(
                f"box.cells: reduced tilt factor {name}={val:.6g} Å "
                f"exceeds the LAMMPS restricted-triclinic bound "
                f"{bound:.6g} Å after integer lattice-vector reduction — "
                "this should be mathematically impossible for "
                "reduce_triclinic_tilts (internal error); check the "
                "cell/space-group inputs."
            )

    box.resolved_h = H.tolist()
    log.info(
        "box.cells: %dx%dx%d lattice multiples of the %s cell -> "
        "box matrix a=(%.6g,0,0) b=(%.6g,%.6g,0) c=(%.6g,%.6g,%.6g) A "
        "(tilts xy=%.6g xz=%.6g yz=%.6g A, LAMMPS-reduced); identity "
        "orientation enforced -> commensurability holds EXACTLY (gate "
        "G14 measures 0 misfit).",
        n1, n2, n3, family,
        H[0, 0], H[0, 1], H[1, 1], H[0, 2], H[1, 2], H[2, 2],
        xy, xz, yz,
    )


def _resolve_path_with_base_dir(
    given: str, base_dir: Path | None, field: str,
) -> tuple[str, bool]:
    """Resolve a config path field with a config-directory fallback.

    Tries *given* AS GIVEN first — absolute, or relative to the
    process's current working directory; this interpretation always wins
    when it exists, taking priority over the fallback below. Only when
    that path does NOT exist, *given* is relative, and *base_dir* is not
    None (i.e. the caller is ``load_config``, not a raw-dict
    ``resolve_config`` call) is ``base_dir / given`` tried as a fallback —
    the directory the YAML file itself lives in, which is what lets an
    example config's asset paths resolve from any CWD.

    Returns ``(resolved, used_fallback)``. *resolved* is *given*
    unchanged when the as-given form won (the caller should then leave
    the config field untouched, so ``resolved_config.yaml`` stays
    byte-identical), or the fallback's absolute path (str) when
    *base_dir* won.

    Raises
    ------
    ConfigError
        If neither location has the file. When the fallback was actually
        attempted (relative path + base_dir given), the message lists
        both locations tried; otherwise it names only the single
        location that was tried.
    """
    as_given = Path(given)
    if as_given.exists():
        return given, False

    if base_dir is not None and not as_given.is_absolute():
        candidate = base_dir / given
        if candidate.exists():
            return str(candidate.resolve()), True
        raise ConfigError(
            f"{field} not found: {given} (tried relative to the working "
            f"directory and to the config file's directory {base_dir})"
        )

    raise ConfigError(f"{field} not found: {as_given}")


def _resolve_crystal_cif(
    crystal_cfg, label: str, *, base_dir: Path | None = None,
) -> None:
    """Rule 27: resolve a single ``CrystalConfig`` block in place.

    Exactly one of ``cif`` or the manual ``space_group`` /
    ``lattice`` / ``wyckoff_sites`` trio must be set. When ``cif`` is
    given, the file is read (ASE) and its symmetry detected (spglib) via
    :func:`grainsmith.crystal.cif.read_cif_crystal`, and the result is
    written back into ``crystal_cfg.space_group`` / ``.lattice`` /
    ``.wyckoff_sites`` — the SAME fields the manual path populates via
    YAML — so no downstream code needs to know which input form was
    used. Mutates *crystal_cfg* in place (CrystalConfig is not frozen).

    ``crystal_cfg.cif.file`` is resolved via
    :func:`_resolve_path_with_base_dir` BEFORE
    :func:`~grainsmith.crystal.cif.read_cif_crystal` reads it, so that
    function's own not-found check never fires through this path (it
    stays in place for direct callers of ``read_cif_crystal``); the
    field is rewritten to the resolved absolute path only when the
    ``base_dir`` fallback wins (see :func:`_resolve_path_with_base_dir`).
    """
    from grainsmith.config.schema import (
        LatticeConfig,
        SpaceGroupConfig,
        WyckoffSiteConfig,
    )

    manual_fields = (crystal_cfg.space_group, crystal_cfg.lattice,
                      crystal_cfg.wyckoff_sites)
    manual_given = any(f is not None for f in manual_fields)
    manual_complete = all(f is not None for f in manual_fields)

    if crystal_cfg.cif is not None and manual_given:
        raise ConfigError(
            f"{label}: 'cif' is mutually exclusive with the manual "
            "'space_group' / 'lattice' / 'wyckoff_sites' trio — set "
            "exactly one of the two input forms."
        )
    if crystal_cfg.cif is None:
        if not manual_given:
            raise ConfigError(
                f"{label}: provide either 'cif' (direct CIF input) or "
                "the manual 'space_group' + 'lattice' + 'wyckoff_sites' "
                "trio."
            )
        if not manual_complete:
            missing = [
                name for name, f in zip(
                    ("space_group", "lattice", "wyckoff_sites"),
                    manual_fields, strict=True)
                if f is None
            ]
            raise ConfigError(
                f"{label}: the manual crystal-input path requires "
                f"space_group, lattice, AND wyckoff_sites all set — "
                f"missing: {missing}."
            )
        return  # manual path — nothing to resolve

    # cif path.
    from grainsmith.crystal.cif import read_cif_crystal

    resolved, used_fallback = _resolve_path_with_base_dir(
        crystal_cfg.cif.file, base_dir, f"{label}.cif.file")
    if used_fallback:
        crystal_cfg.cif.file = resolved

    spec = read_cif_crystal(crystal_cfg.cif.file,
                             symprec=crystal_cfg.cif.symprec)

    log.info(
        "%s.cif: %s -> space group %d (%s)%s, %d atoms/cell, symprec=%g "
        "(traceability: recorded in summary.csv / METHODS.md)",
        label, spec.source_file, spec.sg_number, spec.international,
        f" setting={spec.setting!r}" if spec.setting else "",
        spec.n_atoms_conventional, spec.symprec,
    )
    if spec.declared_sg_mismatch:
        log.warning(
            "%s.cif: file declares space group %s but spglib detected %d "
            "(%s) — the spglib-detected group is used.",
            label, spec.declared_sg_number, spec.sg_number,
            spec.international,
        )

    crystal_cfg.space_group = SpaceGroupConfig(
        number=spec.sg_number, setting=spec.setting,
    )
    crystal_cfg.lattice = LatticeConfig(**spec.lattice_params)
    crystal_cfg.wyckoff_sites = [
        WyckoffSiteConfig(element=s.element, coords=s.coords,
                           letter=s.letter)
        for s in spec.wyckoff_sites
    ]


def _config_yaml(config: RunConfig) -> str:
    """Canonical YAML serialization of a resolved config (sorted keys)."""
    raw: dict[str, Any] = config.model_dump(mode="json")
    return yaml.dump(raw, default_flow_style=False, sort_keys=True)


def config_sha256(config: RunConfig) -> str:
    """Full SHA-256 (64 hex) of the canonical resolved config.

    The hashed payload is EXACTLY ``_config_yaml(config)`` encoded UTF-8 — the
    same bytes ``dump_resolved`` writes after its two ``#`` header lines, which
    is what makes ``grainsmith verify`` able to recompute this digest from the
    shipped ``resolved_config.yaml`` alone. Never add anything to this payload:
    the version binding lives in ``provenance_sha256`` (grainsmith/provenance.py)
    precisely so this one stays a pure *config identity*.
    """
    return hashlib.sha256(_config_yaml(config).encode("utf-8")).hexdigest()


def config_sha12(config: RunConfig) -> str:
    """First 12 hex chars of :func:`config_sha256` — the display form used in
    the §8 provenance header lines."""
    return config_sha256(config)[:12]


def dump_resolved(
    config: RunConfig,
    path: Path,
    timestamp_iso: str | None = None,
    seed: int | None = None,
    provenance: Provenance | None = None,
) -> None:
    """Write the resolved config as YAML to *path* with a provenance header.

    The provenance header contains the package version, UTC timestamp,
    seed value, and the two §8 digests (config sha256, provenance sha256)
    for reproducibility auditing.

    Parameters
    ----------
    config : RunConfig
        Fully-resolved run configuration.
    path : Path
        Destination file (e.g. ``<outdir>/resolved_config.yaml``).
        Parent directories are created if necessary.
    timestamp_iso : str, optional
        UTC timestamp to embed; the pipeline passes its single per-run
        timestamp so all output headers agree.  Defaults to now (or the
        ``SOURCE_DATE_EPOCH`` freeze — see ``grainsmith.provenance``).
    seed : int, optional
        Effective master seed (matters for ``seed.mode: entropy``, where
        ``config.seed.value`` is None).  Defaults to ``config.seed.value``.
    provenance : Provenance, optional
        The pipeline's single per-run record, reused verbatim so every
        header in the run is guaranteed identical. When omitted, one is
        built from *timestamp_iso*/*seed* and this config's own digest
        (used by standalone callers, e.g. tests and ``grainsmith validate``
        tooling, that only need the file and not a full ``RunResult``).
    """
    from grainsmith import __version__
    from grainsmith.provenance import make_provenance, run_timestamp_iso

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_str = _config_yaml(config)

    if provenance is not None:
        prov = provenance
    else:
        seed_val = seed if seed is not None else config.seed.value
        if seed_val is None:
            # No silent fallback (house rule): a caller who reaches here
            # passed neither `seed=` nor `provenance=` for an
            # entropy-mode config, so there is no seed to record. The
            # normal pipeline.run() path never hits this — it always
            # resolves the harvested seed and passes it via `provenance=`.
            raise ConfigError(
                "dump_resolved: cannot record a provenance header without "
                "a known seed (config.seed.mode is 'entropy' and no seed "
                "was given) -- pass the resolved seed via seed=... or the "
                "run's Provenance record via provenance=...")
        prov = make_provenance(
            __version__, timestamp_iso or run_timestamp_iso(), seed_val,
            config_sha256(config), config.meta.title)

    # Critical invariant, relied on by `grainsmith verify` (check C4): the
    # header below is exactly two lines, both starting with "#", and the
    # remainder of the file is exactly `_config_yaml(config)`. `verify`
    # recomputes `config_sha256` by stripping the leading contiguous "#"
    # lines from the shipped file and hashing what's left — so a third
    # header line, a blank line, or any other byte between the header and
    # the YAML body would silently break that round-trip.
    header = (f"# {prov.line()}\n"
              "# This file is auto-generated. Do not edit manually.\n")
    from grainsmith.io.common import atomic_writer
    with atomic_writer(path) as fh:
        fh.write(header + raw_str)
    log.debug("Resolved config written to %s", path)


def load_config(path: Path) -> RunConfig:
    """Load a YAML config file and return the validated RunConfig.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    RunConfig
        Validated and cross-field-checked run configuration.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ConfigError
        If the YAML cannot be parsed or Pydantic validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file {path} must be a YAML mapping, "
            f"got {type(raw).__name__}."
        )

    return resolve_config(raw, base_dir=path.resolve().parent)
