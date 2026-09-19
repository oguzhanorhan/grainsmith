"""Pydantic v2 models for the full grainsmith YAML configuration schema (§7).

Rules enforced here:
- ``model_config = ConfigDict(extra="forbid")`` on **every** model.
- Every field carries a ``Field(description=...)`` that surfaces in
  Pydantic validation error messages.
- Cross-field constraints that require context beyond a single model are
  deferred to ``config/resolve.py``.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

from grainsmith.constants import (
    MEMORY_HARD_LIMIT_BYTES,
    ODF_KERNEL_HALFWIDTH_DEG,
    ODF_NULL_SAMPLES,
)

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


class MetaConfig(BaseModel):
    """Top-level run metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default="grainsmith run",
        description="Free-text title embedded in output file headers and the run log.",
    )
    verbose: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG, "
            "3=DEBUG + extra diagnostic file dumps."
        ),
    )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class SeedConfig(BaseModel):
    """RNG seed configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed", "entropy"] = Field(
        default="fixed",
        description=(
            "Seed mode. 'fixed': use the integer in 'value' (reproducible). "
            "'entropy': harvest OS entropy; the resulting integer is recorded "
            "in summary.csv for later reproduction."
        ),
    )
    value: int | None = Field(
        default=None,
        description="Integer seed. Required when mode == 'fixed'.",
    )


# ---------------------------------------------------------------------------
# Runtime (§13 execution-resource knobs)
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Execution-resource knobs (§13) — never affects a single output byte."""

    model_config = ConfigDict(extra="forbid")

    memory_limit_gb: float = Field(
        default=MEMORY_HARD_LIMIT_BYTES / 1e9,
        gt=0.0,
        description=(
            "Per-allocation hard guard (decimal GB, §13): the estimated peak "
            "size of a single large allocation above which grainsmith raises "
            "a TessellationError instead of risking an OOM-kill. Applies to "
            "the per-grain lattice-enumeration grid (atoms/fill.py), the "
            "analysis voxel grid (tessellation/voxel.py), and the imported "
            "voxel-label EDT margin (tessellation/voxel_import.py). The "
            "EFFECTIVE budget actually enforced is "
            "min(this value, detected physical RAM) -- this value can never "
            "be raised past what the machine actually has (see "
            "memory.resolve_memory_budget). This is NOT a total-process "
            "memory cap -- that is the separate, optional '--max-rss' soft "
            "WARN monitor (memory.py), which watches cumulative RSS. Only "
            "the fill stage (atoms/fill.py) runs inside a '--jobs N' "
            "ProcessPoolExecutor: there, up to N grains fill concurrently, "
            "each allocating its own grid, so the machine needs headroom "
            "for N times this limit in the worst case; the voxel-grid and "
            "voxel-import guards each build exactly one grid on the "
            "single-threaded driver, unaffected by '--jobs'."
        ),
    )


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


class BoxConfig(BaseModel):
    """Simulation-box geometry.

    Exactly one of ``lengths`` (orthogonal box; the polycrystal core and
    most single-crystal builds) or ``cells`` (lattice-multiple
    TRICLINIC box, single-crystal builds only) must be given — enforced
    in ``config/resolve.py`` (Rule 28), since the check needs the sibling
    ``grains``/``crystal``/``orientation`` blocks.
    """

    model_config = ConfigDict(extra="forbid")

    lengths: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description=(
            "Box edge lengths [Lx, Ly, Lz] in Å for an ORTHOGONAL box. "
            "Mutually exclusive with 'cells' (Rule 28)."
        ),
    )

    @field_validator("lengths")
    @classmethod
    def _lengths_finite_positive(cls, v: list[float] | None) -> list[float] | None:
        # Guard zero/negative/NaN/Inf box lengths at the schema boundary so they
        # surface as a clean ConfigError (G1) instead of a raw QhullError /
        # complex-radius TypeError / NaN ValueError downstream.
        if v is None:
            return v
        for i, x in enumerate(v):
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError(
                    f"box.lengths[{i}] must be a finite positive number, got {x!r}"
                )
        return v

    cells: list[int] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description=(
            "[n1, n2, n3] positive integer lattice multiples for a "
            "TRICLINIC single-crystal box: the box vectors are "
            "h_i = n_i * a_i, where a_i are the crystal's own conventional "
            "cell vectors (restricted-triclinic convention: a along +x, b "
            "in the xy-plane with positive y, c with positive z — exactly "
            "the form crystal.cell.cell_matrix() returns). Commensurability "
            "then holds EXACTLY by construction (gate G14 measures 0 "
            "misfit). Requires grains.number: 1, box.periodic all true, "
            "box.vacuum: 0, and an identity crystal orientation (an "
            "arbitrary rotation would break the box-vector/lab-axis "
            "alignment) — enforced in resolve.py Rule 28. Mutually "
            "exclusive with 'lengths'."
        ),
    )

    @field_validator("cells")
    @classmethod
    def _cells_positive_int(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        for i, n in enumerate(v):
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise ValueError(
                    f"box.cells[{i}] must be a positive integer, got {n!r}"
                )
        return v

    resolved_h: list[list[float]] | None = Field(
        default=None,
        description=(
            "DERIVED (not user-set): the resolved (3,3) LAMMPS-tilt-"
            "reduced box matrix for box.cells, written back onto this "
            "config by config/resolve.py Rule 28 — the same "
            "'resolve once, cache on the config' pattern crystal.cif "
            "uses. Row-major nested list (row i = box vector component "
            "i across a, b, c). None for the box.lengths (orthogonal) "
            "path."
        ),
    )
    periodic: list[bool] = Field(
        default_factory=lambda: [True, True, True],
        min_length=3,
        max_length=3,
        description=(
            "Per-axis periodicity flags [px, py, pz]. "
            "Set an axis to false for slab / thin-film geometry."
        ),
    )
    vacuum: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Vacuum thickness in Å added on non-periodic axes. "
            "Requires at least one periodic=false axis (enforced in resolve.py)."
        ),
    )


# ---------------------------------------------------------------------------
# Grains
# ---------------------------------------------------------------------------


class SizeDistributionConfig(BaseModel):
    """Target grain-size distribution (power/Laguerre + SDOT).

    Presence of this block switches the flat backend (or the warp base) to
    a power diagram whose weights are fitted by semi-discrete optimal
    transport so cell volumes match the sampled targets (gate G11).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["lognormal", "equal", "volumes"] = Field(
        default="lognormal",
        description=(
            "Target volume model. 'lognormal': equivalent-sphere diameters "
            "d ~ LogN(0, sigma_log), V ∝ d³ (scale fixed by V_box/N — you "
            "prescribe the SHAPE). 'equal': V_box/N each. 'volumes': "
            "explicit relative volume list."
        ),
    )
    sigma_log: float = Field(
        default=0.35,
        gt=0.0,
        description="Lognormal shape parameter: std of ln(d).",
    )
    volumes: list[float] | None = Field(
        default=None,
        description=(
            "Relative target volumes (type: volumes); length must equal "
            "grains.number; normalized to the box volume."
        ),
    )
    vol_tol: float = Field(
        default=1.0e-3,
        gt=0.0,
        description=(
            "Gate G11 threshold: maximum relative per-grain volume error "
            "|V_i − V_i^target| / V_i^target on the final tessellation."
        ),
    )
    max_iter: int = Field(
        default=30,
        ge=1,
        description="Maximum damped-Newton iterations of the SDOT fit.",
    )
    centroidal_iterations: int = Field(
        default=0,
        ge=0,
        description=(
            "Outer centroidal loop count (Kuhn et al. 2020): alternate "
            "volume fitting with moving seeds to power-cell centroids — "
            "equiaxed grains WITH prescribed volumes. 0 = off."
        ),
    )


class GrainsConfig(BaseModel):
    """Grain count and seeding parameters."""

    model_config = ConfigDict(extra="forbid")

    number: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of grains to generate. Required for all generated "
            "tessellations; for boundaries.geometry 'voxel_import' the "
            "count is DERIVED from the label field — omit it, or set it "
            "as a cross-check (a mismatch is an error)."
        ),
    )
    size_distribution: SizeDistributionConfig | None = Field(
        default=None,
        description=(
            "Optional target grain-size distribution. When set, "
            "the tessellation is a volume-fitted power (Laguerre) diagram; "
            "allowed with flat geometry or as the base of the warp method."
        ),
    )
    min_seed_distance: float | Literal["auto"] = Field(
        default="auto",
        description=(
            "Minimum centre-to-centre seed distance in Å for RSA placement. "
            "'auto' = 1.0 × r_ws (Wigner–Seitz radius), safe below the RSA "
            "jamming limit."
        ),
    )
    lloyd_iterations: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of Lloyd centroidal-relaxation iterations (voxel-grid based). "
            "0 = off (Poisson–Voronoi); >0 produces more equiaxed grains."
        ),
    )


# ---------------------------------------------------------------------------
# Crystal
# ---------------------------------------------------------------------------


class SpaceGroupConfig(BaseModel):
    """Space-group identification."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(
        ge=1,
        le=230,
        description="International (Hermann–Mauguin) space-group number, 1–230.",
    )
    setting: str | None = Field(
        default=None,
        description=(
            "Optional setting string for groups with multiple standard settings, "
            "e.g. '2' (origin choice 2), 'H' or 'R' (rhombohedral). "
            "null = first / standard setting."
        ),
    )


class LatticeConfig(BaseModel):
    """Lattice parameters — only the free parameters for the crystal family.

    Which parameters are required depends on the crystal system and is
    validated in the crystal module.  The schema accepts all parameters
    as optional so that validation errors report exactly the missing ones.
    """

    model_config = ConfigDict(extra="forbid")

    a: float | None = Field(
        default=None,
        gt=0.0,
        description="Lattice parameter a in Å.",
    )
    b: float | None = Field(
        default=None,
        gt=0.0,
        description="Lattice parameter b in Å.",
    )
    c: float | None = Field(
        default=None,
        gt=0.0,
        description="Lattice parameter c in Å.",
    )
    alpha: float | None = Field(
        default=None,
        gt=0.0,
        lt=180.0,
        description="Lattice angle α (bc plane) in degrees.",
    )
    beta: float | None = Field(
        default=None,
        gt=0.0,
        lt=180.0,
        description="Lattice angle β (ac plane) in degrees.",
    )
    gamma: float | None = Field(
        default=None,
        gt=0.0,
        lt=180.0,
        description="Lattice angle γ (ab plane) in degrees.",
    )


class WyckoffSiteConfig(BaseModel):
    """One representative Wyckoff site (fractional coordinates + species)."""

    model_config = ConfigDict(extra="forbid")

    element: str | dict[str, float] = Field(
        description=(
            "Element symbol (e.g. 'Ni') for a fully-occupied site, "
            "or an occupancy dict (e.g. {'Ti': 0.90, 'Al': 0.10}) for a "
            "solid solution.  Occupancy values must sum to 1 ± 1e-6."
        ),
    )
    coords: list[float] = Field(
        min_length=3,
        max_length=3,
        description=(
            "Representative fractional coordinates [x, y, z] of the Wyckoff "
            "position.  Free parameters must be substituted with numeric values."
        ),
    )
    letter: str | None = Field(
        default=None,
        description=(
            "Wyckoff letter (e.g. 'a', 'b', '4c').  Optional; if provided, "
            "the crystal module verifies it matches the detected site symmetry."
        ),
    )


class CifConfig(BaseModel):
    """Direct crystal-structure input from a CIF file (ASE + spglib path).

    Mutually exclusive with the manual ``space_group`` / ``lattice`` /
    ``wyckoff_sites`` trio (enforced in resolve.py, Rule 27): the
    structure is read with ``ase.io.read``, its symmetry is detected by
    ``spglib.get_symmetry_dataset`` at tolerance ``symprec``, and the
    space group, lattice parameters, and symmetry-inequivalent Wyckoff
    sites are derived directly from the file — nothing is entered by
    hand.  The detected quantities (space group number + H-M symbol,
    symprec, lattice parameters, per-site Wyckoff letters) are recorded
    in summary.csv and METHODS.md exactly like the manual path (no
    silent assumptions).
    """

    model_config = ConfigDict(extra="forbid")

    file: str = Field(
        description=(
            "Path to a CIF file (any format ase.io.read accepts for "
            "'.cif'). Read with ASE; the space group is then determined "
            "by spglib — the file's own _symmetry_Int_Tables_number "
            "(if present) is cross-checked against the spglib-detected "
            "number; a mismatch only WARNS (the spglib-detected group is "
            "always the one used), since a CIF legally under-declaring "
            "its own symmetry (e.g. exporting every atom under 'P 1') "
            "is common and not itself an error. A relative path is "
            "resolved against the working directory first, then against "
            "the directory of the YAML file it appears in."
        ),
    )
    symprec: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "spglib symmetry-detection tolerance (Å-scale fractional "
            "distance) used for the CIF structure — analogous to "
            "constants.VERIFY_SYMPREC used for the manual-path G2 "
            "round-trip. Loosen for CIFs with larger coordinate "
            "roundoff (e.g. Rietveld-refined structures); tighten to "
            "separate nearly-degenerate settings."
        ),
    )


class CrystalConfig(BaseModel):
    """Crystal structure definition: EITHER a manual space group +
    Wyckoff-site motif, OR direct input from a CIF file (mutually
    exclusive — resolve.py Rule 27)."""

    model_config = ConfigDict(extra="forbid")

    space_group: SpaceGroupConfig | None = Field(
        default=None,
        description=(
            "Space-group identification (international number + optional "
            "setting). Required for the manual path; forbidden together "
            "with 'cif'."
        ),
    )
    lattice: LatticeConfig | None = Field(
        default=None,
        description=(
            "Lattice parameters for the crystal family (§6.1). Required "
            "for the manual path; forbidden together with 'cif'."
        ),
    )
    wyckoff_sites: list[WyckoffSiteConfig] | None = Field(
        default=None,
        min_length=1,
        description=(
            "List of representative Wyckoff sites defining the motif. "
            "Required for the manual path; forbidden together with 'cif'."
        ),
    )
    cif: CifConfig | None = Field(
        default=None,
        description=(
            "Direct CIF input (ASE read + spglib symmetry detection). "
            "Mutually exclusive with the space_group/lattice/wyckoff_sites "
            "trio — exactly one of the two input forms must be given."
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize_cif_or_manual(self, handler):
        """Dump ONLY the input form actually given, not both.

        ``resolve.py`` (Rule 27) resolves a ``cif`` block by writing the
        derived space_group/lattice/wyckoff_sites back onto this SAME
        instance (so the pipeline can consume them like the manual path)
        — but that makes ``model_dump`` emit both forms at once, which
        would then fail re-validation (the mutual-exclusion guard) the
        moment the dump is reloaded: the resolved-config provenance file
        (``dump_resolved`` / ``resolved_config.yaml``) and the studio
        UI's YAML round-trip (``ui/yaml_builder.config_to_yaml`` ->
        ``load_yaml``) both go through exactly this path. When ``cif``
        is set, the manual trio is dropped from the dump (it is a
        resolved CACHE, not independent input); otherwise the manual
        trio is dumped and ``cif`` (already None) is omitted-in-effect.
        """
        data = handler(self)
        if self.cif is not None:
            data["space_group"] = None
            data["lattice"] = None
            data["wyckoff_sites"] = None
        return data


class PhaseConfig(BaseModel):
    """One phase of a multiphase polycrystal.

    The top-level ``phases:`` list is mutually exclusive with the
    single-phase ``crystal:`` shortcut.  Grains are assigned to phases by
    a deterministic greedy partition of the measured pre-fill grain
    volumes so the achieved VOLUME fractions match the targets as closely
    as the grain-count granularity allows (gate G15, warn).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Phase name (e.g. 'alpha', 'beta', 'Cu_fcc'); must be unique "
            "and match [A-Za-z0-9_-]+ — it appears in grains.csv / "
            "boundaries.csv columns, summary sections, and per-phase "
            "output filenames (mdf_<name>.csv, odf_mtex_<name>.txt)."
        ),
    )
    fraction: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Target VOLUME fraction of this phase (the metallographic "
            "convention). Fractions must sum to 1 ± 1e-6; the achieved "
            "fractions are re-measured by gate G15 (warn — the deviation "
            "floor is set by the largest single grain)."
        ),
    )
    crystal: CrystalConfig = Field(
        description=(
            "Crystal structure of this phase (space group, lattice, "
            "Wyckoff sites) — phases may differ in structure AND in "
            "elements (e.g. HCP Ti / BCC Ti, or FCC Cu / BCC Fe)."
        ),
    )


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


class OrientationFixedHklUvw(BaseModel):
    """Fix orientation by aligning a crystal plane and direction to lab axes."""

    model_config = ConfigDict(extra="forbid")

    plane: list[int] = Field(
        min_length=3,
        max_length=3,
        description="Miller indices (h k l) of the crystal plane to align with lab +z.",
    )
    direction: list[int] = Field(
        min_length=3,
        max_length=3,
        description="Miller direction [u v w] to align with lab +x.",
    )


class AxisAngleConfig(BaseModel):
    """Rotation expressed as an axis vector and an angle."""

    model_config = ConfigDict(extra="forbid")

    axis: list[float] = Field(
        min_length=3,
        max_length=3,
        description="Rotation axis vector (need not be unit length; will be normalised).",
    )
    angle_deg: float = Field(
        description="Rotation angle in degrees.",
    )


class OrientationFixedConfig(BaseModel):
    """Orientation specification for a single grain (one sub-key must be set).

    Exactly one of ``hkl_uvw``, ``euler_bunge_deg``, ``axis_angle``, or
    ``quaternion`` should be provided.  Cross-validation is performed in
    the orientation stage.
    """

    model_config = ConfigDict(extra="forbid")

    hkl_uvw: OrientationFixedHklUvw | None = Field(
        default=None,
        description="Fix orientation by (hkl) plane parallel to lab +z and [uvw] along lab +x.",
    )
    euler_bunge_deg: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Bunge Euler angles (φ1, Φ, φ2) in degrees.",
    )
    axis_angle: AxisAngleConfig | None = Field(
        default=None,
        description="Orientation as a rotation axis + angle.",
    )
    quaternion: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Orientation quaternion (w, x, y, z) — scalar-first, unit norm.",
    )


class OrientationFiberConfig(BaseModel):
    """Fiber-texture orientation parameters."""

    model_config = ConfigDict(extra="forbid")

    crystal_axis: list[float] = Field(
        min_length=3,
        max_length=3,
        description="Crystal axis <u v w> to align with the sample direction.",
    )
    sample_direction: str = Field(
        default="z",
        description="Sample direction to align to: 'x', 'y', or 'z'.",
    )
    spread_deg: float = Field(
        default=10.0,
        ge=0.0,
        le=180.0,
        description="Gaussian angular spread (standard deviation) around the fiber axis in degrees.",
    )


class OdfComponentConfig(BaseModel):
    """One ODF texture component.

    Exactly one of ``euler_bunge_deg``, ``fiber``, ``random`` must be set
    (cross-checked in resolve.py).
    """

    model_config = ConfigDict(extra="forbid")

    euler_bunge_deg: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description=(
            "Component centre as Bunge Euler angles (φ1, Φ, φ2) in degrees, "
            "e.g. Cu/S/Brass rolling components."
        ),
    )
    fiber: OrientationFiberConfig | None = Field(
        default=None,
        description=(
            "Fiber component reusing the fiber machinery "
            "(crystal_axis / sample_direction / spread_deg)."
        ),
    )
    random: bool = Field(
        default=False,
        description=(
            "Haar-uniform 'random fraction' component for partially "
            "textured states. At most one such component is allowed."
        ),
    )
    weight: float = Field(
        gt=0.0,
        description=(
            "Relative component weight (normalized). How it is realised "
            "depends on orientation.component_weight_basis: under "
            "'volume' (default) it is the target VOLUME fraction of "
            "material assigned to this component (deterministic "
            "grain-to-component partition); under 'count' grains draw "
            "their component from the categorical distribution ∝ weight, "
            "so it is a GRAIN-COUNT fraction instead."
        ),
    )
    spread_deg: float = Field(
        default=0.0,
        ge=0.0,
        le=180.0,
        description=(
            "Isotropic angular spread (std of the rotation angle, degrees) "
            "around an euler_bunge_deg centre — SO(3) von Mises–Fisher "
            "small-angle surrogate with exact Haar sin²(θ/2) correction. "
            "0 reduces the component to a fixed orientation. Ignored for "
            "fiber components (they carry their own spread_deg) and "
            "forbidden for random components."
        ),
    )


class MdfTargetConfig(BaseModel):
    """Boundary-area-weighted disorientation-angle target via assignment annealing.

    Simulated annealing permutes WHICH grain gets WHICH orientation to pull
    the neighbor-misorientation histogram toward the target.  This makes the
    NUMBER-weighted discrete orientation distribution invariant by
    construction; the VOLUME-weighted ODF is NOT — a transposition of grains
    a and b moves |V_a - V_b| / Σ_i V_i of its total-variation mass.  Set
    ``odf_drift_max`` to bound that drift; see ``orientation/odf.py`` for
    the full derivation and the measurement/control machinery.  Gate G12
    (warn) reports the final χ² distance.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "mackenzie", "haar_random", "csl_enriched", "sigma3_angle_enriched",
        "histogram",
    ] = Field(
        default="haar_random",
        description=(
            "Target for the boundary-AREA-weighted marginal distribution "
            "of the disorientation ANGLE — this is NOT a misorientation "
            "distribution function (an MDF is a density over the full "
            "misorientation space; this objective never sees the "
            "misorientation axis). 'haar_random': the random-pair "
            "disorientation-angle reference of the ACTUAL point group; "
            "this estimates the Mackenzie cubic law only for the "
            "24-rotation proper group of m-3m. Other point groups have "
            "different references. 'mackenzie' is a deprecated "
            "alias for 'haar_random'. 'sigma3_angle_enriched': "
            "'haar_random' base mixed with a window-only distribution "
            "using sigma3_fraction as the mixture coefficient in the "
            "Σ3 Brandon ANGULAR window 60° ± 15°/√3 — this "
            "enforces only the NECESSARY angular condition; the ⟨111⟩ "
            "axis condition that CSL classification also requires is NOT "
            "part of the objective (cubic only). 'csl_enriched' is a "
            "deprecated alias for 'sigma3_angle_enriched'. 'histogram': "
            "explicit bin_edges + densities."
        ),
    )
    sigma3_fraction: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description=(
            "sigma3_angle_enriched (and its deprecated alias "
            "csl_enriched): mixture coefficient f in "
            "target = (1-f)*haar_random + f*angle_window, where "
            "angle_window is supported on 60° ± 15°/√3. This is NOT "
            "the total target window fraction: the Haar reference "
            "already has mass in that window. This is angle-only — "
            "the realised CSL Σ3 AREA fraction (the true class, axis "
            "condition included) is measured separately; compare it "
            "against the realised angle-window fraction in G23."
        ),
    )
    bin_edges: list[float] | None = Field(
        default=None,
        description=(
            "histogram: increasing misorientation-angle bin edges in "
            "degrees over [0, θ_max]; length = len(densities) + 1."
        ),
    )
    densities: list[float] | None = Field(
        default=None,
        description=(
            "histogram: non-negative target densities per bin "
            "(normalized internally to unit integral)."
        ),
    )
    annealing_steps: int = Field(
        default=20000,
        ge=0,
        description=(
            "Number of swap moves of the simulated annealing. 0 = no "
            "annealing — only measure the χ² distance (gate G12)."
        ),
    )
    t0: float = Field(
        default=0.05,
        gt=0.0,
        description="Initial annealing temperature (χ² energy units).",
    )
    cooling: float = Field(
        default=0.995,
        gt=0.0,
        lt=1.0,
        description="Geometric cooling ratio: T_k = t0 · cooling^k.",
    )
    chi2_max: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Gate G12 threshold (warn): symmetric χ² distance between the "
            "re-measured area-weighted disorientation-angle histogram "
            "and the target. The default "
            "covers the multinomial sampling noise floor ~n_bins/n_pairs "
            "of typical runs (e.g. 40 bins / 150 pairs ≈ 0.27)."
        ),
    )
    odf_drift_max: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Hard cap on d_TV(f_V(π), f_V(π₀)) — the total-variation "
            "drift of the VOLUME-weighted ODF away from the sampler's own "
            "assignment π₀ — enforced on every ACCEPTED swap of the "
            "annealing. None (default): the drift is measured and "
            "reported but never constrained. Measured trade-off on a "
            "60-grain lognormal test case: a cap of 0.01 already recovers "
            "about half of the achievable χ² reduction, and 0.05 about "
            "three quarters — pick a value with that trade-off in mind. "
            "See orientation/odf.py."
        ),
    )
    odf_kernel_halfwidth_deg: float = Field(
        default=ODF_KERNEL_HALFWIDTH_DEG,
        gt=0.0,
        lt=180.0,
        description=(
            "Requested de la Vallée Poussin kernel half-width (degrees) "
            "for the volume-weighted ODF diagnostic. The degree is "
            "rounded to the nearest positive integer to keep the "
            "kernel positive semidefinite; the effective half-width "
            "is reported separately (at most 90 degrees). Configures "
            "the DIAGNOSTIC only — it has no effect on odf_drift_max, "
            "which caps the un-smoothed (atomic) drift and, by the "
            "Markov-contraction argument, therefore already bounds the "
            "TV of kernel-smoothed densities at every half-width "
            "simultaneously (not the separate MMD norm) "
            "(orientation/odf.py)."
        ),
    )
    odf_null_samples: int = Field(
        default=ODF_NULL_SAMPLES,
        ge=1,
        description=(
            "Sample count for the ODF kernel-discrepancy null distribution "
            "(orientation.odf.null_mmd). Configures the DIAGNOSTIC only; "
            "has no effect on the annealing itself or on odf_drift_max."
        ),
    )


class OrientationConfig(BaseModel):
    """Grain-orientation assignment scheme and parameters."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["random_uniform", "fixed", "fiber", "from_list",
                    "odf_components", "imported"] = Field(
        default="random_uniform",
        description=(
            "Orientation assignment scheme. "
            "'random_uniform': Haar-uniform random SO(3). "
            "'fixed': same orientation for all grains (use 'fixed' sub-block). "
            "'fiber': crystal axis aligned to a sample direction with angular spread. "
            "'from_list': explicit per-grain list (length must equal grains.number). "
            "'odf_components': weighted texture components (use 'components' "
            "list). "
            "'imported': per-grain Euler angles from the voxel_import file "
            "(requires boundaries.voxel_import.euler_dataset)."
        ),
    )
    fixed: OrientationFixedConfig | None = Field(
        default=None,
        description="Used when scheme == 'fixed'. Specifies the single shared orientation.",
    )
    fiber: OrientationFiberConfig | None = Field(
        default=None,
        description="Used when scheme == 'fiber'. Specifies fiber-texture parameters.",
    )
    from_list: list[OrientationFixedConfig] | None = Field(
        default=None,
        description=(
            "Used when scheme == 'from_list'. One entry per grain; "
            "length must equal grains.number (checked at generation time)."
        ),
    )
    components: list[OdfComponentConfig] | None = Field(
        default=None,
        description=(
            "Used when scheme == 'odf_components': list of weighted "
            "texture components (Euler centre + spread, fiber, or one "
            "uniform random fraction)."
        ),
    )
    component_weight_basis: Literal["volume", "count"] = Field(
        default="volume",
        description=(
            "How each OdfComponentConfig.weight is realised. 'volume' "
            "(default): weight is the VOLUME fraction of material in the "
            "component — the physical definition of the ODF; grains are "
            "partitioned deterministically "
            "(orientation.odf.volume_balanced_partition) so the realised "
            "volume fractions match the configured weights, at the cost "
            "of a size-component correlation that gate G25 reports. "
            "'count': weight is realised as a GRAIN-COUNT fraction via an "
            "i.i.d. categorical draw (the pre-1.2 behaviour); the "
            "realised VOLUME fractions then differ from the configured "
            "weights by an amount set by the grain-size distribution "
            "(measured 0.745/0.255 for a configured 0.500/0.500 at 24 "
            "grains, sigma_log 0.5). Ignored unless scheme == "
            "'odf_components'."
        ),
    )
    mdf_target: MdfTargetConfig | None = Field(
        default=None,
        description=(
            "Optional boundary-area-weighted disorientation-angle target (any scheme): "
            "orientation-assignment annealing pulls the neighbor "
            "misorientation-angle distribution toward the target. The "
            "NUMBER-weighted orientation distribution is invariant by "
            "construction; the VOLUME-weighted ODF is not, but "
            "mdf_target.odf_drift_max can bound its drift. Gate G12 (warn)."
        ),
    )


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


class CurvedBoundaryConfig(BaseModel):
    """Parameters for curved-boundary methods (§3.3, §6.6)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "warp", "additive_weights", "anisotropic", "perturbed_distance",
    ] = Field(
        default="warp",
        description=(
            "Curved-boundary method. "
            "'warp': domain-warped Voronoi via periodic GRF (default, §6.6). "
            "'additive_weights': Johnson–Mehl / Apollonius diagram. "
            "'anisotropic': per-grain ellipsoidal metric (GBPD family). "
            "'perturbed_distance': level-set membership "
            "argmin_i[d_pbc(x,s_i) − amplitude·η_i(x)] with an "
            "INDEPENDENT per-grain scalar field η_i (the same spectrum/"
            "hurst/l_min/l_max/amplitude schema keys warp also declares, "
            "reused verbatim) — unlike warp's coordinate diffeomorphism "
            "(bijective, so it cannot change the flat-Voronoi surface's "
            "fractal dimension; this is WHY warp does not accept "
            "spectrum: self_affine at all), this method perturbs the "
            "ASSIGNMENT RULE itself and so can produce a genuinely "
            "self-affine grain boundary whose box-counting dimension "
            "responds to hurst — the only method that accepts "
            "spectrum: self_affine (docs/physics.md §5b). G6 "
            "(bijectivity) does not apply; a seed-containment guard and "
            "an exact seed-ownership check replace it. G5 is a "
            "repair-and-report gate here (reassigned_fraction), not a "
            "hard failure."
        ),
    )
    base: Literal["flat", "additive_weights", "anisotropic", "power"] = Field(
        default="flat",
        description=(
            "Base tessellation for the warp method (ignored for other "
            "methods). 'power' requires grains.size_distribution (a "
            "volume-fitted Laguerre base) — and is implied when "
            "size_distribution is present with base 'flat'."
        ),
    )
    amplitude: float = Field(
        default=4.0,
        ge=0.0,
        description=(
            "Warp-field RMS amplitude per component in Å. "
            "Hard constraint: amplitude ≤ min_seed_distance / 4 (enforced "
            "downstream)."
        ),
    )
    correlation_length: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "Gaussian correlation length ℓ in Å — the boundary 'waviness wavelength'. "
            "Larger ℓ produces smoother, more gently curved boundaries. "
            "Used only by the gaussian spectrum; ignored by "
            "perturbed_distance's self_affine spectrum (band limits "
            "l_min/l_max take its place there — warp does not accept "
            "self_affine at all, so this key and self_affine are never "
            "both active together)."
        ),
    )
    spectrum: Literal["gaussian", "self_affine"] = Field(
        default="gaussian",
        description=(
            "Field amplitude spectrum (restricted per-method — see below). "
            "'gaussian': single-scale "
            "exp(−k²ℓ²/4) (default, bit-identical); the ONLY spectrum "
            "'warp' accepts — a ConfigError rejects 'warp' + "
            "'self_affine' (warp is a coordinate diffeomorphism, which "
            "cannot change the flat-Voronoi surface's fractal dimension, "
            "so a self-affine spectrum on it would be undetectable in the "
            "as-built boundary; docs/physics.md §5b). 'self_affine': "
            "band-limited power law √Φ(k) ∝ k^(−(3+2H)/2) for "
            "k ∈ [2π/l_max, 2π/l_min] — scale-free roughness with field "
            "fractal dimension D = 3 − hurst; valid ONLY with method: "
            "perturbed_distance, where perturbing the assignment rule "
            "(not the coordinate) lets the boundary's own box-counting "
            "dimension respond to hurst (gate G20 measures it; gate G13 "
            "is its Hurst-back-estimation analogue, also perturbed_"
            "distance-only)."
        ),
    )
    hurst: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description=(
            "Hurst exponent H ∈ (0, 1] of the self_affine spectrum "
            "(perturbed_distance only; surface PSD C(q) ∝ q^(−2−2H); "
            "typical metal surfaces/GBs: 0.7–0.9)."
        ),
    )
    l_min: float = Field(
        default=8.0,
        gt=0.0,
        description=(
            "Shortest roughness wavelength in Å (self_affine band upper "
            "k-limit 2π/l_min; perturbed_distance only — warp never "
            "consumes l_min at all, since it cannot accept spectrum: "
            "self_affine in the first place). Must satisfy l_min ≥ "
            "2·h_field (grid Nyquist, checked against the actual field "
            "grid at construction) and l_min < l_max. perturbed_distance "
            "has no bijectivity guard (no G6) to tie l_min to an "
            "amplitude ceiling: its amplitude ceiling (ETA_CLIP/"
            "PERTURBED_DISTANCE_SAFETY, seed-containment + exact "
            "seed-ownership) is a function of min_seed_distance only and "
            "does not depend on l_min."
        ),
    )
    l_max: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Longest roughness wavelength in Å (self_affine band lower "
            "k-limit 2π/l_max; perturbed_distance only). Must satisfy "
            "l_min < l_max ≤ min(L)/2."
        ),
    )
    amplitude_convention: Literal["total_rms", "reference_wavelength"] = Field(
        default="total_rms",
        description=(
            "How `amplitude` (self_affine only; opt-in) scales the "
            "band-limited η field. 'total_rms' (default, UNCHANGED "
            "behavior): `amplitude` is the field's total RMS over the "
            "WHOLE synthesis band [l_min, l_max] — widening the band "
            "(more octaves) REDISTRIBUTES this fixed budget across more "
            "scales rather than adding to it, so at fixed `amplitude` "
            "adding a coarse octave can paradoxically REDUCE small-scale "
            "(l_min-scale) roughness — see docs/physics.md §5b for the "
            "measured reversal this causes in specific GB area (S_V). "
            "'reference_wavelength': `amplitude` instead fixes the RMS "
            "power contributed by ONE reference octave anchored at "
            "`reference_wavelength` (default l_max), leaving every other "
            "resolved octave's contribution free to ACCUMULATE on top — "
            "restoring the expected (monotonic) coarse-octave response "
            "and giving `amplitude` a stable physical meaning (Å-RMS at "
            "a named wavelength) independent of how many octaves the "
            "band happens to span. Exact, invertible mapping to the "
            "default convention: `amplitude_total_rms = amplitude_A0 / "
            "kappa(hurst, l_min, l_max, reference_wavelength)` — see "
            "`tessellation.warp.reference_shell_kappa`. Valid only with "
            "spectrum: self_affine (perturbed_distance only); the "
            "seed-containment guard binds the REALIZED total-RMS-"
            "equivalent amplitude in either convention, never just the "
            "raw `amplitude` value, so switching convention cannot "
            "weaken it."
        ),
    )
    reference_wavelength: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Anchor wavelength in Å for the top reference octave "
            "[reference_wavelength/2, reference_wavelength] that "
            "`amplitude_convention: reference_wavelength` fixes the RMS "
            "power of (self_affine only). Default `None` = l_max (an "
            "anchor that MOVES with the band — natural when l_max "
            "already represents the physical grain radius). A fixed "
            "absolute wavelength can be given instead — required for a "
            "band-WIDTH scan at constant grain size to accumulate "
            "variance monotonically in both directions (the l_max-anchored "
            "default does not, "
            "by construction, correct THAT specific diagnostic — see "
            "docs/physics.md §5b). Must lie within [l_min, l_max] when "
            "given; must not be set under `amplitude_convention: "
            "total_rms` (a silently-ignored field is a config error "
            "here, not a no-op)."
        ),
    )
    weight_sigma: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight spread σ_w in Å for the additive_weights method. "
            "Constraint: σ_w ≤ min_seed_distance / 6."
        ),
    )
    aspect_ratio_range: list[float] = Field(
        default_factory=lambda: [1.0, 1.0],
        min_length=2,
        max_length=2,
        description=(
            "Allowed range [min, max] of per-grain aspect ratios "
            "for the anisotropic method."
        ),
    )

    @field_validator("aspect_ratio_range")
    @classmethod
    def _aspect_ratio_range_valid(cls, v: list[float]) -> list[float]:
        lo, hi = v
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(
                f"aspect_ratio_range values must be strictly positive, got [{lo}, {hi}]"
            )
        if lo > hi:
            raise ValueError(
                f"aspect_ratio_range[0] must be <= aspect_ratio_range[1], "
                f"got [{lo}, {hi}]"
            )
        return v


class OverlapRemovalConfig(BaseModel):
    """Atom-overlap removal parameters (§6.9)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether to run the GB overlap-removal pass.",
    )
    cutoff: float | str = Field(
        default="0.85*d_nn",
        description=(
            "Overlap cutoff as an absolute distance in Å, or as an expression "
            "referencing 'd_nn' (nearest-neighbour distance of the ideal crystal). "
            "Example: '0.85*d_nn' (default) or '1.5' (Å)."
        ),
    )
    policy: Literal["delete_shallower", "keep_lower_id", "midpoint_merge"] = Field(
        default="delete_shallower",
        description=(
            "Deletion policy for overlapping inter-grain atom pairs. "
            "'delete_shallower': remove the atom with smaller boundary margin. "
            "'keep_lower_id': always keep the atom belonging to the lower grain id. "
            "'midpoint_merge': replace the pair with one atom at the midpoint "
            "(single-species systems only)."
        ),
    )


class VoxelImportConfig(BaseModel):
    """Imported voxel label field."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(
        description=(
            "Path to the label field: .npy (3D integer array, internal "
            "(Nx, Ny, Nz) layout) or DREAM.3D HDF5 (.dream3d/.h5/.hdf5; "
            "requires the [import] extra / h5py). The grid shape comes "
            "from the file; box.lengths from the config. A relative path "
            "is resolved against the working directory first, then "
            "against the directory of the YAML file it appears in."
        ),
    )
    dataset: str | None = Field(
        default=None,
        description=(
            "HDF5 path to the FeatureIds cell array (required for "
            "DREAM.3D files), e.g. 'DataContainers/SyntheticVolume"
            "DataContainer/CellData/FeatureIds'. Ignored for .npy."
        ),
    )
    relabel: bool = Field(
        default=True,
        description=(
            "Compact arbitrary grain ids to 0 … N−1 (the original → "
            "compact map is recorded in summary.csv). When false, the "
            "field must already be compact."
        ),
    )
    strict_connectivity: bool = Field(
        default=False,
        description=(
            "Gate G5 severity for the imported field: external "
            "microstructures may legitimately contain disconnected or "
            "wrap-spanning grains, so G5 is demoted to WARN by default; "
            "set true to hard-fail on fragmented grains."
        ),
    )
    euler_dataset: str | None = Field(
        default=None,
        description=(
            "Optional HDF5 path to per-grain Bunge Euler angles (N, 3), "
            "indexed by the ORIGINAL feature ids (e.g. DREAM.3D "
            "'.../Grain Data/AvgEulerAngles'). Required by "
            "orientation.scheme 'imported'."
        ),
    )
    euler_degrees: bool = Field(
        default=False,
        description=(
            "Unit of euler_dataset angles: false = radians (the DREAM.3D "
            "convention), true = degrees."
        ),
    )


class BoundariesConfig(BaseModel):
    """Grain-boundary geometry and overlap removal."""

    model_config = ConfigDict(extra="forbid")

    geometry: Literal["flat", "curved", "voxel_import"] = Field(
        default="flat",
        description=(
            "Boundary geometry. "
            "'flat': planar Voronoi faces (Qhull). "
            "'curved': domain-warp or weighted-Voronoi method (use 'curved' sub-block). "
            "'voxel_import': externally produced voxel label field (use "
            "'voxel_import' sub-block)."
        ),
    )
    curved: CurvedBoundaryConfig | None = Field(
        default=None,
        description=(
            "Curved-boundary parameters. Used when geometry == 'curved'. "
            "Providing this block with geometry == 'flat' is a config "
            "error (it would otherwise be silently ignored)."
        ),
    )
    voxel_import: VoxelImportConfig | None = Field(
        default=None,
        description=(
            "Imported label-field parameters. Required when geometry == "
            "'voxel_import'."
        ),
    )
    overlap_removal: OverlapRemovalConfig = Field(
        default_factory=OverlapRemovalConfig,
        description="Atom-overlap removal settings applied after the fill stage.",
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class SectionConfig(BaseModel):
    """EBSD-like 2D section export.

    Writes ``slice_<axis><position>.csv`` with one row per section pixel:
    the two in-plane coordinates (axis order), the grain id, and the
    grain's Bunge Euler angles — directly comparable to an experimental
    EBSD map (e.g. in MTEX).
    """

    model_config = ConfigDict(extra="forbid")

    axis: Literal["x", "y", "z"] = Field(
        default="z",
        description="Box axis NORMAL to the section plane.",
    )
    position: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description=(
            "Fractional position of the plane along the axis "
            "(0 = low box face, 0.5 = midplane)."
        ),
    )


class AnalysisConfig(BaseModel):
    """Post-generation analysis options."""

    model_config = ConfigDict(extra="forbid")

    statistics: bool = Field(
        default=True,
        description=(
            "Write the publication statistics outputs: "
            "statistics.csv (grain-size + log-normal fit, sphericity, "
            "S_V, triple-junction L_V, GB character/CSL area fractions; "
            "phase-aware) and microstructure.json (the machine-readable "
            "record of the whole run — dataset-repository ready)."
        ),
    )
    section: SectionConfig | None = Field(
        default=None,
        description=(
            "Optional EBSD-like 2D section export "
            "(slice_<axis><position>.csv)."
        ),
    )
    voxel_grid: int | Literal["auto"] = Field(
        default="auto",
        description=(
            "Voxel grid resolution. "
            "'auto': derive from ℓ/4 and r_ws/10, capped at VOXEL_GRID_MAX³. "
            "Integer: explicit number of voxels along the shortest box edge."
        ),
    )
    gb_character: bool = Field(
        default=True,
        description=(
            "Compute per-boundary GB character (tilt/twist/mixed) and "
            "boundary-plane Miller indices for both crystal frames."
        ),
    )
    csl: bool = Field(
        default=False,
        description=(
            "Attempt CSL Σ assignment using the built-in cubic CSL table. "
            "Requires a cubic point group (enforced in resolve.py); ignored "
            "otherwise with a ConfigError."
        ),
    )
    gb_curvature: bool = Field(
        default=False,
        description=(
            "Compute local grain-boundary mean/Gaussian curvature (H in "
            "1/Å, K in 1/Å²) at the boundary sample points: appends six "
            "area-weighted per-boundary columns to boundaries.csv and "
            "writes the local samples to gb_curvature.csv (fixed name). "
            "Flat geometry writes exact zeros — planar boundaries have "
            "no curvature by construction.  Sign: H > 0 means grain_i "
            "is locally convex.  Not supported with geometry "
            "'voxel_import' (piecewise-constant margins; rejected at "
            "config time)."
        ),
    )
    per_atom_margin: bool = Field(
        default=False,
        description=(
            "Add a 'gb_margin' column (signed distance to nearest GB in Å) "
            "to the extended XYZ output."
        ),
    )
    mdf_bins: int = Field(
        default=40,
        ge=4,
        description=(
            "Number of misorientation-angle bins for mdf.csv and for the "
            "haar_random/sigma3_angle_enriched angle targets over [0, θ_max] of the "
            "point group."
        ),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class OutputLammpsConfig(BaseModel):
    """LAMMPS data-file output settings."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        default="polycrystal.data",
        description="Output filename for the LAMMPS data file.",
    )
    atom_style: Literal["atomic", "molecular"] = Field(
        default="atomic",
        description=(
            "LAMMPS atom_style. "
            "'atomic': id type x y z. "
            "'molecular': id mol type x y z with mol = grain_id (1-based), "
            "enabling per-grain coloring in OVITO via molecule-id."
        ),
    )
    masses: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional per-element mass overrides in amu for the Masses "
            "section (e.g. {Fe: 55.0}). Elements not listed fall back to "
            "constants.ATOMIC_MASSES; unknown elements without an override "
            "raise a ConfigError."
        ),
    )
    g10_readback: Literal["full", "sampled", "off"] = Field(
        default="sampled",
        description=(
            "How gate G10 verifies the written LAMMPS data file (§9). "
            "'full': re-parse every atom row (O(N), strongest, slowest — "
            "use for CI/paranoid runs). "
            "'sampled' (default): parse the header plus the first and last "
            "g10_readback_sample atom rows — O(1) in N, catches catastrophic "
            "writer/encoding/truncation breakage without the full re-read. "
            "'off': skip the on-disk read-back entirely (the writer's counts "
            "and bounds are still asserted by construction)."
        ),
    )
    g10_readback_sample: int = Field(
        default=1000,
        ge=1,
        description=(
            "Number of leading and trailing atom rows re-parsed when "
            "g10_readback='sampled'. Ignored for 'full'/'off'."
        ),
    )


class OutputXyzConfig(BaseModel):
    """Extended XYZ output settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Write an extended XYZ file.",
    )
    filename: str = Field(
        default="polycrystal.extxyz",
        description="Output filename for the extended XYZ file.",
    )


class OutputCsvConfig(BaseModel):
    """CSV report filenames."""

    model_config = ConfigDict(extra="forbid")

    grains: str = Field(
        default="grains.csv",
        description="Filename for the per-grain CSV report.",
    )
    boundaries: str = Field(
        default="boundaries.csv",
        description="Filename for the per-boundary CSV report.",
    )
    vertices: str = Field(
        default="vertices.csv",
        description=(
            "Filename for the Voronoi vertex CSV report "
            "(flat geometry only — legacy dump.dat parity)."
        ),
    )
    summary: str = Field(
        default="summary.csv",
        description=(
            "Filename for the long-format summary CSV "
            "(QA gate results, versions, timings, composition)."
        ),
    )


class OutputGnuplotConfig(BaseModel):
    """Gnuplot visualization bundle settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "Write gnuplot data files (seeds.dat, edges.dat / gb_points.dat, "
            "box.dat) and a ready-to-run view.plt script."
        ),
    )


class OutputMeshConfig(BaseModel):
    """Boundary mesh output settings (curved geometry, optional)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Write a boundary mesh file (marching-cubes per grain pair). "
            "Requires the 'mesh' optional extra (scikit-image)."
        ),
    )
    format: Literal["ply"] = Field(
        default="ply",
        description="Mesh file format. 'ply' is the only supported format.",
    )


class OutputConfig(BaseModel):
    """All output-file settings."""

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(
        default="./out",
        description=(
            "Output directory.  Created if it does not exist. "
            "The resolved config, run log, and all output files are written here."
        ),
    )
    methods_snippet: bool = Field(
        default=True,
        description=(
            "Write METHODS.md: an auto-generated methods "
            "paragraph describing the exact algorithm chain of THIS run "
            "with parameter values and numbered literature citations — a "
            "copy-paste starting point for the paper."
        ),
    )
    lammps: OutputLammpsConfig = Field(
        default_factory=OutputLammpsConfig,
        description="LAMMPS data-file output settings.",
    )
    xyz: OutputXyzConfig = Field(
        default_factory=OutputXyzConfig,
        description="Extended XYZ output settings.",
    )
    csv: OutputCsvConfig = Field(
        default_factory=OutputCsvConfig,
        description="CSV report filenames.",
    )
    gnuplot: OutputGnuplotConfig = Field(
        default_factory=OutputGnuplotConfig,
        description="Gnuplot visualization bundle settings.",
    )
    mesh: OutputMeshConfig = Field(
        default_factory=OutputMeshConfig,
        description="Optional boundary mesh output (curved geometry only).",
    )


class GbSegregationConfig(BaseModel):
    """GB-segregation control for one dopant."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Enable preferential placement in a shell around "
            "grain boundaries.  Disabled (default): uniform "
            "placement over the grain interior."
        ),
    )
    shell_width: float = Field(
        default=5.0, gt=0.0,
        description=(
            "Shell half-thickness in Å: a site with "
            "distance-to-GB ≤ shell_width belongs to the GB "
            "shell (distance via Tessellation.margin)."
        ),
    )
    enrichment: float = Field(
        default=1.0, gt=0.0,
        description=(
            "Target enrichment E = c_shell / c_bulk.  1.0 means "
            "no preference; large E concentrates dopants at "
            "boundaries.  Infeasible combinations (required "
            "shell probability > 1) raise ConfigError at "
            "placement time."
        ),
    )


class SitesCoordsConfig(BaseModel):
    """Explicit interstitial sublattice (all 230 space groups).

    By default each listed coordinate is expanded to its full
    symmetry orbit under the host crystal's space group (spglib
    operations, same Hall setting as the host), so one representative
    per Wyckoff orbit suffices — exactly like ``crystal.wyckoff_sites``
    for the host.  Listing an already-complete orbit is harmless
    (duplicates are removed), so the expansion is idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    coords: list[list[float]] = Field(
        min_length=1,
        description=(
            "Fractional coordinates [[x, y, z], ...] of the "
            "interstitial sites in the conventional cell.  Each "
            "entry must have exactly 3 components in [0, 1).  With "
            "expand_orbit (default) one representative per orbit "
            "suffices; the full orbit is generated automatically."
        ),
    )
    expand_orbit: bool = Field(
        default=True,
        description=(
            "Expand each coordinate to its full symmetry orbit under "
            "the host space group (spglib, same Hall setting as the "
            "host crystal) with duplicate removal — physically "
            "correct default: an interstitial sublattice must respect "
            "the host's site symmetry.  Set false to use the listed "
            "coordinates verbatim (deliberate symmetry-breaking "
            "escape hatch; you must list every equivalent site "
            "yourself or the sublattice will be incomplete)."
        ),
    )


class DopantConfig(BaseModel):
    """One dopant species and its placement rule."""

    model_config = ConfigDict(extra="forbid")

    element: str = Field(
        description=(
            "Dopant element symbol (e.g. 'C').  Must have a "
            "known mass (constants.ATOMIC_MASSES or an "
            "output.lammps.masses override) — checked at "
            "config time."
        ),
    )
    mode: Literal["substitutional", "interstitial"] = Field(
        description=(
            "substitutional: replace host atoms in place.  "
            "interstitial: insert new atoms on an interstitial "
            "sublattice of the host crystal."
        ),
    )
    concentration: float = Field(
        gt=0.0, lt=1.0,
        description=(
            "Target dopant atom fraction of the FINAL structure "
            "(n_dopant / n_total_after_doping)."
        ),
    )
    sites: Literal[
        "fcc_octahedral", "fcc_tetrahedral",
        "bcc_octahedral", "bcc_tetrahedral",
        "hcp_octahedral", "hcp_tetrahedral",
    ] | SitesCoordsConfig | None = Field(
        default=None,
        description=(
            "Interstitial only: named preset (fcc_* requires "
            "SG 225, bcc_* SG 229, hcp_* SG 194 with the motif "
            "on Wyckoff 2c) or explicit fractional coords for "
            "any other structure (any of the 230 space groups; "
            "orbit-expanded by default, see SitesCoordsConfig)."
        ),
    )
    host: str | None = Field(
        default=None,
        description=(
            "Substitutional only: replace only this host "
            "species.  Default: any host species (earlier "
            "dopants are never replaced)."
        ),
    )
    min_distance: float | None = Field(
        default=None, gt=0.0,
        description=(
            "Interstitial only, Å: candidate sites closer than "
            "this to ANY atom (host, earlier dopants, or "
            "accepted sites of this dopant) are rejected; gate "
            "G18 re-verifies after placement (hard)."
        ),
    )
    gb_segregation: GbSegregationConfig = Field(
        default_factory=GbSegregationConfig,
        description="GB-segregation control (disabled ⇒ uniform).",
    )


class DopingConfig(BaseModel):
    """Dopant insertion stage — runs after overlap removal."""

    model_config = ConfigDict(extra="forbid")

    dopants: list[DopantConfig] = Field(
        min_length=1,
        description=(
            "Dopants applied sequentially in list order; later "
            "dopants see earlier ones as atoms."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Top-level configuration model — the root of every grainsmith YAML file.

    This model maps 1-to-1 to the grainsmith YAML schema (§7).
    All sub-models enforce ``extra='forbid'`` so that typos in field names
    are caught at validation time (gate G1) rather than silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    meta: MetaConfig = Field(
        default_factory=MetaConfig,
        description="Run metadata: title and verbosity level.",
    )
    seed: SeedConfig = Field(
        default_factory=SeedConfig,
        description="RNG seed configuration (fixed or entropy).",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Execution-resource knobs (§13 memory guard); never affects an output byte.",
    )
    box: BoxConfig = Field(
        description="Simulation-box geometry (lengths, periodicity, vacuum).",
    )
    grains: GrainsConfig = Field(
        description="Grain count and seeding parameters.",
    )
    crystal: CrystalConfig | None = Field(
        default=None,
        description=(
            "Crystal structure (space group, lattice, Wyckoff sites) — "
            "the single-phase form. Exactly one of 'crystal' or 'phases' "
            "must be set (resolve Rule 21)."
        ),
    )
    phases: list[PhaseConfig] | None = Field(
        default=None,
        description=(
            "Multiphase polycrystal definition: >= 2 phases, "
            "each with its own crystal structure and a target VOLUME "
            "fraction (sum to 1). Mutually exclusive with 'crystal'. "
            "Grains are partitioned deterministically to match the "
            "fractions; gate G15 (warn) re-measures the achieved values."
        ),
    )
    orientation: OrientationConfig = Field(
        default_factory=OrientationConfig,
        description="Grain-orientation assignment scheme and parameters.",
    )
    boundaries: BoundariesConfig = Field(
        default_factory=BoundariesConfig,
        description="Grain-boundary geometry (flat or curved) and overlap removal.",
    )
    doping: DopingConfig | None = Field(
        default=None,
        description=(
            "Optional dopant insertion: "
            "GB-segregated substitutional and interstitial "
            "dopants, applied deterministically after overlap "
            "removal.  Writes doping.csv; gates G17 (warn) and "
            "G18 (hard).  Single-phase configs only."
        ),
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        description="Post-generation analysis options (voxel grid, GB character, CSL).",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output directory and per-format file settings.",
    )
