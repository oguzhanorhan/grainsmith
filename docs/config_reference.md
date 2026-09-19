# grainsmith configuration reference

> **GENERATED FILE — do not edit by hand.**
> Produced by `tools/gen_config_reference.py` from `grainsmith.config.schema`
> (the single source of truth). Regenerate after any schema change:
> `python -m tools.gen_config_reference`. A drift-proof test
> (`tests/test_docs.py`) fails if this file is out of sync with the schema.

Every model below enforces `extra='forbid'` — an unknown key is a config error
(gate G1), not a silent no-op. Cross-field rules that span models (e.g. "exactly
one of `crystal` or `phases`") live in `config/resolve.py` and are documented in
`docs/gates.md` (G1) and `docs/manual.md`, not here.

## RunConfig (root)

<a id="runconfig"></a>

Top-level configuration model — the root of every grainsmith YAML file.

| field | type | default | constraints | description |
|---|---|---|---|---|
| `meta` | [MetaConfig](#metaconfig) | (sub-block defaults) | — | Run metadata: title and verbosity level. |
| `seed` | [SeedConfig](#seedconfig) | (sub-block defaults) | — | RNG seed configuration (fixed or entropy). |
| `runtime` | [RuntimeConfig](#runtimeconfig) | (sub-block defaults) | — | Execution-resource knobs (§13 memory guard); never affects an output byte. |
| `box` | [BoxConfig](#boxconfig) | **required** | — | Simulation-box geometry (lengths, periodicity, vacuum). |
| `grains` | [GrainsConfig](#grainsconfig) | **required** | — | Grain count and seeding parameters. |
| `crystal` | [CrystalConfig](#crystalconfig) \| null | `None` | — | Crystal structure (space group, lattice, Wyckoff sites) — the single-phase form. Exactly one of 'crystal' or 'phases' must be set (resolve Rule 21). |
| `phases` | list[[PhaseConfig](#phaseconfig)] \| null | `None` | — | Multiphase polycrystal definition: >= 2 phases, each with its own crystal structure and a target VOLUME fraction (sum to 1). Mutually exclusive with 'crystal'. Grains are partitioned deterministically to match the fractions; gate G15 (warn) re-measures the achieved values. |
| `orientation` | [OrientationConfig](#orientationconfig) | (sub-block defaults) | — | Grain-orientation assignment scheme and parameters. |
| `boundaries` | [BoundariesConfig](#boundariesconfig) | (sub-block defaults) | — | Grain-boundary geometry (flat or curved) and overlap removal. |
| `doping` | [DopingConfig](#dopingconfig) \| null | `None` | — | Optional dopant insertion: GB-segregated substitutional and interstitial dopants, applied deterministically after overlap removal. Writes doping.csv; gates G17 (warn) and G18 (hard). Single-phase configs only. |
| `analysis` | [AnalysisConfig](#analysisconfig) | (sub-block defaults) | — | Post-generation analysis options (voxel grid, GB character, CSL). |
| `output` | [OutputConfig](#outputconfig) | (sub-block defaults) | — | Output directory and per-format file settings. |

## MetaConfig

<a id="metaconfig"></a>

Top-level run metadata.

**Config path:** `meta`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `title` | str | `'grainsmith run'` | — | Free-text title embedded in output file headers and the run log. |
| `verbose` | int | `1` | ≥ 0, ≤ 3 | Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG, 3=DEBUG + extra diagnostic file dumps. |

## SeedConfig

<a id="seedconfig"></a>

RNG seed configuration.

**Config path:** `seed`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `mode` | one of: 'fixed' \| 'entropy' | `'fixed'` | — | Seed mode. 'fixed': use the integer in 'value' (reproducible). 'entropy': harvest OS entropy; the resulting integer is recorded in summary.csv for later reproduction. |
| `value` | int \| null | `None` | — | Integer seed. Required when mode == 'fixed'. |

## RuntimeConfig

<a id="runtimeconfig"></a>

Execution-resource knobs (§13) — never affects a single output byte.

**Config path:** `runtime`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `memory_limit_gb` | float | `16.0` | > 0.0 | Per-allocation hard guard (decimal GB, §13): the estimated peak size of a single large allocation above which grainsmith raises a TessellationError instead of risking an OOM-kill. Applies to the per-grain lattice-enumeration grid (atoms/fill.py), the analysis voxel grid (tessellation/voxel.py), and the imported voxel-label EDT margin (tessellation/voxel_import.py). The EFFECTIVE budget actually enforced is min(this value, detected physical RAM) -- this value can never be raised past what the machine actually has (see memory.resolve_memory_budget). This is NOT a total-process memory cap -- that is the separate, optional '--max-rss' soft WARN monitor (memory.py), which watches cumulative RSS. Only the fill stage (atoms/fill.py) runs inside a '--jobs N' ProcessPoolExecutor: there, up to N grains fill concurrently, each allocating its own grid, so the machine needs headroom for N times this limit in the worst case; the voxel-grid and voxel-import guards each build exactly one grid on the single-threaded driver, unaffected by '--jobs'. |

## BoxConfig

<a id="boxconfig"></a>

Simulation-box geometry.

**Config path:** `box`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `lengths` | list[float] \| null | `None` | len ≥ 3, len ≤ 3 | Box edge lengths [Lx, Ly, Lz] in Å for an ORTHOGONAL box. Mutually exclusive with 'cells' (Rule 28). |
| `cells` | list[int] \| null | `None` | len ≥ 3, len ≤ 3 | [n1, n2, n3] positive integer lattice multiples for a TRICLINIC single-crystal box: the box vectors are h_i = n_i * a_i, where a_i are the crystal's own conventional cell vectors (restricted-triclinic convention: a along +x, b in the xy-plane with positive y, c with positive z — exactly the form crystal.cell.cell_matrix() returns). Commensurability then holds EXACTLY by construction (gate G14 measures 0 misfit). Requires grains.number: 1, box.periodic all true, box.vacuum: 0, and an identity crystal orientation (an arbitrary rotation would break the box-vector/lab-axis alignment) — enforced in resolve.py Rule 28. Mutually exclusive with 'lengths'. |
| `resolved_h` | list[list[float]] \| null | `None` | — | DERIVED (not user-set): the resolved (3,3) LAMMPS-tilt-reduced box matrix for box.cells, written back onto this config by config/resolve.py Rule 28 — the same 'resolve once, cache on the config' pattern crystal.cif uses. Row-major nested list (row i = box vector component i across a, b, c). None for the box.lengths (orthogonal) path. |
| `periodic` | list[bool] | `[True, True, True]` | len ≥ 3, len ≤ 3 | Per-axis periodicity flags [px, py, pz]. Set an axis to false for slab / thin-film geometry. |
| `vacuum` | float | `0.0` | ≥ 0.0 | Vacuum thickness in Å added on non-periodic axes. Requires at least one periodic=false axis (enforced in resolve.py). |

## GrainsConfig

<a id="grainsconfig"></a>

Grain count and seeding parameters.

**Config path:** `grains`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `number` | int \| null | `None` | ≥ 1 | Number of grains to generate. Required for all generated tessellations; for boundaries.geometry 'voxel_import' the count is DERIVED from the label field — omit it, or set it as a cross-check (a mismatch is an error). |
| `size_distribution` | [SizeDistributionConfig](#sizedistributionconfig) \| null | `None` | — | Optional target grain-size distribution. When set, the tessellation is a volume-fitted power (Laguerre) diagram; allowed with flat geometry or as the base of the warp method. |
| `min_seed_distance` | float \| one of: 'auto' | `'auto'` | — | Minimum centre-to-centre seed distance in Å for RSA placement. 'auto' = 1.0 × r_ws (Wigner–Seitz radius), safe below the RSA jamming limit. |
| `lloyd_iterations` | int | `0` | ≥ 0 | Number of Lloyd centroidal-relaxation iterations (voxel-grid based). 0 = off (Poisson–Voronoi); >0 produces more equiaxed grains. |

## SizeDistributionConfig

<a id="sizedistributionconfig"></a>

Target grain-size distribution (power/Laguerre + SDOT).

**Config path:** `grains.size_distribution`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | one of: 'lognormal' \| 'equal' \| 'volumes' | `'lognormal'` | — | Target volume model. 'lognormal': equivalent-sphere diameters d ~ LogN(0, sigma_log), V ∝ d³ (scale fixed by V_box/N — you prescribe the SHAPE). 'equal': V_box/N each. 'volumes': explicit relative volume list. |
| `sigma_log` | float | `0.35` | > 0.0 | Lognormal shape parameter: std of ln(d). |
| `volumes` | list[float] \| null | `None` | — | Relative target volumes (type: volumes); length must equal grains.number; normalized to the box volume. |
| `vol_tol` | float | `0.001` | > 0.0 | Gate G11 threshold: maximum relative per-grain volume error \|V_i − V_i^target\| / V_i^target on the final tessellation. |
| `max_iter` | int | `30` | ≥ 1 | Maximum damped-Newton iterations of the SDOT fit. |
| `centroidal_iterations` | int | `0` | ≥ 0 | Outer centroidal loop count (Kuhn et al. 2020): alternate volume fitting with moving seeds to power-cell centroids — equiaxed grains WITH prescribed volumes. 0 = off. |

## CrystalConfig

<a id="crystalconfig"></a>

Crystal structure definition: EITHER a manual space group + Wyckoff-site motif, OR direct input from a CIF file (mutually exclusive — resolve.py Rule 27).

**Config path:** `crystal` · `phases[].crystal`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `space_group` | [SpaceGroupConfig](#spacegroupconfig) \| null | `None` | — | Space-group identification (international number + optional setting). Required for the manual path; forbidden together with 'cif'. |
| `lattice` | [LatticeConfig](#latticeconfig) \| null | `None` | — | Lattice parameters for the crystal family (§6.1). Required for the manual path; forbidden together with 'cif'. |
| `wyckoff_sites` | list[[WyckoffSiteConfig](#wyckoffsiteconfig)] \| null | `None` | len ≥ 1 | List of representative Wyckoff sites defining the motif. Required for the manual path; forbidden together with 'cif'. |
| `cif` | [CifConfig](#cifconfig) \| null | `None` | — | Direct CIF input (ASE read + spglib symmetry detection). Mutually exclusive with the space_group/lattice/wyckoff_sites trio — exactly one of the two input forms must be given. |

## SpaceGroupConfig

<a id="spacegroupconfig"></a>

Space-group identification.

**Config path:** `crystal.space_group` · `phases[].crystal.space_group`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `number` | int | **required** | ≥ 1, ≤ 230 | International (Hermann–Mauguin) space-group number, 1–230. |
| `setting` | str \| null | `None` | — | Optional setting string for groups with multiple standard settings, e.g. '2' (origin choice 2), 'H' or 'R' (rhombohedral). null = first / standard setting. |

## LatticeConfig

<a id="latticeconfig"></a>

Lattice parameters — only the free parameters for the crystal family.

**Config path:** `crystal.lattice` · `phases[].crystal.lattice`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `a` | float \| null | `None` | > 0.0 | Lattice parameter a in Å. |
| `b` | float \| null | `None` | > 0.0 | Lattice parameter b in Å. |
| `c` | float \| null | `None` | > 0.0 | Lattice parameter c in Å. |
| `alpha` | float \| null | `None` | > 0.0, < 180.0 | Lattice angle α (bc plane) in degrees. |
| `beta` | float \| null | `None` | > 0.0, < 180.0 | Lattice angle β (ac plane) in degrees. |
| `gamma` | float \| null | `None` | > 0.0, < 180.0 | Lattice angle γ (ab plane) in degrees. |

## WyckoffSiteConfig

<a id="wyckoffsiteconfig"></a>

One representative Wyckoff site (fractional coordinates + species).

**Config path:** `crystal.wyckoff_sites[]` · `phases[].crystal.wyckoff_sites[]`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `element` | str \| dict[str, float] | **required** | — | Element symbol (e.g. 'Ni') for a fully-occupied site, or an occupancy dict (e.g. {'Ti': 0.90, 'Al': 0.10}) for a solid solution. Occupancy values must sum to 1 ± 1e-6. |
| `coords` | list[float] | **required** | len ≥ 3, len ≤ 3 | Representative fractional coordinates [x, y, z] of the Wyckoff position. Free parameters must be substituted with numeric values. |
| `letter` | str \| null | `None` | — | Wyckoff letter (e.g. 'a', 'b', '4c'). Optional; if provided, the crystal module verifies it matches the detected site symmetry. |

## CifConfig

<a id="cifconfig"></a>

Direct crystal-structure input from a CIF file (ASE + spglib path).

**Config path:** `crystal.cif` · `phases[].crystal.cif`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `file` | str | **required** | — | Path to a CIF file (any format ase.io.read accepts for '.cif'). Read with ASE; the space group is then determined by spglib — the file's own _symmetry_Int_Tables_number (if present) is cross-checked against the spglib-detected number; a mismatch only WARNS (the spglib-detected group is always the one used), since a CIF legally under-declaring its own symmetry (e.g. exporting every atom under 'P 1') is common and not itself an error. A relative path is resolved against the working directory first, then against the directory of the YAML file it appears in. |
| `symprec` | float | `0.0001` | > 0.0 | spglib symmetry-detection tolerance (Å-scale fractional distance) used for the CIF structure — analogous to constants.VERIFY_SYMPREC used for the manual-path G2 round-trip. Loosen for CIFs with larger coordinate roundoff (e.g. Rietveld-refined structures); tighten to separate nearly-degenerate settings. |

## PhaseConfig

<a id="phaseconfig"></a>

One phase of a multiphase polycrystal.

**Config path:** `phases[]`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `name` | str | **required** | len ≥ 1, pattern `^[A-Za-z0-9_-]+$` | Phase name (e.g. 'alpha', 'beta', 'Cu_fcc'); must be unique and match [A-Za-z0-9_-]+ — it appears in grains.csv / boundaries.csv columns, summary sections, and per-phase output filenames (mdf_<name>.csv, odf_mtex_<name>.txt). |
| `fraction` | float | **required** | > 0.0, < 1.0 | Target VOLUME fraction of this phase (the metallographic convention). Fractions must sum to 1 ± 1e-6; the achieved fractions are re-measured by gate G15 (warn — the deviation floor is set by the largest single grain). |
| `crystal` | [CrystalConfig](#crystalconfig) | **required** | — | Crystal structure of this phase (space group, lattice, Wyckoff sites) — phases may differ in structure AND in elements (e.g. HCP Ti / BCC Ti, or FCC Cu / BCC Fe). |

## OrientationConfig

<a id="orientationconfig"></a>

Grain-orientation assignment scheme and parameters.

**Config path:** `orientation`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `scheme` | one of: 'random_uniform' \| 'fixed' \| 'fiber' \| 'from_list' \| 'odf_components' \| 'imported' | `'random_uniform'` | — | Orientation assignment scheme. 'random_uniform': Haar-uniform random SO(3). 'fixed': same orientation for all grains (use 'fixed' sub-block). 'fiber': crystal axis aligned to a sample direction with angular spread. 'from_list': explicit per-grain list (length must equal grains.number). 'odf_components': weighted texture components (use 'components' list). 'imported': per-grain Euler angles from the voxel_import file (requires boundaries.voxel_import.euler_dataset). |
| `fixed` | [OrientationFixedConfig](#orientationfixedconfig) \| null | `None` | — | Used when scheme == 'fixed'. Specifies the single shared orientation. |
| `fiber` | [OrientationFiberConfig](#orientationfiberconfig) \| null | `None` | — | Used when scheme == 'fiber'. Specifies fiber-texture parameters. |
| `from_list` | list[[OrientationFixedConfig](#orientationfixedconfig)] \| null | `None` | — | Used when scheme == 'from_list'. One entry per grain; length must equal grains.number (checked at generation time). |
| `components` | list[[OdfComponentConfig](#odfcomponentconfig)] \| null | `None` | — | Used when scheme == 'odf_components': list of weighted texture components (Euler centre + spread, fiber, or one uniform random fraction). |
| `component_weight_basis` | one of: 'volume' \| 'count' | `'volume'` | — | How each OdfComponentConfig.weight is realised. 'volume' (default): weight is the VOLUME fraction of material in the component — the physical definition of the ODF; grains are partitioned deterministically (orientation.odf.volume_balanced_partition) so the realised volume fractions match the configured weights, at the cost of a size-component correlation that gate G25 reports. 'count': weight is realised as a GRAIN-COUNT fraction via an i.i.d. categorical draw (the pre-1.2 behaviour); the realised VOLUME fractions then differ from the configured weights by an amount set by the grain-size distribution (measured 0.745/0.255 for a configured 0.500/0.500 at 24 grains, sigma_log 0.5). Ignored unless scheme == 'odf_components'. |
| `mdf_target` | [MdfTargetConfig](#mdftargetconfig) \| null | `None` | — | Optional boundary-area-weighted disorientation-angle target (any scheme): orientation-assignment annealing pulls the neighbor misorientation-angle distribution toward the target. The NUMBER-weighted orientation distribution is invariant by construction; the VOLUME-weighted ODF is not, but mdf_target.odf_drift_max can bound its drift. Gate G12 (warn). |

## OrientationFixedConfig

<a id="orientationfixedconfig"></a>

Orientation specification for a single grain (one sub-key must be set).

**Config path:** `orientation.fixed` · `orientation.from_list[]`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `hkl_uvw` | [OrientationFixedHklUvw](#orientationfixedhkluvw) \| null | `None` | — | Fix orientation by (hkl) plane parallel to lab +z and [uvw] along lab +x. |
| `euler_bunge_deg` | list[float] \| null | `None` | len ≥ 3, len ≤ 3 | Bunge Euler angles (φ1, Φ, φ2) in degrees. |
| `axis_angle` | [AxisAngleConfig](#axisangleconfig) \| null | `None` | — | Orientation as a rotation axis + angle. |
| `quaternion` | list[float] \| null | `None` | len ≥ 4, len ≤ 4 | Orientation quaternion (w, x, y, z) — scalar-first, unit norm. |

## OrientationFixedHklUvw

<a id="orientationfixedhkluvw"></a>

Fix orientation by aligning a crystal plane and direction to lab axes.

**Config path:** `orientation.fixed.hkl_uvw` · `orientation.from_list[].hkl_uvw`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `plane` | list[int] | **required** | len ≥ 3, len ≤ 3 | Miller indices (h k l) of the crystal plane to align with lab +z. |
| `direction` | list[int] | **required** | len ≥ 3, len ≤ 3 | Miller direction [u v w] to align with lab +x. |

## AxisAngleConfig

<a id="axisangleconfig"></a>

Rotation expressed as an axis vector and an angle.

**Config path:** `orientation.fixed.axis_angle` · `orientation.from_list[].axis_angle`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `axis` | list[float] | **required** | len ≥ 3, len ≤ 3 | Rotation axis vector (need not be unit length; will be normalised). |
| `angle_deg` | float | **required** | — | Rotation angle in degrees. |

## OrientationFiberConfig

<a id="orientationfiberconfig"></a>

Fiber-texture orientation parameters.

**Config path:** `orientation.fiber` · `orientation.components[].fiber`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `crystal_axis` | list[float] | **required** | len ≥ 3, len ≤ 3 | Crystal axis <u v w> to align with the sample direction. |
| `sample_direction` | str | `'z'` | — | Sample direction to align to: 'x', 'y', or 'z'. |
| `spread_deg` | float | `10.0` | ≥ 0.0, ≤ 180.0 | Gaussian angular spread (standard deviation) around the fiber axis in degrees. |

## OdfComponentConfig

<a id="odfcomponentconfig"></a>

One ODF texture component.

**Config path:** `orientation.components[]`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `euler_bunge_deg` | list[float] \| null | `None` | len ≥ 3, len ≤ 3 | Component centre as Bunge Euler angles (φ1, Φ, φ2) in degrees, e.g. Cu/S/Brass rolling components. |
| `fiber` | [OrientationFiberConfig](#orientationfiberconfig) \| null | `None` | — | Fiber component reusing the fiber machinery (crystal_axis / sample_direction / spread_deg). |
| `random` | bool | `False` | — | Haar-uniform 'random fraction' component for partially textured states. At most one such component is allowed. |
| `weight` | float | **required** | > 0.0 | Relative component weight (normalized). How it is realised depends on orientation.component_weight_basis: under 'volume' (default) it is the target VOLUME fraction of material assigned to this component (deterministic grain-to-component partition); under 'count' grains draw their component from the categorical distribution ∝ weight, so it is a GRAIN-COUNT fraction instead. |
| `spread_deg` | float | `0.0` | ≥ 0.0, ≤ 180.0 | Isotropic angular spread (std of the rotation angle, degrees) around an euler_bunge_deg centre — SO(3) von Mises–Fisher small-angle surrogate with exact Haar sin²(θ/2) correction. 0 reduces the component to a fixed orientation. Ignored for fiber components (they carry their own spread_deg) and forbidden for random components. |

## MdfTargetConfig

<a id="mdftargetconfig"></a>

Boundary-area-weighted disorientation-angle target via assignment annealing.

**Config path:** `orientation.mdf_target`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | one of: 'mackenzie' \| 'haar_random' \| 'csl_enriched' \| 'sigma3_angle_enriched' \| 'histogram' | `'haar_random'` | — | Target for the boundary-AREA-weighted marginal distribution of the disorientation ANGLE — this is NOT a misorientation distribution function (an MDF is a density over the full misorientation space; this objective never sees the misorientation axis). 'haar_random': the random-pair disorientation-angle reference of the ACTUAL point group; this estimates the Mackenzie cubic law only for the 24-rotation proper group of m-3m. Other point groups have different references. 'mackenzie' is a deprecated alias for 'haar_random'. 'sigma3_angle_enriched': 'haar_random' base mixed with a window-only distribution using sigma3_fraction as the mixture coefficient in the Σ3 Brandon ANGULAR window 60° ± 15°/√3 — this enforces only the NECESSARY angular condition; the ⟨111⟩ axis condition that CSL classification also requires is NOT part of the objective (cubic only). 'csl_enriched' is a deprecated alias for 'sigma3_angle_enriched'. 'histogram': explicit bin_edges + densities. |
| `sigma3_fraction` | float | `0.3` | > 0.0, < 1.0 | sigma3_angle_enriched (and its deprecated alias csl_enriched): mixture coefficient f in target = (1-f)*haar_random + f*angle_window, where angle_window is supported on 60° ± 15°/√3. This is NOT the total target window fraction: the Haar reference already has mass in that window. This is angle-only — the realised CSL Σ3 AREA fraction (the true class, axis condition included) is measured separately; compare it against the realised angle-window fraction in G23. |
| `bin_edges` | list[float] \| null | `None` | — | histogram: increasing misorientation-angle bin edges in degrees over [0, θ_max]; length = len(densities) + 1. |
| `densities` | list[float] \| null | `None` | — | histogram: non-negative target densities per bin (normalized internally to unit integral). |
| `annealing_steps` | int | `20000` | ≥ 0 | Number of swap moves of the simulated annealing. 0 = no annealing — only measure the χ² distance (gate G12). |
| `t0` | float | `0.05` | > 0.0 | Initial annealing temperature (χ² energy units). |
| `cooling` | float | `0.995` | > 0.0, < 1.0 | Geometric cooling ratio: T_k = t0 · cooling^k. |
| `chi2_max` | float | `0.5` | > 0.0 | Gate G12 threshold (warn): symmetric χ² distance between the re-measured area-weighted disorientation-angle histogram and the target. The default covers the multinomial sampling noise floor ~n_bins/n_pairs of typical runs (e.g. 40 bins / 150 pairs ≈ 0.27). |
| `odf_drift_max` | float \| null | `None` | ≥ 0.0 | Hard cap on d_TV(f_V(π), f_V(π₀)) — the total-variation drift of the VOLUME-weighted ODF away from the sampler's own assignment π₀ — enforced on every ACCEPTED swap of the annealing. None (default): the drift is measured and reported but never constrained. Measured trade-off on a 60-grain lognormal test case: a cap of 0.01 already recovers about half of the achievable χ² reduction, and 0.05 about three quarters — pick a value with that trade-off in mind. See orientation/odf.py. |
| `odf_kernel_halfwidth_deg` | float | `10.0` | > 0.0, < 180.0 | Requested de la Vallée Poussin kernel half-width (degrees) for the volume-weighted ODF diagnostic. The degree is rounded to the nearest positive integer to keep the kernel positive semidefinite; the effective half-width is reported separately (at most 90 degrees). Configures the DIAGNOSTIC only — it has no effect on odf_drift_max, which caps the un-smoothed (atomic) drift and, by the Markov-contraction argument, therefore already bounds the TV of kernel-smoothed densities at every half-width simultaneously (not the separate MMD norm) (orientation/odf.py). |
| `odf_null_samples` | int | `256` | ≥ 1 | Sample count for the ODF kernel-discrepancy null distribution (orientation.odf.null_mmd). Configures the DIAGNOSTIC only; has no effect on the annealing itself or on odf_drift_max. |

## BoundariesConfig

<a id="boundariesconfig"></a>

Grain-boundary geometry and overlap removal.

**Config path:** `boundaries`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `geometry` | one of: 'flat' \| 'curved' \| 'voxel_import' | `'flat'` | — | Boundary geometry. 'flat': planar Voronoi faces (Qhull). 'curved': domain-warp or weighted-Voronoi method (use 'curved' sub-block). 'voxel_import': externally produced voxel label field (use 'voxel_import' sub-block). |
| `curved` | [CurvedBoundaryConfig](#curvedboundaryconfig) \| null | `None` | — | Curved-boundary parameters. Used when geometry == 'curved'. Providing this block with geometry == 'flat' is a config error (it would otherwise be silently ignored). |
| `voxel_import` | [VoxelImportConfig](#voxelimportconfig) \| null | `None` | — | Imported label-field parameters. Required when geometry == 'voxel_import'. |
| `overlap_removal` | [OverlapRemovalConfig](#overlapremovalconfig) | (sub-block defaults) | — | Atom-overlap removal settings applied after the fill stage. |

## CurvedBoundaryConfig

<a id="curvedboundaryconfig"></a>

Parameters for curved-boundary methods (§3.3, §6.6).

**Config path:** `boundaries.curved`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `method` | one of: 'warp' \| 'additive_weights' \| 'anisotropic' \| 'perturbed_distance' | `'warp'` | — | Curved-boundary method. 'warp': domain-warped Voronoi via periodic GRF (default, §6.6). 'additive_weights': Johnson–Mehl / Apollonius diagram. 'anisotropic': per-grain ellipsoidal metric (GBPD family). 'perturbed_distance': level-set membership argmin_i[d_pbc(x,s_i) − amplitude·η_i(x)] with an INDEPENDENT per-grain scalar field η_i (the same spectrum/hurst/l_min/l_max/amplitude schema keys warp also declares, reused verbatim) — unlike warp's coordinate diffeomorphism (bijective, so it cannot change the flat-Voronoi surface's fractal dimension; this is WHY warp does not accept spectrum: self_affine at all), this method perturbs the ASSIGNMENT RULE itself and so can produce a genuinely self-affine grain boundary whose box-counting dimension responds to hurst — the only method that accepts spectrum: self_affine (docs/physics.md §5b). G6 (bijectivity) does not apply; a seed-containment guard and an exact seed-ownership check replace it. G5 is a repair-and-report gate here (reassigned_fraction), not a hard failure. |
| `base` | one of: 'flat' \| 'additive_weights' \| 'anisotropic' \| 'power' | `'flat'` | — | Base tessellation for the warp method (ignored for other methods). 'power' requires grains.size_distribution (a volume-fitted Laguerre base) — and is implied when size_distribution is present with base 'flat'. |
| `amplitude` | float | `4.0` | ≥ 0.0 | Warp-field RMS amplitude per component in Å. Hard constraint: amplitude ≤ min_seed_distance / 4 (enforced downstream). |
| `correlation_length` | float | `15.0` | > 0.0 | Gaussian correlation length ℓ in Å — the boundary 'waviness wavelength'. Larger ℓ produces smoother, more gently curved boundaries. Used only by the gaussian spectrum; ignored by perturbed_distance's self_affine spectrum (band limits l_min/l_max take its place there — warp does not accept self_affine at all, so this key and self_affine are never both active together). |
| `spectrum` | one of: 'gaussian' \| 'self_affine' | `'gaussian'` | — | Field amplitude spectrum (restricted per-method — see below). 'gaussian': single-scale exp(−k²ℓ²/4) (default, bit-identical); the ONLY spectrum 'warp' accepts — a ConfigError rejects 'warp' + 'self_affine' (warp is a coordinate diffeomorphism, which cannot change the flat-Voronoi surface's fractal dimension, so a self-affine spectrum on it would be undetectable in the as-built boundary; docs/physics.md §5b). 'self_affine': band-limited power law √Φ(k) ∝ k^(−(3+2H)/2) for k ∈ [2π/l_max, 2π/l_min] — scale-free roughness with field fractal dimension D = 3 − hurst; valid ONLY with method: perturbed_distance, where perturbing the assignment rule (not the coordinate) lets the boundary's own box-counting dimension respond to hurst (gate G20 measures it; gate G13 is its Hurst-back-estimation analogue, also perturbed_distance-only). |
| `hurst` | float | `0.8` | > 0.0, ≤ 1.0 | Hurst exponent H ∈ (0, 1] of the self_affine spectrum (perturbed_distance only; surface PSD C(q) ∝ q^(−2−2H); typical metal surfaces/GBs: 0.7–0.9). |
| `l_min` | float | `8.0` | > 0.0 | Shortest roughness wavelength in Å (self_affine band upper k-limit 2π/l_min; perturbed_distance only — warp never consumes l_min at all, since it cannot accept spectrum: self_affine in the first place). Must satisfy l_min ≥ 2·h_field (grid Nyquist, checked against the actual field grid at construction) and l_min < l_max. perturbed_distance has no bijectivity guard (no G6) to tie l_min to an amplitude ceiling: its amplitude ceiling (ETA_CLIP/PERTURBED_DISTANCE_SAFETY, seed-containment + exact seed-ownership) is a function of min_seed_distance only and does not depend on l_min. |
| `l_max` | float | `60.0` | > 0.0 | Longest roughness wavelength in Å (self_affine band lower k-limit 2π/l_max; perturbed_distance only). Must satisfy l_min < l_max ≤ min(L)/2. |
| `amplitude_convention` | one of: 'total_rms' \| 'reference_wavelength' | `'total_rms'` | — | How `amplitude` (self_affine only; opt-in) scales the band-limited η field. 'total_rms' (default, UNCHANGED behavior): `amplitude` is the field's total RMS over the WHOLE synthesis band [l_min, l_max] — widening the band (more octaves) REDISTRIBUTES this fixed budget across more scales rather than adding to it, so at fixed `amplitude` adding a coarse octave can paradoxically REDUCE small-scale (l_min-scale) roughness — see docs/physics.md §5b for the measured reversal this causes in specific GB area (S_V). 'reference_wavelength': `amplitude` instead fixes the RMS power contributed by ONE reference octave anchored at `reference_wavelength` (default l_max), leaving every other resolved octave's contribution free to ACCUMULATE on top — restoring the expected (monotonic) coarse-octave response and giving `amplitude` a stable physical meaning (Å-RMS at a named wavelength) independent of how many octaves the band happens to span. Exact, invertible mapping to the default convention: `amplitude_total_rms = amplitude_A0 / kappa(hurst, l_min, l_max, reference_wavelength)` — see `tessellation.warp.reference_shell_kappa`. Valid only with spectrum: self_affine (perturbed_distance only); the seed-containment guard binds the REALIZED total-RMS-equivalent amplitude in either convention, never just the raw `amplitude` value, so switching convention cannot weaken it. |
| `reference_wavelength` | float \| null | `None` | > 0.0 | Anchor wavelength in Å for the top reference octave [reference_wavelength/2, reference_wavelength] that `amplitude_convention: reference_wavelength` fixes the RMS power of (self_affine only). Default `None` = l_max (an anchor that MOVES with the band — natural when l_max already represents the physical grain radius). A fixed absolute wavelength can be given instead — required for a band-WIDTH scan at constant grain size to accumulate variance monotonically in both directions (the l_max-anchored default does not, by construction, correct THAT specific diagnostic — see docs/physics.md §5b). Must lie within [l_min, l_max] when given; must not be set under `amplitude_convention: total_rms` (a silently-ignored field is a config error here, not a no-op). |
| `weight_sigma` | float | `0.0` | ≥ 0.0 | Weight spread σ_w in Å for the additive_weights method. Constraint: σ_w ≤ min_seed_distance / 6. |
| `aspect_ratio_range` | list[float] | `[1.0, 1.0]` | len ≥ 2, len ≤ 2 | Allowed range [min, max] of per-grain aspect ratios for the anisotropic method. |

## VoxelImportConfig

<a id="voxelimportconfig"></a>

Imported voxel label field.

**Config path:** `boundaries.voxel_import`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `file` | str | **required** | — | Path to the label field: .npy (3D integer array, internal (Nx, Ny, Nz) layout) or DREAM.3D HDF5 (.dream3d/.h5/.hdf5; requires the [import] extra / h5py). The grid shape comes from the file; box.lengths from the config. A relative path is resolved against the working directory first, then against the directory of the YAML file it appears in. |
| `dataset` | str \| null | `None` | — | HDF5 path to the FeatureIds cell array (required for DREAM.3D files), e.g. 'DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds'. Ignored for .npy. |
| `relabel` | bool | `True` | — | Compact arbitrary grain ids to 0 … N−1 (the original → compact map is recorded in summary.csv). When false, the field must already be compact. |
| `strict_connectivity` | bool | `False` | — | Gate G5 severity for the imported field: external microstructures may legitimately contain disconnected or wrap-spanning grains, so G5 is demoted to WARN by default; set true to hard-fail on fragmented grains. |
| `euler_dataset` | str \| null | `None` | — | Optional HDF5 path to per-grain Bunge Euler angles (N, 3), indexed by the ORIGINAL feature ids (e.g. DREAM.3D '.../Grain Data/AvgEulerAngles'). Required by orientation.scheme 'imported'. |
| `euler_degrees` | bool | `False` | — | Unit of euler_dataset angles: false = radians (the DREAM.3D convention), true = degrees. |

## OverlapRemovalConfig

<a id="overlapremovalconfig"></a>

Atom-overlap removal parameters (§6.9).

**Config path:** `boundaries.overlap_removal`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | bool | `True` | — | Whether to run the GB overlap-removal pass. |
| `cutoff` | float \| str | `'0.85*d_nn'` | — | Overlap cutoff as an absolute distance in Å, or as an expression referencing 'd_nn' (nearest-neighbour distance of the ideal crystal). Example: '0.85*d_nn' (default) or '1.5' (Å). |
| `policy` | one of: 'delete_shallower' \| 'keep_lower_id' \| 'midpoint_merge' | `'delete_shallower'` | — | Deletion policy for overlapping inter-grain atom pairs. 'delete_shallower': remove the atom with smaller boundary margin. 'keep_lower_id': always keep the atom belonging to the lower grain id. 'midpoint_merge': replace the pair with one atom at the midpoint (single-species systems only). |

## DopingConfig

<a id="dopingconfig"></a>

Dopant insertion stage — runs after overlap removal.

**Config path:** `doping`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `dopants` | list[[DopantConfig](#dopantconfig)] | **required** | len ≥ 1 | Dopants applied sequentially in list order; later dopants see earlier ones as atoms. |

## DopantConfig

<a id="dopantconfig"></a>

One dopant species and its placement rule.

**Config path:** `doping.dopants[]`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `element` | str | **required** | — | Dopant element symbol (e.g. 'C'). Must have a known mass (constants.ATOMIC_MASSES or an output.lammps.masses override) — checked at config time. |
| `mode` | one of: 'substitutional' \| 'interstitial' | **required** | — | substitutional: replace host atoms in place. interstitial: insert new atoms on an interstitial sublattice of the host crystal. |
| `concentration` | float | **required** | > 0.0, < 1.0 | Target dopant atom fraction of the FINAL structure (n_dopant / n_total_after_doping). |
| `sites` | one of: 'fcc_octahedral' \| 'fcc_tetrahedral' \| 'bcc_octahedral' \| 'bcc_tetrahedral' \| 'hcp_octahedral' \| 'hcp_tetrahedral' \| [SitesCoordsConfig](#sitescoordsconfig) \| null | `None` | — | Interstitial only: named preset (fcc_* requires SG 225, bcc_* SG 229, hcp_* SG 194 with the motif on Wyckoff 2c) or explicit fractional coords for any other structure (any of the 230 space groups; orbit-expanded by default, see SitesCoordsConfig). |
| `host` | str \| null | `None` | — | Substitutional only: replace only this host species. Default: any host species (earlier dopants are never replaced). |
| `min_distance` | float \| null | `None` | > 0.0 | Interstitial only, Å: candidate sites closer than this to ANY atom (host, earlier dopants, or accepted sites of this dopant) are rejected; gate G18 re-verifies after placement (hard). |
| `gb_segregation` | [GbSegregationConfig](#gbsegregationconfig) | (sub-block defaults) | — | GB-segregation control (disabled ⇒ uniform). |

## SitesCoordsConfig

<a id="sitescoordsconfig"></a>

Explicit interstitial sublattice (all 230 space groups).

**Config path:** `doping.dopants[].sites`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `coords` | list[list[float]] | **required** | len ≥ 1 | Fractional coordinates [[x, y, z], ...] of the interstitial sites in the conventional cell. Each entry must have exactly 3 components in [0, 1). With expand_orbit (default) one representative per orbit suffices; the full orbit is generated automatically. |
| `expand_orbit` | bool | `True` | — | Expand each coordinate to its full symmetry orbit under the host space group (spglib, same Hall setting as the host crystal) with duplicate removal — physically correct default: an interstitial sublattice must respect the host's site symmetry. Set false to use the listed coordinates verbatim (deliberate symmetry-breaking escape hatch; you must list every equivalent site yourself or the sublattice will be incomplete). |

## GbSegregationConfig

<a id="gbsegregationconfig"></a>

GB-segregation control for one dopant.

**Config path:** `doping.dopants[].gb_segregation`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | Enable preferential placement in a shell around grain boundaries. Disabled (default): uniform placement over the grain interior. |
| `shell_width` | float | `5.0` | > 0.0 | Shell half-thickness in Å: a site with distance-to-GB ≤ shell_width belongs to the GB shell (distance via Tessellation.margin). |
| `enrichment` | float | `1.0` | > 0.0 | Target enrichment E = c_shell / c_bulk. 1.0 means no preference; large E concentrates dopants at boundaries. Infeasible combinations (required shell probability > 1) raise ConfigError at placement time. |

## AnalysisConfig

<a id="analysisconfig"></a>

Post-generation analysis options.

**Config path:** `analysis`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `statistics` | bool | `True` | — | Write the publication statistics outputs: statistics.csv (grain-size + log-normal fit, sphericity, S_V, triple-junction L_V, GB character/CSL area fractions; phase-aware) and microstructure.json (the machine-readable record of the whole run — dataset-repository ready). |
| `section` | [SectionConfig](#sectionconfig) \| null | `None` | — | Optional EBSD-like 2D section export (slice_<axis><position>.csv). |
| `voxel_grid` | int \| one of: 'auto' | `'auto'` | — | Voxel grid resolution. 'auto': derive from ℓ/4 and r_ws/10, capped at VOXEL_GRID_MAX³. Integer: explicit number of voxels along the shortest box edge. |
| `gb_character` | bool | `True` | — | Compute per-boundary GB character (tilt/twist/mixed) and boundary-plane Miller indices for both crystal frames. |
| `csl` | bool | `False` | — | Attempt CSL Σ assignment using the built-in cubic CSL table. Requires a cubic point group (enforced in resolve.py); ignored otherwise with a ConfigError. |
| `gb_curvature` | bool | `False` | — | Compute local grain-boundary mean/Gaussian curvature (H in 1/Å, K in 1/Å²) at the boundary sample points: appends six area-weighted per-boundary columns to boundaries.csv and writes the local samples to gb_curvature.csv (fixed name). Flat geometry writes exact zeros — planar boundaries have no curvature by construction. Sign: H > 0 means grain_i is locally convex. Not supported with geometry 'voxel_import' (piecewise-constant margins; rejected at config time). |
| `per_atom_margin` | bool | `False` | — | Add a 'gb_margin' column (signed distance to nearest GB in Å) to the extended XYZ output. |
| `mdf_bins` | int | `40` | ≥ 4 | Number of misorientation-angle bins for mdf.csv and for the haar_random/sigma3_angle_enriched angle targets over [0, θ_max] of the point group. |

## SectionConfig

<a id="sectionconfig"></a>

EBSD-like 2D section export.

**Config path:** `analysis.section`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `axis` | one of: 'x' \| 'y' \| 'z' | `'z'` | — | Box axis NORMAL to the section plane. |
| `position` | float | `0.5` | ≥ 0.0, < 1.0 | Fractional position of the plane along the axis (0 = low box face, 0.5 = midplane). |

## OutputConfig

<a id="outputconfig"></a>

All output-file settings.

**Config path:** `output`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `directory` | str | `'./out'` | — | Output directory. Created if it does not exist. The resolved config, run log, and all output files are written here. |
| `methods_snippet` | bool | `True` | — | Write METHODS.md: an auto-generated methods paragraph describing the exact algorithm chain of THIS run with parameter values and numbered literature citations — a copy-paste starting point for the paper. |
| `lammps` | [OutputLammpsConfig](#outputlammpsconfig) | (sub-block defaults) | — | LAMMPS data-file output settings. |
| `xyz` | [OutputXyzConfig](#outputxyzconfig) | (sub-block defaults) | — | Extended XYZ output settings. |
| `csv` | [OutputCsvConfig](#outputcsvconfig) | (sub-block defaults) | — | CSV report filenames. |
| `gnuplot` | [OutputGnuplotConfig](#outputgnuplotconfig) | (sub-block defaults) | — | Gnuplot visualization bundle settings. |
| `mesh` | [OutputMeshConfig](#outputmeshconfig) | (sub-block defaults) | — | Optional boundary mesh output (curved geometry only). |

## OutputLammpsConfig

<a id="outputlammpsconfig"></a>

LAMMPS data-file output settings.

**Config path:** `output.lammps`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `filename` | str | `'polycrystal.data'` | — | Output filename for the LAMMPS data file. |
| `atom_style` | one of: 'atomic' \| 'molecular' | `'atomic'` | — | LAMMPS atom_style. 'atomic': id type x y z. 'molecular': id mol type x y z with mol = grain_id (1-based), enabling per-grain coloring in OVITO via molecule-id. |
| `masses` | dict[str, float] \| null | `None` | — | Optional per-element mass overrides in amu for the Masses section (e.g. {Fe: 55.0}). Elements not listed fall back to constants.ATOMIC_MASSES; unknown elements without an override raise a ConfigError. |
| `g10_readback` | one of: 'full' \| 'sampled' \| 'off' | `'sampled'` | — | How gate G10 verifies the written LAMMPS data file (§9). 'full': re-parse every atom row (O(N), strongest, slowest — use for CI/paranoid runs). 'sampled' (default): parse the header plus the first and last g10_readback_sample atom rows — O(1) in N, catches catastrophic writer/encoding/truncation breakage without the full re-read. 'off': skip the on-disk read-back entirely (the writer's counts and bounds are still asserted by construction). |
| `g10_readback_sample` | int | `1000` | ≥ 1 | Number of leading and trailing atom rows re-parsed when g10_readback='sampled'. Ignored for 'full'/'off'. |

## OutputXyzConfig

<a id="outputxyzconfig"></a>

Extended XYZ output settings.

**Config path:** `output.xyz`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | bool | `True` | — | Write an extended XYZ file. |
| `filename` | str | `'polycrystal.extxyz'` | — | Output filename for the extended XYZ file. |

## OutputCsvConfig

<a id="outputcsvconfig"></a>

CSV report filenames.

**Config path:** `output.csv`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `grains` | str | `'grains.csv'` | — | Filename for the per-grain CSV report. |
| `boundaries` | str | `'boundaries.csv'` | — | Filename for the per-boundary CSV report. |
| `vertices` | str | `'vertices.csv'` | — | Filename for the Voronoi vertex CSV report (flat geometry only — legacy dump.dat parity). |
| `summary` | str | `'summary.csv'` | — | Filename for the long-format summary CSV (QA gate results, versions, timings, composition). |

## OutputGnuplotConfig

<a id="outputgnuplotconfig"></a>

Gnuplot visualization bundle settings.

**Config path:** `output.gnuplot`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | bool | `True` | — | Write gnuplot data files (seeds.dat, edges.dat / gb_points.dat, box.dat) and a ready-to-run view.plt script. |

## OutputMeshConfig

<a id="outputmeshconfig"></a>

Boundary mesh output settings (curved geometry, optional).

**Config path:** `output.mesh`

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | Write a boundary mesh file (marching-cubes per grain pair). Requires the 'mesh' optional extra (scikit-image). |
| `format` | one of: 'ply' | `'ply'` | — | Mesh file format. 'ply' is the only supported format. |

