# grainsmith outputs

Every run writes into `output.directory` (default `./out`). Files fall into six
groups: **atomistic models** (for MD), **reports** (CSV/JSON tables), **the
publication trio** (statistics + machine-readable record + methods), **the
visualization bundle** (gnuplot), an **optional boundary mesh** (PLY), and
**provenance** (log, manifest, resolved config, `verbose >= 3` diagnostics).
Filenames in the *configurable* column can be renamed in the config; the
rest are fixed.

## File inventory

| file | group | configurable? | written when |
|---|---|---|---|
| `polycrystal.data` | model | `output.lammps.filename` | always |
| `polycrystal.extxyz` | model | `output.xyz.filename` (`output.xyz.enabled`) | `xyz.enabled` (default on) |
| `grains.csv` | report | `output.csv.grains` | always |
| `boundaries.csv` | report | `output.csv.boundaries` | always |
| `vertices.csv` | report | `output.csv.vertices` | flat and power/Laguerre (SDOT) geometry only |
| `summary.csv` | report | `output.csv.summary` | always |
| `statistics.csv` | publication | fixed | `analysis.statistics` (default on) |
| `microstructure.json` | publication | fixed | `analysis.statistics` (default on) |
| `METHODS.md` | publication | fixed | `output.methods_snippet` (default on) |
| `mdf.csv` | report | fixed | always (single-phase) |
| `odf_mtex.txt` | report | fixed | always (single-phase) |
| `mdf_<name>.csv` | report | fixed | per phase (multiphase) |
| `odf_mtex_<name>.txt` | report | fixed | per phase (multiphase) |
| `slice_<axis><pos>.csv` | report | fixed | `analysis.section` set |
| `gb_curvature.csv` | report | fixed | `analysis.gb_curvature` |
| `doping.csv` | report | fixed | `doping.dopants` non-empty |
| `doping_profile.csv` | report | fixed | `doping.dopants` non-empty |
| `seeds.dat`, `edges.dat` / `gb_points.dat`, `box.dat`, `view.plt` | viz | fixed | `output.gnuplot.enabled` (default on) |
| `curvature_hist.plt` | viz | fixed | `analysis.gb_curvature` + gnuplot enabled |
| `doping_profile.plt` | viz | fixed | doping non-empty + gnuplot enabled |
| `boundaries.ply` | mesh | fixed (`output.mesh.enabled`/`.format`) | `mesh.enabled` (curved; needs scikit-image) |
| `diagnostics/grain_atom_counts.dat` | provenance | fixed | `meta.verbose >= 3` |
| `diagnostics/warp_field_stats.dat` | provenance | fixed | `meta.verbose >= 3` and `warp` tessellation |
| `diagnostics/voxel_histogram.dat` | provenance | fixed | `meta.verbose >= 3` and curved (non-flat) geometry |
| `resolved_config.yaml` | provenance | fixed | always |
| `run.log` | provenance | fixed | always |
| `MANIFEST.txt` | provenance | fixed | always (SHA-256 of every output) |

`edges.dat` is written for flat geometry, `gb_points.dat` for curved.

## Atomistic models

**`polycrystal.data`** — LAMMPS data file. `atom_style: atomic` → `id type x y z`;
`atom_style: molecular` → `id mol type x y z` with `mol = grain_id + 1` (so OVITO
can colour by molecule = grain). A `Masses` section is written (from
`output.lammps.masses` or `constants.ATOMIC_MASSES`). Gate **G10** re-parses this
file to guarantee round-trip integrity.

**`polycrystal.extxyz`** — extended XYZ. Per-atom columns: `species x y z grain`
(+ `gb_margin` when `analysis.per_atom_margin: true` — signed distance to the
nearest GB in Å). The header `Lattice`/`Properties` line follows the extxyz
convention (OVITO/ASE-readable).

## Reports

**`grains.csv`** — one row per grain:

| column | meaning |
|---|---|
| `grain_id` | 0-based grain index |
| `seed_x/y/z` | seed coordinates (Å) |
| `volume_A3`, `volume_fraction` | cell volume and its fraction of the box |
| `n_atoms` | atoms in the grain after overlap removal and dopant insertion |
| `n_neighbors` | number of distinct neighbour grains |
| `q_w/q_x/q_y/q_z` | orientation quaternion (scalar-first, unit) |
| `euler_phi1_deg/Phi/phi2_deg` | Bunge Euler angles (φ1, Φ, φ2), MTEX/EBSD convention |
| `axis_x/y/z`, `angle_deg` | orientation as axis–angle |
| `z_plane_hkl`, `z_plane_dev_deg` | crystal plane nearest lab +z and its misfit |
| `x_dir_uvw`, `x_dir_dev_deg` | crystal direction nearest lab +x and its misfit |

**`boundaries.csv`** — one row per grain–grain interface:

| column | meaning |
|---|---|
| `grain_i`, `grain_j` | the two grains (i < j) |
| `seed_distance_A` | centre-to-centre distance (Å) |
| `misorientation_deg` | disorientation angle (minimal over symmetry); `NaN` for interphase |
| `axis_u/v/w`, `axis_dev_deg` | disorientation axis (Miller) and its spread |
| `area_A2` | interface area (Å²) |
| `mean_normal_x/y/z`, `normal_spread_deg` | mean boundary-plane normal and its scatter |
| `plane_i_hkl/dev_deg`, `plane_j_hkl/dev_deg` | boundary-plane Miller indices in each crystal frame |
| `character`, `character_angle_deg` | tilt / twist / mixed and the defining angle |
| `csl_sigma` | CSL Σ (cubic only; blank if none/non-cubic) |
| `n_overlap_deleted` | atoms removed along this boundary |
| `H_mean_invA`, `H_std_invA`, `H_abs_mean_invA` | area-weighted mean/std/mean-|H| local mean curvature (1/Å); `analysis.gb_curvature` only |
| `K_mean_invA2`, `K_std_invA2` | area-weighted mean/std local Gaussian curvature (1/Å²); `analysis.gb_curvature` only |
| `curv_n_samples` | number of retained (non-degenerate) local curvature samples for this boundary; `analysis.gb_curvature` only |

**`vertices.csv`** (flat and power/Laguerre geometry only) — Voronoi vertices
(legacy `dump.dat` parity): vertex coordinates and the grains meeting there.

**`gb_curvature.csv`** (`analysis.gb_curvature`) — one row per retained local
curvature sample on a grain-boundary (curved geometry) or per exact face
polygon with `H = K = 0` (flat geometry):

| column | meaning |
|---|---|
| `grain_i`, `grain_j` | the two grains the sample's boundary separates (i < j) |
| `x/y/z` | sample point coordinates (Å) |
| `area_A2` | the sample's area weight (Å²) |
| `H_invA` | local mean curvature (1/Å) |
| `K_invA2` | local Gaussian curvature (1/Å²) |

**Sign convention.** Curvature is evaluated on the level set of the margin
difference oriented `grain_i → grain_j`: `H > 0` means `grain_i` is locally
convex (e.g. a sphere of grain `i` in a matrix `j` has `H = +1/R`); negate `H`
for `grain_j`'s perspective. `K` is orientation-independent. **Per-grain
histogram recipe:** select rows with `grain_i == g` or `grain_j == g`, and
negate `H` where `g == grain_j` (`curvature_hist.plt`'s `grain` variable does
exactly this). Flat geometries are exactly planar, so every row reports
`H = K = 0`. Degenerate samples (near-zero level-set gradient) are dropped
before writing and tracked by gate **G16**. Per-grain, `Σ K·dA` over a
grain's own retained rows is INTENDED to sit well below the grain's full
closed-surface Gauss-Bonnet budget (face-interior sampling excludes the
edge/vertex angle-defects meant to carry most of that budget for a
polyhedral grain) — but in practice is NOT reliably ≈0, or even reliably
small: gate **G21**'s own calibration found near-flat baselines already
cluster around one topological unit (4π), and ordinary curved runs
routinely reach several multiples of it (see `docs/gates.md` G21 for the
full numbers) — a standing reliability caveat on this file's H/K columns
for that run, not proof the model is wrong.

**`doping.csv`** (`doping.dopants` non-empty) — one row per (grain, dopant)
pair:

| column | meaning |
|---|---|
| `grain_id` | 0-based grain index |
| `element` | dopant element symbol |
| `mode` | `substitutional` or `interstitial` |
| `n_candidate_sites` | candidate host sites (substitutional) or interstitial sites (interstitial) considered in this grain |
| `n_rejected_min_distance` | candidates rejected by the `min_distance` cutoff (interstitial only; always 0 for substitutional) |
| `n_dopant` | dopants actually placed in this grain |
| `n_dopant_shell` | of those, in the GB shell (`margin <= shell_width`) |
| `n_dopant_bulk` | of those, in the bulk (`margin > shell_width`) |
| `fraction_shell` | achieved dopant fraction among this grain's shell candidate sites (`n_dopant_shell / n_candidate_shell`; `NaN` when the shell has no candidates) |
| `fraction_bulk` | achieved dopant fraction among this grain's bulk candidate sites (`n_dopant_bulk / n_candidate_bulk`; `NaN` when the bulk has no candidates) |

For interstitial dopants, `n_candidate_sites` is the PRE-rejection count
(before the `min_distance` cutoff), so the `fraction_shell`/`fraction_bulk`
denominators `n_candidate_shell`/`n_candidate_bulk` refer to the
post-rejection survivors and satisfy `n_candidate_shell + n_candidate_bulk
= n_candidate_sites - n_rejected_min_distance`; for substitutional dopants
no rejection applies (`n_rejected_min_distance = 0`) and the identity is
exact with no subtraction needed.

Gate **G17** (warn) compares the per-dopant totals (summed over grains) to
their nominal concentration and target enrichment; gate **G18** (hard)
re-verifies every interstitial dopant's `min_distance` on the final structure.

**`doping_profile.csv`** (`doping.dopants` non-empty) — dopant–GB distance
profile (proxigram), one row per (dopant, GB-distance bin):

| column | meaning |
|---|---|
| `element`, `mode` | dopant element and placement mode |
| `bin_lo`, `bin_hi`, `bin_center` | bin edges/center in Å of distance to the nearest grain boundary |
| `n_dopant` | placed dopant atoms of this element in the bin |
| `n_candidate` | candidate sites in the bin (interstitial: `min_distance` survivors; substitutional: eligible host atoms) |
| `local_fraction` | `n_dopant / n_candidate` — local dopant fraction on the candidate sublattice at this GB distance (`nan` for empty bins) |
| `nominal_fraction`, `shell_width`, `enrichment` | config references, constant per element (`shell_width = 0`, `enrichment = 1` when segregation is off) |

The bin width is `shell_width / 3` when GB segregation is enabled (≥ 3 bins
resolve the shell step), else `max(margin) / 20`. For enrichment `E`,
`local_fraction` should step from ≈ `E·p_bulk` inside the shell to ≈ `p_bulk`
outside; without segregation it is flat at the nominal fraction. The
companion script **`doping_profile.plt`** (gnuplot bundle) renders the
dopant-count histogram and the `local_fraction(d)` profile against the
nominal line and shell edge — run `gnuplot doping_profile.plt` in the output
directory to produce `doping_profile.png`; set its `element` variable to one
symbol to filter multi-dopant runs. See `docs/physics.md` (§ doping) for why
normalising by candidates, not bin volume, is the physically meaningful
profile.

**`mdf.csv`** — misorientation-angle distribution histogram: `bin_center_deg,
area_weighted_density, number_density, haar_random_reference, target_density` (the
last is filled only when an `mdf_target` is active). `haar_random_reference` is
the deterministic random-pair disorientation-angle reference of the actual
crystal point group — it equals the Mackenzie 1958 law only for the cubic proper
point group, and is named for what the curve is everywhere (renamed from the
legacy `mackenzie_reference`). See `docs/physics.md`.

**`odf_mtex.txt`** — per-grain Bunge Euler angles + volume-fraction weights, with
an MTEX import header — drop straight into MTEX as a discrete ODF. `weight` is
the tessellation-volume fraction — the quantity grainsmith controls (gate G22,
`odf_drift_max`). When the run has an atom block with a nonzero total atom
count, a fifth column, `atom_fraction` (each grain's share of the exported
structure's atoms), is added, and the header explains why it differs from
`weight` — the atomistic discretisation floor gate **G26** measures. With no
atom block, or an all-zero atom count, the file stays the original four
columns, byte for byte.

**`summary.csv`** — long format `section,key,value`: `meta` (title, version,
timestamp, the two full 64-hex digests `config_sha256` and `provenance_sha256`,
the payload schema id, and `source_date_epoch`), box, grains, seed, every
QA-gate result, composition, SDOT iterations, timings, dependency `versions`,
and the `environment` section. The first stop for "what happened".

The `environment` section records what the run's floating-point bytes actually
depended on, beyond the library versions: `blas_name`, `blas_version`,
`blas_detection` and `blas_openblas_config` (NumPy's BLAS is where cross-machine
last-bit differences come from); `os`, `os_release`, `platform`, `machine`,
`processor` and `libc`; `python_implementation` and `python_version`; `numba`
(the version and whether the optional kernel was active, or `absent`); and
`jobs_requested` / `jobs_effective` (`--jobs 0` requests "all cores available to
this process", and the number it expanded to is machine-specific). Unavailable
probes record `unknown` rather than failing the run. The `timings,*_s` rows are
absent when `SOURCE_DATE_EPOCH` is set — see *Reproducibility scope* below.
`meta,timings_status` explicitly records `measured` or
`omitted_source_date_epoch`; `timing_stages` lists the stages that ran and
`timings_log` points to `run.log`, which always records their actual wall
times, including at `verbose: 0`. `timings_scope` states the measurement
boundary: `total_s` ends after the primary outputs and excludes summary
serialization, manifest hashing and shutdown. `curvature_s`, when present,
is a submeasurement of `analysis_s`, not an additional stage to add to it.

The `box` section distinguishes the material region from the exported
simulation cell:

| key | meaning |
|---|---|
| `vacuum_A` | Requested total padding on each nonperiodic axis; half on each side |
| `vacuum_per_axis_A`, `export_shift_A` | Applied x/y/z padding and coordinate shift; zero on periodic axes |
| `export_lengths_A` | Exported x/y/z box spans; for triclinic cells these are the restricted-triclinic diagonal spans, with tilts recorded separately |
| `material_volume_A3` | Tessellation-box volume before vacuum padding |
| `export_volume_A3` | Simulation-cell volume including padding |
| `vacuum_volume_A3`, `vacuum_fraction` | Added padding volume and its fraction of the exported cell volume |

`vacuum_fraction` describes the explicitly added vacuum, not atomic-scale
free volume, porosity, or voids left by boundary overlap removal.

The `analysis` section records `csl`, `gb_character`, `gb_curvature`,
`statistics` and the configured `voxel_grid`; `texture,scheme` is present
for every orientation scheme and `overlap,enabled` is always explicit.
The `sigma3` fraction rows have companion `csl_status` and
`angle_window_status` values: `measured`, `not_evaluated`, or
`no_boundary_area`. Empty/`nan` numerical cells are thus distinguishable
from a measured zero without breaking the numerical column contract.

The `composition` section's `{sp}_final_fraction` rows reflect the FINAL
(post-doping) composition by design — this is the atom-count-weighted mixture
actually written to disk, dopant species included. Gate **G9**, by contrast,
measures the PRE-doping host snapshot (species drift from overlap removal
only): comparing dopant achieved-vs-nominal fractions is gate **G17**'s job,
whose per-dopant rows live in the `doping` section, keyed
`dopant{k}_{element}_*` where `k` is the dopant's index in
`doping.dopants` (`dopant{k}_{element}_element`,
`…_nominal_fraction`, `…_achieved_fraction`, `…_enrichment_target`,
`…_enrichment_achieved`, `…_rejected_min_distance`) — see
`doping.csv` above for the per-grain breakdown. The index is part of the key
because `doping.dopants` does not require distinct elements: a bulk pass and
a GB-enriched pass of the SAME species are a documented use, and keying on
the symbol alone would make the two collide on one summary row.

`nominal_basis` and `final_basis` make that pre/post-doping distinction
explicit. Each species also has `{sp}_n_final` and
`{sp}_final_mass_fraction`; `n_species_final` counts the species present.
The `lammps_masses` section records the actual exported masses, including
`output.lammps.masses` overrides. In the `atoms` section, `mass_total_amu`,
`number_density_material_per_A3`, `number_density_export_per_A3`,
`density_material_g_cm3` and `density_export_g_cm3` use the FINAL atom
counts and those same masses. The material/export densities differ only
in their denominators (before/after vacuum padding). They are box-average
densities of the generated, unrelaxed structure, not measurements of an
equilibrated bulk phase; `crystal,density_g_cm3` remains the ideal-crystal
basis density. Multiphase nominal atom fractions use expected atom counts
(`phase volume * ideal atomic number density`), not volume fractions alone.

The `odf` section (gate **G26**, report-only) records the atomistic
ODF-weighting discretisation floor: `weighting_tv_volume_vs_atoms` (the
total-variation distance between the tessellation-volume weight vector
`odf_mtex.txt` exports/`odf_drift_max` controls and the atom-count weight
vector the exported structure actually carries), `weighting_max_dw` (the
largest single-grain weight discrepancy), and `atoms_per_grain_mean`. All
three are absent entirely when the run has no atom block at all (never a
false `0.0`); on a multiphase run each key is instead suffixed
`_<phase name>` and written once per phase (atom number densities differ
by phase, so a global figure would be meaningless), with an empty cell
for a phase whose own atom counts happen to be absent. See `docs/gates.md`
G26 and `docs/physics.md` §4 for the measured floor and its
`(atoms/grain)^(-1/3)` surface-scaling.

For angle-shaping runs, the `mdf` section states the `objective`
(`area_weighted_disorientation_angle`) and `odf_constraint` (`measure_only`
or `tv_cap`). It also records `odf_kernel_halfwidth_requested_deg`,
`odf_kernel_degree`, `odf_kernel_halfwidth_effective_deg` and
`odf_null_samples`. These describe the configured diagnostic; G22 states
whether its memory-limited kernel calculation actually ran. A nonrepresentable
kernel parameter is marked `not_evaluated`, never reported as a zero width.

**`slice_<axis><pos>.csv`** (when `analysis.section` set) — one row per section
pixel: the two in-plane coordinates, the grain id, and the grain's Bunge Euler
angles — directly comparable to an experimental EBSD map.

## The publication trio

**`statistics.csv`** — long format `section,key,value`. Sections:

- `grain_size`: `n_grains`, equivalent-diameter `d_eq_{mean,std,min,max}_A`,
  log-normal `lognormal_{mu,sigma}_hat`, and `lognormal_ks_{statistic,p}`.
- `sphericity`: Wadell sphericity `{estimator, mean, std, min, max}`.
- `topology`: faces-per-grain stats.
- `boundaries`: `n_boundaries`, total GB area, surface density `S_V_per_A`.
- `triple_junctions`: total TJ length and line density `L_V_per_A2`.
- `gb_character`: tilt/twist/mixed/undefined area fractions.
- `csl` (if `analysis.csl`): `area_fraction_sigma3`, `area_fraction_non_csl`.
- `mdf`: `mean_misorientation_deg`, `lagb_area_fraction`.
- `roughness` (`perturbed_distance` + `spectrum: self_affine` only): `hurst_target`, `hurst_estimated`.
- per-phase `grain_size_<name>` / `sphericity_<name>` sections (multiphase), and
  `S_V` split into same-phase vs interphase.

**`microstructure.json`** — schema `grainsmith/microstructure/v1`: a strict-JSON
record of the whole run (config echo, provenance digests, execution environment,
gate results, per-grain and per-boundary records, statistics). NaN → `null`.
This is the Zenodo/dataset-repository artifact.

It carries **no wall-clock stage timings**, but it *does* carry the run's UTC
timestamp (`provenance.timestamp_utc`) and the `environment` section, so two
re-runs of the same config are byte-identical only when the clock is frozen with
`SOURCE_DATE_EPOCH` (see *Reproducibility scope* below) and the environment is
unchanged. Its `provenance` block holds the full 64-hex `config_sha256` and
`provenance_sha256` — the truncated 12-hex forms in the file headers are display
forms of the same two digests.

**`METHODS.md`** — an auto-generated methods paragraph for THIS run: the exact
algorithm chain with parameter values and numbered literature citations — a
copy-paste starting point for the paper's Methods section.

## Estimator-bias notes (honest, stated per section)

The `estimator` key in each `statistics.csv` section records *how* a quantity was
measured, because **voxel-based estimators carry geometric bias that grainsmith
reports rather than silently corrects**:

- **Sphericity / S_V / L_V — flat vs voxel.** Flat geometry uses *exact polyhedral*
  areas, volumes and triple-line lengths. Curved/voxel geometry estimates them
  from the voxel grid, where an oblique boundary face is over-counted by up to a
  factor **√3** (axis-aligned voxel faces approximating a tilted plane) and triple
  lines by the voxel-edge staircase. The bias shrinks with finer
  `analysis.voxel_grid`. The `estimator` field says which applies.
- **Log-normal KS p-value is optimistic.** `lognormal_ks_p` is computed with the
  *fitted* μ̂, σ̂, so it is anti-conservative (the Lilliefors caveat; Lilliefors
  1967). Treat it as a descriptive goodness indicator, not a hypothesis test at
  face value.
- **CSL Σ assignment is cubic-only**; blank for non-cubic point groups.
- **G13 Hurst** carries a constant finite-band/finite-size offset (measured:
  ~0.1, low-biased) — `hurst_estimated` vs `hurst_target` reflects this, which
  is why G13 is a warn gate.

## Diagnostics (`meta.verbose >= 3`)

At verbosity 3 the pipeline dumps extra per-run diagnostics into
`<outdir>/diagnostics/` (plain-text, `#`-commented provenance header):

| file | written when | columns/content |
|---|---|---|
| `grain_atom_counts.dat` | always at verbose ≥ 3 | `grain_id n_atoms` — final atom count per grain |
| `warp_field_stats.dat` | `boundaries.curved.method: warp` only | `max_grad_u`, `rms_u_x/y/z`, `max_norm_u` — warp-field gradient/amplitude summary (the same quantities gate **G6** checks) |
| `voxel_histogram.dat` | any curved (non-flat) geometry | `grain_id n_voxels` — voxel count per grain on the analysis grid |

## Provenance

- **`resolved_config.yaml`** — the fully-resolved config (all defaults filled,
  all `auto` values computed) actually used by the run. Re-running it reproduces
  the result. Everything below its two `#` header lines is exactly the byte
  sequence hashed into `config_sha256`, which is what lets `grainsmith verify`
  re-derive that digest from the shipped file.
- **`run.log`** — the INFO/DEBUG log (verbosity from `meta.verbose`); ASCII-safe
  for Windows cp1254 consoles. It is the one output whose content is wall-clock
  by design, so `MANIFEST.txt` lists it *without* a digest.
- **`MANIFEST.txt`** — SHA-256 and byte size of every output file this run wrote,
  for integrity/FAIR. Its comment header repeats the run's provenance line and
  then the two full-length (64-hex) digests and the payload schema id:

  ```
  # grainsmith 1.1.0 | 2026-09-01T12:00:00Z | seed=555666 | config sha256=fd1c4ecb04ac | provenance sha256=9a3f0c1d2e4b
  # config_sha256 = <64 hex>
  # provenance_sha256 = <64 hex>
  # provenance_payload_schema = grainsmith-provenance/v1
  # sha256  size_bytes  path (relative to this file)
  ```

  Stale files left in a reused output directory are deliberately **not** listed,
  so a leftover artefact is never attributed to this run.

### The two digests

`config_sha256` is the SHA-256 of the canonical resolved config alone — it
identifies the *input*. `provenance_sha256` is the SHA-256 of a small payload
that also contains the grainsmith version string, so it identifies *input +
producer*: the version an output claims cannot be altered without invalidating
the digest. Both are recorded full-length in `MANIFEST.txt`, `summary.csv`
(`meta,config_sha256` / `meta,provenance_sha256`) and `microstructure.json`;
provenance headers show the first 12 hex of each.

### Verifying a directory

```bash
grainsmith verify ./out          # 0 = OK, 1 = mismatch, 2 = cannot verify
grainsmith verify ./out --strict --expect-version 1.1.0
```

See the [user manual](manual.md) for the full check list.

### Reproducibility scope

A fixed `seed.value` reproduces the scientific payload byte-for-byte for any
`--jobs` value, on the same platform and dependency versions. Two caveats that
the table in the [user manual](manual.md) spells out in full:

1. **Timestamps.** Every file that permits comments carries the run's UTC
   timestamp in its provenance header, so two default runs are not byte-identical
   as whole files, and `MANIFEST.txt` — which hashes them — differs in every
   digest. Set `SOURCE_DATE_EPOCH` to freeze the clock (and drop the wall-clock
   `timings,*_s` rows from `summary.csv`); the whole directory except `run.log`
   is then byte-identical between runs.
2. **Platform.** A different BLAS build or CPU architecture can change the last
   floating-point bit of a fraction of atom coordinates, and a different
   NumPy/SciPy/spglib *version* can change the structure outright. The
   `environment` section of `summary.csv` records what actually ran;
   `constraints-repro.txt` pins the versions to reproduce it.
