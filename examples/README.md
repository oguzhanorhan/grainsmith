# grainsmith examples — a guided tour

Every file here runs end-to-end with all QA gates green, with two stated
exceptions — `pdau_perturbed_50nm_H05.yaml` and its 100 nm-box sibling
`pdau_perturbed_54nm_H05_100nm_box.yaml` are validate-only by design
(5.3×10⁷ and 6.7×10⁷ atoms; see their entries below):

```bash
grainsmith validate examples/basics/b2_nial_flat.yaml    # schema (G1) + crystal (G2) only
grainsmith generate examples/basics/b2_nial_flat.yaml    # full run → ./out_b2_flat/
```

To run a whole group at once and have the outputs checked for you, use the
runner that lives here:

```bash
python examples/run_examples.py                 # numbered family menu
python examples/run_examples.py basics          # one family, --jobs 10
python examples/run_examples.py texture -j 6    # six worker processes
python examples/run_examples.py --list          # families, sizes, cost
```

It prints the grainsmith version it is exercising, runs every example in
the family you pick (each into its own `out_*` directory under
`./example_runs/<family>/`), and then reports one line per example:
measured atom count from that run's `summary.csv`, its deviation from the
`~atoms` column below, the PASS/WARN/FAIL gate tally, and how many of the
output files the YAML declares actually appeared. Exit status is non-zero
if anything failed, so it also works as a smoke test. Files declared at
≥ 10⁷ atoms are validated (G1/G2) rather than generated unless you pass
`--force`; families with heavy files ask before they start (`--yes` skips
the prompt). `--jobs` never changes the result — outputs are bit-identical
for every worker count.

The same pattern applies to every file below: `examples/<group>/<name>.yaml`.
Each YAML carries its own `WHAT THIS TEACHES` header and a `HOW TO RUN` block
with the exact commands. The sections below follow a recommended learning
order — basics, then crystallography, CIF input, texture, grain geometry,
multiphase alloys, doping, self-affine boundaries, voxel import, the
`advanced` tier (~10⁶ atoms, plus a 1500 Å `huge_*` sub-tier at ~10⁸),
and finally the cluster-scale `hpc` group with its SLURM recipes. The
rightmost table column is the rough atom count on a laptop. Everything
finishes in seconds except the `advanced` tier, the `hpc` group, the three
large self-affine files and `fcc_cu_curved_warp.yaml` (9.2×10⁶). Atom
count is not the same as disk: `bicrystal_fixed_orient.yaml` runs in ~3 s
at `--jobs 10` and still writes ~140 MB.

## What each output file contains

Every run writes into its `output.directory`. The always-written files:

| file | contents |
|---|---|
| `polycrystal.data` | LAMMPS data file — atom positions, types, molecule id = grain id |
| `polycrystal.extxyz` | extended XYZ — per-atom `grain` column (+ `gb_margin` when `analysis.per_atom_margin` is on) |
| `summary.csv` | gate results, composition, timings, seed (long `section,key,value` format) |
| `grains.csv` | per-grain volume, atom count, orientation |
| `boundaries.csv` | per-GB-pair misorientation, CSL Σ, area |
| `statistics.csv` | grain-size distribution + log-normal fit, sphericity, S_V, L_V, GB character |
| `microstructure.json` | machine-readable record of the whole run |
| `METHODS.md` | auto-generated, citation-ready methods paragraph for this exact run |
| `view.plt` | gnuplot preview bundle — `gnuplot view.plt` renders `view.png` |
| `run.log` | the run's log |
| `resolved_config.yaml` | the fully-resolved config actually used (defaults filled, `auto` values computed) |
| `MANIFEST.txt` | SHA-256 of every output file |

Written only where the corresponding config block is enabled:

| file | written when |
|---|---|
| `vertices.csv` | flat and power/Laguerre (SDOT) geometry only — Voronoi vertices |
| `doping.csv` | `doping.dopants` is non-empty — per-grain dopant placement and shell/bulk segregation split |
| `doping_profile.csv` | `doping.dopants` is non-empty — dopant–GB distance profile |
| `gb_curvature.csv` | `analysis.gb_curvature` — per-boundary-sample local mean/Gaussian curvature (H/K) |
| `mdf.csv` / `odf_mtex.txt` | always for a single-phase run — misorientation and orientation distributions (per-phase as `mdf_<name>.csv` / `odf_mtex_<name>.txt` for multiphase) |
| `slice_<axis><pos>.csv` | `analysis.section` set — an EBSD-like planar section |
| `curvature_hist.plt` | `analysis.gb_curvature` + gnuplot enabled — curvature histogram gnuplot bundle |
| `doping_profile.plt` | `doping.dopants` non-empty + gnuplot enabled — dopant-profile gnuplot bundle |
| `boundaries.ply` | `output.mesh.enabled` — a marching-cubes mesh of the boundary network (curved geometry only) |

See `docs/outputs.md` for the complete inventory, every column's meaning,
and the estimator-bias notes for voxel-based measurements.

## basics

Start here: the plainest constructions grainsmith produces.

| file | teaches | ~atoms |
|---|---|---|
| `b2_nial_flat.yaml` | the baseline: ordered B2 compound, flat Voronoi GBs, legacy-parity scale (NP=10, 150³ Å), `analysis.gb_curvature` (flat ⇒ H = K = 0 exactly) | 2.7×10⁵ |
| `cu_single_crystal.yaml` | `grains.number: 1` single crystal — the perfect-crystal baseline (exactly 4·10³ atoms here), gate G14 box/lattice commensurability, what happens when you rotate the lattice (box-face self-boundary) | 4×10³ |
| `bicrystal_fixed_orient.yaml` | `from_list` per-grain orientations (quaternion + axis-angle), CSL Σ5 assignment, why a fully periodic bicrystal reports an `undefined` mean plane. **the heaviest file in this group** — 240³ Å with 2 grains, ~140 MB of output; shrink `box.lengths` if you only want the Σ5 lesson (it is box-independent) | 1.2×10⁶ |
| `hcp_ti_thin_film.yaml` | hexagonal family, slab geometry with vacuum, single 2c Wyckoff site (= ideal HCP) | 2.7×10⁴ |

## crystallography

Space-group mechanics: settings, free Wyckoff parameters, non-cubic axes.

| file | teaches | ~atoms |
|---|---|---|
| `rutile_tio2_tetragonal.yaml` | tetragonal family (a, c), FREE Wyckoff parameter (O 4f, x=0.305), multi-sublattice stoichiometry, cubic-only CSL guard | 1.1×10⁴ |
| `si_diamond_origin2.yaml` | space-group `setting` (origin choice 2), why spglib Wyckoff letters can differ from ITA in non-default settings, `verbose: 3` diagnostics | 6×10³ |
| `bi_rhombohedral_R.yaml` | the rhombohedral (R) axis setting: free (a, α) instead of (a, c), setting-dependent coordinates and orbit sizes | 6×10³ |

## cif

Reading the crystal structure directly from a CIF file (ASE + spglib)
instead of hand-entering `space_group` / `lattice` / `wyckoff_sites`. All
five use the shipped Materials Project CIFs in `examples/assets/`.

| file | teaches | ~atoms |
|---|---|---|
| `tini_cif_polycrystal.yaml` | `crystal.cif` — TiNi (B19, monoclinic P2_1/m) read DIRECTLY from `TiNi.cif` (ASE + spglib), no manual space_group/lattice/wyckoff_sites; declared and spglib-detected SG agree (both 11) | 1.5×10⁴ |
| `ti2ni_cif_single_crystal.yaml` | `crystal.cif` on the single-crystal backend — Ti2Ni (Fd-3m, 96 atoms/cell, 3 Wyckoff orbits) read from `Ti2Ni.cif`, exactly 3×3×3 conventional cells (gate G14 near-zero commensurability misfit) | 2.6×10³ |
| `tini3_cif_polycrystal.yaml` | `crystal.cif` declared-vs-detected mismatch — `TiNi3.cif` declares `P 1` (SG 1) but spglib detects the true P6_3/mmc (SG 194); grainsmith uses the detected group and only WARNs (see run.log) | 1.3×10⁴ |
| `tini_cif_triclinic_single.yaml` | `box.cells` — TiNi (monoclinic, β=105.2296°) as a TRICLINIC single crystal, box vectors built as 6×5×4 exact multiples of the crystal's own cell (restricted-triclinic LAMMPS tilt, xz=−5.07 Å); gate G14 misfit EXACTLY 0 | 4.8×10² |
| `tini3_cif_hexagonal_single.yaml` | `box.cells` on a hexagonal cell (γ=120°) — 5×5×3 exact multiples produce an XY tilt factor instead (xy=−12.63 Å, landing exactly on the LAMMPS `\|xy\|<=ax/2` bound for n1=n2 — the `reduce_triclinic_tilts` boundary case); gate G14 misfit EXACTLY 0 | 1.2×10³ |

## texture

| file | teaches | ~atoms |
|---|---|---|
| `fcc_cu_fiber_film.yaml` | `fiber` texture (⟨111⟩ ∥ z + spread), slab + vacuum, `per_atom_margin` for OVITO GB coloring | 8×10³ |
| `cu_twin_odf_mdf.yaml` | `odf_components` texture (Σ3 twin-pair components + spread), `component_weight_basis: volume` (weight = physical VOLUME fraction, gate G25), `mdf_target: sigma3_angle_enriched` assignment annealing (gate G12), Σ3 audit via `boundaries.csv` + `mdf.csv` / `odf_mtex.txt` outputs | 1.6×10⁴ |

The three `*_curved.yaml` files below are the **reference texture examples**:
360 Å boxes, 10 grains (~20.7 nm), self-affine curved boundaries, and a
`runtime.memory_limit_gb` guard. Each isolates one question a reader should
ask of a texture generator, and answers it with numbers in `summary.csv`
rather than prose. All three run in 1.5–3 min on a laptop and pass every gate.

| file | teaches | ~atoms |
|---|---|---|
| `cu_sigma3_angle_enriched_curved.yaml` | **angle ≠ CSL, and ODF drift is bounded.** `sigma3_angle_enriched` target on Σ3-twin-related components; G23 prints the Brandon-window area fraction the objective optimises (0.477) next to the TRUE Σ3 fraction with the ⟨111⟩ axis applied (0.175). `odf_drift_max: 0.03` hard-caps the volume-weighted ODF drift; measured 0.026 | 3.8×10⁶ |
| `mg_hcp_basal_fibre_curved.yaml` | **the reference curve is point-group specific.** Mg P6₃/mmc has 12 proper rotations, so its `haar_random_reference` in `mdf.csv` extends to ~94°, against ~63° for cubic — the same file next to the Cu one shows why "the Mackenzie distribution" is the wrong name for the general quantity. Basal fibre ([0001] ∥ ND), the classic rolled-Mg texture | 1.9×10⁶ |
| `fe_bcc_rolling_texture_curved.yaml` | **the sampler term, isolated.** Classic BCC cold-rolling components ({001}⟨110⟩, {112}⟨110⟩, {111}⟨110⟩) and NO `mdf_target`, so the permutation is the identity and the annealer contributes exactly zero. At 10 grains the count basis is quantised to 0.30/0.30/0.40 while the volume basis reaches 0.280/0.302/0.418 against a configured 0.25/0.30/0.45 | 3.9×10⁶ |

## grain_geometry

Grain-shape and boundary-curvature controls.

| file | teaches | ~atoms |
|---|---|---|
| `al_equiaxed_lloyd_weights.yaml` | Lloyd centroidal relaxation (equiaxed grains), `additive_weights` curved GBs, explicit `voxel_grid`, PLY mesh option | 1.2×10⁴ |
| `cu_lognormal_sizes.yaml` | log-normal grain-size targeting via SDOT-fitted Laguerre weights (gate G11, < 0.1 % volume error); statistics.csv re-fits the ACHIEVED diameters (σ̂ vs the prescribed sigma_log) and `analysis.section` writes an EBSD-like midplane slice | 1.6×10⁴ |
| `cu_equal_sizes.yaml` | `size_distribution.type: equal` — every grain SDOT-fitted to the same V_box/N (all volume fractions = 1/12); shows the `type:`-INSIDE-`size_distribution:` nesting (writing it under `grains:` is the G1 "Extra inputs" slip); the σ̂≈0 statistics re-fit is degenerate by design | 1.6×10⁴ |
| `cu_volume_list_sizes.yaml` | `size_distribution.type: volumes` — an explicit RELATIVE per-grain volume list (Rule 11: length == `grains.number`, normalized to V_box) building a bimodal 3:1 structure (fractions 0.1875 / 0.0625); completes the lognormal / equal / volumes trio | 1.7×10⁴ |
| `cu_curved_sizes.yaml` | the ONE legal curved + prescribed-volumes combination: `method: warp` over `base: power` with a `size_distribution` (Rules 10/12 — every other curved method refuses volume targeting, and a `curved:` block under `geometry: flat` is a G1 error); G11 measured on the UNWARPED power base, G6 bijectivity guard | 1.6×10⁴ |
| `fe_bcc_anisotropic.yaml` | `anisotropic` (ellipsoidal-metric) elongated grains, `midpoint_merge` overlap policy, BCC via SG 229 | 3.5×10⁵ |
| `fcc_cu_curved_warp.yaml` | production scale: domain-warp curved GBs at 480³ Å, the G6 clip/gradient interplay, `--jobs 0` parallel fill, `analysis.gb_curvature` → `gb_curvature.csv` (genuinely curved H/K, gate G16) | 9.2×10⁶ |

## alloys_multiphase

| file | teaches | ~atoms |
|---|---|---|
| `cuni_solid_solution.yaml` | stochastic site occupancy (`{Cu: 0.9, Ni: 0.1}`), gate G9 composition drift, why `midpoint_merge` is forbidden for alloys | 1×10⁴ |
| `ti_alpha_beta.yaml` | `phases:` multiphase — HCP α + BCC β Ti with 60/40 VOLUME fractions, gate G15, interphase boundary rows (NaN misorientation, habit-plane indices kept), per-phase mdf_/odf_mtex_ files | 1.1×10⁴ |
| `cufe_composite.yaml` | phases with different ELEMENTS — FCC Cu + BCC Fe composite, two LAMMPS atom types, volume-weighted G9 composition mix, why `midpoint_merge` stays forbidden | 1.3×10⁴ |

## doping

Interstitial and substitutional dopants with GB segregation.

| file | teaches | ~atoms |
|---|---|---|
| `fe_c_gb_interstitial.yaml` | `doping:` interstitial mode — 2 at% C on the `bcc_octahedral` preset sublattice, GB segregation (E=5) via the shell/bulk Bernoulli model, gates G17 (composition + enrichment, warn) and G18 (min-distance re-verification, hard), the `doping.csv` report | 1.7×10⁴ |
| `al_mg_gb_substitutional.yaml` | `doping:` substitutional mode — 2 at% Mg replacing Al in place (`host: Al`), GB segregation (E=5), why substitutional dopants trivially pass G18 (no new atoms, no min_distance check) | 1.2×10⁴ |
| `tio2_cif_li_interstitial.yaml` | `crystal.cif` on a LOW-symmetry CIF (anatase TiO2 declares `P 1`, spglib detects the true I4₁/amd, SG 141) + Li octahedral-interstitial doping via symmetry orbit expansion (1 listed site → 4 sites/cell) with GB segregation (E=4), gates G17/G18 | 1.8×10⁴ |

## self_affine_gb

`boundaries.curved.method: perturbed_distance` — the self-affine
alternative to `warp` (a coordinate diffeomorphism cannot change the
box-counting dimension of the surface it displaces, so `warp` no longer
accepts `spectrum: self_affine` at all — see `docs/physics.md` §5b).

| file | teaches | ~atoms |
|---|---|---|
| `pdau_perturbed_self_affine.yaml` | the ONLY method that accepts `spectrum: self_affine`; a genuinely self-affine grain boundary (level-set assignment), Pd90Au10 target from Braun 2018/2020, gates G13 (Hurst back-estimate, warn) + G19 (repair severity, warn) + G20 (box-counting D_b, warn) | 5.1×10⁵ |
| `pdau_perturbed_50nm_H05.yaml` | `amplitude_convention: reference_wavelength` at a ~50 nm grain diameter, sized to reach the ~5.5-octave regime gate G20's `d_b_estimated` needs to resolve `hurst` at a conventionally LARGE effect size (η²>80%) — every smaller example's `d_b_estimated` is an honest roughness index, not a converged fractal dimension. **NOT run by the test suite or CI** (schema/resolve validation only, gate G1) — a ~5.3×10⁷-atom, multi-GB config; run it manually, only on hardware you have sized for it | 5.3×10⁷ (not generated) |
| `pdau_perturbed_54nm_H05_100nm_box.yaml` | the same construction in a round 100 nm (1000 Å) box: the box edge, not the grain diameter, is the fixed quantity, so 10 grains land at ~54 nm and the band widens to 5.61 octaves. Use this one when the box size must be a round number (e.g. to match an experiment's field of view); use the 50 nm file when the grain diameter must be. **Also validate-only** — ~6.7×10⁷ atoms, and no test touches it | 6.7×10⁷ (not generated) |
| `tio_2_thin_film_self_affine.yaml` | `perturbed_distance` + `spectrum: self_affine` on a CIF-sourced slab/thin-film box (periodic xy, free z + vacuum) — self-affine GBs under a non-fully-periodic box, gates G13/G19/G20 | 8.8×10⁵ |

## voxel_import

| file | teaches | ~atoms |
|---|---|---|
| `fe_voxel_import.yaml` | `voxel_import` — atomize an external voxel label field (shipped 32³ .npy; DREAM.3D HDF5 via `[import]` extra), derived grain count, G5 WARN demotion for imports (the label-field path resolves relative to this file, so it runs from any working directory) | 9×10³ |

## advanced (~10⁶ atoms, plus a ~10⁸ `huge_*` tier)

See [`advanced/README.md`](advanced/README.md) for the tier-by-tier guide,
including why scaling a box is not a multiplication.

The `adv_*.yaml` files scale the same constructions up to ~10⁶ atoms — the
regime the companion LAMMPS benchmark simulations use to show how close the
generated models sit to reality, at a scale where the microstructure
statistics are representative. Each is literature-grounded, with its
sources named in its own file header. The `~Atoms` and `Gates`
columns below are the ACTUAL results of a run on a 32 GB / 24-core laptop
(fixed seed), not estimates.

Two of these use a weighted tessellation (`additive_weights`, `anisotropic`)
whose per-grain fill boxes are large; run **those two with a capped worker
count (`--jobs 6`)** or the fill can be OOM-killed / very slow at 10⁶ atoms.
The header of each such file carries the measured RESOURCE NOTE. The other
four run fine at `--jobs 0` (all cores).

| file | teaches | ~atoms | gates | `--jobs` |
|---|---|---|---|---|
| `adv_pdau_perturbed.yaml` | **flagship**: experimentally-motivated **Pd₉₀Au₁₀** microstructure — `method: perturbed_distance` + `spectrum: self_affine` at H=0.7, a level-set self-affine grain boundary honestly bounded against Braun's D_b≈1.2 target (measured this run: d_b_estimated=1.118, hurst_estimated=0.704 vs target 0.7, reassigned_fraction=0.082% — calibration note in the file header); `element:` site alloy (Au=type1, Pd=type2), gates G13 (Hurst back-estimate) + G19 (repair severity) + G20 (box-counting D_b) | 9.5×10⁵ | 13/13 | 0 |
| `adv_cu_lognormal_curved.yaml` | log-normal grain volumes (SDOT/Laguerre, gate G11) + `centroidal_iterations` (equiaxed WITH target volumes) + a Gaussian curved warp on the `power` base — three literature features composed | 9.7×10⁵ | 11/11 | 0 |
| `adv_cu_rolling_twin.yaml` | rolled-and-annealed FCC Cu: β-fibre rolling ODF (Copper/S/Brass/Cube — Euler centres computed by grainsmith's own `hkl_uvw→bunge`, Brass reproduces the repo-pinned value), `component_weight_basis: volume` (gate G25) + Σ3 annealing-twin angle target (`sigma3_angle_enriched`, gate G12) | 9.5×10⁵ | 11/11 | 0 |
| `adv_cufe_composite.yaml` | two-phase FCC-Cu(60%)/BCC-Fe(40%) composite at scale: greedy LPT phase partition (gate G15), interphase boundaries with NaN misorientation, no imposed OR → stability-only MD | 9.6×10⁵ | 11/11 | 0 |
| `adv_al_equiaxed.yaml` | fully equiaxed FCC Al: Lloyd centroidal relaxation + `additive_weights` (Johnson-Mehl/Apollonius) curved GBs — a clean random-texture FCC GB-energy benchmark case | 9.5×10⁵ | 10/10 | 6 |
| `adv_fe_anisotropic.yaml` | BCC α-Fe with `anisotropic` (ellipsoidal-metric) elongated grains (aspect ≤ 2:1, rolled/columnar) + `midpoint_merge` (legal — elemental); completes the curved-method trio warp / additive_weights / anisotropic | 9.6×10⁵ | 10/10 | 6 |

### huge tier (~10⁸ atoms, 1500 Å box — cluster jobs)

Box-expanded siblings of the three `adv_*` cases whose statistics are most
obviously sample-size limited. **Validated but not run here**: at ~10⁸ atoms a
single run is a cluster job, so the YAML is checked for syntax and for every
constraint the config layer can evaluate, and the resource figures in each
header are computed from the code rather than measured. All three use 256
grains (`d_eq` = 29.3 nm) — at this box edge the grain count is a memory
parameter as much as a physics one, because the per-grain fill box scales as
`box³ / n_grains`.

| file | expands | what the scale buys | ~atoms | sort peak |
|---|---|---|---|---|
| `huge_pdau_perturbed.yaml` | `adv_pdau_perturbed` | self-affine roughness over 4 octaves at 256 grains instead of 28; `l_min` raised to 9.0 Å because the 384-voxel grid cap puts the Nyquist floor at 7.81 Å | 2.3×10⁸ | ~18 GB |
| `huge_cu_rolling_twin.yaml` | `adv_cu_rolling_twin` | a four-component β-fibre ODF and an angle histogram built from ~1700 boundaries; carries `odf_drift_max` (G22) and the angle-window-vs-true-CSL comparison (G23) | 2.9×10⁸ | ~23 GB |
| `huge_cufe_composite.yaml` | `adv_cufe_composite` | a 60/40 phase split whose granularity is set by grain volumes — 256 grains land far closer than 48 (G15) | 2.9×10⁸ | ~23 GB |

*(Gates: `additive_weights` and `anisotropic` report 10/10 rather than 11/11
because gate G6 — warp-field bijectivity — does not apply to a non-warp curved
method. The flagship's `perturbed_distance` also has no G6 to apply, but
reports MORE gates (13/13) than the warp baseline because its self_affine
spectrum triggers three extra warn-only gates: G13 (Hurst back-estimate),
G19 (repair severity) and G20 (box-counting D_b) — see `docs/gates.md`. All
other gates pass.)*

## hpc — cluster-scale runs (SLURM)

The `hpc/` trio scales the three boundary-geometry classes — flat,
curved/anisotropic and self-affine fractal — to ~4×10⁶ atoms each: sized
for a single cluster node rather than a laptop. **`hpc/README.md` is the
full cluster guide** — one-time setup + numba warm-up, an `sbatch`
walkthrough (with `hpc/job.sbatch`, a ready-to-submit template), the
`salloc` interactive case, job arrays, and a pitfalls table;
`1_cu_flat.yaml`'s header carries the short version of the same recipe.
The `~atoms`, `gates` and wall-time columns below are the
ACTUAL results of a `--jobs 10` run on a 32 GB / 24-core laptop (fixed
seed), not estimates.

The parallel model these files teach: grainsmith is a **single-node**
program — one run = one node = one task (`--nodes=1 --ntasks=1`), and all
parallelism is `--jobs` worker processes inside that task. The
best-parallelism operating point is **`--jobs = grains.number`** (here 10)
with `--cpus-per-task` set to match: the fill stage hands each grain its
own worker, so all grains fill simultaneously. `--jobs 0` resolves to the
cores *available to the process* (the cgroup/affinity mask — i.e. your
SLURM allocation, not the whole node). The numba kernels are disk-cached
in the installed package, so one small warm-up run after installation
compiles them once for every node sharing the filesystem. For more nodes,
run independent configs as separate jobs (a job array), each with its own
`output.directory`.

| file | teaches | ~atoms | gates | wall (`--jobs 10`) |
|---|---|---|---|---|
| `1_cu_flat.yaml` | the HPC baseline: flat Voronoi GBs at cluster scale, the single-node SLURM recipe (`sbatch` + `salloc`), jobs == grains == cpus-per-task | 3.9×10⁶ | 10/10 | 11 s |
| `2_cu_curved.yaml` | `anisotropic` elongated grains (≤ 2.5:1) + `midpoint_merge` at cluster scale; measured footprint stays ~1 GB at N = 10 (the advanced-tier `--jobs 6` cap is a high-grain-count concern, not a scale one) | 3.9×10⁶ | 10/10 | 34 s |
| `3_cu_fractal.yaml` | `perturbed_distance` + `spectrum: self_affine` (H = 0.8) at cluster scale — the elemental sibling of the flagship with its three extra warn gates (G13/G19/G20), amplitude recalibrated for this geometry (A_max ≈ 8.6 Å here) | 3.9×10⁶ | 13/13 | 47 s |

## Tips that apply everywhere

- **Reproducibility**: a fixed `seed.value` reproduces every output byte
  for byte (timestamps aside), for ANY `--jobs` value. `seed.mode: entropy`
  draws a fresh seed and records it in `summary.csv`.
- **Reading the results**: start with `summary.csv` (gates, composition,
  timings, versions), then `grains.csv` / `boundaries.csv`. `view.plt`
  renders headless: `gnuplot view.plt` → `view.png`.
- **OVITO**: load the output file into OVITO, then Add modification →
  Color coding. For the extended XYZ file, set Input property = `grain`;
  for the LAMMPS data file, set Input property = `Molecule Identifier`.
  Either way, that colors atoms by grain.
- **When a gate trips**: the error message names the gate, the measured
  value, and the fix (e.g. G6 → increase `correlation_length`; G1 σ_w
  guard → reduce `weight_sigma`). That is by design — gates are the
  user interface of the physics constraints.
