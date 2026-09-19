# grainsmith

**A Generator of Polycrystalline Models for Atomistic Simulations with Statistical and Grain-Boundary Morphology Control**

`grainsmith` builds periodic (or slab) polycrystalline atomistic models for
LAMMPS and other MD codes. It supports all 230 space groups via spglib,
flat and curved (including self-affine) grain boundaries, multiphase
and doped microstructures, reproducible RNG, and structured CSV/JSON
reports covering GB character, CSL Σ assignment, texture, and publication
statistics.

## Quickstart

```bash
pip install -e .
grainsmith validate examples/basics/b2_nial_flat.yaml   # gates G1 + G2 only
grainsmith generate examples/basics/b2_nial_flat.yaml   # full run → ./out_b2_flat
grainsmith generate big_config.yaml --jobs 0            # parallel run, all cores
grainsmith verify ./out_b2_flat                         # re-check output integrity
```

`--jobs N` fans out worker processes over the per-grain fill stage, the
GB-curvature analysis stage, and the per-boundary-pair analysis stage, and
sets the thread count for overlap-removal and the G7 neighbour search
(`0` = all cores, `1` = serial, the default). It is an execution detail
only: every grain draws from its own deterministic rng stream, so outputs
are **byte-identical for every jobs value**.

## Outputs

A `generate` run writes into `output.directory` (default `./out`). Filenames
in the *configurable* column can be renamed in the config; the rest are
fixed. Every file carries a one-line provenance header
(`grainsmith <version> | <UTC timestamp> | seed=<int> | config sha256=<12 hex> | provenance sha256=<12 hex>`)
where the format permits comments. `config sha256` is the digest of the resolved
config; `provenance sha256` additionally covers the grainsmith version string,
so an output's version claim is bound to its digest. Both are recorded
full-length in `MANIFEST.txt`, `summary.csv` and `microstructure.json`, and
`grainsmith verify <outdir>` re-checks all of it.

| file | contents | written when |
|---|---|---|
| `polycrystal.data` | LAMMPS data file (`atom_style: atomic` or `molecular`, molecule-id = grain id) | always (`output.lammps.filename`) |
| `polycrystal.extxyz` | extended XYZ: species, position, grain id (+ GB margin if enabled) | on by default (`output.xyz.enabled`, `output.xyz.filename`) |
| `grains.csv` | one row per grain: seed, volume, atom count, orientation (quaternion/Euler/axis-angle), nearest crystal plane/direction | always (`output.csv.grains`) |
| `boundaries.csv` | one row per grain–grain interface: area, misorientation, boundary plane, character, CSL Σ (+ curvature columns if enabled) | always (`output.csv.boundaries`) |
| `vertices.csv` | Voronoi vertex coordinates and incident grains | flat and power/Laguerre (SDOT) geometry only (`output.csv.vertices`) |
| `summary.csv` | long-format run record: config echo, every QA gate result (G1–G26), composition, stage timings, dependency versions | always (`output.csv.summary`) |
| `mdf.csv` | disorientation-angle histogram vs. the Haar-random reference of the crystal's own point group — equal to the Mackenzie 1958 law only for cubic (+ target, if `mdf_target` is set) | always, single-phase |
| `odf_mtex.txt` | per-grain Bunge Euler angles + volume weights, MTEX-ready | always, single-phase |
| `mdf_<phase>.csv`, `odf_mtex_<phase>.txt` | per-phase versions of the two files above | multiphase runs, one pair per phase |
| `statistics.csv` | grain size (+ log-normal fit), sphericity, GB area density, triple-junction density, GB character/CSL fractions, MDF scalars, roughness — all re-measured | on by default (`analysis.statistics`) |
| `microstructure.json` | strict-JSON machine-readable record of the whole run (config, gates, per-grain/per-boundary data, statistics) | on by default (`analysis.statistics`) |
| `METHODS.md` | auto-generated methods paragraph for this exact run, with author-year citations | on by default (`output.methods_snippet`) |
| `slice_<axis><pos>.csv` | EBSD-like 2D section: pixel grid, grain id, Bunge Euler angles | `analysis.section` set |
| `gb_curvature.csv` | per-sample local mean/Gaussian curvature (H, K) of grain boundaries | `analysis.gb_curvature` |
| `doping.csv` | per-(grain, dopant) placement statistics (candidates, rejections, shell/bulk split) | `doping.dopants` non-empty |
| `doping_profile.csv` | dopant concentration vs. distance-to-GB profile (proxigram) | `doping.dopants` non-empty |
| `seeds.dat`, `edges.dat`/`gb_points.dat`, `box.dat`, `view.plt` | gnuplot visualization bundle (headless `pngcairo` render script + data) | on by default (`output.gnuplot.enabled`); `edges.dat` for flat geometry, `gb_points.dat` for curved |
| `curvature_hist.plt` | gnuplot script: per-grain local-curvature histogram from `gb_curvature.csv` | `analysis.gb_curvature` + gnuplot enabled |
| `doping_profile.plt` | gnuplot script: dopant histogram + proxigram from `doping_profile.csv` | doping non-empty + gnuplot enabled |
| `boundaries.ply` | boundary mesh (marching cubes) | `output.mesh.enabled`, curved geometry only, needs `scikit-image` |
| `resolved_config.yaml` | the fully-resolved config (all defaults filled, all `auto` values computed); reproduces the run | always |
| `run.log` | INFO/DEBUG run log | always |
| `MANIFEST.txt` | SHA-256 hash + size of every output file this run wrote | always |
| `diagnostics/*.dat` | grain atom counts, warp-field stats, voxel histograms | `meta.verbose >= 3` |

See [`docs/outputs.md`](docs/outputs.md) for the full column-by-column glossary
and estimator-bias notes.

Grainsmith ships 47 example configs (`examples/**/*.yaml`, grouped by topic)
forming a guided tour
of every feature — flat/curved/self-affine boundaries, all six orientation
schemes, solid solutions, Lloyd relaxation, slab geometry, non-cubic
families, log-normal grain-size targeting, Σ3-enriched boundary networks,
imported voxel microstructures, single-crystal and multiphase builds,
GB-targeted doping, and multi-million-atom production runs. **Start at
[examples/README.md](examples/README.md)** for the recommended order.

## Conventions

- **Quaternions** are scalar-first `(w, x, y, z)` and encode the **active**
  rotation `v_lab = R(q) @ v_crystal`. Bunge Euler angles `(φ1, Φ, φ2)` are
  the **standard** convention: the orientation matrix `g = R(q)ᵀ`
  (specimen→crystal) factorizes as `Rz(φ2)·Rx(Φ)·Rz(φ1)`, matching MTEX
  `loadOrientation_generic(...,'Bunge')`, EBSD devices, and texture
  textbooks. scipy `Rotation` is touched only through two bridge helpers.
- **float64 everywhere**; every numerical tolerance lives in `constants.py`
  with a written justification.
- **Atom positions are stored unwrapped** in absolute lab coordinates;
  writers wrap to `[0, L)` on periodic axes at write time. On free axes,
  bounds are `0 … L + vacuum` with atoms shifted by `+vacuum/2`.
- **Determinism:** one master seed → named child streams (`seeding`,
  `orientation`, `fields`, `occupancy`, `sizes`, `mdf`, `doping`) via
  `SeedSequence.spawn()`; `np.random.*` module functions are forbidden.
  Re-running an identical config on the same machine reproduces every output
  byte-for-byte (timestamps aside), for any `--jobs` value — see
  [*Reproducibility*](#reproducibility) below for what survives a change of
  machine.
- **LAMMPS** `atom_style: molecular` stores the 1-based grain id in the
  molecule-id column (OVITO-friendly grain coloring); the `grain` column
  of the extended XYZ and the CSVs is the 0-based grain id.

## Reproducibility

What "the same result" means depends on how much of the environment you can
match. Three tiers, in decreasing order of guarantee:

| you match | you get |
|---|---|
| same seed, `pip install -c constraints-repro.txt`, same BLAS vendor, same CPU family | **Byte-identical structure.** Add `SOURCE_DATE_EPOCH` and the rest of the output directory matches too — `run.log` aside, and aside from the rows that measure the run itself: `summary.csv`'s `timings,*` / `memory,*` and the one `MANIFEST.txt` line that hashes it. |
| same library versions, but a **different** BLAS build or CPU | **Scientifically identical** — same grains, atom count, species and gate values — but a minority of coordinates differ by ~1 ulp, so the digests differ. |
| **different** `numpy` / `scipy` / `spglib` versions | **No guarantee.** `np.random.Generator` explicitly disclaims cross-version stream stability, so the structure itself can change. |

Two things keep that honest rather than aspirational:

- **Every run records the environment it ran in** — BLAS vendor and version, OS,
  CPU architecture, numba state, and the effective `--jobs` — into `summary.csv`
  and `microstructure.json`. So when two digests disagree, the files themselves
  tell you which tier you are in, instead of leaving you to guess.
- **`grainsmith verify <outdir>`** makes that checkable by someone who did not
  run the pipeline: it re-hashes every file against `MANIFEST.txt` and re-derives
  both digests from the shipped files, so a second researcher can prove that what
  they were handed is intact and really came from the version it claims.

For the per-file breakdown of what changes between two runs, and the exact
`SOURCE_DATE_EPOCH` semantics, see [*Determinism and
provenance*](docs/manual.md#determinism-and-provenance) in the user manual.

## Physics highlights

- **Seeding:** random sequential addition with per-axis minimum-image
  spacing, optional Lloyd (centroidal) relaxation.
- **Flat boundaries:** exact periodic Voronoi via Qhull, built on the RSA
  hard-core seeds above — a minimum-separation seed process, so the result is
  not a Poisson–Voronoi tessellation; volume-sum and Euler-characteristic
  gates are enforced at construction.
- **Curved boundaries:** domain-warped Voronoi, additive-weight
  (Johnson–Mehl), or anisotropic (ellipsoidal-metric) backends. The warp
  curves faces but preserves Voronoi *topology* — it creates no new
  neighbors; genuinely non-Voronoi shapes come from `perturbed_distance` or
  `voxel_import`.
- **Self-affine roughness:** the `perturbed_distance` level-set backend
  perturbs the grain *assignment rule* itself, producing a band-limited
  self-affine boundary — scale-free across the synthesis band — with a
  tunable Hurst exponent. Band-limited self-affine surface synthesis has
  established precedents; what grainsmith adds is its use as a built-in
  grain-assignment rule rather than a post-processing step.
- **Atom fill:** each grain's compact home cell is filled by direct lattice
  enumeration; home cells tile the torus exactly once, so any orientation
  yields each lattice point exactly once.
- **Overlap removal:** PBC-aware nearest-neighbor queries with
  `delete_shallower` / `keep_lower_id` / `midpoint_merge` (elemental
  systems only) policies; a post-condition re-check proves the final
  minimum distance meets the cutoff.
- **GB character:** tilt/twist/mixed classification from the physical
  minimal-angle lattice map, plus Brandon-criterion CSL Σ assignment
  (cubic point groups only).
- **Grain-size targeting:** a power/Laguerre diagram whose weights are
  fitted by semi-discrete optimal transport to log-normal, equal, or
  explicit volume targets.
- **Texture & boundary-angle shaping:** weighted ODF components (Euler
  center + spread, fiber, or uniform), realised by VOLUME fraction by
  default (`component_weight_basis`), plus assignment annealing toward a
  Haar-random, Σ3-angle-enriched, or user-supplied target for the
  boundary-area-weighted **disorientation-angle** distribution. That
  objective is the angle marginal, not a full MDF: it never sees the
  misorientation axis. The permutation leaves the grain-COUNT-weighted
  orientation distribution invariant by construction; the VOLUME-weighted
  ODF is not invariant, and `mdf_target.odf_drift_max` bounds its drift
  with a hard per-swap veto (re-measured by gate G22).
- **Voxel import:** atomize an external voxel label field (`.npy` or
  DREAM.3D HDF5) with a derived grain count and optional imported
  per-grain orientations.
- **Single crystal:** `grains.number: 1` builds the whole box as one grain,
  with a box/lattice commensurability check.
- **Multiphase:** independent space group/lattice/Wyckoff sites per phase,
  volume-fraction targeting, and interphase boundary reporting (habit-plane
  indices, no misorientation).
- **GB curvature:** local mean and Gaussian curvature from the implicit
  level set of the boundary, with a stated voxel-resolution limit and
  Gauss–Bonnet reliability caveat.
- **Doping:** GB-targeted substitutional or interstitial dopant insertion
  on top of the finished structure, with achieved-fraction/enrichment and
  min-distance re-verification gates.

Every item above has a full write-up, caveats, and citations in
[`docs/physics.md`](docs/physics.md).

## Documentation

Full docs live in [`docs/`](docs/) (Markdown; an HTML mirror is in
`docs/html/`, regenerate with `python -m tools.build_docs_html`):

- [`docs/index.md`](docs/index.md) — documentation landing page.
- [`docs/manual.md`](docs/manual.md) — install, quickstart, the pipeline
  stages, determinism, reading outputs.
- [`docs/architecture.md`](docs/architecture.md) — how a run flows through
  the package, stage by stage.
- [`docs/geometry.md`](docs/geometry.md) — every tessellation backend: what
  it does and the config parameters that control it.
- [`docs/config_reference.md`](docs/config_reference.md) — every config
  field (generated from the schema; drift-pinned by a test).
- [`docs/gates.md`](docs/gates.md) — the QA gates: check, threshold, and
  what to do when one trips.
- [`docs/outputs.md`](docs/outputs.md) — every output file, column
  glossary, and estimator-bias notes.
- [`docs/physics.md`](docs/physics.md) — conventions, tessellation/SDOT,
  texture/MDF, self-affine roughness, multiphase, doping, with citations.
- [`docs/developer.md`](docs/developer.md) — calling grainsmith from
  Python, extending it with a new backend/sampler/writer.
- [`docs/faq.md`](docs/faq.md) — troubleshooting: gate failures,
  performance, Windows.

## Known limits

These are stated up front to pre-empt surprises:

- Polycrystal boxes are orthogonal only; a lattice-multiple triclinic box
  (`box.cells`) is supported for single-crystal builds only.
- A single crystal whose box is incommensurate with its oriented lattice can
  carry an artificial defect where the grain meets its own periodic image,
  even when the minimum-distance check (G7) passes. Gate G14 measures that
  mismatch and warns rather than aborting, so intentional join defects remain
  possible — a passing G7 is not by itself proof of a defect-free periodic
  crystal.
- CSL Σ assignment is restricted to cubic point groups.
- Voxel/curved-geometry estimators (sphericity, S_V, triple-junction L_V)
  carry a geometric bias — oblique boundary area is over-counted by up to
  **√3** — and are reported per section in `statistics.csv` (with an
  `estimator` tag) rather than silently corrected; flat, power/Laguerre and
  single-crystal geometries use exact polyhedral values.
- The GB-curvature estimator samples face interiors only, so the per-grain
  Gauss–Bonnet residual (gate G21) routinely exceeds one topological unit
  (4π) on curved runs, including near-flat ones. It is a warn-only fidelity
  signal for that estimator, not a defect in the structure — G3/G4
  (volume/topology) and G7 (minimum distance) certify the structure
  independently.
- The log-normal grain-size KS p-value uses *fitted* parameters and is
  therefore optimistic (the Lilliefors caveat).
- At the grain counts typical of MD (≈ 8–100), a prescribed log-normal
  `grains.size_distribution.sigma_log` is attained as the mean of the realised-shape
  distribution, not per run: a single 10-grain model can fit a σ well away
  from the target purely through sampling. The per-grain *volume* targets
  themselves are still met to `vol_tol` and re-measured by gate G11.
- The domain warp curves faces but preserves Voronoi topology (no new
  neighbors); genuinely non-Voronoi shapes come from `perturbed_distance`
  or `voxel_import`.
- Composing a `warp` with a `size_distribution` targets the volumes on the
  *unwarped* power base: gate G11 re-measures that base, while the deviation
  the warp then introduces is measured separately by gate G24 against
  `WARP_VOLUME_G24_TOL = 0.10` (calibrated: 0.021–0.041 post-warp against
  ~1e-4 on the base). For equiaxed grains with tight realised volumes, use
  the centroidal loop instead of a warp.
- The self-affine spectrum is band-limited to `k ∈ [2π/l_max, 2π/l_min]`:
  roughness is scale-free across that band only, so the back-estimated
  Hurst exponent (G13) and the box-counting roughness index (G20) describe
  the synthesis band rather than a converged fractal dimension — G20 is
  reported as a threshold-free diagnostic.
- Assignment annealing permutes which grain carries which orientation, so
  it preserves the grain-COUNT-weighted orientation distribution exactly
  but NOT the volume-weighted ODF — with unequal grains a swap moves
  |V_a − V_b| / ΣV of its mass. Set `mdf_target.odf_drift_max` to bound
  that drift; left unset it is only measured, not bounded.
- `odf_drift_max` bounds drift in the *tessellation-volume* weighting only.
  The artifact an MD user consumes is atom-count weighted, and the gap
  between the two weightings is a separate discretisation floor reported by
  gate G26 (0.027 at 275 atoms/grain, decaying roughly as
  (atoms/grain)^(-1/3)) — tightening `odf_drift_max` does not close it.
- The annealing objective is the disorientation-ANGLE marginal, so the Σ3
  Brandon window it optimises is a NECESSARY, not sufficient, condition for
  Σ3: true CSL also requires the ⟨111⟩ axis. Gate G23 reports the window
  area fraction and the true CSL Σ3 fraction side by side, and the
  attainable Σ3 is bounded by the texture's own twin content.
- Lamellar/colony morphologies and interphase orientation relationships
  (Burgers, Kurdjumov–Sachs/Nishiyama–Wassermann) are not generated
  (documented planned extension; use `voxel_import` for lamellar
  morphologies today).
- Curved-boundary mesh output requires `scikit-image`
  (`pip install ".[mesh]"`).
- Doping is single-phase only (`doping:` + `phases:` together is a config
  error).
- Dopant placement is a **geometric** scheme: sites are selected by distance
  to the boundary against a target composition and enrichment, and the
  achieved values are re-measured (G17/G18). No segregation energy,
  temperature or chemical potential enters, so a doped structure is an
  initial condition for a simulation, not an equilibrium segregation
  profile.
- A fully periodic **bicrystal** meets itself through both sides of the
  torus with antiparallel sheet normals: the mean boundary plane is then
  reported as `undefined` by design (use ≥ 3 grains or a slab for a
  defined mean plane).
- The per-allocation memory guard (`runtime.memory_limit_gb`, default 16 GB)
  is clamped to the machine's detected physical RAM and can only ever be
  lowered by it, not raised past it; it is not a cap on total process RSS
  (that's the separate, optional `--max-rss` monitor) — see `docs/faq.md`'s
  *Memory for big runs*.

## Using the output in LAMMPS

```text
units metal
atom_style atomic           # or 'molecular' per output.lammps.atom_style
read_data polycrystal.data
pair_style eam/alloy        # choose your potential
pair_coeff * * Cu.eam.alloy Cu
minimize 1e-8 1e-10 1000 10000
```

## Citation

A software paper describing `grainsmith` is in preparation for *Computer
Physics Communications*. Until it appears, please cite the repository.

## License

MIT — see `LICENSE`.
