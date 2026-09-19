# grainsmith FAQ & troubleshooting

Quick answers to the common failure modes. Gate thresholds and remedies are in
`docs/gates.md`; config fields in `docs/config_reference.md`.

## Gate failures

**G1 "extra fields not permitted" / unknown key.** Every config model is
`extra='forbid'`, so a misspelt key is an error. Check the spelling against
`docs/config_reference.md`. Common slips: `wyckoff_sites` (not `wyckoff`),
`size_distribution` under `grains`, `overlap_removal` under `boundaries`.

**G1 "exactly one of crystal or phases".** Use `crystal:` for single-phase,
`phases:` (a list of ≥ 2) for multiphase — never both.

**G1 "'cif' is mutually exclusive with the manual ... trio".** Inside a
`crystal:` (or per-phase `crystal:`) block, `cif: {file: ...}` and the manual
`space_group` / `lattice` / `wyckoff_sites` trio cannot both be set — pick one
input form. Conversely, "provide either 'cif' ... or the manual trio" means
neither was given; and "requires space_group, lattice, AND wyckoff_sites all
set" means the manual trio is incomplete (no `cif` present either). See
`docs/manual.md` § *Crystal input* for the CIF path, or
`examples/cif/tini_cif_polycrystal.yaml` for a worked example. A CIF whose own
`_symmetry_Int_Tables_number` disagrees with what spglib detects is NOT an
error — only a logged WARNing; spglib's detection is always authoritative.
Partial or mixed site occupancy in a CIF raises a clear ConfigError (use the
manual `wyckoff_sites` path with an explicit `{element: {...}, ...}` dict
instead).

**G2 round-trip failed (wrong space group).** The expanded motif did not
reproduce the requested ITA number. Causes: wrong Wyckoff `coords`, a *free*
Wyckoff parameter left symbolic (substitute a number, e.g. O 4f x=0.305), an
incompatible `setting` (origin choice 2, or R vs H), or two Wyckoff sites placed
at coincident coordinates. Use `grainsmith info --sg <N> --site "x,y,z"` to see
the orbit.

**G6 warp bijectivity (clip/gradient).** The warp amplitude is too large for the
length scale. Reduce `boundaries.curved.amplitude` (hard cap:
≤ `min_seed_distance`/4), or increase `correlation_length`. `warp` only ever
accepts `spectrum: gaussian` — a `ConfigError` (`method 'warp' does not accept
spectrum 'self_affine'`) means you want `boundaries.curved.method:
perturbed_distance` instead, the only method the `self_affine`
spectrum is legal on; see [geometry.md §4b](geometry.md#4b-perturbed-distance--genuinely-self-affine-grain-boundaries).

**G7 atoms closer than cutoff.** Overlap removal is off or too weak. Set
`boundaries.overlap_removal.enabled: true` and/or raise `cutoff` (e.g.
`0.85*d_nn` → `0.9*d_nn`).

**G8 atom count off (WARN > 2 %, FAIL > 5 %).** Large deviation usually means an
unexpectedly large overlap-deletion count or a density/occupancy mismatch. Check
the deletion count and composition in `summary.csv`, and the lattice parameters.

**G11 volume targeting failed.** The SDOT fit could not hit `vol_tol`. Increase
`size_distribution.max_iter`, reduce `sigma_log` (a narrower distribution is
easier), or increase `grains.number`. The target spread may be infeasible for the
seed set. **Note on combining with curved/self-affine boundaries:** when a
`size_distribution` is used *together* with a `warp` boundary, the volume target
is enforced on the **unwarped power base** (the warp then displaces the
boundaries, so the realized per-grain volumes deviate from the targets, bounded
by the G6 gradient guard). G11 reports the base error and its message says so —
do not read a warp+SDOT PASS as a guarantee on the realized warped cells. For
equiaxed shapes *with* tight volumes, prefer the centroidal loop
(`centroidal_iterations`) instead of a warp.

**G12 angle-histogram χ² above threshold (WARN).** Increase `mdf_target.annealing_steps`. The
achievable Σ3 area fraction is **bounded by the twin content of the fixed
orientation set** because the assignment-annealing is **conserved-texture** — it
only permutes a fixed orientation multiset over the (fixed) Voronoi neighbour
graph, and the NUMBER-weighted orientation distribution (not the volume-weighted
ODF) is what stays invariant by construction. Measured behaviour
(`sigma3_angle_enriched` sweep, `sigma3_fraction` 0.1–0.5 x 3 seeds, case `M2` of the MDF
validation campaign — local validation tooling, not published with this repository): the realised **Brandon
angular-window** area fraction (`brandon_area_frac`, angle only) climbs with the
target but stays within roughly **0.24–0.44** across the whole sweep, while the
**true CSL Σ3 area fraction** (`sigma3_area_frac`, angle AND ⟨111⟩ axis, from
`statistics.csv`) stays far smaller throughout, roughly **0.02–0.06** — because
the annealing objective enforces only the 60° Brandon angle window, never the
axis condition CSL classification also requires. The method *shapes* the
disorientation-angle marginal toward the target under that ceiling; it does not hit an
arbitrary target, and reaching a higher *true* Σ3 fraction needs orientations
that are actually twin-related (e.g. two `odf_components` 60° ⟨111⟩ apart), which
this objective alone cannot add — see `docs/physics.md` §4.

**What does annealing actually preserve, and is the ODF itself safe?**
Only the NUMBER-weighted orientation distribution is invariant by
construction (see above) — the ODF as conventionally defined is
**volume**-weighted, and annealing's own drift there is a separate,
optionally controlled term (`mdf_target.odf_drift_max`, gate **G22**).
The default `null` measures drift without constraining it. But that is
only half the picture: the volume-weighted ODF can already be off target
*before* annealing ever runs, through `orientation.scheme: odf_components`
itself — `component_weight_basis: "count"` realises
each component's configured `weight` as a GRAIN-COUNT fraction, not a
volume fraction, and the two can differ substantially on a small,
broad-`size_distribution` grain population (measured 24-grain case:
configured 0.500/0.500 realised as 0.745/0.255 in volume — a 0.245
total-variation gap, twelve times the annealer's typical 0.02 cap). Set
`component_weight_basis: "volume"` (the default from this revision on) to
reduce that gap with deterministic whole-grain partitioning. Finite-grain
granularity remains; exact volume matching is generally impossible.
Gate **G25** reports both the residual component-fraction error and the
size-component correlation. See `docs/physics.md` §4's component-fraction
error budget, which is an inequality rather than an exact decomposition
of the full continuous ODF error.

**G13 Hurst off target (WARN).** A finite-band bias is expected (measured): the
back-estimated **absolute** H is **low-biased** (≈ −0.114 measured), although the
*relative* ordering across target H is faithful. If you need a specific
roughness, raise the target H to compensate, and widen the scale-free band: lower
`l_min`, raise `l_max` (≤ min(L)/2), and use a finer `analysis.voxel_grid` for
more octaves.

**G14 incommensurate single crystal (WARN).** For `grains.number: 1`, use a box
that is an integer number of lattice repeats and/or an identity orientation;
otherwise the periodic faces are self-boundary defects (sometimes intended).

**G15 phase fractions off (WARN).** Fraction resolution is limited by grain count
(deviation floor V_max/V_box). Increase `grains.number`.

**G16 curvature samples degenerate (WARN).** More than 5 % of grain-boundary
level-set curvature samples were dropped as degenerate (gradient magnitude
below `CURV_GRAD_MIN`). Raise `analysis.voxel_grid` for finer level-set
sampling; near-planar curved backends may legitimately report near-zero H
everywhere, which is not itself a problem.

**G17 dopant composition/enrichment off target (WARN).** A dopant's achieved
atom fraction drifted more than 1 percentage point from its nominal
`concentration`, or the achieved enrichment came out more than 15 % off the
target `enrichment`. This is often statistical noise when the expected
bulk-dopant count is small (see the rule of thumb in `docs/gates.md`).
Raise `concentration` or shrink `shell_width` to grow the candidate pools, or
increase `grains.number` for more sites.

**G18 dopant min-distance violation.** An interstitial dopant site ended up
closer than its `min_distance` to another atom on the final re-verification
pass. Lower `min_distance`, choose a different interstitial site preset, or
lower the dopant `concentration`.

**G19 perturbed_distance reassigned-voxel fraction (WARN).** More than 2 % of
voxels needed majority-vote reassignment to repair disconnected fragments
during `perturbed_distance` construction. This is a repair-severity signal,
not a pass/fail test — the structure is always usable (repair either
converges or the constructor raises `TessellationError` outright, which
never reaches `summary.csv`). A large fraction usually means the amplitude
is aggressive relative to `min_seed_distance`; reduce `amplitude` or raise
`correlation_length`/`l_min`.

**G20 box-counting roughness index (informational, WARN-eligible).** `d_b`
in `summary.csv` is a local-roughness diagnostic from a box-counting fit
over the synthesis band, not a converged fractal-dimension certificate at
MD-typical grain sizes (see `docs/physics.md` §5b and `docs/gates.md`).
`d_b = n/a` means no sampled (grain, section) combination yielded a usable
fit (grains too small relative to the section resolution).

**G21 Gauss-Bonnet residual (WARN).** The per-grain face-interior curvature
residual exceeded one topological unit (4π). This is a standing, coarse
fidelity caveat on the `gb_curvature.csv` H/K columns for the current
estimator — it routinely fires on ordinary curved runs (and even near-flat
ones), and does **not** shrink by refining `analysis.voxel_grid` over the
tested range. It never touches the atomistic output itself (G3/G4/G7
already certify that independently); treat a WARN here as a reminder to
read curvature numbers with that caveat in mind, not as an actionable
defect by itself. See `docs/gates.md` G21 for the calibration numbers.

**My crystal isn't fcc/bcc/hcp — can I still add interstitials?** Yes: give
explicit `sites: {coords: [[x, y, z]]}` — any of the 230 space groups works.
One representative per interstitial orbit is enough: by default
(`expand_orbit: true`) it is expanded to the full symmetry orbit of the host
space group (the log reports "N listed site(s) -> M site(s)/cell"). This
also works with `crystal.cif` input, where the spglib-*detected* group
drives the expansion. Set `expand_orbit: false` only if you deliberately
want a symmetry-breaking partial sublattice — then you must list every
equivalent site yourself.

**How do I check the dopants actually segregated to the GBs?** Look at
`doping_profile.csv` / run `gnuplot doping_profile.plt` in the output
directory: the bottom panel plots the local dopant fraction c(d) against
GB distance — with segregation it steps down at the shell edge from
≈ enrichment × bulk level to the bulk level; without it the profile is flat
at the nominal fraction. Gate G17's achieved enrichment is the shell/bulk
ratio of the same quantity.

## Performance

**SDOT is slow for many grains.** The SDOT solve scales with the grain count,
not the box size (measured). N ≈ 100 is seconds; N ≈ 1000 can take ~1–2
minutes on a typical core, and slower hardware scales up from there — these
are honest, hardware-dependent numbers, not errors. The volume *accuracy*
(G11) is unaffected.

**Use `--jobs` for large fills.** `grainsmith generate cfg.yaml --jobs 0` runs
the per-grain atom fill, and (when `analysis.gb_curvature` is on) the
per-boundary-pair GB-curvature analysis, on all cores *available to the
process* (the cgroup/affinity mask — under a SLURM `--cpus-per-task`
allocation that is the allocated cores, not the whole node). Speedup is
near-ideal to ~4 workers and saturates around 8 (measured). `--jobs` also
sets the thread count for the overlap-removal neighbour search and gate G7's
fresh query, though that stage's own pair-processing loop stays serial. The
tessellation and file I/O are serial and dominate very large runs, so total
wall time improves less than the fill time alone. Outputs are bit-identical
for any `--jobs`.

**On SLURM / HPC clusters.** grainsmith is a **single-node** program: one run
= one node = one task, and all parallelism is `--jobs` worker processes
inside that task. Never request `--nodes > 1` or `--ntasks > 1` for one run —
each extra task would execute the full pipeline again into the same output
directory. The best-parallelism operating point on a single node is
`--jobs = grains.number` (the fill stage hands each grain its own worker, so
every grain fills simultaneously; more cores than grains speed up only the
per-boundary-pair analysis stage), with `--cpus-per-task` set to match. The
numba kernels are disk-cached (`cache=True`) inside the installed package, so
one small warm-up run after installation compiles everything once for every
node that shares the filesystem. See `examples/hpc/` for ready-made
cluster-scale configs, a submit-ready `job.sbatch` template, and the full
cluster guide (`examples/hpc/README.md` — `sbatch`, the `salloc`
interactive case, job arrays, pitfalls).

**Memory for big runs.** The 9.2-million-atom flagship example peaks at ~13 GB
resident — budget **≥ 16 GB RAM** for production sizes; that is a measured
total-RSS figure for one run, not a setting. Separately, the per-allocation
memory guard (`runtime.memory_limit_gb`, **16 GB by default**, raised from a
hard-coded 8 GB) blocks any single allocation — the per-grain
lattice-enumeration grid (`atoms/fill.py`), the analysis voxel grid
(`tessellation/voxel.py`), or the imported voxel-label EDT margin
(`tessellation/voxel_import.py`) — whose estimated size exceeds it, with an
actionable error instead of an OOM-kill. The two 16 GB numbers now coincide,
but they are **different things** (a per-allocation cap you can raise in the
YAML vs. a measured total-RSS figure for one example) — see *Memory guards* in
the [user manual](manual.md) for the total-RSS `--max-rss` monitor, which is
what actually watches cumulative RSS. The effective budget enforced is
`min(runtime.memory_limit_gb, detected physical RAM)` — a hard clamp: raising
the config value past what the machine actually has does **not** raise the
ceiling, physical RAM always wins (detected best-effort via `psutil`,
`os.sysconf`, `/proc/meminfo`, or the Windows/macOS equivalents; if every path
fails, the configured value is used unclamped and a warning is logged). The
error names whichever bound tripped: if physical RAM is the smaller, binding
bound, it names the detected RAM and advises reducing the box size / grain
count or moving to a bigger machine; if your own `runtime.memory_limit_gb`
bound with RAM to spare, it advises raising that setting instead. With `--jobs
N`, up to N grains fill concurrently, each allocating its own grid, so budget
for N times the per-allocation limit — but **only** for the fill stage: the
analysis voxel grid and the voxel-import EDT margin are each one whole-box
allocation on the single-threaded driver, unaffected by `--jobs`. On a
RAM-limited host a very large run can still swap without the guard firing —
size the box/grain count to your machine, or watch memory during the run. The
effective budget is machine-dependent and never written to any output file —
only your own `runtime.memory_limit_gb` is recorded, in
`resolved_config.yaml`, so cross-machine reproducibility is unaffected.

## Windows notes

- **Console encoding (cp1254 etc.).** Logs and CLI output are ASCII-safe so they
  never crash on a non-UTF-8 Windows console. Output *files* are written UTF-8.
- **Paths with spaces / non-ASCII** (e.g. `Masaüstü`) work; quote paths on the
  command line.
- **gnuplot / OVITO are optional viewers.** `view.plt` needs gnuplot installed to
  render `view.png` (`gnuplot view.plt`); otherwise load `polycrystal.data` in
  OVITO. grainsmith itself does not require either.
- **Optional extras.** `.ply` mesh output needs `pip install ".[mesh]"`
  (scikit-image); DREAM.3D HDF5 import needs `pip install ".[import]"` (h5py).
  Without them, those specific features are skipped (and the matching tests are
  skipped) — the core pipeline runs fully.

## Reproducibility

**Will I get the same result twice?** The *scientific payload* — yes,
byte-for-byte, with a fixed `seed.value`, on the same platform and dependency
versions, for any `--jobs` value. The *whole directory* — only with a frozen
clock. Every file that permits comments carries a one-line provenance header
containing the run's UTC timestamp, so by default two runs of the same config
differ in that one line of each file (and `MANIFEST.txt`, which hashes those
files, differs in every digest). Set `SOURCE_DATE_EPOCH` to freeze it:

```bash
SOURCE_DATE_EPOCH=1700000000 grainsmith generate config.yaml
```

That stamps the given epoch second instead of the wall clock and drops the
wall-clock `timings,*_s` rows from `summary.csv`, which makes the entire output
directory byte-identical between two runs — `MANIFEST.txt` included, with
`run.log` the sole exception (it is a log, and `MANIFEST.txt` deliberately lists
it without a digest).

**Across machines?** Not guaranteed, and not merely in the last bit. The atom
fill multiplies matrices through NumPy's BLAS, so a different BLAS build or CPU
architecture shifts a measurable fraction of coordinates by ~1 ulp — the grain
structure, atom counts and species are unaffected, but the digests change. A
different *version* of NumPy, SciPy or spglib can change the result outright:
`numpy.random.Generator` gives no cross-version bit-stream guarantee. Install
with `pip install -c constraints-repro.txt -e .` to pin the versions this
repository was developed against, and check the `environment` section of
`summary.csv` (BLAS name/version, OS, CPU architecture, numba state, `--jobs`)
to see what actually produced a given run.

**How do I check an archived run is intact?** `grainsmith verify <outdir>` — see
*Determinism and provenance* in the [user manual](manual.md).

Use `seed.mode: entropy` to draw and record a fresh seed (saved in
`summary.csv`).
