# grainsmith user manual

grainsmith generates **space-group-aware atomistic polycrystals** for molecular
dynamics, with statistically controlled microstructure (grain size, texture,
boundary network, boundary roughness), construction-time QA gates, and
reproducible, publication-grade outputs.

This manual is the quickstart and the conceptual map. See also:
`docs/architecture.md` (how the pipeline and modules fit together),
`docs/geometry.md` (grain-geometry backends and their parameters),
`docs/developer.md` (programmatic API and extension points),
`docs/config_reference.md` (every config field), `docs/gates.md` (the QA gates),
`docs/outputs.md` (every output file + columns), `docs/physics.md` (conventions
and math), `docs/faq.md` (troubleshooting).

## Install

Requires Python ≥ 3.10. From the repository root:

```
pip install .
```

Runtime dependencies: numpy, scipy, spglib, pyyaml, pydantic. Optional extras:

```
pip install ".[mesh]"     # scikit-image — boundary .ply mesh output
pip install ".[import]"   # h5py — DREAM.3D HDF5 voxel import
pip install ".[cif]"      # ase — read crystal structures directly from a CIF file
pip install ".[monitor]"  # psutil — best-effort total-RSS monitor (--max-rss)
pip install ".[perf]"     # numba — JIT-accelerated owns()/margin()/grain_of()
                           #   kernels for the additive-weights, anisotropic and
                           #   perturbed_distance backends; pure NumPy is the
                           #   default and always available (GRAINSMITH_NO_NUMBA=1
                           #   forces it even when numba is installed) — results
                           #   are bit-identical either way, so this is a pure
                           #   speed-up, never a correctness dependency.
pip install ".[test]"     # pytest, ase — run the test suite
```

Verify the install:

```
grainsmith --version              # confirm which grainsmith is on PATH
grainsmith info --sg 225          # inspect a space group (FCC here)
grainsmith --help
```

## Quickstart

Generate your first polycrystal from a shipped example:

```
grainsmith validate examples/basics/b2_nial_flat.yaml    # schema (G1) + crystal (G2) only
grainsmith generate examples/basics/b2_nial_flat.yaml    # full run → ./out_*/
grainsmith verify ./out_b2_flat                          # re-check the outputs' digests
```

`generate` prints `OK: N/N QA gates passed` (or `WARN: …` with the failed warn
gates). Then read `summary.csv` first, then `grains.csv` / `boundaries.csv`, and
visualize with `gnuplot view.plt` → `view.png` or by loading `polycrystal.data`
in OVITO (colour by molecule-id = grain).

`b2_nial_flat.yaml` also demonstrates `analysis.gb_curvature` (writes
`gb_curvature.csv`; flat geometry ⇒ exact zeros).

A minimal config looks like:

```yaml
seed: {mode: fixed, value: 42}
box: {lengths: [60.0, 60.0, 60.0]}
grains: {number: 12}
crystal:
  space_group: {number: 225}        # Fm-3m (FCC)
  lattice: {a: 3.615}               # Cu
  wyckoff_sites: [{element: Cu, coords: [0.0, 0.0, 0.0]}]
orientation: {scheme: random_uniform}
```

The `examples/` directory is a guided tour from this baseline up to multiphase,
textured, self-affine and 9-million-atom production runs — see
`examples/README.md`.

### Crystal input: manual Wyckoff sites, or directly from a CIF file

The `crystal` block above (`space_group` + `lattice` + `wyckoff_sites`) is one
of two mutually exclusive ways to define the crystal structure. The other
reads a CIF file directly — no manual space-group number, lattice parameters,
or Wyckoff coordinates:

```yaml
crystal:
  cif:
    file: ../assets/TiNi.cif   # any file ase.io.read accepts; relative to
                               # the CWD first, else to this YAML's own dir
    symprec: 1.0e-4           # spglib symmetry-detection tolerance (default)
```

The file is parsed with [ASE](https://wiki.fysik.dtu.dk/ase/) and its symmetry
is *independently detected* with spglib at `symprec` — the file's own
`_symmetry_Int_Tables_number` (if present) is only cross-checked, never
trusted outright: a CIF that under-declares its symmetry (e.g. exporting every
atom under `P 1`, a common pymatgen/CIF-export habit) still resolves to its
true, higher-symmetry space group, with a WARNING logged naming both numbers.
spglib's own standardized cell then supplies the space group, family-reduced
lattice parameters, and one representative Wyckoff site per symmetry orbit —
exactly the same internal representation the manual path builds, so every
downstream stage (orientation, tessellation, fill, LAMMPS output) is
unaffected by which input form was used. `crystal.cif` requires the optional
`ase` extra (`pip install ".[cif]"`); only full site occupancy is supported
(partial/mixed occupancy needs the manual `wyckoff_sites` path with an
explicit occupancy dict).

Traceability: the detected space-group number + Hermann–Mauguin symbol, the
`symprec` used, and the per-site Wyckoff letters are written to
`summary.csv` (section `crystal`, rows `cif_file` / `cif_symprec` alongside
the usual `detected_sg` / `international` / `wyckoff_letters` rows); the
resolved lattice parameters are spelled out in `METHODS.md` and echoed in
the run log — nothing about the CIF-derived structure is a silent
assumption. See `examples/cif/tini_cif_polycrystal.yaml`,
`examples/cif/ti2ni_cif_single_crystal.yaml`, and
`examples/cif/tini3_cif_polycrystal.yaml` for worked examples (the last
demonstrates the declared-vs-detected mismatch WARNing).

### Triclinic boxes for single crystals (`box.cells`)

`box` normally takes `lengths: [Lx, Ly, Lz]` for an axis-aligned orthogonal
box. For a **single crystal** (`grains.number: 1`) whose own conventional
cell is NOT orthogonal (monoclinic, hexagonal, triclinic, …), an
axis-aligned box can never exactly match the crystal's periodicity — the
box faces become genuine (if often small) self-boundary defects, which
gate G14 reports as a nonzero commensurability strain and which overlap
removal then cleans up by deleting atoms along the faces.

`box.cells: [n1, n2, n3]` sidesteps this: it replaces `lengths` with an
integer triple, and the box vectors are built AS the crystal's own
conventional-cell vectors scaled by `[n1, n2, n3]` — the box literally IS
`n1 × n2 × n3` repeats of the true unit cell, so it is exactly periodic
**by construction**, for any crystal family. Gate G14 then reports a
misfit of 0 (to float64 roundoff) rather than "small but nonzero".

```yaml
box:
  cells: [6, 5, 4]          # box = 6x5x4 exact repeats of the crystal cell
  periodic: [true, true, true]
  vacuum: 0.0

grains:
  number: 1                  # box.cells is single-crystal only

orientation:
  scheme: fixed
  fixed:
    euler_bunge_deg: [0.0, 0.0, 0.0]   # box.cells REQUIRES identity
```

Preconditions (enforced at config-resolve time with a clear `ConfigError`
naming the violated rule):

- `box.cells` and `box.lengths` are mutually exclusive — exactly one must
  be given.
- `grains.number: 1` — the polycrystal tessellation core stays
  orthogonal-box only; an arbitrarily-oriented grain-boundary network has
  no natural lattice-multiple box shape (see README "Known limits").
- `box.periodic: [true, true, true]` and `box.vacuum: 0.0` — the box is
  exactly periodic by construction, so there is no meaningful free/vacuum
  axis.
- `orientation.scheme: fixed` with an **identity** rotation (Euler
  `[0,0,0]`, zero-angle axis-angle, or the identity quaternion) — the box
  vectors are built directly from the crystal's own UNROTATED cell, so any
  other orientation would misalign the lattice with the box/lab frame.
- Not combined with `phases`, `doping`, `analysis.section`, or
  `analysis.gb_curvature` (each assumes either multiple crystals, an
  orthogonal-box neighbor search, or a boundary that a single crystal does
  not have).

The resolved box matrix — in the LAMMPS **restricted-triclinic**
convention (`a` along `+x`, `b` in the `xy`-plane with `+y`, `c` with
`+z`) — is written back onto `config.box.resolved_h` and echoed in
`summary.csv` (section `box`, rows `cell_matrix_a_A` / `_b_A` / `_c_A` /
`tilt_xy_xz_yz_A`) and `METHODS.md`. If the raw `A @ diag(n1,n2,n3)`
product has an off-diagonal tilt factor exceeding the LAMMPS bound
(`|xy|, |xz| <= ax/2`, `|yz| <= by/2`), grainsmith automatically applies an
integer lattice-vector reduction (subtracting an integer multiple of one
box vector from another — the same represented lattice, just a
differently-shaped fundamental domain) to bring it back into bounds; this
is exact and requires no changes to the atom basis. The LAMMPS data file
then carries an `xy xz yz` tilt line instead of a plain orthogonal box,
and the extended-XYZ `Lattice=` string carries the full 3×3 matrix
(row-major) instead of just the diagonal. See
`examples/cif/tini_cif_triclinic_single.yaml` (monoclinic, xz tilt) and
`examples/cif/tini3_cif_hexagonal_single.yaml` (hexagonal, xy tilt) for worked
examples.

## The four subcommands

- **`grainsmith info --sg <N> [--setting S] [--site "x,y,z"]`** — inspect a space
  group: point group, and (with `--site`) the Wyckoff orbit of a position. No
  config needed.
- **`grainsmith validate <config.yaml>`** — run only the cheap gates G1 (schema +
  cross-field rules) and G2 (crystal symmetry round-trip). Use it to check a
  config before a long run.
- **`grainsmith generate <config.yaml> [--jobs N] [--max-rss GB]`** — the full
  pipeline. `--jobs` parallelizes the per-grain atom fill and the GB-curvature
  analysis stage (`0` = all cores available to the process — the
  cgroup/affinity mask, e.g. a SLURM `--cpus-per-task` allocation), and also
  sets the thread count for the overlap-removal neighbour search and gate G7's
  fresh query; outputs are bit-identical for every `--jobs` value (see
  *Determinism*).
  `--max-rss` sets a soft WARN ceiling (GB) for the best-effort total-RSS
  monitor (needs the optional `psutil` extra); it only logs a warning and never
  changes the output (see *Performance notes*).
- **`grainsmith verify <outdir> [--strict] [--expect-version X.Y.Z] [--quiet]`**
  — check an output directory against its own `MANIFEST.txt`: recompute every
  file digest, re-derive the config hash from the shipped
  `resolved_config.yaml`, and confirm the recorded grainsmith version is the one
  cryptographically bound to the run. Exits 0 (OK), 1 (mismatch) or 2 (cannot
  verify). See *Determinism and provenance* below.

`grainsmith --version` prints the installed version and exits.

### Running on a cluster (SLURM)

grainsmith is a **single-node** program: one run = one node = one task, and
all parallelism is `--jobs` worker processes inside that task — nothing spans
nodes, so never request `--nodes > 1` or `--ntasks > 1` for a single run
(each extra task would run the full pipeline again into the same output
directory). For the best parallelism on a single node set
`--jobs = grains.number` — the fill stage hands each grain its own worker
process, so every grain fills simultaneously — and pair `--cpus-per-task`
with it (more cores than grains speed up only the per-boundary-pair analysis
stage):

```bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10   # == grains.number
export OMP_NUM_THREADS=1
grainsmith generate config.yaml --jobs $SLURM_CPUS_PER_TASK
```

The numba kernels are disk-cached (`cache=True`) inside the installed
package, so a single small warm-up run after installation compiles them once
for every node sharing the filesystem. `examples/hpc/` carries ready-made
cluster-scale configs, a submit-ready `job.sbatch` template, and the full
cluster guide (`examples/hpc/README.md`: setup + warm-up, `sbatch`, the
`salloc` interactive case, job arrays, pitfalls).

## How a run is built (the pipeline stages)

A `generate` run is a fixed sequence of stages; `summary.csv` records the timing
of each:

1. **crystal** — expand the Wyckoff sites to a unit cell; verify symmetry (G2).
2. **seeding** — place grain seeds (RSA with `min_seed_distance`; optional Lloyd
   relaxation).
3. **tessellation** — build the grain geometry: flat Voronoi, a volume-fitted
   power/Laguerre diagram (if `size_distribution` is set, via SDOT — G11), a
   domain-warp curved variant (G6), a level-set `perturbed_distance` variant
   for genuinely self-affine boundaries (G19/G20; see `docs/geometry.md` and
   `docs/physics.md` §5b), or an imported voxel field. Gates G3–G6.
4. **orientation** — assign each grain an orientation (random, fixed, fiber,
   explicit list, or ODF components); optional MDF assignment annealing (G12).
5. **fill** — fill each grain with atoms by tiling the rotated lattice into its
   cell (the `owns()` torus-tiling rule — no wrap-and-dedupe). Parallelized by
   `--jobs`.
6. **overlap** — remove atoms that overlap across boundaries (G7).
7. **doping** (optional) — substitutional/interstitial dopants placed after
   overlap removal (`doping:`), gates G17/G18. Interstitial sublattices come
   from a preset (SG 225/229/194) or explicit `sites.coords` for any of the
   230 space groups — coords are expanded to their full symmetry orbit by
   default (`expand_orbit`), so one representative per orbit suffices.
8. **analysis** — grains, boundaries, GB character, CSL, statistics (G8, G9,
   G16 + G21 when `analysis.gb_curvature` is on, and the publication outputs).
9. **write** — all output files + the SHA-256 `MANIFEST.txt`.

For the full dataflow, the data objects passed between stages, and the module
dependency graph, see [`architecture.md`](architecture.md). For how to choose and
tune the grain geometry in stage 3, see [`geometry.md`](geometry.md). To drive
this pipeline from Python instead of the CLI, see [`developer.md`](developer.md).

## Gates are the user interface

Every physics constraint is a **QA gate**. Hard gates abort with an actionable
message (e.g. G6 → "reduce amplitude / increase correlation_length"); warn gates
complete the run but flag a quality concern in `summary.csv`. When something
"doesn't work", the gate message names the check, the measured value, and the
fix. The full catalogue with thresholds and remedies is `docs/gates.md`.

## Determinism and provenance

A fixed `seed.value` makes the **scientific payload** byte-for-byte reproducible
— including for any `--jobs` value (each grain draws from its own deterministic
RNG sub-stream, so the worker count never changes the result). `seed.mode:
entropy` draws a fresh seed and records it in `summary.csv` so the run can be
reproduced later.

### What is byte-identical, and when

| | same machine, default | same machine, `SOURCE_DATE_EPOCH` set | different machine, same versions | different library versions |
|---|---|---|---|---|
| data CSVs (`grains`, `boundaries`, `vertices`, `statistics`, `mdf`, `gb_curvature`) | identical | identical | last-bit differences possible | no guarantee |
| files with a provenance header (`polycrystal.data`, `.extxyz`, `resolved_config.yaml`, `METHODS.md`, gnuplot bundle, …) | identical except the header timestamp | identical | last-bit differences possible | no guarantee |
| `summary.csv` | differs (`timings` + `memory` rows) | differs (`timings` + `memory` rows only) | `environment` section differs | no guarantee |
| `microstructure.json` | differs (timestamp) | identical | `environment` section differs | no guarantee |
| `MANIFEST.txt` | differs (hashes the above) | differs in its `summary.csv` line only | differs | no guarantee |
| `run.log` | differs (it is a log) | differs | differs | differs |

`SOURCE_DATE_EPOCH` is the standard [reproducible-builds][rb] environment
variable: an integer count of seconds since the Unix epoch, interpreted as UTC.
When it is set, grainsmith stamps that instant into every provenance header
instead of the wall clock. An unparseable value is an error, not a silent
fallback.

It does **not** freeze the two row families in `summary.csv` that are a
measurement of the run itself: the wall-clock stage timings (`timings,*_s`)
and the peak driver RSS (`memory,peak_driver_rss_gb`, `memory,rss_warn_tripped`).
Those keep their real values, so `summary.csv` differs between two otherwise
identical runs — and, because `MANIFEST.txt` hashes `summary.csv`, so does that
one manifest line. Every other manifest digest, and every scientific output
byte, is identical. The trade is deliberate: an archived run stays honest about
how long it took and how much memory it needed, which is exactly what a reader
of an archive wants to know, and the checks that must ignore it say so
explicitly — see `tests/test_end_to_end.py::test_byte_identical_rerun_source_date_epoch`
and `tools/golden_hashes.py`'s `double_run`, which compare `summary.csv` with
those rows stripped and `MANIFEST.txt` with that line dropped.

[rb]: https://reproducible-builds.org/docs/source-date-epoch/

### Cross-machine scope (honest)

Byte-identity across machines needs three things to match, not one: the library
versions, the BLAS build behind NumPy, and the CPU architecture. The atom-fill
hot path multiplies matrices through BLAS `dgemm`, so switching OpenBLAS ↔ MKL ↔
a generic reference BLAS, or x86 ↔ ARM, shifts a measurable fraction of atom
coordinates by about one unit in the last place. The *physics* is unaffected —
grain ownership, atom counts, species and every gate value are identical — but
the digests in `MANIFEST.txt` are not. A different NumPy/SciPy/spglib **version**
is a stronger break: `numpy.random.Generator` explicitly disclaims cross-version
bit-stream stability, so a version bump can change the structure entirely.

Two things make this auditable rather than a leap of faith. First, every run
records what produced it: the `environment` section of `summary.csv` (and the
`environment` object of `microstructure.json`) carries the BLAS name and version,
the OS, kernel release and CPU architecture, the C library, whether the optional
numba kernel was active, and both the requested and effective `--jobs` values.
Second, `constraints-repro.txt` in the repository root pins the exact dependency
versions grainsmith was developed against:

```bash
python -m venv .venv-repro && . .venv-repro/bin/activate
pip install -c constraints-repro.txt -e ".[test]"
```

That file pins versions; it cannot pin the Python interpreter or the BLAS a
NumPy build links against, and it says so in its own header comments.

### The version is bound to the hash

Each output records two SHA-256 digests:

- **`config_sha256`** — of the canonical resolved config alone. It identifies
  the *input*: two grainsmith versions run on the same config produce the same
  value.
- **`provenance_sha256`** — of a payload that **contains the grainsmith version
  string** alongside `config_sha256`. It identifies *input + producer*, so an
  output's claim to have been "produced with 1.1.0" is cryptographically bound
  to its digest and cannot be edited after the fact without detection.

Both are stored full-length (64 hex) in `MANIFEST.txt`, `summary.csv` and
`microstructure.json`; the one-line provenance header at the top of every
commented file shows the first 12 hex of each:

```
# grainsmith 1.1.0 | 2026-09-01T12:00:00Z | seed=555666 | config sha256=fd1c4ecb04ac | provenance sha256=9a3f0c1d2e4b
```

### Checking an archived run

```bash
grainsmith verify ./out
```

`verify` works from the shipped files alone — no config, no network, no re-run.
It re-reads `MANIFEST.txt`, recomputes the SHA-256 of every listed file,
re-derives `config_sha256` from the shipped `resolved_config.yaml`, recomputes
`provenance_sha256` from that digest plus the recorded version string, and
checks that the version agrees across `MANIFEST.txt`, `resolved_config.yaml`,
`summary.csv` and `microstructure.json`. It exits **0** when everything checks
out, **1** on any mismatch, and **2** when the directory cannot be verified at
all (no `MANIFEST.txt`, or an unreadable one).

Files left over in a reused output directory are reported as a warning, not a
failure, because `MANIFEST.txt` deliberately lists only the files *this* run
wrote; `--strict` promotes that warning to a failure. `--expect-version X.Y.Z`
turns "was this produced by version X?" into a hard check.

## Reading the outputs

Start with `summary.csv` (gates, composition, timings, versions). Then:

- **per-grain / per-boundary**: `grains.csv`, `boundaries.csv`.
- **statistics for the paper**: `statistics.csv` (with the estimator-bias notes),
  `microstructure.json` (machine-readable), `METHODS.md` (auto methods text).
- **texture**: `odf_mtex.txt` (→ MTEX), `mdf.csv` (misorientation histogram).
- **GB curvature** (`analysis.gb_curvature` only): `gb_curvature.csv`
  (per-sample H/K), plus `curvature_hist.plt` when gnuplot output is enabled.
- **dopants** (`doping.dopants` non-empty only): `doping.csv` (per-grain,
  per-dopant placement counts and achieved shell/bulk fractions),
  `doping_profile.csv` (dopant–GB distance proxigram) and its plot script
  `doping_profile.plt` (gnuplot bundle) — the visual check that GB
  segregation landed where requested.
- **visualize**: `gnuplot view.plt`, or OVITO on `polycrystal.data` / `.extxyz`.

Every file and column is documented in `docs/outputs.md`.

## Performance notes

Single-core SDOT fits scale with the grain count (measured): small N is
seconds; large N (≳ 1000 grains) can take noticeably longer on a given machine
— these are honest, hardware-dependent numbers, not a fixed target. Atom fill
parallelizes well to ~8 workers (measured); the warp tessellation, overlap
removal and I/O are serial and dominate very large runs. The flagship
9.2-million-atom example peaks around 13 GB RAM — budget **≥ 16 GB** for
production sizes (see `docs/faq.md`).

**Memory guards.** Two independent layers protect against out-of-memory
failures — keep them separate. (1) A *per-allocation* hard guard refuses any
single allocation whose estimated size exceeds the effective §13 budget — the
per-grain lattice fill (`atoms/fill.py`), the analysis voxel grid
(`tessellation/voxel.py`), and the imported-label EDT margin
(`tessellation/voxel_import.py`). The nominal budget is
`runtime.memory_limit_gb` (config-file only, **16 GB by default**, raised from
a hard-coded 8 GB); the EFFECTIVE budget actually enforced is
`min(runtime.memory_limit_gb, detected physical RAM)` — a hard clamp, so
setting the knob above the machine's own RAM does not raise the ceiling.
Physical RAM is detected best-effort (`psutil` if installed, else
`os.sysconf`, `/proc/meminfo`, the Windows `GlobalMemoryStatusEx` API, or
macOS's `sysctl hw.memsize`); if every path fails, the configured value is
used unclamped and a warning is logged. The resulting error names whichever
bound actually tripped: when physical RAM is the smaller, binding bound, it
states that the machine does not have enough RAM, names the detected RAM, and
advises reducing the box size / grain count or moving to a bigger machine
(raising `runtime.memory_limit_gb` would just be re-clamped); when the
configured value binds with RAM to spare, it advises raising
`runtime.memory_limit_gb` instead. Only the fill stage runs inside a `--jobs
N` worker pool — there, up to N grains fill concurrently, each allocating its
own grid, so the machine needs headroom for N times the limit in the worst
case; the voxel-grid and voxel-import guards each build exactly one whole-box
grid on the single-threaded driver and are unaffected by `--jobs`. (2) Because
the cumulative resident set of the driver can exceed any single allocation
(the flagship 9.2-million-atom run peaked ~13.1 GB while every per-allocation
estimate stayed under the — then lower — per-allocation guard in effect at the
time; a measurement, not today's 16 GB default, which now happens to
numerically coincide with it), an optional *total-RSS* monitor (`--max-rss
GB`, needs `pip install ".[monitor]"`) samples the process RSS on a
background thread and logs a WARN when the ceiling — explicit, or an
auto-fraction of physical RAM — is crossed. The monitor is best-effort and
**WARN-only**: it never aborts a run and never alters a single output byte
(RSS is non-deterministic, so the peak is reported via the log and `RunResult`
only, never a reproducible output file). The resolved, RAM-clamped §13 budget
is itself machine-dependent, for the same reason, and is likewise never
written to a reproducible output file — only your own
`runtime.memory_limit_gb` is recorded, in `resolved_config.yaml`.
