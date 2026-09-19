# Developer guide

This page is for users who want to **call grainsmith from Python** or **extend
it** — add a tessellation backend, an orientation sampler, or an output writer
— and for anyone working ON the codebase itself. It documents the repository
layout, the programmatic entry point, the data-object contracts, the public
API of each scientific subpackage, the determinism contract and the parallel
pattern that implements it, the extension points, the test suite's
conventions (including the drift-proof documentation tests), and the dev
install.

Read [architecture.md](architecture.md) first for the pipeline and module map;
this page assumes you know the stage chain.

## 1. Repository layout

```
src/grainsmith/            the engine (importable as `grainsmith`)
  cli.py                    generate / validate / verify / info / ui subcommands
  provenance.py             version-bound digest, SOURCE_DATE_EPOCH, environment capture
  verify.py                 `grainsmith verify`: MANIFEST re-hash + version-binding checks
  pipeline.py                run() orchestrator: the stage chain, summary.csv assembly
  qa.py                      the QAGates registry + every gate_gN_* evaluator
  constants.py                physics/gate thresholds and tolerances live here
                              as the single source for those constants
  rng.py, seeding.py          RNGBundle (deterministic child streams), RSA + Lloyd seeding
  phases.py, memory.py, errors.py   multiphase LPT partition, the total-RSS
                              monitor, the exception hierarchy
  config/                     pydantic schema, YAML -> RunConfig resolution,
                              schema introspection (drives config_reference.md)
  crystal/                    lattice geometry, Wyckoff expansion via spglib,
                              space-group verification, CIF ingestion
  orientation/                 quaternion algebra, disorientation, texture/MDF sampling
  tessellation/                 every grain-partition backend (flat, power, warp,
                              weighted/anisotropic, imported-voxel via voxel_import,
                              perturbed, single) plus voxel.py's shared VoxelGrid
                              GB-membership helper and the base.Tessellation contract
  atoms/                       per-grain lattice fill, PBC-aware overlap removal,
                              dopant insertion
  analysis/                    grain/boundary reports, publication statistics, GB curvature
  io/                          every output writer + the LAMMPS parse-back (gate G10)
  ui/                          FastAPI backend for an interactive UI ("grainsmith
                              studio") planned for a future release; not enabled
                              in this version (architecture.md §4)
tests/                      pytest suite: one test_<module>.py per src module
                              (roughly), plus test_docs.py, test_end_to_end.py,
                              test_fuzz.py; conftest.py holds the shared fixtures
tools/                      docs generators — gen_config_reference.py, build_docs_html.py
docs/                       this documentation set: the Markdown under docs/ is
                              the source of truth; docs/html/ and
                              docs/config_reference.md are generated (§9)
examples/                   worked configs, grouped by topic (basics/,
                              crystallography/, cif/, texture/, grain_geometry/,
                              alloys_multiphase/, doping/, self_affine_gb/,
                              voxel_import/, advanced/), plus examples/assets/
                              for shared CIF files and imported label fields
pyproject.toml              package metadata + the optional-extras map (§10)
```

See [architecture.md §4](architecture.md#4-package-structure) for the full
subpackage responsibility table and the measured dependency-edge counts
between subpackages; this section is only the filesystem map.

## 2. Programmatic entry point

The whole pipeline is one call:

```python
from grainsmith.config.resolve import load_config
from grainsmith.pipeline import run

config = load_config("examples/grain_geometry/cu_lognormal_sizes.yaml")   # -> RunConfig
result = run(config)                                         # -> RunResult
print(result.seed, result.n_generated, result.n_deleted)
print([f.name for f in result.files])                        # files written
```

`run()` has the signature:

```python
def run(config: RunConfig, jobs: int = 1, max_rss_gb: float | None = None) -> RunResult
```

- **`config`** — a validated `RunConfig` (build it with
  `config.resolve.load_config(path)` from YAML, or `resolve_config(raw_dict)` from
  a dict).
- **`jobs`** — worker processes for the per-grain fill stage, the
  GB-curvature analysis stage, and `analyze_boundaries`'s per-(i, j)-pair
  geometry/character/CSL computation on curved/voxel backends (flat stays
  serial regardless — its exact-face lookup is too cheap to be worth a
  worker pool), and the thread count for the overlap-removal neighbour
  search and gate G7's fresh query — all named in the CLI's `--jobs` help.
  `1` = serial; `0` = all cores; `n` = n workers. The same knob also drives
  one further stage not named in that one-line help: the LAMMPS/extxyz
  writers' row-chunk formatting (`io.lammps.write_lammps`,
  `io.xyz.write_extxyz`) for large atom counts. The result is
  **bit-identical** regardless of `jobs` in every one of these stages
  (determinism is preserved across parallelisation — see §5–§6 below for the
  contract and the house pattern that makes this true).
- **`max_rss_gb`** — optional total-RSS WARN ceiling for the soft, best-effort
  monitor (see `memory.MemoryMonitor`): it samples the driver's resident set on
  a background thread and logs a WARNING when the ceiling is crossed. It never
  raises, never aborts the run, and never alters an output byte. The hard,
  *per-allocation* guard that does abort is a separate mechanism — the §13
  budget `min(runtime.memory_limit_gb, detected physical RAM)`, enforced in
  `atoms/fill.py`, `tessellation/voxel.py` and `tessellation/voxel_import.py`.

`run()` both **writes all outputs** to `config.output.directory` and **returns**
the in-memory `RunResult`, so you can either consume the files or work with the
objects directly.

### Building a config in code

```python
from grainsmith.config.resolve import resolve_config
raw = {
    "seed": {"mode": "fixed", "value": 42},
    "box": {"lengths": [60.0, 60.0, 60.0]},
    "grains": {"number": 12},
    "crystal": {
        "space_group": {"number": 225},
        "lattice": {"a": 3.615},
        "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
    },
}
config = resolve_config(raw)     # runs the full pydantic validation + resolution
```

`resolve_config` applies the schema **and** the cross-field resolution rules
(e.g. inferring the power backend when a size distribution is present). It raises
`ConfigError` on an invalid config — this is gate **G1**.

## 3. Data-object contracts

### `RunResult`

The single return value of `run()`. All 18 fields (`pipeline.RunResult`):

| Field | Type | Meaning |
|-------|------|---------|
| `config` | `RunConfig` | the resolved config that produced this run |
| `seed` | `int` | the seed actually used (regenerate with this) |
| `outdir` | `Path` | resolved output directory |
| `atoms` | `AtomBlock` | filled model, sorted by `(grain, generation order)` |
| `tess` | `Tessellation` | the grain partition object |
| `quats` | `(N,4) ndarray` | per-grain orientation quaternions |
| `grain_reports` | `list[GrainReport]` | per-grain analysis |
| `boundary_reports` | `list[BoundaryReport]` | per-boundary analysis |
| `gates` | `qa.QAGates` | all gate results (see `docs/gates.md`) |
| `timings` | `dict[str, float]` | wall-clock seconds per stage |
| `n_generated` | `int` | atoms placed before overlap removal |
| `n_deleted` | `int` | atoms removed by the overlap gate |
| `d_nn` | `float` | ideal-crystal nearest-neighbour distance (Å) |
| `files` | `list[Path]` | every file written |
| `phase_of` | `ndarray \| None` | grain → phase index (multiphase runs) |
| `statistics` | `dict \| None` | publication statistics sections |
| `peak_rss_bytes` | `float \| None` | peak resident memory, if monitored |
| `provenance` | `io.common.Provenance \| None` | full config and producer-bound digests, seed and timestamp |

### `CrystalData`

Crystal-stage output (`pipeline.CrystalData`), one per phase:

| Field | Type | Meaning |
|-------|------|---------|
| `cellpar` | `crystal.cell.CellPar` | the six lattice parameters |
| `A` | `(3,3) ndarray` | cell matrix, **columns** = `a₁,a₂,a₃` (Å) |
| `basis` | `crystal.spacegroup.Basis` | symmetry-expanded atom basis |
| `sym_quats` | `(Nsym,4) ndarray` | proper point-group rotations |
| `dataset` | `dict` | spglib verification subset (gate G2) |
| `d_nn` | `float` | nearest-neighbour distance (Å) |
| `rho_atom` | `float` | atom number density (atoms/Å³) |
| `density_g_cm3` | `float` | mass density (NaN if a mass is unknown) |

### `AtomBlock`

The filled model (`atoms.fill.AtomBlock`). Array attributes, all row-aligned:
`pos` `(M,3)` positions in Å, `species` `(M,)` element symbols, `grain` `(M,)`
grain id, and (optionally) `gb_margin` `(M,)` signed distance to the nearest
boundary. `concatenate()` merges per-grain blocks. **Note the attribute is
`.pos`, not `.positions`.**

## 4. Public API by subpackage

Every symbol below is importable from its module (e.g. `from
grainsmith.tessellation.sdot import fit_power_weights`). Signatures are the real
ones; only the most useful entry points per subpackage are listed — see the
source or `help()` for the full set.

### `crystal` — lattice & symmetry

```python
cell.cell_matrix(a, b, c, alpha, beta, gamma) -> (3,3) ndarray
cell.validate_cellpar(family, params) -> CellPar
spacegroup.symmetry_ops(hall) -> (rotations, translations)
spacegroup.expand_wyckoff(sites, rots, trans) -> Basis
verify.verify_spacegroup(cellpar, basis, requested_number, symprec=…) -> dict   # gate G2
```

### `orientation` — orientation algebra & sampling

```python
quaternion.axis_angle_to_quat(axis, angle_deg) -> (4,) ndarray     # angle in DEGREES
quaternion.quat_to_matrix(q) -> (3,3) ndarray                       # active: v_lab = R @ v_crystal
quaternion.quat_to_bunge(q) -> (phi1, Phi, phi2)                    # Bunge ZXZ, MTEX/EBSD convention
misorientation.disorientation(q_i, q_j, sym_quats) -> Disorientation
samplers.random_uniform(n, rng) -> (n,4)
samplers.fiber_texture(n, crystal_axis, sample_direction, spread_deg, rng, cell_matrix=None) -> (n,4)
samplers.odf_components(n, components, rng, cell_matrix=None) -> (n,4)
```

### `tessellation` — grain-partition backends

```python
sdot.fit_power_weights(seeds, box_lengths, periodic, target_volumes,
                       vol_tol=1e-3, max_iter=30, centroidal_iterations=0) -> SDOTResult
sdot.sample_target_volumes(dist_type, n, box_volume, rng, sigma_log=0.35, volumes=None) -> (n,)
warp.synthesize_grf(grid_shape, correlation_length, box_lengths, rng,
                    spectrum='gaussian', hurst=0.8, l_min=8.0, l_max=60.0) -> ndarray
warp.estimate_hurst(field, box_lengths, l_min, l_max, n_bins=16) -> float
```

```python
perturbed.box_count_dimension(mask, eps_min_px=3, ...) -> float | None    # G20's D_b estimator
```

Backend classes: `FlatTessellation`, `PowerTessellation`, `WarpTessellation`,
`WeightedTessellation`, `AnisotropicTessellation`,
`VoxelTessellation` (`tessellation/voxel_import.py` — `voxel.py` itself
holds only the shared `VoxelGrid` GB-membership helper, not a second
backend), `PerturbedDistanceTessellation`, `SingleCrystalTessellation` —
all subclasses of `base.Tessellation`.

### `atoms` — fill & overlap

```python
overlap.resolve_cutoff(spec, d_nn=None) -> float           # parses "0.85*d_nn" etc.
fill.fill_grains(tess, frac_basis, basis_species, basis_occupancy, A, quats,
                 box_lengths, periodic, rngs, jobs=1) -> list[AtomBlock]
overlap.remove_overlaps(atoms, tess, cutoff, policy, periodic, box_lengths,
                        jobs=1) -> (AtomBlock, OverlapLedger)   # gate G7
doping.run_doping(doping_cfg, atoms, tess, rng_bundle, crystal, quats, n_grains,
                  box_lengths, periodic) -> (AtomBlock, DopingResult)  # gates G17/G18
doping.resolve_sites(sites, sg_number, what) -> (M,3) ndarray  # preset name or explicit coords
doping.INTERSTITIAL_PRESETS: dict[str, tuple[int, ndarray]]  # preset -> (space group, frac coords)
```

### `analysis` — reports & statistics

```python
grains.grain_volumes(tess, box_lengths) -> (N,) ndarray
grains.analyze_grains(tess, quats, A, box_lengths, ...) -> list[GrainReport]
boundaries.analyze_boundaries(tess, quats, sym_quats, A, box_lengths, periodic,
                              csl=False, character=True, jobs=1, ...) -> list[BoundaryReport]
statistics.lognormal_fit(diameters) -> dict     # + KS test (Lilliefors caveat)
curvature.analyze_curvature(tess, box_lengths, periodic, vg=None, jobs=1) -> CurvatureResult
curvature.attach_curvature(reports, result) -> None   # fills H/K fields of BoundaryReport in place
curvature.global_curvature_rows(result) -> list[(section, key, value)]  # summary.csv rows
```

### `io` — output writers

```python
lammps.write_lammps(path, atoms, box_lengths, periodic, atom_style, provenance,
                    vacuum=0.0, masses=None, jobs=1) -> dict
lammps.parse_lammps(path) -> dict               # round-trip parse-back (gate G10)
xyz.write_extxyz(path, atoms, box_lengths, periodic, provenance, ...) -> None
reports.write_grains_csv / write_boundaries_csv / write_summary_csv(...)
publication.write_microstructure_json(path, config, provenance, gates, ...) -> None
```

### `config` — schema & resolution

```python
resolve.load_config(path) -> RunConfig          # YAML file -> validated config
resolve.resolve_config(raw) -> RunConfig         # dict -> validated config  (gate G1)
resolve.config_sha256(config) -> str             # full config hash (summary.csv, MANIFEST.txt)
resolve.config_sha12(config) -> str              # its first 12 hex (in output headers)
introspect.field_specs(model) -> list[FieldSpec] # drives the config-reference generator
```

## 5. The determinism contract

grainsmith's central promise (manual.md, "Determinism and provenance") is
code-enforced, not just documented: a fixed `seed.value` reproduces the
scientific payload byte-for-byte for ANY `--jobs`/`jobs=` value, on the same
platform and dependency versions. Whole-file identity additionally requires a
frozen clock (`SOURCE_DATE_EPOCH`), because every commented output carries the
run's UTC timestamp in its provenance header; `manual.md`'s "Determinism and
provenance" table states which files are identical under which conditions.
Three mechanisms, at three different levels of the stack, make this true.

### 5.1 One seed, many independent streams

`np.random.*` module-level calls are forbidden everywhere in the engine (a
project-wide guardrail — see `rng.py`'s module docstring); every stochastic
draw goes through an `RNGBundle` (`rng.make_rng(seed)`). `make_rng` spawns
**seven** independent PCG64 streams from one `numpy.random.SeedSequence(seed)`,
in a fixed order (`STAGE_NAMES`: `seeding, orientation, fields, occupancy,
sizes, mdf, doping`) — reordering that tuple, or inserting a new stage before
its end, would silently change every later stream and break reproducibility
of existing configs, which is why the module docstring pins "add new stages
at the end only".

Two of those streams (`occupancy`, `doping`) are themselves the PARENT of a
further per-grain fan-out: `RNGBundle.occupancy_streams(n)` /
`.doping_streams(n)` derive grain `i`'s own stream by extending the parent
`SeedSequence`'s `spawn_key` with `(i,)` — a STATELESS construction (plain
`SeedSequence.spawn()` counts children as a side effect, so calling it twice
would not reproduce the same streams; extending `spawn_key` by hand has no
such state). A grain's stream therefore depends only on `(master seed, grain
id)` — never on `--jobs`, execution order, or how many times the function is
called — which is what makes the parallel fill (§6) reproducible in the
first place.

### 5.2 Ordered reassembly after a process pool

Parallelizing over grains or boundary pairs changes only WHEN each unit of
work runs, never what it produces (each is a pure function of its own
inputs) or where its result lands: every parallel driver in the engine
(`atoms.fill.fill_grains`, `analysis.curvature.analyze_curvature`,
`analysis.boundaries.analyze_boundaries`, the LAMMPS/extxyz writers) submits
work through `ProcessPoolExecutor.map`, which preserves input order in its
output, and reassembles results positionally (`fill_grains`: grain order;
`analyze_curvature`/`analyze_boundaries`: `tess.adjacency()` order) rather
than in whatever order workers happen to finish. `--jobs` therefore never
feeds back into the DATA — only into how many OS processes computed it, and
how long that took.

The one place parallelism touches data ORDER rather than just execution
order is neighbour-pair DISCOVERY in `atoms.overlap`/
`qa.gate_g7_min_distance`: `jobs > 1` swaps `scipy.spatial.KDTree.query_pairs`
for a thread-parallel `query_ball_point` two-pass scan. `overlap.py`'s module
docstring proves the two routes return the identical pair SET (both run the
same `d² <= cutoff²` comparison in the same cKDTree C machinery) and both are
sorted into one canonical lexicographic order before any deletion logic sees
them — so the deletion sequence, and hence the final atom set, does not
depend on `jobs`, only the discovery wall time does.

### 5.3 Numba kernels are IEEE-exact, never an approximation

The `perf` extra (`pip install ".[perf]"`, `numba>=0.59`) enables fused
`owns()`/`margin()`/`grain_of()` kernels at three sites, all following the
same soft-dependency pattern:

- `tessellation/_local_owns.py` — the shared owns()-tie-break kernel behind
  `FlatTessellation` and `PowerTessellation`; always uses numba when it is
  importable (no environment override outside the equality tests' own
  `use_numba=` argument).
- `tessellation/weighted.py` — `WeightedTessellation`/`AnisotropicTessellation`'s
  owns()/margin() kernels.
- `tessellation/perturbed.py` — `PerturbedDistanceTessellation`'s three
  fused kernels (K1 `owns`, K2 `margin`, K3 `grain_of`), all built on one
  shared `_eval_candidate`/`_interp_eta` primitive.

The latter two read `GRAINSMITH_NO_NUMBA` (`1`/`true`/`yes`/`on`,
case-insensitive) to force the plain-NumPy path even when numba is installed.
Every kernel is compiled with `@numba.njit(cache=True)`; a few (in
`weighted.py`) also pass `nogil=True`, which only releases the GIL during
execution and has no bearing on exactness. What matters for the
IEEE-exactness argument is what's absent instead: no `fastmath` (which
would let LLVM reorder floating-point operations and change rounding) and
no `prange` (parallel numba would reintroduce a reduction-order dependency
of exactly the kind §5.2 is designed to avoid) — so the compiled kernel
evaluates the same operations in the same order as
the NumPy reference it replaces, term by term (see `perturbed.py`'s
`_interp_eta` docstring: "same trilinear 8-term sum in the SAME textual term
order").

Numba is a SOFT dependency everywhere it is used: the NumPy implementation is
kept byte-for-byte as the reference — never deleted or left to bit-rot —
specifically so it can serve as the always-available fallback AND as the
other half of an equality test. Every numba site is regression-pinned by
tests that force each path explicitly (monkeypatching the module's
`_USE_NUMBA` flag) and assert the SAME output bytes come out either way
(`tests/test_owns_kernel.py`, `tests/test_perturbed_kernel.py`) — including
engineered TIE cases (candidates exactly equidistant, at a periodic-boundary
half-`L`, etc.) where a naive reduction order would be the first thing to
disagree (§8 below has more on this test family).

### 5.4 How the contract is tested

Three layers, deliberately independent:

1. `tests/test_end_to_end.py::test_byte_identical_rerun` — two runs of the same
   config from two working directories, compared after normalizing timestamps
   and dropping the wall-clock `timings` rows. This is the wall-clock-mode
   guarantee.
2. `tests/test_end_to_end.py::test_byte_identical_rerun_source_date_epoch` — the
   same two runs with `SOURCE_DATE_EPOCH` set, compared with **no** normalization
   at all, `MANIFEST.txt` included and `run.log` the only exclusion. This is the
   strict form.
3. `tools/golden_hashes.py` + `tests/test_reproducibility.py` — a committed
   fixture config, a recorded digest set, and a same-environment double run. The
   golden check degrades to a warning (never a failure) when the environment it
   was recorded in no longer matches the one it runs in, because byte-identity
   across BLAS builds and CPU architectures is not something grainsmith can
   promise; its bootstrap and update procedure is in the tool's module docstring.

## 6. The parallel pattern

Three stages — the per-grain fill (`atoms/fill.py`), the per-boundary-pair
GB-curvature analysis (`analysis/curvature.py`), and the per-boundary-pair
geometry/character/CSL analysis on curved/voxel backends
(`analysis/boundaries.py`) — all parallelize the SAME shape of embarrassingly
parallel problem (one independent unit of work per grain, or per adjacent
grain pair) with the SAME house pattern. Understand one and you understand
all three; a future stage that fans out over grains or boundary pairs should
follow it too:

1. **`jobs <= 1` short-circuits to the untouched serial path.** Every driver
   checks `jobs <= 1` (or `jobs <= 1 or len(units) <= 1`) FIRST and, on that
   branch, runs a plain Python loop/comprehension calling the same pure
   per-unit function it would otherwise ship to a worker — no
   `ProcessPoolExecutor` is even imported on this path. `jobs=1` (the
   default) therefore stays exactly as cheap, and exactly as behaved, as
   before any of this parallel machinery existed.
2. **A module-level `_XXX_CTX` global + `_init_xxx_worker` initializer**
   ships the (potentially large) shared, read-only inputs — the
   `Tessellation` object, the crystal basis, `h_vec`, per-phase symmetry/cell
   tables — to each worker process ONCE, via `ProcessPoolExecutor`'s
   `initializer=`/`initargs=`, instead of once per task. Because
   `ProcessPoolExecutor` workers are separate OS processes (POSIX fork or
   Windows spawn), `_FILL_CTX`/`_CURVATURE_CTX`/`_BOUNDARY_CTX` are
   process-local state, not shared mutable module state in the sense the
   codebase otherwise forbids (the same "no module-level `np.random`"
   guardrail §5.1 mentions) — each worker sets its OWN copy once and reads
   it for every task that lands on it.
3. **A thin per-task function** (`_fill_one`, `_curvature_pair_task`,
   `_boundary_pair_task`) reads the worker's `_XXX_CTX` plus its own task
   arguments (a grain id + that grain's own rng stream; a boundary pair +
   its sample points; …) and calls the SAME pure function the serial path
   calls — `fill_grain`, `_pair_curvature`, `_pair_report_fields` — so the
   worker and the serial loop provably compute the identical thing, not two
   hand-kept-in-sync implementations.
4. **`pool.map` in a fixed, meaningful order** — grain id order for the
   fill, `tess.adjacency()` order for curvature/boundaries — because
   `ProcessPoolExecutor.map` preserves input order in its output regardless
   of which worker finishes first, so **positional reassembly**
   (`list(pool.map(...))`, or a lazy generator pulled in lockstep for the
   curvature/boundaries drivers, which additionally skip empty-input pairs
   without breaking that lockstep) is all that is needed to recover the
   same grain-ordered/adjacency-ordered result the serial path would have
   built. No result ever needs to carry its own index back to the parent —
   the ORDER work was submitted in is enough.
5. **`max_workers=min(jobs, n_units)`** avoids spinning up worker processes
   that would never receive a task (more workers than grains/pairs).

The parallelism is therefore an execution-order optimization ONLY: every
worker computes the exact bytes the serial loop would have computed for that
same unit, and the parent reassembles them into the exact order the serial
loop would have produced. This is what makes §5's determinism contract
compatible with `--jobs` at all, and it is the same shape the LAMMPS/extxyz
writers (`io/lammps.py`, `io/xyz.py`) use for their row-chunk formatting, and
that `atoms/overlap.py` uses at the THREAD (not process) level — scipy's own
`workers=` argument — for its neighbour-pair discovery.

## 7. Extension points

grainsmith is designed so that the whole fill/overlap/analysis stack depends only
on abstract interfaces. The three most useful extension points:

### 7.1 A new tessellation backend

Subclass `tessellation.base.Tessellation` and implement its interface — the fill,
overlap, and analysis stages will then work with your backend unchanged. The
required contract is five abstract methods and two abstract properties:

| Member | Signature | Purpose |
|--------|-----------|---------|
| `grain_of` | `(X:(N,3)) -> (N,) int32` | grain index for each point (periodic images folded to home id) |
| `owns` | `(X:(N,3), i:int) -> (N,) bool` | does the **home** cell of grain `i` own each unwrapped point? (the fill uses this to place each torus point exactly once) |
| `margin` | `(X:(N,3), i:int) -> (N,) float64` | signed distance to grain `i`'s boundary (+ inside), Å |
| `adjacency` | `() -> list[(i,j)]` | adjacent grain pairs, `i < j` |
| `bounding_radius` | `(i:int) -> float` | bounding-sphere radius at `seeds[i]` (bounds the fill's lattice enumeration) |
| `seeds` | property `-> (N,3)` | seed positions |
| `n_grains` | property `-> int` | grain count |

The distinction between `grain_of` and `owns` is the key to grainsmith's
image-exact periodic fill: `grain_of` folds periodic images of a grain onto its
home id, while `owns` distinguishes the home cell from its images so the home
cells tile the torus exactly once — no wrap-and-deduplicate step, valid for any
incommensurate lattice orientation.

Two further methods are OPTIONAL overrides — a subclass that skips them still
works correctly, just without the optimization each one unlocks:

- **`cell_vertices_rel(i) -> (V,3) | None`** — cell vertices relative to
  `seeds[i]` for a tight oriented fill bounding box; return `None` (the base
  default) for curved/voxel backends, which falls back to the
  `bounding_radius` sphere.
- **`gb_shell_lower_bound(pos, grain, cutoff, workers=1) -> (N,) bool | None`**
  — a certified pre-filter mask that lets `atoms.overlap.remove_overlaps`
  and gate G7 search a certified SUBSET of atoms instead of the full
  population before building a neighbour-search tree. The base
  implementation always returns `None` ("no certified bound; search
  everything"), which is always correct, just not the fastest option.

  **The 1-Lipschitz certificate.** A backend may only return a real mask
  when its membership "distance" `d(x)` is **1-Lipschitz in `x`**, i.e.
  `|d(x) - d(y)| <= |x - y|`. Under that condition the *replica-aware
  margin* `m_rep(x) = (second-best generalized distance to a periodic
  replica seed − best) / 2` satisfies `m_rep(x) <= |x - y|` for any pair
  `(x, y)` within `cutoff`, so the shell `{x : m_rep(x) <= cutoff}` is a
  certified SUPERSET of every true overlap-pair endpoint — searching only
  the shell finds the identical pair set as searching everything, just from
  a smaller tree (`tessellation/_shell_knn.py`'s `certified_knn_shell_mask`
  implements this k-NN route; `FlatTessellation` and the additive-weights
  `WeightedTessellation` use it, since plain-Euclidean and constant-offset
  additive distances are both 1-Lipschitz). A SECOND, independent
  certificate needs no Lipschitz bound at all: for a backend whose cells are
  exact convex polytopes with stored planar faces, the true margin equals
  the minimum distance to any of the cell's own supporting hyperplanes — an
  EXACT computation, not an approximation — which is what
  `PowerTessellation` uses instead of inheriting `Flat`'s k-NN shortcut,
  because the power (Laguerre) distance `‖x−c‖² − w` is quadratic (its
  Lipschitz constant grows without bound as `‖x−c‖` grows), so the
  1-Lipschitz argument does not carry over to nonzero weights even though
  the polytope-face argument still does.

  Backends whose distance is neither 1-Lipschitz nor backed by an exact
  convex-polytope margin correctly decline the optimization by overriding
  the method to always return `None`, each with its own documented reason:
  `AnisotropicTessellation` (Lipschitz constant `1/s_min` for the smallest
  metric semi-axis, `> 1` for any non-degenerate aspect ratio),
  `WarpTessellation` (the warped margin is a documented PROXY, not proven
  1-Lipschitz in the unwarped coordinate the cutoff is measured in),
  `PerturbedDistanceTessellation` (Lipschitz constant `1 + amplitude ·
  |grad η|`, `> 1` whenever the perturbation is nonzero — the entire point
  of that backend), `SingleCrystalTessellation` (no other-grain boundary to
  overlap against on a non-periodic axis), and `VoxelTessellation`
  (the margin is a voxel-center EDT estimate with a documented `~h` bias,
  not an exact distance). A new backend is under no obligation to implement
  this override — the base `None` always works.

### 7.2 A new orientation sampler

Add a function to `orientation.samplers` returning an `(n, 4)` array of unit
quaternions (scalar-first, `w ≥ 0`), then wire it into `OrientationConfig` in
`config.schema`. Existing samplers (`random_uniform`, `fiber_texture`,
`odf_components`, `fixed_orientation`, `from_list`) are the templates; reuse the
`quaternion` helpers so your output matches the active-rotation convention
(`v_lab = R(q) @ v_crystal`).

### 7.3 A new output writer

Add a `write_*` function to a module under `io/`, using
`io.common.atomic_writer(path)` for crash-safe writes and accepting a
`Provenance` object for the reproducibility header. Register it in the write
section of `pipeline.run()` (near the other `write_*` calls) and expose a
toggle in `OutputConfig`.

## 8. Testing conventions

```bash
pip install ".[test]"        # pytest, pytest-cov, ase, httpx, psutil
pytest                       # full suite
pytest -k bunge              # a focused subset
```

The suite lives under `tests/`, roughly one `test_<module>.py` per
`src/grainsmith/<module>.py`, plus cross-cutting files: `test_end_to_end.py`
(full pipeline runs against shipped example configs), `test_fuzz.py`
(randomized config generation), and `test_docs.py` (§9). Session-scoped
fixtures in `conftest.py` resolve a handful of example configs by name (e.g.
the B2 NiAl flat-boundary and FCC Cu curved-warp examples) so tests don't
hardcode paths independently. The suite runs entirely from `tests/`; a
handful of tests exercise optional local tooling and skip gracefully when
it isn't present in the checkout.

Key test families: crystallography round-trips (`test_spacegroup`,
`test_cif`), the Bunge/MTEX convention pin (`test_quaternion`), CSL/Σ tables
(`test_analysis`), the SDOT Jacobian/convergence checks (`test_power`), and
the overlap gate (`test_overlap`).

**Determinism/`jobs`-invariance batteries.** Anywhere the codebase claims an
output is independent of `jobs` or of the numba/NumPy dispatch (§5, §6), a
test proves it with exact byte equality rather than a numeric tolerance —
typically `array.tobytes()` compared directly, or a written file's SHA-256
hash. Representative examples: `test_curvature.py`'s
`test_analyze_curvature_jobs_parallel_matches_serial*` family compares
`PairCurvature.H/K/areas/points.tobytes()` between `jobs=1` and `jobs>1`
runs (including a degenerate-mask case and a mixed-empty-pair case);
`test_overlap.py`'s `test_determinism_jobs1_vs_jobs8_synthetic_ties`
and its `.pos/.species/.grain.tobytes()` comparisons do the same for
overlap removal; `test_owns_kernel.py` and `test_perturbed_kernel.py`
monkeypatch each module's `_USE_NUMBA` flag to force both dispatch paths on
the same input and assert identical output. Several of these also run a
true end-to-end `generate` and byte-compare the resulting
`polycrystal.data` file across `--jobs` values
(`test_e2e_curved_rerun_byte_identical_across_jobs`) — the same guarantee
manual.md's "Determinism" section promises the CLI user, pinned at the
file level, not just the in-memory array level.

**Engineered tie cases.** Several kernels have an explicit, documented
tie-break rule (e.g. "first-minimum wins, strict `<`" on scan order — see
§5.3 and the K1/K3 docstrings in `perturbed.py`). Rather than hoping a
randomized test happens to hit a tie, these are pinned with HAND-BUILT
inputs engineered so specific candidates are exactly equidistant, or land
exactly on a periodic-boundary half-`L` `np.rint` rounding boundary:
`test_perturbed_kernel.py`'s R1–R3 cases (`test_r1_margin_min_image_half_l_np_rint_tie`,
`test_r3_engineered_tie_four_cases`, parametrized over "replica index below
home_idx" / "above home_idx" combinations) and `test_owns_kernel.py`'s
`test_owns_kernel_tie_rule` / `test_sqrt_rounding_collision_tie`. A test
that only exercises typical, well-separated geometry would never catch a
reduction-order regression in the tie path — these exist specifically
because that path is otherwise invisible to coverage.

**Env-gated slow validations.** A few tests replay an expensive, realistic
scenario (a full curved system at production scale) rather than a synthetic
unit case, and are skipped by default so the ordinary `pytest` run stays
fast: `test_owns_kernel.py`'s numba-vs-NumPy mask-equality replay against
the shipped paper Fig. 2 curved system is gated by
`GRAINSMITH_RUN_FIG2_VALIDATION=1` (registered as the `fig2_validation`
pytest marker in `conftest.py`), and `test_perturbed_kernel.py` has the
analogous `GRAINSMITH_RUN_PERTURBED_FIG2_VALIDATION=1` gate for
`perturbed_distance`. Set the environment variable (any of `1`/`true`/
`yes`/`on`) to opt in when touching the numba kernels or their dispatch
logic; CI does not run them by default.

## 9. Docs tooling

Two files under `docs/` are **generated**, not hand-written, and a test
(`tests/test_docs.py`) fails the build if either has drifted from its
generator:

### The drift-proof documentation tests

`docs/config_reference.md` is **generated** from the pydantic schema by
`tools/gen_config_reference.py`, and `test_config_reference_is_not_stale`
asserts the committed file matches what the generator produces from the
current schema. If you change `config/schema.py`, regenerate it:

```bash
python -m tools.gen_config_reference        # rewrite docs/config_reference.md
```

Then the drift test passes again. **Never hand-edit `config_reference.md`** — the
schema is the source of truth, and the test will fail on any manual change.
The same test file also checks that every `G<n>` id referenced in `qa.py`
appears in `docs/gates.md`, and that every configurable and fixed output
filename appears in `docs/outputs.md` — so a new gate or a new output file
that never made it into the prose docs fails CI too, not just a schema
change.

### Rebuilding the HTML docs mirror

`docs/html/` is a themed HTML mirror of the Markdown pages under `docs/`,
built by `tools/build_docs_html.py` (a pure-Python Markdown engine; no
external binary needed, though `pandoc` is an optional fallback). The
Markdown under `docs/` is the source of truth — only regenerate the mirror
after editing a page:

```bash
pip install ".[docs]"                  # markdown
python -m tools.build_docs_html         # rewrite docs/html/
```

`test_html_docs_are_not_stale` re-renders every page in memory (pinned to
the `markdown` engine specifically, since that is what produced the
committed files — a `pandoc`-rendered comparison would flag drift that
isn't there) and byte-compares it against the committed `docs/html/*.html`.
Both generators are deterministic (no timestamps, no filesystem-order
dependence), so a mismatch is always genuine drift, never CI noise.

## 10. Dev install & QA gates as the runtime interface

For working on the engine itself, an editable install skips repackaging on
every change:

```bash
pip install -e . --no-deps        # into the project's own env; --no-deps
                                   # avoids re-resolving already-installed
                                   # dependencies on every reinstall
pip install ".[test,docs,perf,dev]"   # pytest suite, HTML docs builder,
                                       # numba speed-up, ruff + mypy
```

The optional-extras map lives in `pyproject.toml`: `test` (pytest,
pytest-cov, ase, httpx, psutil), `docs` (markdown, for §9), `perf` (numba,
for §5.3), `dev` (ruff, mypy), plus the feature extras a user install would
also reach for — `cif` (ase), `mesh` (scikit-image), `import` (h5py),
`monitor` (psutil), `bench` (matplotlib).

**QA gates are the runtime interface, not just a validation afterthought.**
Every physical constraint the engine enforces surfaces through one registry:
`qa.QAGates` accumulates one `GateResult(gate, passed, measured, message)`
per check into a single ordered list, written verbatim into `summary.csv`'s
`gates` section regardless of which gates a given run exercised. Two verbs
distinguish severity: `QAGates.require(result)` raises
`errors.QAGateError` immediately on a failing HARD gate, aborting the run
with the gate's own actionable message; `QAGates.record(result)` only
appends, for WARN gates that should complete the run but still flag a
quality concern. The CLI's one-line summary (`OK: <atoms> atoms, <grains>
grains, <boundaries> boundaries; N/N QA gates passed; outputs in <dir>
(<s> s).`, or `WARN: ...; k/N QA gates passed [WARN gates: g1, g2]; ...`
when any gate failed) is generated straight from this same list
(`cli._cmd_generate`) — so a caller working programmatically gets the exact
same information via `RunResult.gates.results()` that the CLI prints, just
as `GateResult` objects instead of a formatted string. See
[gates.md](gates.md) for the full G1–G26 catalogue: what each one checks,
its threshold constant (collected in `constants.py` as the single source
for those constants, per the module-layout convention in §1), and what to
do when it fires.

## 11. Where to go next

- **Pipeline & module map** → [architecture](architecture.md)
- **Grain-geometry backends** → [geometry](geometry.md)
- **Every config field** → [configuration reference](config_reference.md)
- **Physics & conventions** → [physics](physics.md)
- **The QA gate catalogue** → [gates](gates.md)
