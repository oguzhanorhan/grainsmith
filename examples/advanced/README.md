# advanced — production-scale examples

Two tiers live here, and the difference is not only the box edge.

| tier | prefix | box | atoms | where it runs |
|---|---|---|---|---|
| **advanced** | `adv_*` | 228–255 Å | ~10⁶ | a 32 GB laptop |
| **huge** | `huge_*` | 1500 Å | 2.3–2.9 × 10⁸ | a cluster node, ≥ 64 GB |

The `adv_*` files are the reference constructions: each is literature-grounded,
names its sources in its own header, and has been **run** — the atom counts and
gate results quoted in [`../README.md`](../README.md) are measured, not
estimated. Start there.

The `huge_*` files are box-expanded siblings of the three `adv_*` cases whose
statistics are most obviously sample-size limited. They are **validated but not
run** in this repository: at ~10⁸ atoms a single run is a cluster job, so the
YAML is verified for syntax and for every constraint the config layer can check,
and the resource numbers in each header are computed from the code rather than
measured.

| file | expands | what the extra scale buys |
|---|---|---|
| `huge_pdau_perturbed.yaml` | `adv_pdau_perturbed.yaml` | self-affine boundary roughness measured over 4 octaves at 256 grains instead of 28 |
| `huge_cu_rolling_twin.yaml` | `adv_cu_rolling_twin.yaml` | a four-component ODF and a misorientation-angle histogram built from ~1700 boundaries, not a few hundred |
| `huge_cufe_composite.yaml` | `adv_cufe_composite.yaml` | a 60/40 phase split whose granularity is set by the grain volumes — 256 grains land far closer than 48 |

## Scaling a box is not a multiplication

Three parameters change non-linearly with the edge, and each header works its
own case. Read this before writing a `huge_*` variant of your own.

**The analysis grid caps out.** `VOXEL_GRID_MAX` is 384 voxels per axis, so
`h_field = edge / 384` grows linearly with the box: 3.906 Å at 1500 Å. A
self-affine field cannot represent a wavelength below `2 · h_field`, so
`l_min` has a floor that rises with the box — 7.81 Å here, against the 5.53 Å
(`2 · d_nn`) the small-box examples use. **This is not caught by `grainsmith
validate`**: the grid is only resolved when the tessellation is built, so a
too-small `l_min` passes validation and raises minutes into the run. The
amplitude ceiling, by contrast, *is* caught at resolve time.

**Grain count is a memory parameter.** `fill_grain` enumerates a lattice box
around each grain whose volume scales as `box³ / n_grains`. Holding a small
example's grain count at 1500 Å puts tens of GB into *every* fill worker at
once. The `huge_*` files use 256 grains, which keeps the per-grain grid at
~0.4 GB transient — so `--jobs 32` needs ~13 GB between the workers — while
still landing `d_eq = 29.3 nm`, inside the nanocrystalline regime these
examples are about.

**The amplitude ceiling moves with the grain size.** The seed-containment guard
caps the perturbation at `PERTURBED_DISTANCE_SAFETY · min_seed_distance /
(2 · ETA_CLIP)`, and `min_seed_distance: auto` is `r_ws`, which grows with
`(box³/n)^(1/3)`. At 1500 Å and 256 grains that ceiling is 12.21 Å, against
4.19 Å in the 246 Å sibling (28 grains → `r_ws` = 50.26 Å).

## Memory: what the numbers actually are

`runtime.memory_limit_gb` bounds any **single** allocation. It is not a cap on
total resident memory, and the peak is never the atom arrays alone: the
post-fill sort holds the old arrays, the new arrays and an index
simultaneously.

| | atom arrays (36 B/atom) | sort peak (80 B/atom) |
|---|---|---|
| `huge_pdau_perturbed` (2.26 × 10⁸) | 8.1 GB | ~18 GB |
| `huge_cu_rolling_twin` (2.86 × 10⁸) | 10.3 GB | ~23 GB |
| `huge_cufe_composite` (2.87 × 10⁸) | 10.3 GB | ~23 GB |

Budget a node with ≥ 64 GB and expect 35–45 GB resident. The driver also
divides `runtime.memory_limit_gb` by the worker count when it decides how many
fill workers actually fit, so lowering it throttles the pool rather than
failing the run.

## Running

```bash
grainsmith validate examples/advanced/huge_pdau_perturbed.yaml
grainsmith generate examples/advanced/huge_pdau_perturbed.yaml --jobs 32
```

Outputs are bit-identical for every `--jobs` value. See
[`../hpc/job.sbatch`](../hpc/job.sbatch) for a SLURM submission sketch; the
`hpc/` group itself is a smaller (360 Å, ~3.9 × 10⁶ atom) cluster tier aimed at
throughput studies rather than single large structures.

Two of the `adv_*` files use a weighted tessellation (`additive_weights`,
`anisotropic`) whose per-grain fill boxes are large — run **those two with a
capped worker count (`--jobs 6`)**; the measured resource note is in each
file's header. The other four run fine at `--jobs 0` (all cores).
