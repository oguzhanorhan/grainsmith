# Running grainsmith on an HPC cluster (SLURM)

This note is the practical companion to the three configs in this folder.
It applies to any SLURM cluster (partition names below are placeholders —
e.g. `orfoz` on TRUBA).

## The model in one paragraph

grainsmith is a **single-node** program: one run = one node = one task
(`--nodes=1 --ntasks=1`), and all parallelism is `--jobs` worker processes
inside that task. Nothing spans nodes. The best-parallelism operating point
is **`--jobs = grains.number`** with `--cpus-per-task` set to match — the
fill stage hands each grain its own worker, so every grain fills
simultaneously. More cores than grains speed up only the per-boundary-pair
analysis stage; on fewer cores, workers pick grains up dynamically. Outputs
are bit-identical for every `--jobs` value.

## One-time setup (login node)

```bash
git clone https://github.com/oguzhanorhan/grainsmith
cd grainsmith                              # on the shared filesystem
python -m venv ~/venvs/grainsmith
source ~/venvs/grainsmith/bin/activate
pip install ".[perf]"                      # [perf] = numba, from the clone

# Warm the numba disk cache ONCE — compiles every kernel into the venv,
# so every later job on every node skips JIT compilation entirely:
grainsmith generate examples/basics/b2_nial_flat.yaml
```

## Batch jobs (`sbatch`)

`job.sbatch` in this folder is a ready template — submit any config with:

```bash
sbatch examples/hpc/job.sbatch examples/hpc/1_cu_flat.yaml
```

Its core (see the file for the commented version):

```bash
#!/bin/bash
#SBATCH --partition=<partition>       # e.g. orfoz on TRUBA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10            # == grains.number of the config
#SBATCH --time=00:30:00
export OMP_NUM_THREADS=1
source ~/venvs/grainsmith/bin/activate
grainsmith generate "$1" --jobs "$SLURM_CPUS_PER_TASK"
```

Notes:

- **`--cpus-per-task` is the only size knob.** Match it to the config's
  `grains.number` (10 for the files here) and pass the same number as
  `--jobs` — `$SLURM_CPUS_PER_TASK` keeps the two in lockstep.
- `--jobs 0` also works: it resolves to the cores *available to the
  process* (the cgroup/affinity mask, i.e. your allocation — not the whole
  node). The explicit form above is simply easier to audit.
- `export OMP_NUM_THREADS=1` is cheap insurance against numpy's BLAS
  spawning its own threads inside each worker.
- Memory: these ~4×10⁶-atom runs peak near ~1 GB driver RSS — any normal
  node allocation is fine. `--max-rss` (a WARN-only monitor) reads the
  node's *physical* RAM, not your job's memory cgroup limit, so pass an
  explicit value on shared nodes if you want meaningful warnings.

## Interactive sessions (`salloc`)

An `salloc` allocation behaves exactly like a batch job — same process
model, same `--jobs` rules:

```bash
salloc --partition=<partition> --nodes=1 --ntasks=1 --cpus-per-task=10 --time=01:00:00
# The salloc shell may still be ON THE LOGIN NODE (site-dependent).
# Launch the run onto the allocated compute node with srun:
srun grainsmith generate examples/hpc/1_cu_flat.yaml --jobs 10
```

Two things to watch:

- **Always launch with `srun` (or ssh to the node)** unless your site
  configures salloc to drop you onto the compute node — otherwise the run
  executes on the login node.
- **Keep `-n 1`.** `srun` without an explicit task count inherits the
  allocation's; if you allocated more than one task, `srun grainsmith ...`
  would start that many *independent full pipelines* writing into the same
  output directory.

The first interactive run pays the numba JIT cost only if you skipped the
warm-up above; with a warm cache it starts immediately.

## Many runs: job arrays, not bigger jobs

More nodes never make one run faster. To use them, run *independent*
configs (parameter sweeps, seeds) as separate jobs — each with its own
`output.directory`:

```bash
#SBATCH --array=0-2
CFGS=(1_cu_flat.yaml 2_cu_curved.yaml 3_cu_fractal.yaml)
grainsmith generate "examples/hpc/${CFGS[$SLURM_ARRAY_TASK_ID]}" \
    --jobs "$SLURM_CPUS_PER_TASK"
```

Concurrent array tasks share the numba cache safely (writes are atomic;
a cold cache costs each first task one compile, nothing worse).

## Pitfalls, summarized

| don't | because |
|---|---|
| `--nodes > 1` or `--ntasks > 1` for one run | each task runs the FULL pipeline again into the same `output.directory` |
| `srun -n 4 grainsmith ...` | same duplicate-pipeline trap, interactively |
| assume `salloc` put you on the compute node | on many sites the salloc shell stays on the login node — use `srun` |
| point two jobs at one `output.directory` | outputs overwrite each other; give every run its own directory |
