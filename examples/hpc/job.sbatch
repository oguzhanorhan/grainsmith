#!/bin/bash
# grainsmith single-node SLURM template — submit any config with:
#   sbatch examples/hpc/job.sbatch examples/hpc/1_cu_flat.yaml
# See examples/hpc/README.md for the full guide (salloc, arrays, pitfalls).
#
#SBATCH --job-name=grainsmith
#SBATCH --partition=<partition>       # e.g. orfoz on TRUBA
##SBATCH --account=<account>          # uncomment if your site requires it
#SBATCH --nodes=1                     # grainsmith is single-node: never >1
#SBATCH --ntasks=1                    # one task = one pipeline
#SBATCH --cpus-per-task=10            # == grains.number of the config
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Insurance against numpy/BLAS spawning threads inside each worker:
export OMP_NUM_THREADS=1

source ~/venvs/grainsmith/bin/activate

# --jobs == --cpus-per-task, kept in lockstep via SLURM's own variable.
grainsmith generate "${1:?usage: sbatch examples/hpc/job.sbatch <config.yaml>}" \
    --jobs "$SLURM_CPUS_PER_TASK"
