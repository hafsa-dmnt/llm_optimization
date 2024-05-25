#!/bin/bash
#SBATCH --job-name=fib_mpi
#SBATCH --output=fib_mpi.out
#SBATCH --error=fib_mpi.err
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=compute

module load mpi
srun ./fib 100000
