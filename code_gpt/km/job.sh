#!/bin/bash
#SBATCH --job-name=kmeans_mpi
#SBATCH --output=kmeans_mpi.out
#SBATCH --error=kmeans_mpi.err
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=compute

module load mpi
srun ./km 10000 10 5 0.01 42
