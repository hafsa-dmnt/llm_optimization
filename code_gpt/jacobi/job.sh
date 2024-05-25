#!/bin/bash
#SBATCH --job-name=laplace_omp
#SBATCH --output=laplace_omp_%j.out
#SBATCH --error=laplace_omp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00

# Exécuter laplace_seq avec différentes valeurs de OMP_NUM_THREADS
for threads in 1 2 4 8 16 20; do
    echo "Running laplace_seq with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./laplace_seq
done
