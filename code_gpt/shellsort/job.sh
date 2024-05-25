#!/bin/bash
#SBATCH --job-name=shellsort_omp
#SBATCH --output=shellsort_omp_%j.out
#SBATCH --error=shellsort_omp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00

for threads in 1 2 4 8 16 20; do
    echo "Running shellsort with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./shellsort
done
