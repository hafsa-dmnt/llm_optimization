#!/bin/bash
#SBATCH --job-name=dijkstra_omp
#SBATCH --output=dijkstra_omp_%j.out
#SBATCH --error=dijkstra_omp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00

for threads in 1 2 4 8 16 20; do
    echo "Running dijkstra with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./dijkstra 50 20 1000000
done
