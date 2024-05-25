#!/bin/bash

#SBATCH --job-name=kmeans
#SBATCH --output=kmeans_%j.out
#SBATCH --error=kmeans_%j.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

module load openmpi/4.0.3

mpirun -np 4 ./km 1000 10 5 0.1 12345