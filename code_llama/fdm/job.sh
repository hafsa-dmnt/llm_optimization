#!/bin/bash

#SBATCH --job-name=mdf
#SBATCH --output=mdf_%j.out
#SBATCH --error=mdf_%j.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

module load openmpi/4.0.3

mpirun -np 4 ./mdf