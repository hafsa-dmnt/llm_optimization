#!/bin/bash
#SBATCH --job-name=mdf_mpi
#SBATCH --output=mdf_mpi.out
#SBATCH --error=mdf_mpi.err
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

source /home/hdemnati/projetm2/env.sh
mpirun -np 4 ./mdf
