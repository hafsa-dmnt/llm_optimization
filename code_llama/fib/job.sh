#!/bin/bash

#SBATCH --job-name=fib_mpi
#SBATCH --output=fib_mpi_%j.out
#SBATCH --error=fib_mpi_%j.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=00:10:00

module load mpi/openmpi-4.0.3

mpirun -np 4 ./fib_mpi $1