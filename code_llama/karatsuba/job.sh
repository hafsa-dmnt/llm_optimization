#!/bin/bash

#SBATCH --job-name=karatsuba_mpi
#SBATCH --output=karatsuba_mpi_%j.out
#SBATCH --error=karatsuba_mpi_%j.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=00:10:00

module load intel/compiler/2021.1.1
module load pgi/
module load openmpi/4.1.4.2-intel-ilp64

mpirun -np 4 ./karatsuba_mpi karatsuba.in