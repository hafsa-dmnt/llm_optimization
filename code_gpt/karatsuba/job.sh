#!/bin/bash
#SBATCH --job-name=karatsuba
#SBATCH --output=karatsuba.out
#SBATCH --error=karatsuba.err
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

module load intel/compiler/2021.1.1
module load pgi/
module load openmpi/4.1.4.2-intel-ilp64

mpirun -np 4 ./karatsuba karatsuba.in