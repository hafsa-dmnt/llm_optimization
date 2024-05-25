#!/bin/bash

#SBATCH --job-name=password_bf_cuda
#SBATCH --output=password_bf_cuda_%j.out
#SBATCH --error=password_bf_cuda_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

source /home/hdemnati/projetm2/env.sh
module load cuda/11.4

./password_bf afa345bc5ced1b9bf90a3ff76d8ac111