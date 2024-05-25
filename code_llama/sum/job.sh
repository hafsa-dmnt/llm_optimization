#!/bin/bash

#SBATCH --job-name=cuda_job
#SBATCH --output=cuda_job_%j.out
#SBATCH --error=cuda_job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

source /home/hdemnati/projetm2/env.sh
module load cuda/11.4
./problem