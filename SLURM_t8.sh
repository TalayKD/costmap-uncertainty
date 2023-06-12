#!/bin/bash

# SLURM Resource Parameters

#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 4-00:00 # D-HH:MM
#SBATCH -p a100-gpu-full
#SBATCH --gres=gpu:2
#SBATCH --mem=124G
#SBATCH --job-name=t_8
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tkondhor@andrew.cmu.edu

# Executable
EXE=/bin/bash

singularity run --nv /data2/datasets/tkondhor/images/pytorch.sif sh run_talay_t8.sh