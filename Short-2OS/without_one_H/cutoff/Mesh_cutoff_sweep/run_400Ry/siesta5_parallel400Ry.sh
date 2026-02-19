#!/bin/sh
#
#SBATCH --job-name="siesta5_003"
#SBATCH --output=siesta_400Ry.out
#SBATCH --error=siesta_400Ry.out
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --mem-per-cpu=3968M
#SBATCH --account=research-as-qn

module load 2024r1

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/rconte1/siesta5

mpirun -np 4 siesta < short-2OS_400Ry.fdf

conda deactivate

