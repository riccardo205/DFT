#!/bin/sh
#
#SBATCH --job-name="siesta5_parallel"
#SBATCH --output=short-2OS.out
#SBATCH --error=short-2OS.err
#SBATCH --partition=compute-p1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-as-qn

module load 2024r1

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/rconte1/siesta5

mpirun -np 4 siesta < short-2OS.fdf

conda deactivate

