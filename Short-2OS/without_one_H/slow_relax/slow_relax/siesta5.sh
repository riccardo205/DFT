#!/bin/sh
#
#SBATCH --job-name="siesta5"
#SBATCH --output=short-2OS.out
#SBATCH --error=short-2OS.err
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-as-qn

module load 2024r1

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/rconte1/siesta5

srun siesta < short-2OS.fdf

conda deactivate

