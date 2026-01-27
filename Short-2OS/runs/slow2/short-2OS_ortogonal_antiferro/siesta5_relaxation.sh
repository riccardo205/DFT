#!/bin/sh
#
#SBATCH --job-name="siesta5"
#SBATCH --output=siesta.out
#SBATCH --error=siesta.err
#SBATCH --partition=compute-p1
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3968M
#SBATCH --account=research-as-qn

module load 2024r1 openmpi
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-w6w5qi5/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-h2uag4k/Core/

module load siesta

srun siesta < short-2OS_ortogonal_antiferro.fdf


