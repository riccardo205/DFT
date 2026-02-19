#!/bin/sh
#
#SBATCH --job-name=%EXECUTABLE%
#SBATCH --output=%OUTPUTFILE%
#SBATCH --error=%ERRORFILE%
#SBATCH --partition=compute-p1
#SBATCH --time=%WALLTIME%
#SBATCH --ntasks=%NTASKS%
#SBATCH --cpus-per-task=%CPUSPERTASK%
#SBATCH --mem-per-cpu=%MEMPERCPU%
#SBATCH --account=research-as-qn

module load 2024r1 openmpi
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-w6w5qi5/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-h2uag4k/Core/

module load siesta

srun siesta < %INPUTFILE%