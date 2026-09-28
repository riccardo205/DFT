#!/bin/bash

#SBATCH --job-name=orca
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00

#SBATCH --account=research-as-qn

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch orca_job.sh calculation.inp" >&2
    exit 2
fi

INPUT_FILE="$1"
BASENAME="${INPUT_FILE%.inp}"
OUTPUT_FILE="${BASENAME}.out"

module purge
module load 2025
module load gcc

export OPENMPI_DIR=/home/rconte1/software/openmpi-4.1.8
export ORCA_DIR=/home/rconte1/software/orca-6.1.1

export PATH="${OPENMPI_DIR}/bin:${ORCA_DIR}:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_DIR}/lib:${ORCA_DIR}:${LD_LIBRARY_PATH:-}"

# Single-node jobs: avoid OpenFabrics warnings / unnecessary network transport.
export OMPI_MCA_btl="^openib"

# Prevent hidden threading on top of the MPI ranks.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR="${SLURM_SUBMIT_DIR}"

# Safety check: calculations must run from /scratch, never from /home.
case "${RUN_DIR}" in
    /scratch/${USER}/*)
        ;;
    *)
        echo "ERROR: SLURM_SUBMIT_DIR is not below /scratch/${USER}/" >&2
        echo "Current directory: ${RUN_DIR}" >&2
        exit 2
        ;;
esac

cd "${RUN_DIR}"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: input file not found: ${RUN_DIR}/${INPUT_FILE}" >&2
    exit 2
fi

if [ ! -f "geometry.xyz" ]; then
    echo "ERROR: geometry.xyz not found in ${RUN_DIR}" >&2
    exit 2
fi

echo "============================================================"
echo "ORCA job information"
echo "============================================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : $(hostname)"
echo "Run directory   : ${RUN_DIR}"
echo "Input           : ${INPUT_FILE}"
echo "Output          : ${OUTPUT_FILE}"
echo "Slurm tasks     : ${SLURM_NTASKS}"
echo "Memory / CPU    : ${SLURM_MEM_PER_CPU:-unknown} MB"
echo "ORCA            : $(which orca)"
echo "MPI             : $(which mpirun)"
echo "Fortran         : $(which gfortran)"
echo

mpirun --version | head -n 1

echo
echo "OpenMPI Fortran support:"
ompi_info | grep -E \
"Fort mpif.h:|Fort use mpi:|Fort use mpi_f08:|Fort compiler:" \
| head -n 10

echo
echo "MPI libraries used by ORCA LEANSCF:"
ldd "${ORCA_DIR}/orca_leanscf_mpi" \
    | grep -E "mpi|open-rte|open-pal" \
    || true

echo
echo "Scratch filesystem before ORCA:"
df -h "${RUN_DIR}" || true
df -i "${RUN_DIR}" || true
du -sh "${RUN_DIR}" 2>/dev/null || true

echo
echo "Starting ORCA at $(date)"
echo "============================================================"

# ORCA sometimes reports an internal error while the top-level driver still
# returns exit code 0. Therefore we check BOTH the shell exit code and the
# presence of "ORCA TERMINATED NORMALLY" in the output.
set +e

"${ORCA_DIR}/orca" "${INPUT_FILE}" > "${OUTPUT_FILE}" 2>&1

ORCA_EXIT_CODE=$?

set -e

echo
echo "============================================================"
echo "ORCA driver exit code: ${ORCA_EXIT_CODE}"
echo "Finished ORCA command at $(date)"
echo "============================================================"

if grep -Fq "ORCA TERMINATED NORMALLY" "${OUTPUT_FILE}" \
    && [ "${ORCA_EXIT_CODE}" -eq 0 ]; then

    echo "ORCA TERMINATED NORMALLY"

    echo
    echo "Removing ORCA temporary files from successful run..."

    # Delete disposable ORCA temporary matrices only after a successful run.
    find "${RUN_DIR}" -maxdepth 1 -type f \
        \( -name '*.tmp' -o -name '*.tmp.*' \) \
        -print -delete || true

    echo
    echo "Final run-directory usage:"
    du -sh "${RUN_DIR}" 2>/dev/null || true
    df -h "${RUN_DIR}" || true

    exit 0
fi

echo >&2
echo "============================================================" >&2
echo "ERROR: ORCA DID NOT TERMINATE NORMALLY" >&2
echo "ORCA driver exit code: ${ORCA_EXIT_CODE}" >&2
echo "Output file: ${RUN_DIR}/${OUTPUT_FILE}" >&2
echo "Temporary files are being KEPT for diagnosis." >&2
echo "============================================================" >&2

echo
echo "Run-directory usage after failure:"
du -sh "${RUN_DIR}" 2>/dev/null || true
df -h "${RUN_DIR}" || true
df -i "${RUN_DIR}" || true

# Ensure Slurm marks this job as FAILED even if the ORCA driver returned 0.
if [ "${ORCA_EXIT_CODE}" -ne 0 ]; then
    exit "${ORCA_EXIT_CODE}"
else
    exit 1
fi
