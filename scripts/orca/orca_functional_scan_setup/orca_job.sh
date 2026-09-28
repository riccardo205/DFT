#!/bin/bash

#SBATCH --job-name=orca
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00
#SBATCH --account=research-as-qn

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch orca_job.sh calculation.inp"
    exit 2
fi

INPUT_FILE="$1"
BASENAME="${INPUT_FILE%.inp}"
OUTPUT_FILE="${BASENAME}.out"

export OPENMPI_DIR=/home/rconte1/software/openmpi-4.1.8
export ORCA_DIR=/home/rconte1/software/orca-6.1.1

export PATH="${OPENMPI_DIR}/bin:${ORCA_DIR}:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_DIR}/lib:${ORCA_DIR}:${LD_LIBRARY_PATH:-}"

export OMPI_MCA_btl="^openib"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR="${SLURM_SUBMIT_DIR}"
TMP_DIR="/tmp/${USER}/orca_${SLURM_JOB_ID}"

mkdir -p "${TMP_DIR}"

cleanup() {
    exit_code=$?

    echo
    echo "=========================================="
    echo "Copying files back to:"
    echo "${RUN_DIR}"
    echo "=========================================="

    cp -a "${TMP_DIR}/." "${RUN_DIR}/" || true
    rm -rf "${TMP_DIR}" || true

    exit "${exit_code}"
}

trap cleanup EXIT

echo "=========================================="
echo "ORCA DelftBlue job"
echo "=========================================="
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : $(hostname)"
echo "Slurm tasks     : ${SLURM_NTASKS}"
echo "Input           : ${INPUT_FILE}"
echo "Output          : ${OUTPUT_FILE}"
echo "ORCA executable : ${ORCA_DIR}/orca"
echo "MPI executable  : $(which mpirun)"
echo
mpirun --version | head -n 1
echo "=========================================="
echo

cp "${RUN_DIR}/${INPUT_FILE}" "${TMP_DIR}/"
cp "${RUN_DIR}/geometry.xyz" "${TMP_DIR}/"

cd "${TMP_DIR}"

echo "Starting ORCA at $(date)"

"${ORCA_DIR}/orca" "${INPUT_FILE}" > "${OUTPUT_FILE}"

echo "ORCA command finished at $(date)"
