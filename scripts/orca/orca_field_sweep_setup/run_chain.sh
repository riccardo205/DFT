#!/bin/bash

#SBATCH --job-name=orca_field
#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00

#SBATCH --account=research-as-qn

# Runs one continuation chain of ORCA single points, sequentially.
#
# Usage (working directory = a <structure>/<state> folder on /scratch):
#     sbatch run_chain.sh <branch>        e.g. F0, axis_p, axis_m, perp_p, perp_m
#
# chain_<branch>.txt lists "point previous_point" in execution order. Before a
# point runs, previous_point/<state>.gbw is copied to point/guess.gbw (the
# input has MORead + %moinp "guess.gbw"). Points that already terminated
# normally are skipped, so resubmitting resumes the chain. The chain stops at
# the first failed point: continuing from a bad density would defeat the
# purpose of continuation.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run_chain.sh <branch>" >&2
    exit 2
fi

BRANCH="$1"
STATE_DIR="$(pwd)"   # set by sbatch --chdir (SLURM_SUBMIT_DIR would be the sbatch caller dir)
STATE="$(basename "${STATE_DIR}")"
CHAIN_FILE="${STATE_DIR}/chain_${BRANCH}.txt"
LOG="${STATE_DIR}/chain_${BRANCH}.log"

case "${STATE_DIR}" in
    /scratch/${USER}/*) ;;
    *)
        echo "ERROR: ${STATE_DIR} is not below /scratch/${USER}/" >&2
        exit 2
        ;;
esac

[ -f "${CHAIN_FILE}" ] || { echo "ERROR: ${CHAIN_FILE} not found" >&2; exit 2; }

module purge
module load 2025
module load gcc

export OPENMPI_DIR=/home/rconte1/software/openmpi-4.1.8
export ORCA_DIR=/home/rconte1/software/orca-6.1.1
export PATH="${OPENMPI_DIR}/bin:${ORCA_DIR}:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_DIR}/lib:${ORCA_DIR}:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_btl="^openib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | ${STATE_DIR} | branch ${BRANCH} | $(date)" | tee -a "${LOG}"

# The chain file is read on fd 3 and ORCA gets </dev/null: mpirun otherwise
# inherits stdin and swallows the remaining lines (the chain then stops after
# its first point with no error).
while read -r POINT PREV <&3; do
    [ -z "${POINT}" ] && continue
    PDIR="${STATE_DIR}/${POINT}"
    OUT="${PDIR}/${STATE}.out"

    if [ -f "${OUT}" ] && grep -Fq "ORCA TERMINATED NORMALLY" "${OUT}"; then
        echo "SKIP ${POINT} (already done)" | tee -a "${LOG}"
        continue
    fi

    # leftovers from an earlier failed attempt (keep inputs and geometry)
    # (F0 keeps its seed guess.gbw; later points get a fresh copy just below)
    if [ "${PREV}" = "-" ]; then
        find "${PDIR}" -maxdepth 1 -type f ! -name '*.inp' ! -name 'geometry.xyz' ! -name 'guess.gbw' -delete || true
    else
        find "${PDIR}" -maxdepth 1 -type f ! -name '*.inp' ! -name 'geometry.xyz' -delete || true
    fi

    if [ "${PREV}" != "-" ]; then
        PREV_GBW="${STATE_DIR}/${PREV}/${STATE}.gbw"
        PREV_OUT="${STATE_DIR}/${PREV}/${STATE}.out"
        if [ ! -f "${PREV_GBW}" ] || ! grep -Fq "ORCA TERMINATED NORMALLY" "${PREV_OUT}" 2>/dev/null; then
            echo "ERROR: previous point ${PREV} not finished; cannot continue to ${POINT}" | tee -a "${LOG}" >&2
            exit 1
        fi
        cp "${PREV_GBW}" "${PDIR}/guess.gbw"
    fi

    cd "${PDIR}"
    echo "RUN  ${POINT} (guess from ${PREV}) at $(date)" | tee -a "${LOG}"

    set +e
    "${ORCA_DIR}/orca" "${STATE}.inp" > "${OUT}" 2>&1 < /dev/null
    CODE=$?
    set -e

    if [ "${CODE}" -ne 0 ] || ! grep -Fq "ORCA TERMINATED NORMALLY" "${OUT}"; then
        echo "FAIL ${POINT} (exit ${CODE}); chain stopped, temporary files kept" | tee -a "${LOG}" >&2
        exit 1
    fi

    E=$(grep "FINAL SINGLE POINT ENERGY" "${OUT}" | tail -1 | awk '{print $5}')
    S2=$(grep "Expectation value of <S\*\*2>" "${OUT}" | tail -1 | awk '{print $NF}')
    NCYC=$(grep -o "SCF CONVERGED AFTER *[0-9]*" "${OUT}" | tail -1 | awk '{print $NF}')
    echo "DONE ${POINT}  E=${E}  <S2>=${S2}  cycles=${NCYC}" | tee -a "${LOG}"

    find "${PDIR}" -maxdepth 1 -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name 'guess.gbw' \) -delete || true
    cd "${STATE_DIR}"
done 3< "${CHAIN_FILE}"

echo "Chain ${BRANCH} complete at $(date)" | tee -a "${LOG}"
