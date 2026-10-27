#!/bin/bash
# Submit the whole field sweep from the angle_scan_field_sweep folder on /scratch.
#
#   bash submit_field_sweep.sh                 # all structures
#   bash submit_field_sweep.sh M1 L090_C080_R090
#
# One Slurm job per chain_<job>.txt in each <structure>/<state> folder:
#   chain_F0.txt              zero field
#   chain_<dir>_<s><|F|>.txt  one point, guess from F0          (|F| <= 0.001)
#   chain_<dir>_<s>_tail.txt  0.002 -> 0.005, guess from 0.001  (continuation)
# A job waits (afterok) for the job that produces the point it starts from,
# unless that point already terminated normally. Jobs whose points are all
# done are not submitted, so rerunning this script only fills in what is missing.
#
# Do NOT submit the point folders individually (e.g. with the generic
# orca_job.sh / submit_functional_scan.py): each point reads guess.gbw, which
# run_chain.sh copies from the previous point right before ORCA starts.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -gt 0 ]; then
    STRUCTURES=("$@")
else
    STRUCTURES=(L090_C080_R090 L135_C120_R090 L090_C120_R135 M1)
fi

done_point() {  # <state_dir> <point>
    grep -Fqs "ORCA TERMINATED NORMALLY" "$1/$2/$(basename "$1").out"
}

for S in "${STRUCTURES[@]}"; do
    for STATE in high_spin broken_symmetry; do
        D="${ROOT}/${S}/${STATE}"
        [ -d "${D}" ] || { echo "missing ${D}" >&2; continue; }
        TAG="${S}_${STATE:0:2}"

        PRODUCER=" "   # " point=jobid point=jobid ... " for the jobs submitted here
        # F0 first, then single points, then tails (tails depend on single points)
        CHAINS=("${D}/chain_F0.txt")
        for C in "${D}"/chain_*.txt; do
            case "$(basename "${C}")" in chain_F0.txt|*_tail.txt) ;; *) CHAINS+=("${C}") ;; esac
        done
        for C in "${D}"/chain_*_tail.txt; do [ -f "${C}" ] && CHAINS+=("${C}"); done

        for CHAIN in "${CHAINS[@]}"; do
            B="$(basename "${CHAIN}" .txt)"; B="${B#chain_}"

            TODO=0
            while read -r PT _; do
                [ -n "${PT}" ] && ! done_point "${D}" "${PT}" && TODO=1
            done < "${CHAIN}"
            if [ "${TODO}" -eq 0 ]; then
                echo "${TAG} ${B}: done"
                continue
            fi

            FIRST_PREV="$(head -n 1 "${CHAIN}" | awk '{print $2}')"
            DEP=()
            if [ "${FIRST_PREV}" != "-" ] && ! done_point "${D}" "${FIRST_PREV}"; then
                PJ="$(echo "${PRODUCER}" | tr ' ' '\n' | awk -F= -v p="${FIRST_PREV}" '$1 == p {print $2}')"
                if [ -n "${PJ}" ]; then
                    DEP=(--dependency=afterok:"${PJ}")
                else
                    echo "${TAG} ${B}: ${FIRST_PREV} not done and not queued here; skipped" >&2
                    continue
                fi
            fi

            TIME=04:00:00
            case "${B}" in *_tail) TIME=12:00:00 ;; esac

            J=$(sbatch --parsable --chdir="${D}" --job-name="${B}_${TAG}" \
                ${DEP[@]+"${DEP[@]}"} --time="${TIME}" \
                --output="${D}/slurm_${B}.out" --error="${D}/slurm_${B}.err" \
                "${ROOT}/run_chain.sh" "${B}")
            echo "${J}" > "${D}/job_id_${B}.txt"
            while read -r PT _; do [ -n "${PT}" ] && PRODUCER="${PRODUCER}${PT}=${J} "; done < "${CHAIN}"
            echo "${TAG} ${B} -> ${J} ${DEP[*]:-}"
        done
    done
done
