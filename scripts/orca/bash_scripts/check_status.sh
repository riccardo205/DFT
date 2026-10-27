#!/bin/bash
# Status of every ORCA calculation (*.inp) below a folder, one line per calculation folder.
#
#   DONE         ORCA TERMINATED NORMALLY (for optimisations also: converged / NOT converged)
#   RUNNING      job id from job_id.txt is running in squeue
#   PENDING      job id is waiting in squeue
#   FAILED       output exists, ORCA did not terminate normally and the job is no longer in the
#                queue (reason from slurm.err / the output: time limit, SCF not converged, error)
#   NOT RUN      job_id.txt exists but the job left the queue without writing an output
#   NOT STARTED  no job_id.txt and no output
#
# Usage:  check_status.sh [ROOT] [STATUS]
#   ROOT    folder to search (default: .)
#   STATUS  only print lines with this status, e.g. FAILED or RUNNING (summary is always printed)
# Without squeue (e.g. on a laptop) jobs with an output are reported as UNFINISHED and jobs
# with only a job_id.txt as SUBMITTED, since their queue state cannot be checked.

ROOT="${1:-.}"
FILTER="${2:-}"

# active jobs: lines "<jobid> <STATE>", fetched once (plain text: works with bash 3 and 4+)
QUEUE=""
HAVE_SQUEUE=0
if command -v squeue >/dev/null 2>&1; then
    HAVE_SQUEUE=1
    QUEUE=$(squeue -u "$USER" -h -o "%i %T" 2>/dev/null)
fi
STATUSES=$(mktemp)
trap 'rm -f "$STATUSES"' EXIT
printf "%-70s %-12s %s\n" "CALCULATION" "STATUS" "DETAILS"
printf '%.0s=' {1..120}; echo

while read -r inp; do
    dir=$(dirname "$inp")
    base=$(basename "$inp" .inp)
    out="$dir/$base.out"
    err="$dir/slurm.err"
    jobid=""
    [ -f "$dir/job_id.txt" ] && jobid=$(tr -d '[:space:]' < "$dir/job_id.txt")
    qstate=""
    [ -n "$jobid" ] && qstate=$(printf '%s\n' "$QUEUE" | awk -v id="$jobid" '$1 == id {print $2; exit}')
    details=""

    if [ -f "$out" ] && tail -n 30 "$out" | grep -q "ORCA TERMINATED NORMALLY"; then
        status="DONE"
        if grep -qi "^ *! .*opt" "$inp"; then
            if grep -q "THE OPTIMIZATION HAS CONVERGED" "$out"; then
                details="optimisation converged"
            else
                details="optimisation NOT converged"
            fi
        fi
        energy=$(grep "FINAL SINGLE POINT ENERGY" "$out" | tail -1 | awk '{print $5}')
        [ -n "$energy" ] && details="${details:+$details, }E = $energy Eh"
    elif [ "$qstate" = "RUNNING" ] || [ "$qstate" = "COMPLETING" ]; then
        status="RUNNING"
        details="job $jobid"
        if [ -f "$out" ]; then
            ncyc=$(grep -c "GEOMETRY OPTIMIZATION CYCLE" "$out")
            [ "$ncyc" -gt 0 ] && details="$details, opt cycle $ncyc"
            grep -q "Leaving SCF to start the TRAH" "$out" && details="$details, in TRAH"
        fi
    elif [ -n "$qstate" ]; then
        status="PENDING"
        details="job $jobid ($qstate)"
    elif [ "$HAVE_SQUEUE" -eq 0 ] && [ -f "$out" ]; then
        status="UNFINISHED"
        details="no squeue here: running or stopped"
    elif [ -f "$out" ]; then
        status="FAILED"
        if [ -f "$err" ] && grep -q "DUE TO TIME LIMIT" "$err"; then
            details="time limit"
        elif grep -q "SCF NOT CONVERGED" "$out"; then
            details="SCF not converged"
        else
            reason=$(grep -m1 -i "error\|aborting\|cannot open" "$out" "$err" 2>/dev/null | head -1 | cut -d: -f2- | tr -s ' ' | cut -c1-60)
            details="${reason:-stopped without a message}"
        fi
        [ -n "$jobid" ] && details="$details (job $jobid)"
    elif [ -n "$jobid" ] && [ "$HAVE_SQUEUE" -eq 0 ]; then
        status="SUBMITTED"
        details="job $jobid, no output here (no squeue to check)"
    elif [ -n "$jobid" ]; then
        status="NOT RUN"
        details="job $jobid not in queue, no output"
        [ -f "$err" ] && [ -s "$err" ] && details="$details; slurm.err: $(tail -1 "$err" | cut -c1-50)"
    else
        status="NOT STARTED"
    fi

    echo "$status" >> "$STATUSES"
    if [ -z "$FILTER" ] || [ "$status" = "$FILTER" ]; then
        printf "%-70s %-12s %s\n" "${inp#./}" "$status" "$details"
    fi
done < <(find "$ROOT" -type f -name "*.inp" -not -path "*/.*" | sort -V)

printf '%.0s=' {1..120}; echo
for s in DONE RUNNING PENDING UNFINISHED SUBMITTED FAILED "NOT RUN" "NOT STARTED"; do
    n=$(grep -cx "$s" "$STATUSES")
    [ "$n" -gt 0 ] && printf "%-12s %5d\n" "$s" "$n"
done
printf "%-12s %5d\n" "TOTAL" "$(wc -l < "$STATUSES" | tr -d ' ')"
