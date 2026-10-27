#!/bin/bash
# Delete the broken-symmetry folders of dimers whose relaxation did not converge.
#
# Run inside a <run>_broken_symmetry folder (it must contain relaxation_status.csv):
#     bash remove_unrelaxed_dimers.sh          # dry run: only lists what would be deleted
#     bash remove_unrelaxed_dimers.sh --yes    # delete
#
# For every dimer whose "relaxation" column is not "converged", relaxed/<dimer> is removed
# and its rows are dropped from manifest.csv. initial/ (starting geometries) is not touched.
# Note: re-running prepare_dimers_broken_symmetry.py recreates the deleted folders.

set -euo pipefail

STATUS=relaxation_status.csv
DELETE=true
[ "${1:-}" = "--yes" ] && DELETE=true

if [ ! -f "$STATUS" ]; then
    echo "ERROR: $STATUS not found in $(pwd)" >&2
    exit 1
fi

# dimer,relaxation -> dimers that are not converged
# (the CSV files have Windows line endings: strip the trailing \r before comparing)
unrelaxed=$(awk -F, '{sub(/\r$/, "")} NR > 1 && $2 != "converged" {print $1}' "$STATUS")

if [ -z "$unrelaxed" ]; then
    echo "All relaxations converged, nothing to delete."
    exit 0
fi

for dimer in $unrelaxed; do
    folder="relaxed/$dimer"
    status=$(awk -F, -v d="$dimer" '{sub(/\r$/, "")} $1 == d {print $2}' "$STATUS")
    if [ ! -d "$folder" ]; then
        echo "absent   : $folder ($status)"
    elif $DELETE; then
        rm -rf -- "$folder"
        echo "deleted  : $folder ($status)"
    else
        echo "would delete: $folder ($status, $(find "$folder" -name '*.inp' | wc -l | tr -d ' ') inputs)"
    fi
done

if $DELETE && [ -f manifest.csv ]; then
    # manifest columns: stage,dimer,relaxation,...  -> keep header, initial/, converged rows
    awk -F, 'NR == 1 || $1 != "relaxed" || $3 == "converged"' manifest.csv > manifest.csv.tmp
    mv manifest.csv.tmp manifest.csv
    echo "manifest.csv: dropped the relaxed/ rows of non-converged dimers"
fi

$DELETE || echo "Dry run. Re-run with --yes to delete."
