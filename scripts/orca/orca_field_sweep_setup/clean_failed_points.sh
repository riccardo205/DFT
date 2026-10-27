#!/bin/bash
# Reset field-sweep points that did not terminate normally, keeping only the
# input and geometry (and the seed guess.gbw of F0). Run from the
# angle_scan_field_sweep folder. Dry run by default; pass --delete to act.
#
#   bash clean_failed_points.sh            # list what would be removed
#   bash clean_failed_points.sh --delete

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ACT="${1:-}"

for INP in "${ROOT}"/*/*/*/*.inp; do
    PDIR="$(dirname "${INP}")"
    OUT="${INP%.inp}.out"
    grep -Fqs "ORCA TERMINATED NORMALLY" "${OUT}" && continue
    N=$(find "${PDIR}" -maxdepth 1 -type f ! -name '*.inp' ! -name 'geometry.xyz' ! -name 'guess.gbw' | wc -l)
    [ "${N}" -eq 0 ] && continue
    echo "${PDIR#${ROOT}/}: ${N} files"
    if [ "${ACT}" = "--delete" ]; then
        find "${PDIR}" -maxdepth 1 -type f ! -name '*.inp' ! -name 'geometry.xyz' ! -name 'guess.gbw' -delete
    fi
done
[ "${ACT}" = "--delete" ] || echo "(dry run; pass --delete to remove)"
