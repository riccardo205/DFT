#!/usr/bin/env bash
# Remove ORCA *.densities files from the repo (density scratch, regenerable
# from the .gbw). Nothing in scripts/ or visualisation/ reads them; restarts
# use MORead/%moinp on .gbw files. *.densitiesinfo (small) is kept.
#
# Usage:
#   remove_heavy_backups.sh            # dry run: list files and total size
#   remove_heavy_backups.sh --delete   # actually delete them
#
# Optional: ROOT=/path/to/dir remove_heavy_backups.sh   (default: repo root)

# If sourced (or run by zsh), re-run in a separate bash process so that
# `exit` and `set -e` cannot close the calling terminal.
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file*) bash "$0" "$@"; return $? ;;
        *)       exec bash "$0" "$@" ;;
    esac
elif [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    bash "${BASH_SOURCE[0]}" "$@"; return $?
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DELETE=0
[[ "${1:-}" == "--delete" ]] && DELETE=1

LIST="$(mktemp -t heavy_densities)"
trap 'rm -f "$LIST"' EXIT

find "$ROOT" -path "$ROOT/.git" -prune -o -type f -name '*.densities' -print0 > "$LIST"

N=$(tr -cd '\0' < "$LIST" | wc -c | tr -d ' ')
if [[ "$N" -eq 0 ]]; then
    echo "Nothing to remove under $ROOT"
    exit 0
fi

echo "Root: $ROOT"
xargs -0 stat -f '%z' < "$LIST" | awk -v n="$N" '{ t+=$1 } END { printf "  *.densities  %d files  %.1f GB\n", n, t/1e9 }'

if [[ "$DELETE" -eq 0 ]]; then
    echo
    echo "Dry run. First 20 files:"
    tr '\0' '\n' < "$LIST" | awk -v r="$ROOT/" 'NR<=20 { sub("^" r, "  "); print }'
    echo
    echo "Re-run with --delete to remove these $N files."
    exit 0
fi

read -r -p "Delete these $N files? Type 'yes' to confirm: " ans
[[ "$ans" == "yes" ]] || { echo "Aborted."; exit 1; }

xargs -0 rm -f < "$LIST"
echo "Deleted $N files."
