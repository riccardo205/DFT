#!/bin/bash
# Copy the converged high-spin wavefunction uuuu/uuuu.gbw of every dimer to the sibling
# spin-configuration folders (duuu, uduu, ..., duud) under the same name, to be read with
#     ! MORead
#     %moinp "uuuu.gbw"
# (the FlipSpin inputs are not modified by this script).
#
# Run from any parent folder, e.g. dimers_run2 or one <run>_broken_symmetry folder:
#     bash copy_uuuu_gbw.sh          # dry run: only lists what would be copied
#     bash copy_uuuu_gbw.sh --yes    # copy
#
# A dimer is used only if uuuu.out contains "ORCA TERMINATED NORMALLY" and uuuu.gbw exists.
# Folders whose copy of uuuu.gbw is already identical are skipped.

set -euo pipefail

COPY=true
[ "${1:-}" = "--yes" ] && COPY=true

n_copy=0; n_same=0; n_skip=0

while read -r hs_dir; do
    dimer_dir=$(dirname "$hs_dir")
    out="$hs_dir/uuuu.out"
    gbw="$hs_dir/uuuu.gbw"

    if [ ! -f "$out" ] || ! grep -q "ORCA TERMINATED NORMALLY" "$out"; then
        echo "SKIP (uuuu not finished): ${dimer_dir#./}"
        n_skip=$((n_skip + 1)); continue
    fi
    if [ ! -f "$gbw" ]; then
        echo "SKIP (no uuuu.gbw)      : ${dimer_dir#./}"
        n_skip=$((n_skip + 1)); continue
    fi

    for target in "$dimer_dir"/*/; do
        target=${target%/}
        name=$(basename "$target")
        [ "$name" = "uuuu" ] && continue
        [ -f "$target/$name.inp" ] || continue          # only spin-configuration folders

        if [ -f "$target/uuuu.gbw" ] && cmp -s "$gbw" "$target/uuuu.gbw"; then
            n_same=$((n_same + 1)); continue
        fi
        if $COPY; then
            cp -p "$gbw" "$target/uuuu.gbw"
            echo "copied      : ${target#./}/uuuu.gbw"
        else
            echo "would copy  : ${target#./}/uuuu.gbw"
        fi
        n_copy=$((n_copy + 1))
    done
done < <(find . -type d -name uuuu | sort -V)

echo
if $COPY; then
    echo "Copied $n_copy, already up to date $n_same, dimers skipped $n_skip."
else
    echo "Would copy $n_copy, already up to date $n_same, dimers skipped $n_skip."
    echo "Dry run. Re-run with --yes to copy."
fi
