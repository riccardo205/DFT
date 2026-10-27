#!/usr/bin/env python3
"""
PBE0 high-spin (uuuu, quintet, no FlipSpin) single points on the converged dimer relaxations
of one dimers_run2 run, as <run>_broken_symmetry_high_spin/relaxed/<dimer>/uuuu/.

These are the reference calculations whose uuuu.gbw is later read (MORead) by the FlipSpin
configurations in <run>_broken_symmetry (prepare_dimers_broken_symmetry.py, MOREAD_FROM).
Same level of theory and input template as prepare_dimers_broken_symmetry.py.

Handles both layouts: <run>/<dimer>/high_spin (run2, run3) and <run>/<set>/<dimer>/high_spin
(run4: fixC_dihedral, fixC_double90, free); the <set>/<dimer> structure is kept.
Only relaxations that converged are used (relaxed geometry = high_spin.xyz); the others are
listed in relaxation_status.csv. Existing folders are left alone, so re-run the script when
more relaxations converge. All jobs are independent: submit with submit_functional_scan.py.

    python3 prepare_dimers_high_spin_reference.py run4
    python3 prepare_dimers_high_spin_reference.py run4 --only free/packmol_1 free/docked_4 ...
    python3 prepare_dimers_high_spin_reference.py run2 --only conf_12 conf_25 \
        --output run2_broken_symmetry_high_spin_reconverged
With --only, only the listed dimers are prepared and kept in the manifest; --output sets the
experiment folder name (default <run>_broken_symmetry_high_spin), e.g. for relaxations that
converged after the first high-spin experiment was made.
"""

import argparse
import csv
import shutil
from pathlib import Path

from prepare_dimers_broken_symmetry import TEMPLATE, relaxation_state

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"

FUNCTIONAL = "PBE0"
BASIS = "def2-TZVP"
CHARGE = 0
MULTIPLICITY = 5
NPROCS = 16
MAXCORE_MB = 2000
MAX_ITER = 500
CONV_CHECK_MODE_0 = False            # as in prepare_dimers_broken_symmetry.py


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="source run folder in dimers_run2, e.g. run4")
    parser.add_argument("--only", nargs="+", help="prepare only these dimers, e.g. free/docked_4")
    parser.add_argument("--output", help="experiment folder name (default <run>_broken_symmetry_high_spin)")
    args = parser.parse_args()

    source_root = RUN_ROOT / args.run
    output_root = RUN_ROOT / (args.output or f"{args.run}_broken_symmetry_high_spin")
    relax_dirs = sorted(source_root.glob("*/high_spin")) + sorted(source_root.glob("*/*/high_spin"))
    if not relax_dirs:
        raise SystemExit(f"No <dimer>/high_spin folders found in {source_root}")
    output_root.mkdir(exist_ok=True)

    text = TEMPLATE.format(
        functional=FUNCTIONAL, basis=BASIS, moread="", moinp="", max_iter=MAX_ITER,
        conv_check="    ConvCheckMode 0\n" if CONV_CHECK_MODE_0 else "", flip="",
        nprocs=NPROCS, maxcore=MAXCORE_MB, charge=CHARGE, mult=MULTIPLICITY,
    )

    status_rows, manifest_rows, created, existing, not_converged = [], [], 0, 0, []
    for relax in relax_dirs:
        dimer = relax.parent.relative_to(source_root).as_posix()          # conf_1 or free/docked_1
        if args.only and dimer not in args.only:
            continue
        state = relaxation_state(relax / "high_spin.out")
        status_rows.append({"dimer": dimer, "relaxation": state})
        if state != "converged":
            not_converged.append(f"{dimer} ({state})")
            continue

        run_folder = output_root / "relaxed" / dimer / "uuuu"
        manifest_rows.append({
            "dimer": dimer,
            "configuration": "++++",
            "multiplicity": MULTIPLICITY,
            "source_geometry": str((relax / "high_spin.xyz").relative_to(REPO_ROOT)),
            "input_file": f"relaxed/{dimer}/uuuu/uuuu.inp",
        })
        if run_folder.exists():
            existing += 1
            continue
        run_folder.mkdir(parents=True)
        shutil.copy2(relax / "high_spin.xyz", run_folder / "geometry.xyz")
        (run_folder / "uuuu.inp").write_text(text, encoding="utf-8")
        created += 1

    if args.only:
        unknown = set(args.only) - {r["dimer"] for r in status_rows}
        if unknown:
            raise SystemExit(f"Not found in {source_root}: {sorted(unknown)}")
    for name, rows in (("relaxation_status.csv", status_rows), ("manifest.csv", manifest_rows)):
        with (output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"{output_root.name}: {len(manifest_rows)} uuuu jobs ({created} new, {existing} already present)")
    print(f"Relaxations NOT converged, skipped ({len(not_converged)}):")
    for item in not_converged:
        print(f"  {item}")


if __name__ == "__main__":
    main()
