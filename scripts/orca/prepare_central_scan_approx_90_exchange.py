#!/usr/bin/env python3
"""
PBE0 HS/BS single points, with natural orbitals, on the relaxed geometries of
angle_scan_central_approx_90 (free_outer/ and fixed_outer/).

Same protocol as prepare_outer_scan_exchange.py (PBE0 D4 / def2-TZVP, BS from GuessMix),
plus ! UNO UCO and the UNO occupation / Hirshfeld printout of prepare_natural_orbitals.py,
so .uno/.uco files come out of the same job (no MORead restart needed).

Writes angle_scan_central_approx_90_exchange/<set>/<label>/<state>/{<state>.inp, geometry.xyz}.
Relaxations that have not terminated normally are skipped; points that already exist
are left alone, so the script can be re-run to add relaxations that finish later.
All jobs are independent: the tree can be submitted with submit_functional_scan.py.
"""

import csv
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "orca_angle_scan_exchange_setup"))
from prepare_angle_scan_exchange import relaxation_converged  # noqa: E402

TEMPLATE = """! {functional} {basis} TightSCF UKS DEFGRID3 NoAutoStart{guess} UNO UCO

%scf
    MaxIter {max_iter}
    ConvCheckMode 0
end

%output
    Print[P_UNO_OccNum] 1
    Print[P_Hirshfeld] 1
end

%pal
    nprocs {nprocs}
end

%maxcore {maxcore}

* xyzfile {charge} {mult} geometry.xyz
"""


def main():
    # ---- USER SETTINGS ----
    SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
    SOURCE_ROOT = SCANS / "angle_scan_central_approx_90"
    OUTPUT_ROOT = SCANS / "angle_scan_central_approx_90_exchange"
    SETS = ["free_outer", "fixed_outer"]

    FUNCTIONAL_KEYWORD = "PBE0 D4"
    BASIS = "def2-TZVP"
    CHARGE = 0
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500

    STATES = [
        {"folder": "high_spin", "multiplicity": 3, "guess_mix": False},
        {"folder": "broken_symmetry", "multiplicity": 1, "guess_mix": True},
    ]

    manifest_rows, created, existing, skipped = [], [], [], []
    for set_name in SETS:
        for relax_dir in sorted(d for d in (SOURCE_ROOT / set_name).iterdir() if d.is_dir()):
            point = f"{set_name}/{relax_dir.name}"
            geometry_source = relax_dir / "high_spin.xyz"
            if not (geometry_source.exists() and relaxation_converged(relax_dir / "high_spin.out")):
                skipped.append(point)
                print(f"SKIP (relaxation not finished): {point}")
                continue

            for state in STATES:
                run_folder = OUTPUT_ROOT / set_name / relax_dir.name / state["folder"]
                input_file = run_folder / f"{state['folder']}.inp"
                manifest_rows.append({
                    "set": set_name,
                    "scan_point": relax_dir.name,
                    "source_geometry": str(geometry_source.relative_to(REPO_ROOT)),
                    "state": state["folder"],
                    "multiplicity": state["multiplicity"],
                    "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
                })
                if run_folder.exists():
                    existing.append(f"{point}/{state['folder']}")
                    continue
                run_folder.mkdir(parents=True)
                shutil.copy2(geometry_source, run_folder / "geometry.xyz")
                input_file.write_text(
                    TEMPLATE.format(
                        functional=FUNCTIONAL_KEYWORD, basis=BASIS, guess=" GuessMix" if state["guess_mix"] else "",
                        max_iter=MAX_ITER, nprocs=NPROCS, maxcore=MAXCORE_MB, charge=CHARGE,
                        mult=state["multiplicity"],
                    ),
                    encoding="utf-8",
                )
                created.append(f"{point}/{state['folder']}")

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Central-scan (C ~ 90) PBE0 HS/BS single points with UNO/UCO prepared")
    print("=" * 70)
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Points   : {len(manifest_rows) // len(STATES)} ({len(manifest_rows)} jobs in manifest)")
    print(f"New jobs : {len(created)} (left existing: {len(existing)})")
    print(f"Skipped  : {len(skipped)} {skipped}")


if __name__ == "__main__":
    main()
