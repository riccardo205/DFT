#!/usr/bin/env python3
"""
PBE0 HS/BS single-point inputs on the relaxed geometries of the detailed
outer-torsion scans (angle_scan_C100_outer_relaxation, angle_scan_C100_L090_relaxation).
Same protocol as orca_angle_scan_exchange_setup/prepare_angle_scan_exchange.py.

Points whose relaxed geometry is identical (< 5e-3 Å) to a coarse-grid point that
already has PBE0 results in angle_scan_exchange_analysis are skipped: the analysis
(visualisation/analyze_outer_scans.py) reuses those results.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "orca_angle_scan_exchange_setup"))
from prepare_angle_scan_exchange import relaxation_converged, write_orca_input  # noqa: E402


def main():
    # ---- USER SETTINGS ----
    SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
    JOBS = {
        SCANS / "angle_scan_C100_outer_relaxation": SCANS / "angle_scan_C100_outer_exchange",
        SCANS / "angle_scan_C100_L090_relaxation": SCANS / "angle_scan_C100_L090_exchange",
    }
    COARSE_ROOT = SCANS / "angle_scan_exchange_analysis"
    SAME_GEOMETRY_TOL_ANG = 5e-3

    FUNCTIONAL_KEYWORD = "PBE0 D4"
    BASIS = "def2-TZVP"
    CHARGE = 0
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500

    OVERWRITE = False

    STATES = [
        {"folder": "high_spin", "input_name": "high_spin.inp", "multiplicity": 3, "guess_mix": False},
        {"folder": "broken_symmetry", "input_name": "broken_symmetry.inp", "multiplicity": 1, "guess_mix": True},
    ]

    for source_root, output_root in JOBS.items():
        if output_root.exists() and not OVERWRITE:
            raise FileExistsError(f"Output directory already exists: {output_root.resolve()}")
        if output_root.exists() and OVERWRITE:
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)

        manifest_rows, skipped = [], []
        for relax_dir in sorted(d for d in source_root.iterdir() if d.is_dir() and d.name.startswith("L")):
            geometry_source = relax_dir / "high_spin.xyz"
            if not (geometry_source.exists() and relaxation_converged(relax_dir / "high_spin.out")):
                print(f"SKIP (not converged): {relax_dir.name}")
                continue

            coarse_geom = COARSE_ROOT / relax_dir.name / "high_spin" / "geometry.xyz"
            coarse_out = COARSE_ROOT / relax_dir.name / "broken_symmetry" / "broken_symmetry.out"
            if coarse_geom.exists() and coarse_out.exists():
                dmax = np.abs(read(coarse_geom).get_positions() - read(geometry_source).get_positions()).max()
                if dmax < SAME_GEOMETRY_TOL_ANG:
                    skipped.append(relax_dir.name)
                    print(f"SKIP (PBE0 exists in {COARSE_ROOT.name}, max Δx {dmax:.0e} Å): {relax_dir.name}")
                    continue

            for state in STATES:
                run_folder = output_root / relax_dir.name / state["folder"]
                run_folder.mkdir(parents=True)
                shutil.copy2(geometry_source, run_folder / "geometry.xyz")
                write_orca_input(
                    output_file=run_folder / state["input_name"], functional_keyword=FUNCTIONAL_KEYWORD,
                    basis=BASIS, charge=CHARGE, multiplicity=state["multiplicity"], nprocs=NPROCS,
                    maxcore_mb=MAXCORE_MB, max_iter=MAX_ITER, guess_mix=state["guess_mix"],
                )
                manifest_rows.append({
                    "scan_point": relax_dir.name,
                    "source_geometry": str(geometry_source.relative_to(REPO_ROOT)),
                    "state": state["folder"],
                    "multiplicity": state["multiplicity"],
                    "input_file": f"{relax_dir.name}/{state['folder']}/{state['input_name']}",
                })
            print(f"Created: {output_root.name}/{relax_dir.name}")

        with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"--> {output_root.name}: {len(manifest_rows)} jobs, reused {len(skipped)} ({', '.join(skipped)})\n")


if __name__ == "__main__":
    main()
