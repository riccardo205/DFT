#!/usr/bin/env python3
"""
High-spin R2SCAN-3C relaxations around C = 90 deg, in two sets:

  free_outer/  : C = 80, 85, ..., 100 constrained; outer torsions initialised at
                 (L, R) = (45, 45), (135, 135), (45, 135) and then FREE.
  fixed_outer/ : C = 90 with L <= R in {0, 45, 90, 135, 180} (mirror pairs L<->R skipped);
                 L, C and R all constrained (fills the C = 90 gap of angle_scan_relaxation).

Starting geometries are the HS-relaxed constrained-scan geometries in
angle_scan_relaxation, which only exist at C = 80 and 100 (20 deg steps).
For every target C the base with the nearest C at the same (L, R) is taken and
its central torsion is rotated rigidly to the target (<= 10 deg; ties -> lower C).

Broken-symmetry states and UNO/UCO orbitals are left to later single points.
All jobs are independent: the tree can be submitted with submit_functional_scan.py.
"""

import csv
import shutil
import sys
from itertools import product
from pathlib import Path

from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Short-2OS" / "molecule_creation" / "generate_angle_scan_SM"))
from generate_angle_scan import (  # noqa: E402
    angle_diff_deg,
    build_connectivity,
    fmt_angle,
    render_orca_input,
    set_dihedral,
)


def main():
    # ---- USER SETTINGS ----
    SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
    SOURCE_ROOT = SCANS / "angle_scan_relaxation"
    OUTPUT_ROOT = SCANS / "angle_scan_central_approx_90"

    BASE_CENTRAL = [80, 100]                         # C available in angle_scan_relaxation
    OUTER_GRID = [0, 45, 90, 135, 180]
    SETS = {
        "free_outer": {
            "points": [(l, c, r) for l, r in [(45, 45), (135, 135), (45, 135)] for c in [80, 85, 90, 95, 100]],
            "constrain_outer": False,
        },
        "fixed_outer": {
            "points": [(l, 90, r) for l, r in product(OUTER_GRID, OUTER_GRID) if l <= r],  # L<->R mirror skipped
            "constrain_outer": True,
        },
    }

    LEFT = (42, 41, 0, 5)
    CENTER = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)
    ANGLE_TOLERANCE_DEG = 0.01

    CHARGE = 0
    MULTIPLICITY = 3
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 500
    MAX_ITER_GEOM = 200

    OVERWRITE = False

    # ---- SETUP ----
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source scan root not found: {SOURCE_ROOT.resolve()}")
    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = False only if you really want to regenerate it."
        )
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    manifest_rows = []
    for set_name, spec in SETS.items():
        print(f"--- {set_name} ---")
        for left, central, right in spec["points"]:
            base_c = min(BASE_CENTRAL, key=lambda c: (abs(c - central), c))
            base_label = f"L{fmt_angle(left)}_C{fmt_angle(base_c)}_R{fmt_angle(right)}"
            base_geometry = SOURCE_ROOT / base_label / "high_spin.xyz"
            if not base_geometry.exists():
                raise FileNotFoundError(f"No optimized geometry found for {base_label}: {base_geometry.resolve()}")

            atoms = read(base_geometry)
            conn = build_connectivity(atoms)
            set_dihedral(atoms, conn, CENTER, float(central), "CENTER")
            obtained = atoms.get_dihedral(*CENTER)
            if abs(angle_diff_deg(obtained, central)) > ANGLE_TOLERANCE_DEG:
                raise RuntimeError(f"{base_label} -> C={central}: off target, obtained {obtained:.3f}")
            l_start, r_start = atoms.get_dihedral(*LEFT), atoms.get_dihedral(*RIGHT)

            constraints = [("CENTER", CENTER, float(central))]
            if spec["constrain_outer"]:
                constraints += [("LEFT", LEFT, float(left)), ("RIGHT", RIGHT, float(right))]

            tag = f"L{fmt_angle(left)}_C{fmt_angle(central)}_R{fmt_angle(right)}"
            run_folder = OUTPUT_ROOT / set_name / tag
            run_folder.mkdir(parents=True, exist_ok=False)
            write(run_folder / "geometry.xyz", atoms)
            input_file = run_folder / "high_spin.inp"
            input_file.write_text(
                render_orca_input(constraints, charge=CHARGE, multiplicity=MULTIPLICITY, nprocs=NPROCS,
                                  maxcore_mb=MAXCORE_MB, max_iter_scf=MAX_ITER_SCF, max_iter_geom=MAX_ITER_GEOM),
                encoding="utf-8",
            )
            manifest_rows.append({
                "set": set_name,
                "label": tag,
                "source_geometry": str(base_geometry.relative_to(REPO_ROOT)),
                "central_angle": central,
                "left_init": left,
                "right_init": right,
                "left_start_actual": round(l_start, 3),
                "right_start_actual": round(r_start, 3),
                "outer_constrained": spec["constrain_outer"],
                "state": "high_spin",
                "multiplicity": MULTIPLICITY,
                "nprocs": NPROCS,
                "maxcore_mb": MAXCORE_MB,
                "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
            })
            print(f"{tag} (from {base_label}): C={obtained:.3f}, L={l_start:.3f}, R={r_start:.3f}")

    manifest_file = OUTPUT_ROOT / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Central-torsion scan around C = 90 prepared (high spin)")
    print("=" * 70)
    print(f"Source   : {SOURCE_ROOT.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
