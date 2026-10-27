#!/usr/bin/env python3
"""
Outer-torsion scans at FIXED central torsion C = 100 deg, built from the
HS-relaxed constrained-scan geometries at C = 100 (angle_scan_relaxation).

SCAN_TYPE selects the (L, R) targets:
  "symmetric" : L = R = 0, 10, ..., 180
  "left90"    : L = 90 fixed, R = 0, 10, ..., 180

Each target is generated from the relaxed scan geometry with the NEAREST
(L, R) among the available bases (outer angles 0/45/90/135/180), so every
rigid rotation is <= 22.5 deg. Rotating a single base (e.g. L045_C100_R045,
or the M1 free minimum) through the whole symmetric range makes fragments 1
and 4 collide at L = R ~ 70-100 deg (H-H down to 0.2-0.6 Å).

Fragments 1 and 4 are rotated rigidly about their linking bonds with L = R,
spanning 0..180 deg. The central dihedral (fragment 2 | fragment 3) is kept at
the M1 value. Each geometry gets a constrained high-spin R2SCAN-3C relaxation
input (L, C, R all constrained), in the same layout as angle_scan_relaxation.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
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


def min_nonbonded_distance(atoms, conn):
    pos = atoms.get_positions()
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    rows, cols = conn.nonzero()
    dmat[rows, cols] = np.inf
    i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
    return float(dmat[i, j]), (int(i), int(j))


def main():
    # ---- USER SETTINGS ----
    SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
    SOURCE_ROOT = SCANS / "angle_scan_relaxation"
    CENTER_SOURCE = 100
    SCAN_ANGLES = np.arange(0, 181, 10).tolist()  # 0, 10, ..., 180
    BASE_ANGLES = [0, 45, 90, 135, 180]            # outer angles available in angle_scan_relaxation

    SCAN_TYPE = "left90"  # "symmetric" or "left90"
    if SCAN_TYPE == "symmetric":
        TARGETS = [(a, a) for a in SCAN_ANGLES]
        BASE_LR = [(a, a) for a in BASE_ANGLES]
        OUTPUT_ROOT = SCANS / "angle_scan_C100_outer_relaxation"
    elif SCAN_TYPE == "left90":
        TARGETS = [(90, a) for a in SCAN_ANGLES]
        BASE_LR = [(90, a) for a in BASE_ANGLES]
        OUTPUT_ROOT = SCANS / "angle_scan_C100_L090_relaxation"
    else:
        raise ValueError(f"Unknown SCAN_TYPE: {SCAN_TYPE!r}")

    LEFT = (42, 41, 0, 5)
    CENTER = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)

    ANGLE_TOLERANCE_DEG = 0.01
    CLASH_WARN_ANG = 1.5

    CHARGE = 0
    MULTIPLICITY = 3
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 500
    MAX_ITER_GEOM = 200

    OVERWRITE = False

    # ---- SETUP ----
    bases = {}
    for lr in BASE_LR:
        path = SOURCE_ROOT / f"L{lr[0]:03d}_C{CENTER_SOURCE:03d}_R{lr[1]:03d}" / "high_spin.xyz"
        if not path.exists():
            raise FileNotFoundError(f"Base geometry not found: {path.resolve()}")
        bases[lr] = path
    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    center_angle = float(CENTER_SOURCE)
    print(f"Central dihedral fixed at {center_angle} deg")
    print("Base geometries (torsions L / C / R):")
    for path in bases.values():
        a = read(path)
        print(f"  {path.parent.name}: {a.get_dihedral(*LEFT):.3f} / {a.get_dihedral(*CENTER):.3f} / "
              f"{a.get_dihedral(*RIGHT):.3f} deg")
    print()

    manifest_rows = []
    for left, right in TARGETS:
        tag = f"L{fmt_angle(left)}_C{fmt_angle(center_angle)}_R{fmt_angle(right)}"
        base = min(bases, key=lambda b: abs(b[0] - left) + abs(b[1] - right))
        atoms = read(bases[base])
        conn = build_connectivity(atoms)
        order = [("CENTER", CENTER, center_angle), ("LEFT", LEFT, left), ("RIGHT", RIGHT, right)]
        for name, quartet, angle in order:
            set_dihedral(atoms, conn, quartet, angle, name)

        for name, quartet, angle in order:
            obtained = atoms.get_dihedral(*quartet)
            if abs(angle_diff_deg(obtained, angle)) > ANGLE_TOLERANCE_DEG:
                raise RuntimeError(f"{tag}: {name} off target: target={angle:.3f} obtained={obtained:.3f}")

        dmin, pair = min_nonbonded_distance(atoms, conn)
        warn = "  <-- CLASH" if dmin < CLASH_WARN_ANG else ""
        print(f"{tag} (from {bases[base].parent.name}): min non-bonded distance {dmin:.3f} Å (atoms {pair[0]}-{pair[1]}){warn}")

        run_folder = OUTPUT_ROOT / tag
        run_folder.mkdir(parents=True, exist_ok=False)
        write(run_folder / "geometry.xyz", atoms)
        input_text = render_orca_input(
            order, charge=CHARGE, multiplicity=MULTIPLICITY, nprocs=NPROCS, maxcore_mb=MAXCORE_MB,
            max_iter_scf=MAX_ITER_SCF, max_iter_geom=MAX_ITER_GEOM,
        )
        (run_folder / "high_spin.inp").write_text(input_text, encoding="utf-8")

        manifest_rows.append({
            "label": tag,
            "source_geometry": str(bases[base].relative_to(REPO_ROOT)),
            "left_angle": left,
            "center_angle": center_angle,
            "right_angle": right,
            "state": "high_spin",
            "multiplicity": MULTIPLICITY,
            "min_nonbonded_distance_A": round(dmin, 3),
            "min_nonbonded_pair": f"{pair[0]}-{pair[1]}",
            "input_file": f"{tag}/high_spin.inp",
        })

    manifest_file = OUTPUT_ROOT / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Outer-torsion scan (fixed C) prepared")
    print("=" * 70)
    print(f"Scan     : {SCAN_TYPE}")
    print(f"Bases    : {SOURCE_ROOT.resolve()} ({', '.join(p.parent.name for p in bases.values())})")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
