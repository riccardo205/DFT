#!/usr/bin/env python3
"""
PBE0 HS/BS single points (no relaxation) at an EXACTLY orthogonal central link.

Question: does the residual exchange at C ~ 90 deg survive when the actual central
ring-plane angle theta_C is exactly 90 deg (instead of the ~84 deg obtained with a
90 deg central dihedral)?

Starting geometry: R2SCAN-3C HS relaxation with L, C, R fixed at 45/90/45
(angle_scan_central_approx_90/fixed_outer/L045_C090_R045). Rigid rotations only:
  1. outer dihedrals set to L = 40, R = 80 deg (rotation about the linker bonds)
  2. two central variants:
       dihedral90 : central dihedral = 90 deg  (theta_C ~ 83.6 deg, reference)
       plane90    : central dihedral rotated (0.01 deg grid) to the theta_C maximum,
                    then the far side (fragments 3+4) is rotated rigidly about the
                    planes' intersection line, pivoting on atom 9, until
                    theta_C = 90.000 deg. For this geometry the torsion alone gives
                    theta_C = 90.00 at a central dihedral of 83.55 deg and the extra
                    rotation is 0.002 deg, i.e. effectively a pure torsion (bond
                    length and bond angles at 9 and 17 unchanged; see manifest).
The outer plane angles are identical in both variants, so the pair isolates the
effect of theta_C alone. Same level of theory and input layout as
angle_scan_central_approx_90_exchange (every job independent, submit with
submit_functional_scan.py).
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
sys.path.insert(0, str(REPO_ROOT / "visualisation"))
from generate_angle_scan import build_connectivity, fragment_to_rotate, set_dihedral  # noqa: E402
from analyze_angle_scan import build_adjacency, check_chemical_integrity, reference_bonds  # noqa: E402

# ---- USER SETTINGS ----
SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
SOURCE = SCANS / "angle_scan_central_approx_90" / "fixed_outer" / "L045_C090_R045" / "high_spin.xyz"
OUTPUT_ROOT = SCANS / "central_plane90_singlepoints"
BASELINE_XYZ = REPO_ROOT / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
L_TARGET, R_TARGET = 40.0, 80.0
THETA_C_TARGET = 90.0
NPROCS, MAXCORE = 16, 2000
OVERWRITE = False

CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
RINGS = {"L": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]),
         "C": ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25]),
         "R": ([23, 21, 25, 24, 26, 27], [28, 29, 30, 31, 32])}


def normal(pos, ring):
    c = pos[ring].mean(axis=0)
    return np.linalg.svd(pos[ring] - c)[2][-1]


def plane_angle(pos, key):
    n1, n2 = normal(pos, RINGS[key][0]), normal(pos, RINGS[key][1])
    return float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), 0, 1))))


def rotate(pos, idx, axis, pivot, deg):
    axis = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    v = pos[idx] - pivot
    rot = v * np.cos(t) + np.cross(axis, v) * np.sin(t) + np.outer(v @ axis, axis) * (1 - np.cos(t))
    out = pos.copy()
    out[idx] = pivot + rot
    return out


def make_plane90(atoms, conn):
    """Central dihedral to the theta_C maximum, then rigid rotation about the plane intersection line."""
    grid = np.arange(60.0, 120.0, 0.01)
    best_c = max(grid, key=lambda c: plane_angle(_with_c(atoms, conn, c).get_positions(), "C"))
    a = _with_c(atoms, conn, best_c)
    far = fragment_to_rotate(conn, CENTER[1], CENTER[2], CENTER[3], "CENTER")   # atoms on the 17 side
    pos0 = a.get_positions()
    n1, n2 = normal(pos0, RINGS["C"][0]), normal(pos0, RINGS["C"][1])
    axis = np.cross(n1, n2)
    pivot = pos0[CENTER[1]]                                                    # atom 9 keeps the 9-17 bond length
    f = lambda d: abs(normal(rotate(pos0, far, axis, pivot, d), RINGS["C"][0]) @
                      normal(rotate(pos0, far, axis, pivot, d), RINGS["C"][1]))
    ds = np.arange(-5.0, 5.0, 0.001)
    d_best = min(ds, key=f)
    for step in (1e-4, 1e-5):                                                  # refine
        ds = np.arange(d_best - 20 * step, d_best + 20 * step, step)
        d_best = min(ds, key=f)
    a.set_positions(rotate(pos0, far, axis, pivot, d_best))
    return a, best_c, d_best


def _with_c(atoms, conn, c):
    b = atoms.copy()
    set_dihedral(b, conn, CENTER, c, "CENTER")
    return b


def render(mult, guessmix):
    return (f"! PBE0 D4 def2-TZVP TightSCF UKS DEFGRID3 NoAutoStart{' GuessMix' if guessmix else ''} UNO UCO\n\n"
            "%scf\n    MaxIter 500\n    ConvCheckMode 0\nend\n\n"
            "%output\n    Print[P_UNO_OccNum] 1\n    Print[P_Hirshfeld] 1\nend\n\n"
            f"%pal\n    nprocs {NPROCS}\nend\n\n%maxcore {MAXCORE}\n\n* xyzfile 0 {mult} geometry.xyz\n")


def main():
    if OUTPUT_ROOT.exists() and not OVERWRITE:
        sys.exit(f"{OUTPUT_ROOT} exists; set OVERWRITE = True to regenerate.")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    ref = read(BASELINE_XYZ)
    adj = build_adjacency(ref)
    bonds = reference_bonds(adj, ref)
    bonded = {(i, j) for i in range(len(ref)) for j in adj[i] if j > i}

    base = read(SOURCE)
    conn = build_connectivity(base)
    set_dihedral(base, conn, LEFT, L_TARGET, "LEFT")
    set_dihedral(base, conn, RIGHT, R_TARGET, "RIGHT")
    variants = {"L040_C090dih_R080": _with_c(base, conn, 90.0)}
    plane90, c_dih, delta = make_plane90(base, conn)
    variants["L040_Cplane90_R080"] = plane90

    rows = []
    for name, atoms in variants.items():
        pos = atoms.get_positions()
        integ = check_chemical_integrity(atoms, bonds, bonded, 0.30, 1.9, 1.5)
        if not integ["valid"]:
            raise RuntimeError(f"{name}: {integ['reasons']}")
        info = {"point": name,
                "L_dihedral": atoms.get_dihedral(*LEFT), "C_dihedral": atoms.get_dihedral(*CENTER),
                "R_dihedral": atoms.get_dihedral(*RIGHT),
                **{f"theta_{k}_plane": plane_angle(pos, k) for k in "LCR"},
                "bond_9_17_A": atoms.get_distance(9, 17),
                "angle_4_9_17": atoms.get_angle(4, 9, 17), "angle_8_9_17": atoms.get_angle(8, 9, 17),
                "angle_9_17_18": atoms.get_angle(9, 17, 18), "angle_9_17_25": atoms.get_angle(9, 17, 25),
                "min_nonbonded_A": integ["min_nonbonded_distance"], "source": str(SOURCE.relative_to(REPO_ROOT))}
        if "Cplane90" in name:
            info["note"] = (f"central dihedral {c_dih:.2f} (theta_C max) + {delta:+.4f} deg rigid rotation about the "
                            "plane intersection line, pivot atom 9")
        rows.append(info)
        for state, mult, gm in (("high_spin", 3, False), ("broken_symmetry", 1, True)):
            d = OUTPUT_ROOT / name / state
            d.mkdir(parents=True)
            write(d / "geometry.xyz", atoms)
            (d / f"{state}.inp").write_text(render(mult, gm))
        print(f"{name}: L/C/R dihedral {info['L_dihedral']:.2f}/{info['C_dihedral']:.2f}/{info['R_dihedral']:.2f}, "
              f"theta L/C/R {info['theta_L_plane']:.3f}/{info['theta_C_plane']:.3f}/{info['theta_R_plane']:.3f}, "
              f"min non-bonded {info['min_nonbonded_A']:.2f} A")
    src = base.copy()
    with open(OUTPUT_ROOT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows(rows)
    s = read(SOURCE)
    print(f"source angles 4-9-17 {s.get_angle(4, 9, 17):.2f}, 8-9-17 {s.get_angle(8, 9, 17):.2f}, "
          f"9-17-18 {s.get_angle(9, 17, 18):.2f}, 9-17-25 {s.get_angle(9, 17, 25):.2f}")
    print(f"Wrote {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
