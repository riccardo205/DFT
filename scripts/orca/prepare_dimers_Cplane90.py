#!/usr/bin/env python3
"""
Quintet R2SCAN-3C relaxations of dimers of the central-plane-90 monomer, as dimers_run2/run4.

Monomer: angle_scans/central_plane90_singlepoints/L040_Cplane90_R080 (rigid single-point geometry,
outer dihedrals 40/80 deg, central ring planes exactly orthogonal, theta_C = 90.000 deg, reached
with a central dihedral 4-9-17-18 of 83.55 deg).

Dimers (same construction as prepare_dimers_L135_C120_R090.py / run3): N_DOCKED rigid-body docked
poses + packmol poses. The SAME starting dimers are relaxed in three ways:

  fixC_dihedral/ : central dihedral 4-9-17-18 of both monomers fixed at its starting value
                   (83.55 deg, the value giving theta_C = 90 in the isolated monomer); L, R free.
  fixC_double90/ : two opposite central dihedrals of each monomer fixed at 90 deg,
                   4-9-17-18 and 8-9-17-25 (83.55 and 94.04 deg at the start), so the asymmetric
                   ring tilt at the 9-17 junction is removed and the planes are held near 90 deg;
                   L, R free.
  free/          : no constraints.

All jobs are independent: submit with submit_functional_scan.py.
"""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_dimers_L135_C120_R090 import contacts, dock, packmol_dimer  # noqa: E402

RINGS_C = ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25])


def render_input(constraints, opt_keyword, nprocs, maxcore_mb, max_iter_scf, max_iter_geom, multiplicity):
    geom = f"%geom\n    MaxIter {max_iter_geom}\n"
    if constraints:
        lines = "\n".join(f"        {{ D {a} {b} {c} {d} {angle:.6f} C }}" for (a, b, c, d), angle in constraints)
        geom += f"    Constraints\n{lines}\n    end\n"
    geom += "end\n\n"
    return (
        f"! R2SCAN-3C TightSCF {opt_keyword} UKS\n\n"
        "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
        "end\n\n"
        f"{geom}"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        f"* xyzfile 0 {multiplicity} geometry.xyz\n"
    )


def theta_c(pos, offset=0):
    normals = []
    for ring in RINGS_C:
        p = pos[[i + offset for i in ring]]
        normals.append(np.linalg.svd(p - p.mean(0))[2][-1])
    return float(np.degrees(np.arccos(np.clip(abs(normals[0] @ normals[1]), 0, 1))))


def main():
    # ---- USER SETTINGS ----
    MONOMER_LABEL = "L040_Cplane90_R080"
    MONOMER_XYZ = (REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "central_plane90_singlepoints"
                   / MONOMER_LABEL / "high_spin" / "geometry.xyz")
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2" / "run4"

    CENTER = (4, 9, 17, 18)
    CENTER_OPPOSITE = (8, 9, 17, 25)
    VARIANTS = {
        "fixC_dihedral": "start",        # CENTER fixed at the monomer's starting value
        "fixC_double90": 90.0,           # CENTER and CENTER_OPPOSITE both fixed at 90
        "free": None,                    # no constraints
    }

    MIN_ANY = 2.4               # Angstrom, first contact for docked dimers (H...H)
    MIN_HEAVY = 3.4             # Angstrom, first C...C contact for docked dimers
    CONTACT_CC = 4.5            # Angstrom, C...C pairs counted in the docking score
    N_DOCK_SAMPLES = 3000
    N_DOCKED = 4
    DOCK_RMSD_MIN = 3.0         # Angstrom, minimum RMSD of B between kept poses
    DOCK_SEED = 0
    PACKMOL_SEEDS = [1, 2]
    PACKMOL_TOLERANCE = 2.0
    PACKMOL_SPHERE = 11.0

    OPT_KEYWORD = "NormalOpt"
    MULTIPLICITY = 5
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 150
    MAX_ITER_GEOM = 300

    OVERWRITE = False

    # ---- SETUP ----
    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(f"Output directory already exists: {OUTPUT_ROOT.resolve()}")
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    monomer = read(MONOMER_XYZ)
    n = len(monomer)
    c_start = float(monomer.get_dihedral(*CENTER))
    print(f"Monomer {MONOMER_LABEL}: central dihedral {c_start:.3f}, opposite {monomer.get_dihedral(*CENTER_OPPOSITE):.3f}, "
          f"theta_C {theta_c(monomer.positions):.3f} deg")

    dimers = {}
    docked = dock(monomer, N_DOCK_SAMPLES, N_DOCKED, DOCK_RMSD_MIN, MIN_ANY, MIN_HEAVY, CONTACT_CC, DOCK_SEED)
    for rank, (score, pb, shift) in enumerate(docked, start=1):
        dimers[f"docked_{rank}"] = (Atoms(np.concatenate([monomer.numbers] * 2),
                                          positions=np.vstack([monomer.positions, pb])), f"docked, {score} C-C<{CONTACT_CC}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write(tmp / "monomer.xyz", monomer)
        for seed in PACKMOL_SEEDS:
            dimers[f"packmol_{seed}"] = (packmol_dimer(tmp / "monomer.xyz", seed, PACKMOL_TOLERANCE,
                                                       PACKMOL_SPHERE, tmp), f"packmol seed {seed}")

    constraint_sets = {}
    for variant, spec in VARIANTS.items():
        if spec is None:
            constraint_sets[variant] = []
        elif spec == "start":
            constraint_sets[variant] = [(tuple(i + off for i in CENTER), c_start) for off in (0, n)]
        else:
            constraint_sets[variant] = [(tuple(i + off for i in q), spec) for off in (0, n) for q in (CENTER, CENTER_OPPOSITE)]

    manifest_rows = []
    for name, (dimer, how) in dimers.items():
        if len(dimer) != 2 * n or list(dimer.numbers[:n]) != list(monomer.numbers) \
                or list(dimer.numbers[n:]) != list(monomer.numbers):
            raise ValueError(f"{name}: atom order differs from monomer")
        # rigid copies: both monomers must keep the starting central geometry
        for off in (0, n):
            if abs(dimer.get_dihedral(*(i + off for i in CENTER)) - c_start) > 0.05 or abs(theta_c(dimer.positions, off) - 90) > 0.05:
                raise ValueError(f"{name}: monomer at offset {off} is not the starting monomer geometry")
        heavy = dimer.numbers[:n] > 1
        d_any, d_heavy = contacts(dimer.positions[:n], dimer.positions[n:], heavy, heavy)
        n_cc = int((np.linalg.norm(dimer.positions[:n][heavy][:, None] - dimer.positions[n:][heavy][None], axis=-1)
                    < CONTACT_CC).sum())
        com_sep = np.linalg.norm(dimer.positions[n:].mean(0) - dimer.positions[:n].mean(0))
        dimer.positions -= dimer.positions.mean(0)

        for variant, constraints in constraint_sets.items():
            run_folder = OUTPUT_ROOT / variant / name / "high_spin"
            run_folder.mkdir(parents=True)
            write(run_folder / "geometry.xyz", dimer, comment=f"{name}: 2 x {MONOMER_LABEL}, {how}")
            (run_folder / "high_spin.inp").write_text(
                render_input(constraints, OPT_KEYWORD, NPROCS, MAXCORE_MB, MAX_ITER_SCF, MAX_ITER_GEOM, MULTIPLICITY),
                encoding="utf-8",
            )
            manifest_rows.append({
                "variant": variant,
                "dimer": name,
                "construction": how,
                "constraints": "; ".join(f"D {' '.join(map(str, q))} = {v:.3f}" for q, v in constraints) or "none",
                "monomer_source": str(MONOMER_XYZ.relative_to(REPO_ROOT)),
                "n_CC_below_contact": n_cc,
                "min_contact_A": round(d_any, 3),
                "min_CC_contact_A": round(d_heavy, 3),
                "centre_separation_A": round(com_sep, 3),
                "opt": OPT_KEYWORD,
                "multiplicity": MULTIPLICITY,
                "input_file": f"{variant}/{name}/high_spin/high_spin.inp",
            })
        print(f"{name:12s} {how:24s} C-C<{CONTACT_CC}: {n_cc:3d}, min contact {d_any:.2f} A, min C-C {d_heavy:.2f} A, centres {com_sep:.2f} A")

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print(f"{MONOMER_LABEL} dimers prepared (run4, quintet, {len(VARIANTS)} constraint variants)")
    print("=" * 70)
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")


if __name__ == "__main__":
    main()
