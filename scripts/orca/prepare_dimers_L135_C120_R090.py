#!/usr/bin/env python3
"""
Quintet R2SCAN-3C relaxations of dimers of the L135_C120_R090 monomer
(outer torsions 135/90 deg, central 120 deg), as dimers_run2/run3.

Monomer: HS-relaxed geometry of angle_scans/angle_scan_relaxation/L135_C120_R090.
Two kinds of starting dimers, both built from two copies of that rigid monomer:

  docked   : rigid-body docking scan. For N_DOCK_SAMPLES random orientations of B and
             random approach directions, B is slid in towards A until the first contact
             reaches MIN_HEAVY (C...C) or MIN_ANY (any pair). Poses are ranked by the number
             of C...C pairs closer than CONTACT_CC (a proxy for dispersion / pi overlap);
             the best N_DOCKED poses that differ by more than DOCK_RMSD_MIN (RMSD of B) are kept.
             (Sliding along fixed principal axes was tried first: the twisted monomer then
             only touches at protruding H atoms, centres 11-15 A apart.)
  packmol  : random poses as in Short-2OS/molecule_creation/dimers_creation/dimer.inp
             (A fixed, B inside an 11 Angstrom sphere, 2 Angstrom tolerance), fixed seeds.

All six torsions (L, C, R of both monomers) are constrained during the relaxation;
everything else, including the relative position of the monomers, is free.
SCF/opt settings follow run2 (default SCF convergence check, SCF MaxIter 150, NormalOpt).
All jobs are independent: submit with submit_functional_scan.py.
"""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def render_input(constraints, opt_keyword, nprocs, maxcore_mb, max_iter_scf, max_iter_geom, multiplicity):
    lines = "\n".join(f"        {{ D {a} {b} {c} {d} {angle:.6f} C }}" for (a, b, c, d), angle in constraints)
    return (
        f"! R2SCAN-3C TightSCF {opt_keyword} UKS\n\n"
        "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
        "end\n\n"
        "%geom\n"
        f"    MaxIter {max_iter_geom}\n"
        "    Constraints\n"
        f"{lines}\n"
        "    end\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        f"* xyzfile 0 {multiplicity} geometry.xyz\n"
    )


def random_rotation(rng):
    q = rng.normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def contacts(pa, pb, heavy_a, heavy_b):
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return d.min(), d[np.ix_(heavy_a, heavy_b)].min()


def place_by_contact(monomer, rot, direction, min_any, min_heavy, coarse=0.5, tol=0.01, start=40.0):
    """B = rot(A) about A's centre, slid in along `direction` until the first contact."""
    centre = monomer.positions.mean(0)
    pa = monomer.positions
    pb0 = (pa - centre) @ rot.T + centre
    heavy = monomer.numbers > 1
    direction = direction / np.linalg.norm(direction)

    def clash(shift):
        d_any, d_heavy = contacts(pa, pb0 + shift * direction, heavy, heavy)
        return d_any < min_any or d_heavy < min_heavy

    outer = start
    while outer - coarse > 0 and not clash(outer - coarse):
        outer -= coarse
    inner = outer - coarse                       # clashing (or through the centre)
    while outer - inner > tol:
        mid = 0.5 * (outer + inner)
        outer, inner = (mid, inner) if not clash(mid) else (outer, mid)
    return pb0 + outer * direction, outer


def dock(monomer, n_samples, n_keep, rmsd_min, min_any, min_heavy, contact_cc, seed):
    rng = np.random.default_rng(seed)
    carbons = monomer.numbers == 6
    poses = []
    for _ in range(n_samples):
        direction = rng.normal(size=3)
        pb, shift = place_by_contact(monomer, random_rotation(rng), direction, min_any, min_heavy)
        d = np.linalg.norm(monomer.positions[carbons][:, None] - pb[carbons][None], axis=-1)
        poses.append((int((d < contact_cc).sum()), pb, shift))
    poses.sort(key=lambda p: -p[0])
    kept = []
    for score, pb, shift in poses:
        if all(np.sqrt(((pb - k[1]) ** 2).sum(1).mean()) > rmsd_min for k in kept):
            kept.append((score, pb, shift))
        if len(kept) == n_keep:
            break
    return kept


def packmol_dimer(monomer_xyz, seed, tolerance, sphere_radius, workdir):
    out = workdir / f"packmol_{seed}.xyz"
    inp = (
        f"tolerance {tolerance}\nfiletype xyz\noutput {out.name}\nseed {seed}\n\n"
        f"structure {monomer_xyz.name}\n  number 1\n  fixed 0. 0. 0.  0. 0. 0.\nend structure\n\n"
        f"structure {monomer_xyz.name}\n  number 1\n  inside sphere 0. 0. 0. {sphere_radius}\nend structure\n"
    )
    (workdir / f"packmol_{seed}.inp").write_text(inp)
    with (workdir / f"packmol_{seed}.inp").open() as stdin:
        subprocess.run(["packmol"], stdin=stdin, cwd=workdir, check=True, capture_output=True)
    return read(out)


def main():
    # ---- USER SETTINGS ----
    MONOMER_LABEL = "L135_C120_R090"
    MONOMER_XYZ = (REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation"
                   / MONOMER_LABEL / "high_spin.xyz")
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2" / "run3"

    # monomer torsions (0-based ORCA indices) and their values
    TORSIONS = {"LEFT": ((42, 41, 0, 5), 135.0), "CENTER": ((4, 9, 17, 18), 120.0), "RIGHT": ((21, 23, 28, 29), 90.0)}

    MIN_ANY = 2.4               # Angstrom, first contact for docked dimers (H...H)
    MIN_HEAVY = 3.4             # Angstrom, first C...C contact for docked dimers
    CONTACT_CC = 4.5            # Angstrom, C...C pairs counted in the docking score
    N_DOCK_SAMPLES = 3000
    N_DOCKED = 6
    DOCK_RMSD_MIN = 3.0         # Angstrom, minimum RMSD of B between kept poses
    DOCK_SEED = 0
    PACKMOL_SEEDS = [1, 2, 3]
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
    for name, (quartet, target) in TORSIONS.items():
        if abs((monomer.get_dihedral(*quartet) - target + 180) % 360 - 180) > 0.05:
            raise ValueError(f"{MONOMER_XYZ}: {name} torsion is {monomer.get_dihedral(*quartet):.3f}, expected {target}")

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

    constraints = [(tuple(i + offset for i in quartet), target)
                   for offset in (0, n) for quartet, target in TORSIONS.values()]

    manifest_rows = []
    for name, (dimer, how) in dimers.items():
        if len(dimer) != 2 * n or list(dimer.numbers[:n]) != list(monomer.numbers) \
                or list(dimer.numbers[n:]) != list(monomer.numbers):
            raise ValueError(f"{name}: atom order differs from monomer")
        for (quartet, target) in constraints:
            if abs((dimer.get_dihedral(*quartet) - target + 180) % 360 - 180) > 0.05:
                raise ValueError(f"{name}: torsion {quartet} is {dimer.get_dihedral(*quartet):.3f}, expected {target}")
        heavy = dimer.numbers[:n] > 1
        d_any, d_heavy = contacts(dimer.positions[:n], dimer.positions[n:], heavy, heavy)
        n_cc = int((np.linalg.norm(dimer.positions[:n][heavy][:, None] - dimer.positions[n:][heavy][None], axis=-1)
                    < CONTACT_CC).sum())
        com_sep = np.linalg.norm(dimer.positions[n:].mean(0) - dimer.positions[:n].mean(0))
        dimer.positions -= dimer.positions.mean(0)

        run_folder = OUTPUT_ROOT / name / "high_spin"
        run_folder.mkdir(parents=True)
        write(run_folder / "geometry.xyz", dimer, comment=f"{name}: 2 x {MONOMER_LABEL}, {how}")
        (run_folder / "high_spin.inp").write_text(
            render_input(constraints, OPT_KEYWORD, NPROCS, MAXCORE_MB, MAX_ITER_SCF, MAX_ITER_GEOM, MULTIPLICITY),
            encoding="utf-8",
        )
        manifest_rows.append({
            "dimer": name,
            "construction": how,
            "monomer_source": str(MONOMER_XYZ.relative_to(REPO_ROOT)),
            "n_CC_below_contact": n_cc,
            "min_contact_A": round(d_any, 3),
            "min_CC_contact_A": round(d_heavy, 3),
            "centre_separation_A": round(com_sep, 3),
            "opt": OPT_KEYWORD,
            "multiplicity": MULTIPLICITY,
            "input_file": f"{name}/high_spin/high_spin.inp",
        })
        print(f"{name:12s} {how:24s} C-C<{CONTACT_CC}: {n_cc:3d}, min contact {d_any:.2f} A, min C-C {d_heavy:.2f} A, centres {com_sep:.2f} A")

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print(f"{MONOMER_LABEL} dimers prepared (run3, quintet, 6 torsions constrained)")
    print("=" * 70)
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")


if __name__ == "__main__":
    main()
