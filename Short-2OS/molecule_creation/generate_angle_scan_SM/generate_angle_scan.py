#!/usr/bin/env python3

import shutil
from itertools import product
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import NeighborList, get_connectivity_matrix, natural_cutoffs
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent


def build_connectivity(atoms):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    # ase.neighborlist.get_connectivity_matrix(nl, ...) reads nl.cutoffs, which only
    # exists on the primitive list (nl.nl), not on the NeighborList wrapper itself.
    return get_connectivity_matrix(nl.nl, sparse=True)


def fragment_to_rotate(conn, b, c, d, bond_name):
    if conn[b, c] == 0:
        raise ValueError(
            f"{bond_name}: atoms {b}-{c} are not bonded according to "
            f"the connectivity built from natural_cutoffs()."
        )

    cut = lil_matrix(conn, copy=True)
    cut[b, c] = 0
    cut[c, b] = 0

    n_components, labels = connected_components(cut.tocsr(), directed=False)
    if n_components != 2:
        raise ValueError(
            f"{bond_name}: removing bond {b}-{c} gave {n_components} "
            f"fragment(s) instead of 2 (the bond is likely part of a ring)."
        )

    return np.where(labels == labels[d])[0]


def set_dihedral(atoms, conn, quartet, angle, bond_name):
    a, b, c, d = quartet
    rotating_fragment = fragment_to_rotate(conn, b, c, d, bond_name)
    atoms.set_dihedral(a, b, c, d, angle, indices=rotating_fragment)


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def fmt_angle(angle):
    normalized = angle % 360.0
    if abs(normalized - round(normalized)) < 1e-6:
        return f"{int(round(normalized)):03d}"
    return f"{normalized:06.2f}".replace(".", "p")


def render_orca_input(targets, charge, multiplicity, nprocs, maxcore_mb, max_iter_scf, max_iter_geom):
    constraint_lines = "\n".join(
        f"        {{ D {a} {b} {c} {d} {angle:.6f} C }}"
        for _, (a, b, c, d), angle in targets
    )
    return (
        "! R2SCAN-3C TightSCF TightOpt UKS\n\n"
        "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
        "end\n\n"
        "%geom\n"
        f"    MaxIter {max_iter_geom}\n"
        "    Constraints\n"
        f"{constraint_lines}\n"
        "    end\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        f"* xyzfile {charge} {multiplicity} geometry.xyz\n"
    )


def generate_geometry(atoms0, conn, left_angle, center_angle, right_angle, quartets, tol_deg):
    atoms = atoms0.copy()

    order = [
        ("CENTER", quartets["CENTER"], center_angle),
        ("LEFT", quartets["LEFT"], left_angle),
        ("RIGHT", quartets["RIGHT"], right_angle),
    ]
    for name, quartet, angle in order:
        set_dihedral(atoms, conn, quartet, angle, name)

    obtained = {name: atoms.get_dihedral(*quartet) for name, quartet, _ in order}
    target = {name: angle for name, _, angle in order}

    print(
        f"Target:   L={target['LEFT']:.3f} "
        f"C={target['CENTER']:.3f} R={target['RIGHT']:.3f}"
    )
    print(
        f"Obtained: L={obtained['LEFT']:.3f} "
        f"C={obtained['CENTER']:.3f} R={obtained['RIGHT']:.3f}"
    )

    for name, _, angle in order:
        if abs(angle_diff_deg(obtained[name], angle)) > tol_deg:
            raise RuntimeError(
                f"{name} dihedral off target by more than {tol_deg} deg: "
                f"target={angle:.3f} obtained={obtained[name]:.3f}"
            )

    return atoms, order


def build_angle_combinations(mode, left_angles, center_angles, right_angles, explicit_combinations):
    if mode == "explicit":
        return list(explicit_combinations)
    if mode == "product":
        return list(product(left_angles, center_angles, right_angles))
    raise ValueError(f"Unknown MODE: {mode!r} (expected 'product' or 'explicit')")


def main():
    # ---- USER SETTINGS ----
    INPUT_GEOMETRY = HERE / "relaxed_main_conf_4.xyz"
    OUTPUT_ROOT = HERE / "angle_scan"

    # Zero-based ASE/ORCA atom indices (a, b, c, d) for each dihedral.
    LEFT = (42, 41, 0, 5)
    CENTER = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)

    # MODE = "product"  -> Cartesian product of LEFT_ANGLES x CENTER_ANGLES x RIGHT_ANGLES
    #                      (use single-element lists to keep an angle fixed -> 1D scan)
    # MODE = "explicit" -> use ANGLE_COMBINATIONS as explicit (left, center, right) triples
    MODE = "product"

    LEFT_ANGLES = np.arange(0, 190, 45).tolist()  # [0, 10, 20, ..., 180]
    CENTER_ANGLES = np.arange(20, 190, 20).tolist()  # [0, 10, 20, ..., 180]
    RIGHT_ANGLES = np.arange(0, 190, 45).tolist()  # [0, 10, 20, ..., 180]

    ANGLE_COMBINATIONS = [
        (40, 60, 50),
        (40, 70, 50),
    ]

    ANGLE_TOLERANCE_DEG = 0.01

    CHARGE = 0
    MULTIPLICITY = 3
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 500
    MAX_ITER_GEOM = 200

    OVERWRITE = True

    # ---- SETUP ----
    if not INPUT_GEOMETRY.exists():
        raise FileNotFoundError(f"Geometry file not found: {INPUT_GEOMETRY.resolve()}")

    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    atoms0 = read(INPUT_GEOMETRY)
    conn = build_connectivity(atoms0)

    quartets = {"LEFT": LEFT, "CENTER": CENTER, "RIGHT": RIGHT}

    print("Original torsions:")
    for name, quartet in quartets.items():
        fragment_to_rotate(conn, quartet[1], quartet[2], quartet[3], name)
        print(f"  {name} {quartet}: {atoms0.get_dihedral(*quartet):.3f} deg")
    print()

    combinations = build_angle_combinations(
        MODE, LEFT_ANGLES, CENTER_ANGLES, RIGHT_ANGLES, ANGLE_COMBINATIONS
    )

    for left_angle, center_angle, right_angle in combinations:
        tag = f"L{fmt_angle(left_angle)}_C{fmt_angle(center_angle)}_R{fmt_angle(right_angle)}"
        print(f"--- {tag} ---")

        atoms, order = generate_geometry(
            atoms0, conn, left_angle, center_angle, right_angle, quartets, ANGLE_TOLERANCE_DEG
        )

        run_folder = OUTPUT_ROOT / tag
        run_folder.mkdir(parents=True, exist_ok=False)

        write(run_folder / "geometry.xyz", atoms)

        targets = [(name, quartet, angle) for name, quartet, angle in order]
        input_text = render_orca_input(
            targets,
            charge=CHARGE,
            multiplicity=MULTIPLICITY,
            nprocs=NPROCS,
            maxcore_mb=MAXCORE_MB,
            max_iter_scf=MAX_ITER_SCF,
            max_iter_geom=MAX_ITER_GEOM,
        )
        (run_folder / "high_spin.inp").write_text(input_text, encoding="utf-8")

        print(f"Created: {tag}")
        print()

    print("=" * 70)
    print("Angle scan prepared")
    print("=" * 70)
    print(f"Geometry : {INPUT_GEOMETRY.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(combinations)}")


if __name__ == "__main__":
    main()
