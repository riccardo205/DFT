#!/usr/bin/env python3

from pathlib import Path

import csv
import shutil


FUNCTIONALS = [
    ("pbe", "PBE"),
    ("r2scan", "R2SCAN"),
    ("b3lyp", "B3LYP"),
    ("pbe0", "PBE0"),
    ("r2scan0", "R2SCAN0"),
    ("cam_b3lyp", "CAM-B3LYP"),
]


STATES = [
    {
        "folder": "high_spin",
        "input_name": "high_spin.inp",
        "multiplicity": 3,
        "guessmix": False,
    },
    {
        "folder": "broken_symmetry",
        "input_name": "broken_symmetry.inp",
        "multiplicity": 1,
        "guessmix": True,
    },
]


def validate_xyz(xyz_file: Path) -> int:
    lines = xyz_file.read_text(
        encoding="utf-8",
        errors="strict",
    ).splitlines()

    if len(lines) < 3:
        raise ValueError(
            f"XYZ file is too short: {xyz_file}"
        )

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"First XYZ line is not an atom count: {xyz_file}"
        ) from exc

    coordinate_lines = lines[2:]

    if len(coordinate_lines) != natoms:
        raise ValueError(
            f"XYZ atom-count mismatch in {xyz_file}: "
            f"header says {natoms}, "
            f"found {len(coordinate_lines)} coordinate lines"
        )

    return natoms


def make_orca_input(
    functional: str,
    multiplicity: int,
    guessmix: bool,
    basis: str,
    nprocs: int,
    maxcore_mb: int,
    max_iter: int,
) -> str:

    keywords = [
        functional,
        basis,
        "TightSCF",
        "UKS",
        "DEFGRID3",
        "NoAutoStart",
    ]

    if guessmix:
        keywords.append("GuessMix")

    keyword_line = "! " + " ".join(keywords)

    return (
        f"{keyword_line}\n"
        "\n"
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "    ConvCheckMode 0\n"
        "end\n"
        "\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n"
        "\n"
        f"%maxcore {maxcore_mb}\n"
        "\n"
        f"* xyzfile 0 {multiplicity} geometry.xyz\n"
    )


def main():

    # ============================================================
    # USER SETTINGS
    # ============================================================

    # Edit this if your master geometry.xyz is stored elsewhere.
    SOURCE_GEOMETRY = Path(
        "/scratch/rconte1/orca/short-2OS/"
        "main_conf_4_antiferro/geometry.xyz"
    )

    OUTPUT_ROOT = Path(
        "/scratch/rconte1/orca/short-2OS/"
        "main_conf_4_antiferro/functional_scan"
    )

    BASIS = "def2-TZVP"
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500

    # False is safest if a scan already exists.
    OVERWRITE = False

    # ============================================================

    if not SOURCE_GEOMETRY.exists():
        raise FileNotFoundError(
            f"Geometry not found: {SOURCE_GEOMETRY}\n"
            "Edit SOURCE_GEOMETRY in this script."
        )

    if not str(OUTPUT_ROOT).startswith("/scratch/"):
        raise RuntimeError(
            f"OUTPUT_ROOT must be below /scratch: {OUTPUT_ROOT}"
        )

    natoms = validate_xyz(SOURCE_GEOMETRY)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for folder_name, functional in FUNCTIONALS:

        for state in STATES:

            run_folder = (
                OUTPUT_ROOT
                / folder_name
                / state["folder"]
            )

            run_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            geometry_target = run_folder / "geometry.xyz"
            input_target = run_folder / state["input_name"]

            if not OVERWRITE:
                existing = [
                    path
                    for path in [geometry_target, input_target]
                    if path.exists()
                ]

                if existing:
                    print(
                        "SKIP existing   : "
                        + ", ".join(str(path) for path in existing)
                    )

                    manifest_rows.append(
                        {
                            "functional_folder": folder_name,
                            "functional": functional,
                            "state": state["folder"],
                            "input_file": str(input_target),
                            "status": "existing",
                        }
                    )

                    continue

            shutil.copy2(
                SOURCE_GEOMETRY,
                geometry_target,
            )

            input_text = make_orca_input(
                functional=functional,
                multiplicity=state["multiplicity"],
                guessmix=state["guessmix"],
                basis=BASIS,
                nprocs=NPROCS,
                maxcore_mb=MAXCORE_MB,
                max_iter=MAX_ITER,
            )

            input_target.write_text(
                input_text,
                encoding="utf-8",
            )

            print(f"CREATED         : {input_target}")

            manifest_rows.append(
                {
                    "functional_folder": folder_name,
                    "functional": functional,
                    "state": state["folder"],
                    "input_file": str(input_target),
                    "status": "created",
                }
            )

    manifest_file = OUTPUT_ROOT / "manifest.csv"

    with manifest_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "functional_folder",
                "functional",
                "state",
                "input_file",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Geometry atoms : {natoms}")
    print(f"Run root       : {OUTPUT_ROOT}")
    print(f"Manifest       : {manifest_file}")


if __name__ == "__main__":
    main()
