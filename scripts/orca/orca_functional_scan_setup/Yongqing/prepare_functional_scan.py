#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
import csv
import shutil


def write_orca_input(
    output_file: Path,
    functional_keyword: str,
    basis: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
    maxcore_mb: int,
    max_iter: int,
    guess_mix: bool,
    dispersion: Optional[str],
):
    keywords = [
        functional_keyword,
        basis,
        "TightSCF",
        "UKS",
        "DEFGRID3",
        "NoAutoStart",
    ]

    if dispersion:
        keywords.append(dispersion)

    if guess_mix:
        keywords.append("GuessMix")

    text = (
        "! " + " ".join(keywords) + "\n\n"
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "    ConvCheckMode 0\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        f"* xyzfile {charge} {multiplicity} geometry.xyz\n"
    )

    output_file.write_text(text, encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def main():
    # USER SETTINGS
    GEOMETRY_FILE = SCRIPT_DIR / "geometries" / "relaxed_main_conf_4.xyz"
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "main_conf_4_antiferro" / "relaxed_main_conf_4" / "functional_scan_D4"

    BASIS = "def2-TZVP"
    CHARGE = 0

    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500
    DISPERSION = "D4"  # e.g. "D4", or None to disable

    OVERWRITE = False

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
            "guess_mix": False,
        },
        {
            "folder": "broken_symmetry",
            "input_name": "broken_symmetry.inp",
            "multiplicity": 1,
            "guess_mix": True,
        },
    ]

    if not GEOMETRY_FILE.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {GEOMETRY_FILE.resolve()}"
        )

    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n"
            f"  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )

    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    first_line = GEOMETRY_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    try:
        int(first_line)
    except ValueError as exc:
        raise ValueError(
            f"{GEOMETRY_FILE} does not look like an XYZ file: "
            f"first line is not an atom count."
        ) from exc

    manifest_rows = []

    for functional_folder, functional_keyword in FUNCTIONALS:
        for state in STATES:
            run_folder = OUTPUT_ROOT / functional_folder / state["folder"]
            run_folder.mkdir(parents=True, exist_ok=False)

            shutil.copy2(
                GEOMETRY_FILE,
                run_folder / "geometry.xyz",
            )

            input_file = run_folder / state["input_name"]

            write_orca_input(
                output_file=input_file,
                functional_keyword=functional_keyword,
                basis=BASIS,
                charge=CHARGE,
                multiplicity=state["multiplicity"],
                nprocs=NPROCS,
                maxcore_mb=MAXCORE_MB,
                max_iter=MAX_ITER,
                guess_mix=state["guess_mix"],
                dispersion=DISPERSION,
            )

            manifest_rows.append(
                {
                    "functional_folder": functional_folder,
                    "functional_keyword": functional_keyword,
                    "state": state["folder"],
                    "charge": CHARGE,
                    "multiplicity": state["multiplicity"],
                    "basis": BASIS,
                    "dispersion": DISPERSION or "",
                    "nprocs": NPROCS,
                    "maxcore_mb": MAXCORE_MB,
                    "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
                }
            )

            print(f"Created: {functional_folder}/{state['folder']}")

    manifest_file = OUTPUT_ROOT / "manifest.csv"

    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Functional scan prepared")
    print("=" * 70)
    print(f"Geometry : {GEOMETRY_FILE.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
