#!/usr/bin/env python3

from pathlib import Path
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
):
    keywords = [
        functional_keyword,
        basis,
        "TightSCF",
        "UKS",
        "DEFGRID3",
        "NoAutoStart",
    ]

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


def relaxation_converged(out_file: Path) -> bool:
    if not out_file.exists():
        return False
    return "ORCA TERMINATED NORMALLY" in out_file.read_text(
        encoding="utf-8", errors="ignore"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def main():
    # USER SETTINGS
    SOURCE_ROOT = (
        REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scan_relaxation"
    )
    OUTPUT_ROOT = (
        REPO_ROOT / "runs" / "orca" / "short-2OS"
        / "angle_scan_exchange_analysis"
    )

    FUNCTIONAL_KEYWORD = "PBE0"
    BASIS = "def2-TZVP"
    CHARGE = 0

    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500

    OVERWRITE = False

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

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    relaxation_dirs = sorted(
        d for d in SOURCE_ROOT.iterdir()
        if d.is_dir() and d.name != "plots"
    )

    manifest_rows = []

    for relax_dir in relaxation_dirs:
        geometry_source = relax_dir / "high_spin.xyz"
        if not geometry_source.exists():
            print(f"SKIP (no high_spin.xyz): {relax_dir.name}")
            continue

        converged = relaxation_converged(relax_dir / "high_spin.out")
        scan_name = relax_dir.name if converged else f"{relax_dir.name}_unconverged"

        for state in STATES:
            run_folder = OUTPUT_ROOT / scan_name / state["folder"]

            if run_folder.exists() and not OVERWRITE:
                print(f"SKIP existing: {scan_name}/{state['folder']}")
                continue

            run_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy2(geometry_source, run_folder / "geometry.xyz")

            write_orca_input(
                output_file=run_folder / state["input_name"],
                functional_keyword=FUNCTIONAL_KEYWORD,
                basis=BASIS,
                charge=CHARGE,
                multiplicity=state["multiplicity"],
                nprocs=NPROCS,
                maxcore_mb=MAXCORE_MB,
                max_iter=MAX_ITER,
                guess_mix=state["guess_mix"],
            )

            manifest_rows.append(
                {
                    "scan_point": relax_dir.name,
                    "relaxation_converged": converged,
                    "output_folder": scan_name,
                    "state": state["folder"],
                    "multiplicity": state["multiplicity"],
                    "input_file": str(
                        (OUTPUT_ROOT / scan_name / state["folder"] / state["input_name"])
                        .relative_to(OUTPUT_ROOT)
                    ),
                }
            )

        print(f"Created: {scan_name}")

    manifest_file = OUTPUT_ROOT / "manifest.csv"

    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_unconverged = sum(1 for d in relaxation_dirs if not relaxation_converged(d / "high_spin.out"))

    print()
    print("=" * 70)
    print("Angle scan exchange analysis prepared")
    print("=" * 70)
    print(f"Source   : {SOURCE_ROOT.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Scan points : {len(relaxation_dirs)} ({n_unconverged} marked _unconverged)")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
