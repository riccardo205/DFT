#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil


def render_high_spin_input(nprocs, maxcore_mb, max_iter_scf, max_iter_geom):
    return (
        "! R2SCAN-3C TightSCF TightOpt UKS\n\n"
        "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
        "    ConvCheckMode 0\n"
        "end\n\n"
        "%geom\n"
        f"    MaxIter {max_iter_geom}\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        "* xyzfile 0 3 geometry.xyz\n"
    )


def render_broken_symmetry_input(nprocs, maxcore_mb, max_iter_scf, max_iter_geom):
    return (
        "! R2SCAN-3C TightSCF TightOpt UKS GuessMix\n\n"
        "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
        "    ConvCheckMode 0\n"
        "end\n\n"
        "%geom\n"
        f"    MaxIter {max_iter_geom}\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        "* xyzfile 0 1 geometry.xyz\n"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    # USER SETTINGS
    SOURCE_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation"
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_free_relaxation"

    CENTRAL_ANGLES = ["080", "100"]
    SIDE_ANGLES = ["000", "045", "090", "135"]  # symmetric L=R initializations

    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 500
    MAX_ITER_GEOM = 200

    OVERWRITE = False

    STATES = [
        {"folder": "high_spin", "input_name": "high_spin.inp", "multiplicity": 3, "render": render_high_spin_input},
        {"folder": "broken_symmetry", "input_name": "broken_symmetry.inp", "multiplicity": 1, "render": render_broken_symmetry_input},
    ]

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source scan root not found: {SOURCE_ROOT.resolve()}")

    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    manifest_rows = []

    for central in CENTRAL_ANGLES:
        for side in SIDE_ANGLES:
            source_label = f"L{side}_C{central}_R{side}"
            source_folder = SOURCE_ROOT / source_label
            source_geometry = source_folder / "high_spin.xyz"

            if not source_geometry.exists():
                raise FileNotFoundError(
                    f"No optimized geometry found for {source_label}: {source_geometry.resolve()}"
                )

            point_folder = OUTPUT_ROOT / source_label
            point_folder.mkdir(parents=True, exist_ok=False)

            for state in STATES:
                run_folder = point_folder / state["folder"]
                run_folder.mkdir(parents=True, exist_ok=False)

                shutil.copy2(source_geometry, run_folder / "geometry.xyz")

                input_file = run_folder / state["input_name"]
                input_text = state["render"](NPROCS, MAXCORE_MB, MAX_ITER_SCF, MAX_ITER_GEOM)
                input_file.write_text(input_text, encoding="utf-8")

                manifest_rows.append(
                    {
                        "source_point": source_label,
                        "source_geometry": str(source_geometry.relative_to(REPO_ROOT)),
                        "central_angle": central,
                        "side_angle": side,
                        "state": state["folder"],
                        "multiplicity": state["multiplicity"],
                        "nprocs": NPROCS,
                        "maxcore_mb": MAXCORE_MB,
                        "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
                    }
                )

                print(f"Created: {source_label}/{state['folder']}")

    manifest_file = OUTPUT_ROOT / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Free-relaxation follow-up scan prepared")
    print("=" * 70)
    print(f"Source   : {SOURCE_ROOT.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
