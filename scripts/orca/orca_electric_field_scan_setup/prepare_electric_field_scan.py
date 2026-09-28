#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil


def field_vector_str(axis, value):
    components = {"x": 0.0, "y": 0.0, "z": 0.0}
    components[axis] = value
    return f"{components['x']:g}, {components['y']:g}, {components['z']:g}"


def render_high_spin_input(field_vector, nprocs, maxcore_mb, max_iter):
    return (
        "! PBE0 D4 def2-TZVP TightSCF UKS DEFGRID3 NoAutoStart\n\n"
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        f"    EField {field_vector}\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        "* xyzfile 0 3 geometry.xyz\n"
    )


def render_broken_symmetry_input(field_vector, nprocs, maxcore_mb, max_iter):
    return (
        "! PBE0 def2-TZVP TightSCF UKS DEFGRID3 NoAutoStart D4 GuessMix\n\n"
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "    ConvCheckMode 0\n"
        f"    EField {field_vector}\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        "* xyzfile 0 1 geometry.xyz\n"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def main():
    # USER SETTINGS
    GEOMETRY_FILE = REPO_ROOT / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "main_conf_4_antiferro" / "electric_field_scan"

    # Which Cartesian axes to generate a field scan for.
    AXES = ["x"]

    # Field strengths, atomic units (Eh / (e a0)), applied one axis at a time.
    FIELD_VALUES = [-0.005, -0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002, 0.005]

    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500

    # Existing field_<axis><value> folders are left untouched (never deleted or
    # regenerated) so completed/running ORCA jobs are never clobbered. Set to True
    # only to regenerate a specific folder that has no results in it yet.
    OVERWRITE_EXISTING_FOLDERS = False

    STATES = [
        {
            "folder": "high_spin",
            "input_name": "high_spin.inp",
            "multiplicity": 3,
            "render": render_high_spin_input,
        },
        {
            "folder": "broken_symmetry",
            "input_name": "broken_symmetry.inp",
            "multiplicity": 1,
            "render": render_broken_symmetry_input,
        },
    ]

    if not GEOMETRY_FILE.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {GEOMETRY_FILE.resolve()}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_file = OUTPUT_ROOT / "manifest.csv"
    manifest_rows = []
    if manifest_file.exists():
        with manifest_file.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))

        # Migrate the older single-axis manifest format (field_z_au, no axis column).
        for row in manifest_rows:
            if "axis" not in row:
                row["axis"] = "z"
                row["field_au"] = row.pop("field_z_au")

    seen_keys = {(row["axis"], row["field_au"], row["state"]) for row in manifest_rows}

    created = 0
    skipped = 0

    for axis in AXES:
        for field_value in FIELD_VALUES:
            field_folder = OUTPUT_ROOT / f"field_{axis}{field_value:g}"
            field_vector = field_vector_str(axis, field_value)

            for state in STATES:
                run_folder = field_folder / state["folder"]
                tag = f"field_{axis}{field_value:g}/{state['folder']}"

                if run_folder.exists() and not OVERWRITE_EXISTING_FOLDERS:
                    print(f"SKIP existing  : {tag}")
                    skipped += 1
                    continue

                if run_folder.exists() and OVERWRITE_EXISTING_FOLDERS:
                    shutil.rmtree(run_folder)

                run_folder.mkdir(parents=True, exist_ok=False)

                shutil.copy2(
                    GEOMETRY_FILE,
                    run_folder / "geometry.xyz",
                )

                input_file = run_folder / state["input_name"]
                input_text = state["render"](field_vector, NPROCS, MAXCORE_MB, MAX_ITER)
                input_file.write_text(input_text, encoding="utf-8")

                key = (axis, f"{field_value:g}", state["folder"])
                if key not in seen_keys:
                    manifest_rows.append(
                        {
                            "axis": axis,
                            "field_au": f"{field_value:g}",
                            "state": state["folder"],
                            "multiplicity": state["multiplicity"],
                            "nprocs": NPROCS,
                            "maxcore_mb": MAXCORE_MB,
                            "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
                        }
                    )
                    seen_keys.add(key)

                print(f"Created        : {tag}")
                created += 1

    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["axis", "field_au", "state", "multiplicity", "nprocs", "maxcore_mb", "input_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Electric field scan prepared")
    print("=" * 70)
    print(f"Geometry : {GEOMETRY_FILE.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Created  : {created}")
    print(f"Skipped  : {skipped}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
