#!/usr/bin/env python3

# Prepare ORCA high-spin (quintet) relaxations for the Short-2OS dimer
# configurations of the SIESTA run dimers_run2/slow_lowT.
#
# In SIESTA every conf_N was started from the same geometry with four spin
# initializations (antiferro1, antiferro2, ferro, spin2). Here only one state
# is kept: all four radical spins parallel -> charge 0, multiplicity 5.
# The starting geometry is the input geometry of the SIESTA runs
# (AtomicCoordinatesAndAtomicSpecies block of the ferro .fdf).

from pathlib import Path
import csv
import re
import shutil


def render_high_spin_input(nprocs, maxcore_mb, max_iter_scf, max_iter_geom, multiplicity):
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
        f"* xyzfile 0 {multiplicity} geometry.xyz\n"
    )


def read_fdf_geometry(fdf_file):
    text = fdf_file.read_text()

    species = {}
    species_block = re.search(
        r"%block ChemicalSpeciesLabel(.*?)%endblock ChemicalSpeciesLabel", text, re.S
    ).group(1)
    for line in species_block.splitlines():
        fields = line.split("#")[0].split()
        if len(fields) >= 3:
            species[int(fields[0])] = fields[2]

    fmt = re.search(r"^\s*AtomicCoordinatesFormat\s+(\S+)", text, re.M | re.I).group(1)
    if fmt.lower() not in ("ang", "notscaledcartesianang"):
        raise ValueError(f"Unsupported AtomicCoordinatesFormat '{fmt}' in {fdf_file}")

    coord_block = re.search(
        r"%block AtomicCoordinatesAndAtomicSpecies(.*?)%endblock AtomicCoordinatesAndAtomicSpecies",
        text,
        re.S,
    ).group(1)
    atoms = []
    for line in coord_block.splitlines():
        fields = line.split("#")[0].split()
        if len(fields) >= 4:
            x, y, z = (float(v) for v in fields[:3])
            atoms.append((species[int(fields[3])], x, y, z))

    n_atoms = int(re.search(r"^\s*NumberOfAtoms\s+(\d+)", text, re.M).group(1))
    if len(atoms) != n_atoms:
        raise ValueError(f"{fdf_file}: read {len(atoms)} atoms, NumberOfAtoms = {n_atoms}")

    return atoms


def write_xyz(path, atoms, comment):
    lines = [str(len(atoms)), comment]
    lines += [f"{el:<2s} {x:14.6f} {y:14.6f} {z:14.6f}" for el, x, y, z in atoms]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    # USER SETTINGS
    SOURCE_ROOT = REPO_ROOT / "Short-2OS" / "runs" / "siesta" / "dimers_run2" / "slow_lowT"
    OUTPUT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2_slow_lowT"

    SOURCE_SPIN_STATE = "ferro"  # all SIESTA spin variants share the same starting geometry
    CHECK_SPIN_VARIANTS = ["antiferro1", "antiferro2", "ferro", "spin2"]

    MULTIPLICITY = 5  # four parallel spins
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 500
    MAX_ITER_GEOM = 200

    OVERWRITE = False

    STATE_FOLDER = "high_spin"
    INPUT_NAME = "high_spin.inp"

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SIESTA run root not found: {SOURCE_ROOT.resolve()}")

    conf_ids = sorted(
        {
            int(m.group(1))
            for p in SOURCE_ROOT.iterdir()
            if (m := re.fullmatch(r"conf_(\d+)_\w+", p.name)) and p.is_dir()
        }
    )
    if not conf_ids:
        raise FileNotFoundError(f"No conf_* folders found in {SOURCE_ROOT.resolve()}")

    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output directory already exists:\n  {OUTPUT_ROOT.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    manifest_rows = []

    for conf_id in conf_ids:
        conf_label = f"conf_{conf_id}"
        source_name = f"{conf_label}_{SOURCE_SPIN_STATE}"
        source_fdf = SOURCE_ROOT / source_name / f"{source_name}.fdf"
        atoms = read_fdf_geometry(source_fdf)

        # Sanity check: the other spin initializations must start from the same geometry
        for variant in CHECK_SPIN_VARIANTS:
            variant_name = f"{conf_label}_{variant}"
            variant_fdf = SOURCE_ROOT / variant_name / f"{variant_name}.fdf"
            if variant_fdf.exists() and read_fdf_geometry(variant_fdf) != atoms:
                raise ValueError(f"Starting geometry of {variant_name} differs from {source_name}")

        run_folder = OUTPUT_ROOT / conf_label / STATE_FOLDER
        run_folder.mkdir(parents=True, exist_ok=False)

        write_xyz(
            run_folder / "geometry.xyz",
            atoms,
            f"{conf_label} initial geometry from {source_fdf.relative_to(REPO_ROOT)}",
        )

        input_file = run_folder / INPUT_NAME
        input_file.write_text(
            render_high_spin_input(NPROCS, MAXCORE_MB, MAX_ITER_SCF, MAX_ITER_GEOM, MULTIPLICITY),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "conf": conf_label,
                "source_fdf": str(source_fdf.relative_to(REPO_ROOT)),
                "n_atoms": len(atoms),
                "state": STATE_FOLDER,
                "charge": 0,
                "multiplicity": MULTIPLICITY,
                "nprocs": NPROCS,
                "maxcore_mb": MAXCORE_MB,
                "input_file": str(input_file.relative_to(OUTPUT_ROOT)),
            }
        )

        print(f"Created: {conf_label}/{STATE_FOLDER}")

    manifest_file = OUTPUT_ROOT / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 70)
    print("Dimer quintet relaxations prepared")
    print("=" * 70)
    print(f"Source   : {SOURCE_ROOT.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)}")
    print(f"Manifest : {manifest_file.resolve()}")


if __name__ == "__main__":
    main()
