#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone


# ============================================================
# Repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

METHOD_FILE = REPO_ROOT / "configs" / "orca_methods.json"
RUN_ROOT = REPO_ROOT / "runs" / "orca"


# ============================================================
# Utility functions
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)

    return h.hexdigest()


def normal_termination(output_file):

    if not output_file.exists():
        return False

    text = output_file.read_text(errors="ignore")

    return "ORCA TERMINATED NORMALLY" in text


def parse_s2(output_file):

    text = output_file.read_text(errors="ignore")

    match = re.search(
        r"Expectation value of <S\*\*2>\s*:\s*([-+]?\d+\.\d+)",
        text,
    )

    if match is None:
        return None

    return float(match.group(1))


def parse_mulliken_spin_populations(output_file):

    lines = output_file.read_text(
        errors="ignore"
    ).splitlines()

    populations = {}

    inside_section = False

    pattern = re.compile(
        r"^\s*(\d+)\s+([A-Za-z]+)\s*:"
        r"\s*([-+]?\d+(?:\.\d+)?)"
        r"\s+([-+]?\d+(?:\.\d+)?)"
    )

    for line in lines:

        if (
            "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
            in line
        ):
            inside_section = True
            continue

        if not inside_section:
            continue

        if "Sum of atomic charges" in line:
            break

        match = pattern.match(line)

        if match is None:
            continue

        atom_index = int(match.group(1))
        element = match.group(2)
        charge = float(match.group(3))
        spin = float(match.group(4))

        populations[atom_index] = {
            "element": element,
            "charge": charge,
            "spin": spin,
        }

    return populations


# ============================================================
# Directory handling
# ============================================================

def prepare_directory(path, overwrite=False):

    path = Path(path)

    if path.exists():

        contents = list(path.iterdir())

        if contents and not overwrite:

            raise RuntimeError(
                f"\nJob directory already contains files:\n"
                f"{path}\n\n"
                f"Set OVERWRITE = True if you want "
                f"to replace this calculation."
            )

        if contents and overwrite:
            shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# ORCA execution
# ============================================================

def run_orca(
    job_dir,
    input_name,
    output_name,
):

    orca_executable = shutil.which("orca")

    if orca_executable is None:
        raise RuntimeError(
            "ORCA executable not found in PATH."
        )

    job_dir = Path(job_dir)

    input_file = job_dir / input_name
    output_file = job_dir / output_name

    print()
    print("=" * 60)
    print("Running ORCA")
    print("=" * 60)

    print(f"Directory : {job_dir}")
    print(f"Input     : {input_name}")
    print(f"Output    : {output_name}")
    print()

    with open(output_file, "w") as out:

        result = subprocess.run(
            [
                orca_executable,
                input_name,
            ],
            cwd=job_dir,
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:

        raise RuntimeError(
            f"ORCA exited with return code "
            f"{result.returncode}.\n"
            f"Check:\n{output_file}"
        )

    if not normal_termination(output_file):

        raise RuntimeError(
            f"ORCA did not terminate normally.\n"
            f"Check:\n{output_file}"
        )

    print("ORCA terminated normally.")

    return output_file


# ============================================================
# Create a normal single-state calculation
# ============================================================

def prepare_single_state(
    xyz_source,
    calculation_root,
    method,
    charge,
    multiplicity,
    nprocs,
    overwrite,
):

    state_name = f"q{charge}_m{multiplicity}"

    job_dir = (
        calculation_root
        / state_name
    )

    prepare_directory(
        job_dir,
        overwrite=overwrite,
    )

    # --------------------------------------------------------
    # Copy geometry
    # --------------------------------------------------------

    geometry_file = (
        job_dir
        / "geometry.xyz"
    )

    shutil.copy2(
        xyz_source,
        geometry_file,
    )

    # --------------------------------------------------------
    # ORCA settings
    # --------------------------------------------------------

    keyword_line = " ".join(
        method["keywords"]
    )

    max_iter = method.get(
        "max_iter",
        300,
    )

    input_text = (
        f"! {keyword_line} NoAutoStart\n\n"
    )

    input_text += (
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "end\n\n"
    )

    if nprocs > 1:

        input_text += (
            "%pal\n"
            f"    nprocs {nprocs}\n"
            "end\n\n"
        )

    input_text += (
        f"* xyzfile "
        f"{charge} "
        f"{multiplicity} "
        f"geometry.xyz\n"
    )

    input_file = (
        job_dir
        / "orca.inp"
    )

    input_file.write_text(
        input_text
    )

    return job_dir


# ============================================================
# Prepare high-spin reference calculation
# ============================================================

def prepare_high_spin(
    xyz_source,
    calculation_root,
    method,
    charge,
    high_spin_multiplicity,
    nprocs,
    overwrite,
):

    job_dir = (
        calculation_root
        / "high_spin"
    )

    prepare_directory(
        job_dir,
        overwrite=overwrite,
    )

    geometry_file = (
        job_dir
        / "geometry.xyz"
    )

    shutil.copy2(
        xyz_source,
        geometry_file,
    )

    keyword_line = " ".join(
        method["keywords"]
    )

    max_iter = method.get(
        "max_iter",
        300,
    )

    input_text = (
        f"! {keyword_line} NoAutoStart\n\n"
    )

    input_text += (
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "end\n\n"
    )

    if nprocs > 1:

        input_text += (
            "%pal\n"
            f"    nprocs {nprocs}\n"
            "end\n\n"
        )

    input_text += (
        f"* xyzfile "
        f"{charge} "
        f"{high_spin_multiplicity} "
        f"geometry.xyz\n"
    )

    input_file = (
        job_dir
        / "high_spin.inp"
    )

    input_file.write_text(
        input_text
    )

    return job_dir


# ============================================================
# Validate high-spin state
# ============================================================

def validate_high_spin(
    output_file,
    radical_atoms,
    minimum_spin_population,
):

    populations = (
        parse_mulliken_spin_populations(
            output_file
        )
    )

    if not populations:

        raise RuntimeError(
            "Could not find Mulliken spin populations "
            "in high-spin output."
        )

    print()
    print("=" * 60)
    print("High-spin population check")
    print("=" * 60)

    valid = True

    for atom in radical_atoms:

        if atom not in populations:

            raise RuntimeError(
                f"Atom {atom} was not found "
                f"in the Mulliken population analysis."
            )

        element = (
            populations[atom]["element"]
        )

        spin = (
            populations[atom]["spin"]
        )

        print(
            f"Atom {atom:3d} "
            f"({element}) : "
            f"spin = {spin:+.6f}"
        )

        if spin < minimum_spin_population:
            valid = False

    s2 = parse_s2(output_file)

    if s2 is not None:
        print()
        print(f"<S^2> = {s2:.6f}")

    if not valid:

        raise RuntimeError(
            "\nThe high-spin calculation did not "
            "produce the expected positive spin "
            "population on all radical centers.\n"
            "The broken-symmetry calculation "
            "will not be started."
        )

    print()
    print(
        "High-spin spin localization looks correct."
    )


# ============================================================
# Prepare broken-symmetry calculation
# ============================================================

def prepare_broken_symmetry(
    xyz_source,
    calculation_root,
    high_spin_gbw,
    method,
    charge,
    high_spin_multiplicity,
    final_ms,
    flip_atoms,
    nprocs,
    overwrite,
):

    job_dir = (
        calculation_root
        / "broken_symmetry"
    )

    prepare_directory(
        job_dir,
        overwrite=overwrite,
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry_file = (
        job_dir
        / "geometry.xyz"
    )

    shutil.copy2(
        xyz_source,
        geometry_file,
    )

    # --------------------------------------------------------
    # Copy converged high-spin wavefunction
    # --------------------------------------------------------

    reference_gbw = (
        job_dir
        / "high_spin_reference.gbw"
    )

    shutil.copy2(
        high_spin_gbw,
        reference_gbw,
    )

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    keyword_line = " ".join(
        method["keywords"]
    )

    max_iter = method.get(
        "max_iter",
        300,
    )

    flip_string = ", ".join(
        str(atom)
        for atom in flip_atoms
    )

    input_text = (
        f"! {keyword_line} NoAutoStart\n\n"
    )

    input_text += (
        "%scf\n"
        '    Guess MORead\n'
        '    MOInp "high_spin_reference.gbw"\n'
        f"    FlipSpin {flip_string}\n"
        f"    FinalMs {final_ms}\n"
        f"    MaxIter {max_iter}\n"
        "end\n\n"
    )

    if nprocs > 1:

        input_text += (
            "%pal\n"
            f"    nprocs {nprocs}\n"
            "end\n\n"
        )

    # Important:
    #
    # The multiplicity here remains the HIGH-SPIN
    # reference multiplicity.
    #
    # FlipSpin + FinalMs then construct the
    # broken-symmetry state.
    input_text += (
        f"* xyzfile "
        f"{charge} "
        f"{high_spin_multiplicity} "
        f"geometry.xyz\n"
    )

    input_file = (
        job_dir
        / "broken_symmetry.inp"
    )

    input_file.write_text(
        input_text
    )

    return job_dir


# ============================================================
# SINGLE-STATE WORKFLOW
# ============================================================

def run_single_state_workflow(
    xyz_source,
    system,
    geometry,
    method_name,
    charge,
    multiplicity,
    nprocs,
    overwrite,
):

    methods = load_json(
        METHOD_FILE
    )

    if method_name not in methods:

        raise ValueError(
            f"Unknown method: {method_name}"
        )

    method = (
        methods[method_name]
    )

    calculation_root = (
        RUN_ROOT
        / system
        / geometry
        / method_name
        / "single_states"
    )

    calculation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    job_dir = prepare_single_state(
        xyz_source=xyz_source,
        calculation_root=calculation_root,
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        nprocs=nprocs,
        overwrite=overwrite,
    )

    output_file = run_orca(
        job_dir=job_dir,
        input_name="orca.inp",
        output_name="orca.out",
    )

    print()
    print("=" * 60)
    print("SINGLE-STATE CALCULATION COMPLETE")
    print("=" * 60)

    print(f"Charge       : {charge}")
    print(f"Multiplicity : {multiplicity}")

    s2 = parse_s2(output_file)

    if s2 is not None:
        print(f"<S^2>        : {s2:.6f}")


# ============================================================
# BROKEN-SYMMETRY WORKFLOW
# ============================================================

def run_broken_symmetry_workflow(
    xyz_source,
    system,
    geometry,
    method_name,
    charge,
    high_spin_multiplicity,
    final_ms,
    flip_atoms,
    radical_atoms,
    minimum_spin_population,
    nprocs,
    overwrite,
):

    methods = load_json(
        METHOD_FILE
    )

    if method_name not in methods:

        raise ValueError(
            f"Unknown method: {method_name}"
        )

    method = (
        methods[method_name]
    )

    calculation_root = (
        RUN_ROOT
        / system
        / geometry
        / method_name
        / "broken_symmetry_workflow"
    )

    calculation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # STEP 1
    # High-spin state
    # ========================================================

    print()
    print("#" * 60)
    print("STEP 1: HIGH-SPIN REFERENCE")
    print("#" * 60)

    high_spin_dir = prepare_high_spin(
        xyz_source=xyz_source,
        calculation_root=calculation_root,
        method=method,
        charge=charge,
        high_spin_multiplicity=high_spin_multiplicity,
        nprocs=nprocs,
        overwrite=overwrite,
    )

    high_spin_output = run_orca(
        job_dir=high_spin_dir,
        input_name="high_spin.inp",
        output_name="high_spin.out",
    )

    validate_high_spin(
        output_file=high_spin_output,
        radical_atoms=radical_atoms,
        minimum_spin_population=minimum_spin_population,
    )

    high_spin_gbw = (
        high_spin_dir
        / "high_spin.gbw"
    )

    if not high_spin_gbw.exists():

        raise RuntimeError(
            f"High-spin GBW file not found:\n"
            f"{high_spin_gbw}"
        )

    # ========================================================
    # STEP 2
    # Broken symmetry
    # ========================================================

    print()
    print("#" * 60)
    print("STEP 2: BROKEN-SYMMETRY STATE")
    print("#" * 60)

    bs_dir = prepare_broken_symmetry(
        xyz_source=xyz_source,
        calculation_root=calculation_root,
        high_spin_gbw=high_spin_gbw,
        method=method,
        charge=charge,
        high_spin_multiplicity=high_spin_multiplicity,
        final_ms=final_ms,
        flip_atoms=flip_atoms,
        nprocs=nprocs,
        overwrite=overwrite,
    )

    bs_output = run_orca(
        job_dir=bs_dir,
        input_name="broken_symmetry.inp",
        output_name="broken_symmetry.out",
    )

    print()
    print("=" * 60)
    print("BROKEN-SYMMETRY WORKFLOW COMPLETE")
    print("=" * 60)

    print()
    print("High-spin output:")
    print(high_spin_output)

    print()
    print("Broken-symmetry output:")
    print(bs_output)

    bs_s2 = parse_s2(
        bs_output
    )

    if bs_s2 is not None:
        print()
        print(
            f"Broken-symmetry <S^2> = "
            f"{bs_s2:.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # USER SETTINGS
    # ========================================================

    XYZ_FILE = "Short-2OS/runs/siesta/junction_SM2/slow_2026_03_13_vs2/conf_4_ferro/short-2OS.xyz"

    SYSTEM = "short-2OS"

    GEOMETRY = "Junction_SM2_conf_4_ferro"

    METHOD = "pbe_tzvp"

    CHARGE = 0

    NPROCS = 1

    OVERWRITE = True


    # ========================================================
    # TYPE OF CALCULATION
    # ========================================================
    # Options:
    #     "single_state"
    #     "broken_symmetry"

    RUN_TYPE = "broken_symmetry"

    MULTIPLICITY = 3

    HIGH_SPIN_MULTIPLICITY = 3 #broken_symmetry

    FINAL_MS = 0

    # ORCA atom numbering starts from zero.
    RADICAL_ATOMS = [
        28,
        41,
    ]

    # Flip only one radical center.
    FLIP_ATOMS = [
        28,
    ]

    # Require substantial positive spin density
    # on the expected radical centers before
    # performing FlipSpin.
    MINIMUM_HIGH_SPIN_POPULATION = 0.20


    # ========================================================
    # Run selected workflow
    # ========================================================

    xyz_source = Path(
        XYZ_FILE
    ).resolve()

    if not xyz_source.exists():

        raise FileNotFoundError(
            f"XYZ file does not exist:\n"
            f"{xyz_source}"
        )

    if RUN_TYPE == "single_state":

        run_single_state_workflow(
            xyz_source=xyz_source,
            system=SYSTEM,
            geometry=GEOMETRY,
            method_name=METHOD,
            charge=CHARGE,
            multiplicity=MULTIPLICITY,
            nprocs=NPROCS,
            overwrite=OVERWRITE,
        )

    elif RUN_TYPE == "broken_symmetry":

        run_broken_symmetry_workflow(
            xyz_source=xyz_source,
            system=SYSTEM,
            geometry=GEOMETRY,
            method_name=METHOD,
            charge=CHARGE,
            high_spin_multiplicity=HIGH_SPIN_MULTIPLICITY,
            final_ms=FINAL_MS,
            flip_atoms=FLIP_ATOMS,
            radical_atoms=RADICAL_ATOMS,
            minimum_spin_population=MINIMUM_HIGH_SPIN_POPULATION,
            nprocs=NPROCS,
            overwrite=OVERWRITE,
        )

    else:

        raise ValueError(
            f"Unknown RUN_TYPE: {RUN_TYPE}\n"
            f"Use 'single_state' or "
            f"'broken_symmetry'."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
