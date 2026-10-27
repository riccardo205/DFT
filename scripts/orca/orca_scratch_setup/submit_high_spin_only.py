#!/usr/bin/env python3
"""
Submit only the high-spin (no spin flip) single points of the dimer broken-symmetry trees,
e.g. <run>_broken_symmetry/<stage>/<dimer>/uuuu/uuuu.inp.

An input counts as high spin when it contains no FlipSpin keyword. The FlipSpin jobs are
left untouched, so they can later restart from the converged uuuu wavefunction.

Same behaviour as submit_functional_scan.py (whose helpers are reused): completed jobs
are skipped, active Slurm jobs are never duplicated, and failed/incomplete attempts are
archived before resubmission when RESUBMIT = True.
"""

import re
import subprocess
from pathlib import Path

from submit_functional_scan import (
    archive_previous_run,
    get_input_nprocs,
    get_slurm_ntasks,
    orca_terminated_normally,
    slurm_job_is_active,
)


def is_high_spin(input_file: Path) -> bool:
    text = input_file.read_text(encoding="utf-8", errors="ignore")
    return re.search(r"^\s*FlipSpin\b", text, flags=re.IGNORECASE | re.MULTILINE) is None


def main():

    # ============================================================
    # USER SETTINGS
    # ============================================================

    SCRIPT_DIR = Path(__file__).resolve().parent

    RUNS_ROOTS = [
        Path("/scratch/rconte1/orca/short-2OS/dimers_run2/run2_broken_symmetry"),
        Path("/scratch/rconte1/orca/short-2OS/dimers_run2/run3_broken_symmetry"),
    ]

    BATCH_FILE = SCRIPT_DIR / "orca_job.sh"

    SUBMIT = True            # False = dry run, only print what would be submitted
    SKIP_COMPLETED = True
    RESUBMIT = True          # archive and resubmit failed/incomplete attempts

    # ============================================================

    if not BATCH_FILE.exists():
        raise FileNotFoundError(f"Batch file not found: {BATCH_FILE}")

    slurm_ntasks = get_slurm_ntasks(BATCH_FILE)

    input_files = []
    for root in RUNS_ROOTS:
        if not root.exists():
            raise FileNotFoundError(f"Run directory not found: {root}")
        if not str(root).startswith("/scratch/"):
            raise RuntimeError(f"Run root must be below /scratch: {root}")
        all_inputs = sorted(root.rglob("*.inp"))
        high_spin = [p for p in all_inputs if is_high_spin(p)]
        print(f"{root}: {len(high_spin)} high-spin inputs (of {len(all_inputs)})")
        input_files += [(root, p) for p in high_spin]

    if not input_files:
        raise RuntimeError("No high-spin inputs found")

    print(f"Batch file    : {BATCH_FILE}")
    print(f"Slurm ntasks  : {slurm_ntasks}")
    print(f"Submit        : {SUBMIT}")
    print(f"Skip complete : {SKIP_COMPLETED}")
    print(f"Resubmit      : {RESUBMIT}")
    print()

    submitted = []
    skipped = []

    for root, input_file in input_files:
        run_folder = input_file.parent
        relative = input_file.relative_to(root.parent)

        if len(list(run_folder.glob("*.inp"))) != 1:
            raise RuntimeError(f"Expected exactly one .inp in {run_folder}")

        input_nprocs = get_input_nprocs(input_file)
        if input_nprocs != slurm_ntasks:
            raise RuntimeError(
                f"CPU mismatch for {input_file}\n"
                f"  ORCA %pal nprocs = {input_nprocs}\n"
                f"  Slurm --ntasks   = {slurm_ntasks}"
            )

        output_file = run_folder / f"{input_file.stem}.out"
        job_id_file = run_folder / "job_id.txt"

        if SKIP_COMPLETED and orca_terminated_normally(output_file):
            print(f"SKIP completed : {relative}")
            skipped.append((relative, "completed"))
            continue

        if job_id_file.exists():
            old_job_id = job_id_file.read_text().strip()

            if slurm_job_is_active(old_job_id):
                print(f"SKIP active    : {relative} (job {old_job_id})")
                skipped.append((relative, "active"))
                continue

            if not RESUBMIT:
                print(f"SKIP submitted : {relative} (old job {old_job_id})")
                skipped.append((relative, "already submitted"))
                continue

        if RESUBMIT and (
            output_file.exists()
            or job_id_file.exists()
            or (run_folder / "slurm.out").exists()
            or (run_folder / "slurm.err").exists()
        ):
            print(f"PREP resubmit  : {relative}")
            archive_previous_run(run_folder, input_file.stem)

        # e.g. conf_1_uuuu
        job_name = f"{run_folder.parent.name}_{run_folder.name}"
        command = ["sbatch", f"--job-name={job_name}", str(BATCH_FILE), input_file.name]

        print(f"SUBMIT         : {relative}")

        if not SUBMIT:
            print("  DRY RUN: " + " ".join(command))
            continue

        result = subprocess.run(command, cwd=run_folder, capture_output=True, text=True, check=True)
        stdout = result.stdout.strip()
        print(f"  {stdout}")

        match = re.search(r"Submitted batch job\s+(\d+)", stdout)
        job_id = match.group(1) if match else stdout
        job_id_file.write_text(str(job_id) + "\n", encoding="utf-8")
        submitted.append((relative, job_id))

    print()
    print("=" * 70)
    print("High-spin submission summary")
    print("=" * 70)
    print(f"Submitted : {len(submitted)}")
    print(f"Skipped   : {len(skipped)}")

    if submitted:
        print()
        for relative, job_id in submitted:
            print(f"{job_id:>10}  {relative}")


if __name__ == "__main__":
    main()
