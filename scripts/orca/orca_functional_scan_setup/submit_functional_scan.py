#!/usr/bin/env python3

from pathlib import Path
import re
import subprocess


def orca_terminated_normally(output_file: Path) -> bool:
    if not output_file.exists():
        return False

    try:
        text = output_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return False

    return "ORCA TERMINATED NORMALLY" in text


def get_input_nprocs(input_file: Path) -> int:
    text = input_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    match = re.search(
        r"\bnprocs\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise RuntimeError(
            f"No 'nprocs' found in {input_file}"
        )

    return int(match.group(1))


def get_slurm_ntasks(batch_file: Path) -> int:
    text = batch_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    match = re.search(
        r"^\s*#SBATCH\s+--ntasks=(\d+)\s*$",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise RuntimeError(
            f"No '#SBATCH --ntasks=N' found in {batch_file}"
        )

    return int(match.group(1))


def main():
    # USER SETTINGS
    SCRIPT_DIR = Path(__file__).resolve().parent

    RUNS_ROOT = Path.home() / "runs" / "short-2OS" / "main_conf_4_antiferro" / "electric_field_scan" / "functional_scan_D4"
    BATCH_FILE = SCRIPT_DIR / "orca_job.sh"

    SUBMIT = True
    RESUBMIT = False
    SKIP_COMPLETED = True

    if not RUNS_ROOT.exists():
        raise FileNotFoundError(
            f"Run directory not found: {RUNS_ROOT.resolve()}"
        )

    if not BATCH_FILE.exists():
        raise FileNotFoundError(
            f"Batch file not found: {BATCH_FILE}"
        )

    slurm_ntasks = get_slurm_ntasks(BATCH_FILE)

    input_files = sorted(RUNS_ROOT.rglob("*.inp"))

    if not input_files:
        raise RuntimeError(
            f"No .inp files found below {RUNS_ROOT.resolve()}"
        )

    print(f"Found {len(input_files)} ORCA input files.")
    print(f"Slurm ntasks: {slurm_ntasks}")
    print()

    submitted = []
    skipped = []

    for input_file in input_files:
        run_folder = input_file.parent

        sibling_inputs = list(run_folder.glob("*.inp"))
        if len(sibling_inputs) != 1:
            raise RuntimeError(
                f"Expected exactly one .inp in {run_folder}, "
                f"found {len(sibling_inputs)}"
            )

        input_nprocs = get_input_nprocs(input_file)

        if input_nprocs != slurm_ntasks:
            raise RuntimeError(
                f"CPU mismatch for {input_file}\n"
                f"  ORCA %pal nprocs = {input_nprocs}\n"
                f"  Slurm --ntasks   = {slurm_ntasks}"
            )

        output_file = run_folder / f"{input_file.stem}.out"
        job_id_file = run_folder / "job_id.txt"
        relative = input_file.relative_to(RUNS_ROOT)

        if SKIP_COMPLETED and orca_terminated_normally(output_file):
            print(f"SKIP completed : {relative}")
            skipped.append((relative, "completed"))
            continue

        if job_id_file.exists() and not RESUBMIT:
            old_job_id = job_id_file.read_text().strip()
            print(
                f"SKIP submitted : {relative} "
                f"(job {old_job_id})"
            )
            skipped.append((relative, "already submitted"))
            continue

        functional = run_folder.parent.name
        state = run_folder.name
        job_name = f"{functional}_{state}"

        command = [
            "sbatch",
            f"--job-name={job_name}",
            str(BATCH_FILE),
            input_file.name,
        ]

        print(f"SUBMIT         : {relative}")

        if not SUBMIT:
            print("  DRY RUN: " + " ".join(command))
            continue

        result = subprocess.run(
            command,
            cwd=run_folder,
            capture_output=True,
            text=True,
            check=True,
        )

        stdout = result.stdout.strip()
        print(f"  {stdout}")

        match = re.search(
            r"Submitted batch job\s+(\d+)",
            stdout,
        )

        job_id = match.group(1) if match else stdout

        job_id_file.write_text(str(job_id) + "\n")
        submitted.append((relative, job_id))

    print()
    print("=" * 70)
    print("Submission summary")
    print("=" * 70)
    print(f"Submitted : {len(submitted)}")
    print(f"Skipped   : {len(skipped)}")

    if submitted:
        print()
        for relative, job_id in submitted:
            print(f"{job_id:>10}  {relative}")


if __name__ == "__main__":
    main()
