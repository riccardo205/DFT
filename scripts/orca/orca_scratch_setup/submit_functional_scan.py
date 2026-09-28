#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import re
import subprocess


def orca_terminated_normally(output_file: Path) -> bool:
    if not output_file.exists():
        return False

    try:
        with output_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            for line in handle:
                if "ORCA TERMINATED NORMALLY" in line:
                    return True
    except OSError:
        return False

    return False


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


def slurm_job_is_active(job_id: str) -> bool:
    """
    Return True if the previous job ID is still present in squeue.

    This prevents RESUBMIT=True from accidentally launching a duplicate
    while an earlier calculation is still queued or running.
    """
    if not job_id.strip().isdigit():
        return False

    result = subprocess.run(
        [
            "squeue",
            "-h",
            "-j",
            job_id.strip(),
            "-o",
            "%T",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return False

    return bool(result.stdout.strip())


def archive_previous_run(
    run_folder: Path,
    stem: str,
) -> None:
    """
    Preserve important files from a failed/incomplete previous attempt.

    Renaming on the same filesystem is cheap. Temporary *.tmp files are
    deleted before resubmission so the new ORCA run starts cleanly.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    names = [
        f"{stem}.out",
        f"{stem}.gbw",
        f"{stem}.property.txt",
        f"{stem}.densities",
        f"{stem}.densitiesinfo",
        f"{stem}.bibtex",
        "slurm.out",
        "slurm.err",
        "job_id.txt",
    ]

    for name in names:
        path = run_folder / name

        if not path.exists():
            continue

        archived = run_folder / f"{name}.previous_{timestamp}"
        print(f"  archive        : {path.name} -> {archived.name}")
        path.rename(archived)

    tmp_files = set(run_folder.glob("*.tmp"))
    tmp_files.update(run_folder.glob("*.tmp.*"))

    for path in sorted(tmp_files):
        if path.is_file():
            print(f"  remove stale tmp: {path.name}")
            path.unlink()


def main():

    # ============================================================
    # USER SETTINGS
    # ============================================================

    SCRIPT_DIR = Path(__file__).resolve().parent

    RUNS_ROOT = Path(
        "/scratch/rconte1/orca/short-2OS/"
        "main_conf_4_antiferro/functional_scan"
    )

    BATCH_FILE = SCRIPT_DIR / "orca_job.sh"

    SUBMIT = True

    # With SKIP_COMPLETED=True:
    #   - ORCA jobs containing "ORCA TERMINATED NORMALLY" are untouched.
    #   - failed/incomplete jobs are eligible for resubmission.
    SKIP_COMPLETED = True

    # Set True when you want failed/incomplete old jobs resubmitted.
    # Active jobs are ALWAYS skipped to avoid duplicates.
    RESUBMIT = True

    # ============================================================

    if not RUNS_ROOT.exists():
        raise FileNotFoundError(
            f"Run directory not found: {RUNS_ROOT}"
        )

    if not str(RUNS_ROOT).startswith("/scratch/"):
        raise RuntimeError(
            f"RUNS_ROOT must be below /scratch: {RUNS_ROOT}"
        )

    if not BATCH_FILE.exists():
        raise FileNotFoundError(
            f"Batch file not found: {BATCH_FILE}"
        )

    slurm_ntasks = get_slurm_ntasks(BATCH_FILE)

    input_files = sorted(RUNS_ROOT.rglob("*.inp"))

    if not input_files:
        raise RuntimeError(
            f"No .inp files found below {RUNS_ROOT}"
        )

    print(f"Run root      : {RUNS_ROOT}")
    print(f"Batch file    : {BATCH_FILE}")
    print(f"Found inputs  : {len(input_files)}")
    print(f"Slurm ntasks  : {slurm_ntasks}")
    print(f"Submit        : {SUBMIT}")
    print(f"Skip complete : {SKIP_COMPLETED}")
    print(f"Resubmit      : {RESUBMIT}")
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

        # This is the authoritative success test.
        if SKIP_COMPLETED and orca_terminated_normally(output_file):
            print(f"SKIP completed : {relative}")
            skipped.append((relative, "completed"))
            continue

        # Never duplicate an active Slurm job.
        old_job_id = None

        if job_id_file.exists():
            old_job_id = job_id_file.read_text().strip()

            if slurm_job_is_active(old_job_id):
                print(
                    f"SKIP active    : {relative} "
                    f"(job {old_job_id})"
                )
                skipped.append((relative, "active"))
                continue

            if not RESUBMIT:
                print(
                    f"SKIP submitted : {relative} "
                    f"(old job {old_job_id})"
                )
                skipped.append((relative, "already submitted"))
                continue

        # Failed/incomplete calculation + RESUBMIT=True:
        # archive the previous attempt and remove stale *.tmp files.
        if RESUBMIT:
            has_old_files = (
                output_file.exists()
                or job_id_file.exists()
                or (run_folder / "slurm.out").exists()
                or (run_folder / "slurm.err").exists()
            )

            if has_old_files:
                print(f"PREP resubmit  : {relative}")
                archive_previous_run(
                    run_folder,
                    input_file.stem,
                )

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

        job_id_file.write_text(
            str(job_id) + "\n",
            encoding="utf-8",
        )

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
