ORCA direct-/scratch setup

Files
-----
orca_job.sh
    Runs ORCA directly in the Slurm submission folder under /scratch/rconte1.
    No node-local /tmp copy is used.
    ORCA stdout+stderr are combined into the main *.out file.
    The script verifies "ORCA TERMINATED NORMALLY".
    Successful jobs delete *.tmp / *.tmp.* files.
    Failed jobs keep temporary files for diagnosis and exit non-zero.

submit_functional_scan.py
    Scans:
      /scratch/rconte1/orca/short-2OS/main_conf_4_antiferro/functional_scan

    Recommended current settings:
      SKIP_COMPLETED = True
      RESUBMIT = True

    Successful calculations are identified ONLY by:
      ORCA TERMINATED NORMALLY

    Active Slurm jobs are skipped, preventing duplicate submissions.
    Failed/incomplete previous runs are archived and stale temporary files
    are removed before resubmission.

prepare_functional_scan.py
    Only needed when creating/recreating the functional-scan tree.
    OUTPUT_ROOT points directly to /scratch.
    OVERWRITE=False by default.

Recommended script location
---------------------------
/home/rconte1/orca/scripts/

Recommended run location
------------------------
/scratch/rconte1/orca/short-2OS/main_conf_4_antiferro/functional_scan/

Typical use
-----------
1. Copy these scripts to /home/rconte1/orca/scripts/
2. chmod +x orca_job.sh
3. If the functional_scan tree already exists, DO NOT rerun prepare_functional_scan.py.
4. From /home/rconte1/orca/scripts/:
       python3 submit_functional_scan.py

Current batch memory:
    --ntasks=16
    --mem-per-cpu=4G

Current ORCA input:
    %maxcore 2000

This gives Slurm ~64 GB total allocation for 16 ranks while keeping ORCA's
per-rank MaxCore at 2000 MB.
