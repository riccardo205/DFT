ORCA functional scan setup
==========================

Before preparation:
    project/
        geometry.xyz
        prepare_functional_scan.py
        submit_functional_scan.py
        orca_job.sh

Generate the nested tree:
    python3 prepare_functional_scan.py

This creates 6 functionals x 2 spin states = 12 jobs:
    runs/short2os_functional_scan/
        manifest.csv
        pbe/
            high_spin/
                geometry.xyz
                high_spin.inp
            broken_symmetry/
                geometry.xyz
                broken_symmetry.inp
        r2scan/
        b3lyp/
        pbe0/
        r2scan0/
        cam_b3lyp/

On DelftBlue:
    python3 submit_functional_scan.py

Check:
    squeue --me

Important:
- Do not use "srun orca" or "mpirun orca".
- Slurm requests 16 tasks and each ORCA input uses %pal nprocs 16.
- Jobs run in node-local /tmp and copy results back.
- NoAutoStart prevents accidental reuse of an old .gbw.
- DEFGRID3 is used because the target energy differences are very small.
