#!/usr/bin/env python3
"""
Restart the dimer quintet relaxations of dimers_run2/run1 as dimers_run2/run2.

run1 (prepare_dimers_relaxation.py) was killed by the 12 h walltime for all 39 confs:
  - ConvCheckMode 0 never let DIIS converge (DIIS error stalls ~4e-4 while dE ~1e-8),
    so every geometry cycle ran ~50 SCF iterations and then fell into Auto-TRAH;
  - in 13 confs TRAH diverged and burned the remaining walltime in a single SCF.

Changes for run2:
  - start from the last run1 geometry (run1/conf_N/high_spin/high_spin.xyz);
  - ORCA default SCF convergence check (no ConvCheckMode 0);
  - SCF MaxIter 150, so a diverging SCF aborts the job instead of eating the walltime;
  - NormalOpt instead of TightOpt (OPT_KEYWORD);
  - MORead of the run1 orbitals (previous.gbw), except where run1 ended inside a
    diverging TRAH SCF: those start from the default guess.

Writes run2/conf_N/high_spin/{high_spin.inp, geometry.xyz, geometry_initial.xyz, [previous.gbw]}.
geometry_initial.xyz is the original SIESTA starting structure, kept for reference.
run1 is only read. All jobs are independent: submit with submit_functional_scan.py.
"""

import csv
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def render_input(opt_keyword, moread, nprocs, maxcore_mb, max_iter_scf, max_iter_geom, multiplicity):
    keywords = f"R2SCAN-3C TightSCF {opt_keyword} UKS" + (" MORead" if moread else "")
    return (
        f"! {keywords}\n\n"
        + ('%moinp "previous.gbw"\n\n' if moread else "")
        + "%scf\n"
        f"    MaxIter {max_iter_scf}\n"
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


def last_cycle_trah_iterations(out_text):
    """TRAH macro-iterations in the last geometry cycle of a run1 output."""
    last_cycle = out_text.split("GEOMETRY OPTIMIZATION CYCLE")[-1]
    return last_cycle.count("(TRAH MAcro)")


def main():
    # ---- USER SETTINGS ----
    RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
    SOURCE_ROOT = RUN_ROOT / "run1"
    OUTPUT_ROOT = RUN_ROOT / "run2"

    OPT_KEYWORD = "NormalOpt"        # run1 used TightOpt
    MULTIPLICITY = 5
    N_ATOMS = 172
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER_SCF = 150               # run1: 500
    MAX_ITER_GEOM = 200
    DIVERGED_TRAH_ITER = 10          # more TRAH macro-iterations in the last cycle -> no MORead

    OVERWRITE = False

    # ---- SETUP ----
    if OUTPUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(f"Output directory already exists: {OUTPUT_ROOT.resolve()}")
    if OUTPUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    conf_dirs = sorted(
        (d for d in SOURCE_ROOT.iterdir() if re.fullmatch(r"conf_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
    )

    manifest_rows = []
    for conf_dir in conf_dirs:
        src = conf_dir / "high_spin"
        out_text = (src / "high_spin.out").read_text(encoding="utf-8", errors="ignore")
        if "ORCA TERMINATED NORMALLY" in out_text:
            print(f"SKIP (run1 finished): {conf_dir.name}")
            continue

        last_xyz = src / "high_spin.xyz"
        if int(last_xyz.read_text().split()[0]) != N_ATOMS:
            raise ValueError(f"{last_xyz}: expected {N_ATOMS} atoms")
        n_cycles = out_text.count("GEOMETRY OPTIMIZATION CYCLE")
        trah_iter = last_cycle_trah_iterations(out_text)
        moread = trah_iter <= DIVERGED_TRAH_ITER and (src / "high_spin.gbw").exists()

        dst = OUTPUT_ROOT / conf_dir.name / "high_spin"
        dst.mkdir(parents=True)
        shutil.copy2(last_xyz, dst / "geometry.xyz")
        shutil.copy2(src / "geometry.xyz", dst / "geometry_initial.xyz")
        if moread:
            shutil.copy2(src / "high_spin.gbw", dst / "previous.gbw")
        (dst / "high_spin.inp").write_text(
            render_input(OPT_KEYWORD, moread, NPROCS, MAXCORE_MB, MAX_ITER_SCF, MAX_ITER_GEOM, MULTIPLICITY),
            encoding="utf-8",
        )

        manifest_rows.append({
            "conf": conf_dir.name,
            "source_geometry": str(last_xyz.relative_to(REPO_ROOT)),
            "run1_cycles": n_cycles,
            "run1_last_cycle_trah_iter": trah_iter,
            "run1_end": "scf_diverged" if trah_iter > DIVERGED_TRAH_ITER else "walltime",
            "moread": moread,
            "opt": OPT_KEYWORD,
            "multiplicity": MULTIPLICITY,
            "input_file": f"{conf_dir.name}/high_spin/high_spin.inp",
        })
        print(f"{conf_dir.name}: run1 {n_cycles} cycles, last-cycle TRAH {trah_iter}, MORead={moread}")

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_moread = sum(r["moread"] for r in manifest_rows)
    print()
    print("=" * 70)
    print("Dimer quintet relaxations restarted (run2)")
    print("=" * 70)
    print(f"Source   : {SOURCE_ROOT.resolve()}")
    print(f"Runs     : {OUTPUT_ROOT.resolve()}")
    print(f"Jobs     : {len(manifest_rows)} ({n_moread} MORead, {len(manifest_rows) - n_moread} fresh guess)")


if __name__ == "__main__":
    main()
