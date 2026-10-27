#!/usr/bin/env python3
"""
FlipSpin broken-symmetry single points for the run4 dimers, same protocol as run2/run3
(prepare_dimers_broken_symmetry.py + set_carbon_flips.py):

  - one folder per dimer that has a finished high-spin run in run4_broken_symmetry_high_spin
    (relaxed/<set>/<dimer>/uuuu, set = free / fixC_dihedral / fixC_double90);
  - all 8 configurations uuuu duuu uduu uudu uuud dduu dudu duud (u/d = spin of radicals A B C D);
  - PBE0 def2-TZVP TightSCF UKS DEFGRID3, default SCF convergence check, UNO UCO + UNO occupations;
  - every job (uuuu included) starts from the converged high-spin orbitals:
    MORead NoAutoStart, %moinp "../../../../../run4_broken_symmetry_high_spin/relaxed/<set>/<dimer>/uuuu/uuuu.gbw";
  - geometry.xyz is copied from that high-spin folder (identical to the .gbw geometry);
  - FlipSpin lists only the CARBON atoms of the flipped radical halves (27 per radical: single flips
    27 atoms, pair flips 54 < 64, ORCA's FlipSpin limit); quintet reference (* xyzfile 0 5), FinalMs set.

Atom assignment, rebuilt from the geometry for every dimer (as set_carbon_flips.py):
two molecules = two bonded components; each mapped on the reference monomer by graph isomorphism;
A / B = halves of molecule 1 (atoms 0..N-1) around radical carbons 28 / 41, C / D = molecule 2
around 28+N / 41+N. The script checks that these four atoms are the radical carbons, that every half
has 43 atoms / 27 C, and every FlipSpin list before writing. Writes radical_fragments.json per dimer
(read by Read_output_bs_orca.py) and run4_broken_symmetry/manifest.csv.
All jobs are independent: submit with submit_functional_scan.py.
"""

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_dimers_broken_symmetry import CONFIGURATIONS, RADICALS, TEMPLATE, bond_graph, folder_name  # noqa: E402
from set_carbon_flips import radical_halves  # noqa: E402

RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
HS_ROOT = RUN_ROOT / "run4_broken_symmetry_high_spin"
OUT_ROOT = RUN_ROOT / "run4_broken_symmetry"
REF_XYZ = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L135_C120_R090" / "high_spin.xyz"

FUNCTIONAL, BASIS, CHARGE, MULTIPLICITY = "PBE0", "def2-TZVP", 0, 5
NPROCS, MAXCORE_MB, MAX_ITER = 16, 2000, 500
FLIP_LIMIT = 64
OVERWRITE = False


def finished(out):
    return out.exists() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="ignore")[-3000:]


def main():
    if OUT_ROOT.exists() and not OVERWRITE:
        raise FileExistsError(f"{OUT_ROOT} exists; set OVERWRITE = True to regenerate")
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ref = read(REF_XYZ)
    ref_graph = bond_graph(ref)
    n_ref = Counter(ref.get_chemical_symbols())

    hs_dirs = sorted(p.parent for p in HS_ROOT.glob("relaxed/*/*/uuuu/uuuu.out"))
    manifest, skipped = [], []
    for hs in hs_dirs:
        set_name, dimer_name = hs.parent.parent.name, hs.parent.name
        label = f"{set_name}/{dimer_name}"
        if not (finished(hs / "uuuu.out") and (hs / "uuuu.gbw").exists()):
            skipped.append(f"{label} (high spin not finished)")
            continue
        geo_src = hs / "geometry.xyz"
        dimer = read(geo_src)
        sym = dimer.get_chemical_symbols()
        if Counter(sym) != Counter({k: 2 * v for k, v in n_ref.items()}):
            raise ValueError(f"{label}: composition differs from 2 x reference")
        halves, radc = radical_halves(dimer, ref_graph)          # checks radical carbons 28/41(+N)
        carbons = {lab: [a for a in h if sym[a] == "C"] for lab, h in halves.items()}
        for lab in RADICALS:
            if len(halves[lab]) != 43 or len(carbons[lab]) != 27:
                raise ValueError(f"{label} {lab}: {len(halves[lab])} atoms / {len(carbons[lab])} C")

        dimer_out = OUT_ROOT / "relaxed" / set_name / dimer_name
        dimer_out.mkdir(parents=True)
        (dimer_out / "radical_fragments.json").write_text(json.dumps({
            "fragments": halves, "radical_carbons": radc, "configurations": CONFIGURATIONS,
            "flip_atoms": "carbons", "source_geometry": str(geo_src.relative_to(REPO_ROOT)),
        }, indent=1))
        gbw = f"../../../../../{HS_ROOT.name}/relaxed/{set_name}/{dimer_name}/uuuu/uuuu.gbw"
        for config, final_ms in CONFIGURATIONS.items():
            name = folder_name(config)
            down = [r for r, s in zip(RADICALS, config) if s == "-"]
            flip = sorted(a for r in down for a in carbons[r])
            up_atoms = {a for r in RADICALS if r not in down for a in halves[r]}
            if any(sym[a] != "C" for a in flip) or set(flip) & up_atoms or len(flip) != 27 * len(down) \
                    or len(flip) >= FLIP_LIMIT:
                raise RuntimeError(f"{label}/{name}: FlipSpin check failed")
            flip_block = (f"\n    FlipSpin {', '.join(map(str, flip))}\n    FinalMs {final_ms}\n") if flip else ""
            run = dimer_out / name
            run.mkdir()
            shutil.copy2(geo_src, run / "geometry.xyz")
            (run / f"{name}.inp").write_text(TEMPLATE.format(
                functional=FUNCTIONAL, basis=BASIS, max_iter=MAX_ITER, moread=" MORead NoAutoStart",
                moinp=f'\n%moinp "{gbw}"\n', conv_check="", flip=flip_block, nprocs=NPROCS, maxcore=MAXCORE_MB,
                charge=CHARGE, mult=MULTIPLICITY), encoding="utf-8")
            # the relative %moinp path must point to the high-spin .gbw from the job folder
            if not (run / gbw).resolve().exists():
                raise FileNotFoundError(f"{label}/{name}: {gbw} not found from the job folder")
            manifest.append({"stage": "relaxed", "set": set_name, "dimer": dimer_name, "configuration": config,
                             "folder": name, "flipped_radicals": "".join(down), "FinalMs": final_ms,
                             "n_flipped_atoms": len(flip), "moread_from": gbw,
                             "input_file": f"relaxed/{set_name}/{dimer_name}/{name}/{name}.inp"})
        print(f"{label:24s} radical C A={radc['A']} B={radc['B']} C={radc['C']} D={radc['D']}  -> 8 inputs")

    with (OUT_ROOT / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)
    print(f"\n{len(manifest)} inputs for {len(manifest) // 8} dimers in {OUT_ROOT}")
    for s in skipped:
        print(f"SKIPPED {s}")


if __name__ == "__main__":
    main()
