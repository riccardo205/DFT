#!/usr/bin/env python3
"""
Collect the FlipSpin broken-symmetry single points of prepare_dimers_broken_symmetry.py.

For every <run>_broken_symmetry/<stage>/<dimer>/<config> folder: FINAL SINGLE POINT ENERGY,
<S^2>, and the final Mulliken spin population summed over each radical fragment A/B/C/D
(fragments from <stage>/<dimer>/radical_fragments.json, which differ between dimers because
the run2 monomers use two atom numberings). The converged spin topology is checked against the requested one;
convergence alone is not taken as proof of the intended state.

status:
  ok                 every fragment has the requested sign and |spin| >= MIN_ABS_SPIN,
                     and the total spin matches 2*FinalMs
  wrong_topology     at least one fragment converged with the opposite sign
  quenched           a fragment's |spin| < MIN_ABS_SPIN (radical paired / delocalised)
  wrong_Ms           total spin differs from 2*FinalMs
  scf_not_converged  ORCA finished but reports an unconverged SCF
  not_finished       output present without ORCA TERMINATED NORMALLY
  missing            no output

Writes <stage>/<dimer>/summary.csv per dimer; relative_energy_meV is relative to "++++" of
that dimer. The relaxation state of the source geometry (converged / not_converged) is
printed with each dimer.
"""

import csv
import json
import re
from pathlib import Path

from prepare_dimers_broken_symmetry import RADICALS, folder_name, mulliken_spins

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EH_TO_MEV = 27211.386245988


def classify(config, final_ms, spins, min_abs_spin, ms_tol):
    if any(abs(s) < min_abs_spin for s in spins.values()):
        return "quenched"
    if any((s > 0) != (c == "+") for s, c in zip(spins.values(), config)):
        return "wrong_topology"
    if abs(sum(spins.values()) - 2 * final_ms) > ms_tol:
        return "wrong_Ms"
    return "ok"


def parse(out_file, fragments):
    if not out_file.exists():
        return {"status": "missing"}
    text = out_file.read_text(encoding="utf-8", errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in text:
        return {"status": "not_finished"}
    if "SCF NOT CONVERGED" in text or "SCF CONVERGED AFTER" not in text:
        return {"status": "scf_not_converged"}
    energy = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)[-1])
    s2 = float(re.findall(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)", text)[-1])
    atom_spins = mulliken_spins(out_file)
    spins = {r: sum(atom_spins[a] for a in atoms) for r, atoms in fragments.items()}
    return {"energy": energy, "S2": s2, "spins": spins}


def main():
    # ---- USER SETTINGS ----
    RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
    ROOTS = [RUN_ROOT / "run2_broken_symmetry", RUN_ROOT / "run3_broken_symmetry"]
    MIN_ABS_SPIN = 0.5          # minimum |fragment spin| to count as an unpaired S=1/2
    MS_TOL = 0.1                # tolerance on sum of fragment spins vs 2*FinalMs

    dimer_dirs = sorted(
        (d for root in ROOTS for stage in ("initial", "relaxed") if (root / stage).is_dir()
         for d in (root / stage).iterdir() if (d / "radical_fragments.json").exists()),
        key=lambda d: (str(d.parent), len(d.name), d.name),
    )
    for dimer_dir in dimer_dirs:
        setup = json.loads((dimer_dir / "radical_fragments.json").read_text())
        fragments = setup["fragments"]
        configurations = setup["configurations"]
        rows = []
        for config, final_ms in configurations.items():
            name = folder_name(config)
            result = parse(dimer_dir / name / f"{name}.out", fragments)
            row = {
                "configuration": config,
                "flipped_radicals": "".join(r for r, c in zip(RADICALS, config) if c == "-"),
                "FinalMs": final_ms,
                "energy_Eh": result.get("energy"),
                "relative_energy_meV": None,
                "S2": result.get("S2"),
                **{f"spin_{r}": round(result["spins"][r], 4) if "spins" in result else None for r in RADICALS},
                "status": result.get("status")
                or classify(config, final_ms, result["spins"], MIN_ABS_SPIN, MS_TOL),
            }
            rows.append(row)

        reference = next((r["energy_Eh"] for r in rows if r["configuration"] == "++++"), None)
        for row in rows:
            if row["energy_Eh"] is not None and reference is not None:
                row["relative_energy_meV"] = round((row["energy_Eh"] - reference) * EH_TO_MEV, 3)

        with (dimer_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"== {dimer_dir.parent.parent.name}/{dimer_dir.parent.name}/{dimer_dir.name}"
              f"  (relaxation: {setup.get('relaxation', '?')})")
        for row in rows:
            spins = " ".join(f"{row[f'spin_{r}']:+.2f}" if row[f"spin_{r}"] is not None else "  -  " for r in RADICALS)
            e = f"{row['relative_energy_meV']:9.2f} meV" if row["relative_energy_meV"] is not None else "        -    "
            s2 = f"{row['S2']:.3f}" if row["S2"] is not None else "  -  "
            print(f"  {row['configuration']}  Ms={row['FinalMs']}  {e}  S2={s2}  [{spins}]  {row['status']}")


if __name__ == "__main__":
    main()
