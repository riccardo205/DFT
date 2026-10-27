#!/usr/bin/env python3
"""
Pairwise exchange couplings of one four-radical dimer from its broken-symmetry energies.

Convention (visualisation/exchange_convention.py, docs/Riccardo/exchange_couplings.tex):
    H = - sum_{i<j} J_ij S_i . S_j          J > 0 ferromagnetic, J < 0 antiferromagnetic
    E_k = E_0 - 1/4 sum_{i<j} J_ij sigma_i sigma_j,     sigma = +1 (u) / -1 (d)
fitted for the physical J_ij directly. Sites 1-4 are the radicals A B C D, so J_12 = J_AB, ...,
J_34 = J_CD. The 8 configurations uuuu, duuu, uduu, uudu, uuud, dduu, dudu, duud are the 8 classes of
the 16 spin arrangements (a global flip gives the same energy), so E0 + 6 J (7 unknowns) are fitted by
least squares to 8 energies; the remaining degree of freedom is the four-spin term, so the residual
measures how far the energies are from a purely pairwise model.

A configuration enters the fit only if pattern_ok (signs of A-D as requested; for M_S = 0 the global
reversal is also accepted) and local_moments_ok (|spin| >= 0.5 on every radical).

Usage: python3 fit_dimer_exchange.py <run>_broken_symmetry/<stage>/<dimer> [...]
Writes <dimer>/bs_energies.csv and <dimer>/exchange_couplings.csv.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "visualisation"))
from prepare_dimers_broken_symmetry import RADICALS, mulliken_spins  # noqa: E402
import exchange_convention as XC  # noqa: E402

HARTREE_TO_MEV = XC.HARTREE_TO_MEV
CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
PAIRS = XC.pair_labels(4)
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
SCF_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)")


def fit(dimer_dir):
    d = Path(dimer_dir)
    frags = json.loads((d / "radical_fragments.json").read_text())["fragments"]
    geo = read(d / "uuuu" / "geometry.xyz")
    n = len(geo) // 2
    centres = [28, 41, 28 + n, 41 + n]
    kind = {}
    f4 = d / "fragments4.json"
    if f4.exists():
        lab = json.loads(f4.read_text()).get("json_label_to_half", {})
        kind = {k: re.sub(r"M\d-", "", v) for k, v in lab.items()}

    rows = []
    for c in CONFIGS:
        out = d / c / f"{c}.out"
        text = out.read_text(errors="ignore") if out.exists() else ""
        if "ORCA TERMINATED NORMALLY" not in text:
            print(f"  {c}: missing or not terminated normally, skipped")
            continue
        spins = mulliken_spins(out)
        fs = [sum(spins[i] for i in frags[r]) for r in RADICALS]
        rows.append({"config": c, "flipped": "".join(r for r, s in zip(RADICALS, c) if s == "d") or "-",
                     "Ms": (c.count("u") - c.count("d")) / 2,
                     "E_Eh": float(ENERGY_RE.findall(text)[-1]), "S2": float(S2_RE.findall(text)[-1]),
                     "spins": fs, "scf_cycles": "+".join(SCF_RE.findall(text)),
                     "pattern_ok": XC.pattern_ok(c, fs), "local_moments_ok": XC.local_moments_ok(fs)})
    for r in rows:
        r["used_in_fit"] = r["pattern_ok"] and r["local_moments_ok"]
        if not r["used_in_fit"]:
            print(f"  {r['config']}: excluded (pattern_ok={r['pattern_ok']}, local_moments_ok={r['local_moments_ok']}, "
                  f"spins {' '.join(f'{x:+.2f}' for x in r['spins'])})")

    e_ref = min(r["E_Eh"] for r in rows)
    res = XC.fit_heisenberg({r["config"]: (r["E_Eh"] - e_ref) * HARTREE_TO_MEV for r in rows if r["used_in_fit"]})
    if res["quality"] == "insufficient":
        raise RuntimeError(f"{d}: only {res['n_configs']} usable configurations, J not determined")
    e_uuuu = next((r["E_Eh"] for r in rows if r["config"] == "uuuu"), np.nan)

    with (d / "bs_energies.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "flipped_radicals", "Ms", "E_Eh", "E_minus_E_uuuu_meV", "E_minus_E_min_meV",
                    "fit_residual_meV", "S2", "S2_ideal_BS", "spin_A", "spin_B", "spin_C", "spin_D",
                    "pattern_ok", "local_moments_ok", "used_in_fit", "scf_cycles"])
        for r in rows:
            nd = r["config"].count("d")
            resid = res["residuals"].get(r["config"])
            w.writerow([r["config"], r["flipped"], f"{r['Ms']:.0f}", f"{r['E_Eh']:.9f}",
                        f"{(r['E_Eh'] - e_uuuu) * HARTREE_TO_MEV:.4f}", f"{(r['E_Eh'] - e_ref) * HARTREE_TO_MEV:.4f}",
                        "" if resid is None else f"{resid:+.4f}", f"{r['S2']:.4f}", {0: 6.0, 1: 3.0, 2: 2.0}[nd],
                        *(f"{x:+.4f}" for x in r["spins"]), r["pattern_ok"], r["local_moments_ok"],
                        r["used_in_fit"], r["scf_cycles"]])

    with (d / "exchange_couplings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "radicals", "relation", "kind_i", "kind_j", "centre_i", "centre_j", "centre_distance_A",
                    "J_meV", "type", "fit", "n_configs_in_fit", "fit_rms_residual_meV"])
        for p in PAIRS:
            i, j = int(p[0]) - 1, int(p[1]) - 1
            a, b = RADICALS[i], RADICALS[j]
            J = res["J"][p]
            w.writerow([f"J_{p}", a + b, "intramolecular" if (i < 2) == (j < 2) else "intermolecular",
                        kind.get(a, ""), kind.get(b, ""), centres[i], centres[j],
                        f"{geo.get_distance(centres[i], centres[j]):.3f}", f"{J:+.4f}", "FM" if J > 0 else "AFM",
                        res["quality"], res["n_configs"], f"{res['rms']:.4f}"])

    print(f"{d}: {res['n_configs']} configurations, fit {res['quality']}, rms residual {res['rms']:.4f} meV")
    for p in PAIRS:
        i, j = int(p[0]) - 1, int(p[1]) - 1
        print(f"  J_{p} ({RADICALS[i]}{RADICALS[j]}) = {res['J'][p]:+.4f} meV  "
              f"({'intra' if (i < 2) == (j < 2) else 'inter'}, r = {geo.get_distance(centres[i], centres[j]):.2f} A)")
    return res


if __name__ == "__main__":
    for path in sys.argv[1:]:
        fit(path)
