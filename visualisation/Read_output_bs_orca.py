#!/usr/bin/env python3
"""
Broken-symmetry energies and pairwise exchange couplings of the dimers_run2 four-radical dimers
(PBE0 FlipSpin single points in <run>_broken_symmetry/<stage>/<dimer>/<config>/<config>.out).
Companion of Read_output_relax_orca.py.

Convention (exchange_convention.py, docs/Riccardo/exchange_couplings.tex):
    H = - sum_{i<j} J_ij S_i . S_j,   J > 0 FM, J < 0 AFM;   sites 1-4 = radicals A B C D
    E_k = E_0 - 1/4 sum_{i<j} J_ij sigma_i sigma_j,  sigma = +1 (u) / -1 (d)   -> physical J_ij fitted directly

Per configuration (uuuu, duuu, uduu, uudu, uuud, dduu, dudu, duud; u/d = spin of radicals A B C D):
  status (done / NOT finished + reason), final energy, energy relative to uuuu, <S^2> and its ideal
  broken-symmetry value, fragment spins of A-D (Mulliken, halves from radical_fragments.json), SCF cycles,
  and two separate checks:
    pattern_ok        signs of A-D equal the requested pattern (M_S = 0: global reversal also accepted)
    local_moments_ok  |spin| >= 0.5 on every radical
  used_in_fit = pattern_ok and local_moments_ok.

Per dimer: J_12 ... J_34 (meV) from exchange_convention.fit_heisenberg() on the used configurations.
With 8 the fit is overdetermined by one (the four-spin term), and its residual measures how pairwise
the energies are; with 7 it is solved exactly (no check); with fewer, no couplings are reported.

Outputs (per run):
  visualisation/output/orca/dimers_run2/<run>_broken_symmetry/bs_energies.csv
  visualisation/output/orca/dimers_run2/<run>_broken_symmetry/exchange_couplings.csv
  visualisation/latex_tables/orca/dimers_run2/<run>_broken_symmetry/<run>_bs_energies.tex
  visualisation/latex_tables/orca/dimers_run2/<run>_broken_symmetry/<run>_exchange.tex
  visualisation/output/orca/dimers_run2/<run>_broken_symmetry/<run>_exchange_4spin.csv
      column layout of the SIESTA table of latex_exhange_couplings.py
      (Config,E_lower,J_12 (meV),...,J_34 (meV),F_lower,E0 (eV),Residual (eV),Rank), but with the PHYSICAL
      Heisenberg J_ij of the convention above (the SIESTA script reports J/4: multiply those by 4 to compare);
        sites 1-4 = radicals A-D (radical carbons 28, 41, 28+N, 41+N = SIESTA atoms 29, 42, 115, 128);
        E_lower = lowest of the fitted configuration energies (eV); F_lower = remaining MAX force of the
        dimer geometry (eV/Angstrom: last cycle of its R2SCAN-3C relaxation for stage "relaxed", first
        cycle for "initial"), since the broken-symmetry single points have no forces;
        Residual (eV) = sqrt(sum of squared fit residuals); Rank = rank of the fit matrix.
"""

import csv
import json
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange_convention as XC  # noqa: E402

# =========================
# USER SETTINGS
# =========================
REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = ["run2", "run3"]
DIMERS_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
OUT_CSV_ROOT = REPO_ROOT / "visualisation" / "output" / "orca" / "dimers_run2"
OUT_TEX_ROOT = REPO_ROOT / "visualisation" / "latex_tables" / "orca" / "dimers_run2"

HARTREE_TO_MEV = XC.HARTREE_TO_MEV
RADICALS = "ABCD"                                   # sites 1..4
CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
PAIRS = XC.pair_labels(4)                           # "12", "13", "14", "23", "24", "34"
PAIR_LETTERS = {p: RADICALS[int(p[0]) - 1] + RADICALS[int(p[1]) - 1] for p in PAIRS}
INTRA = {"12", "34"}                                # A-B and C-D
STRONG_J_MEV = 50.0          # |J| above which the Ising (broken-symmetry) mapping is only qualitative
SPIN_DEVIATION = 0.15        # uuuu fragment spin further than this from 1 -> radical not localized on its half

ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
SCF_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)")
MAX_GRAD_RE = re.compile(r"MAX gradient\s+(-?\d+\.\d+)\s+\d+\.\d+\s+(?:YES|NO)")
EH_PER_BOHR_TO_EV_PER_A = 27.211386245988 / 0.529177210903
MULLIKEN_RE = re.compile(r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\nSum of atomic", re.S)
ATOM_RE = re.compile(r"^\s*(\d+)\s+\S+\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.M)


# =========================
# EXCHANGE FIT
# =========================
def fit_exchange(energies_eh):
    """
    J_12 ... J_34 (meV) from broken-symmetry energies {config: Eh} via exchange_convention.fit_heisenberg
    (E_k = E_0 - 1/4 sum J_ij sigma_i sigma_j, physical J fitted directly). Energies are shifted by the
    lowest one before fitting (numerical conditioning only; E_0 is shifted back).
    """
    configs = [c for c in CONFIGS if c in energies_eh]
    e_min = min(energies_eh.values()) if energies_eh else 0.0
    fit = XC.fit_heisenberg({c: (energies_eh[c] - e_min) * HARTREE_TO_MEV for c in configs})
    fit["E0_Eh"] = fit["E0"] / HARTREE_TO_MEV + e_min if np.isfinite(fit["E0"]) else np.nan
    fit["max_res"] = max((abs(v) for v in fit["residuals"].values()), default=np.nan)
    fit["J"] = {p: fit["J"].get(p, np.nan) for p in PAIRS}        # all six pairs, NaN if not determined
    return fit


# =========================
# PARSING
# =========================
def parse_config(cfg_dir, fragments):
    name = cfg_dir.name
    out, err = cfg_dir / f"{name}.out", cfg_dir / "slurm.err"
    row = {"config": name, "flipped": "".join(r for r, s in zip(RADICALS, name) if s == "d") or "-",
           "Ms": (name.count("u") - name.count("d")) / 2, "S2_ideal": {0: 6.0, 1: 3.0, 2: 2.0}[name.count("d")]}
    if not out.exists():
        row["status"], row["reason"] = "NOT finished", "no output"
        row["pattern_ok"] = row["local_moments_ok"] = row["used_in_fit"] = False
        return row
    text = out.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in text:
        err_text = err.read_text(errors="ignore") if err.exists() else ""
        row["status"] = "NOT finished"
        row["reason"] = ("time limit" if "DUE TO TIME LIMIT" in err_text else
                         "SCF not converged" if "SCF NOT CONVERGED" in text else "running or killed")
        row["pattern_ok"] = row["local_moments_ok"] = row["used_in_fit"] = False
        return row
    row["status"], row["reason"] = "done", ""
    row["E_Eh"] = float(ENERGY_RE.findall(text)[-1])
    s2 = S2_RE.findall(text)
    row["S2"] = float(s2[-1]) if s2 else np.nan
    row["scf_cycles"] = "+".join(SCF_RE.findall(text))
    blocks = MULLIKEN_RE.findall(text)
    if blocks and fragments:
        spins = {int(m.group(1)): float(m.group(3)) for m in ATOM_RE.finditer(blocks[-1])}
        for r in RADICALS:
            row[f"spin_{r}"] = sum(spins.get(a, 0.0) for a in fragments[r])
        frag_spins = [row[f"spin_{r}"] for r in RADICALS]
        row["pattern_ok"] = XC.pattern_ok(name, frag_spins)
        row["local_moments_ok"] = XC.local_moments_ok(frag_spins)
    else:
        row["pattern_ok"] = row["local_moments_ok"] = False
    row["used_in_fit"] = bool(row["pattern_ok"] and row["local_moments_ok"])
    return row


def relaxation_max_force(run, stage, dimer):
    """Remaining MAX Cartesian gradient (eV/Angstrom) of the dimer geometry used for the single points."""
    out = DIMERS_ROOT / run / dimer / "high_spin" / "high_spin.out"
    if not out.exists():
        return np.nan
    grads = MAX_GRAD_RE.findall(out.read_text(errors="ignore"))
    if not grads:
        return np.nan
    return float(grads[0 if stage == "initial" else -1]) * EH_PER_BOHR_TO_EV_PER_A


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def process_run(run):
    bs_root = DIMERS_ROOT / f"{run}_broken_symmetry"
    relax_status = {}
    st_file = bs_root / "relaxation_status.csv"
    if st_file.exists():
        relax_status = {r["dimer"]: r["relaxation"] for r in csv.DictReader(st_file.open())}

    e_rows, j_rows = [], []
    for stage_dir in sorted(p for p in bs_root.iterdir() if p.is_dir()):
        for dimer_dir in sorted((p for p in stage_dir.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
            frag_file = dimer_dir / "radical_fragments.json"
            fragments = json.loads(frag_file.read_text())["fragments"] if frag_file.exists() else {}
            kinds = {}
            f4 = dimer_dir / "fragments4.json"
            if f4.exists():
                kinds = {k: re.sub(r"M\d-", "", v) for k, v in json.loads(f4.read_text()).get("json_label_to_half", {}).items()}
            relax = relax_status.get(dimer_dir.name, "") if stage_dir.name == "relaxed" else "initial geometry"

            rows = [parse_config(dimer_dir / c, fragments) for c in CONFIGS if (dimer_dir / c).is_dir()]
            if not any(r["status"] == "done" for r in rows):
                continue
            e_ref = next((r["E_Eh"] for r in rows if r["config"] == "uuuu" and r["status"] == "done"), np.nan)
            for r in rows:
                r.update({"stage": stage_dir.name, "dimer": dimer_dir.name, "relaxation": relax})
                if r["status"] == "done":
                    r["E_minus_uuuu_meV"] = (r["E_Eh"] - e_ref) * HARTREE_TO_MEV
                e_rows.append(r)

            usable = {r["config"]: r["E_Eh"] for r in rows if r["status"] == "done" and r["used_in_fit"]}
            fit = fit_exchange(usable)
            jr = {"stage": stage_dir.name, "dimer": dimer_dir.name, "relaxation": relax,
                  "n_configs_done": sum(r["status"] == "done" for r in rows),
                  "n_configs_in_fit": fit["n_configs"], "fit": fit["quality"]}
            for p in PAIRS:
                jr[f"J_{p}_meV"] = fit["J"][p]
            jr.update({"rms_residual_meV": fit["rms"], "max_residual_meV": fit["max_res"],
                       "E0_Eh": fit["E0_Eh"], "residual_norm_meV": fit["residual_norm"], "rank": fit["rank"],
                       "E_lower_Eh": min((usable[c] for c in fit["residuals"]), default=np.nan),
                       "F_lower_eV_A": relaxation_max_force(run, stage_dir.name, dimer_dir.name)})
            for r in RADICALS:
                jr[f"kind_{r}"] = kinds.get(r, "")
            uuuu = next((r for r in rows if r["config"] == "uuuu" and r["status"] == "done"), None)
            dev = max(abs(abs(uuuu.get(f"spin_{r}", 1.0)) - 1.0) for r in RADICALS) if uuuu else np.nan
            strong = [p for p in PAIRS if abs(fit["J"][p]) > STRONG_J_MEV]
            jr["max_uuuu_spin_deviation"] = dev
            jr["strong_coupling_pairs"] = " ".join(strong)
            jr["regime"] = ("strong coupling: Ising mapping qualitative" if strong or dev > SPIN_DEVIATION
                            else "weak coupling" if fit["quality"] != "insufficient" else "")
            jr["wrong_pattern_configs"] = " ".join(r["config"] for r in rows if r["status"] == "done" and not r["pattern_ok"])
            jr["weak_moment_configs"] = " ".join(r["config"] for r in rows if r["status"] == "done" and not r["local_moments_ok"])
            for c, v in fit["residuals"].items():
                next(x for x in rows if x["config"] == c)["fit_residual_meV"] = v
            j_rows.append(jr)
            print(f"  {stage_dir.name}/{dimer_dir.name:10s} done {jr['n_configs_done']}/8, in fit {fit['n_configs']}, "
                  f"fit: {fit['quality']}" + (f", rms {fit['rms']:.4f} meV" if fit["quality"] == "overdetermined" else "")
                  + (f"   wrong pattern: {jr['wrong_pattern_configs']}" if jr["wrong_pattern_configs"] else "")
                  + (f"   |spin|<0.5: {jr['weak_moment_configs']}" if jr["weak_moment_configs"] else ""))
    return pd.DataFrame(e_rows), pd.DataFrame(j_rows)


# =========================
# LATEX
# =========================
def fmt(v, f):
    return f"{v:{f}}" if pd.notna(v) else "--"


def tex_dimer(stage, dimer, relaxation, multi_stage):
    name = dimer.replace("_", r"\_")
    if multi_stage:
        name = f"{stage}/{name}"
    return name + (r"$^{\dagger}$" if relaxation == "not_converged" else "")


def write_energy_table(edf, run, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    multi = edf["stage"].nunique() > 1
    cols = CONFIGS[1:]
    head = "Dimer & " + " & ".join(cols) + r" & lowest \\"
    with path.open("w", encoding="utf-8") as tex:
        tex.write(r"""
\begin{longtable}{@{}l""" + "r" * len(cols) + r"""l@{}}
\caption{PBE0 broken-symmetry energies of the """ + run + r""" dimers, relative to the high-spin
         configuration uuuu (meV). u/d give the spin of radicals A B C D; A, B belong to monomer 1 and
         C, D to monomer 2. ``lowest'' is the configuration of lowest energy. Configurations that are not
         finished, did not converge to the intended spin pattern, or have a radical with $|$spin$|<0.5$ are
         shown as -- (they are not used in the exchange fit).
         $^{\dagger}$: relaxation of the dimer geometry not converged.}\label{tab:""" + run + r"""_bs_energies}\\
\toprule
""" + head + r"""
\midrule
\endfirsthead
\toprule
""" + head + r"""
\midrule
\endhead
\bottomrule
\endlastfoot
""")
        for (stage, dimer), g in edf.groupby(["stage", "dimer"], sort=False):
            g = g.set_index("config")
            ok = g[(g["status"] == "done") & (g["used_in_fit"] == True)]  # noqa: E712
            vals = [fmt(g.loc[c, "E_minus_uuuu_meV"], "+.3f") if c in ok.index else "--" for c in cols]
            lowest = ok["E_Eh"].idxmin() if len(ok) else "--"
            tex.write(f"{tex_dimer(stage, dimer, g['relaxation'].iloc[0], multi)} & " + " & ".join(vals)
                      + f" & {lowest} \\\\\n")
        tex.write(r"\end{longtable}" + "\n")


def write_exchange_table(jdf, run, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    multi = jdf["stage"].nunique() > 1
    head = ("Dimer & " + " & ".join(f"$J_{{{p}}}$" for p in PAIRS) + r" & rms (meV) & fit \\")
    with path.open("w", encoding="utf-8") as tex:
        tex.write(r"""
\begin{longtable}{@{}lrrrrrrrl@{}}
\caption{Pairwise exchange couplings of the """ + run + r""" dimers (meV), from a least-squares fit of the
         broken-symmetry energies to $\hat H = -\sum_{i<j} J_{ij}\,\hat S_i\cdot\hat S_j$ ($S=\tfrac12$ per
         radical), mapped as $E_k=E_0-\tfrac14\sum_{i<j}J_{ij}\sigma_i^{(k)}\sigma_j^{(k)}$; $J>0$ ferromagnetic,
         $J<0$ antiferromagnetic. Sites 1--4 are radicals A--D; $J_{12}$ and $J_{34}$ are intramolecular,
         the other four intermolecular. ``rms'' is the fit residual (8 configurations, one
         redundant: a measure of non-pairwise terms); ``exact'' = 7 configurations, no check.
         $^{\dagger}$: relaxation of the dimer geometry not converged. $^{\S}$: strong coupling
         ($|J|>""" + f"{STRONG_J_MEV:.0f}" + r"""$\,meV, or high-spin radical spins deviating from 1): the broken-symmetry
         spins are far from $\pm1$ and these couplings are only qualitative.}\label{tab:""" + run + r"""_exchange}\\
\toprule
""" + head + r"""
\midrule
\endfirsthead
\toprule
""" + head + r"""
\midrule
\endhead
\bottomrule
\endlastfoot
""")
        for _, r in jdf.iterrows():
            vals = [fmt(r[f"J_{p}_meV"], "+.3f") for p in PAIRS]
            fit = {"overdetermined": "8 conf.", "exact": "exact (7)"}.get(r["fit"], r"\textit{insufficient}")
            mark = r"$^{\S}$" if str(r.get("regime", "")).startswith("strong") else ""
            tex.write(f"{tex_dimer(r['stage'], r['dimer'], r['relaxation'], multi)}{mark} & " + " & ".join(vals)
                      + f" & {fmt(r['rms_residual_meV'], '.4f')} & {fit} \\\\\n")
        tex.write(r"\end{longtable}" + "\n")


def write_siesta_style_csv(jdf, path):
    """Column layout of latex_exhange_couplings.py, with the physical Heisenberg J_ij (no rescaling)."""
    multi = jdf["stage"].nunique() > 1
    rows = []
    for _, r in jdf[jdf["fit"] != "insufficient"].iterrows():
        m = re.fullmatch(r"conf_(\d+)", r["dimer"])
        cfg = int(m.group(1)) if (m and not multi) else (f"{r['stage']}/{r['dimer']}" if multi else r["dimer"])
        row = {"Config": cfg, "E_lower": r["E_lower_Eh"] * HARTREE_TO_MEV / 1000}
        for p in PAIRS:
            row[f"J_{p} (meV)"] = r[f"J_{p}_meV"]
        row["F_lower"] = r["F_lower_eV_A"]
        row["E0 (eV)"] = r["E0_Eh"] * HARTREE_TO_MEV / 1000
        row["Residual (eV)"] = r["residual_norm_meV"] / 1000
        row["Rank"] = int(r["rank"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    for run in RUNS:
        print(f"=== {run}_broken_symmetry")
        edf, jdf = process_run(run)
        if edf.empty:
            print("  no finished configurations")
            continue
        out_dir = OUT_CSV_ROOT / f"{run}_broken_symmetry"
        out_dir.mkdir(parents=True, exist_ok=True)
        first = ["stage", "dimer", "relaxation", "config", "flipped", "Ms", "status", "reason", "E_Eh",
                 "E_minus_uuuu_meV", "fit_residual_meV", "S2", "S2_ideal", "spin_A", "spin_B", "spin_C", "spin_D",
                 "pattern_ok", "local_moments_ok", "used_in_fit", "scf_cycles"]
        edf = edf[[c for c in first if c in edf.columns] + [c for c in edf.columns if c not in first]]
        edf.to_csv(out_dir / "bs_energies.csv", index=False)
        jdf.to_csv(out_dir / "exchange_couplings.csv", index=False)
        write_siesta_style_csv(jdf, out_dir / f"{run}_exchange_4spin.csv")
        tex_dir = OUT_TEX_ROOT / f"{run}_broken_symmetry"
        write_energy_table(edf, run, tex_dir / f"{run}_bs_energies.tex")
        write_exchange_table(jdf, run, tex_dir / f"{run}_exchange.tex")
        n_fit = int((jdf["fit"] != "insufficient").sum())
        print(f"  {len(jdf)} dimers with results, couplings for {n_fit}")
        print(f"  -> {out_dir}/bs_energies.csv, exchange_couplings.csv, {run}_exchange_4spin.csv\n  -> {tex_dir}/")


if __name__ == "__main__":
    main()
