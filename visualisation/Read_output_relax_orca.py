#!/usr/bin/env python3
"""
ORCA counterpart of Read_output_relax.py + build_latex_table.py for the dimer relaxations of
dimers_run2 (R2SCAN-3C quintet geometry optimisations, one <conf>/high_spin/high_spin.out each).
Runs with variants (e.g. run4/<set>/<conf>/high_spin) are handled too: the configuration is then
labelled <set>/<conf>, and E_rel_in_set_meV is relative to the lowest energy of the same set.

For every configuration it extracts:
  - status: converged (ORCA's "THE OPTIMIZATION HAS CONVERGED" banner) or NOT converged, with the
    reason (time limit, ORCA error, still running / no termination message);
  - energies: first and last FINAL SINGLE POINT ENERGY (Eh and eV), relaxation energy gained in
    this run, energy relative to the lowest configuration (meV);
  - remaining "forces": ORCA's geometry-convergence table of the LAST cycle (energy change, RMS/MAX
    gradient, RMS/MAX step with tolerances and YES/NO), with the gradients also in eV/Angstrom
    (the SIESTA-like force measure); the MAX gradient of the first cycle is F_init;
  - number of geometry cycles, SCF cycles per geometry step (mean/max) and TRAH activations;
  - <S^2> and Mulliken spins of the last population analysis: total spin, sum |spin| and, if
    <run>_broken_symmetry/relaxed/<conf>/radical_fragments.json exists, the spin of the four
    radical halves A-D; atoms with |spin| > SPIN_LIST_THRESHOLD as an index:spin list;
  - geometry of the last structure: closest contact between the two monomers (any atom, C...C)
    and distance between the monomer centres.
For unconverged runs the LAST available values are reported and the row is flagged.

Outputs:
  visualisation/output/dimers_run2/<run>/relaxation_summary.csv      one row per configuration
  visualisation/output/dimers_run2/<run>/relaxation_trajectory.csv   energy / gradient per cycle
  visualisation/latex_tables/dimers_run2/<run>/<run>_relaxation.tex  longtable, lowest E in grey,
                                                                      unconverged rows marked
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_angle_scan import CONV_TABLE_RE, CONV_CRITERIA_NAMES, MULLIKEN_BLOCK_RE, ATOM_LINE_RE  # noqa: E402

# =========================
# USER SETTINGS
# =========================
REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = "run2"
RUN_DIR = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2" / RUN
FRAGMENT_DIR = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2" / f"{RUN}_broken_symmetry" / "relaxed"
OUT_CSV_DIR = REPO_ROOT / "visualisation" / "output" / "orca" / "dimers_run2" / RUN
OUT_TEX = REPO_ROOT / "visualisation" / "latex_tables" / "orca" /"dimers_run2" / RUN / f"{RUN}_relaxation.tex"
SPIN_LIST_THRESHOLD = 0.05          # |spin| for the index:spin list (as in Read_output_relax.py)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_MEV = 1000 * HARTREE_TO_EV
EH_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / 0.529177210903

# =========================
# REGEX PATTERNS
# =========================
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
SCF_CYCLES_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
TRAH_RE = re.compile(r"Leaving SCF to start the TRAH")
CONVERGED_BANNER = "THE OPTIMIZATION HAS CONVERGED"


def last_spins(text):
    blocks = list(MULLIKEN_BLOCK_RE.finditer(text))
    if not blocks:
        return {}
    spins = {}
    for line in blocks[-1].group(1).splitlines():
        m = ATOM_LINE_RE.match(line)
        if m:
            spins[int(m.group(1))] = float(m.group(4))
    return spins


def label_of(conf_dir):
    """<conf> for run/<conf>/high_spin, <set>/<conf> for run/<set>/<conf>/high_spin."""
    return conf_dir.parent.relative_to(RUN_DIR).as_posix()


def status_of(conf_dir, text):
    if CONVERGED_BANNER in text:
        return "converged", ""
    err = conf_dir / "slurm.err"
    err_text = err.read_text(errors="ignore") if err.exists() else ""
    if "DUE TO TIME LIMIT" in err_text:
        reason = "time limit"
    elif "ORCA TERMINATED NORMALLY" in text:
        reason = "terminated without convergence (geometry MaxIter?)"
    elif re.search(r"aborting the run|ORCA finished by error|SCF NOT CONVERGED", text):
        reason = "ORCA error / SCF not converged"
    else:
        reason = "no termination message (running or killed)"
    return "NOT converged", reason


def parse_conf(conf_dir):
    out = conf_dir / "high_spin.out"
    if not out.exists():
        return None, []
    text = out.read_text(errors="ignore")
    energies = [float(x) for x in ENERGY_RE.findall(text)]
    tables = CONV_TABLE_RE.findall(text)
    if not energies:
        return None, []
    status, reason = status_of(conf_dir, text)

    label = label_of(conf_dir)
    row = {"Config": label, "Set": label.rpartition("/")[0], "Status": status, "Reason": reason,
           "Geom_cycles": len(CYCLE_RE.findall(text)),
           "E_init_Eh": energies[0], "E_final_Eh": energies[-1],
           "E_final_eV": energies[-1] * HARTREE_TO_EV,
           "Relax_energy_meV": (energies[-1] - energies[0]) * HARTREE_TO_MEV}

    # convergence table: first (F_init) and last cycle
    if tables:
        for name, i in zip(CONV_CRITERIA_NAMES, range(5)):
            row[f"{name}"] = float(tables[-1][3 * i])
            row[f"{name}_tol"] = float(tables[-1][3 * i + 1])
            row[f"{name}_ok"] = tables[-1][3 * i + 2]
        row["Criteria_met"] = sum(tables[-1][3 * i + 2] == "YES" for i in range(5))
        row["F_init_eV_A"] = float(tables[0][6]) * EH_PER_BOHR_TO_EV_PER_A
        row["F_final_eV_A"] = row["max_gradient"] * EH_PER_BOHR_TO_EV_PER_A
        row["F_rms_final_eV_A"] = row["rms_gradient"] * EH_PER_BOHR_TO_EV_PER_A

    scf = [int(x) for x in SCF_CYCLES_RE.findall(text)]
    row["SCF_cycles_mean"] = float(np.mean(scf)) if scf else np.nan
    row["SCF_cycles_max"] = max(scf) if scf else np.nan
    row["TRAH_activations"] = len(TRAH_RE.findall(text))
    s2 = S2_RE.findall(text)
    row["S2_final"] = float(s2[-1]) if s2 else np.nan

    spins = last_spins(text)
    if spins:
        row["S_tot"] = sum(spins.values())
        row["Sum_abs_spin"] = sum(abs(s) for s in spins.values())
        row["Spin_final_list"] = "; ".join(f"{i}:{s:.3f}" for i, s in sorted(spins.items())
                                            if abs(s) > SPIN_LIST_THRESHOLD)
        frag_file = FRAGMENT_DIR / label / "radical_fragments.json"
        if frag_file.exists():
            import json
            frags = json.loads(frag_file.read_text())["fragments"]
            for lab, atoms in frags.items():
                row[f"Spin_{lab}"] = sum(spins.get(a, 0.0) for a in atoms)

    # last geometry: monomer contact
    xyz = conf_dir / "high_spin.xyz"
    if xyz.exists():
        atoms = read(xyz)
        n = len(atoms) // 2
        pos, heavy = atoms.positions, atoms.numbers > 1
        d = np.linalg.norm(pos[:n, None] - pos[None, n:], axis=-1)
        row["Min_contact_A"] = float(d.min())
        row["Min_CC_contact_A"] = float(d[np.ix_(heavy[:n], heavy[n:])].min())
        row["Centre_separation_A"] = float(np.linalg.norm(pos[:n].mean(0) - pos[n:].mean(0)))

    traj = []
    for k, e in enumerate(energies):
        t = {"Config": row["Config"], "Cycle": k + 1, "E_Eh": e,
             "E_minus_E_init_meV": (e - energies[0]) * HARTREE_TO_MEV}
        if k < len(tables):
            t["max_gradient_eV_A"] = float(tables[k][6]) * EH_PER_BOHR_TO_EV_PER_A
            t["rms_gradient_eV_A"] = float(tables[k][3]) * EH_PER_BOHR_TO_EV_PER_A
        traj.append(t)
    return row, traj


def write_latex(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    i_min = df["E_final_Eh"].idxmin()
    with path.open("w", encoding="utf-8") as tex:
        tex.write(r"""
\begin{longtable}{@{}lrrrrrrrl@{}}
\caption{R2SCAN-3C quintet relaxations of the dimers (""" + RUN + r"""). $E_{\mathrm{rel}}$ is relative to the
         lowest final energy (grey row); $F_{\mathrm{max}}$ and $F_{\mathrm{rms}}$ are the maximum and RMS
         Cartesian gradient of the last cycle; $d_{\mathrm{CC}}$ is the closest C$\cdots$C contact between the
         two monomers. Rows marked $^{\dagger}$ did not converge: their values are those of the last
         completed cycle.}\label{tab:""" + RUN + r"""_relaxation}\\
\toprule
Config & $E_{\mathrm{final}}$ (eV) & $E_{\mathrm{rel}}$ (meV) & $F_{\mathrm{max}}$ (eV\,\AA$^{-1}$) &
$F_{\mathrm{rms}}$ (eV\,\AA$^{-1}$) & Cycles & $\langle S^2\rangle$ & $d_{\mathrm{CC}}$ (\AA) & Status\\
\midrule
\endfirsthead
\multicolumn{9}{c}{{\tablename}~\thetable{} --- continued from previous page}\\
\toprule
Config & $E_{\mathrm{final}}$ (eV) & $E_{\mathrm{rel}}$ (meV) & $F_{\mathrm{max}}$ (eV\,\AA$^{-1}$) &
$F_{\mathrm{rms}}$ (eV\,\AA$^{-1}$) & Cycles & $\langle S^2\rangle$ & $d_{\mathrm{CC}}$ (\AA) & Status\\
\midrule
\endhead
\midrule
\multicolumn{9}{r}{\emph{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
""")
        for idx, r in df.iterrows():
            hl = r"\rowcolor{gray!20} " if idx == i_min else ""
            conv = r["Status"] == "converged"
            cfg = r["Config"].replace("_", r"\_") + ("" if conv else r"$^{\dagger}$")
            status = "converged" if conv else r"\textbf{not conv.}"
            fmt = lambda v, f: f"{v:{f}}" if pd.notna(v) else "--"
            tex.write(f"{hl}{cfg} & {r['E_final_eV']:.4f} & {fmt(r['E_rel_meV'], '.1f')} & "
                      f"{fmt(r.get('F_final_eV_A'), '.4f')} & {fmt(r.get('F_rms_final_eV_A'), '.4f')} & "
                      f"{int(r['Geom_cycles'])} & {fmt(r['S2_final'], '.3f')} & "
                      f"{fmt(r.get('Min_CC_contact_A'), '.2f')} & {status} \\\\\n")
        tex.write(r"\end{longtable}" + "\n")


def main():
    rows, traj = [], []
    def natural_key(path):                      # conf_2 < conf_10, docked_1 < packmol_1
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", label_of(path))]

    confs = sorted(list(RUN_DIR.glob("*/high_spin")) + list(RUN_DIR.glob("*/*/high_spin")), key=natural_key)
    if not confs:
        raise SystemExit(f"No <dimer>/high_spin folders found in {RUN_DIR}")
    for conf in confs:
        row, t = parse_conf(conf)
        if row is None:
            print(f"{label_of(conf)}: no energies yet, skipped")
            continue
        rows.append(row)
        traj += t
        flag = "" if row["Status"] == "converged" else f"   <-- NOT converged ({row['Reason']})"
        print(f"{row['Config']:24s} E = {row['E_final_Eh']:.8f} Eh, F_max = {row.get('F_final_eV_A', np.nan):.4f} eV/A, "
              f"{row['Geom_cycles']:3d} cycles, criteria met {row.get('Criteria_met', '-')}/5{flag}")

    if not rows:
        raise SystemExit(f"No high_spin.out with energies found in {RUN_DIR}")
    df = pd.DataFrame(rows)
    df["E_rel_meV"] = (df["E_final_Eh"] - df["E_final_Eh"].min()) * HARTREE_TO_MEV
    df["E_rel_in_set_meV"] = (df["E_final_Eh"] - df.groupby("Set")["E_final_Eh"].transform("min")) * HARTREE_TO_MEV
    df = df.sort_values("E_final_Eh").reset_index(drop=True)
    first = ["Config", "Set", "Status", "Reason", "E_final_Eh", "E_final_eV", "E_rel_meV", "E_rel_in_set_meV", "E_init_Eh", "Relax_energy_meV",
             "F_init_eV_A", "F_final_eV_A", "F_rms_final_eV_A", "Criteria_met", "Geom_cycles"]
    df = df[first + [c for c in df.columns if c not in first]]

    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_DIR / "relaxation_summary.csv", index=False)
    pd.DataFrame(traj).to_csv(OUT_CSV_DIR / "relaxation_trajectory.csv", index=False)
    write_latex(df, OUT_TEX)
    n_conv = int((df["Status"] == "converged").sum())
    print(f"\n{len(df)} configurations: {n_conv} converged, {len(df) - n_conv} NOT converged")
    print(f"-> {OUT_CSV_DIR / 'relaxation_summary.csv'}\n-> {OUT_CSV_DIR / 'relaxation_trajectory.csv'}\n-> {OUT_TEX}")


if __name__ == "__main__":
    main()
