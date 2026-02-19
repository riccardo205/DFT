r"""
latex_exhange_couplings.py  --  build LaTeX table of exchange couplings
---------------------------------------------------------------------

For each Config:
  - If SPIN_NR=2: find ferro and antiferro, compute J = E_antiferro - E_ferro (keeps your original sign usage)
  - If SPIN_NR=4: solve 6 pairwise couplings from 8 collinear spin configurations via least-squares fit

IMPORTANT CONVENTION NOTE (kept on purpose)
-------------------------------------------
In *proper* spin-1/2 Heisenberg mapping one would write:

    H = - Σ_{i<j} J_ij (S_i · S_j)
and for collinear states with s_i = ±1:
    <S_i · S_j> = (1/4) s_i s_j

So the energy model is:
    E = E0 - (1/4) Σ J_ij s_i s_j

However, THIS SCRIPT PURPOSELY DOES *NOT* include the 1/4 factor.
Instead it fits the rescaled couplings J_res such that:

    E = E0 - Σ J_res,ij (s_i s_j)

Therefore:
    J_res,ij = (1/4) * J_Heisenberg,ij
or equivalently:
    J_Heisenberg,ij = 4 * J_res,ij

So the couplings written by this script are the *rescaled* Js,
which are 4x smaller than true Heisenberg spin-1/2 Js.

Sign convention (kept):
  - J_res < 0 : antiferromagnetic (prefers antiparallel, s_i s_j = -1 gives lower E)
  - J_res > 0 : ferromagnetic (prefers parallel, s_i s_j = +1 gives lower E)

For 4 spins (sites at indices 29, 42, 115, 128):
  Spin configs used (s1,s2,s3,s4):
    - ferro:      ( 1,  1, -1, -1)   # NOTE: name historical; not all-parallel
    - antiferro1: ( 1, -1,  1, -1)
    - antiferro2: ( 1, -1, -1,  1)
    - spin1_a:    (-1,  1,  1,  1)
    - spin1_b:    ( 1, -1,  1,  1)
    - spin1_c:    ( 1,  1, -1,  1)
    - spin1_d:    ( 1,  1,  1, -1)
    - spin2:      ( 1,  1,  1,  1)

6 exchanges: J_12, J_13, J_14, J_23, J_24, J_34

Preamble needed in your .tex:
    \usepackage{booktabs,longtable}
"""

import os
import pandas as pd
import numpy as np
import math

run_label = "dimers_run2"  # subfolder in Short-2OS/runs/ and visualisation/
fast_slow = "slow_lowT_lowest_forces"    # subfolder for fast or slow screening
INFILE  = f"visualisation/{run_label}/{fast_slow}/summary_results.csv"
OUTFILE_2SPIN = f"visualisation/latex_tables/{run_label}/{fast_slow}/{run_label}_exchange_2spin.tex"
OUTFILE_4SPIN = f"visualisation/latex_tables/{run_label}/{fast_slow}/{run_label}_exchange_4spin.tex"

# Mode: 2 or 4 spins
SPIN_NR = 4

# Color mode: True for colored J values, False for black
COLORED_LATEX = True

# ======================================================================
# HELPER: Color formatting for J values
# ======================================================================
def format_j_value(j_val, max_abs_j, colored=True):
    """
    Format a coupling J value with optional color.

    Input j_val is stored in eV in the dataframe; the function prints meV.

    colored=True:
      - Negative J (antiferro): blue
      - Positive J (ferromagnetic): red
      - Intensity (1-100) based on logarithmic scale of |J|
    colored=False:
      - Black text

    Returns LaTeX formatted string.
    """
    if pd.isna(j_val) or max_abs_j == 0:
        return r"--"

    j_meV = j_val * 1000  # eV -> meV

    if not colored:
        return f"{j_meV:.1f}"

    # Intensity (20..100) based on log scaling
    intensity = int(20 + 80 * math.log(1 + abs(j_val)) / math.log(1 + max_abs_j))

    if j_val < 0:
        color = f"blue!{intensity}"
    else:
        color = f"red!{intensity}"

    return r"\textcolor{" + color + r"}{" + f"{j_meV:.1f}" + r"}"


# ======================================================================
# SPIN-2 MODE (ferro vs antiferro)
# ======================================================================
def process_2spin(df):
    """Extract J = E_antiferro - E_ferro for each Config (keeps original sign usage)."""
    _df = df.copy()
    _df["E_final"] = pd.to_numeric(_df["E_final"], errors="coerce")
    _df["F_final"] = pd.to_numeric(_df["F_final"], errors="coerce")
    _df = _df.dropna(subset=["Config", "Spin init", "E_final"])

    spin_lower = _df["Spin init"].str.lower().fillna("")
    _df["is_ferro"] = spin_lower == "ferro"
    _df["is_antiferro"] = spin_lower == "antiferro"
    _df = _df[_df["is_ferro"] | _df["is_antiferro"]]

    rows = []
    for cfg, grp in _df.groupby("Config"):
        ferro = grp[grp["is_ferro"]]
        antiferro = grp[grp["is_antiferro"]]
        if ferro.empty or antiferro.empty:
            continue

        ferro_min = ferro.loc[ferro["E_final"].idxmin()]
        antiferro_min = antiferro.loc[antiferro["E_final"].idxmin()]

        e_ferro = float(ferro_min["E_final"])
        e_antiferro = float(antiferro_min["E_final"])

        f_ferro = ferro_min["F_final"]
        f_antiferro = antiferro_min["F_final"]

        e_lower = min(e_ferro, e_antiferro)
        f_lower = min(f for f in [f_ferro, f_antiferro] if pd.notna(f)) if (
            pd.notna(f_ferro) or pd.notna(f_antiferro)
        ) else pd.NA

        rows.append({
            "Config": int(cfg),
            "E_lower": e_lower,
            "J": (e_antiferro - e_ferro),  # eV
            "F_lower": f_lower,
            "Residual": 0.0,  # 2-spin case: exact fit with 2 points
        })

    return pd.DataFrame(rows).sort_values("Config").reset_index(drop=True)


# ======================================================================
# SPIN-4 MODE (8 configurations → 6 exchange couplings via least-squares)
# ======================================================================
def get_4spin_matrix():
    """
    Return (config_names, A) for 4-spin system.
    A: 8 x 6, where entry A[row, col] = s_a*s_b for that config and that pair.

    MODEL FITTED BY THIS SCRIPT (rescaled):
        E = E0 - Σ_{i<j} J_res,ij * (s_i s_j)

    where s_i = ±1 are *collinear* spins.

    Relation to proper spin-1/2 Heisenberg couplings:
        E = E0 - (1/4) Σ J_Heis,ij * (s_i s_j)
    Therefore:
        J_res,ij = (1/4) J_Heis,ij
        J_Heis,ij = 4 * J_res,ij

    Sign convention:
      - J_res < 0: antiferromagnetic (antiparallel preferred)
      - J_res > 0: ferromagnetic (parallel preferred)

    Couplings (columns): J_12, J_13, J_14, J_23, J_24, J_34
    Configs (rows): ferro, antiferro1, antiferro2, spin1_a, spin1_b, spin1_c, spin1_d, spin2
    Sites (spins): 1=idx29, 2=idx42, 3=idx115, 4=idx128
    """
    configs = {
        'ferro':      ( 1,  1, -1, -1),
        'antiferro1': ( 1, -1,  1, -1),
        'antiferro2': ( 1, -1, -1,  1),
        'spin1_a':    (-1,  1,  1,  1),
        'spin1_b':    ( 1, -1,  1,  1),
        'spin1_c':    ( 1,  1, -1,  1),
        'spin1_d':    ( 1,  1,  1, -1),
        'spin2':      ( 1,  1,  1,  1)
    }

    config_names = list(configs.keys())
    matrix = []

    for name in config_names:
        s1, s2, s3, s4 = configs[name]
        # Pairwise products: J_12, J_13, J_14, J_23, J_24, J_34
        row = [s1*s2, s1*s3, s1*s4, s2*s3, s2*s4, s3*s4]
        matrix.append(row)

    return config_names, np.array(matrix, dtype=float)


def process_4spin(df):
    """
    For each Config, extract energies from the 8 spin variants,
    and solve a least-squares system for (E0, 6 couplings).

    FITTED MODEL (rescaled):
        E = E0 - A @ J_res

    where A is 8x6 with A[row,col] = s_i*s_j for the corresponding pair.
    J_res are returned in eV and printed in meV.

    Reminder:
      J_res = (1/4) * J_Heisenberg for spin-1/2 isotropic Heisenberg mapping.
    """
    _df = df.copy()
    _df["E_final"] = pd.to_numeric(_df["E_final"], errors="coerce")
    _df["F_final"] = pd.to_numeric(_df["F_final"], errors="coerce")
    _df = _df.dropna(subset=["Config", "Spin init", "E_final"])

    spin_variants = {'ferro', 'antiferro1', 'antiferro2', 'spin1_a', 'spin1_b', 'spin1_c', 'spin1_d', 'spin2'}
    spin_lower_all = _df["Spin init"].str.lower().fillna("")
    _df = _df[spin_lower_all.isin(spin_variants)]

    config_names, A = get_4spin_matrix()
    coupling_names = ['J_12', 'J_13', 'J_14', 'J_23', 'J_24', 'J_34']

    rows = []
    for cfg, grp in _df.groupby("Config"):
        # Recompute spin_lower for this group index slice
        spin_lower = grp["Spin init"].str.lower().fillna("")

        # Extract minimum-energy entry for each spin variant (in fixed order)
        energy_vec = []
        f_min = float('inf')

        for variant in config_names:
            variant_grp = grp[spin_lower == variant]
            if variant_grp.empty:
                energy_vec.append(np.nan)
                continue

            min_row = variant_grp.loc[variant_grp["E_final"].idxmin()]
            energy_vec.append(float(min_row["E_final"]))

            if pd.notna(min_row["F_final"]):
                f_min = min(f_min, float(min_row["F_final"]))

        if any(np.isnan(e) for e in energy_vec):
            print(f"[!] Config {cfg}: missing some spin variants, skipping.")
            continue

        E = np.array(energy_vec, dtype=float)  # eV

        # Add intercept: E = E0 - A @ J_res
        A1 = np.column_stack([np.ones(len(E)), A])  # 8 x 7
        b = E

        # Solve A1 @ x ≈ b, with x = [E0, (-J_res)]
        # because b = E0*1 + A @ (-J_res)
        x, residuals, rank, svals = np.linalg.lstsq(A1, b, rcond=None)

        E0 = float(x[0])
        J_res = -x[1:]  # eV

        # Diagnostics
        e_lower = float(np.min(E))
        f_lower = f_min if f_min != float('inf') else pd.NA

        row_dict = {
            "Config": int(cfg),
            "E_lower": e_lower,
            "F_lower": f_lower,
            "Residual": float(np.sqrt(residuals[0])) if residuals.size else 0.0,
            "_E0": E0,  # optional; you can drop this column later if you don't want it
            "_rank": int(rank),
        }

        for j_name, j_val in zip(coupling_names, J_res):
            row_dict[j_name] = float(j_val)

        rows.append(row_dict)
        print(f"Config {cfg}: J_res (eV) = {J_res}, resid={row_dict['Residual']:.6f} rank={rank}")

    return pd.DataFrame(rows).sort_values("Config").reset_index(drop=True)


# ======================================================================
# MAIN
# ======================================================================
df = pd.read_csv(INFILE)

# split Folder into numeric Config and a Spin-init string
tmp = df["Folder"].str.split("_", n=2, expand=True)
df["Config"]    = pd.to_numeric(tmp[1], errors="coerce")
df["Spin init"] = tmp[2]

# Process based on SPIN_NR mode
if SPIN_NR == 2:
    out = process_2spin(df)
    outfile = OUTFILE_2SPIN
    table_header = r"""
\begin{longtable}{@{}crrr@{}}
\caption{Exchange coupling summary (2-spin) per dimer configuration.}
\label{tab:exchange_couplings_2spin}\\
\toprule
Config & $E_{\mathrm{lower}}$ (eV) & $J$ (meV) & $F_{\mathrm{lower}}$ (eV\,\AA$^{-1}$)\\
\midrule
\endfirsthead
\multicolumn{4}{c}{{\tablename}~\thetable{} --- continued from previous page}\\
\toprule
Config & $E_{\mathrm{lower}}$ (eV) & $J$ (meV) & $F_{\mathrm{lower}}$ (eV\,\AA$^{-1}$)\\
\midrule
\endhead
\midrule
\multicolumn{4}{r}{\emph{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
"""

elif SPIN_NR == 4:
    out = process_4spin(df)
    outfile = OUTFILE_4SPIN
    table_header = r"""
\begin{longtable}{@{}crrrrrrrrr@{}}
\caption{Exchange couplings (4-spin least-squares fit, rescaled: $E=E_0-\sum J_{\mathrm{res}} s_i s_j$; true Heisenberg $J=4J_{\mathrm{res}}$ for $S=\tfrac12$) per dimer configuration.}
\label{tab:exchange_couplings_4spin}\\
\toprule
Config & $E_{\mathrm{lower}}$ (eV) & $J_{12}$ (meV) & $J_{13}$ (meV) & $J_{14}$ (meV) & $J_{23}$ (meV) & $J_{24}$ (meV) & $J_{34}$ (meV) & $F_{\mathrm{lower}}$ (eV\,\AA$^{-1}$) & Residual (meV)\\
\midrule
\endfirsthead
\multicolumn{10}{c}{{\tablename}~\thetable{} --- continued from previous page}\\
\toprule
Config & $E_{\mathrm{lower}}$ (eV) & $J_{12}$ (meV) & $J_{13}$ (meV) & $J_{14}$ (meV) & $J_{23}$ (meV) & $J_{24}$ (meV) & $J_{34}$ (meV) & $F_{\mathrm{lower}}$ (eV\,\AA$^{-1}$) & Residual (meV)\\
\midrule
\endhead
\midrule
\multicolumn{10}{r}{\emph{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
"""

else:
    raise ValueError(f"SPIN_NR must be 2 or 4, got {SPIN_NR}")

# Write output
os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

with open(outfile, "w", encoding="utf-8") as tex:
    tex.write(table_header)

    if SPIN_NR == 2:
        max_abs_j = out["J"].abs().max() if not out["J"].isna().all() else 1.0

        for _, r in out.iterrows():
            cfg = int(r["Config"])
            e_low = f"{r['E_lower']:.4f}" if pd.notna(r["E_lower"]) else r"--"
            j = format_j_value(r["J"], max_abs_j, colored=COLORED_LATEX)
            f_low = f"{r['F_lower']:.3f}" if pd.notna(r["F_lower"]) else r"--"
            tex.write(f"{cfg} & {e_low} & {j} & {f_low} \\\\\n")

    elif SPIN_NR == 4:
        j_cols = ['J_12', 'J_13', 'J_14', 'J_23', 'J_24', 'J_34']
        max_abs_j = out[j_cols].abs().max().max() if not out[j_cols].isna().all().all() else 1.0

        for _, r in out.iterrows():
            cfg = int(r["Config"])
            e_low = f"{r['E_lower']:.4f}" if pd.notna(r["E_lower"]) else r"--"
            j12 = format_j_value(r['J_12'], max_abs_j, colored=COLORED_LATEX)
            j13 = format_j_value(r['J_13'], max_abs_j, colored=COLORED_LATEX)
            j14 = format_j_value(r['J_14'], max_abs_j, colored=COLORED_LATEX)
            j23 = format_j_value(r['J_23'], max_abs_j, colored=COLORED_LATEX)
            j24 = format_j_value(r['J_24'], max_abs_j, colored=COLORED_LATEX)
            j34 = format_j_value(r['J_34'], max_abs_j, colored=COLORED_LATEX)
            f_low = f"{r['F_lower']:.3f}" if pd.notna(r["F_lower"]) else r"--"
            resid = f"{r['Residual']*1000:.3f}" if pd.notna(r["Residual"]) else r"--"
            tex.write(f"{cfg} & {e_low} & {j12} & {j13} & {j14} & {j23} & {j24} & {j34} & {f_low} & {resid} \\\\\n")

    tex.write(r"\end{longtable}")

print(f"✔  LaTeX written to {outfile}")
