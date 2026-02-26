r"""
build_dimer_table.py  —  generate LaTeX longtable from dimer CSV
---------------------------------------------------------------

• Reads:   dimer_results.csv   (comma-separated, headers on first line)
• Writes:  dimer_fast_table.tex  (ready to \input{} in LaTeX)

Preamble needed in your .tex:
    \usepackage{booktabs,longtable,tabularx}
    \usepackage[table]{xcolor}   % for  \rowcolor
"""

import pandas as pd
import re
import os

run_label = "junction_SM"  # subfolder in Short-2OS/runs/ and visualisation/
fast_slow = "conf_4"    # subfolder for fast or slow screening
INFILE  = f"visualisation/{run_label}/{fast_slow}/summary_results.csv"
OUTFILE = f"visualisation/latex_tables/{run_label}/{fast_slow}/{run_label}_latex.tex"

# ----------------------------------------------------------------------
# helper: strip “index:value” → “value”  and return an en-dash if empty
# ----------------------------------------------------------------------
def clean_spin_list(raw: str) -> str:
    if pd.isna(raw) or raw.strip() == "":
        return r"\textemdash"
    vals = [seg.split(":")[-1].strip() for seg in raw.split(";") if seg.strip()]
    return ";\\,".join(vals)              # thin space after each semicolon

# ----------------------------------------------------------------------
# 1.  read CSV  ➜  tidy / augment columns
# ----------------------------------------------------------------------
df = pd.read_csv(INFILE)

# split Folder into numeric Config and a Spin-init string
tmp = df["Folder"].str.split("_", n=2, expand=True)
df["Config"]    = pd.to_numeric(tmp[1], errors="coerce")
df["Spin init"] = tmp[2]

# keep only rows that have a numeric E_final
df["E_final"] = pd.to_numeric(df["E_final"], errors="coerce")
df = df.dropna(subset=["E_final"])

# clean numeric columns
df["F_final"]  = pd.to_numeric(df["F_final"],  errors="coerce")
df["DM_steps"] = pd.to_numeric(df["DM_steps"], errors="coerce", downcast="integer")

# # prepare the cleaned spin list
# df["Spin_clean"] = df["Spin_final_list"].apply(clean_spin_list)
df["S_tot"] = pd.to_numeric(df["Stot"], errors="coerce")
df["Unpaired_spins"] = pd.to_numeric(df["Unpaired_spins"], errors="coerce")

# mark the lowest-energy row per configuration
df["highlight"] = ""
for idx in df.groupby("Config")["E_final"].idxmin():
    df.loc[idx, "highlight"] = r"\rowcolor{gray!20} "

# sort by Config (numeric) then E_final
df = df.sort_values(["Config", "E_final"]).reset_index(drop=True)

# ----------------------------------------------------------------------
# 2.  ensure output directory exists
# ----------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)

# ----------------------------------------------------------------------
# 3.  write LaTeX longtable with an X column
# ----------------------------------------------------------------------
with open(OUTFILE, "w", encoding="utf-8") as tex:
    tex.write(r"""
\begin{longtable}{@{}ccrrrrr@{}}
\caption{Fast–screening DFT results for all dimer configurations.
         For each numeric \textbf{Config} the spin initialisation with the
         lowest $E_{\mathrm{final}}$ is highlighted in grey.}\label{tab:dimer_full}\\
\toprule
Config & Spin init & $E_{\mathrm{final}}$ (eV) &
$F_{\mathrm{final}}$ (eV\,\AA$^{-1}$) & DM steps & S_{tot} & Unpaired Electrons\\
\midrule
\endfirsthead
\multicolumn{6}{c}{{\tablename}~\thetable{} — continued from previous page}\\
\toprule
Config & Spin init & $E_{\mathrm{final}}$ (eV) &
$F_{\mathrm{final}}$ (eV\,\AA$^{-1}$) & DM steps & S_{tot} & Unpaired Electrons\\
\midrule
\endhead
\midrule
\multicolumn{6}{r}{\emph{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
""")

    for _, r in df.iterrows():
        cfg   = int(r["Config"])
        init  = r["Spin init"].replace("_", r"\_")  # escape underscores
        efin  = f"{r['E_final']:.4f}"
        ffin  = f"{r['F_final']:.3f}" if pd.notna(r["F_final"]) else r"--"
        steps = int(r["DM_steps"]) if pd.notna(r["DM_steps"]) else r"--"
        stot = f"{r['S_tot']:.3f}" if pd.notna(r["F_final"]) else r"--"
        unpspins = f"{r['Unpaired_spins']:.3f}" if pd.notna(r["Unpaired_spins"]) else r"--"
        tex.write(f"{r['highlight']}{cfg} & {init} & {efin} & {ffin} & {steps} & {stot} & {unpspins} \\\\\n")

    tex.write(r"\end{longtable}")

print(f"✔  LaTeX written to {OUTFILE}")