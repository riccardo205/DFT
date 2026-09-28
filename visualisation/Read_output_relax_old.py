import re
import pandas as pd
from pathlib import Path
import os
import warnings

# ─── REGEX PATTERNS ────────────────────────────────────────────────────────────
freeeng_pattern = re.compile(
    r"^\s*scf:\s*\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+([-+]?\d+\.\d+)",
    re.MULTILINE
)
force_pattern  = re.compile(r"^\s*Max\s+([0-9]+\.[0-9]+)", re.MULTILINE)
dm_pattern     = re.compile(r"new_DM\s+--\s*step:")
mulliken_block = re.compile(
    r"Mulliken Atomic Populations:\s*Atom\s+#.*?\n"
    r"((?:\s*\d+\s+\S+\s+\S+\s+[-\d.]+\s+\S+.*\n)+)",
    re.MULTILINE
)
qtot_pattern = re.compile(r"mulliken:\s*Qtot\s*=\s*([-+]?\d+\.\d+)", re.IGNORECASE)

def parse_siesta(path: Path):
    txt = path.read_text()

    # Energies
    all_fe = [float(m.group(1)) for m in freeeng_pattern.finditer(txt)]
    e_init = all_fe[0] if all_fe else None
    e_last = all_fe[-1] if all_fe else None

    # Forces
    all_f = [float(m.group(1)) for m in force_pattern.finditer(txt)]
    f_init = all_f[0] if all_f else None
    f_last = all_f[-1] if all_f else None

    # DM steps
    dm_steps = len(dm_pattern.findall(txt))

    # Split roughly in half for init vs final
    half = len(txt)//2
    init_txt, final_txt = txt[:half], txt[half:]

    def pick_spins(block):
        m = mulliken_block.search(block)
        if not m:
            return [],[]
        all = []
        selection = []
        for line in m.group(1).strip().splitlines():
            cols = line.split()
            atom = int(cols[0])
            sz   = float(cols[3])
            all.append((atom, sz))
            if abs(sz) > 0.05:
                selection.append((atom, sz))
        return all, selection

    spin_init_all, spin_init_selection  = pick_spins(init_txt)
    spin_final_all, spin_final_selection = pick_spins(final_txt)

    Stot = sum(sz for _, sz in spin_final_all)
    unpaired_spins = sum(abs(sz) for _, sz in spin_final_all)/1.75 # N(1+0.75) including the 0.75 due to fluctuations 

    # Qtot imbalance warning
    qtots = [float(m.group(1)) for m in qtot_pattern.finditer(txt)]
    for i in range(0, len(qtots)-1, 2):
        diff = abs(qtots[i] - qtots[i+1])
        if diff > 2.0:
            warnings.warn(
                f"Large Qtot imbalance |{qtots[i]}-{qtots[i+1]}| = {diff:.3f}"
                f" at pair #{i//2+1}"
            )

    return {
        "E_init":        e_init,
        "E_final":      e_last,
        "F_init":        f_init,
        "F_final":      f_last,
        "DM_steps":     dm_steps,
        "Stot":          Stot,
        "Unpaired_spins": unpaired_spins,
        "Spin_init_all":   spin_init_all,
        "Spin_final_all": spin_final_all,
        "Spin_init":   spin_init_selection,
        "Spin_final":   spin_final_selection,
    }

def save_summary(rows, path):
    df = pd.DataFrame(rows, columns=[
        "Folder","E_init","E_final","F_init","F_final",
        "DM_steps","Stot","Unpaired_spins","Spin_final_list","Spin_init_list"
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")

def save_all_spins(rows, path):
    # we only need Folder and Spin_final_list_all
    df = pd.DataFrame(rows, columns=["Folder","Spin_final_all"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")

# ─── DIRECTORY WALK ────────────────────────────────────────────────────────────
dir_runs = "/Users/riccardo/Documenti/TuDelft/phd/Analysis_Code/DFT/Short-2OS/runs/"
run      = "dimers_run2/slow_lowT_lowest_forces"  # Change this to your specific run folder
base     = Path(dir_runs) / run

summary_rows = []
# allspin_rows = []

for sub in sorted(base.iterdir()):
    out = sub / "siesta.out"
    if not out.is_file():
        continue

    data = parse_siesta(out)
    data["Folder"] = sub.name
    data["Spin_init_list"]  = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_init"])
    data["Spin_final_list"] = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_final"])
    data["Spin_final_all"] = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_final_all"])
    summary_rows.append(data)  
     
    # # summary (|Sz|>0.15)
    # data = parse_siesta(out, all_spins=False)
    # data["Folder"] = sub.name
    # data["Spin_init_list"]  = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_init"])
    # data["Spin_final_list"] = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_final"])
    # summary_rows.append(data)

    # # all spins (no filter)
    # data_all = parse_siesta(out, all_spins=True)
    # data_all["Folder"] = sub.name
    # data_all["Spin_final_list"] = "; ".join(f"{i}:{sz:.3f}" for i, sz in data_all["Spin_final"])
    # allspin_rows.append(data_all)

# ─── SAVE BOTH CSVs ────────────────────────────────────────────────────────────
save_summary(
    summary_rows,
    Path(f"visualisation/output/{run}/summary_results.csv")
)
save_all_spins(
    summary_rows,
    Path(f"visualisation/output/{run}/all_spins.csv")
)
