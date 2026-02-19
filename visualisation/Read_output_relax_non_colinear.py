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

# Capture the whole Mulliken table body (robust to both collinear and non-collinear)
# Match lines starting with whitespace and a digit, stopping at a blank line or end
mulliken_block = re.compile(
    r"Mulliken Atomic Populations:\s*\n"
    r"Atom\s+#[^\n]*\n"
    r"(?P<body>(?:\s+\d+[^\n]*\n)+)",
    re.MULTILINE
)

# Non-collinear line: atom charge valence S Sx Sy Sz Species
_noncol_line = re.compile(
    r"^\s*(?P<atom>\d+)\s+"
    r"(?P<charge>[-\d.Ee+]+)\s+"
    r"(?P<valence>[-\d.Ee+]+)\s+"
    r"(?P<S>[-\d.Ee+]+)\s+"
    r"(?P<Sx>[-\d.Ee+]+)\s+"
    r"(?P<Sy>[-\d.Ee+]+)\s+"
    r"(?P<Sz>[-\d.Ee+]+)\s+"
    r"(?P<species>\S+)\s*$",
    re.MULTILINE
)

# Collinear: atom charge valence Sz Species (exactly 5 columns)
_col_line = re.compile(
    r"^\s*(?P<atom>\d+)\s+"
    r"(?P<charge>[-\d.Ee+]+)\s+"
    r"(?P<valence>[-\d.Ee+]+)\s+"
    r"(?P<sz>[-\d.Ee+]+)\s+"
    r"(?P<species>\S+)\s*$",
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
            return {"Sx": [], "Sy": [], "Sz": []}, {"Sx": [], "Sy": [], "Sz": []}

        body = m.group("body")
        has_noncol = _noncol_line.search(body) is not None

        all_spins = {"Sx": [], "Sy": [], "Sz": []}
        selection = {"Sx": [], "Sy": [], "Sz": []}

        if has_noncol:
            for lm in _noncol_line.finditer(body):
                atom = int(lm.group("atom"))
                sx = float(lm.group("Sx"))
                sy = float(lm.group("Sy"))
                sz = float(lm.group("Sz"))
                all_spins["Sx"].append((atom, sx))
                all_spins["Sy"].append((atom, sy))
                all_spins["Sz"].append((atom, sz))
                if abs(sx) > 0.05:
                    selection["Sx"].append((atom, sx))
                if abs(sy) > 0.05:
                    selection["Sy"].append((atom, sy))
                if abs(sz) > 0.05:
                    selection["Sz"].append((atom, sz))
            return all_spins, selection

        # Collinear: only Sz meaningful; Sx/Sy -> 0
        for lm in _col_line.finditer(body):
            atom = int(lm.group("atom"))
            sz = float(lm.group("sz"))
            all_spins["Sx"].append((atom, 0.0))
            all_spins["Sy"].append((atom, 0.0))
            all_spins["Sz"].append((atom, sz))
            if abs(sz) > 0.05:
                selection["Sz"].append((atom, sz))
        return all_spins, selection

    spin_init_all, spin_init_selection   = pick_spins(init_txt)
    spin_final_all, spin_final_selection = pick_spins(final_txt)

    # Calculate total spin magnitude considering all components
    Stot_x = sum(sx for _, sx in spin_final_all["Sx"])
    Stot_y = sum(sy for _, sy in spin_final_all["Sy"])
    Stot_z = sum(sz for _, sz in spin_final_all["Sz"])
    Stot = sum(
        (sx**2 + sy**2 + sz**2)**0.5
        for (_, sx), (_, sy), (_, sz) in zip(
            spin_final_all["Sx"],
            spin_final_all["Sy"],
            spin_final_all["Sz"]
        )
    )
    unpaired_spins = Stot / 1.75  # heuristic

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
        "E_init":         e_init,
        "E_final":        e_last,
        "F_init":         f_init,
        "F_final":        f_last,
        "DM_steps":       dm_steps,
        "Stot_x":           Stot_x,
        "Stot_y":           Stot_y,
        "Stot_z":           Stot_z,
        "Unpaired_spins": unpaired_spins,
        "Spin_init_all_Sx":  spin_init_all["Sx"],
        "Spin_init_all_Sy":  spin_init_all["Sy"],
        "Spin_init_all_Sz":  spin_init_all["Sz"],
        "Spin_final_all_Sx": spin_final_all["Sx"],
        "Spin_final_all_Sy": spin_final_all["Sy"],
        "Spin_final_all_Sz": spin_final_all["Sz"],
        "Spin_init_Sx":      spin_init_selection["Sx"],
        "Spin_init_Sy":      spin_init_selection["Sy"],
        "Spin_init_Sz":      spin_init_selection["Sz"],
        "Spin_final_Sx":     spin_final_selection["Sx"],
        "Spin_final_Sy":     spin_final_selection["Sy"],
        "Spin_final_Sz":     spin_final_selection["Sz"],
    }

def save_summary(rows, path):
    df = pd.DataFrame(rows, columns=[
        "Folder","E_init","E_final","F_init","F_final",
        "DM_steps","Stot_x","Stot_y","Stot_z","Unpaired_spins",
        "Spin_final_Sx_list","Spin_final_Sy_list","Spin_final_Sz_list",
        "Spin_init_Sx_list","Spin_init_Sy_list","Spin_init_Sz_list"
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")

def save_all_spins(rows, path):
    # we only need Folder and Spin_final_all components
    df = pd.DataFrame(rows, columns=[
        "Folder",
        "Spin_final_all_Sx",
        "Spin_final_all_Sy",
        "Spin_final_all_Sz"
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")

# ─── DIRECTORY WALK ────────────────────────────────────────────────────────────
dir_runs = "/Users/riccardo/Documenti/TuDelft/phd/Analysis_Code/DFT/Short-2OS/runs/"
run      = "dimers_run2/non_colinear_fast2"  # Change this to your specific run folder
base     = Path(dir_runs) / run

# Check if directory exists
if not base.exists():
    print(f"[ERROR] Directory not found: {base}")
    exit(1)

summary_rows = []

# Get list of subdirectories first to avoid issues
subdirs = [sub for sub in sorted(base.iterdir()) if sub.is_dir()]
print(f"Found {len(subdirs)} subdirectories to process")

for sub in subdirs:
    out = sub / "siesta.out"
    if not out.is_file():
        print(f"[SKIP] No siesta.out in {sub.name}")
        continue

    print(f"[PROCESSING] {sub.name}...", flush=True)
    try:
        data = parse_siesta(out)
    except Exception as e:
        print(f"[ERROR] Failed to parse {sub.name}: {e}")
        continue
    
    data["Folder"] = sub.name
    def fmt_spin_list(items):
        return "; ".join(f"{i}:{s:.3f}" for i, s in items)

    data["Spin_init_Sx_list"]  = fmt_spin_list(data["Spin_init_Sx"])
    data["Spin_init_Sy_list"]  = fmt_spin_list(data["Spin_init_Sy"])
    data["Spin_init_Sz_list"]  = fmt_spin_list(data["Spin_init_Sz"])
    data["Spin_final_Sx_list"] = fmt_spin_list(data["Spin_final_Sx"])
    data["Spin_final_Sy_list"] = fmt_spin_list(data["Spin_final_Sy"])
    data["Spin_final_Sz_list"] = fmt_spin_list(data["Spin_final_Sz"])
    data["Spin_final_all_Sx"]  = fmt_spin_list(data["Spin_final_all_Sx"])
    data["Spin_final_all_Sy"]  = fmt_spin_list(data["Spin_final_all_Sy"])
    data["Spin_final_all_Sz"]  = fmt_spin_list(data["Spin_final_all_Sz"])
    summary_rows.append(data)
    print(f"[OK] {sub.name}", flush=True)

# ─── SAVE BOTH CSVs ────────────────────────────────────────────────────────────
print(f"\n[SAVING] {len(summary_rows)} results to CSV files...", flush=True)
save_summary(
    summary_rows,
    Path(f"visualisation/{run}/summary_results.csv")
)
save_all_spins(
    summary_rows,
    Path(f"visualisation/{run}/all_spins.csv")
)
print("[COMPLETE] Script finished successfully!", flush=True)
