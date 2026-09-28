import re
import pandas as pd
from pathlib import Path
import warnings

# =========================
# USER SETTINGS
# =========================
MAX_FORCE_ACCEPTED_FOR_MIN_FREEENG = 0.1  # eV/Ang (change as you like)

# =========================
# REGEX PATTERNS
# =========================
# Matches lines like:
# scf:   17   Eharris   E_KS   FreeEng  ...
scf_line_re = re.compile(
    r"^\s*scf:\s*\d+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)",
    re.MULTILINE
)

# Geometry move headers (CG)
cg_move_re = re.compile(
    r"^\s*Begin\s+CG\s+opt\.\s+move\s*=\s*(\d+)",
    re.MULTILINE
)

# Force summary: we want the first pure "Max    <num>" line (not the "... constrained" one)
force_max_re = re.compile(r"^\s*Max\s+([0-9]+\.[0-9]+)\s*$", re.MULTILINE)

dm_pattern = re.compile(r"new_DM\s+--\s*step:")

mulliken_block_re = re.compile(
    r"Mulliken Atomic Populations:\s*Atom\s+#.*?\n"
    r"((?:\s*\d+\s+\S+\s+\S+\s+[-\d.]+\s+\S+.*\n)+)",
    re.MULTILINE
)
qtot_pattern = re.compile(r"mulliken:\s*Qtot\s*=\s*([-+]?\d+\.\d+)", re.IGNORECASE)


def _extract_moves(txt: str):
    """
    Return list of (move_index, segment_text) for CG moves.
    Segment is from this move header up to the next move header (or EOF).
    """
    hits = list(cg_move_re.finditer(txt))
    if not hits:
        return []

    moves = []
    for i, m in enumerate(hits):
        move_idx = int(m.group(1))
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(txt)
        moves.append((move_idx, txt[start:end]))
    return moves


def _parse_spin_block(block_text: str, threshold: float = 0.05):
    """
    Parse a Mulliken Atomic Populations block and return:
      all_spins = [(atom_index, sz), ...]
      selection = [(atom_index, sz), ...] where |sz|>threshold
    """
    m = mulliken_block_re.search(block_text)
    if not m:
        return [], []

    all_spins = []
    selection = []
    for line in m.group(1).strip().splitlines():
        cols = line.split()
        # Your previous code assumed:
        # cols[0] = atom index
        # cols[3] = Sz
        # Keep it, but protect against unexpected formatting:
        try:
            atom = int(cols[0])
            sz = float(cols[3])
        except Exception:
            continue
        all_spins.append((atom, sz))
        if abs(sz) > threshold:
            selection.append((atom, sz))

    return all_spins, selection


def parse_siesta(path: Path):
    txt = path.read_text(errors="ignore")

    # DM steps (global count)
    dm_steps = len(dm_pattern.findall(txt))

    # ---- Build per-move records for completed moves only ----
    records = []
    for move_idx, seg in _extract_moves(txt):
        conv_pos = seg.find("SCF cycle converged")
        if conv_pos == -1:
            # interrupted before SCF convergence -> ignore this move
            continue

        # Last scf line BEFORE convergence is the "end-of-cycle" energy
        scfs = list(scf_line_re.finditer(seg[:conv_pos]))
        if not scfs:
            continue
        last_scf = scfs[-1]
        eharris = float(last_scf.group(1))
        e_ks = float(last_scf.group(2))
        freeeng = float(last_scf.group(3))

        # Forces must be present after SCF
        forces_pos = seg.find("siesta: Atomic forces")
        if forces_pos == -1:
            # interrupted after SCF but before forces printed -> ignore
            continue

        fm = force_max_re.search(seg[forces_pos:])
        if not fm:
            # no Max line -> ignore
            continue
        fmax = float(fm.group(1))

        records.append({
            "Move": move_idx,
            "Eharris": eharris,
            "E_KS": e_ks,
            "FreeEng": freeeng,
            "Fmax": fmax,
        })

    if not records:
        return None  # nothing safe to parse

    # Init/final from completed moves
    e_init = records[0]["FreeEng"]
    f_init = records[0]["Fmax"]


    # Picks
    min_force_rec = min(records, key=lambda r: r["Fmax"])

    # Min free energy among "acceptable force" steps
    acceptable = [r for r in records if r["Fmax"] <= MAX_FORCE_ACCEPTED_FOR_MIN_FREEENG]
    min_free_rec = None
    if acceptable:
        min_free_rec = min(acceptable, key=lambda r: r["FreeEng"])
    else:
        warnings.warn(
            f"{path.parent.name}: no steps with Fmax <= {MAX_FORCE_ACCEPTED_FOR_MIN_FREEENG} eV/Ang "
            f"-> skipping min_freeeng row."
        )

    # ---- Mulliken / spins: first and last block in whole output ----
    mull_hits = list(mulliken_block_re.finditer(txt))
    if mull_hits:
        first_block = mull_hits[0].group(0)
        last_block = mull_hits[-1].group(0)
    else:
        first_block = ""
        last_block = ""

    spin_init_all, spin_init_sel = _parse_spin_block(first_block, threshold=0.05)
    spin_final_all, spin_final_sel = _parse_spin_block(last_block, threshold=0.05)

    Stot = sum(sz for _, sz in spin_final_all)
    unpaired_spins = sum(abs(sz) for _, sz in spin_final_all) / 1.75  # your heuristic

    # Qtot imbalance warning (kept)
    qtots = [float(m.group(1)) for m in qtot_pattern.finditer(txt)]
    for i in range(0, len(qtots) - 1, 2):
        diff = abs(qtots[i] - qtots[i + 1])
        if diff > 2.0:
            warnings.warn(
                f"{path.parent.name}: Large Qtot imbalance |{qtots[i]}-{qtots[i+1]}| = {diff:.3f} "
                f"at pair #{i//2+1}"
            )

    return {
        "records": records,
        "E_init": e_init,
        "F_init": f_init,
        "DM_steps": dm_steps,
        "Stot": Stot,
        "Unpaired_spins": unpaired_spins,
        "Spin_init": spin_init_sel,
        "Spin_final": spin_final_sel,
        "Spin_final_all": spin_final_all,
        "min_force": min_force_rec,
        "min_free": min_free_rec,
    }


def save_summary(rows, path: Path):
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")


def save_all_spins(rows, path: Path):
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"→ {path} written")


# =========================
# DIRECTORY WALK
# =========================
dir_runs = "/Users/riccardo/Documenti/TuDelft/phd/Analysis_Code/DFT/Short-2OS/runs/"
run = "junction_SM2/slow_2026_03_13_vs2"
base = Path(dir_runs) / run

summary_rows = []
allspin_rows = []

for sub in sorted(base.iterdir()):
    out = sub / "siesta.out"
    if not out.is_file():
        continue

    data = parse_siesta(out)
    if data is None:
        continue

    spin_init_list = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_init"])
    spin_final_list = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_final"])
    spin_final_all = "; ".join(f"{i}:{sz:.3f}" for i, sz in data["Spin_final_all"])

    common = {
        "Folder": sub.name,
        "E_init": data["E_init"],
        "F_init": data["F_init"],
        "E_final": None,
        "F_final": None, 
        "DM_step": data["DM_steps"],
        "Stot": data["Stot"],
        "Unpaired_spins": data["Unpaired_spins"],
        "Spin_init_list": spin_init_list,
        "Spin_final_list": spin_final_list,
    }

    # Row 1: minimum force configuration
    r1 = dict(common)
    r1.update({
        "Pick": "min_force",
        "DM_step": data["min_force"]["Move"],
        "E_final": data["min_force"]["FreeEng"],
        "E_KS": data["min_force"]["E_KS"],
        "F_final": data["min_force"]["Fmax"],
    })
    print(f"Parsed {sub.name}: min_force Fmax = {data['min_force']['Fmax']:.3f} eV/Ang")
    summary_rows.append(r1)

    # Row 2: minimum free-energy configuration (subject to force threshold)
    if data["min_free"] is not None:
        r2 = dict(common)
        r2.update({
            "Pick": "min_freeeng",
            "DM_step": data["min_free"]["Move"],
            "E_final": data["min_free"]["FreeEng"],
            "E_KS": data["min_free"]["E_KS"],
            "F_final": data["min_free"]["Fmax"],
            "ForceThresholdUsed": MAX_FORCE_ACCEPTED_FOR_MIN_FREEENG,
        })
        summary_rows.append(r2)

    # all_spins should be ONE row per folder (not duplicated)
    allspin_rows.append({
        "Folder": sub.name,
        "Spin_final_all": spin_final_all
    })

# Save outputs
save_summary(summary_rows, Path(f"visualisation/output/{run}/summary_results.csv"))
save_all_spins(allspin_rows, Path(f"visualisation/output/{run}/all_spins.csv"))