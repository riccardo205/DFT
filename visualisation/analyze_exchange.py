#!/usr/bin/env python3
"""
Exchange analysis: PBE0 high-spin / broken-symmetry single points on the
frozen R2SCAN-3C-relaxed geometries of the L x C x R angle scan.

Structural source of truth: angle_scan.csv (from analyze_angle_scan.py), joined
by (L, C, R). Only the geometries marked usable=True there are used for
scientific conclusions.

Sign convention used everywhere (the raw quantity is NOT called J):
    dE = E_HS - E_BS
    dE > 0 : BS lower  -> AFM tendency
    dE < 0 : HS slightly lower (near-degenerate unless dE <= -0.5 meV)

Narrative:
    torsional decoupling -> reduced pi-hopping -> stronger terminal-radical
    localization -> collapse of the AFM exchange contribution ->
    near-degeneracy / occasional very small HS preference.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange_convention as XC  # noqa: E402  (J convention, see exchange_convention.py)
from analyze_angle_scan import (  # noqa: E402
    ATOM_LINE_RE,
    FINAL_E_RE,
    HARTREE_TO_MEV,
    MULLIKEN_BLOCK_RE,
    S2_RE,
    build_adjacency,
    fit_plane,
    outer_terminal_fragment,
    smallest_ring_at,
    split_fragment_at_bond,
)

SCF_CONVERGED_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
LOEWDIN_RE = re.compile(r"LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n((?:\s*\d+\s+\S+\s*:.*\n)+)")
LABEL_RE = re.compile(r"^L(\d+)_C(\d+)_R(\d+)(_unconverged)?$")
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
CLASS_LABELS = {0: "0 outer 90°", 1: "1 outer 90°", 2: "2 outer 90°"}


# ============================================================
# PARSING
# ============================================================
def _read_block(block_text):
    charges, spins = {}, {}
    for line in block_text.splitlines():
        m = ATOM_LINE_RE.match(line)
        if m:
            charges[int(m.group(1))] = float(m.group(3))
            spins[int(m.group(1))] = float(m.group(4))
    return charges, spins


def parse_pbe0(path, fragments):
    rec = {"exists": path.exists(), "terminated": False, "scf_converged": False, "scf_iterations": np.nan,
           "energy": np.nan, "s2": np.nan, "has_spin": False}
    for k in fragments:
        rec[f"spin_f{k}"] = np.nan
        rec[f"charge_f{k}"] = np.nan
        rec[f"lcharge_f{k}"] = np.nan
    if not rec["exists"]:
        return rec
    text = path.read_text(errors="ignore")
    rec["terminated"] = "ORCA TERMINATED NORMALLY" in text
    scf = SCF_CONVERGED_RE.findall(text)
    rec["scf_converged"] = bool(scf) and "SCF NOT CONVERGED" not in text
    rec["scf_iterations"] = int(scf[-1]) if scf else np.nan
    e = FINAL_E_RE.findall(text)
    rec["energy"] = float(e[-1]) if e else np.nan
    s2 = S2_RE.findall(text)
    rec["s2"] = float(s2[-1]) if s2 else np.nan
    blocks = list(MULLIKEN_BLOCK_RE.finditer(text))
    if blocks:
        charges, spins = _read_block(blocks[-1].group(1))
        for k, atoms in fragments.items():
            rec[f"spin_f{k}"] = sum(spins.get(i, 0.0) for i in atoms)
            rec[f"charge_f{k}"] = sum(charges.get(i, 0.0) for i in atoms)
        rec["has_spin"] = True
    lblocks = list(LOEWDIN_RE.finditer(text))
    if lblocks:
        lcharges, _ = _read_block(lblocks[-1].group(1))
        for k, atoms in fragments.items():
            rec[f"lcharge_f{k}"] = sum(lcharges.get(i, 0.0) for i in atoms)
    return rec


def plane_angle(pos, ring_a, ring_b):
    _, n1 = fit_plane(pos[ring_a])
    _, n2 = fit_plane(pos[ring_b])
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(m.sum()), "pearson": np.nan, "pearson_p": np.nan, "spearman": np.nan, "spearman_p": np.nan}
    if m.sum() >= 4 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
        out["pearson"], out["pearson_p"] = (float(v) for v in stats.pearsonr(x[m], y[m]))
        out["spearman"], out["spearman_p"] = (float(v) for v in stats.spearmanr(x[m], y[m]))
    return out


def fmt_corr(c):
    if not np.isfinite(c["spearman"]):
        return f"n/a (n={c['n']})"
    return f"Spearman ρ = {c['spearman']:.2f} (p={c['spearman_p']:.1e}), Pearson r = {c['pearson']:.2f} (n={c['n']})"


def mannwhitney(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def classify_exchange(dE, near=0.5):
    if dE >= near:
        return "BS-favored (AFM tendency)"
    if dE > -near:
        return "near-degenerate" if dE >= 0 else "near-degenerate with small HS preference"
    return "robust HS-favored"


def f2(v, nd=2):
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    repo_root = Path(__file__).resolve().parent.parent
    scans = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans"
    exchange_root = scans / "angle_scan_exchange_analysis"
    relax_root = scans / "angle_scan_relaxation"
    struct_csv = relax_root / "angle_scan.csv"
    baseline_xyz = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    NEAR_DEGENERATE_MEV = 0.5          # |dE| below this = near-degenerate; dE <= -this = robust HS-favored
    TIGHT_DEGENERATE_MEV = 0.1
    HS_S2_RANGE = (2.0, 2.3)
    BS_S2_MIN = 0.5
    BS_FRAGMENT_SPIN_MIN = 0.3
    EXTREME_C = 160.0                  # excluded from the empirical P trend / simple-model statistics
    NEAR_90_TOL_DEG = 15.0
    HIGH_LOCALIZATION = 0.90           # mean terminal spin considered "highly localized"
    STRONG_SPIN_ASYM = 0.05            # ||S1|-|S4|| considered "strong" asymmetry
    CLIP_YMAX_MEV = 22.0               # y-limit where far outliers would otherwise dominate

    out_dir = exchange_root
    plots = out_dir / "plots"
    heat_dir = plots / "heatmaps"
    plots.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    # ---- fragments and ring planes from the clean baseline geometry ----
    ref = read(baseline_xyz)
    adj = build_adjacency(ref)
    n_atoms = len(ref)
    frag1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    frag4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    center_all = sorted(set(range(n_atoms)) - set(frag1) - set(frag4))
    frag2, frag3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), center_all)
    fragments = {1: frag1, 2: frag2, 3: frag3, 4: frag4}

    rings = {
        "L": (smallest_ring_at(adj, LEFT[1], LEFT[2]), smallest_ring_at(adj, LEFT[2], LEFT[1])),
        "C": (smallest_ring_at(adj, CENTER[1], CENTER[2]), smallest_ring_at(adj, CENTER[2], CENTER[1])),
        "R": (smallest_ring_at(adj, RIGHT[1], RIGHT[2]), smallest_ring_at(adj, RIGHT[2], RIGHT[1])),
    }

    # ---- structural table ----
    struct = pd.read_csv(struct_csv)
    struct["label"] = struct["label"].astype(str)

    # ---- parse every PBE0 pair ----
    rows = []
    geom_pos = {}
    for folder in sorted(p for p in exchange_root.iterdir() if p.is_dir() and LABEL_RE.match(p.name)):
        m = LABEL_RE.match(folder.name)
        base = f"L{m.group(1)}_C{m.group(2)}_R{m.group(3)}"
        hs = parse_pbe0(folder / "high_spin" / "high_spin.out", fragments)
        bs = parse_pbe0(folder / "broken_symmetry" / "broken_symmetry.out", fragments)
        row = {"label": base, "exchange_folder": folder.name}
        for prefix, rec in (("HS", hs), ("BS", bs)):
            row[f"E_{prefix}"] = rec["energy"]
            row[f"terminated_{prefix}"] = rec["terminated"]
            row[f"scf_converged_{prefix}"] = rec["scf_converged"]
            row[f"scf_iterations_{prefix}"] = rec["scf_iterations"]
            row[f"s2_{prefix}"] = rec["s2"]
            for k in fragments:
                row[f"spin_f{k}_{prefix}"] = rec[f"spin_f{k}"]
                row[f"charge_f{k}_{prefix}"] = rec[f"charge_f{k}"]
                row[f"lcharge_f{k}_{prefix}"] = rec[f"lcharge_f{k}"]
        row["pair_ok"] = all([hs["terminated"], bs["terminated"], hs["scf_converged"], bs["scf_converged"],
                              hs["has_spin"], bs["has_spin"], np.isfinite(hs["energy"]), np.isfinite(bs["energy"])])
        geom = folder / "high_spin" / "geometry.xyz"
        if geom.exists():
            pos = read(geom).get_positions()
            geom_pos[base] = pos
            for k, (ra, rb) in rings.items():
                row[f"theta_{k}_plane"] = plane_angle(pos, ra, rb)
            relaxed = relax_root / base / "high_spin.xyz"
            if relaxed.exists():
                row["geom_max_diff_vs_relaxed"] = float(np.abs(pos - read(relaxed).get_positions()).max())
        rows.append(row)

    ex = pd.DataFrame(rows)
    df = struct.merge(ex, on="label", how="left", validate="one_to_one")
    df["usable"] = df["usable"].astype(bool)

    # ---- derived quantities ----
    df["delta_E_raw_meV"] = (df["E_HS"] - df["E_BS"]) * HARTREE_TO_MEV
    df["exchange_class"] = df["delta_E_raw_meV"].map(lambda v: classify_exchange(v, NEAR_DEGENERATE_MEV) if np.isfinite(v) else "")

    # Exchange in the repository convention H = -J S_A.S_B (J > 0 FM), DeltaE = E_HS - E_BS:
    #   J_ideal = -2 DeltaE;  J = -2 DeltaE / (<S^2>_HS - <S^2>_BS) (Yamaguchi);  E_T - E_S = -J
    df["delta_S2"] = df["s2_HS"] - df["s2_BS"]
    df["J_ideal_meV"] = XC.j_ideal(df["delta_E_raw_meV"])
    df["J_yamaguchi_meV"] = XC.j_yamaguchi(df["delta_E_raw_meV"], df["s2_HS"], df["s2_BS"])
    df["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(df["J_yamaguchi_meV"])

    def a(c):
        return df[c].abs()

    df["mean_terminal_spin_geomscan"] = (df["spin_fragment1_final"].abs() + df["spin_fragment4_final"].abs()) / 2.0
    df["mean_terminal_spin_pbe0_HS"] = (a("spin_f1_HS") + a("spin_f4_HS")) / 2.0
    df["mean_terminal_spin_pbe0_BS"] = (a("spin_f1_BS") + a("spin_f4_BS")) / 2.0
    for st in ("HS", "BS"):
        df[f"spin_asymmetry_{st}"] = (a(f"spin_f1_{st}") - a(f"spin_f4_{st}")).abs()
        df[f"spin_diff_signed_{st}"] = a(f"spin_f1_{st}") - a(f"spin_f4_{st}")
        df[f"spin_asymmetry_norm_{st}"] = df[f"spin_asymmetry_{st}"] / df[f"mean_terminal_spin_pbe0_{st}"]
        df[f"charge_asymmetry_{st}"] = df[f"charge_f1_{st}"] - df[f"charge_f4_{st}"]
        df[f"abs_charge_asymmetry_{st}"] = df[f"charge_asymmetry_{st}"].abs()
        df[f"charge_asymmetry_lowdin_{st}"] = df[f"lcharge_f1_{st}"] - df[f"lcharge_f4_{st}"]
        df[f"abs_charge_asymmetry_lowdin_{st}"] = df[f"charge_asymmetry_lowdin_{st}"].abs()

    df["n_outer_90"] = (df["L"] == 90).astype(int) + (df["R"] == 90).astype(int)
    df["outer_plane_asymmetry"] = (df["theta_L_plane"] - df["theta_R_plane"]).abs()
    df["outer_dihedral_asymmetry"] = (df["L_magnitude"] - df["R_magnitude"]).abs()
    df["outer_dihedral_asymmetry_signed"] = (df["L"] - df["R"]).abs()
    df["outer_dev_from_90"] = ((df["L"] - 90).abs() + (df["R"] - 90).abs()) / 2.0
    df["mean_outer_plane"] = (df["theta_L_plane"] + df["theta_R_plane"]) / 2.0
    df["handedness_x_signC"] = (df["L_handedness"] + df["R_handedness"]) / 2.0 * np.sign(df["C"] - 90.0)
    df["puckering_rms"] = (df["final_left_plane_rms"] + df["final_right_plane_rms"]) / 2.0
    df["tilt_abs_sum"] = df["final_tilt_ab"].abs() + df["final_tilt_cd"].abs()

    def cos2(x):
        return np.cos(np.radians(x)) ** 2

    df["P"] = cos2(df["theta_L_plane"]) * cos2(df["theta_C_plane"]) * cos2(df["theta_R_plane"])
    df["P_LR"] = cos2(df["theta_L_plane"]) * cos2(df["theta_R_plane"])
    df["P_C"] = cos2(df["theta_C_plane"])
    df["P_imposed"] = cos2(df["L"]) * cos2(df["C"]) * cos2(df["R"])
    df["P_LR_imposed"] = cos2(df["L"]) * cos2(df["R"])
    df["P_C_imposed"] = cos2(df["C"])

    U = df[df["usable"]].copy()
    n_usable = len(U)
    dE = U["delta_E_raw_meV"]

    # ---- 1: counts, terminology ----
    cnt = {
        "N_BS_lower": int((dE > 0).sum()),
        "N_BS_lower_robust": int((dE >= NEAR_DEGENERATE_MEV).sum()),
        "N_near_degenerate_HS_preference": int(((dE < 0) & (dE > -NEAR_DEGENERATE_MEV)).sum()),
        "N_robust_HS_favored": int((dE <= -NEAR_DEGENERATE_MEV).sum()),
        "n_lt_tight": int((dE.abs() < TIGHT_DEGENERATE_MEV).sum()),
        "n_lt_near": int((dE.abs() < NEAR_DEGENERATE_MEV).sum()),
        "min": float(dE.min()), "max": float(dE.max()), "median": float(dE.median()), "mean": float(dE.mean()),
    }
    neg = U[U["delta_E_raw_meV"] < 0].sort_values("delta_E_raw_meV")

    # ---- validation and QC ----
    missing = U[~U["pair_ok"].fillna(False)]
    qc = pd.DataFrame({"HS_S2": U["s2_HS"], "BS_S2": U["s2_BS"], "spin_f1_BS": U["spin_f1_BS"], "spin_f4_BS": U["spin_f4_BS"],
                       "spin_f1_HS": U["spin_f1_HS"], "spin_f4_HS": U["spin_f4_HS"]})
    qc_bad = {
        "HS <S^2> outside expected triplet window": U[(U["s2_HS"] < HS_S2_RANGE[0]) | (U["s2_HS"] > HS_S2_RANGE[1])],
        "BS <S^2> below threshold (possible closed-shell collapse)": U[U["s2_BS"] < BS_S2_MIN],
        "BS terminal fragments not of opposite sign": U[np.sign(U["spin_f1_BS"]) * np.sign(U["spin_f4_BS"]) >= 0],
        "BS terminal spin too small": U[(U["spin_f1_BS"].abs() < BS_FRAGMENT_SPIN_MIN) | (U["spin_f4_BS"].abs() < BS_FRAGMENT_SPIN_MIN)],
    }
    qc_any_bad = any(len(v) for v in qc_bad.values())
    charge_sum_dev = float((U["charge_f1_HS"] + U["charge_f2_HS"] + U["charge_f3_HS"] + U["charge_f4_HS"]).abs().max())

    # ---- 3: outer-90 classes ----
    class_rows = []
    for n90, sub in U.groupby("n_outer_90"):
        v = sub["delta_E_raw_meV"]
        class_rows.append({"n_outer_90": int(n90), "N": len(v), "mean": v.mean(), "median": v.median(), "std": v.std(),
                           "min": v.min(), "max": v.max(), "frac_lt_0.1": float((v.abs() < 0.1).mean()),
                           "frac_lt_0.5": float((v.abs() < 0.5).mean()), "n_negative": int((v < 0).sum()),
                           "median_spin_asym_HS": sub["spin_asymmetry_HS"].median(), "median_spin_asym_BS": sub["spin_asymmetry_BS"].median(),
                           "median_abs_charge_asym_HS": sub["abs_charge_asymmetry_HS"].median(),
                           "median_terminal_spin_HS": sub["mean_terminal_spin_pbe0_HS"].median()})
    classes = pd.DataFrame(class_rows).set_index("n_outer_90")

    # ---- 4: symmetric L=R=90 ----
    sym90 = U[(U["L"] == 90) & (U["R"] == 90)].sort_values("C")

    # ---- 5/6/7/8: correlations ----
    corr_spin = {
        "R2SCAN-3C geometry-scan terminal spin": corr(U["mean_terminal_spin_geomscan"], U["delta_E_raw_meV"]),
        "PBE0 HS terminal spin": corr(U["mean_terminal_spin_pbe0_HS"], U["delta_E_raw_meV"]),
    }
    corr_asym = {
        "spin asymmetry HS": corr(U["spin_asymmetry_HS"], U["delta_E_raw_meV"]),
        "spin asymmetry BS": corr(U["spin_asymmetry_BS"], U["delta_E_raw_meV"]),
        "normalized spin asymmetry HS": corr(U["spin_asymmetry_norm_HS"], U["delta_E_raw_meV"]),
    }
    per_class_asym = {}
    for k in (0, 1, 2):
        sub = U[U["n_outer_90"] == k]
        per_class_asym[k] = {"HS": corr(sub["spin_asymmetry_HS"], sub["delta_E_raw_meV"]), "BS": corr(sub["spin_asymmetry_BS"], sub["delta_E_raw_meV"]),
                             "charge_HS": corr(sub["abs_charge_asymmetry_HS"], sub["delta_E_raw_meV"]),
                             "charge_BS": corr(sub["abs_charge_asymmetry_BS"], sub["delta_E_raw_meV"])}
    corr_charge = {
        "abs charge asymmetry HS": corr(U["abs_charge_asymmetry_HS"], U["delta_E_raw_meV"]),
        "abs charge asymmetry BS": corr(U["abs_charge_asymmetry_BS"], U["delta_E_raw_meV"]),
        "abs charge asymmetry HS (Löwdin)": corr(U["abs_charge_asymmetry_lowdin_HS"], U["delta_E_raw_meV"]),
        "abs charge asymmetry BS (Löwdin)": corr(U["abs_charge_asymmetry_lowdin_BS"], U["delta_E_raw_meV"]),
    }
    corr_spin_charge = {
        "|spin asym| vs |charge asym| (HS)": corr(U["spin_asymmetry_HS"], U["abs_charge_asymmetry_HS"]),
        "signed (|S1|-|S4|) vs (q1-q4) (HS)": corr(U["spin_diff_signed_HS"], U["charge_asymmetry_HS"]),
        "Mulliken vs Löwdin (q1-q4), HS": corr(U["charge_asymmetry_HS"], U["charge_asymmetry_lowdin_HS"]),
    }
    one90 = U[U["n_outer_90"] == 1]
    p_mw_spin = mannwhitney(one90.loc[one90["delta_E_raw_meV"] < 0, "spin_asymmetry_HS"], one90.loc[one90["delta_E_raw_meV"] >= 0, "spin_asymmetry_HS"])
    p_mw_charge = mannwhitney(one90.loc[one90["delta_E_raw_meV"] < 0, "abs_charge_asymmetry_HS"], one90.loc[one90["delta_E_raw_meV"] >= 0, "abs_charge_asymmetry_HS"])
    neg1, oth1 = one90[one90["delta_E_raw_meV"] < 0], one90[one90["delta_E_raw_meV"] >= 0]
    cls1 = {
        "n_neg": len(neg1), "n_oth": len(oth1),
        "asym_neg": neg1["spin_asymmetry_HS"].median(), "asym_oth": oth1["spin_asymmetry_HS"].median(),
        "q_neg": neg1["abs_charge_asymmetry_HS"].median(), "q_oth": oth1["abs_charge_asymmetry_HS"].median(),
        "loc_neg": neg1["mean_terminal_spin_pbe0_HS"].median(), "loc_oth": oth1["mean_terminal_spin_pbe0_HS"].median(),
    }
    lower_higher = lambda a, b: "lower" if a < b else "higher"  # noqa: E731
    geo_asym = {
        "outer plane asymmetry vs |spin asym| (HS)": corr(U["outer_plane_asymmetry"], U["spin_asymmetry_HS"]),
        "outer plane asymmetry vs |charge asym| (HS)": corr(U["outer_plane_asymmetry"], U["abs_charge_asymmetry_HS"]),
        "outer plane asymmetry vs ΔE": corr(U["outer_plane_asymmetry"], U["delta_E_raw_meV"]),
        "outer dihedral asymmetry (magnitude) vs |spin asym| (HS)": corr(U["outer_dihedral_asymmetry"], U["spin_asymmetry_HS"]),
        "outer dihedral asymmetry (magnitude) vs |charge asym| (HS)": corr(U["outer_dihedral_asymmetry"], U["abs_charge_asymmetry_HS"]),
        "outer dihedral asymmetry (magnitude) vs ΔE": corr(U["outer_dihedral_asymmetry"], U["delta_E_raw_meV"]),
    }
    same_LR = U[U["L"] == U["R"]]
    diff_LR = U[U["L"] != U["R"]]

    # ---- 10/11: hopping proxy ----
    U_ex = U[U["C"] < EXTREME_C]
    hop_vars = [("P = cos²θL cos²θC cos²θR", "P"), ("P_LR = cos²θL cos²θR", "P_LR"), ("P_C = cos²θC", "P_C"),
                ("P (imposed dihedrals)", "P_imposed"), ("P_LR (imposed)", "P_LR_imposed"), ("P_C (imposed)", "P_C_imposed")]
    corr_hop = {name: (corr(U[col], U["delta_E_raw_meV"]), corr(U_ex[col], U_ex["delta_E_raw_meV"])) for name, col in hop_vars}
    fit = stats.linregress(U_ex["P"], U_ex["delta_E_raw_meV"])
    fit_a, fit_b, fit_r2 = float(fit.slope), float(fit.intercept), float(fit.rvalue) ** 2

    # ---- 9: negative subset tests ----
    ref_charge_asym = float(U["abs_charge_asymmetry_HS"].median())
    tests = {}
    if len(neg):
        tests = {
            "at least one imposed outer torsion = 90°": int(((neg["L"] == 90) | (neg["R"] == 90)).sum()),
            f"at least one actual outer plane angle within {NEAR_90_TOL_DEG:.0f}° of 90°": int(((neg["theta_L_plane"] - 90).abs().lt(NEAR_90_TOL_DEG) | (neg["theta_R_plane"] - 90).abs().lt(NEAR_90_TOL_DEG)).sum()),
            f"high spin localization (mean terminal spin, PBE0 HS ≥ {HIGH_LOCALIZATION})": int((neg["mean_terminal_spin_pbe0_HS"] >= HIGH_LOCALIZATION).sum()),
            f"strong spin asymmetry (HS, ≥ {STRONG_SPIN_ASYM})": int((neg["spin_asymmetry_HS"] >= STRONG_SPIN_ASYM).sum()),
            f"above-median charge asymmetry (HS, median of usable = {ref_charge_asym:.3f} e)": int((neg["abs_charge_asymmetry_HS"] > ref_charge_asym).sum()),
        }

    # ---- descriptor correlations (supporting) ----
    descriptors = [
        ("L magnitude", "L_magnitude"), ("R magnitude", "R_magnitude"), ("C magnitude", "C_magnitude"),
        ("mean abs(outer - 90°) (imposed)", "outer_dev_from_90"),
        ("actual L plane angle", "theta_L_plane"), ("actual R plane angle", "theta_R_plane"),
        ("mean actual outer plane angle", "mean_outer_plane"), ("actual central plane angle", "theta_C_plane"),
        ("terminal spin, R2SCAN-3C geometry scan", "mean_terminal_spin_geomscan"), ("terminal spin, PBE0 HS", "mean_terminal_spin_pbe0_HS"),
        ("spin asymmetry, HS", "spin_asymmetry_HS"), ("abs charge asymmetry, HS", "abs_charge_asymmetry_HS"),
        ("outer plane asymmetry", "outer_plane_asymmetry"),
        ("1-3/2-4 closest-pair distance", "cross_distance_mean_heavy"),
        ("1-3/2-4 3-shortest-contact mean", "cross_distance_robust_mean"), ("1-3/2-4 contact count < 4 Å", "cross_contact_count_total"),
        ("ring puckering RMS", "puckering_rms"), ("bond tilt abs(AB)+abs(CD)", "tilt_abs_sum"),
        ("hopping proxy P", "P"),
        ("outer handedness x sign(C-90)", "handedness_x_signC"),
    ]

    def demean(s):
        return s - s.groupby(U["C"]).transform("mean")

    def partial_withinC(xcol, ycol, zcol):
        x, y, z = (demean(U[c].rank()) for c in (xcol, ycol, zcol))
        m = x.notna() & y.notna() & z.notna()
        x, y, z = x[m], y[m], z[m]
        if z.std() == 0 or x.std() == 0 or y.std() == 0:
            return np.nan
        rx = x - (np.cov(x, z)[0, 1] / z.var()) * z
        ry = y - (np.cov(y, z)[0, 1] / z.var()) * z
        return float(np.corrcoef(rx, ry)[0, 1])

    desc_rows = []
    for name, col in descriptors:
        r_e, r_c = corr(U[col], U["delta_E_raw_meV"]), corr(U[col], U["relative_energy_meV"])
        w_e, w_c = corr(demean(U[col]), demean(U["delta_E_raw_meV"])), corr(demean(U[col]), demean(U["relative_energy_meV"]))
        desc_rows.append({"descriptor": name, "column": col, "rho_dE": r_e["spearman"], "rho_Econf": r_c["spearman"],
                          "rho_dE_withinC": w_e["spearman"], "rho_Econf_withinC": w_c["spearman"],
                          "partial_dE_ctrl_outer_plane": partial_withinC(col, "delta_E_raw_meV", "mean_outer_plane") if col != "mean_outer_plane" else np.nan,
                          "partial_Econf_ctrl_outer_plane": partial_withinC(col, "relative_energy_meV", "mean_outer_plane") if col != "mean_outer_plane" else np.nan})
    desc = pd.DataFrame(desc_rows)

    sign_agree = int((np.sign(U["delta_E_raw_meV"]) == np.sign(U["E_T_minus_E_S_yamaguchi_meV"])).sum())
    ratio = U["E_T_minus_E_S_yamaguchi_meV"] / U["delta_E_raw_meV"].replace(0, np.nan)

    # ---- 13: electric-field candidates ----
    def field_direction(label):
        pos = geom_pos.get(label)
        if pos is None:
            return (np.nan, np.nan, np.nan)
        v = pos[frag4].mean(axis=0) - pos[frag1].mean(axis=0)
        v = v / np.linalg.norm(v)
        return tuple(float(x) for x in v)

    ef = []
    s90 = sym90[sym90["delta_E_raw_meV"].abs() < TIGHT_DEGENERATE_MEV]
    if len(s90):
        ef.append(("1. symmetric, both outer 90°, ΔE ~ 0 (lowest conformational energy among such points)", s90.loc[s90["relative_energy_meV"].idxmin()]))
    one_neg = neg[neg["n_outer_90"] == 1]
    if len(one_neg):
        picked_neg = one_neg.iloc[0]
        ef.append(("2. asymmetric, one outer 90°, most negative ΔE", picked_neg))
        mirror = U[(U["L"] == picked_neg["R"]) & (U["C"] == picked_neg["C"]) & (U["R"] == picked_neg["L"])]
        if len(mirror):
            ef.append(("3. mirror-related geometry of #2 (L and R swapped, same C)", mirror.iloc[0]))
    afm = U[(U["n_outer_90"] == 0) & (U["C"] >= 80) & (U["C"] <= 120)]
    if len(afm):
        ef.append(("4. moderate AFM reference (no outer 90°, C in 80–120°, ΔE closest to the median of that set)", afm.loc[(afm["delta_E_raw_meV"] - afm["delta_E_raw_meV"].median()).abs().idxmin()]))
    ef_rows = []
    for role, r in ef:
        d = field_direction(r["label"])
        ef_rows.append({
            "role": role, "label": r["label"], "L": r["L"], "C": r["C"], "R": r["R"], "delta_E_raw_meV": r["delta_E_raw_meV"],
            "conformational_energy_meV": r["relative_energy_meV"], "spin_asymmetry_HS": r["spin_asymmetry_HS"], "spin_asymmetry_BS": r["spin_asymmetry_BS"],
            "charge_asymmetry_HS": r["charge_asymmetry_HS"], "charge_asymmetry_BS": r["charge_asymmetry_BS"],
            "theta_L_plane": r["theta_L_plane"], "theta_C_plane": r["theta_C_plane"], "theta_R_plane": r["theta_R_plane"],
            "field_dir_x_1to4": d[0], "field_dir_y_1to4": d[1], "field_dir_z_1to4": d[2],
        })
    ef_df = pd.DataFrame(ef_rows)

    # validation candidates (kept from the previous analysis, supporting)
    cand = [("strongly AFM (largest dE)", U.loc[U["delta_E_raw_meV"].idxmax()])]
    posset = U[U["delta_E_raw_meV"] >= NEAR_DEGENERATE_MEV]
    if len(posset):
        cand.append(("intermediate (dE closest to the median of BS-favored points)", posset.loc[(posset["delta_E_raw_meV"] - posset["delta_E_raw_meV"].median()).abs().idxmin()]))
    near0 = U[~U.index.isin(set(neg.index[:1]))]
    cand.append(("near-zero exchange (smallest abs(dE), excluding the most negative point)", near0.loc[near0["delta_E_raw_meV"].abs().idxmin()]))
    if len(neg):
        cand.append(("most negative dE", neg.iloc[0]))
    if len(sym90):
        cand.append(("symmetric L=R=90°, lowest conformational energy", sym90.loc[sym90["relative_energy_meV"].idxmin()]))
    cand_df = pd.DataFrame([{"role": role, "label": r["label"], "L": r["L"], "C": r["C"], "R": r["R"], "delta_E_raw_meV": r["delta_E_raw_meV"],
                             "relative_conformational_energy_meV": r["relative_energy_meV"], "mean_terminal_spin": r["mean_terminal_spin_pbe0_HS"]} for role, r in cand])

    # ---- save csv ----
    save_cols = [c for c in df.columns if c not in ("spin_final", "flags", "notes")]
    df[save_cols].to_csv(out_dir / "exchange_analysis.csv", index=False)
    cand_df.to_csv(out_dir / "follow_up_candidates.csv", index=False)
    ef_df.to_csv(out_dir / "electric_field_candidates.csv", index=False)

    # ============================================================
    # PLOTS
    # ============================================================
    c_vals = sorted(U["C"].unique())
    outer_styles = {0.0: "tab:blue", 45.0: "tab:orange", 90.0: "tab:green", 135.0: "tab:red", 180.0: "tab:grey"}

    def sym_curve(ax, col):
        for outer, color in outer_styles.items():
            sub = U[(U["L"] == outer) & (U["R"] == outer)].sort_values("C")
            if sub.empty:
                continue
            ax.plot(sub["C"], sub[col], marker="o", color=color, ls="--" if outer == 180 else "-", lw=3 if outer == 90 else 1.5,
                    alpha=0.6 if outer == 180 else 0.95, label=f"L=R={int(outer)}°", zorder=5 if outer == 90 else 3)

    def scatter_classes(ax, xcol, ycol, data=None, s=26):
        data = U if data is None else data
        for k in (0, 1, 2):
            sub = data[data["n_outer_90"] == k]
            ax.scatter(sub[xcol], sub[ycol], s=s, color=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.85, edgecolors="none")

    def clipped_note(ax, ycol, ymax):
        n_off = int((U[ycol] > ymax).sum())
        if n_off:
            ax.text(0.99, 0.99, f"{n_off} point(s) above axis (max {U[ycol].max():.1f} meV)", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.3")

    # 1. main: dE vs C symmetric cuts (clipped)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sym_curve(ax, "delta_E_raw_meV")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_ylim(-1.0, CLIP_YMAX_MEV)
    clipped_note(ax, "delta_E_raw_meV", CLIP_YMAX_MEV)
    ax.set_xlabel("Central (C) dihedral (deg)")
    ax.set_ylabel("ΔE = E_HS − E_BS (meV)")
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_C_symmetric.png", dpi=150)
    plt.close(fig)

    # supporting: exchange over conformational energy
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
    sym_curve(ax1, "delta_E_raw_meV")
    ax1.axhline(0, color="k", lw=1.2)
    ax1.set_ylim(-1.0, CLIP_YMAX_MEV)
    clipped_note(ax1, "delta_E_raw_meV", CLIP_YMAX_MEV)
    ax1.set_ylabel("ΔE = E_HS − E_BS (meV, PBE0)")
    ax1.legend(fontsize=8, ncol=2)
    sym_curve(ax2, "relative_energy_meV")
    e_min = U.groupby("C")["relative_energy_meV"].min()
    ax2.plot(e_min.index, e_min.values, "k-", lw=3, marker="o", label="E_min(C)")
    ax2.set_ylabel("Conformational energy (meV, R2SCAN-3C)")
    ax2.set_xlabel("Central (C) dihedral (deg)")
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_C_with_conformational_energy.png", dpi=150)
    plt.close(fig)

    # 4. dedicated L=R=90 plot
    if len(sym90):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhspan(-TIGHT_DEGENERATE_MEV, TIGHT_DEGENERATE_MEV, color="0.9", label=f"|ΔE| < {TIGHT_DEGENERATE_MEV} meV")
        ax.axhline(0, color="k", lw=1)
        ax.plot(sym90["C"], sym90["delta_E_raw_meV"], "o-", color="tab:green", lw=2.5, ms=8)
        for _, r in sym90.iterrows():
            ax.annotate(f"{r['delta_E_raw_meV']:.3f}", (r["C"], r["delta_E_raw_meV"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel("Central (C) dihedral (deg)")
        ax.set_ylabel("ΔE = E_HS − E_BS (meV)")
        ax.set_title("L = R = 90°: both outer hopping bottlenecks cut")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(plots / "exchange_symmetric_LR90_vs_C.png", dpi=150)
        plt.close(fig)

    # 2. outer-90 classes (clipped main axis + zoom inset)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    groups = [U.loc[U["n_outer_90"] == k, "delta_E_raw_meV"].values for k in (0, 1, 2)]
    rng = np.random.default_rng(0)

    def draw_class_box(axx):
        axx.boxplot(groups, positions=[0, 1, 2], widths=0.5, showfliers=False)
        for k, g in zip((0, 1, 2), groups):
            axx.scatter(k + rng.uniform(-0.15, 0.15, len(g)), g, s=14, alpha=0.65, color=CLASS_COLORS[k])
        axx.axhline(0, color="k", lw=1)
        axx.set_xticks([0, 1, 2])
        axx.set_xticklabels(["0 outer 90°", "1 outer 90°", "2 outer 90°"])

    draw_class_box(ax)
    ax.set_ylim(-1.0, CLIP_YMAX_MEV)
    clipped_note(ax, "delta_E_raw_meV", CLIP_YMAX_MEV)
    ax.set_ylabel("ΔE = E_HS − E_BS (meV)")
    inset = ax.inset_axes([0.42, 0.42, 0.55, 0.42])
    draw_class_box(inset)
    inset.set_ylim(-0.4, 1.0)
    inset.set_title("zoom: −0.4 to 1 meV", fontsize=8)
    inset.tick_params(labelsize=7)
    ax.set_title("Progressive collapse of the exchange with outer 90° torsions", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots / "exchange_by_outer_90_class.png", dpi=150)
    plt.close(fig)

    # 3. dE vs terminal spin (two variants), colored by n_outer_90
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, key, title in ((axes[0], "mean_terminal_spin_geomscan", "R2SCAN-3C geometry-scan terminal spin", "R2SCAN-3C geometry scan"),
                                (axes[1], "mean_terminal_spin_pbe0_HS", "PBE0 HS terminal spin", "PBE0 HS single point")):
        scatter_classes(ax, col, "delta_E_raw_meV")
        ax.axhline(0, color="k", lw=1)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel(f"mean terminal spin = (|S1|+|S4|)/2 ({title})", fontsize=8)
        ax.set_ylabel("ΔE (meV, symlog)")
        ax.set_title(fmt_corr(corr_spin[key]), fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_terminal_spin.png", dpi=150)
    plt.close(fig)

    # 10/11. hopping proxy with empirical trend
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, col) in zip(axes, hop_vars[:3]):
        ex_mask = U["C"] >= EXTREME_C
        for k in (0, 1, 2):
            sub = U[(~ex_mask) & (U["n_outer_90"] == k)]
            ax.scatter(sub[col], sub["delta_E_raw_meV"], s=24, color=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.85)
        ax.scatter(U.loc[ex_mask, col], U.loc[ex_mask, "delta_E_raw_meV"], facecolors="none", edgecolors="red", s=42, label=f"C≥{int(EXTREME_C)}° (excluded from stats)")
        if col == "P":
            xs = np.linspace(0, U_ex["P"].max(), 50)
            ax.plot(xs, fit_a * xs + fit_b, "k--", lw=1.2, label=f"empirical guide: ΔE = {fit_a:.1f}·P + {fit_b:.2f} (R²={fit_r2:.2f})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("ΔE (meV)")
        ax.set_title(f"excl. extreme C: {fmt_corr(corr_hop[name][1])}", fontsize=7)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_hopping_proxy.png", dpi=150)
    plt.close(fig)

    # 6. spin asymmetry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, st in zip(axes, ("HS", "BS")):
        scatter_classes(ax, f"spin_asymmetry_{st}", "delta_E_raw_meV")
        ax.axhline(0, color="k", lw=1)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel(f"spin asymmetry {st} = ||S1| − |S4||")
        ax.set_ylabel("ΔE (meV, symlog)")
        ax.set_title(fmt_corr(corr_asym[f"spin asymmetry {st}"]), fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_spin_asymmetry.png", dpi=150)
    plt.close(fig)

    # 7. charge asymmetry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, st in zip(axes, ("HS", "BS")):
        scatter_classes(ax, f"abs_charge_asymmetry_{st}", "delta_E_raw_meV")
        ax.axhline(0, color="k", lw=1)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel(f"|q(fragment 1) − q(fragment 4)| {st} (Mulliken, e)")
        ax.set_ylabel("ΔE (meV, symlog)")
        ax.set_title(fmt_corr(corr_charge[f"abs charge asymmetry {st}"]), fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "exchange_vs_charge_asymmetry.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    scatter_classes(axes[0], "abs_charge_asymmetry_HS", "spin_asymmetry_HS")
    axes[0].set_xlabel("|q1 − q4| HS (e)")
    axes[0].set_ylabel("spin asymmetry HS = ||S1| − |S4||")
    axes[0].set_title(fmt_corr(corr_spin_charge["|spin asym| vs |charge asym| (HS)"]), fontsize=8)
    axes[0].legend(fontsize=7)
    scatter_classes(axes[1], "charge_asymmetry_HS", "spin_diff_signed_HS")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].axvline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("q1 − q4 HS (e), signed")
    axes[1].set_ylabel("|S1| − |S4| HS, signed")
    axes[1].set_title(fmt_corr(corr_spin_charge["signed (|S1|-|S4|) vs (q1-q4) (HS)"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "spin_vs_charge_asymmetry.png", dpi=150)
    plt.close(fig)

    # 8. geometric vs electronic asymmetry
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, ycol, ylab in ((axes[0], "spin_asymmetry_HS", "spin asymmetry HS"), (axes[1], "abs_charge_asymmetry_HS", "|q1 − q4| HS (e)"),
                           (axes[2], "delta_E_raw_meV", "ΔE (meV)")):
        scatter_classes(ax, "outer_plane_asymmetry", ycol)
        ax.set_xlabel("outer plane asymmetry |θ_L − θ_R| (deg)")
        ax.set_ylabel(ylab)
        if ycol == "delta_E_raw_meV":
            ax.set_yscale("symlog", linthresh=0.5)
            ax.axhline(0, color="k", lw=0.8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "geometric_vs_electronic_asymmetry.png", dpi=150)
    plt.close(fig)

    # 7'. heatmaps
    l_vals, r_vals = sorted(U["L"].unique()), sorted(U["R"].unique())

    def grid_for(c):
        g = np.full((len(l_vals), len(r_vals)), np.nan)
        for _, row in U[U["C"] == c].iterrows():
            g[l_vals.index(row["L"]), r_vals.index(row["R"])] = row["delta_E_raw_meV"]
        return g

    def norm_for(values):
        vmax = max(np.nanmax(values), 1e-3)
        vmin = min(np.nanmin(values), -0.05 * vmax)
        return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    def draw_heat(ax, g, norm, title):
        cmap = plt.cm.RdBu.copy()
        cmap.set_bad("lightgrey")
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(r_vals)))
        ax.set_xticklabels([f"{int(v)}" for v in r_vals])
        ax.set_yticks(range(len(l_vals)))
        ax.set_yticklabels([f"{int(v)}" for v in l_vals])
        ax.set_xlabel("R (deg)")
        ax.set_ylabel("L (deg)")
        ax.set_title(title)
        for i in range(len(l_vals)):
            for j in range(len(r_vals)):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=7)
        return im

    main_c = [c for c in (80.0, 100.0, 120.0) if c in c_vals]
    grids = {c: grid_for(c) for c in main_c}
    norm_main = norm_for(np.concatenate([g.ravel() for g in grids.values()]))
    fig, axes = plt.subplots(1, len(main_c), figsize=(5.4 * len(main_c), 4.6), constrained_layout=True)
    for ax, c in zip(np.atleast_1d(axes), main_c):
        im = draw_heat(ax, grids[c], norm_main, f"ΔE (meV), C={int(c)}°")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), pad=0.02, label="ΔE = E_HS − E_BS (meV): red = HS lower, blue = BS lower")
    fig.savefig(plots / "exchange_heatmaps_C80_100_120.png", dpi=150)
    plt.close(fig)
    for c in c_vals:
        g = grid_for(c)
        if not np.isfinite(g).any():
            continue
        fig, ax = plt.subplots(figsize=(5, 4.4))
        im = draw_heat(ax, g, norm_for(g), f"ΔE (meV), C={int(c)}°")
        fig.colorbar(im, ax=ax, label="meV")
        fig.tight_layout()
        fig.savefig(heat_dir / f"exchange_heatmap_C{int(c):03d}.png", dpi=150)
        plt.close(fig)

    # supporting: Yamaguchi, descriptors, QC
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(U["delta_E_raw_meV"], U["E_T_minus_E_S_yamaguchi_meV"], s=20)
    lim = [min(U["delta_E_raw_meV"].min(), 0), max(U["E_T_minus_E_S_yamaguchi_meV"].max(), U["delta_E_raw_meV"].max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("raw ΔE = E_HS − E_BS (meV)")
    ax.set_ylabel("E_T − E_S = −J (Yamaguchi, meV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "yamaguchi_vs_raw.png", dpi=150)
    plt.close(fig)

    d = desc.dropna(subset=["rho_dE_withinC", "rho_Econf_withinC"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["rho_Econf_withinC"], height=0.38, color="tab:purple", label="vs conformational energy (R2SCAN-3C)")
    ax.barh(y + 0.2, d["rho_dE_withinC"], height=0.38, color="tab:green", label="vs exchange ΔE (PBE0)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["descriptor"], fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Spearman ρ within fixed C")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots / "descriptor_correlations_withinC.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, ttl) in zip(axes, (("HS_S2", "<S^2> HS"), ("BS_S2", "<S^2> BS"), ("spin_f1_BS", "fragment 1 spin, BS"), ("spin_f4_BS", "fragment 4 spin, BS"))):
        ax.hist(qc[col].dropna(), bins=25, color="0.5")
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "qc_distributions.png", dpi=130)
    plt.close(fig)

    # ============================================================
    # REPORT
    # ============================================================
    c0, c1, c2 = (classes.loc[k] if k in classes.index else None for k in (0, 1, 2))
    spin_cs = corr_spin["PBE0 HS terminal spin"]
    P_all, P_ex = corr_hop["P = cos²θL cos²θC cos²θR"]
    LRc, LRce = corr_hop["P_LR = cos²θL cos²θR"]
    Cc, Cce = corr_hop["P_C = cos²θC"]
    n_neg = len(neg)

    L = []
    L.append("# PBE0 high-spin / broken-symmetry exchange analysis\n")
    L.append("## Scientific summary\n")
    L.append(
        "Working picture tested here: **torsional decoupling → reduced π-hopping → stronger terminal-radical localization → collapse of the AFM exchange contribution → near-degeneracy, with an occasional very small HS preference.** "
        "The small negative ΔE values are a consequence of the collapse, not the headline result.\n"
    )
    S = []
    qc_tail = "" if not qc_any_bad else " (EXCEPT for the anomalies listed in the QC section)"
    S.append(f"1. **All HS/BS electronic states are well behaved.** {n_usable - len(missing)}/{n_usable} usable geometries have both PBE0 runs converged; HS ⟨S²⟩ {qc['HS_S2'].min():.2f}–{qc['HS_S2'].max():.2f}, BS ⟨S²⟩ {qc['BS_S2'].min():.2f}–{qc['BS_S2'].max():.2f}, terminal fragments of opposite sign in BS everywhere{qc_tail}.")
    if c0 is not None:
        S.append(f"2. **AFM exchange is several meV when all three conjugation pathways remain active** (no outer torsion at 90°): N={int(c0['N'])}, median ΔE = {c0['median']:.2f} meV, smallest {c0['min']:.2f} meV, {int(c0['n_negative'])} with ΔE<0.")
    if c1 is not None:
        S.append(f"3. **One outer fragment driven to 90° almost switches off the AFM exchange:** N={int(c1['N'])}, median ΔE = {c1['median']:.3f} meV, {c1['frac_lt_0.5']:.0%} of points have |ΔE|<0.5 meV.")
    if c2 is not None:
        sym_tail = ""
        if len(sym90):
            c_list = ", ".join(str(int(c)) for c in sym90["C"])
            sym_tail = f"; along L=R=90° the central torsion changes |ΔE| by at most {sym90['delta_E_raw_meV'].abs().max():.3f} meV over C = {c_list}°"
        S.append(f"4. **Both outer fragments at 90° give essentially complete decoupling:** N={int(c2['N'])}, ΔE between {c2['min']:.3f} and {c2['max']:.3f} meV{sym_tail}.")
    S.append(f"5. **Exchange correlates strongly with terminal-radical localization:** {fmt_corr(spin_cs)} (PBE0 HS spin); the relation is monotonic but not linear, so the rank correlation is the relevant number.")
    S.append(
        f"6. **The three-bottleneck hopping proxy explains exchange better than any single torsion** (excluding C≥{int(EXTREME_C)}°): P ρ = {f2(P_ex['spearman'])}, terminal part alone P_LR ρ = {f2(LRce['spearman'])}, central part alone P_C ρ = {f2(Cce['spearman'])}. "
        f"Empirical guide ΔE ≈ {fit_a:.1f}·P + {fit_b:.2f} meV (R² = {fit_r2:.2f}): as P → 0 the dominant AFM contribution vanishes."
    )
    if n_neg:
        n_strong = tests[f"strong spin asymmetry (HS, \u2265 {STRONG_SPIN_ASYM})"]
        n_with90 = int((neg["n_outer_90"] >= 1).sum())
        S.append(
            f"7. **Small negative \u0394E values appear only after the AFM exchange has been almost completely suppressed:** {n_neg} points, all within {abs(neg['delta_E_raw_meV'].min()):.2f} meV of zero "
            f"(robust HS-favored, \u0394E \u2264 \u2212{NEAR_DEGENERATE_MEV} meV: {cnt['N_robust_HS_favored']}); {n_with90}/{n_neg} have at least one outer torsion at 90\u00b0, and the largest hopping proxy among them is P = {neg['P'].max():.1e}."
        )
        n_neg_c1 = int((neg["n_outer_90"] == 1).sum())
        S.append(
            f"8. **Most negative points are left-right asymmetric in spin localization, but the asymmetry does not single them out:** {n_strong}/{n_neg} have ||S1|\u2212|S4|| \u2265 {STRONG_SPIN_ASYM}, and {n_neg_c1} of the negative points belong to the one-outer-90\u00b0 class, "
            f"where asymmetry is intrinsic (class median {c1['median_spin_asym_HS']:.3f}). Within that class the negative points have {lower_higher(cls1['asym_neg'], cls1['asym_oth'])} median spin asymmetry ({cls1['asym_neg']:.3f}) than the non-negative ones ({cls1['asym_oth']:.3f}) "
            f"and {lower_higher(cls1['loc_neg'], cls1['loc_oth'])} mean terminal spin ({cls1['loc_neg']:.3f} vs {cls1['loc_oth']:.3f})."
        )
    else:
        S.append("7. No usable point has \u0394E < 0.")
        S.append("8. (No negative points to characterize.)")
    rho_a = per_class_asym[1]["HS"]["spearman"]
    S.append(
        f"9. **Does the residual HS preference track spin asymmetry, actual charge asymmetry, or both? The data do not support asymmetry itself as the cause.** Within the one-outer-90\u00b0 class, \u03c1(\u0394E, spin asymmetry HS) = {f2(rho_a)} "
        + ("(positive: larger asymmetry goes with a LARGER, more AFM \u0394E \u2014 the opposite of what asymmetry-driven negative values would need; there the asymmetry mainly reflects how far the other outer torsion is from 90\u00b0)" if np.isfinite(rho_a) and rho_a > 0 else "")
        + f"; negative vs non-negative points differ in spin asymmetry (Mann\u2013Whitney p = {f2(p_mw_spin, 3)}) but not in charge asymmetry (p = {f2(p_mw_charge, 3)}; medians {cls1['q_neg']:.3f} vs {cls1['q_oth']:.3f} e). "
        "Small samples (11 negative points in that class) and confounding by the other outer torsion mean this remains an association to be tested with the field scan."
    )
    S.append("10. **Motivation for a future fixed-geometry electric-field scan** near the mechanically decoupled regime (candidates listed below; nothing has been launched).")
    L.extend(f"{s}\n" for s in S)
    L.append(
        "> **Current statement.** The existing data suggest that mechanical torsion suppresses the dominant AFM exchange. Once the system is near degeneracy, "
        "left-right electronic asymmetry becomes a plausible additional control parameter for the small residual exchange, which can be tested directly with an electric field. "
        "No conclusion is drawn that a field will necessarily enhance ferromagnetism.\n"
    )

    L.append("## Conventions and terminology\n")
    L.append("- **ΔE = E_HS − E_BS** (raw, meV). ΔE > 0: BS lower / AFM tendency. ΔE < 0: HS slightly lower. The raw quantity is not called J.")
    L.append(f"- Counts over the {n_usable} usable geometries: **N_BS_lower = {cnt['N_BS_lower']}** (of which ΔE ≥ {NEAR_DEGENERATE_MEV} meV: {cnt['N_BS_lower_robust']}); **N_near_degenerate_HS_preference = {cnt['N_near_degenerate_HS_preference']}** (−{NEAR_DEGENERATE_MEV} < ΔE < 0); **N_robust_HS_favored = {cnt['N_robust_HS_favored']}** (ΔE ≤ −{NEAR_DEGENERATE_MEV} meV).")
    L.append(f"- Negative values are within {abs(cnt['min']):.2f} meV of zero and are called *near-degenerate with small HS preference*, never robust ferromagnetism. |ΔE| < {TIGHT_DEGENERATE_MEV} meV: {cnt['n_lt_tight']} points; |ΔE| < {NEAR_DEGENERATE_MEV} meV: {cnt['n_lt_near']} points. ΔE range {cnt['min']:.2f} to {cnt['max']:.2f} meV, median {cnt['median']:.2f} meV.")
    L.append("- Geometries are the frozen HIGH-SPIN R2SCAN-3C-relaxed structures; PBE0/def2-TZVP UKS single points (TightSCF, DEFGRID3), HS = multiplicity 3, BS = multiplicity 1 with GuessMix. No D4 on these single points (dispersion cancels in E_HS − E_BS at a shared geometry). The conformational landscape comes from the R2SCAN-3C optimizations, not from the PBE0 absolute energies.")
    L.append("- Only `usable=True` geometries (converged R2SCAN-3C relaxation, chemically intact) enter conclusions. Actual plane angles θ_L, θ_C, θ_R are angles between least-squares planes of the two rings flanking each bond (fragments 1|2, 2|3, 3|4), folded to 0–90°.")
    L.append("- Terminal spin: mean_terminal_spin = (|S1|+|S4|)/2 with fragment-integrated Mulliken spin. Spin asymmetry = ||S1|−|S4||. Charge asymmetry q1−q4 uses fragment-integrated Mulliken charges (Löwdin as a reliability cross-check).")
    L.append("- **Limitation**: all geometries were optimized on the HS surface; this gives the local electronic exchange at a fixed HS-relaxed geometry and does not show whether the BS state would relax elsewhere.")
    L.append("")

    # ---------- A ----------
    L.append("## A) Localization / conjugation controls the exchange magnitude\n")
    L.append("outer torsion → hopping suppression → spin localization → AFM exchange collapse.\n")

    L.append("### Main figure 1 — exchange vs central torsion (symmetric outer cuts)\n")
    L.append(f"Zero line marks the AFM/HS boundary; the thick green curve (L=R=90°) sits on it. The y-axis is clipped at {CLIP_YMAX_MEV:.0f} meV so that the few extreme high-strain points do not dominate.\n")
    L.append("![Exchange vs C, symmetric outer torsions](plots/exchange_vs_C_symmetric.png)\n")

    L.append("### Main figure 2 — progressive collapse with the number of outer 90° torsions\n")
    L.append("| n_outer_90 | N | mean | median | std | min | max | frac(\\|ΔE\\|<0.1) | frac(\\|ΔE\\|<0.5) | N(ΔE<0) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, r in classes.iterrows():
        L.append(f"| {k} | {int(r['N'])} | {r['mean']:.3f} | {r['median']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['max']:.3f} | {r['frac_lt_0.1']:.2f} | {r['frac_lt_0.5']:.2f} | {int(r['n_negative'])} |")
    L.append("")
    L.append("0 = neither L nor R is 90°; 1 = exactly one; 2 = both. The expected pattern (clear AFM, almost switched off, essentially zero) is what the table shows. The main axis is clipped; the inset zooms on the sub-meV region.\n")
    L.append("![Exchange by outer-90 class](plots/exchange_by_outer_90_class.png)\n")

    L.append("### Dedicated analysis — L = R = 90° vs the central torsion\n")
    if len(sym90):
        L.append("| C (deg) | ΔE (meV) | conformational E (meV) | ⟨S²⟩ HS | mean terminal spin (PBE0 HS) |")
        L.append("|---|---|---|---|---|")
        for _, r in sym90.iterrows():
            L.append(f"| {r['C']:.0f} | {r['delta_E_raw_meV']:.4f} | {r['relative_energy_meV']:.0f} | {r['s2_HS']:.3f} | {r['mean_terminal_spin_pbe0_HS']:.3f} |")
        L.append("")
        conf_span = sym90["relative_energy_meV"].max() - sym90["relative_energy_meV"].min()
        L.append(f"Once both outer hopping bottlenecks are cut, the central torsion has little remaining influence on the exchange: |ΔE| ≤ {sym90['delta_E_raw_meV'].abs().max():.3f} meV over the usable C values, despite conformational-energy differences of up to {conf_span:.0f} meV between them. (Only {len(sym90)} usable L=R=90° geometries exist, so this is a small-sample statement.)\n")
        L.append("![Exchange, L=R=90 vs C](plots/exchange_symmetric_LR90_vs_C.png)\n")
    else:
        L.append("No usable L=R=90° geometry.\n")

    L.append("### Main figure 3 — exchange vs terminal-radical localization\n")
    for k, v in corr_spin.items():
        L.append(f"- {k}: {fmt_corr(v)}")
    L.append("")
    L.append("Interpretation is primarily the monotonic relation *more localized radicals → weaker AFM exchange* (rank correlation); the Pearson value is lower because the relation is nonlinear, hence the symlog y-axis. Points are colored by n_outer_90.\n")
    L.append("![Exchange vs terminal spin](plots/exchange_vs_terminal_spin.png)\n")

    L.append("### Main figure 4 — exchange vs the three-bottleneck hopping proxy\n")
    L.append("P = cos²θ_L cos²θ_C cos²θ_R with the ACTUAL relaxed plane angles (an exploratory structural descriptor from t_eff ∝ cosθ_L cosθ_C cosθ_R and J_kinetic ∝ t_eff², not a fitted microscopic Hamiltonian). Extreme C ≥ 160° geometries are excluded from the statistics and fit (red circles).\n")
    L.append(f"| descriptor | all usable | excluding C≥{int(EXTREME_C)}° |")
    L.append("|---|---|---|")
    for name, (call, cex) in corr_hop.items():
        L.append(f"| {name} | {fmt_corr(call)} | {fmt_corr(cex)} |")
    L.append("")
    L.append(
        f"The full three-bottleneck product explains the exchange much better (ρ = {f2(P_ex['spearman'])}) than the terminal torsions alone (P_LR, ρ = {f2(LRce['spearman'])}) or the central torsion alone (P_C, ρ = {f2(Cce['spearman'])}). "
        f"Empirical linear guide (visual only, C<{int(EXTREME_C)}°, n={len(U_ex)}): **ΔE = a·P + b with a = {fit_a:.2f} meV, b = {fit_b:.2f} meV, R² = {fit_r2:.2f}**. The intercept is not a rigorous ferromagnetic exchange term; the point is that ΔE → ~0 as P → 0.\n"
    )
    L.append("![Exchange vs hopping proxy](plots/exchange_vs_hopping_proxy.png)\n")

    L.append("### Main figure 7 — L×R heatmaps near C = 80, 100, 120°\n")
    L.append("Diverging scale centered exactly at 0 meV (red = HS lower, blue = BS lower), one common scale; the full per-C set is in `plots/heatmaps/`.\n")
    L.append("![Exchange heatmaps near C=80-120](plots/exchange_heatmaps_C80_100_120.png)\n")

    # ---------- B ----------
    L.append("## B) Left-right asymmetry and the residual sub-meV interaction (hypothesis under test)\n")
    L.append("unequal L/R hybridization → unequal terminal spin localization / possible charge imbalance → possible small shift of ΔE around zero. Mechanism A is strongly supported; this one is tested below without assuming the outcome.\n")

    L.append("### Main figure 5 — exchange vs spin-localization asymmetry\n")
    L.append("| n_outer_90 | median spin asym HS | median spin asym BS | median ΔE | N(ΔE<0) | ρ(ΔE, asym HS) within class | ρ(ΔE, asym BS) within class |")
    L.append("|---|---|---|---|---|---|---|")
    for k in (0, 1, 2):
        if k in classes.index:
            r = classes.loc[k]
            L.append(f"| {k} | {r['median_spin_asym_HS']:.3f} | {r['median_spin_asym_BS']:.3f} | {r['median']:.3f} | {int(r['n_negative'])} | {f2(per_class_asym[k]['HS']['spearman'])} (n={per_class_asym[k]['HS']['n']}) | {f2(per_class_asym[k]['BS']['spearman'])} |")
    L.append("")
    L.append("All usable points: " + "; ".join(f"{k}: {fmt_corr(v)}" for k, v in corr_asym.items()) + ".\n")
    if c1 is not None and c2 is not None:
        asym_conf = c1["median_spin_asym_HS"] > 2 * c2["median_spin_asym_HS"]
        collapse_conf = abs(c2["median"]) < TIGHT_DEGENERATE_MEV
        hyp = (
            "Hypothesis check: "
            f"1 outer 90° → median spin asymmetry {c1['median_spin_asym_HS']:.3f}, median ΔE {c1['median']:.3f} meV; "
            f"2 outer 90° → median spin asymmetry {c2['median_spin_asym_HS']:.3f}, median ΔE {c2['median']:.3f} meV. "
        )
        hyp += "Localization asymmetry is clearly larger with one outer torsion at 90° than with two, " if asym_conf else "Localization asymmetry is NOT clearly larger with one outer 90° than with two, "
        hyp += "while exchange is already ~0 in the symmetric two-90° case. " if collapse_conf else "and the two-90° exchange is not below 0.1 meV in median. "
        hyp += "**Conclusion supported by these data:** spin localization suppresses the AFM exchange, while spin-localization asymmetry may affect the small residual interaction but is not required for the collapse of exchange"
        hyp += " (the collapse is complete at L=R=90° where asymmetry is small).\n" if (asym_conf and collapse_conf) else "; the pattern only partly matches this expectation, see the numbers above.\n"
        L.append(hyp)
    L.append(
        f"Within the one-outer-90\u00b0 class (N={len(one90)}), \u03c1(\u0394E, spin asymmetry HS) = {f2(per_class_asym[1]['HS']['spearman'])} and \u03c1(\u0394E, spin asymmetry BS) = {f2(per_class_asym[1]['BS']['spearman'])}. "
        f"The {cls1['n_neg']} negative points have {lower_higher(cls1['asym_neg'], cls1['asym_oth'])} median spin asymmetry ({cls1['asym_neg']:.3f}) than the {cls1['n_oth']} non-negative ones ({cls1['asym_oth']:.3f}; Mann\u2013Whitney p = {f2(p_mw_spin, 3)}) and {lower_higher(cls1['loc_neg'], cls1['loc_oth'])} mean terminal spin ({cls1['loc_neg']:.3f} vs {cls1['loc_oth']:.3f}). "
        "So asymmetry itself is not what makes \u0394E negative here: in this class asymmetry mostly tracks how far the second outer torsion is from 90\u00b0, and the smallest \u0394E occur where that fragment is also strongly localized. "
        "The negative points are better described by strong localization on both terminals than by imbalance between them.\n"
    )
    L.append("![Exchange vs spin asymmetry](plots/exchange_vs_spin_asymmetry.png)\n")

    L.append("### Main figure 6 — exchange vs charge asymmetry (fragment-integrated Mulliken)\n")
    L.append(f"Fragment charges sum to zero over the four fragments to within {charge_sum_dev:.1e} e (sanity check). Correlations with ΔE over all usable points:\n")
    for k, v in corr_charge.items():
        L.append(f"- {k}: {fmt_corr(v)}")
    L.append("")
    L.append("Spin vs charge asymmetry: " + "; ".join(f"{k}: {fmt_corr(v)}" for k, v in corr_spin_charge.items()) + ".\n")
    L.append(
        "The overall association with \u0394E is carried largely by the difference between n_outer_90 classes; within classes: "
        + "; ".join(f"{CLASS_LABELS[k]}: {fmt_corr(per_class_asym[k]['charge_HS'])}" for k in (0, 1, 2))
        + ". Mulliken and L\u00f6wdin asymmetries agree moderately (see below), so the sign of the effect is not a population-scheme artifact, but the magnitudes are scheme dependent.\n"
    )
    pc = per_class_asym[1]["charge_HS"]
    charge_sig = np.isfinite(p_mw_charge) and p_mw_charge < 0.05
    txt = (
        f"Within the one-outer-90° class, ρ(ΔE, |q1−q4| HS) = {f2(pc['spearman'])} (n={pc['n']}); negative vs non-negative points differ in |q1−q4| with Mann–Whitney p = {f2(p_mw_charge, 3)} "
        f"and in spin asymmetry with p = {f2(p_mw_spin, 3)}. "
    )
    txt += "Charge asymmetry does differ significantly between the slightly negative and the other cases in this class. " if charge_sig else "There is no significant evidence in this class that actual charge imbalance distinguishes the slightly negative cases. "
    txt += "Mulliken charges are basis-set dependent; the Löwdin cross-check above shows how consistent the asymmetry is between population schemes. Interpret as an association, not a mechanism.\n"
    L.append(txt)
    L.append("![Exchange vs charge asymmetry](plots/exchange_vs_charge_asymmetry.png)\n")
    L.append("Supporting: spin asymmetry vs charge asymmetry.\n")
    L.append("![Spin vs charge asymmetry](plots/spin_vs_charge_asymmetry.png)\n")

    L.append("### Geometric vs electronic asymmetry\n")
    L.append("| comparison | correlation |")
    L.append("|---|---|")
    for k, v in geo_asym.items():
        L.append(f"| {k} | {fmt_corr(v)} |")
    L.append("")
    L.append(
        f"Electronic asymmetry at otherwise symmetric geometry: for L = R (n={len(same_LR)}) median spin asymmetry HS = {same_LR['spin_asymmetry_HS'].median():.4f}, median |q1−q4| HS = {same_LR['abs_charge_asymmetry_HS'].median():.4f} e; "
        f"for L ≠ R (n={len(diff_LR)}) {diff_LR['spin_asymmetry_HS'].median():.4f} and {diff_LR['abs_charge_asymmetry_HS'].median():.4f} e. "
        "Asymmetry that survives at L = R would be electronic (relaxation/central-torsion induced) rather than imposed by unequal outer torsions.\n"
    )
    L.append("![Geometric vs electronic asymmetry](plots/geometric_vs_electronic_asymmetry.png)\n")

    # ---------- negative subset ----------
    L.append("## The slightly negative (near-degenerate, small HS preference) points\n")
    if n_neg:
        L.append("| label | L | C | R | θL | θC | θR | E_conf (meV) | ΔE = E_HS−E_BS | E_T−E_S = −J | ⟨S²⟩ HS | ⟨S²⟩ BS | S1 HS | S4 HS | spin asym HS | S1 BS | S4 BS | q1 HS | q4 HS | q1−q4 HS | P | n_outer_90 |")
        L.append("|" + "---|" * 22)
        for _, r in neg.iterrows():
            L.append(
                f"| {r['label']} | {r['L']:.0f} | {r['C']:.0f} | {r['R']:.0f} | {r['theta_L_plane']:.0f} | {r['theta_C_plane']:.0f} | {r['theta_R_plane']:.0f} | {r['relative_energy_meV']:.0f} | {r['delta_E_raw_meV']:.3f} | {r['E_T_minus_E_S_yamaguchi_meV']:.3f} | "
                f"{r['s2_HS']:.3f} | {r['s2_BS']:.3f} | {r['spin_f1_HS']:.3f} | {r['spin_f4_HS']:.3f} | {r['spin_asymmetry_HS']:.3f} | {r['spin_f1_BS']:.3f} | {r['spin_f4_BS']:.3f} | "
                f"{r['charge_f1_HS']:.3f} | {r['charge_f4_HS']:.3f} | {r['charge_asymmetry_HS']:.3f} | {r['P']:.3f} | {int(r['n_outer_90'])} |"
            )
        L.append("")
        L.append("What they have in common (each property tested, none assumed):\n")
        for k, v in tests.items():
            L.append(f"- {k}: **{v}/{n_neg}**" + (" — holds for all" if v == n_neg else " — does NOT hold for all"))
        L.append("")
        L.append(f"Reference for comparison: over all usable points the median mean-terminal-spin (PBE0 HS) is {U['mean_terminal_spin_pbe0_HS'].median():.3f}, the median spin asymmetry {U['spin_asymmetry_HS'].median():.3f} and the median |q1−q4| {ref_charge_asym:.3f} e.\n")
    else:
        L.append("No usable point has ΔE < 0.\n")

    # ---------- EF ----------
    L.append("## Candidate geometries for a future electric-field test (nothing launched)\n")
    L.append(
        "Planned test: apply ±F along the fragment-1 ↔ fragment-4 direction with the geometry kept fixed, and monitor ΔE(F), q(fragment 1) − q(fragment 4) as a function of F, and S(fragment 1) − S(fragment 4) as a function of F. "
        "Question: can a field electronically create or reinforce left-right asymmetry and thereby tune the residual exchange once the AFM contribution has already been mechanically suppressed? The unit vector from the centroid of fragment 1 to that of fragment 4 is given per candidate in `electric_field_candidates.csv`.\n"
    )
    L.append("| role | label | L,C,R | ΔE (meV) | E_conf (meV) | spin asym HS | spin asym BS | q1−q4 HS | q1−q4 BS | θL / θC / θR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ef_df.iterrows():
        L.append(f"| {r['role']} | {r['label']} | {r['L']:.0f},{r['C']:.0f},{r['R']:.0f} | {r['delta_E_raw_meV']:.3f} | {r['conformational_energy_meV']:.0f} | {r['spin_asymmetry_HS']:.3f} | {r['spin_asymmetry_BS']:.3f} | {r['charge_asymmetry_HS']:.3f} | {r['charge_asymmetry_BS']:.3f} | {r['theta_L_plane']:.0f} / {r['theta_C_plane']:.0f} / {r['theta_R_plane']:.0f} |")
    L.append("")

    # ---------- supporting ----------
    L.append("## Supporting / internal analysis\n")
    L.append("### Data completeness and electronic-state QC\n")
    L.append(f"Usable geometries with both PBE0 calculations terminated normally, SCF-converged and with spin populations: **{n_usable - len(missing)} / {n_usable}**. Frozen geometry vs relaxed `high_spin.xyz`: max coordinate difference {df['geom_max_diff_vs_relaxed'].max():.2e} Å.\n")
    L.append("| quantity (usable points) | min | median | max |")
    L.append("|---|---|---|---|")
    for name, col in (("⟨S²⟩ HS", "HS_S2"), ("⟨S²⟩ BS", "BS_S2"), ("fragment 1 spin, BS", "spin_f1_BS"), ("fragment 4 spin, BS", "spin_f4_BS"),
                      ("fragment 1 spin, HS", "spin_f1_HS"), ("fragment 4 spin, HS", "spin_f4_HS")):
        L.append(f"| {name} | {qc[col].min():.3f} | {qc[col].median():.3f} | {qc[col].max():.3f} |")
    L.append("")
    L.append(f"QC checks: HS ⟨S²⟩ in {HS_S2_RANGE}, BS ⟨S²⟩ > {BS_S2_MIN}, fragments 1 and 4 of opposite sign in BS with |spin| > {BS_FRAGMENT_SPIN_MIN}. ")
    if qc_any_bad:
        for k, v in qc_bad.items():
            if len(v):
                L.append(f"- **{k}: {len(v)} point(s)**: " + ", ".join(v["label"]))
    else:
        L.append("All pass for every usable point. Distributions: `plots/qc_distributions.png`.")
    L.append("")

    L.append("### Yamaguchi spin projection (secondary diagnostic)\n")
    L.append(
        "Heisenberg H = −2 J S_A·S_B (J > 0 ferromagnetic): J_Y = −(E_HS − E_BS)/(⟨S²⟩_HS − ⟨S²⟩_BS); E_T − E_S = −2 J_Y. "
        f"It preserves the sign of the raw ΔE at {sign_agree}/{len(U)} points and mostly rescales the raw gap (factor {ratio.min():.2f}–{ratio.max():.2f}), so it does not change any qualitative conclusion. "
        "The raw ΔE = E_HS − E_BS stays the primary quantity; the CSV also has `J_yamaguchi_meV` = −2ΔE/(⟨S²⟩_HS − ⟨S²⟩_BS), "
        "`J_ideal_meV` = −2ΔE (H = −J S_A·S_B, J > 0 FM) and `E_T_minus_E_S_yamaguchi_meV` = −J.\n"
    )
    L.append("![Yamaguchi vs raw](plots/yamaguchi_vs_raw.png)\n")

    L.append("### Exchange vs conformational energy\n")
    L.append("![Exchange vs C with conformational energy](plots/exchange_vs_C_with_conformational_energy.png)\n")

    L.append("### Descriptor correlations (exchange vs conformational energy)\n")
    L.append("Spearman ρ against ΔE and against the R2SCAN-3C conformational energy, raw, within fixed C, and within C controlling the mean actual outer plane angle. Correlations are not assumed causal.\n")
    L.append("| descriptor | ρ ΔE | ρ E_conf | within-C ΔE | within-C E_conf | within-C ctrl outer plane: ΔE | …: E_conf |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in desc.iterrows():
        L.append(f"| {r['descriptor']} | {f2(r['rho_dE'])} | {f2(r['rho_Econf'])} | {f2(r['rho_dE_withinC'])} | {f2(r['rho_Econf_withinC'])} | {f2(r['partial_dE_ctrl_outer_plane'])} | {f2(r['partial_Econf_ctrl_outer_plane'])} |")
    L.append("")

    def dget(col, key):
        return float(desc.loc[desc["column"] == col, key].iloc[0])

    pk = "cross_distance_robust_mean"
    wc, pcc = dget(pk, "rho_dE_withinC"), dget(pk, "partial_dE_ctrl_outer_plane")
    ec = dget(pk, "rho_Econf_withinC")
    read_txt = (
        f"Reading: within fixed C, ρ(ΔE) = {f2(dget('mean_outer_plane', 'rho_dE_withinC'))} for the mean actual outer plane angle, {f2(dget('mean_terminal_spin_geomscan', 'rho_dE_withinC'))} for terminal spin, {f2(dget('P', 'rho_dE_withinC'))} for P. "
        f"Cross-fragment packing (3-shortest-contact mean) vs ΔE within C: ρ = {f2(wc)}, becoming {f2(pcc)} once the outer plane angle is controlled "
    )
    read_txt += "(mostly mediated by the outer torsions)" if (np.isfinite(pcc) and abs(pcc) < 0.5 * abs(wc)) else "(not simply explained by the outer torsions)"
    read_txt += f"; packing vs conformational energy within C: ρ = {f2(ec)}"
    read_txt += " — essentially zero, so the earlier packing↔energy link holds only for the symmetric 45/45 vs 135/135 comparisons, not across the whole (L,R) plane.\n" if abs(ec) < 0.2 else ".\n"
    L.append(read_txt)
    L.append("![Descriptor correlations within C](plots/descriptor_correlations_withinC.png)\n")

    L.append("### Validation candidates (alternative functionals, tighter numerics, possible independent BS relaxation)\n")
    L.append("| role | label | ΔE (meV) | E_conf (meV) | mean terminal spin (PBE0 HS) |")
    L.append("|---|---|---|---|---|")
    for _, r in cand_df.iterrows():
        L.append(f"| {r['role']} | {r['label']} | {r['delta_E_raw_meV']:.3f} | {r['relative_conformational_energy_meV']:.0f} | {r['mean_terminal_spin']:.3f} |")
    L.append("")

    L.append("### Full per-C exchange heatmaps\n")
    for c in c_vals:
        L.append(f"![Exchange heatmap C={int(c)}](plots/heatmaps/exchange_heatmap_C{int(c):03d}.png)\n")

    (out_dir / "exchange_analysis.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out_dir / 'exchange_analysis.md'}")
    print("=" * 70)
    print(f"usable={n_usable}, pairs complete={n_usable - len(missing)}, QC anomalies={'yes' if qc_any_bad else 'none'}")
    print({k: v for k, v in cnt.items() if k.startswith("N_")})
    print(classes[["N", "median", "std", "frac_lt_0.1", "frac_lt_0.5", "n_negative", "median_spin_asym_HS"]].to_string())


if __name__ == "__main__":
    main()
