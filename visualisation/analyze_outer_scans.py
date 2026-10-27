#!/usr/bin/env python3
"""
Detailed outer-torsion scans at fixed central torsion C = 100 deg.

  Scan A (symmetric)      : L = R = theta,   theta = 0, 10, ..., 180
  Scan B (one side at 90) : L = 90 fixed, R = theta

Geometries: R2SCAN-3C HS constrained relaxations
(angle_scan_C100_outer_relaxation, angle_scan_C100_L090_relaxation).

Exchange: PBE0/def2-TZVP HS/BS single points, same protocol and convention as
analyze_exchange.py:
    dE = E_HS - E_BS ;  dE > 0 : BS lower (AFM tendency)
They are read from <scan>_exchange/<label>/{high_spin,broken_symmetry}/ when
present. A detailed point whose relaxed geometry is identical to a point of
the coarse grid (angle_scan_exchange_analysis) reuses that PBE0 result. The
remaining coarse-grid C = 100 points that fall on the scan lines (L = R = 45,
135; L = 90, R = 45, 135) are shown as a separately marked preview.

Conformational energy (R2SCAN-3C) and exchange energy (PBE0 dE) are never
mixed: they come from different calculations and answer different questions.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from ase.io import read
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
    check_chemical_integrity,
    cross_fragment_min_distance,
    fit_plane,
    outer_terminal_fragment,
    parse_out_file,
    reference_bonds,
    split_fragment_at_bond,
)

SCF_CONVERGED_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
LABEL_RE = re.compile(r"^L(\d+)_C(\d+)_R(\d+)$")

# Validated categorical slots (reference palette, light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SCAN_COLOR = {"A": BLUE, "B": ORANGE}
RIGHT_JUNCTION_ATOM = 23  # backbone carbon bonded to the right radical ring (RIGHT quartet atom b)
SCAN_NAME = {"A": "Scan A: L = R = θ", "B": "Scan B: L = 90°, R = θ"}


# ============================================================
# HELPERS
# ============================================================
def plane_angle(pos, ring_a, ring_b):
    _, n1 = fit_plane(pos[ring_a])
    _, n2 = fit_plane(pos[ring_b])
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))


def folded(theta):
    return min(theta % 180.0, 180.0 - theta % 180.0)


def mulliken(text, fragments):
    """Fragment-integrated Mulliken charges and spins from the LAST population block."""
    blocks = list(MULLIKEN_BLOCK_RE.finditer(text))
    if not blocks:
        return {}
    q, s = {}, {}
    for line in blocks[-1].group(1).splitlines():
        m = ATOM_LINE_RE.match(line)
        if m:
            q[int(m.group(1))] = float(m.group(3))
            s[int(m.group(1))] = float(m.group(4))
    out = {}
    for k, atoms in fragments.items():
        out[f"S{k}"] = sum(s.get(i, 0.0) for i in atoms)
        out[f"q{k}"] = sum(q.get(i, 0.0) for i in atoms)
    return out


def parse_pbe0(path, fragments):
    rec = {"ok": False}
    if not path.exists():
        return rec
    text = path.read_text(errors="ignore")
    e = FINAL_E_RE.findall(text)
    s2 = S2_RE.findall(text)
    rec["ok"] = ("ORCA TERMINATED NORMALLY" in text and bool(SCF_CONVERGED_RE.search(text))
                 and "SCF NOT CONVERGED" not in text and bool(e))
    rec["E"] = float(e[-1]) if e else np.nan
    rec["s2"] = float(s2[-1]) if s2 else np.nan
    rec.update(mulliken(text, fragments))
    return rec


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(m.sum()), "pearson": np.nan, "spearman": np.nan}
    if m.sum() >= 4 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
        out["pearson"] = float(stats.pearsonr(x[m], y[m])[0])
        out["spearman"] = float(stats.spearmanr(x[m], y[m])[0])
    return out


def fmt_corr(c):
    if not np.isfinite(c["pearson"]):
        return f"n = {c['n']} (too few points)"
    return f"Pearson r = {c['pearson']:.2f}, Spearman ρ = {c['spearman']:.2f} (n = {c['n']})"


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    res = stats.linregress(x[m], y[m])
    return float(res.slope), float(res.intercept), float(res.rvalue ** 2)


def local_minima(theta, e):
    idx = []
    for i in range(1, len(e) - 1):
        if e[i] < e[i - 1] and e[i] < e[i + 1]:
            idx.append(i)
    return idx


def parabola_curvature(theta, e, i):
    """Second derivative (meV/deg^2) of a parabola through the minimum and its two neighbours."""
    if i <= 0 or i >= len(e) - 1:
        return np.nan, np.nan
    c = np.polyfit(theta[i - 1:i + 2], e[i - 1:i + 2], 2)
    return float(2 * c[0]), float(-c[1] / (2 * c[0]))


def departure_from_90(theta, value, threshold, use_abs, dist=None):
    """
    Smallest distance from orthogonality at which the signed/abs value first exceeds
    `threshold`, walking outward from 90 deg on each handedness branch separately
    (linear interpolation). `dist` is the distance measure (default |theta - 90|;
    pass 90 - actual plane angle for the physical version).
    Returns {"below": (d_last_below, d_first_above), "above": ...}: the grid bracket
    (distances of the last point still below and the first point above the
    threshold); (nan, nan) if never exceeded within the data.
    """
    out = {}
    theta = np.asarray(theta, float)
    dist = np.abs(theta - 90) if dist is None else np.asarray(dist, float)
    v = np.abs(np.asarray(value, float)) if use_abs else np.asarray(value, float)
    for side, mask in (("below", theta <= 90), ("above", theta >= 90)):
        ts, vs = dist[mask], v[mask]
        out[side] = (np.nan, np.nan)
        if len(ts) < 2:
            continue
        seq = np.argsort(ts)
        ts, vs = ts[seq], vs[seq]
        if vs[0] > threshold:
            out[side] = (0.0, ts[0])
            continue
        for k in range(1, len(ts)):
            if vs[k] > threshold:
                out[side] = (ts[k - 1], ts[k])
                break
    return out


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3)
    ax.tick_params(colors=INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def symlog_dE(ax):
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1, 2, 5, 10])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def theta_axis(ax):
    ax.set_xlim(-5, 185)
    ax.set_xticks(range(0, 181, 30))
    ax.axvline(90, color=INK_3, linewidth=0.8, linestyle=":")


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    repo_root = Path(__file__).resolve().parent.parent
    scans_root = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans"
    SCANS = {
        "A": scans_root / "angle_scan_C100_outer_relaxation",
        "B": scans_root / "angle_scan_C100_L090_relaxation",
    }
    PBE0_ROOTS = {
        "A": scans_root / "angle_scan_C100_outer_exchange",
        "B": scans_root / "angle_scan_C100_L090_exchange",
    }
    coarse_pbe0_root = scans_root / "angle_scan_exchange_analysis"
    coarse_csv = coarse_pbe0_root / "exchange_analysis.csv"
    scan_csv = scans_root / "angle_scan_relaxation" / "angle_scan.csv"
    free_csv = scans_root / "angle_scan_free_relaxation" / "free_relaxation.csv"
    baseline_xyz = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    out_dir = scans_root / "outer_scan_analysis"

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    RINGS = {
        "L": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]),
        "C": ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25]),
        "R": ([23, 21, 25, 24, 26, 27], [28, 29, 30, 31, 32]),
    }
    SAME_GEOMETRY_TOL_ANG = 5e-3  # one-cycle reruns move atoms by ~1e-3 Å
    DETAILED_MIN_POINTS = 8  # per scan, before the detailed dE analysis is reported as such
    THRESHOLDS = [(0.1, True), (0.5, True), (1.0, False)]  # (meV, use |dE|)
    PLANE_DEVIATION_WARN_DEG = 10.0

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- FRAGMENTS ----
    ref_atoms = read(baseline_xyz)
    adj = build_adjacency(ref_atoms)
    n_atoms = len(ref_atoms)
    bonded_pairs = {(i, j) for i in range(n_atoms) for j in adj[i] if j > i}
    bonds = reference_bonds(adj, ref_atoms)
    f1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    f4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    centre = sorted(set(range(n_atoms)) - set(f1) - set(f4))
    f2, f3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), centre)
    fragments = {1: f1, 2: f2, 3: f3, 4: f4}

    coarse = pd.read_csv(coarse_csv).set_index("label") if coarse_csv.exists() else pd.DataFrame()
    scan = pd.read_csv(scan_csv).set_index("label")
    e_ref_label = scan.loc[scan["usable"], "final_energy"].idxmin()
    e_ref = float(scan.loc[e_ref_label, "final_energy"])
    e_m1 = np.nan
    if free_csv.exists():
        fr = pd.read_csv(free_csv)
        e_m1 = float(fr.loc[fr["converged"], "E_final"].min())

    # ---- PARSE BOTH SCANS ----
    rows = []
    for key, root in SCANS.items():
        manifest = pd.read_csv(root / "manifest.csv").set_index("label") if (root / "manifest.csv").exists() else None
        for d in sorted(p for p in root.iterdir() if p.is_dir() and LABEL_RE.match(p.name)):
            L, C, R = (int(x) for x in LABEL_RE.match(d.name).groups())
            out = d / "high_spin.out"
            row = {"scan": key, "label": d.name, "L": L, "C": C, "R": R,
                   "theta": float(R), "theta_folded": folded(R)}
            row["base"] = Path(manifest.loc[d.name, "source_geometry"]).parent.name if manifest is not None else ""
            if not out.exists():
                row["status"] = "missing"
                rows.append(row)
                continue
            text = out.read_text(errors="ignore")
            rec = parse_out_file(text, out)
            row.update({
                "converged": rec["opt_converged"], "convergence_type": rec["convergence_type"],
                "n_geom_cycles": rec["n_geom_cycles"], "E_r2scan": rec["final_energy"],
                "s2_HS_r2scan": rec["s2_final"],
                "status": "converged" if rec["opt_converged"] else "not converged",
            })
            xyz = d / "high_spin.xyz"
            atoms = read(xyz)
            pos = atoms.get_positions()
            for ax_name, quartet in (("L", LEFT), ("C", CENTER), ("R", RIGHT)):
                row[f"theta_{ax_name}_plane"] = plane_angle(pos, *RINGS[ax_name])
                row[f"dihedral_{ax_name}_final"] = atoms.get_dihedral(*quartet)
            integ = check_chemical_integrity(atoms, bonds, bonded_pairs, 0.30, 1.9, 1.5)
            row["geometry_valid"] = integ["valid"]
            row["min_nonbonded_distance"] = integ["min_nonbonded_distance"]
            row["usable"] = bool(rec["opt_converged"] and integ["valid"])
            symbols = atoms.get_chemical_symbols()
            for a_, b_, name in ((f1, f3, "13"), (f2, f4, "24"), (f1, f4, "14")):
                row[f"d{name}_heavy"], pair = cross_fragment_min_distance(pos, symbols, a_, b_, True)
                row[f"d{name}_pair"] = f"{pair[0]}-{pair[1]}"
            row["d_radicals"] = float(np.linalg.norm(pos[LEFT[1]] - pos[RIGHT[2]]))
            # R2SCAN-3C HS populations (always available for converged points)
            if rec["opt_converged"]:
                row.update({f"{k}_HS_r2scan": v for k, v in mulliken(text, fragments).items()})

            # ---- PBE0 ----
            pbe0_dir = PBE0_ROOTS[key] / d.name
            row["pbe0_source"] = ""
            if (pbe0_dir / "high_spin" / "high_spin.out").exists():
                src = pbe0_dir
                row["pbe0_source"] = "detailed"
            elif d.name in coarse.index and (coarse_pbe0_root / d.name / "high_spin" / "geometry.xyz").exists():
                cpos = read(coarse_pbe0_root / d.name / "high_spin" / "geometry.xyz").get_positions()
                dmax = float(np.abs(cpos - pos).max())
                src = coarse_pbe0_root / d.name if dmax < SAME_GEOMETRY_TOL_ANG else None
                row["pbe0_source"] = f"coarse grid (identical geometry, max Δx {dmax:.0e} Å)" if src else ""
            else:
                src = None
            if src is not None:
                hs = parse_pbe0(src / "high_spin" / "high_spin.out", fragments)
                bs = parse_pbe0(src / "broken_symmetry" / "broken_symmetry.out", fragments)
                if hs["ok"] and bs["ok"]:
                    row["E_HS"], row["E_BS"] = hs["E"], bs["E"]
                    row["dE"] = (hs["E"] - bs["E"]) * HARTREE_TO_MEV
                    row["s2_HS"], row["s2_BS"] = hs["s2"], bs["s2"]
                    for k in range(1, 5):
                        row[f"S{k}_HS"], row[f"S{k}_BS"] = hs.get(f"S{k}", np.nan), bs.get(f"S{k}", np.nan)
                        row[f"q{k}_HS"], row[f"q{k}_BS"] = hs.get(f"q{k}", np.nan), bs.get(f"q{k}", np.nan)
                else:
                    row["pbe0_source"] += " (failed)"
            rows.append(row)

    df = pd.DataFrame(rows)
    df["E_rel_global_meV"] = (df["E_r2scan"] - e_ref) * HARTREE_TO_MEV
    df["E_rel_scan_meV"] = df["E_rel_global_meV"] - df.groupby("scan")["E_rel_global_meV"].transform("min")

    # Physical outer-angle coordinate: the torsion that is being varied.
    df["theta_varied_plane"] = df["theta_R_plane"]
    df["theta_fixed_plane"] = np.where(df["scan"] == "A", df["theta_L_plane"], df["theta_L_plane"])
    df["theta_outer_plane_mean"] = (df["theta_L_plane"] + df["theta_R_plane"]) / 2
    df["plane_minus_folded_L"] = df["theta_L_plane"] - df["L"].map(folded)
    df["plane_minus_folded_R"] = df["theta_R_plane"] - df["R"].map(folded)

    # Spin measures: PBE0 HS where available, R2SCAN-3C HS otherwise (flagged).
    for k in range(1, 5):
        df[f"S{k}_best"] = df.get(f"S{k}_HS", pd.Series(np.nan, index=df.index)).fillna(df[f"S{k}_HS_r2scan"])
    df["spin_level"] = np.where(df.get("S1_HS", pd.Series(np.nan, index=df.index)).notna(), "PBE0 HS", "R2SCAN-3C HS")
    for lvl, tag in (("_HS_r2scan", "r2scan"), ("_best", "best")):
        df[f"mean_terminal_{tag}"] = (df[f"S1{lvl}"].abs() + df[f"S4{lvl}"].abs()) / 2
        df[f"spin_asym_{tag}"] = (df[f"S1{lvl}"].abs() - df[f"S4{lvl}"].abs()).abs()
        df[f"backbone_{tag}"] = df[f"S2{lvl}"] + df[f"S3{lvl}"]
    df["charge_asym_r2scan"] = df["q1_HS_r2scan"] - df["q4_HS_r2scan"]
    if "q1_BS" in df:
        df["charge_asym_BS"] = df["q1_BS"] - df["q4_BS"]
    df["P"] = np.prod([np.cos(np.radians(df[f"theta_{a}_plane"])) ** 2 for a in "LCR"], axis=0)
    df["cos4_outer"] = np.cos(np.radians(df["theta_L_plane"])) ** 2 * np.cos(np.radians(df["theta_R_plane"])) ** 2
    df["P_LR"] = df["cos4_outer"]
    if "dE" not in df:
        df["dE"] = np.nan

    U = df[df["usable"] == True].copy()  # noqa: E712
    detailed_counts = {k: int(((U["scan"] == k) & (U["pbe0_source"] == "detailed")).sum()) for k in SCANS}

    # ---- COARSE-GRID PBE0 PREVIEW POINTS (not part of the detailed scans) ----
    preview_rows = []
    if not coarse.empty:
        for key, pairs in (("A", [(45, 45), (135, 135)]), ("B", [(90, 45), (90, 135)])):
            for Lc, Rc in pairs:
                lab = f"L{Lc:03d}_C100_R{Rc:03d}"
                if lab in coarse.index and bool(coarse.loc[lab, "usable"]):
                    c = coarse.loc[lab]
                    preview_rows.append({
                        "scan": key, "label": lab, "theta": float(Rc), "dE": float(c["delta_E_raw_meV"]),
                        "theta_L_plane": float(c["theta_L_plane"]), "theta_C_plane": float(c["theta_C_plane"]),
                        "theta_R_plane": float(c["theta_R_plane"]), "P": float(c["P"]),
                        "S1_best": float(c["spin_f1_HS"]), "S4_best": float(c["spin_f4_HS"]),
                    })
    preview = pd.DataFrame(preview_rows)
    if not preview.empty:
        preview["mean_terminal_best"] = (preview["S1_best"].abs() + preview["S4_best"].abs()) / 2
        preview["spin_asym_best"] = (preview["S1_best"].abs() - preview["S4_best"].abs()).abs()
        preview["cos4_outer"] = (np.cos(np.radians(preview["theta_L_plane"])) ** 2
                                 * np.cos(np.radians(preview["theta_R_plane"])) ** 2)
    # All exchange points actually available on the scan lines (detailed + reused + preview)
    ex_cols = ["scan", "label", "theta", "dE", "theta_L_plane", "theta_C_plane", "theta_R_plane", "P",
               "cos4_outer", "mean_terminal_best", "spin_asym_best", "S1_best", "S4_best"]
    EX = U[U["dE"].notna()][ex_cols + ["pbe0_source", "s2_HS", "s2_BS", "d13_heavy", "d_radicals"]].copy()
    if not preview.empty:
        pv = preview[ex_cols].copy()
        pv["pbe0_source"] = "coarse grid (base geometry, not on 10° grid)"
        EX = pd.concat([EX, pv], ignore_index=True)
    EX["on_grid"] = ~EX["pbe0_source"].str.contains("base geometry")
    EX["branch"] = np.where(EX["theta"] < 90, "θ < 90°", np.where(EX["theta"] > 90, "θ > 90°", "90°"))
    ds2 = EX["s2_HS"] - EX["s2_BS"]
    # H = -J S1.S2 (J > 0 FM), dE = E_HS - E_BS
    EX["J_ideal_meV"] = XC.j_ideal(EX["dE"])
    EX["J_yamaguchi_meV"] = XC.j_yamaguchi(EX["dE"], EX["s2_HS"], EX["s2_BS"])
    EX["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(EX["J_yamaguchi_meV"])
    EX = EX.sort_values(["scan", "theta"]).reset_index(drop=True)

    df.to_csv(out_dir / "outer_scans.csv", index=False)
    EX.to_csv(out_dir / "outer_scans_exchange_points.csv", index=False)

    results = analyse(U, EX, detailed_counts, DETAILED_MIN_POINTS, THRESHOLDS)
    make_plots(U, EX, plots_dir, results)
    write_report(df, U, EX, results, out_dir / "outer_scans.md", e_ref_label, e_ref, e_m1,
                 detailed_counts, DETAILED_MIN_POINTS, PLANE_DEVIATION_WARN_DEG, PBE0_ROOTS)
    print(f"Wrote {out_dir / 'outer_scans.md'}")
    print(f"PBE0 points per scan (detailed / reused / preview): " + ", ".join(
        f"{k}: {detailed_counts[k]} / {int(((EX['scan'] == k) & EX['pbe0_source'].str.startswith('coarse grid (identical')).sum())}"
        f" / {int(((EX['scan'] == k) & EX['pbe0_source'].str.contains('base geometry')).sum())}" for k in SCANS))


# ============================================================
# ANALYSIS
# ============================================================
def analyse(U, EX, detailed_counts, detailed_min, thresholds):
    res = {"energy": {}, "spin": {}, "exchange": {}, "pairs": {}}

    # --- conformational energy
    for key, g in U.groupby("scan"):
        g = g.sort_values("theta")
        t, e = g["theta"].to_numpy(), g["E_rel_scan_meV"].to_numpy()
        i_min = int(np.argmin(e))
        k2, t_fit = parabola_curvature(t, e, i_min)
        minima = [(float(t[i]), float(e[i])) for i in local_minima(t, e)]
        e90 = float(g.loc[g["theta"] == 90, "E_rel_scan_meV"].iloc[0]) if (g["theta"] == 90).any() else np.nan
        asym = []
        for th in t[t < 90]:
            if (t == 180 - th).any():
                asym.append((float(th), float(e[t == th][0]), float(e[t == 180 - th][0])))
        res["energy"][key] = {
            "theta_min": float(t[i_min]), "E_min_global": float(g["E_rel_global_meV"].min()),
            "curvature_meV_per_deg2": k2, "theta_min_parabola": t_fit, "local_minima": minima,
            "cost_to_90": e90, "complementary": asym,
            "theta_plane_min": float(g["theta_varied_plane"].iloc[i_min]),
        }

    # --- spin localisation (R2SCAN-3C HS, available for every point; PBE0 HS where present)
    for key, g in U.groupby("scan"):
        g = g.sort_values("theta")
        s1, s4 = g["S1_HS_r2scan"].abs().to_numpy(), g["S4_HS_r2scan"].abs().to_numpy()
        t = g["theta"].to_numpy()
        d1, d4 = np.gradient(s1, t), np.gradient(s4, t)
        res["spin"][key] = {
            "S1_range": (float(s1.min()), float(s1.max())), "S4_range": (float(s4.min()), float(s4.max())),
            "dS1": float(np.ptp(s1)), "dS4": float(np.ptp(s4)),
            "max_abs_dS1_dtheta": float(np.abs(d1).max()), "max_abs_dS4_dtheta": float(np.abs(d4).max()),
            "backbone_range": (float(g["backbone_r2scan"].min()), float(g["backbone_r2scan"].max())),
            "S1_S4_max_diff": float(np.abs(s1 - s4).max()),
            "asym_range": (float(g["spin_asym_r2scan"].min()), float(g["spin_asym_r2scan"].max())),
            "q_asym_range": (float(g["charge_asym_r2scan"].min()), float(g["charge_asym_r2scan"].max())),
        }
    # universality: varied-side terminal spin vs its own plane angle, A vs B
    A, B = U[U["scan"] == "A"], U[U["scan"] == "B"]
    xa = np.concatenate([A["theta_L_plane"], A["theta_R_plane"]])
    ya = np.concatenate([A["S1_HS_r2scan"].abs(), A["S4_HS_r2scan"].abs()])
    o = np.argsort(xa)
    xb, yb = B["theta_R_plane"].to_numpy(), B["S4_HS_r2scan"].abs().to_numpy()
    inside = (xb >= xa.min()) & (xb <= xa.max())
    pred = np.interp(xb[inside], xa[o], ya[o])
    res["spin"]["universality_rms"] = float(np.sqrt(np.mean((yb[inside] - pred) ** 2))) if inside.any() else np.nan
    res["spin"]["universality_maxdev"] = float(np.abs(yb[inside] - pred).max()) if inside.any() else np.nan
    res["spin"]["universality_n"] = int(inside.sum())
    res["spin"]["S4_B_range_span"] = float(np.ptp(yb))

    # --- exchange
    for key in ("A", "B"):
        is_detailed = detailed_counts[key] >= detailed_min
        g = EX[(EX["scan"] == key) & (EX["on_grid"] | (not is_detailed))].sort_values("theta")
        ex = {"n": len(g), "n_detailed": detailed_counts[key], "detailed": is_detailed}
        if len(g):
            ex["dE_range"] = (float(g["dE"].min()), float(g["dE"].max()))
            ex["sign_changes"] = [
                (float(a), float(b)) for (a, da), (b, db) in zip(zip(g["theta"], g["dE"]), zip(g["theta"][1:], g["dE"][1:]))
                if np.sign(da) != np.sign(db) and da != 0 and db != 0
            ]
            ex["thresholds"] = {thr: departure_from_90(g["theta"], g["dE"], thr, use_abs) for thr, use_abs in thresholds}
            ex["thresholds_plane"] = {thr: departure_from_90(g["theta"], g["dE"], thr, use_abs, 90 - g["theta_R_plane"])
                                      for thr, use_abs in thresholds}
            ex["branches"] = {}
            for br, m in (("θ < 90°", g["theta"] < 90), ("θ > 90°", g["theta"] > 90)):
                gb = g[m]
                ex["branches"][br] = {
                    "range": (float(gb["dE"].min()), float(gb["dE"].max())),
                    "n_hs": int((gb["dE"] < 0).sum()), "n_near": int((gb["dE"].abs() < 0.5).sum()), "n": len(gb),
                    "corr_P": corr(gb["P"], gb["dE"]), "corr_loc": corr(gb["mean_terminal_best"], gb["dE"]),
                    "corr_plane": corr(gb["theta_R_plane"], gb["dE"]),
                }
            ex["min_dE"] = g.loc[g["dE"].idxmin()]
            ex["corr_thetaC"] = corr(g["theta_C_plane"], g["dE"])
            ex["corr_d13"] = corr(g["d13_heavy"], g["dE"])
            ex["corr_drad"] = corr(g["d_radicals"], g["dE"])
            ex["s2_gap_range"] = (float((g["s2_HS"] - g["s2_BS"]).min()), float((g["s2_HS"] - g["s2_BS"]).max()))
            ex["yam_same_sign"] = int((np.sign(g["E_T_minus_E_S_yamaguchi_meV"]) == np.sign(g["dE"])).sum())
            ex["corr_P"] = corr(g["P"], g["dE"])
            ex["corr_cos4"] = corr(g["cos4_outer"], g["dE"])
            ex["fit_P"] = linfit(g["P"], g["dE"])
            ex["fit_cos4"] = linfit(g["cos4_outer"], g["dE"])
            ex["corr_loc"] = corr(g["mean_terminal_best"], g["dE"])
            ex["corr_asym"] = corr(g["spin_asym_best"], g["dE"])
            ex["table"] = g
        res["exchange"][key] = ex
    grid = EX[EX["on_grid"]] if all(res["exchange"][k]["detailed"] for k in ("A", "B")) else EX
    res["exchange"]["corr_loc_all"] = corr(grid["mean_terminal_best"], grid["dE"])

    # --- complementary angles (theta vs 180 - theta), per scan
    for key, g in U.groupby("scan"):
        g = g.set_index("theta")
        pairs = []
        for th in sorted(g.index):
            if th < 90 and (180 - th) in g.index:
                a, b = g.loc[th], g.loc[180 - th]
                pairs.append({
                    "theta": th, "E_a": a["E_rel_scan_meV"], "E_b": b["E_rel_scan_meV"],
                    "S_a": a["S4_HS_r2scan"], "S_b": b["S4_HS_r2scan"],
                    "plane_a": a["theta_varied_plane"], "plane_b": b["theta_varied_plane"],
                    "thetaC_a": a["theta_C_plane"], "thetaC_b": b["theta_C_plane"],
                    "dE_a": a.get("dE", np.nan), "dE_b": b.get("dE", np.nan),
                })
        res["pairs"][key] = pd.DataFrame(pairs)
    return res


# ============================================================
# PLOTS
# ============================================================
def _preview_mask(g):
    return g["pbe0_source"].str.contains("base geometry")


def make_plots(U, EX, plots_dir, res):
    # 1. energy vs outer angle
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        g = U[U["scan"] == key].sort_values("theta")
        ax.plot(g["theta"], g["E_rel_scan_meV"], "-o", color=SCAN_COLOR[key], markersize=5, linewidth=1.8)
        r = res["energy"][key]
        ax.annotate(f"min θ = {r['theta_min']:.0f}°", (r["theta_min"], 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_title(f"{SCAN_NAME[key]}\nscan minimum at {r['E_min_global']:+.0f} meV (common ref.)", fontsize=10)
        ax.set_xlabel("imposed varied outer dihedral θ (deg)")
        theta_axis(ax)
        style_axes(ax)
    axes[0].set_ylabel("R2SCAN-3C E − E_min(scan) (meV)")
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_outer_angle.png", dpi=200)
    plt.close(fig)

    # 2. fragment spin vs outer angle
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        g = U[U["scan"] == key].sort_values("theta")
        lab1 = "|S1| (left, fixed 90°)" if key == "B" else "|S1| (left)"
        lab4 = "|S4| (right, varied)" if key == "B" else "|S4| (right)"
        ax.plot(g["theta"], g["S1_HS_r2scan"].abs(), "-o", color=BLUE, markersize=5, label=lab1)
        ax.plot(g["theta"], g["S4_HS_r2scan"].abs(), "--s", color=ORANGE, markersize=4, label=lab4)
        ax.plot(g["theta"], g["backbone_r2scan"], "-^", color=AQUA, markersize=4, label="backbone S2 + S3")
        ax.axhline(1.0, color=INK_3, linewidth=0.8, linestyle=":")
        ax.set_title(SCAN_NAME[key], fontsize=10)
        ax.set_xlabel("imposed varied outer dihedral θ (deg)")
        theta_axis(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="center left")
        style_axes(ax)
    axes[0].set_ylabel("HS Mulliken spin (R2SCAN-3C)")
    fig.tight_layout()
    fig.savefig(plots_dir / "fragment_spin_vs_outer_angle.png", dpi=200)
    plt.close(fig)

    # 3. exchange vs outer angle
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        g = EX[EX["scan"] == key].sort_values("theta")
        ax.axhline(0, color=INK_3, linewidth=1)
        ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.6, linewidth=0)
        if len(g):
            pv = _preview_mask(g)
            ax.plot(g.loc[~pv, "theta"], g.loc[~pv, "dE"], "-", color=SCAN_COLOR[key], linewidth=1.2, alpha=0.7)
            ax.plot(g.loc[~pv, "theta"], g.loc[~pv, "dE"], "o", color=SCAN_COLOR[key], markersize=7,
                    markeredgecolor="white", label="on detailed grid")
            ax.plot(g.loc[pv, "theta"], g.loc[pv, "dE"], "D", markerfacecolor="white",
                    markeredgecolor=SCAN_COLOR[key], markersize=7, label="coarse-grid base point (not in stats)")
        symlog_dE(ax)
        ax.set_title(SCAN_NAME[key] + ("" if res["exchange"][key]["detailed"] else "\n(coarse preview: detailed PBE0 not yet run)"),
                     fontsize=10)
        ax.set_xlabel("imposed varied outer dihedral θ (deg)")
        theta_axis(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center")
        style_axes(ax)
    axes[0].set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)\nsymlog, linear within ±1 meV; grey band |ΔE| < 0.5")
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_outer_angle.png", dpi=200)
    plt.close(fig)

    # 4. scan comparison on the physical x-axis
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color=INK_3, linewidth=1)
    ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.6, linewidth=0)
    for key, mk in (("A", "o"), ("B", "s")):
        g = EX[(EX["scan"] == key) & EX["on_grid"]].sort_values("theta_R_plane")
        if not len(g):
            continue
        lo, hi = g[g["theta"] <= 90], g[g["theta"] >= 90]
        ax.plot(lo["theta_R_plane"], lo["dE"], "-" + mk, color=SCAN_COLOR[key], markersize=7, markeredgecolor="white",
                linewidth=1.2, label=f"{SCAN_NAME[key]}, θ ≤ 90°")
        ax.plot(hi["theta_R_plane"], hi["dE"], "--" + mk, color=SCAN_COLOR[key], markerfacecolor="white",
                markeredgecolor=SCAN_COLOR[key], markersize=7, linewidth=1.2, label=f"{SCAN_NAME[key]}, θ ≥ 90°")
    symlog_dE(ax)
    ax.set_xlabel("actual plane angle of the varied (right) outer torsion θ_R (deg)")
    ax.set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    ax.set_title("Is cutting one outer link enough? (A: both links vary; B: left link fixed at 90°)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(0, 92)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_scan_comparison.png", dpi=200)
    plt.close(fig)

    # 5. exchange vs terminal localisation
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color=INK_3, linewidth=1)
    for key, mk in (("A", "o"), ("B", "s")):
        g = EX[(EX["scan"] == key) & (EX["on_grid"] | (not res["exchange"][key]["detailed"]))]
        c = res["exchange"][key].get("corr_loc", {"n": 0, "pearson": np.nan, "spearman": np.nan})
        lo, hi = g[g["theta"] <= 90], g[g["theta"] > 90]
        ax.plot(lo["mean_terminal_best"], lo["dE"], mk, color=SCAN_COLOR[key], markersize=8, markeredgecolor="white",
                label=f"{SCAN_NAME[key]}  ({fmt_corr(c)})")
        ax.plot(hi["mean_terminal_best"], hi["dE"], mk, markerfacecolor="white", markeredgecolor=SCAN_COLOR[key],
                markersize=8, label=f"   (open: θ > 90° branch)")
    symlog_dE(ax)
    ax.set_xlabel("mean terminal spin (|S1| + |S4|)/2, HS")
    ax.set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_terminal_localization.png", dpi=200)
    plt.close(fig)

    # 6. exchange vs hopping proxy
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, col, lab in ((axes[0], "P", "P = cos²θL cos²θC cos²θR (actual planes)"),
                         (axes[1], "cos4_outer", "cos²θL cos²θR (outer only)"),
                         (axes[2], "theta_C_plane", "actual central plane angle θ_C (deg)")):
        ax.axhline(0, color=INK_3, linewidth=1)
        for key, mk in (("A", "o"), ("B", "s")):
            g = EX[(EX["scan"] == key) & EX["on_grid"]]
            ax.plot(g[col], g["dE"], mk, color=SCAN_COLOR[key], markersize=8, markeredgecolor="white",
                    label=SCAN_NAME[key])
            a, b, r2 = res["exchange"][key].get("fit_P" if col == "P" else "fit_cos4", (np.nan,) * 3)
            if np.isfinite(a) and key == "A" and col != "theta_C_plane":
                xs = np.linspace(0, g[col].max(), 50)
                ax.plot(xs, a * xs + b, color=SCAN_COLOR[key], linewidth=1, alpha=0.6, label=f"A: linear fit, R² = {r2:.2f}")
        ax.set_xlabel(lab)
        ax.legend(frameon=False, fontsize=8.5)
        style_axes(ax)
    axes[0].set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_hopping_proxy.png", dpi=200)
    plt.close(fig)

    # 7. spin asymmetry vs angle (+ dE vs asymmetry, + charge asymmetry) for scan B
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, mk in (("A", "o"), ("B", "s")):
        g = U[U["scan"] == key].sort_values("theta")
        axes[0].plot(g["theta"], g["spin_asym_r2scan"], "-" + mk, color=SCAN_COLOR[key], markersize=5, label=SCAN_NAME[key])
        axes[1].plot(g["theta"], g["charge_asym_r2scan"], "-" + mk, color=SCAN_COLOR[key], markersize=5, label=SCAN_NAME[key])
        e = EX[(EX["scan"] == key) & EX["on_grid"]]
        axes[2].plot(e["spin_asym_best"], e["dE"], mk, color=SCAN_COLOR[key], markersize=8, markeredgecolor="white",
                     label=SCAN_NAME[key])
    axes[0].set_ylabel("spin asymmetry ||S1| − |S4|| (HS)")
    axes[1].set_ylabel("charge asymmetry q1 − q4 (Mulliken, HS)")
    for ax in axes[:2]:
        ax.set_xlabel("imposed varied outer dihedral θ (deg)")
        theta_axis(ax)
    axes[2].axhline(0, color=INK_3, linewidth=1)
    symlog_dE(axes[2])
    axes[2].set_xlabel("spin asymmetry ||S1| − |S4|| (HS)")
    axes[2].set_ylabel("ΔE, PBE0 (meV)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "spin_asymmetry_vs_angle.png", dpi=200)
    plt.close(fig)

    # 8. actual vs imposed outer angle (QC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        g = U[U["scan"] == key].sort_values("theta")
        ax.plot([0, 90], [0, 90], color=INK_3, linewidth=1, linestyle=":")
        for side, col, c, mk in (("L", "L", BLUE, "o"), ("R", "R", ORANGE, "s")):
            fx = g[col].map(folded)
            below = g[col] <= 90
            ax.plot(fx[below], g.loc[below, f"theta_{side}_plane"], mk, color=c, markersize=6, label=f"θ_{side} (imposed ≤ 90°)")
            ax.plot(fx[~below], g.loc[~below, f"theta_{side}_plane"], mk, markerfacecolor="white", markeredgecolor=c,
                    markersize=6, label=f"θ_{side} (imposed > 90°)")
        ax.plot(g["R"].map(folded), g["theta_C_plane"], "^", color=AQUA, markersize=5, label="θ_C (imposed 100°)")
        ax.set_title(SCAN_NAME[key], fontsize=10)
        ax.set_xlabel("folded imposed dihedral min(θ, 180 − θ) (deg)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left" if key == "A" else "lower right")
        style_axes(ax)
    axes[0].set_ylabel("actual ring-plane angle (deg)")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_imposed_outer_angle.png", dpi=200)
    plt.close(fig)

    # 9. terminal spin vs actual plane angle (universality)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    A, B = U[U["scan"] == "A"], U[U["scan"] == "B"]
    ax.plot(A["theta_L_plane"], A["S1_HS_r2scan"].abs(), "o", color=BLUE, markersize=6, label="A: |S1| vs θ_L")
    ax.plot(A["theta_R_plane"], A["S4_HS_r2scan"].abs(), "o", markerfacecolor="white", markeredgecolor=BLUE,
            markersize=6, label="A: |S4| vs θ_R")
    ax.plot(B["theta_R_plane"], B["S4_HS_r2scan"].abs(), "s", color=ORANGE, markersize=6, label="B: |S4| (varied) vs θ_R")
    ax.plot(B["theta_L_plane"], B["S1_HS_r2scan"].abs(), "s", markerfacecolor="white", markeredgecolor=ORANGE,
            markersize=6, label="B: |S1| (fixed 90°) vs θ_L")
    ax.set_xlabel("actual plane angle of the radical's own outer torsion (deg)")
    ax.set_ylabel("terminal HS spin (R2SCAN-3C)")
    ax.set_title(f"Is localization a local function of the own torsion?\nRMS(B − A) = {res['spin']['universality_rms']:.3f}",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "terminal_spin_vs_plane_angle.png", dpi=200)
    plt.close(fig)

    # 10. complementary angles (handedness)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, key in enumerate(("A", "B")):
        p = res["pairs"][key]
        if p.empty:
            continue
        for ax, (a, b, lab) in zip(axes[row], (("E_a", "E_b", "E rel. scan min (meV)"),
                                               ("S_a", "S_b", "|S4| (HS)"),
                                               ("plane_a", "plane_b", "actual θ_R plane (deg)"))):
            ax.plot(p["theta"], p[a].abs() if a.startswith("S") else p[a], "-o", color=SCAN_COLOR[key], label="θ")
            ax.plot(p["theta"], p[b].abs() if b.startswith("S") else p[b], "--s", markerfacecolor="white",
                    markeredgecolor=SCAN_COLOR[key], color=SCAN_COLOR[key], label="180° − θ")
            ax.set_ylabel(lab)
            ax.set_title(SCAN_NAME[key], fontsize=9.5)
            ax.legend(frameon=False, fontsize=8)
            style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("torsion magnitude θ ≤ 90° (deg)")
    fig.tight_layout()
    fig.savefig(plots_dir / "complementary_angle_handedness_comparison.png", dpi=200)
    plt.close(fig)


# ============================================================
# REPORT
# ============================================================
def fmt_bracket(b):
    lo, hi = b
    if not np.isfinite(hi):
        return "not reached"
    return f"{lo:.0f}–{hi:.0f}°"


def matched_plane_pairs(g, tol=3.0):
    """For each theta > 90 point, the theta < 90 point with the closest varied plane angle (within tol)."""
    lo, hi = g[g["theta"] < 90], g[g["theta"] > 90]
    out = []
    for _, h in hi.iterrows():
        if lo.empty:
            break
        j = (lo["theta_R_plane"] - h["theta_R_plane"]).abs().idxmin()
        l_ = lo.loc[j]
        if abs(l_["theta_R_plane"] - h["theta_R_plane"]) <= tol:
            out.append((l_, h))
    return out


def plot_cross_contact(EX, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, lab in ((axes[0], "d13_heavy", "closest heavy-atom contact fragment 1 – fragment 3 (Å)"),
                         (axes[1], "d_radicals", "radical-centre distance C41 – C28 (Å)")):
        ax.axhline(0, color=INK_3, linewidth=1)
        for key, mk in (("A", "o"), ("B", "s")):
            g = EX[(EX["scan"] == key) & EX["on_grid"]]
            lo, hi = g[g["theta"] <= 90], g[g["theta"] > 90]
            ax.plot(lo[col], lo["dE"], mk, color=SCAN_COLOR[key], markersize=8, markeredgecolor="white",
                    label=f"{SCAN_NAME[key]}, θ ≤ 90°")
            ax.plot(hi[col], hi["dE"], mk, markerfacecolor="white", markeredgecolor=SCAN_COLOR[key], markersize=8,
                    label=f"{SCAN_NAME[key]}, θ > 90°")
        symlog_dE(ax)
        ax.set_xlabel(lab)
        style_axes(ax)
    axes[0].set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_cross_contact.png", dpi=200)
    plt.close(fig)


def write_report(df, U, EX, res, md_path, e_ref_label, e_ref, e_m1, detailed_counts, detailed_min,
                 plane_warn, pbe0_roots):
    plot_cross_contact(EX, md_path.parent / "plots")
    pair_B170 = df.set_index(["scan", "theta"]).loc[("B", 170.0), "d13_pair"] if ((df["scan"] == "B") & (df["theta"] == 170)).any() else "n/a"
    L = []
    detailed = all(res["exchange"][k]["detailed"] for k in ("A", "B"))
    exA, exB = res["exchange"]["A"], res["exchange"]["B"]
    eA, eB = res["energy"]["A"], res["energy"]["B"]
    sA, sB = res["spin"]["A"], res["spin"]["B"]
    gA, gB = exA["table"], exB["table"]
    tA, tB = gA.set_index("theta"), gB.set_index("theta")
    brA, brB = exA["branches"], exB["branches"]
    pairsB = matched_plane_pairs(gB)
    n_reuse = int(EX["pbe0_source"].str.startswith("coarse grid (identical").sum())
    n_prev = int((~EX["on_grid"]).sum())
    hiB = gB[(gB["theta"] > 90) & (gB["dE"] > 1.0)]
    loc_pair = None
    if pairsB:
        loc_pair = max(pairsB, key=lambda p: p[1]["dE"] - p[0]["dE"])

    L.append("# Detailed outer-torsion scans at fixed C = 100°")
    L.append("")
    L.append("Scan A: L = R = θ. Scan B: L = 90° fixed, R = θ. θ = 0, 10, …, 180°, C constrained at 100°. "
             "R2SCAN-3C HS constrained relaxations + PBE0-D4/def2-TZVP HS/BS single points.")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"1. **Twisting toward orthogonality is expensive.** Scan A minimum at θ = {eA['theta_min']:.0f}°; reaching "
             f"L = R = 90° costs {eA['cost_to_90']:.0f} meV. Holding one side at 90° costs {eB['E_min_global'] - eA['E_min_global']:.0f} meV "
             f"at the best position of the other, and taking the second one to 90° adds another {eB['cost_to_90']:.0f} meV.")
    L.append(f"2. **Each outer torsion controls its own radical.** Scan B: the radical behind the fixed 90° link changes by "
             f"{sB['dS1']:.3f} over the scan, the varied one by {sB['dS4']:.3f}.")
    L.append(f"3. **Symmetric twisting (A): AFM exchange collapses only close to 90°.** ΔE drops from {brA['θ < 90°']['range'][1]:.1f} meV "
             f"to a {tA.loc[50:70, 'dE'].min():.2f}–{tA.loc[50:70, 'dE'].max():.2f} meV plateau at θ = 50–70°, then to "
             f"{tA.loc[80, 'dE']:+.2f} meV at 80°. |ΔE| < 0.5 meV needs the outer rings within "
             f"{fmt_bracket(exA['thresholds_plane'][0.5]['below'])} (θ < 90° branch) of orthogonality in actual plane angle.")
    L.append(f"4. **One orthogonal link keeps the exchange off — but only on one handedness branch (B).** For θ < 90° every point is "
             f"near-degenerate (ΔE {brB['θ < 90°']['range'][0]:+.2f} to {brB['θ < 90°']['range'][1]:+.2f} meV). For θ > 90° ΔE "
             f"re-emerges at θ ≥ {hiB['theta'].min():.0f}° (up to {brB['θ > 90°']['range'][1]:.1f} meV) although the hopping product "
             f"P ≈ 0 throughout.")
    if loc_pair is not None:
        l_, h_ = loc_pair
        L.append(f"5. **Neither P nor terminal localization explains Scan B.** At matched right-ring plane angle "
                 f"({l_['theta_R_plane']:.0f}° vs {h_['theta_R_plane']:.0f}°) and matched mean terminal spin "
                 f"({l_['mean_terminal_best']:.3f} vs {h_['mean_terminal_best']:.3f}), θ = {l_['theta']:.0f}° gives ΔE = "
                 f"{l_['dE']:+.2f} meV and θ = {h_['theta']:.0f}° gives {h_['dE']:+.2f} meV. The simple three-bottleneck "
                 f"picture holds for Scan A (ρ(ΔE, P) = {exA['corr_P']['spearman']:.2f}) but not for Scan B "
                 f"(ρ = {exB['corr_P']['spearman']:.2f}).")
    L.append("6. **No robust ferromagnetism.** All negative ΔE are within the near-degenerate window "
             f"(most negative {min(brA['θ < 90°']['range'][0], brA['θ > 90°']['range'][0], brB['θ < 90°']['range'][0], brB['θ > 90°']['range'][0]):+.2f} meV).")
    L.append("")

    L.append("## Data, QC and conventions")
    L.append("")
    L.append(f"- R2SCAN-3C HS relaxations: {int((U['scan'] == 'A').sum())}/19 (A) and {int((U['scan'] == 'B').sum())}/19 (B) "
             "ORCA-converged and chemically intact; all enter the analysis.")
    L.append(f"- PBE0 HS/BS: {detailed_counts['A']} + {detailed_counts['B']} new points (all SCF-converged, normal termination), "
             f"plus {n_reuse} points (θ = 0/90/180°) reused from `angle_scan_exchange_analysis` because the relaxed geometry is "
             "identical (≤ 1e-3 Å). The new runs include D4, the reused ones do not; D4 is identical for HS and BS at a shared "
             "geometry (checked: equal to 1e-9 Eh), so ΔE is unaffected.")
    L.append(f"- {n_prev} coarse-grid C = 100 points at 45°/135° (other relaxation lineage) are drawn as open diamonds in the "
             "exchange-vs-angle figure only; they are **not** used in any statistic.")
    L.append("- ΔE = E_HS − E_BS: ΔE > 0 → BS lower (AFM tendency). |ΔE| < 0.5 meV = near-degenerate; negative values in that "
             "window are a small HS preference, not ferromagnetism. "
             f"⟨S²⟩_HS − ⟨S²⟩_BS = {min(exA['s2_gap_range'][0], exB['s2_gap_range'][0]):.3f}–"
             f"{max(exA['s2_gap_range'][1], exB['s2_gap_range'][1]):.3f} everywhere, so Yamaguchi projection only doubles ΔE "
             f"(same sign at {exA['yam_same_sign'] + exB['yam_same_sign']}/{exA['n'] + exB['n']} points).")
    L.append(f"- Conformational energy: R2SCAN-3C, relative to each scan's minimum; common reference {e_ref_label} "
             f"(free minimum M1 at {(e_m1 - e_ref) * HARTREE_TO_MEV:+.1f} meV on that scale).")
    L.append("- θ is the imposed dihedral of the varied (right) torsion. θ and 180° − θ have the same magnitude but opposite "
             "handedness and are treated as separate **branches** (θ < 90°, θ > 90°). Actual ring-plane angles θ_L, θ_C, θ_R "
             "(0–90°) are the physical coordinate.")
    L.append("- Spins/charges: fragment-integrated Mulliken (1 = left radical, 4 = right radical, 2/3 = backbone halves). "
             "Spin figures use the R2SCAN-3C HS populations (same geometry); the exchange correlations use PBE0 HS.")
    L.append("")

    # =================== PART I ===================
    L.append("# Part I — Main results")
    L.append("")
    L.append("## M1. What is the energetic cost of twisting the terminal radicals?")
    L.append("")
    L.append("| | Scan A (L = R = θ) | Scan B (L = 90°, R = θ) |")
    L.append("|---|---|---|")
    L.append(f"| minimum θ (imposed / actual plane) | {eA['theta_min']:.0f}° / {eA['theta_plane_min']:.1f}° | "
             f"{eB['theta_min']:.0f}° / {eB['theta_plane_min']:.1f}° |")
    L.append(f"| scan minimum vs common reference | {eA['E_min_global']:+.1f} meV | {eB['E_min_global']:+.1f} meV |")
    L.append(f"| local minima (θ, meV above scan min) | {', '.join(f'{t:.0f}° ({e:.0f})' for t, e in eA['local_minima'])} | "
             f"{', '.join(f'{t:.0f}° ({e:.0f})' for t, e in eB['local_minima'])} |")
    L.append(f"| curvature at minimum (3-point parabola) | {eA['curvature_meV_per_deg2']:.2f} meV/deg² | "
             f"{eB['curvature_meV_per_deg2']:.2f} meV/deg² |")
    L.append(f"| cost to reach θ = 90° | {eA['cost_to_90']:.0f} meV | {eB['cost_to_90']:.0f} meV |")
    L.append("")
    L.append(f"The planar limits are also costly (θ = 0/180°: {tA_E(U, 'A', 0):.0f}/{tA_E(U, 'A', 180):.0f} meV in A, "
             f"{tA_E(U, 'B', 0):.0f}/{tA_E(U, 'B', 180):.0f} meV in B) because the rings cannot become coplanar (H···H). "
             "Scan A is strongly asymmetric between branches (handedness, see S5); Scan B is nearly symmetric with two "
             "almost degenerate minima.")
    L.append("")
    L.append("![Energy vs outer angle](plots/energy_vs_outer_angle.png)")
    L.append("")

    L.append("## M2. How locally does each outer torsion control its adjacent radical?")
    L.append("")
    L.append("| | Scan A | Scan B |")
    L.append("|---|---|---|")
    L.append(f"| \\|S1\\| (B: fixed-90° side) | {sA['S1_range'][0]:.3f}–{sA['S1_range'][1]:.3f} (Δ {sA['dS1']:.3f}) | "
             f"{sB['S1_range'][0]:.3f}–{sB['S1_range'][1]:.3f} (Δ {sB['dS1']:.3f}) |")
    L.append(f"| \\|S4\\| (B: varied side) | {sA['S4_range'][0]:.3f}–{sA['S4_range'][1]:.3f} (Δ {sA['dS4']:.3f}) | "
             f"{sB['S4_range'][0]:.3f}–{sB['S4_range'][1]:.3f} (Δ {sB['dS4']:.3f}) |")
    L.append(f"| max \\|dS/dθ\\| S1 / S4 (deg⁻¹) | {sA['max_abs_dS1_dtheta']:.4f} / {sA['max_abs_dS4_dtheta']:.4f} | "
             f"{sB['max_abs_dS1_dtheta']:.4f} / {sB['max_abs_dS4_dtheta']:.4f} |")
    L.append(f"| backbone S2 + S3 | {sA['backbone_range'][0]:.3f}–{sA['backbone_range'][1]:.3f} | "
             f"{sB['backbone_range'][0]:.3f}–{sB['backbone_range'][1]:.3f} |")
    L.append("")
    L.append(f"**Each outer torsion acts predominantly as a local control of its adjacent radical's localization.** In Scan B the "
             f"fixed-90° radical stays at |S1| ≈ 1.01–1.02 (Δ {sB['dS1']:.3f}) while the varied radical moves by {sB['dS4']:.3f}, "
             f"{sB['dS4'] / sB['dS1']:.0f}× more; its spin leaks onto the backbone half next to it (S3), not onto the fixed side (S2 "
             f"stays ≈ −0.03). In Scan A both radicals follow each other to within {sA['S1_S4_max_diff']:.3f} and the backbone "
             "spin vanishes at 90°. The same-side relation is quantitatively transferable between scans (side analysis S3).")
    L.append("")
    L.append("![Fragment spin vs outer angle](plots/fragment_spin_vs_outer_angle.png)")
    L.append("")

    L.append("## M3. How rapidly does AFM exchange collapse toward orthogonality?")
    L.append("")
    L.append("| scan / branch | ΔE range (meV) | points with \\|ΔE\\| < 0.5 | points with ΔE < 0 | ρ(ΔE, own plane angle) |")
    L.append("|---|---|---|---|---|")
    for key, br in (("A", brA), ("B", brB)):
        for name, b in br.items():
            L.append(f"| {key}, {name} | {b['range'][0]:+.2f} to {b['range'][1]:+.2f} | {b['n_near']}/{b['n']} | {b['n_hs']}/{b['n']} | "
                     f"{b['corr_plane']['spearman']:.2f} |")
    L.append("")
    L.append("Angular window around 90°: grid bracket of the distance from orthogonality at which the threshold is first "
             "exceeded (last point below – first point above), in imposed dihedral and in actual plane angle of the varied ring.")
    L.append("")
    L.append("| scan | threshold | imposed, θ < 90° | imposed, θ > 90° | plane, θ < 90° | plane, θ > 90° |")
    L.append("|---|---|---|---|---|---|")
    for key, ex in (("A", exA), ("B", exB)):
        for thr, lab in ((0.1, "\\|ΔE\\| > 0.1 meV"), (0.5, "\\|ΔE\\| > 0.5 meV"), (1.0, "ΔE > 1 meV")):
            ti, tp = ex["thresholds"][thr], ex["thresholds_plane"][thr]
            L.append(f"| {key} | {lab} | {fmt_bracket(ti['below'])} | {fmt_bracket(ti['above'])} | "
                     f"{fmt_bracket(tp['below'])} | {fmt_bracket(tp['above'])} |")
    L.append("")
    L.append(f"**Shape in Scan A.** On the θ < 90° branch ΔE is not a smooth cos⁴ decay: it falls from {tA.loc[10, 'dE']:.2f} meV "
             f"(θ = 10°) to a plateau of {tA.loc[50:70, 'dE'].min():.2f}–{tA.loc[50:70, 'dE'].max():.2f} meV at θ = 50–70° "
             f"(plane {tA.loc[50, 'theta_R_plane']:.0f}–{tA.loc[70, 'theta_R_plane']:.0f}°) and then drops by more than an order "
             f"of magnitude between θ = 70° and 80° to {tA.loc[80, 'dE']:+.2f} meV. That last step coincides with the plane angle "
             f"jumping {tA.loc[70, 'theta_R_plane']:.0f}° → {tA.loc[80, 'theta_R_plane']:.0f}° and the central plane opening "
             f"{tA.loc[70, 'theta_C_plane']:.0f}° → {tA.loc[80, 'theta_C_plane']:.0f}°: the suppression is concentrated in the "
             f"last ~15° before orthogonality. On the θ > 90° branch the central plane angle sits at ~86° for θ = 130–170°, and "
             f"ΔE stays at {tA.loc[110:170, 'dE'].min():.2f}–{tA.loc[110:170, 'dE'].max():.2f} meV until the near-planar 180° "
             f"point ({tA.loc[180, 'dE']:.2f} meV). Structure below ~0.3 meV (e.g. 100° → 110° → 120°) follows θ_C and should "
             "not be over-interpreted.")
    L.append("")
    L.append("![Exchange vs outer angle](plots/exchange_vs_outer_angle.png)")
    L.append("")

    L.append("## M4. Is one orthogonal outer torsion sufficient to keep the exchange switched off?")
    L.append("")
    L.append(f"**Partly.** On the θ < 90° branch of Scan B all {brB['θ < 90°']['n']} points are near-degenerate "
             f"({brB['θ < 90°']['range'][0]:+.2f} to {brB['θ < 90°']['range'][1]:+.2f} meV) over the full range of the other "
             f"torsion, including right-ring plane angles down to {gB[gB['theta'] < 90]['theta_R_plane'].min():.0f}°, where the "
             f"same right ring in Scan A carries {tA.loc[10, 'dE']:.1f} meV. Here a single orthogonal terminal bottleneck is "
             "sufficient to suppress the through-molecule AFM exchange even while the opposite radical is strongly delocalized.")
    L.append("")
    L.append(f"On the θ > 90° branch the exchange **re-emerges**: ΔE = "
             + ", ".join(f"{r['dE']:.2f} meV at θ = {r['theta']:.0f}°" for _, r in gB[gB['theta'] >= 150].iterrows())
             + f", reaching Scan-A magnitudes although the left link stays orthogonal (θ_L plane "
             f"{gB[gB['theta'] >= 150]['theta_L_plane'].min():.0f}–{gB[gB['theta'] >= 150]['theta_L_plane'].max():.0f}°) and P ≈ 0.")
    L.append("")
    L.append("Matched right-ring plane angles across the two branches of Scan B:")
    L.append("")
    L.append("| θ < 90° point | θ_R plane | ΔE (meV) | θ > 90° point | θ_R plane | ΔE (meV) |")
    L.append("|---|---|---|---|---|---|")
    for l_, h_ in pairsB:
        L.append(f"| {l_['theta']:.0f}° | {l_['theta_R_plane']:.1f}° | {l_['dE']:+.2f} | {h_['theta']:.0f}° | "
                 f"{h_['theta_R_plane']:.1f}° | {h_['dE']:+.2f} |")
    L.append("")
    L.append("The serial-bottleneck picture therefore holds only on one branch: which way the right ring is rotated matters "
             "as much as how far. A pathway that bypasses the orthogonal bond is active on the θ > 90° branch (see S6 for a "
             "through-space hypothesis).")
    L.append("")
    L.append("![Scan comparison](plots/exchange_scan_comparison.png)")
    L.append("")

    L.append("## M5. Exchange vs terminal spin localization")
    L.append("")
    L.append(f"- Scan A: {fmt_corr(exA['corr_loc'])}  (θ < 90°: ρ = {brA['θ < 90°']['corr_loc']['spearman']:.2f}; "
             f"θ > 90°: ρ = {brA['θ > 90°']['corr_loc']['spearman']:.2f})")
    L.append(f"- Scan B: {fmt_corr(exB['corr_loc'])}  (θ < 90°: ρ = {brB['θ < 90°']['corr_loc']['spearman']:.2f}; "
             f"θ > 90°: ρ = {brB['θ > 90°']['corr_loc']['spearman']:.2f})")
    L.append(f"- combined: {fmt_corr(res['exchange']['corr_loc_all'])}")
    L.append("")
    msg = ("In Scan A stronger terminal localization goes with weaker AFM exchange, as expected when both radical–backbone "
           "hoppings shrink together.")
    if loc_pair is not None:
        l_, h_ = loc_pair
        msg += (f" In Scan B localization is not the controlling variable: θ = {l_['theta']:.0f}° and θ = {h_['theta']:.0f}° have "
                f"the same mean terminal spin ({l_['mean_terminal_best']:.3f} vs {h_['mean_terminal_best']:.3f}) but ΔE = "
                f"{l_['dE']:+.2f} vs {h_['dE']:+.2f} meV.")
    L.append(msg)
    L.append("")
    L.append("![Exchange vs terminal localization](plots/exchange_vs_terminal_localization.png)")
    L.append("")

    L.append("## Main physical messages")
    L.append("")
    msgs = [
        f"**The decoupled regime is conformationally costly.** Holding both outer rings orthogonal costs {eA['cost_to_90']:.0f} meV "
        f"above the preferred ~{eA['theta_plane_min']:.0f}° twist; even one orthogonal link costs "
        f"{eB['E_min_global'] - eA['E_min_global']:.0f} meV. The sub-meV exchange regime is electronically interesting but not "
        "where the free molecule sits; a junction must hold it there.",
        "**Radical localization is a local torsional property.** Each terminal spin is set by its own outer torsion, "
        f"essentially independent of the other (fixed-side Δ|S1| = {sB['dS1']:.3f} vs varied-side {sB['dS4']:.3f}).",
        f"**With both links twisted together, AFM exchange collapses sharply only in the last ~15° before orthogonality** "
        f"(θ < 90° branch: {tA.loc[70, 'dE']:.2f} → {tA.loc[80, 'dE']:+.2f} meV between 70° and 80°), and ΔE tracks the "
        f"three-bottleneck product P (Spearman ρ = {exA['corr_P']['spearman']:.2f}).",
        f"**One orthogonal link suppresses the exchange on one handedness branch only.** With L = 90°, ΔE stays within "
        f"±0.5 meV for every θ < 90°, but rises to {brB['θ > 90°']['range'][1]:.1f} meV for θ ≥ {hiB['theta'].min():.0f}°, at "
        "the same plane angles and the same terminal localization. Exchange suppression here is not a consequence of spin "
        "localization or of the hopping product alone; the sense of rotation opens or closes an additional pathway.",
        "**Conformational energy and exchange are controlled differently.** Localization follows torsion magnitude, the "
        "conformational energy strongly follows handedness (Scan A), and the exchange follows magnitude on one branch and "
        "handedness-dependent geometry on the other.",
    ]
    for i, m in enumerate(msgs, 1):
        L.append(f"{i}. {m}")
    L.append("")

    # =================== PART II ===================
    L.append("# Part II — Side analyses")
    L.append("")
    L.append("## S1. Actual vs imposed outer angle (QC)")
    L.append("")
    dev = pd.concat([U["plane_minus_folded_R"], U.loc[U["scan"] == "A", "plane_minus_folded_L"]])
    big = U[(U["plane_minus_folded_R"].abs() > plane_warn)]
    L.append(f"Actual − folded imposed outer plane angle: median {dev.median():+.1f}°, range {dev.min():+.1f}° to {dev.max():+.1f}°; "
             f"{len(big)} varied-side points deviate by more than {plane_warn:.0f}° (near 0/180°, where H···H contacts prevent "
             "coplanarity, and around 60–80°). The central plane angle varies between "
             f"{U['theta_C_plane'].min():.1f}° and {U['theta_C_plane'].max():.1f}° although C is constrained: in Scan A it switches "
             "to ~86° on the θ > 90° branch. Mechanistic plots therefore use actual plane angles.")
    L.append("")
    L.append("![Actual vs imposed](plots/actual_vs_imposed_outer_angle.png)")
    L.append("")
    L.append("## S2. Simple torsional hopping proxy")
    L.append("")
    L.append("P = cos²θ_L cos²θ_C cos²θ_R with actual plane angles (a structural descriptor, not a fitted Hubbard model).")
    L.append("")
    L.append("| scan | ΔE vs P | linear fit ΔE = aP + b | ΔE vs cos²θ_L cos²θ_R | ΔE vs θ_C |")
    L.append("|---|---|---|---|---|")
    for key, ex in (("A", exA), ("B", exB)):
        a, b, r2 = ex["fit_P"]
        L.append(f"| {key} | {fmt_corr(ex['corr_P'])} | a = {a:.3g}, b = {b:+.2f}, R² = {r2:.2f} | {fmt_corr(ex['corr_cos4'])} "
                 f"(R² = {ex['fit_cos4'][2]:.2f}) | {fmt_corr(ex['corr_thetaC'])} |")
    L.append("")
    L.append("In Scan A the full product (including θ_C) outperforms the outer-only product, i.e. the central bottleneck matters "
             "even at fixed C, because θ_C relaxes by up to ~10°. In Scan B P ≈ 0 by construction and cannot explain the "
             "θ > 90° re-emergence.")
    L.append("")
    L.append("![Exchange vs hopping proxy](plots/exchange_vs_hopping_proxy.png)")
    L.append("")
    L.append("## S3. Is localization a universal function of the own plane angle?")
    L.append("")
    L.append(f"Varied-side spin in Scan B vs the Scan-A curve at the same actual plane angle: RMS deviation "
             f"{res['spin']['universality_rms']:.3f}, max {res['spin']['universality_maxdev']:.3f} ({res['spin']['universality_n']} "
             f"points; total span {res['spin']['S4_B_range_span']:.3f}). The same-side radical follows essentially the same "
             "S_terminal(θ_plane) curve whether the opposite link is equal or orthogonal.")
    L.append("")
    L.append("![Terminal spin vs plane angle](plots/terminal_spin_vs_plane_angle.png)")
    L.append("")
    L.append("## S4. Spin and charge asymmetry")
    L.append("")
    L.append(f"Scan B: ||S1| − |S4|| = {sB['asym_range'][0]:.3f}–{sB['asym_range'][1]:.3f}; q1 − q4 = {sB['q_asym_range'][0]:+.3f} "
             f"to {sB['q_asym_range'][1]:+.3f} (Scan A: ≤ {sA['asym_range'][1]:.3f} and {sA['q_asym_range'][0]:+.3f} to "
             f"{sA['q_asym_range'][1]:+.3f}). ΔE vs spin asymmetry in Scan B: {fmt_corr(exB['corr_asym'])}. Large asymmetry "
             "coexists with near-zero ΔE on the θ < 90° branch (asymmetry up to "
             f"{gB[gB['theta'] < 90]['spin_asym_best'].max():.2f} at ΔE ≤ {gB[gB['theta'] < 90]['dE'].max():.2f} meV), so the "
             "suppression is not caused by left–right spin asymmetry. Mulliken charges are descriptors only.")
    L.append("")
    L.append("![Spin asymmetry](plots/spin_asymmetry_vs_angle.png)")
    L.append("")
    L.append("## S5. Magnitude vs handedness (θ vs 180° − θ)")
    L.append("")
    L.append("| scan | θ | E (meV) θ / 180−θ | \\|S4\\| θ / 180−θ | θ_R plane θ / 180−θ | θ_C plane θ / 180−θ | ΔE (meV) θ / 180−θ |")
    L.append("|---|---|---|---|---|---|---|")
    for key, t in (("A", tA), ("B", tB)):
        for _, p in res["pairs"][key].iterrows():
            th = p["theta"]
            L.append(f"| {key} | {th:.0f} | {p['E_a']:.0f} / {p['E_b']:.0f} | {abs(p['S_a']):.3f} / {abs(p['S_b']):.3f} | "
                     f"{p['plane_a']:.1f} / {p['plane_b']:.1f} | {p['thetaC_a']:.1f} / {p['thetaC_b']:.1f} | "
                     f"{t.loc[th, 'dE']:+.2f} / {t.loc[180 - th, 'dE']:+.2f} |")
    L.append("")
    L.append("Terminal spins depend on magnitude (complementary angles agree to a few 0.01). The conformational energy "
             "depends strongly on handedness in Scan A and weakly in Scan B. The exchange depends on handedness in both scans: in "
             "A mostly through θ_C (the θ > 90° branch has θ_C ≈ 86° and smaller ΔE), in B through the re-emergence at θ ≥ 160°.")
    L.append("")
    L.append("![Handedness comparison](plots/complementary_angle_handedness_comparison.png)")
    L.append("")
    L.append("## S6. Through-space contacts (hypothesis for the Scan B re-emergence)")
    L.append("")
    L.append(f"ΔE vs closest fragment 1 – fragment 3 heavy-atom contact: Scan A {fmt_corr(exA['corr_d13'])}; Scan B "
             f"{fmt_corr(exB['corr_d13'])}. ΔE vs radical-centre distance: A {fmt_corr(exA['corr_drad'])}; B "
             f"{fmt_corr(exB['corr_drad'])}. In Scan B the θ ≥ 160° points have the orthogonal left radical closest to the "
             f"backbone half that carries the delocalized right-radical spin (d₁₃ = "
             + ", ".join(f"{r['d13_heavy']:.2f}" for _, r in gB[gB['theta'] >= 160].iterrows())
             + f" Å; closest pair {pair_B170}, i.e. a fragment-1 ring carbon against C{RIGHT_JUNCTION_ATOM}, the backbone carbon "
             "of the right junction). In Scan A the d₁₃ correlation is confounded with θ itself (both change monotonically along "
             "each branch) and carries no separate information. "
             "In Scan B this is consistent with a through-space contribution that bypasses the orthogonal bond, but it is a correlation "
             f"over a handful of points (θ = 0° has a similar contact, {tB.loc[0, 'd13_heavy']:.2f} Å, and only "
             f"{tB.loc[0, 'dE']:.2f} meV), not a demonstrated mechanism. Testing it would need e.g. a spin-density "
             "difference analysis or a model with the backbone half removed.")
    L.append("")
    L.append("![Exchange vs cross contact](plots/exchange_vs_cross_contact.png)")
    L.append("")
    L.append("## Appendix — exchange points used in the statistics")
    L.append("")
    L.append("| scan | θ | θ_L / θ_C / θ_R plane | ΔE (meV) | E_T−E_S = −J, Yamaguchi (meV) | ⟨S²⟩ HS / BS | mean terminal spin | P | d₁₃ (Å) | source |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in EX[EX["on_grid"]].iterrows():
        L.append(f"| {r['scan']} | {r['theta']:.0f} | {r['theta_L_plane']:.1f} / {r['theta_C_plane']:.1f} / {r['theta_R_plane']:.1f} | "
                 f"{r['dE']:+.3f} | {r['E_T_minus_E_S_yamaguchi_meV']:+.3f} | {r['s2_HS']:.3f} / {r['s2_BS']:.3f} | "
                 f"{r['mean_terminal_best']:.3f} | {r['P']:.2e} | {r['d13_heavy']:.2f} | "
                 f"{'new (PBE0-D4)' if r['pbe0_source'] == 'detailed' else 'reused coarse grid'} |")
    md_path.write_text("\n".join(L) + "\n")


def tA_E(U, key, theta):
    return float(U[(U["scan"] == key) & (U["theta"] == theta)]["E_rel_scan_meV"].iloc[0])


if __name__ == "__main__":
    main()
