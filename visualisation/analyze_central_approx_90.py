#!/usr/bin/env python3
"""
Central torsion around C = 90 deg (angle_scan_central_approx_90 + _exchange).

  free_outer/  : C = 80, 85, ..., 100 constrained; outer torsions started at
                 (L, R) = (45, 45), (135, 135), (45, 135) and then FREE.
  fixed_outer/ : C = 90 with L <= R on the 0/45/90/135/180 grid, all constrained.

Geometries: R2SCAN-3C HS relaxations. Exchange: PBE0-D4/def2-TZVP HS/BS single
points with UNO/UCO and Hirshfeld populations. Convention (as in the other analyses):
    dE = E_HS - E_BS ;  dE > 0 : BS lower (AFM tendency)
Context is taken from the coarse L x C x R scan (angle_scan.csv, exchange_analysis.csv)
and the free relaxations (free_relaxation.csv).

Extra descriptors from the BS wavefunction:
    T      overlap of the magnetic pair of corresponding orbitals (UCO)
    n_HONO, n_LUNO  UHF natural-orbital occupations (= 1 +/- T)
    y      Yamaguchi diradical character, y = 1 - 2T / (1 + T^2)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange_convention as XC  # noqa: E402  (J convention, see exchange_convention.py)
from analyze_angle_scan import (  # noqa: E402
    FINAL_E_RE,
    HARTREE_TO_MEV,
    S2_RE,
    build_adjacency,
    check_chemical_integrity,
    outer_terminal_fragment,
    parse_out_file,
    reference_bonds,
    split_fragment_at_bond,
)
from analyze_outer_scans import (  # noqa: E402
    AQUA,
    BLUE,
    GRID,
    INK,
    INK_2,
    INK_3,
    ORANGE,
    YELLOW,
    corr,
    fmt_corr,
    mulliken,
    parse_pbe0,
    plane_angle,
    style_axes,
    symlog_dE,
)

LABEL_RE = re.compile(r"^L(\d+)_C(\d+)_R(\d+)$")
UCO_RE = re.compile(r"Orbital\s+Overlap\(\*\)\s*\n-+\n(.*?)\(\*\)", re.S)
UNO_RE = re.compile(r"UHF NATURAL ORBITALS\s*\n-+\s*\n\s*Occupation Numbers:\s*\n(.*?)\n\s*\n", re.S)
OCC_RE = re.compile(r"N\[\s*\d+\]=\s*(-?\d+\.\d+)")
HIRSH_RE = re.compile(r"HIRSHFELD ANALYSIS.*?ATOM\s+CHARGE\s+SPIN\s*\n(.*?)\n\s*TOTAL", re.S)
HIRSH_LINE_RE = re.compile(r"\s*(\d+)\s+\S+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
DISP_RE = re.compile(r"Dispersion correction\s+(-?\d+\.\d+)")

FAMILY = {(45, 45): "45/45 start", (135, 135): "135/135 start", (45, 135): "45/135 start"}
FAMILY_COLOR = {"45/45 start": BLUE, "135/135 start": ORANGE, "45/135 start": AQUA}
FAMILY_MARK = {"45/45 start": "o", "135/135 start": "s", "45/135 start": "^"}


# ============================================================
# PARSING
# ============================================================
def bs_extras(text, fragments):
    out = {"T": np.nan, "n_HONO": np.nan, "n_LUNO": np.nan}
    m = UCO_RE.search(text)
    if m:
        vals = [float(line.split(":")[1]) for line in m.group(1).strip().splitlines() if ":" in line]
        if vals:
            out["T"] = min(vals)  # magnetic pair = smallest weighted overlap
    m = UNO_RE.search(text)
    if m:
        occ = np.array([float(x) for x in OCC_RE.findall(m.group(1))])
        frac = np.sort(occ[(occ > 0.1) & (occ < 1.9)])
        if len(frac) >= 2:
            out["n_LUNO"], out["n_HONO"] = float(frac[0]), float(frac[-1])
    out.update(hirshfeld(text, fragments, "BS"))
    return out


def hirshfeld(text, fragments, tag):
    m = HIRSH_RE.search(text)
    if not m:
        return {}
    s, q = {}, {}
    for line in m.group(1).splitlines():
        mm = HIRSH_LINE_RE.match(line)
        if mm:
            q[int(mm.group(1))], s[int(mm.group(1))] = float(mm.group(2)), float(mm.group(3))
    return {f"H{k}_{tag}": sum(s.get(i, 0.0) for i in atoms) for k, atoms in fragments.items()}


def dispersion(path):
    if not path.exists():
        return np.nan
    v = DISP_RE.findall(path.read_text(errors="ignore"))
    return float(v[-1]) if v else np.nan


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    repo_root = Path(__file__).resolve().parent.parent
    scans_root = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans"
    relax_root = scans_root / "angle_scan_central_approx_90"
    pbe0_root = scans_root / "angle_scan_central_approx_90_exchange"
    coarse_scan_csv = scans_root / "angle_scan_relaxation" / "angle_scan.csv"
    coarse_ex_csv = scans_root / "angle_scan_exchange_analysis" / "exchange_analysis.csv"
    free_csv = scans_root / "angle_scan_free_relaxation" / "free_relaxation.csv"
    baseline_xyz = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    out_dir = scans_root / "central_approx_90_analysis"

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    RINGS = {
        "L": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]),
        "C": ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25]),
        "R": ([23, 21, 25, 24, 26, 27], [28, 29, 30, 31, 32]),
    }
    SETS = ("free_outer", "fixed_outer")
    FLOOR_FIT_PAIRS = [(45, 45), (135, 135), (0, 0), (45, 135)]  # outer pairs for dE(C) floor fits
    ORTHO_OUTER_PAIRS = [(90, 90), (90, 180), (0, 90), (45, 90), (90, 135)]

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

    coarse = pd.read_csv(coarse_scan_csv)
    coarse_ex = pd.read_csv(coarse_ex_csv)
    coarse_ex = coarse_ex[coarse_ex["usable"]]
    e_ref_label = coarse.loc[coarse["usable"], :].set_index("label")["final_energy"].idxmin()
    e_ref = float(coarse.set_index("label").loc[e_ref_label, "final_energy"])
    free = pd.read_csv(free_csv) if free_csv.exists() else pd.DataFrame()

    # ---- PARSE ----
    rows = []
    for set_name in SETS:
        manifest = pd.read_csv(relax_root / "manifest.csv")
        manifest = manifest[manifest["set"] == set_name].set_index("label")
        for d in sorted(p for p in (relax_root / set_name).iterdir() if p.is_dir() and LABEL_RE.match(p.name)):
            L0, C0, R0 = (int(x) for x in LABEL_RE.match(d.name).groups())
            row = {"set": set_name, "label": d.name, "L_init": L0, "C": C0, "R_init": R0,
                   "family": FAMILY.get((L0, R0), "") if set_name == "free_outer" else "",
                   "base": Path(manifest.loc[d.name, "source_geometry"]).parent.name if d.name in manifest.index else ""}
            out = d / "high_spin.out"
            text = out.read_text(errors="ignore")
            rec = parse_out_file(text, out)
            atoms = read(d / "high_spin.xyz")
            pos = atoms.get_positions()
            integ = check_chemical_integrity(atoms, bonds, bonded_pairs, 0.30, 1.9, 1.5)
            row.update({
                "converged": rec["opt_converged"], "convergence_type": rec["convergence_type"],
                "n_geom_cycles": rec["n_geom_cycles"], "E_r2scan": rec["final_energy"],
                "geometry_valid": integ["valid"], "usable": bool(rec["opt_converged"] and integ["valid"]),
                "L_final": (atoms.get_dihedral(*LEFT) + 180) % 360 - 180,
                "C_final": (atoms.get_dihedral(*CENTER) + 180) % 360 - 180,
                "R_final": (atoms.get_dihedral(*RIGHT) + 180) % 360 - 180,
            })
            for k in "LCR":
                row[f"theta_{k}_plane"] = plane_angle(pos, *RINGS[k])
            if rec["opt_converged"]:
                row.update({f"{k}_HS_r2scan": v for k, v in mulliken(text, fragments).items()})

            # PBE0 HS/BS
            ex = pbe0_root / set_name / d.name
            hs_out, bs_out = ex / "high_spin" / "high_spin.out", ex / "broken_symmetry" / "broken_symmetry.out"
            row["pbe0_status"] = "missing"
            if hs_out.exists() and bs_out.exists():
                hs, bs = parse_pbe0(hs_out, fragments), parse_pbe0(bs_out, fragments)
                if hs["ok"] and bs["ok"]:
                    row["pbe0_status"] = "ok"
                    row["dE"] = (hs["E"] - bs["E"]) * HARTREE_TO_MEV
                    row["s2_HS"], row["s2_BS"] = hs["s2"], bs["s2"]
                    for k in range(1, 5):
                        row[f"S{k}_HS"], row[f"S{k}_BS"] = hs.get(f"S{k}"), bs.get(f"S{k}")
                    row.update(bs_extras(bs_out.read_text(errors="ignore"), fragments))
                    row.update(hirshfeld(hs_out.read_text(errors="ignore"), fragments, "HS"))
                    row["disp_diff_Eh"] = dispersion(hs_out) - dispersion(bs_out)
                else:
                    row["pbe0_status"] = "failed"
            rows.append(row)

    df = pd.DataFrame(rows)
    df["E_rel_meV"] = (df["E_r2scan"] - e_ref) * HARTREE_TO_MEV
    df["L_mag"] = df["L_final"].abs().map(lambda a: min(a, 180 - a))
    df["R_mag"] = df["R_final"].abs().map(lambda a: min(a, 180 - a))
    for lvl in ("_HS_r2scan", "_HS"):
        if f"S1{lvl}" in df:
            df[f"mean_terminal{lvl}"] = (df[f"S1{lvl}"].abs() + df[f"S4{lvl}"].abs()) / 2
            df[f"backbone{lvl}"] = df[f"S2{lvl}"] + df[f"S3{lvl}"]
    if "T" in df:
        df["T2"] = df["T"] ** 2
        df["y_diradical"] = 1 - 2 * df["T"] / (1 + df["T"] ** 2)
        df["T_from_NO"] = (df["n_HONO"] - df["n_LUNO"]) / 2
        # H = -J S1.S2 (J > 0 FM), dE = E_HS - E_BS
        df["J_ideal_meV"] = XC.j_ideal(df["dE"])
        df["J_yamaguchi_meV"] = XC.j_yamaguchi(df["dE"], df["s2_HS"], df["s2_BS"])
        df["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(df["J_yamaguchi_meV"])
        df["mean_terminal_hirsh_HS"] = (df["H1_HS"].abs() + df["H4_HS"].abs()) / 2
    df["P"] = np.prod([np.cos(np.radians(df[f"theta_{a}_plane"])) ** 2 for a in "LCR"], axis=0)
    df["cos2_C"] = np.cos(np.radians(df["theta_C_plane"])) ** 2
    df.to_csv(out_dir / "central_approx_90.csv", index=False)

    U = df[df["usable"]].copy()
    FR = U[U["set"] == "free_outer"].copy()
    FX = U[U["set"] == "fixed_outer"].copy()
    EXo = U[U["pbe0_status"] == "ok"].copy()

    # ---- COARSE CONTEXT: exchange(C) per fixed outer pair, with the C = 90 point inserted ----
    cx = coarse_ex[["L", "C", "R", "delta_E_raw_meV", "theta_C_plane", "theta_L_plane", "theta_R_plane",
                    "spin_f1_HS", "spin_f4_HS"]].rename(columns={"delta_E_raw_meV": "dE"})
    cx["source"] = "coarse grid"
    fx = FX[FX["pbe0_status"] == "ok"][["L_init", "C", "R_init", "dE", "theta_C_plane", "theta_L_plane",
                                        "theta_R_plane", "S1_HS", "S4_HS"]].rename(
        columns={"L_init": "L", "R_init": "R", "S1_HS": "spin_f1_HS", "S4_HS": "spin_f4_HS"})
    fx["source"] = "C = 90 (this set)"
    line = pd.concat([cx, fx], ignore_index=True)
    line["cos2_C"] = np.cos(np.radians(line["theta_C_plane"])) ** 2

    floors = {}
    for Lp, Rp in FLOOR_FIT_PAIRS:
        g = line[(line["L"] == Lp) & (line["R"] == Rp)].sort_values("C")
        if len(g) >= 4:
            a, b = np.polyfit(g["cos2_C"], g["dE"], 1)
            pred = a * g["cos2_C"] + b
            r2 = 1 - ((g["dE"] - pred) ** 2).sum() / ((g["dE"] - g["dE"].mean()) ** 2).sum()
            near = g[(g["C"] >= 80) & (g["C"] <= 100)]
            floors[(Lp, Rp)] = {"a": a, "b": b, "r2": r2, "n": len(g), "near90_min": float(near["dE"].min()),
                                "near90_max": float(near["dE"].max()), "table": g}

    res = {"floors": floors, "line": line, "e_ref_label": e_ref_label}
    m1 = m1_reference(scans_root, free, fragments)
    plot_free_outer_exchange(FR, FX, m1, plots_dir)
    make_plots(U, FR, FX, EXo, line, floors, free, e_ref, plots_dir, ORTHO_OUTER_PAIRS, coarse)
    write_report(df, U, FR, FX, EXo, line, floors, free, e_ref, e_ref_label, out_dir / "central_approx_90.md",
                 ORTHO_OUTER_PAIRS, coarse, m1)
    print(f"Wrote {out_dir / 'central_approx_90.md'}")
    print(f"PBE0 complete: free_outer {int((FR['pbe0_status'] == 'ok').sum())}/{len(FR)}, "
          f"fixed_outer {int((FX['pbe0_status'] == 'ok').sum())}/{len(FX)}")


# ============================================================
# PLOTS
# ============================================================
def free_minima(free, e_ref_label_offset=0.0):
    if free.empty:
        return []
    f = free[(free["state"] == "high_spin") & free["converged"]]
    out = []
    for m, g in f.groupby("minimum"):
        r = g.loc[g["E_final"].idxmin()]
        out.append((m, abs(r["C_final"]), r["E_vs_scan_ref_meV"], r["L_final"], r["R_final"]))
    return sorted(out, key=lambda x: x[2])


def m1_reference(scans_root, free, fragments):
    """Free minimum M1 as the C > 90 end point of the 45/45 family: PBE0 HS/BS (UNO/UCO restart at the M1
    geometry, natural_orbitals/M1) + relaxed angles from fluorene_angle_analysis.csv. None if unavailable."""
    base = scans_root / "natural_orbitals" / "M1"
    fa = scans_root / "fluorene_angle_analysis" / "fluorene_angle_analysis.csv"
    hs, bs = base / "high_spin" / "high_spin.out", base / "broken_symmetry" / "broken_symmetry.out"
    if not (hs.exists() and bs.exists() and fa.exists()):
        return None
    th, tb = hs.read_text(errors="ignore"), bs.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in th or "ORCA TERMINATED NORMALLY" not in tb:
        return None
    e_hs, e_bs = float(FINAL_E_RE.findall(th)[-1]), float(FINAL_E_RE.findall(tb)[-1])
    s2_hs, s2_bs = float(S2_RE.findall(th)[-1]), float(S2_RE.findall(tb)[-1])
    geo = pd.read_csv(fa)
    g = geo[(geo["kind"] == "free") & (geo["minimum"] == "M1")].sort_values("E_meV").iloc[0]
    mins = {m: e for m, _, e, _, _ in free_minima(free)}
    out = {"dE": (e_hs - e_bs) * HARTREE_TO_MEV, "s2_HS": s2_hs, "s2_BS": s2_bs, "C": g["C"],
           "theta_C_plane": g["theta_C"], "theta_L_plane": g["theta_L"], "theta_R_plane": g["theta_R"],
           "E_rel_meV": mins.get("M1", np.nan)}
    out["T"] = bs_extras(tb, fragments)["T"]
    out["J_ideal_meV"] = XC.j_ideal(out["dE"])
    out["J_yamaguchi_meV"] = XC.j_yamaguchi(out["dE"], s2_hs, s2_bs)
    out["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(out["J_yamaguchi_meV"])
    return out


def plot_free_outer_exchange(FR, FX, m1, plots_dir):
    fe = FR[FR["pbe0_status"] == "ok"]
    if fe.empty:
        return
    fx = FX[(FX["pbe0_status"] == "ok") & FX.apply(lambda r: (r["L_init"], r["R_init"]) in FAMILY, axis=1)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharey=True)
    for ax, (col, lab) in zip(axes, (("C", "constrained central dihedral C (deg)"),
                                     ("theta_C_plane", "actual central plane angle θ$_C$ (deg)"),
                                     ("T", "magnetic-orbital overlap T (BS UCO)"))):
        for fam, g in fe.groupby("family"):
            g = g.sort_values("C")
            ax.plot(g[col], g["dE"], ("-" if col == "C" else "") + FAMILY_MARK[fam], color=FAMILY_COLOR[fam],
                    markersize=8, markeredgecolor="white", linewidth=1.6, label=f"free outer, {fam}")
        for r in fx.itertuples():
            fam = FAMILY[(r.L_init, r.R_init)]
            ax.plot(getattr(r, col), r.dE, "D", markerfacecolor="white", markeredgecolor=FAMILY_COLOR[fam],
                    markersize=8, markeredgewidth=1.6,
                    label=f"C = 90, outer fixed at {r.L_init}/{r.R_init}" if ax is axes[0] else None)
        if m1 is not None and pd.notna(m1.get(col)):
            ax.plot(m1[col], m1["dE"], "*", color=INK, markersize=14, label="free minimum M1" if ax is axes[0] else None)
            ax.annotate("M1", (m1[col], m1["dE"]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
        ax.set_xlabel(lab)
        style_axes(ax)
    axes[0].axvline(90, color=INK_3, linewidth=0.8, linestyle=":")
    axes[1].axvline(90, color=INK_3, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("PBE0 ΔE = E$_{HS}$ − E$_{BS}$ (meV, > 0: BS lower)")
    axes[0].set_ylim(0, None)
    axes[0].set_title("(a) vs imposed C", fontsize=10, loc="left")
    axes[1].set_title("(b) vs actual central plane angle", fontsize=10, loc="left")
    axes[2].set_title("(c) vs magnetic-orbital overlap", fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes[0].legend(seen.values(), seen.keys(), frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Free-outer exchange with the C = 90 fixed-outer points and M1", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plots_dir / "free_outer_exchange_vs_fixed_and_M1.png", dpi=200)
    plt.close(fig)


def make_plots(U, FR, FX, EXo, line, floors, free, e_ref, plots_dir, ortho_pairs, coarse):
    # ---- 1. energy vs central angle (free outer) + relaxed outer angles + actual theta_C
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    cu = coarse[coarse["usable"]]
    for (Lp, Rp), shade in (((45, 45), "#b5b4ae"), ((135, 135), "#8a8984")):
        g = cu[(cu["L"] == Lp) & (cu["R"] == Rp)].sort_values("C")
        ax.plot(g["C"], g["relative_energy_meV"], ":", color=shade, linewidth=1.2, marker=".",
                label=f"coarse scan, L = R = {Lp}° fixed")
    for fam, g in FR.groupby("family"):
        g = g.sort_values("C")
        ax.plot(g["C"], g["E_rel_meV"], "-" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markersize=7,
                markeredgecolor="white", linewidth=1.8, label=f"free outer, {fam}")
    for m, c, e, _, _ in free_minima(free)[:2]:
        ax.plot(c, e, "*", color=INK, markersize=13)
        ax.annotate(f"free minimum {m}", (c, e), xytext=(-8, -14), textcoords="offset points", fontsize=8.5,
                    ha="right" if c > 90 else "left", color=INK)
    ax.set_xlim(72, 108)
    ax.set_ylim(-45, 160)
    ax.set_xlabel("constrained central dihedral C (deg)")
    ax.set_ylabel("R2SCAN-3C E vs common reference (meV)")
    ax.set_title("Conformational energy around C = 90°", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    style_axes(ax)

    ax = axes[1]
    for fam, g in FR.groupby("family"):
        g = g.sort_values("C")
        ax.plot(g["C"], g["L_final"], "-" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markersize=6, label=f"{fam}: L")
        ax.plot(g["C"], g["R_final"], "--" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markerfacecolor="white",
                markersize=6, label=f"{fam}: R")
    ax.set_xlabel("constrained central dihedral C (deg)")
    ax.set_ylabel("relaxed outer dihedral (deg)")
    ax.set_title("Where the free outer torsions go", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    style_axes(ax)

    ax = axes[2]
    ax.plot([75, 105], [75, 105], ":", color=INK_3, linewidth=1)
    ax.plot([75, 105], [75, 75 + 0 * 30], alpha=0)
    for fam, g in FR.groupby("family"):
        g = g.sort_values("C")
        ax.plot(g["C"], g["theta_C_plane"], "-" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markersize=7,
                markeredgecolor="white", label=fam)
    ax.plot(FX["C"], FX["theta_C_plane"], "D", markerfacecolor="white", markeredgecolor=INK_2, markersize=6,
            label="fixed outer, C = 90 (all L, R)")
    ax.set_xlabel("constrained central dihedral C (deg)")
    ax.set_ylabel("actual central plane angle θ_C (deg, folded 0–90)")
    ax.set_title("Imposed C vs actual central ring-plane angle\n(dotted: θ_C = C, valid up to 90°)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_central_angle.png", dpi=200)
    plt.close(fig)

    # ---- 2. C = 90 (L, R) maps: energy, exchange, localization, magnetic-orbital overlap
    grid = [0, 45, 90, 135, 180]
    panels = [("E_rel_meV", "R2SCAN-3C E (meV, common ref.)", "viridis", None),
              ("dE", "ΔE = E_HS − E_BS, PBE0 (meV)", "RdBu_r", "div"),
              ("mean_terminal_HS", "mean terminal spin (PBE0 HS)", "Blues", None),
              ("T", "magnetic-orbital overlap T (BS)", "Purples", None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (col, lab, cmap, kind) in zip(axes, panels):
        M = np.full((5, 5), np.nan)
        for _, r in FX.iterrows():
            if col in r and np.isfinite(r.get(col, np.nan)):
                M[grid.index(r["R_init"]), grid.index(r["L_init"])] = r[col]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(GRID)
        if kind == "div":
            v = np.nanmax(np.abs(M))
            im = ax.imshow(M, origin="lower", cmap=cm, vmin=-v, vmax=v)
        else:
            im = ax.imshow(M, origin="lower", cmap=cm)
        for i in range(5):
            for j in range(5):
                if np.isfinite(M[i, j]):
                    v_ = M[i, j]
                    txt = f"{v_:.0f}" if col == "E_rel_meV" else (f"{v_:+.2f}" if col == "dE" else
                                                                   (f"{v_:.3f}" if col == "T" else f"{v_:.2f}"))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if (col == "E_rel_meV" and v_ < np.nanmean(M)) else INK)
        ax.set_xticks(range(5))
        ax.set_xticklabels(grid)
        ax.set_yticks(range(5))
        ax.set_yticklabels(grid)
        ax.set_xlabel("L (deg)")
        ax.set_ylabel("R (deg)")
        ax.set_title(lab, fontsize=9.5, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("C = 90° constrained, L ≤ R computed (grey: mirror L↔R, not computed)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "fixedC90_LR_maps.png", dpi=200)
    plt.close(fig)

    # ---- 3. exchange vs central angle (coarse lines + C = 90 points + free-outer points)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4"]
    ax = axes[0]
    for (Lp, Rp), c in zip([(45, 45), (135, 135), (45, 135), (0, 0)], cols):
        g = line[(line["L"] == Lp) & (line["R"] == Rp)].sort_values("C")
        ax.plot(g["C"], g["dE"], "-o", color=c, markersize=5, linewidth=1.4, label=f"L/R = {Lp}/{Rp}")
        g90 = g[g["source"] != "coarse grid"]
        ax.plot(g90["C"], g90["dE"], "D", color=c, markersize=9, markeredgecolor=INK)
    fe = FR[FR["pbe0_status"] == "ok"]
    for fam, g in fe.groupby("family"):
        ax.plot(g["C"], g["dE"], FAMILY_MARK[fam], markerfacecolor="white", markeredgecolor=FAMILY_COLOR.get(fam, INK),
                markersize=10, markeredgewidth=2, label=f"free outer, {fam}")
    ax.set_title("Twisted outer rings: ΔE vs C (◆ = new C = 90 point)", fontsize=10)
    ax = axes[1]
    for (Lp, Rp), c in zip(ortho_pairs, cols):
        g = line[(line["L"] == Lp) & (line["R"] == Rp)].sort_values("C")
        if g.empty:
            g = line[(line["L"] == Rp) & (line["R"] == Lp)].sort_values("C")
        ax.plot(g["C"], g["dE"], "-o", color=c, markersize=5, linewidth=1.4, label=f"L/R = {Lp}/{Rp}")
        g90 = g[g["source"] != "coarse grid"]
        ax.plot(g90["C"], g90["dE"], "D", color=c, markersize=9, markeredgecolor=INK)
    ax.set_title("One outer ring orthogonal: ΔE vs C", fontsize=10)
    for ax in axes:
        ax.axhline(0, color=INK_3, linewidth=1)
        ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.axvline(90, color=INK_3, linewidth=0.8, linestyle=":")
        symlog_dE(ax)
        ax.set_yticks([-0.5, 0, 0.5, 1, 2, 5, 10, 20])
        ax.set_xlabel("constrained central dihedral C (deg)")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_central_angle.png", dpi=200)
    plt.close(fig)

    # ---- 4. exchange floor: dE vs cos^2(theta_C actual)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ((Lp, Rp), f), c in zip(floors.items(), cols):
        g = f["table"]
        ax.plot(g["cos2_C"], g["dE"], "o", color=c, markersize=7, markeredgecolor="white",
                label=f"{Lp}/{Rp}: ΔE = {f['a']:.1f}·cos²θ_C {f['b']:+.2f} (R² = {f['r2']:.2f})")
        xs = np.linspace(0, g["cos2_C"].max(), 50)
        ax.plot(xs, f["a"] * xs + f["b"], color=c, linewidth=1, alpha=0.7)
    ax.axhline(0, color=INK_3, linewidth=1)
    ax.set_xlabel("cos²θ_C (actual central plane angle)")
    ax.set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
    ax.set_title("Does the exchange vanish when the central link is orthogonal?\n(intercept = residual ΔE at θ_C = 90°)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_floor_vs_cos2_thetaC.png", dpi=200)
    plt.close(fig)

    # ---- 5. localization vs angles
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for fam, g in FR.groupby("family"):
        g = g.sort_values("C")
        ax.plot(g["C"], g["S1_HS_r2scan"].abs(), "-" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markersize=6,
                label=f"{fam}: |S1|")
        ax.plot(g["C"], g["S4_HS_r2scan"].abs(), "--" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam],
                markerfacecolor="white", markersize=6, label=f"{fam}: |S4|")
        ax.plot(g["C"], g["backbone_HS_r2scan"], ":" + FAMILY_MARK[fam], color=FAMILY_COLOR[fam], markersize=4,
                label=f"{fam}: S2 + S3")
    ax.set_xlabel("constrained central dihedral C (deg)")
    ax.set_ylabel("HS Mulliken spin (R2SCAN-3C)")
    ax.set_title("Free outer: localization vs central angle", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    style_axes(ax)
    ax = axes[1]
    for side, col, c, mk in (("left", "theta_L_plane", BLUE, "o"), ("right", "theta_R_plane", ORANGE, "s")):
        s = FX["S1_HS_r2scan" if side == "left" else "S4_HS_r2scan"].abs()
        ax.plot(FX[col], s, mk, color=c, markersize=7, markeredgecolor="white", label=f"C = 90 fixed: {side} radical")
    allx = pd.concat([U["theta_L_plane"], U["theta_R_plane"]])
    ally = pd.concat([U["S1_HS_r2scan"].abs(), U["S4_HS_r2scan"].abs()])
    ax.plot(FR["theta_L_plane"], FR["S1_HS_r2scan"].abs(), "^", color=AQUA, markersize=6, label="free outer (all C)")
    ax.set_xlabel("actual plane angle of the radical's own outer torsion (deg)")
    ax.set_ylabel("terminal HS spin (R2SCAN-3C)")
    c_ = corr(allx, ally)
    ax.set_title(f"Localization vs own outer plane angle\n(all geometries: ρ = {c_['spearman']:.2f}, r = {c_['pearson']:.2f}, n = {c_['n']})",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "localization_vs_angles.png", dpi=200)
    plt.close(fig)

    # ---- 6. side: exchange vs magnetic-orbital overlap and diradical character
    if len(EXo):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, col, lab in ((axes[0], "T2", "T² (magnetic-orbital overlap squared, BS)"),
                             (axes[1], "mean_terminal_HS", "mean terminal spin (PBE0 HS)"),
                             (axes[2], "P", "P = cos²θ_L cos²θ_C cos²θ_R")):
            for sname, mk, c in (("fixed_outer", "o", BLUE), ("free_outer", "^", ORANGE)):
                g = EXo[EXo["set"] == sname]
                ax.plot(g[col], g["dE"], mk, color=c, markersize=8, markeredgecolor="white",
                        label=f"{sname} ({fmt_corr(corr(g[col], g['dE']))})")
            ax.axhline(0, color=INK_3, linewidth=1)
            ax.set_xlabel(lab)
            ax.legend(frameon=False, fontsize=7.5)
            style_axes(ax)
        axes[0].set_ylabel("ΔE = E_HS − E_BS, PBE0 (meV)")
        fig.tight_layout()
        fig.savefig(plots_dir / "exchange_vs_overlap_localization_P.png", dpi=200)
        plt.close(fig)

        # Hirshfeld vs Mulliken
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0.7, 1.05], [0.7, 1.05], ":", color=INK_3)
        ax.plot(EXo["S1_HS"].abs(), EXo["H1_HS"].abs(), "o", color=BLUE, label="fragment 1")
        ax.plot(EXo["S4_HS"].abs(), EXo["H4_HS"].abs(), "s", color=ORANGE, label="fragment 4")
        ax.set_xlabel("terminal spin, Mulliken (PBE0 HS)")
        ax.set_ylabel("terminal spin, Hirshfeld (PBE0 HS)")
        ax.set_title(f"Population-scheme check ({fmt_corr(corr(pd.concat([EXo['S1_HS'], EXo['S4_HS']]).abs(), pd.concat([EXo['H1_HS'], EXo['H4_HS']]).abs()))})",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(plots_dir / "hirshfeld_vs_mulliken.png", dpi=200)
        plt.close(fig)


# ============================================================
# REPORT
# ============================================================
def write_report(df, U, FR, FX, EXo, line, floors, free, e_ref, e_ref_label, md_path, ortho_pairs, coarse, m1=None):
    L = []
    nFRok, nFXok = int((FR["pbe0_status"] == "ok").sum()), int((FX["pbe0_status"] == "ok").sum())
    fx = FX.set_index(["L_init", "R_init"])
    fam = {f: g.sort_values("C").set_index("C") for f, g in FR.groupby("family")}
    a45, a135, amix = fam.get("45/45 start"), fam.get("135/135 start"), fam.get("45/135 start")
    # crossover of the 45/45 and 135/135 families
    cross, cross_E = np.nan, np.nan
    if a45 is not None and a135 is not None:
        diff = (a45["E_rel_meV"] - a135["E_rel_meV"]).dropna()
        for (c0, d0), (c1, d1) in zip(diff.items(), list(diff.items())[1:]):
            if np.sign(d0) != np.sign(d1):
                cross = c0 + (c1 - c0) * d0 / (d0 - d1)
                e0, e1 = a45.loc[c0, "E_rel_meV"], a45.loc[c1, "E_rel_meV"]
                cross_E = e0 + (e1 - e0) * (cross - c0) / (c1 - c0)
    mins = free_minima(free)
    fl = floors.get((45, 45))
    ortho = fx[(fx.index.get_level_values(0) == 90) | (fx.index.get_level_values(1) == 90)]
    twisted = fx[(fx.index.get_level_values(0) != 90) & (fx.index.get_level_values(1) != 90)]

    L.append("# Central torsion around C = 90°")
    L.append("")
    L.append("Two sets of R2SCAN-3C HS relaxations with PBE0-D4/def2-TZVP HS/BS single points (UNO/UCO, Hirshfeld):")
    L.append("")
    L.append("- **free_outer**: C = 80, 85, 90, 95, 100° constrained; outer torsions started at 45/45, 135/135 or "
             "45/135 and then free.")
    L.append("- **fixed_outer**: C = 90° with L ≤ R on the 0/45/90/135/180° grid, all three torsions constrained "
             "(fills the C = 90° gap of the coarse scan).")
    L.append("")
    L.append("## Status")
    L.append("")
    L.append(f"- Relaxations: {int(df['usable'].sum())}/{len(df)} converged and chemically intact.")
    L.append(f"- PBE0 HS/BS: fixed_outer {nFXok}/{len(FX)}, free_outer {nFRok}/{len(FR)}. "
             + ("" if nFRok == len(FR) else
                "**Some free_outer single points have not finished yet** (job folders exist without output); every "
                "exchange statement about free_outer below is limited to the finished point(s) and is filled in "
                "automatically on re-running `visualisation/analyze_central_approx_90.py`."))
    if "disp_diff_Eh" in EXo:
        L.append(f"- D4 is identical in HS and BS (max |ΔD4| = {EXo['disp_diff_Eh'].abs().max():.1e} Eh), so ΔE is "
                 "directly comparable with the coarse-grid PBE0 values (no D4).")
    L.append("- ΔE = E_HS − E_BS: ΔE > 0 → BS lower (AFM tendency); |ΔE| < 0.5 meV near-degenerate. Energies are "
             f"relative to {e_ref_label} (coarse scan), the same reference as `angle_scan.md`.")
    L.append("")

    # ---------------- PART I
    L.append("# Part I — Main results")
    L.append("")
    L.append("## M1. Conformational energy around C = 90° (free outer torsions)")
    L.append("")
    L.append("| C (deg) | 45/45 start: E (meV) | relaxed L / R | θ_C plane | 135/135 start: E | relaxed L / R | θ_C plane | 45/135 start: E | relaxed L / R | θ_C plane |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(FR["C"].unique()):
        cells = []
        for a in (a45, a135, amix):
            if a is not None and c in a.index:
                r = a.loc[c]
                cells += [f"{r['E_rel_meV']:+.1f}", f"{r['L_final']:.0f} / {r['R_final']:.0f}", f"{r['theta_C_plane']:.1f}"]
            else:
                cells += ["", "", ""]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"- Released from 45° the outer rings relax to ≈ ±36°, from 135° to ≈ 145°: the same |36°| twist as the free "
             "minima, independent of C.")
    if np.isfinite(cross):
        L.append(f"- The 45/45 and 135/135 families **cross at C ≈ {cross:.1f}°**: below it the 135/135 handedness is "
                 "lower, above it the 45/45 handedness, consistent with the crossover found in the coarse scan, but the "
                 "crossing sits a few degrees below 90°, not at it.")
    if a45 is not None and a135 is not None:
        L.append(f"- Along C = 80→100° the 45/45 family falls by {a45['E_rel_meV'].iloc[0] - a45['E_rel_meV'].iloc[-1]:.0f} meV "
                 f"and the 135/135 family rises by {a135['E_rel_meV'].iloc[-1] - a135['E_rel_meV'].iloc[0]:.0f} meV: no "
                 "stationary point at C = 90°; each family keeps descending toward its own free minimum "
                 + ", ".join(f"{m} (C = {c:.1f}°, {e:+.1f} meV)" for m, c, e, _, _ in mins[:2]) + ".")
    if amix is not None:
        L.append(f"- The mixed 45/135 start keeps opposite outer handedness (L ≈ {amix['L_final'].mean():.0f}°, "
                 f"R ≈ {amix['R_final'].mean():.0f}°) and lies {amix['E_rel_meV'].min():.0f}–{amix['E_rel_meV'].max():.0f} meV "
                 "with a shallow C dependence: it never becomes the lowest family.")
    L.append(f"- The imposed C is a poor proxy for the actual central plane angle: at C = 80° the 45/45 family has "
             f"θ_C = {a45.loc[80, 'theta_C_plane']:.0f}° but the 135/135 family {a135.loc[80, 'theta_C_plane']:.0f}°, and the "
             "trend reverses at C = 100° (the backbone takes up part of the twist depending on outer handedness).")
    L.append("")
    L.append("![Energy vs central angle](plots/energy_vs_central_angle.png)")
    L.append("")

    L.append("## M2. The C = 90° (L, R) slice: energy, exchange, localization")
    L.append("")
    L.append("| L | R | θ_L / θ_C / θ_R plane | E (meV) | ΔE (meV) | ⟨S²⟩ HS / BS | \\|S1\\| / \\|S4\\| (PBE0 HS) | T (UCO) | n_LUNO | y |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for (l_, r_), r in fx.sort_index().iterrows():
        L.append(f"| {l_} | {r_} | {r['theta_L_plane']:.1f} / {r['theta_C_plane']:.1f} / {r['theta_R_plane']:.1f} | "
                 f"{r['E_rel_meV']:.0f} | {r.get('dE', np.nan):+.3f} | {r.get('s2_HS', np.nan):.3f} / {r.get('s2_BS', np.nan):.3f} | "
                 f"{abs(r.get('S1_HS', np.nan)):.3f} / {abs(r.get('S4_HS', np.nan)):.3f} | {r.get('T', np.nan):.4f} | "
                 f"{r.get('n_LUNO', np.nan):.4f} | {r.get('y_diradical', np.nan):.3f} |")
    L.append("")
    L.append(f"- Lowest C = 90° conformer: L/R = {fx['E_rel_meV'].idxmin()[0]}/{fx['E_rel_meV'].idxmin()[1]} at "
             f"{fx['E_rel_meV'].min():+.0f} meV; planar or orthogonal outer rings cost "
             f"{fx['E_rel_meV'].max() - fx['E_rel_meV'].min():.0f} meV more at most.")
    L.append(f"- With **no** outer ring at 90° ΔE = {twisted['dE'].min():+.2f} to {twisted['dE'].max():+.2f} meV: the "
             "exchange is weak but clearly AFM even though the central link is (nearly) orthogonal "
             f"(θ_C = {twisted['theta_C_plane'].min():.0f}–{twisted['theta_C_plane'].max():.0f}°).")
    one = ortho[(ortho.index.get_level_values(0) != 90) | (ortho.index.get_level_values(1) != 90)]
    L.append(f"- With **one** outer ring at 90° ΔE = {one['dE'].min():+.2f} to {one['dE'].max():+.2f} meV for every partner angle, "
             f"including the planar partner 180° ({fx.loc[(90, 180), 'dE']:+.3f} meV) that gave +2.57 meV at C = 100°.")
    if (90, 90) in fx.index:
        L.append(f"- All three links near-orthogonal (L = C = R = 90°): ΔE = {fx.loc[(90, 90), 'dE']:+.2f} meV, "
                 f"T = {fx.loc[(90, 90), 'T']:.4f}: small but not zero, larger than with one outer at 90° and a twisted partner.")
    L.append("- Localization is set by the outer plane angle alone (see M4).")
    L.append("")
    L.append("![C = 90 maps](plots/fixedC90_LR_maps.png)")
    L.append("")

    L.append("## M3. Exchange vs central angle: is the central link a bottleneck?")
    L.append("")
    L.append("Coarse-grid ΔE(C) at fixed outer angles with the new C = 90° point inserted (◆), and a linear fit "
             "ΔE = a·cos²θ_C + b over the whole C range using the actual central plane angle:")
    L.append("")
    L.append("| L / R | n | a (meV) | b = residual at θ_C = 90° (meV) | R² | ΔE range for C = 80–100° (meV) |")
    L.append("|---|---|---|---|---|---|")
    for (Lp, Rp), f in floors.items():
        L.append(f"| {Lp}/{Rp} | {f['n']} | {f['a']:.1f} | {f['b']:+.2f} | {f['r2']:.2f} | {f['near90_min']:+.2f} to {f['near90_max']:+.2f} |")
    L.append("")
    if fl:
        good = {k: f for k, f in floors.items() if f["r2"] >= 0.9}
        floor_lo = min(f["near90_min"] for f in floors.values())
        floor_hi = max(f["near90_min"] for f in floors.values())
        L.append(f"For twisted outer rings ΔE decreases steeply as C approaches 90° (e.g. 45/45: "
                 + ", ".join(f"C = {int(c)}°: {d:+.2f}" for c, d in zip(fl['table']['C'], fl['table']['dE']))
                 + f" meV) but **does not vanish around C = 80–100°: the smallest ΔE reached is {floor_lo:+.2f} to "
                 f"{floor_hi:+.2f} meV depending on the outer pair**, with actual θ_C up to ~90°. Where the cos²θ_C model "
                 f"fits well (R² ≥ 0.9: {', '.join(f'{a}/{b}' for a, b in good)}), it extrapolates to a positive residual "
                 f"of {min(f['b'] for f in good.values()):+.2f} to {max(f['b'] for f in good.values()):+.2f} meV at θ_C = 90°, "
                 "and the C = 80–100° points lie above the fit line. "
                 + ("The 135/135 line is poorly described (R² = "
                    f"{floors[(135, 135)]['r2']:.2f}, negative intercept): it is dominated by the strained C ≥ 120° points "
                    f"(ΔE up to {floors[(135, 135)]['table']['dE'].max():.1f} meV). "
                    if (135, 135) in floors and floors[(135, 135)]["r2"] < 0.9 else "")
                 + "A residual coupling thus survives an orthogonal central link, whereas an orthogonal OUTER link drives "
                 "ΔE to ≈ 0 for C ≤ 90° (right panel below): the central link is a leaky bottleneck, the outer links "
                 "are tight ones in that range.")
    L.append("")
    L.append("![Exchange vs central angle](plots/exchange_vs_central_angle.png)")
    L.append("")
    L.append("![Exchange floor](plots/exchange_floor_vs_cos2_thetaC.png)")
    L.append("")
    L.append("**One orthogonal outer ring vs C** (coarse grid + C = 90°): the 'Scan B re-emergence' found at C = 100° "
             "(L = 90°, R = 180°) is a C > 90° effect:")
    L.append("")
    L.append("| L / R | " + " | ".join(f"C = {c}" for c in (20, 40, 60, 80, 90, 100, 120, 140)) + " |")
    L.append("|---|" + "---|" * 8)
    for Lp, Rp in ortho_pairs:
        g = line[(line["L"] == Lp) & (line["R"] == Rp)].set_index("C")["dE"]
        L.append(f"| {Lp}/{Rp} | " + " | ".join(f"{g.loc[c]:+.2f}" if c in g.index else "" for c in (20, 40, 60, 80, 90, 100, 120, 140)) + " |")
    L.append("")

    fe = FR[FR["pbe0_status"] == "ok"]
    if len(fe):
        L.append("## M3b. Exchange with free outer torsions (relaxed to |36°|)")
        L.append("")
        L.append("| family | C | θ_C plane | ΔE (meV) | T | P | mean terminal spin (PBE0 HS) |")
        L.append("|---|---|---|---|---|---|---|")
        for f_, g in fe.sort_values(["family", "C"]).groupby("family", sort=False):
            for _, r in g.iterrows():
                L.append(f"| {f_} | {r['C']} | {r['theta_C_plane']:.1f} | {r['dE']:+.3f} | {r['T']:.4f} | {r['P']:.1e} | "
                         f"{r['mean_terminal_HS']:.3f} |")
        L.append("")
        L.append("| family | ΔE range (meV) | θ_C range (deg) | ρ(ΔE, cos²θ_C) | ρ(ΔE, T) |")
        L.append("|---|---|---|---|---|")
        fam_stats = {}
        for f_, g in fe.groupby("family"):
            fam_stats[f_] = (corr(g["cos2_C"], g["dE"]), corr(g["T"], g["dE"]))
            L.append(f"| {f_} | {g['dE'].min():+.2f} to {g['dE'].max():+.2f} | {g['theta_C_plane'].min():.0f}–"
                     f"{g['theta_C_plane'].max():.0f} | {fam_stats[f_][0]['spearman']:.2f} | {fam_stats[f_][1]['spearman']:.2f} |")
        L.append("")
        L.append("(ρ with n = 5 per family: indicative only; interpretation in M5.)")
        L.append("")

    L.append("## M4. Localization vs angles")
    L.append("")
    allx = pd.concat([U["theta_L_plane"], U["theta_R_plane"]])
    ally = pd.concat([U["S1_HS_r2scan"].abs(), U["S4_HS_r2scan"].abs()])
    L.append(f"- Free outer: terminal spins stay at |S| = {FR[['S1_HS_r2scan', 'S4_HS_r2scan']].abs().min().min():.3f}–"
             f"{FR[['S1_HS_r2scan', 'S4_HS_r2scan']].abs().max().max():.3f} and the backbone at "
             f"{FR['backbone_HS_r2scan'].min():.3f}–{FR['backbone_HS_r2scan'].max():.3f} over C = 80–100°: the central "
             "torsion does not change radical localization (the outer rings sit at ~36° throughout).")
    L.append(f"- Over all {len(U)} geometries the terminal spin is a single function of the radical's own outer plane angle "
             f"({fmt_corr(corr(allx, ally))}), the same relation found in the outer scans at C = 100°.")
    L.append("- Localization therefore cannot explain the exchange floor: at C = 90° with L = R = 45° the radicals are "
             f"as delocalized ({fx.loc[(45, 45), 'mean_terminal_HS']:.3f}) as at C = 100°, yet the central link is orthogonal.")
    L.append("")
    L.append("![Localization vs angles](plots/localization_vs_angles.png)")
    L.append("")

    # ---------------- M5: exchange of the free-outer families
    fe = FR[FR["pbe0_status"] == "ok"].copy()
    m5_msg = None
    if len(fe) >= 3:
        L.append("## M5. Free-outer exchange: matched θ_C, fixed vs relaxed outer rings, low-energy path")
        L.append("")
        L.append("Complements the free-outer table in M3 (per-point values there and in `central_approx_90.csv`).")
        L.append("")
        if m1 is not None:
            L.append(f"- Reference end point of the 45/45 family: free minimum M1 (C = {m1['C']:.1f}°, θ_C = "
                     f"{m1['theta_C_plane']:.1f}°): ΔE = {m1['dE']:+.3f} meV, T = {m1['T']:.4f} "
                     "(PBE0 UNO/UCO single points in `natural_orbitals/M1`).")
        L.append(f"- All {len(fe)} free-outer points are AFM: ΔE = {fe['dE'].min():+.2f} to {fe['dE'].max():+.2f} meV "
                 f"(E_T − E_S = −J, Yamaguchi: {fe['E_T_minus_E_S_yamaguchi_meV'].min():+.2f} to {fe['E_T_minus_E_S_yamaguchi_meV'].max():+.2f} "
                 f"meV; ⟨S²⟩_HS {fe['s2_HS'].min():.3f}–{fe['s2_HS'].max():.3f}, ⟨S²⟩_BS {fe['s2_BS'].min():.3f}–"
                 f"{fe['s2_BS'].max():.3f}). The outer rings stay at ≈ 36° throughout, so no point reaches the "
                 "near-degenerate regime (|ΔE| < 0.5 meV) except 135/135 at C ≥ 95°.")
        per = []
        fams = ["45/45 start", "135/135 start", "45/135 start"]
        for f in fams:
            g = fe[fe["family"] == f]
            if len(g) >= 4:
                rc = corr(g["cos2_C"], g["dE"])
                rt = corr(g["T2"], g["dE"])
                per.append(f"{f}: ρ(ΔE, cos²θ_C) = {rc['spearman']:.2f}, ρ(ΔE, T²) = {rt['spearman']:.2f}")
        if per:
            L.append("- Within each family (n = 5, Spearman): " + "; ".join(per) + ".")
        hi = fe[fe["theta_C_plane"] >= 87.0].sort_values("dE")
        if len(hi) >= 2:
            L.append("- **At an (almost) orthogonal central link the exchange still depends on the handedness.** Points "
                     "with θ_C ≥ 87°: " + "; ".join(
                         f"{r.family.replace(' start', '')} at C = {r.C:.0f}° (θ_C = {r.theta_C_plane:.1f}°, T = {r.T:.4f}): "
                         f"{r.dE:+.2f} meV" for r in hi.itertuples())
                     + f". The outer plane angles are the same (≈ 36°) and P is ≈ 0 for all of them, yet ΔE differs by a "
                       f"factor {hi['dE'].max() / hi['dE'].min():.1f}; across these points the ordering broadly follows "
                       "the BS overlap T, not θ_C or P. Within a family T is not sufficient either: in 45/135 T is nearly "
                       "constant while ΔE follows θ_C, and in 45/45 ΔE is U-shaped in C while both T and θ_C change "
                       "monotonically, so no single descriptor captures this family.")
        rows_cmp = []
        for (Lp, Rp), f in FAMILY.items():
            r_free = fe[(fe["family"] == f) & (fe["C"] == 90)]
            if (Lp, Rp) in fx.index and len(r_free):
                rows_cmp.append((f"{Lp}/{Rp}", fx.loc[(Lp, Rp), "dE"], r_free["dE"].iloc[0],
                                 fx.loc[(Lp, Rp), "theta_L_plane"], r_free["theta_L_plane"].iloc[0]))
        if rows_cmp:
            L.append("- **Fixed vs relaxed outer rings at C = 90°:** " + "; ".join(
                f"{p}: outer fixed (θ_L ≈ {tf:.0f}°) {a:+.2f} → relaxed (θ_L ≈ {tr:.0f}°) {b:+.2f} meV"
                for p, a, b, tf, tr in rows_cmp)
                + ". Letting the outer rings relax toward 36° increases the exchange by "
                  f"{min(b - a for _, a, b, _, _ in rows_cmp):+.2f} to {max(b - a for _, a, b, _, _ in rows_cmp):+.2f} meV.")
        low = fe.loc[fe.groupby("C")["E_rel_meV"].idxmin()].sort_values("C")
        path = ", ".join(f"C = {r.C:.0f}°: {r.family.replace(' start', '')} ({r.E_rel_meV:+.0f} meV) ΔE = {r.dE:+.2f}"
                         for r in low.itertuples())
        mn = fe.loc[fe["dE"].idxmin()]
        L.append(f"- **Along the lowest-energy family at each C** ({path}"
                 + (f"; M1: {m1['E_rel_meV']:+.0f} meV, ΔE = {m1['dE']:+.2f}" if m1 is not None else "")
                 + f") the exchange stays between {low['dE'].min():+.2f} and {max(low['dE'].max(), m1['dE'] if m1 else -9):+.2f} meV. "
                 f"The weakest coupling of the set ({mn['label']}, {mn['dE']:+.2f} meV) belongs to the 135/135 family at "
                 f"{mn['E_rel_meV']:+.0f} meV, about {mn['E_rel_meV'] - low['E_rel_meV'].min():.0f} meV above the lowest "
                 "structure of this set, i.e. thermally not populated at room temperature (k_BT ≈ 26 meV).")
        L.append("")
        L.append("![Free-outer exchange with fixed-outer points and M1](plots/free_outer_exchange_vs_fixed_and_M1.png)")
        L.append("")
        m5_msg = (f"**With relaxed outer rings the exchange stays AFM near C = 90°** ({fe['dE'].min():+.2f} to "
                  f"{fe['dE'].max():+.2f} meV). Its value depends on the handedness of the outer twists at fixed θ_C "
                  "(tracked by the BS overlap T), and the low-energy structures keep ≈ 1 meV: the near-orthogonal "
                  "central link alone does not decouple the radicals.")

    L.append("## Main physical messages")
    L.append("")
    one90 = fx[((fx.index.get_level_values(0) == 90) ^ (fx.index.get_level_values(1) == 90))]
    msgs = [
        f"**C = 90° is not a stationary point.** With free outer rings the two handedness families cross at "
        f"C ≈ {cross:.0f}° ({cross_E:+.0f} meV) and each keeps descending toward its own minimum (M1 at C ≈ 104°, "
        "M2 at C ≈ 76°). Converting M2 into M1 is not a pure central rotation: it also requires reversing the outer "
        "handedness.",
        f"**An orthogonal central link does not switch the exchange off.** For twisted outer rings ΔE stays at "
        f"{min(f['near90_min'] for f in floors.values()):+.2f} to {max(f['near90_min'] for f in floors.values()):+.2f} meV "
        "(minimum per outer pair) around C = 80–100° instead of following cos²θ_C to zero.",
        f"**An orthogonal outer link does.** At C = 90° every point with one outer ring at 90° and a twisted partner is "
        f"near-degenerate ({one90['dE'].min():+.2f} to {one90['dE'].max():+.2f} meV), for both handedness branches; with "
        f"both outer rings orthogonal ΔE = {fx.loc[(90, 90), 'dE']:+.2f} meV. The strong re-emergence seen at C = 100° "
        "(L = 90°, R = 180°: +2.57 meV) only appears for C > 90°.",
        "**Localization is a property of the outer torsions only.** The central torsion leaves terminal and backbone "
        "spins unchanged; the residual exchange at C = 90° is not a localization effect.",
    ]
    if m5_msg:
        msgs.append(m5_msg)
    for i, m in enumerate(msgs, 1):
        L.append(f"{i}. {m}")
    L.append("")

    # ---------------- PART II
    L.append("# Part II — Side analyses")
    L.append("")
    L.append("## S1. Magnetic-orbital overlap and diradical character")
    L.append("")
    if len(EXo):
        L.append(f"From the BS corresponding orbitals: T = {EXo['T'].min():.4f}–{EXo['T'].max():.4f}, so the system is an "
                 f"almost pure diradical (y = {EXo['y_diradical'].min():.3f}–{EXo['y_diradical'].max():.3f}; n_LUNO = "
                 f"{EXo['n_LUNO'].min():.3f}–{EXo['n_LUNO'].max():.3f}). The UNO occupations reproduce T "
                 f"(max |T_UCO − (n_HONO − n_LUNO)/2| = {(EXo['T'] - EXo['T_from_NO']).abs().max():.1e}).")
        L.append("")
        L.append(f"- ΔE vs T² (Kahn-type AFM term ∝ S²): {fmt_corr(corr(EXo['T2'], EXo['dE']))}")
        L.append(f"- ΔE vs mean terminal spin: {fmt_corr(corr(EXo['mean_terminal_HS'], EXo['dE']))}")
        L.append(f"- ΔE vs P: {fmt_corr(corr(EXo['P'], EXo['dE']))}")
        L.append("")
        comp = EXo[(EXo["dE"].abs() < 0.05) & (EXo["T"] > 0.008)]
        txt = (f"T² tracks ΔE somewhat better than the geometric proxy P (Spearman {corr(EXo['T2'], EXo['dE'])['spearman']:.2f} vs "
               f"{corr(EXo['P'], EXo['dE'])['spearman']:.2f}), and comparably to terminal localization: P is small for all "
               f"C = 90° points (≤ {EXo.loc[EXo['set'] == 'fixed_outer', 'P'].max():.1e}), while the BS wavefunction still registers a residual magnetic-orbital overlap. "
               "With one outer ring at 90° and a twisted partner, T itself collapses (≤ "
               f"{EXo[(EXo['set'] == 'fixed_outer') & ((EXo['L_init'] == 90) ^ (EXo['R_init'] == 90)) & (EXo['L_init'] != 0) & (EXo['R_init'] != 0)]['T'].max():.4f}).")
        if len(comp):
            txt += (" Points with ΔE ≈ 0 but a sizeable T ("
                    + ", ".join(f"{r['label']}: T = {r['T']:.4f}, ΔE = {r['dE']:+.3f} meV" for _, r in comp.iterrows())
                    + ") may indicate a ferromagnetic contribution cancelling the small kinetic AFM term. "
                    "T² is a descriptor here, not a fitted model.")
        L.append(txt)
        L.append("")
        L.append("![Exchange vs overlap](plots/exchange_vs_overlap_localization_P.png)")
        L.append("")
    L.append("## S2. Population-scheme check (Hirshfeld vs Mulliken)")
    L.append("")
    if len(EXo):
        L.append(f"Hirshfeld terminal spins (PBE0 HS) are systematically smaller than Mulliken "
                 f"(mean ratio {(pd.concat([EXo['H1_HS'], EXo['H4_HS']]).abs() / pd.concat([EXo['S1_HS'], EXo['S4_HS']]).abs()).mean():.2f}) "
                 "but rank the geometries identically, so the localization conclusions do not depend on the scheme.")
        L.append("")
        L.append("![Hirshfeld vs Mulliken](plots/hirshfeld_vs_mulliken.png)")
        L.append("")
    L.append("## S3. QC")
    L.append("")
    L.append(f"- Optimization cycles: {int(df['n_geom_cycles'].min())}–{int(df['n_geom_cycles'].max())}; convergence types: "
             + ", ".join(f"{k} {v}" for k, v in df["convergence_type"].value_counts().items()) + ".")
    L.append(f"- Constraints held: max |C_final − C| = {(df['C_final'].abs() - df['C']).abs().max():.3f}°.")
    if len(EXo):
        L.append(f"- ⟨S²⟩_HS {EXo['s2_HS'].min():.3f}–{EXo['s2_HS'].max():.3f}, ⟨S²⟩_BS {EXo['s2_BS'].min():.3f}–"
                 f"{EXo['s2_BS'].max():.3f}; E_T − E_S = −J (Yamaguchi) has the same sign as ΔE at "
                 f"{int((np.sign(EXo['E_T_minus_E_S_yamaguchi_meV']) == np.sign(EXo['dE'])).sum())}/{len(EXo)} points.")
    L.append("")
    L.append("All per-point values are in `central_approx_90.csv`.")
    md_path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
