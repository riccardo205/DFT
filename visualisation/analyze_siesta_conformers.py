#!/usr/bin/env python3
"""
SIESTA (PBE, spin-polarized) relaxed random conformers of short-2OS, ferro (FM) and
antiferro (AFM) starts, placed on the ORCA torsion landscape.

Sources (ferro / antiferro runs only; *_4radical runs are ignored):
  Short-2OS/runs/siesta/main_conf/slow_lowT_lowest_forces/conf_*_{ferro,antiferro}
  Short-2OS/runs/siesta/conformers/slow_lowT_lowest_forces/conf_*_{ferro,antiferro}

For every run: final geometry (atom order mapped onto the ORCA numbering by an
element-aware graph isomorphism), free energy, max force vs MD.MaxForceTol,
total spin moment, Hirshfeld fragment spins. Electronic state is classified from
the spins (FM: |S| ~ 2; AFM: terminal fragments of opposite sign) so collapsed
closed-shell solutions are never used as FM or AFM.

Then the same descriptors as analyze_fluorene_angle.py (L, C, R, theta_L/C/R,
theta_FF, family) are computed and the conformers are overlaid on:
  * fluorene_angle_analysis (E vs theta_FF, families, ORCA PBE0 exchange)
  * angle_scan_relaxation (E vs central torsion C, E_min(C))
and matched by RMSD (over graph automorphisms and mirror images) to every
ORCA geometry, in particular the free minima M1-M5.

Energies of the two codes are never put on one absolute scale: each set is
relative to its own lowest proper state of the same spin (SIESTA FM <-> ORCA HS).
Exchange: dE = E_FM - E_AFM (> 0: AFM lower), adiabatic (each state relaxed on
its own) and PBE, so only qualitatively comparable to the PBE0 vertical values.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from matplotlib.lines import Line2D
from networkx.algorithms import isomorphism as iso
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_angle_scan import (  # noqa: E402
    build_adjacency,
    check_chemical_integrity,
    outer_terminal_fragment,
    reference_bonds,
    split_fragment_at_bond,
)
from analyze_fluorene_angle import (  # noqa: E402
    CENTER,
    FAMILIES,
    FAMILY_ORDER,
    FAMILY_STYLE,
    GRID,
    INK,
    INK_2,
    INK_3,
    LEFT,
    MUTED,
    RIGHT,
    RINGS,
    SERIES,
    family_of,
    fluorene_descriptors,
    fold180,
    style_axes,
)
from analyze_free_relaxation import kabsch_rmsd, plane_angle, signed  # noqa: E402

EV_TO_MEV = 1000.0
SIESTA_C = SERIES[1]   # orange: SIESTA, everywhere
ORCA_C = "#b9b8b2"     # ORCA constrained points (background)

FREE_E_RE = re.compile(r"siesta: FreeEng =\s+(-?\d+\.\d+)")
SPIN_RE = re.compile(r"spin moment: \{S\} , \|S\| = \{\s*(\S+)\s+(\S+)\s+(\S+)\s*\}\s+(\S+)")
MAXF_RE = re.compile(r"^\s+Max\s+(\d+\.\d+)", re.M)
FTOL_RE = re.compile(r"^\s*MD\.MaxForceTol\s+(\S+)", re.M | re.I)
NSTEP_RE = re.compile(r"^\s*MD\.Steps\s+(\d+)", re.M | re.I)


# ============================================================
# PARSING
# ============================================================
def element_graph(atoms):
    adj = build_adjacency(atoms)
    g = nx.Graph()
    for i, s in enumerate(atoms.get_chemical_symbols()):
        g.add_node(i, el=s)
    g.add_edges_from((i, j) for i in range(len(atoms)) for j in adj[i] if j > i)
    return g


def match_el(a, b):
    return a["el"] == b["el"]


def last_hirshfeld_sz(text):
    blocks = text.split("Hirshfeld Atomic Populations:")
    if len(blocks) < 2:
        return None
    sz = {}
    for line in blocks[-1].splitlines()[2:]:
        p = line.split()
        if len(p) != 5 or not p[0].isdigit():
            break
        sz[int(p[0]) - 1] = float(p[3])
    return sz


def parse_run(d):
    text = (d / "siesta.out").read_text(errors="ignore")
    fdf = next(d.glob("conf_*.fdf")).read_text(errors="ignore")
    e = [float(x) for x in FREE_E_RE.findall(text)]
    spins = SPIN_RE.findall(text)
    maxf = [float(x) for x in MAXF_RE.findall(text)]
    ftol = float(FTOL_RE.search(fdf).group(1)) if FTOL_RE.search(fdf) else 0.04
    nsteps = int(NSTEP_RE.search(fdf).group(1)) if NSTEP_RE.search(fdf) else None
    n_moves = text.count("Begin CG opt. move")
    return {
        "E_eV": e[-1] if e else np.nan,
        "S_total": float(spins[-1][2]) if spins else np.nan,
        "S_abs": float(spins[-1][3]) if spins else np.nan,
        "max_force": maxf[-1] if maxf else np.nan,
        "force_tol": ftol,
        "n_cg_moves": n_moves,
        "max_steps": nsteps,
        "normal_exit": (d / "0_NORMAL_EXIT").exists(),
        "sz": last_hirshfeld_sz(text),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    repo = Path(__file__).resolve().parent.parent
    siesta_roots = {
        "main_conf": repo / "Short-2OS" / "runs" / "siesta" / "main_conf" / "slow_lowT_lowest_forces",
        "conformers": repo / "Short-2OS" / "runs" / "siesta" / "conformers" / "slow_lowT_lowest_forces",
    }
    scans = repo / "runs" / "orca" / "short-2OS" / "angle_scans"
    orca_csv = scans / "fluorene_angle_analysis" / "fluorene_angle_analysis.csv"
    out_dir = scans / "siesta_conformer_comparison"
    baseline_xyz = repo / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    SAME_MINIMUM_RMSD = 0.30   # A, as in analyze_free_relaxation
    FM_S_TARGET, S_TOL = 2.0, 0.2
    AFM_MIN_TERMINAL_SPIN = 0.3
    out_dir.mkdir(exist_ok=True)

    ref = read(baseline_xyz)
    g_ref = element_graph(ref)
    n_atoms = len(ref)
    adj = build_adjacency(ref)
    bonded_pairs = {(i, j) for i in range(n_atoms) for j in adj[i] if j > i}
    bonds = reference_bonds(adj, ref)
    autos = list(iso.GraphMatcher(g_ref, g_ref, node_match=match_el).isomorphisms_iter())
    f1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    f4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    centre = sorted(set(range(n_atoms)) - set(f1) - set(f4))
    f2, f3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), centre)
    sym = ref.get_chemical_symbols()
    flu1 = [i for i in f1 if sym[i] == "C"]
    flu4 = [i for i in f4 if sym[i] == "C"]

    # ---- ORCA reference data ----
    orca = pd.read_csv(orca_csv)
    orca_pos = {r.label: read(scans / r.geometry_file).get_positions() for r in orca.itertuples()}

    # ---- parse SIESTA runs ----
    rows, pos_of = [], {}
    for set_name, root in siesta_roots.items():
        for d in sorted(root.glob("conf_*_*ferro")):
            m = re.match(r"conf_(\d+)_(ferro|antiferro)$", d.name)
            if not m:
                continue
            rec = parse_run(d)
            atoms = read(d / "short-2OS.xyz")
            g = element_graph(atoms)
            gm = iso.GraphMatcher(g_ref, g, node_match=match_el)
            if not gm.is_isomorphic():
                raise RuntimeError(f"{d}: connectivity differs from the reference molecule")
            mp = next(gm.isomorphisms_iter())   # ORCA index -> SIESTA index
            order = [mp[i] for i in range(n_atoms)]
            pos = atoms.get_positions()[order]
            mapped = Atoms(sym, positions=pos)
            sz = rec.pop("sz")
            sz = [sz[mp[i]] for i in range(n_atoms)] if sz else None
            label = f"{set_name}/conf_{m.group(1)}"
            row = {"set": set_name, "conf": int(m.group(1)), "label": label, "state": m.group(2),
                   "run_dir": str(d.relative_to(repo))}
            row.update(rec)
            row["converged"] = bool(rec["max_force"] <= rec["force_tol"] and rec["normal_exit"])
            if sz:
                row["S1"] = float(sum(sz[i] for i in f1))
                row["S4"] = float(sum(sz[i] for i in f4))
                row["S2"] = float(sum(sz[i] for i in f2))
                row["S3"] = float(sum(sz[i] for i in f3))
            integ = check_chemical_integrity(mapped, bonds, bonded_pairs, 0.30, 1.9, 1.5)
            row["geometry_valid"] = integ["valid"]
            L, C, R = (signed(mapped.get_dihedral(*q)) for q in (LEFT, CENTER, RIGHT))
            if C < 0:  # mirror image -> C > 0 representative
                L, C, R = -L, -C, -R
            row.update({"L": L, "C": C, "R": R, "L_fold": fold180(L), "R_fold": fold180(R)})
            row["outer_hi"], row["outer_lo"] = max(row["L_fold"], row["R_fold"]), min(row["L_fold"], row["R_fold"])
            row.update({f"theta_{a}": plane_angle(pos, *RINGS[a]) for a in "LCR"})
            row.update(fluorene_descriptors(pos, flu1, flu4))
            row["family"] = family_of(row["L_fold"], C, row["R_fold"])
            row["P_hop"] = float(np.prod([np.cos(np.radians(row[f"theta_{a}"])) ** 2 for a in "LCR"]))
            rows.append(row)
            pos_of[(label, m.group(2))] = pos
    df = pd.DataFrame(rows)

    # ---- electronic state ----
    def state_kind(r):
        if r["state"] == "ferro":
            return "FM triplet" if abs(r["S_total"] - FM_S_TARGET) < S_TOL else "collapsed (closed shell)"
        if pd.notna(r.get("S1")) and min(abs(r["S1"]), abs(r["S4"])) > AFM_MIN_TERMINAL_SPIN and r["S1"] * r["S4"] < 0:
            return "AFM broken symmetry"
        return "collapsed (closed shell)"
    df["electronic_state"] = df.apply(state_kind, axis=1)

    # ---- RMSD matching to ORCA (min over graph automorphisms and mirror image) ----
    def best_rmsd(p, q):
        best = np.inf
        for a in autos:
            perm = [a[i] for i in range(n_atoms)]
            for mirror in (1.0, -1.0):
                pp = p[perm] * np.array([mirror, 1.0, 1.0])
                best = min(best, kabsch_rmsd(pp, q))
        return best

    near_lab, near_rmsd, near_min, near_min_rmsd = [], [], [], []
    free = orca[orca["kind"] == "free"]
    free_best = free.loc[free.groupby("minimum")["E_meV"].idxmin()]
    for r in df.itertuples():
        p = pos_of[(r.label, r.state)]
        d_all = {lab: best_rmsd(p, q) for lab, q in orca_pos.items()}
        lab = min(d_all, key=d_all.get)
        near_lab.append(lab)
        near_rmsd.append(d_all[lab])
        dm = {fr.minimum: d_all[fr.label] for fr in free_best.itertuples()}
        mm = min(dm, key=dm.get)
        near_min.append(mm)
        near_min_rmsd.append(dm[mm])
    df["nearest_orca"] = near_lab
    df["nearest_orca_rmsd"] = near_rmsd
    df["nearest_orca_minimum"] = near_min
    df["nearest_orca_minimum_rmsd"] = near_min_rmsd
    df["same_as_orca_minimum"] = df["nearest_orca_minimum_rmsd"] < SAME_MINIMUM_RMSD
    orca_by = orca.set_index("label")
    df["nearest_orca_E_meV"] = [orca_by.loc[lab, "E_meV"] for lab in df["nearest_orca"]]
    df["nearest_orca_dE_PBE0_meV"] = [orca_by.loc[lab, "dE_PBE0_meV"] for lab in df["nearest_orca"]]

    # ---- relative energies (per spin state, own reference) ----
    fm = df[(df["state"] == "ferro") & (df["electronic_state"] == "FM triplet") & df["geometry_valid"]]
    afm = df[(df["state"] == "antiferro") & (df["electronic_state"] == "AFM broken symmetry") & df["geometry_valid"]]
    e_fm0 = fm["E_eV"].min()
    e_afm0 = afm["E_eV"].min()
    df["E_rel_meV"] = np.where(df["state"] == "ferro", (df["E_eV"] - e_fm0) * EV_TO_MEV,
                               (df["E_eV"] - e_afm0) * EV_TO_MEV)
    ref_fm = df.loc[fm["E_eV"].idxmin(), "label"]
    # anchor: the converged FM conformer that is closest to an ORCA free minimum (same structure, RMSD < 0.3 A)
    # is placed at that minimum's ORCA energy, so both codes share one structure as reference
    cand = df[(df["state"] == "ferro") & (df["electronic_state"] == "FM triplet") & df["converged"]
              & df["same_as_orca_minimum"]]
    anchor = cand.loc[cand["nearest_orca_minimum_rmsd"].idxmin()]
    m_e = orca[(orca["kind"] == "free") & (orca["minimum"] == anchor["nearest_orca_minimum"])]["E_meV"].min()
    shift = m_e - anchor["E_rel_meV"]
    df["E_anchored_meV"] = np.where(df["state"] == "ferro", df["E_rel_meV"] + shift, np.nan)

    # ---- FM/AFM pairs: exchange ----
    pairs = []
    for lab, g in df.groupby("label"):
        g = g.set_index("state")
        if not {"ferro", "antiferro"} <= set(g.index):
            continue
        a, b = g.loc["ferro"], g.loc["antiferro"]
        ok = a["electronic_state"] == "FM triplet" and b["electronic_state"] == "AFM broken symmetry"
        pairs.append({
            "label": lab, "set": a["set"], "conf": a["conf"], "valid_pair": bool(ok),
            "both_converged": bool(a["converged"] and b["converged"]),
            "dE_FM_AFM_meV": (a["E_eV"] - b["E_eV"]) * EV_TO_MEV if ok else np.nan,
            "rmsd_FM_AFM": kabsch_rmsd(pos_of[(lab, "ferro")], pos_of[(lab, "antiferro")]),
            "S1_AFM": b.get("S1"), "S4_AFM": b.get("S4"), "S1_FM": a.get("S1"), "S4_FM": a.get("S4"),
            **{k: a[k] for k in ("L", "C", "R", "L_fold", "R_fold", "theta_L", "theta_C", "theta_R", "theta_FF",
                                 "family", "P_hop", "E_rel_meV", "E_anchored_meV", "nearest_orca", "nearest_orca_rmsd",
                                 "nearest_orca_minimum", "nearest_orca_minimum_rmsd", "nearest_orca_dE_PBE0_meV",
                                 "max_force", "converged")},
            "max_force_AFM": b["max_force"], "converged_AFM": b["converged"],
            "state_FM": a["electronic_state"], "state_AFM": b["electronic_state"],
        })
    pairs = pd.DataFrame(pairs)

    ctx = {"ref_fm": ref_fm, "e_fm0": e_fm0, "e_afm0": e_afm0, "n_autos": len(autos),
           "anchor_label": anchor["label"], "anchor_min": anchor["nearest_orca_minimum"],
           "anchor_rmsd": anchor["nearest_orca_minimum_rmsd"], "shift": shift}
    df.drop(columns=[]).to_csv(out_dir / "siesta_conformers.csv", index=False, float_format="%.6f")
    pairs.to_csv(out_dir / "siesta_exchange_pairs.csv", index=False, float_format="%.6f")
    make_plots(df, pairs, orca, scans, out_dir)
    write_report(df, pairs, orca, ctx, out_dir / "siesta_conformer_comparison.md")
    print(df[["label", "state", "electronic_state", "converged", "max_force", "E_rel_meV", "theta_FF", "C",
              "family", "nearest_orca_minimum", "nearest_orca_minimum_rmsd"]].round(2).to_string())
    print(pairs[["label", "valid_pair", "both_converged", "dE_FM_AFM_meV", "rmsd_FM_AFM"]].round(3).to_string())
    print(f"Wrote {out_dir}")


# ============================================================
# PLOTS
# ============================================================
def orca_grid_emin_c(scans):
    grid = pd.read_csv(scans / "angle_scan_relaxation" / "angle_scan.csv")
    grid = grid[grid["usable"]]
    return grid


def siesta_scatter(ax, sub, x, y, label=True, ms=70):
    for r in sub.itertuples():
        filled = r.converged
        ax.scatter(getattr(r, x), getattr(r, y), marker="D", s=ms, zorder=7, linewidth=1.4,
                   facecolor=SIESTA_C if filled else "white", edgecolor=SIESTA_C)
        if label:
            short = ("m" if r.set == "main_conf" else "c") + str(r.conf)
            ax.annotate(short, (getattr(r, x), getattr(r, y)), xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=INK_2, zorder=8)


def orca_minima(ax, orca, x, y="E_meV", ymax=None):
    f = orca[orca["kind"] == "free"]
    f = f.loc[f.groupby("minimum")["E_meV"].idxmin()]
    for r in f.itertuples():
        if ymax is not None and getattr(r, y) > ymax:
            continue
        ax.scatter(getattr(r, x), getattr(r, y), marker="*", s=230, color="white", edgecolor=INK, lw=1.4, zorder=6)
        ax.annotate(r.minimum, (getattr(r, x), getattr(r, y)), xytext=(6, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=INK, zorder=6)


def legend_handles(extra=()):
    h = [Line2D([], [], marker="o", ls="", color=ORCA_C, label="ORCA R2SCAN-3C HS, constrained geometries"),
         Line2D([], [], marker="*", ls="", ms=13, mfc="white", mec=INK, label="ORCA free minima M1–M5"),
         Line2D([], [], marker="D", ls="", mfc=SIESTA_C, mec=SIESTA_C, label="SIESTA PBE FM, converged"),
         Line2D([], [], marker="D", ls="", mfc="white", mec=SIESTA_C, label="SIESTA PBE FM, force > tol (50-step cap)")]
    return h + list(extra)


def make_plots(df, pairs, orca, scans, out_dir):
    fm = df[(df["state"] == "ferro") & (df["electronic_state"] == "FM triplet") & df["geometry_valid"]]
    con = orca[orca["kind"] == "constrained"]
    xlab_ff = "angle between terminal fluorene planes θ$_{FF}$ (°)"
    ylab = "HS / FM energy rel. to ORCA M1 (meV)"

    # ---- 1. energy vs theta_FF ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for ax, ymax in zip(axes, (None, 300)):
        ax.scatter(con["theta_FF"], con["E_meV"], color=ORCA_C, s=16, zorder=2)
        orca_minima(ax, orca, "theta_FF", ymax=ymax)
        siesta_scatter(ax, fm, "theta_FF", "E_anchored_meV", label=ymax is not None)
        ax.set_xlim(-2, 92)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xlabel(xlab_ff)
        ax.set_ylabel(ylab)
        if ymax:
            ax.set_ylim(-15, ymax)
        style_axes(ax)
    axes[0].set_title("(a) all geometries", loc="left", fontsize=10)
    axes[1].set_title("(b) low-energy region; labels m = main_conf, c = conformers", loc="left", fontsize=10)
    axes[0].legend(handles=legend_handles(), fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("SIESTA FM conformers on the ORCA fluorene-angle landscape (SIESTA anchored on the shared M2 "
                 "structure)", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "energy_vs_fluorene_angle_siesta_orca.png", dpi=200)
    plt.close(fig)

    # ---- 2. energy vs central torsion ----
    grid = orca_grid_emin_c(scans)
    emin = grid.groupby("C")["relative_energy_meV"].min()
    offset = float(orca.loc[(orca["source"] == "grid") & (orca["label"] == "L045_C100_R045"), "E_meV"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for ax, ymax in zip(axes, (None, 400)):
        ax.scatter(con["C"], con["E_meV"], color=ORCA_C, s=16, zorder=2)
        ax.plot(emin.index, emin.values + offset, color=INK, lw=2, marker="o", ms=4, zorder=4,
                label="ORCA E$_{min}$(C) over the L×R grid")
        orca_minima(ax, orca, "C", ymax=ymax)
        siesta_scatter(ax, fm, "C", "E_anchored_meV", label=ymax is not None)
        ax.axvline(90, color=INK_3, lw=0.8, ls=":")
        ax.set_xlim(10, 170)
        ax.set_xticks(range(20, 161, 20))
        ax.set_xlabel("central torsion C (°, relaxed; mirror images folded to C > 0)")
        ax.set_ylabel(ylab)
        if ymax:
            ax.set_ylim(-15, ymax)
        style_axes(ax)
    axes[0].set_title("(a) all geometries", loc="left", fontsize=10)
    axes[1].set_title("(b) low-energy region", loc="left", fontsize=10)
    axes[0].legend(handles=legend_handles([Line2D([], [], color=INK, lw=2, marker="o", ms=4,
                                                  label="ORCA E$_{min}$(C) over the L×R grid")]),
                   fontsize=7.5, frameon=False, loc="upper center")
    fig.suptitle("SIESTA FM conformers vs central torsion, with the ORCA angle-scan profile", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "energy_vs_central_angle_siesta_orca.png", dpi=200)
    plt.close(fig)

    # ---- 3. torsion maps ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    for f in FAMILY_ORDER:
        g = con[con["family"] == f]
        if g.empty:
            continue
        col, mk = FAMILY_STYLE[f]
        ax.scatter(g["C"], g["theta_FF"], color=col, marker=mk, s=18, alpha=0.45, edgecolor="none",
                   label=f"ORCA {FAMILIES[f][0]} ({f})", zorder=2)
    orca_minima(ax, orca, "C", y="theta_FF")
    siesta_scatter(ax, fm, "C", "theta_FF")
    ax.set_xlabel("central torsion C (°)")
    ax.set_ylabel(xlab_ff)
    ax.set_title("(a) where the conformers sit: C vs θ$_{FF}$", loc="left", fontsize=10)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", ncol=2)
    style_axes(ax)
    ax = axes[1]
    ax.scatter(np.maximum(con["L_fold"], con["R_fold"]), np.minimum(con["L_fold"], con["R_fold"]), color=ORCA_C,
               s=14, zorder=2)
    f = orca[orca["kind"] == "free"]
    f = f.loc[f.groupby("minimum")["E_meV"].idxmin()]
    for r in f.itertuples():
        hi, lo = max(r.L_fold, r.R_fold), min(r.L_fold, r.R_fold)
        ax.scatter(hi, lo, marker="*", s=230, color="white", edgecolor=INK, lw=1.4, zorder=6)
        ax.annotate(r.minimum, (hi, lo), xytext=(-22, 6), textcoords="offset points", fontsize=9, fontweight="bold")
    siesta_scatter(ax, fm, "outer_hi", "outer_lo")
    for v in (0,):
        ax.axhline(v, color=INK_3, lw=0.6)
        ax.axvline(v, color=INK_3, lw=0.6)
    ax.set_xlim(-92, 92)
    ax.set_ylim(-92, 92)
    ax.set_xlabel("larger outer torsion, folded mod 180° (°)")
    ax.set_ylabel("smaller outer torsion, folded mod 180° (°)")
    ax.set_title("(b) outer torsions, L↔R ordered (fluorenyl flip folded; C > 0)", loc="left", fontsize=10)
    ax.set_aspect("equal")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "torsion_maps_siesta_orca.png", dpi=200)
    plt.close(fig)

    # ---- 4. energy spectrum (ladders) ----
    fig, ax = plt.subplots(figsize=(9, 6.4))
    sf = df[(df["state"] == "ferro") & (df["electronic_state"] == "FM triplet")].sort_values("E_anchored_meV")
    ymax = 420
    last_ty = -np.inf
    for r in sf.itertuples():
        y = r.E_anchored_meV
        if y > ymax:
            continue
        ax.plot([-0.3, 0.3], [y, y], color=SIESTA_C, lw=2, ls="-" if r.converged else "--")
        tag = ("m" if r.set == "main_conf" else "c") + str(r.conf) + f"  θ$_{{FF}}$={r.theta_FF:.0f}°, C={r.C:.0f}°"
        ty = max(y, last_ty + 9.0)
        last_ty = ty
        ax.plot([-0.62, -0.3], [ty, y], color=INK_3, lw=0.5)
        ax.text(-0.65, ty, tag, fontsize=6.8, va="center", ha="right", color=INK_2)
        if r.same_as_orca_minimum:
            ye = f.loc[f["minimum"] == r.nearest_orca_minimum, "E_meV"].iloc[0]
            ax.plot([0.3, 1.7], [y, ye], color=INK_3, lw=0.8, ls=":")
    for r in f.itertuples():
        ax.plot([1.7, 2.3], [r.E_meV, r.E_meV], color=INK, lw=2)
        ax.text(2.33, r.E_meV, f"{r.minimum}  θ$_{{FF}}$={r.theta_FF:.0f}°, C={r.C:.0f}°", fontsize=7.5,
                va="center", color=INK_2)
    n_hidden = int((sf["E_anchored_meV"] > ymax).sum())
    ax.set_xticks([0, 2])
    ax.set_xticklabels(["SIESTA PBE, FM", "ORCA R2SCAN-3C, HS free minima"])
    ax.set_xlim(-1.9, 3.4)
    ax.set_ylim(-15, ymax)
    ax.set_ylabel(ylab)
    ax.set_title("Energy spectrum of the relaxed conformers; dotted = same structure (RMSD < 0.3 Å), dashed = force "
                 f"above tolerance{f'; {n_hidden} level(s) > {ymax} meV not shown' if n_hidden else ''}",
                 loc="left", fontsize=9)
    style_axes(ax)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_spectrum_siesta_orca.png", dpi=200)
    plt.close(fig)

    # ---- 5. exchange vs angles ----
    ok = pairs[pairs["valid_pair"]]
    ox = orca.dropna(subset=["dE_PBE0_meV"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, (col, lab) in zip(axes, (("theta_FF", xlab_ff), ("C", "central torsion C (°)"),
                                     ("P_hop", "P = cos²θ$_L$ cos²θ$_C$ cos²θ$_R$"))):
        ax.scatter(ox[col], ox["dE_PBE0_meV"], color=ORCA_C, s=16, zorder=2)
        for r in ok.itertuples():
            filled = r.both_converged
            ax.scatter(getattr(r, col), r.dE_FM_AFM_meV, marker="D", s=70, zorder=7, lw=1.4,
                       facecolor=SIESTA_C if filled else "white", edgecolor=SIESTA_C)
            ax.annotate(("m" if r.set == "main_conf" else "c") + str(r.conf), (getattr(r, col), r.dE_FM_AFM_meV),
                        xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK_2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel(lab)
        style_axes(ax)
    axes[2].set_xscale("symlog", linthresh=1e-3)
    axes[2].set_xlim(0, 1.1 * max(ox["P_hop"].max(), ok["P_hop"].max() if len(ok) else 0))
    axes[0].set_ylabel("E$_{HS}$ − E$_{BS}$ (meV, symlog; > 0: AFM/BS lower)")
    h = [Line2D([], [], marker="o", ls="", color=ORCA_C, label="ORCA PBE0, vertical (HS and BS at the same geometry)"),
         Line2D([], [], marker="D", ls="", mfc=SIESTA_C, mec=SIESTA_C,
                label="SIESTA PBE, E$_{FM}$ − E$_{AFM}$, adiabatic, both converged"),
         Line2D([], [], marker="D", ls="", mfc="white", mec=SIESTA_C, label="SIESTA, ≥ 1 of the pair above force tol")]
    axes[0].legend(handles=h, fontsize=7.5, frameon=False, loc="upper left")
    if len(ok) > 2:
        rho = spearmanr(ok["theta_FF"], ok["dE_FM_AFM_meV"])[0], spearmanr(ok["P_hop"], ok["dE_FM_AFM_meV"])[0]
        axes[2].set_title(f"SIESTA: ρ(ΔE, θ$_{{FF}}$) = {rho[0]:.2f}, ρ(ΔE, P) = {rho[1]:.2f} (n = {len(ok)})",
                          loc="left", fontsize=9.5)
    fig.suptitle("Exchange splitting vs geometry: SIESTA conformers on top of the ORCA PBE0 data", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "exchange_vs_angles_siesta_orca.png", dpi=200)
    plt.close(fig)


# ============================================================
# REPORT
# ============================================================
def md_table(df, fmt="{:.1f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("–" if pd.isna(v) else str(v))
            for v in (r[c] for c in cols)) + " |")
    return "\n".join(out)


def write_report(df, pairs, orca, ctx, path):
    L = []
    w = L.append
    fm = df[df["state"] == "ferro"].set_index("label")
    af = df[df["state"] == "antiferro"].set_index("label")
    pr = pairs.set_index("label")
    ok = pairs[pairs["valid_pair"]]
    f = orca[orca["kind"] == "free"]
    f = f.loc[f.groupby("minimum")["E_meV"].idxmin()].set_index("minimum")
    collapsed = sorted(set(df.loc[df["electronic_state"] == "collapsed (closed shell)", "label"]))
    n_conv = int(pairs["both_converged"].sum())
    # ORCA PBE0 linear guide from exchange_analysis.md (C < 160): dE = 46.67 P + 0.43 meV
    ok = ok.assign(orca_guide=46.67 * ok["P_hop"] + 0.43)
    rr = ok["dE_FM_AFM_meV"] / ok["orca_guide"]
    lo_r, hi_r = rr[ok["P_hop"] < 0.01].median(), rr[ok["P_hop"] >= 0.01].median()
    matched = df[(df["state"] == "ferro") & df["same_as_orca_minimum"]]

    w("# SIESTA relaxed conformers on the ORCA torsion landscape\n")
    w("Generated by `visualisation/analyze_siesta_conformers.py`. SIESTA runs: "
      "`Short-2OS/runs/siesta/{main_conf,conformers}/slow_lowT_lowest_forces/conf_*_{ferro,antiferro}` "
      "(the `*_4radical` runs are not used). ORCA references: `fluorene_angle_analysis/` and "
      "`angle_scan_relaxation/angle_scan.md`.\n")
    w("## Summary\n")
    w(f"1. **{len(collapsed)} of the {df['label'].nunique()} conformers ({', '.join(collapsed)}) are not diradicals in "
      "SIESTA.** Both the ferro and the antiferro start converged to the same closed-shell solution (total and fragment "
      "spin 0, identical energy). These are the two near-planar structures (C ≈ 2°, θ_FF ≈ 0°), about 1.3 eV above "
      "the others; they carry no exchange information and are excluded from the energy and exchange comparisons.")
    w(f"2. **The SIESTA conformers fall into the ORCA low-energy basins, but none of them is M1.** The lowest SIESTA "
      f"conformer ({ctx['anchor_label']}) is ORCA's M2 (RMSD {ctx['anchor_rmsd']:.2f} Å, θ_FF "
      f"{fm.loc[ctx['anchor_label'], 'theta_FF']:.0f}° vs {f.loc['M2', 'theta_FF']:.0f}°); "
      + ", ".join(f"{r.label} = {r.nearest_orca_minimum} ({r.nearest_orca_minimum_rmsd:.2f} Å)"
                  for r in matched.itertuples() if r.label != ctx["anchor_label"])
      + f". The closest SIESTA structure to the ORCA global minimum M1 is "
      f"{fm['nearest_orca_minimum_rmsd'].where(fm['nearest_orca_minimum'] == 'M1').idxmin() if (fm['nearest_orca_minimum'] == 'M1').any() else '–'} "
      f"at {fm.loc[fm['nearest_orca_minimum'] == 'M1', 'nearest_orca_minimum_rmsd'].min():.2f} Å, i.e. a different "
      "structure. The random sampling never reached the C ≈ 104°, both-outer-co-twisted basin of M1.")
    c90 = fm[(fm["C"].between(85, 96)) & (fm["electronic_state"] == "FM triplet")]
    members = ", ".join(f"{i} [{fam}]" for i, fam in c90["family"].items())
    w(f"3. **A cluster of twisted conformers at C ≈ 85–96° that ORCA has not optimized freely** "
      f"({members}; θ_FF {c90['theta_FF'].min():.0f}–{c90['theta_FF'].max():.0f}°, "
      f"{c90['E_anchored_meV'].min():.0f}–{c90['E_anchored_meV'].max():.0f} meV on the ORCA scale). Their nearest "
      "ORCA geometries are the constrained C = 90 points (0.3–0.4 Å). Whether they are R2SCAN-3C minima is exactly "
      "what the pending `angle_scan_central_approx_90/free_outer` runs (or an ORCA relaxation started from these "
      "SIESTA structures) would decide.")
    w(f"4. **Same picture on both coordinates.** On E(C) every converged SIESTA conformer sits on or near the ORCA "
      "E_min(C) curve between C = 75° and 107°; on E(θ_FF) they occupy 50–80°, the region of M2–M4. Nothing near "
      "θ_FF ≈ 20° was found, consistent with the ORCA conclusion that small terminal-plane angles are high in energy "
      "for the isolated molecule.")
    w(f"5. **Exchange: SIESTA PBE gives small, always AFM splittings.** E_FM − E_AFM = "
      f"{ok['dE_FM_AFM_meV'].min():.2f}–{ok['dE_FM_AFM_meV'].max():.2f} meV over the {len(ok)} valid pairs "
      f"({n_conv} with both runs force-converged). FM and AFM relaxed to the same structure (RMSD ≤ "
      f"{ok['rmsd_FM_AFM'].max():.3f} Å), so the values are effectively vertical. Compared with the ORCA PBE0 trend at "
      f"the same hopping proxy P: near the decoupled limit (P < 0.01) they are similar ({lo_r:.2f}× the PBE0 guide, "
      f"median), for the better-conjugated conformers (P ≥ 0.01) about {hi_r:.2f}× (median). The "
      "ordering follows P loosely (Spearman ρ below), not θ_FF, as in the ORCA data, but the SIESTA set spans only a "
      "narrow geometry range, so the correlations are weak and not significant.\n")

    w("## Data and quality control\n")
    w("- Level of theory: SIESTA, PBE, TZP-type numerical basis (PAO.EnergyShift 5 meV), MeshCutoff 500 Ry, collinear "
      "spin, CG relaxation with MD.MaxForceTol 0.01 eV/Å and MD.Steps 50. FM start: +1 μB on C29 and C42 "
      "(SIESTA 1-based = ORCA radical atoms 28, 41); AFM start: +1 / −1.")
    w("- **Atom order**: SIESTA files are re-indexed onto the ORCA numbering by an element-aware graph isomorphism "
      f"(all runs have the reference connectivity; {ctx['n_autos']} automorphisms = fluorenyl flips × end swap, all "
      "of which leave the descriptors unchanged or folded). Structural comparison to ORCA uses the minimum Kabsch RMSD "
      "over these automorphisms and the mirror image.")
    w("- **Convergence**: a run counts as converged when the last max force ≤ MD.MaxForceTol. "
      f"{int((~df['converged']).sum())} of {len(df)} runs stopped at the 50-step cap with max force "
      f"{df.loc[~df['converged'], 'max_force'].min():.3f}–{df.loc[~df['converged'], 'max_force'].max():.3f} eV/Å; they are "
      "shown with open symbols and their energies/angles are provisional.")
    w("- **Electronic state** from the final spins: FM triplet if the total moment is 2 μB; AFM broken symmetry if "
      "the Hirshfeld spins of the two terminal fluorenyls exceed 0.3 with opposite sign.")
    w(f"- **Energy reference**: SIESTA PBE and ORCA R2SCAN-3C absolute energies are unrelated. The converged SIESTA FM "
      f"conformer that is the same structure as an ORCA free minimum ({ctx['anchor_label']} ≡ {ctx['anchor_min']}, "
      f"{ctx['anchor_rmsd']:.2f} Å) is placed at that minimum's ORCA energy (+{ctx['shift']:.1f} meV), so all "
      "plotted energies are relative to ORCA M1 through one shared structure. Relative energies *within* each code "
      "are unchanged by this; differences *between* codes at other structures are real method differences.")
    w("")

    w("## Conformers\n")
    rows = []
    for lab in sorted(set(df["label"])):
        a, b = fm.loc[lab], af.loc[lab]
        rows.append({
            "conformer": lab, "FM state": a["electronic_state"], "AFM state": b["electronic_state"],
            "max F FM/AFM (eV/Å)": f"{a['max_force']:.3f}/{b['max_force']:.3f}",
            "E FM (meV, ORCA scale)": a["E_anchored_meV"], "E FM − E_FM,min (meV)": a["E_rel_meV"],
            "E_FM − E_AFM (meV)": pr.loc[lab, "dE_FM_AFM_meV"],
            "L, C, R (°)": f"{a['L']:.0f}, {a['C']:.0f}, {a['R']:.0f}",
            "θ_L, θ_C, θ_R (°)": f"{a['theta_L']:.0f}, {a['theta_C']:.0f}, {a['theta_R']:.0f}",
            "θ_FF (°)": a["theta_FF"], "family": a["family"],
            "nearest ORCA minimum (Å)": f"{a['nearest_orca_minimum']} ({a['nearest_orca_minimum_rmsd']:.2f})",
            "nearest ORCA geometry (Å)": f"{a['nearest_orca']} ({a['nearest_orca_rmsd']:.2f})",
        })
    w(md_table(pd.DataFrame(rows), "{:.1f}"))
    w("\nFamilies as defined in `fluorene_angle_analysis.md` (handedness of the outer twists relative to C). "
      "SIESTA Hirshfeld terminal spins are ±0.72–0.80 (AFM) — a different population scheme from the ORCA Mulliken "
      "values, so not directly comparable in magnitude.\n")

    w("## Energy vs the fluorene-plane angle\n")
    w("![Energy vs fluorene angle](energy_vs_fluorene_angle_siesta_orca.png)\n")
    w("## Energy vs the central torsion\n")
    w("![Energy vs central angle](energy_vs_central_angle_siesta_orca.png)\n")
    w("The black curve is E_min(C) of `angle_scan.md` (minimum over the 5×5 (L,R) grid at each C), on the M1 scale. "
      "The converged SIESTA conformers lie between C = 75° and 92°, close to the curve; the ORCA minimum of the curve "
      "(C = 100°) and M1 (C = 104°) have no SIESTA counterpart.\n")
    w("## Where the conformers sit in torsion space\n")
    w("![Torsion maps](torsion_maps_siesta_orca.png)\n")
    w("Outer torsions are folded modulo the fluorenyl flip and ordered (larger, smaller) so that L↔R-swapped copies "
      "coincide. Most SIESTA conformers have one outer ring at ≈ +35° and the other at ≈ −30° (mixed twist, like M3/M4); "
      "only the M2 match has both rings on the same side.\n")
    w("## Energy spectrum\n")
    w("![Energy spectrum](energy_spectrum_siesta_orca.png)\n")
    w("Dotted connectors join SIESTA and ORCA levels of the same structure (RMSD < 0.3 Å). The M2 level coincides by "
      "construction (anchor); the M4 match shows the method difference directly: "
      + (f"{matched[matched['nearest_orca_minimum'] == 'M4'].iloc[0]['E_anchored_meV']:.0f} meV (SIESTA) vs "
         f"{f.loc['M4', 'E_meV']:.0f} meV (ORCA), i.e. PBE puts the mixed-twist M4 closer to M2 than R2SCAN-3C does."
         if (matched["nearest_orca_minimum"] == "M4").any() else "no second match.") + "\n")

    w("## Exchange splitting vs angles\n")
    w("![Exchange vs angles](exchange_vs_angles_siesta_orca.png)\n")
    t = ok[["label", "dE_FM_AFM_meV", "orca_guide", "theta_FF", "C", "P_hop", "both_converged", "rmsd_FM_AFM"]].copy()
    t.columns = ["conformer", "SIESTA E_FM − E_AFM (meV)", "ORCA PBE0 guide 46.7·P + 0.43 (meV)", "θ_FF (°)", "C (°)",
                 "P", "both converged", "RMSD FM–AFM (Å)"]
    t["P"] = [f"{v:.1e}" for v in t["P"]]
    w(md_table(t, "{:.3f}"))
    w("")
    if len(ok) > 2:
        r1, r2, r3 = (spearmanr(ok[c], ok["dE_FM_AFM_meV"]) for c in ("theta_FF", "C", "P_hop"))
        w(f"- Spearman (n = {len(ok)}): ρ(ΔE, θ_FF) = {r1[0]:.2f} (p = {r1[1]:.2f}), ρ(ΔE, C) = {r2[0]:.2f} "
          f"(p = {r2[1]:.2f}), ρ(ΔE, P) = {r3[0]:.2f} (p = {r3[1]:.2f}).")
    w("- ΔE = E_FM − E_AFM is the raw SIESTA energy difference (same sign convention as the ORCA E_HS − E_BS: > 0 → "
      "AFM lower). SIESTA gives no ⟨S²⟩, so no spin projection is applied. With Zuo's definition "
      "J = −(E_HS − E_BS)/(⟨S²⟩_HS − ⟨S²⟩_BS) and ⟨S²⟩_HS − ⟨S²⟩_BS ≈ 1 (as in the ORCA runs), J ≈ −ΔE.")
    w(f"- For P < 0.01 the SIESTA values sit close to the ORCA PBE0 guide (median ratio {lo_r:.2f}; the guide's "
      f"0.43 meV intercept dominates there), for P ≥ 0.01 they are about {hi_r:.2f}× the guide: the PBE splitting "
      "grows less with conjugation than the PBE0 one. Functional (GGA vs hybrid) and code/basis (numerical PAOs vs "
      "def2-TZVP) both enter; an ORCA PBE single point on the same geometries would separate the two.")
    c3 = ok[ok["family"] == "O1"]
    for r in c3.itertuples():
        w(f"- {r.label} (one outer ring nearly orthogonal, P = {r.P_hop:.1e}) has ΔE = {r.dE_FM_AFM_meV:.2f} meV — "
          "the same orthogonal-ring decoupling seen in the ORCA O1 family.")
    w("")
    w("## Files\n")
    w("- `siesta_conformers.csv` — one row per SIESTA run (state, convergence, energies, spins, all angles, family, "
      "nearest ORCA structures).")
    w("- `siesta_exchange_pairs.csv` — one row per conformer: FM/AFM validity, E_FM − E_AFM, RMSD FM–AFM, angles.")
    w("- figures: `energy_vs_fluorene_angle_siesta_orca.png`, `energy_vs_central_angle_siesta_orca.png`, "
      "`torsion_maps_siesta_orca.png`, `energy_spectrum_siesta_orca.png`, `exchange_vs_angles_siesta_orca.png`.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
