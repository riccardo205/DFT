#!/usr/bin/env python3
"""
Energy landscape of short-2OS projected on the angle between the two terminal
(radical-bearing) fluorene planes -- the collective coordinate theta of
Zuo et al., J. Phys. Chem. Lett. 15, 5761 (2024), Fig. 2(a) / SI Fig. S1.

The molecule has three internal torsions (L, C, R). Many (L, C, R) map onto
the same terminal-plane angle theta_FF; this script keeps them apart:
  1. theta_FF (0-90 deg, unoriented planes) + a signed relative-orientation
     descriptor, for every converged, chemically intact HS geometry
  2. E(theta_FF) colored by C, free minima M1-M5 overlaid
  3. degeneracy: +/-2 deg windows, pairs with matched theta_FF and different E
  4. conformational families from the handedness of L, C, R
  5. sampled lower envelope (5 deg bins) -- NOT a relaxed profile
  6. comparison with Zuo (junction Fig. 2a, isolated molecule SI Fig. S1a)
  7. PBE0 E_HS - E_BS vs theta_FF

Data (all R2SCAN-3C TightOpt UKS, multiplicity 3, same 86-atom molecule):
  angle_scan_relaxation              L x C x R grid, all three torsions constrained
  angle_scan_C100_outer_relaxation   C = 100, L = R = 0..180 (10 deg), constrained
  angle_scan_C100_L090_relaxation    C = 100, L = 90, R = 0..180, constrained
  angle_scan_central_approx_90/fixed_outer   C = 90, L <= R on {0,45,90,135,180}, constrained
  angle_scan_free_relaxation         unconstrained HS minima M1-M5
Broken-symmetry R2SCAN-3C energies exist only at the free minima, so no BS
conformational dataset is built (HS and BS are never mixed).

Symmetry facts used (checked on the relaxed geometry, see report):
  * each terminal fluorenyl is locally C2-symmetric about its linker bond
    (180 deg flip RMSD ~0.08 A)  -> L == L + 180, R == R + 180
  * the backbone fragments are not (flip RMSD ~2-3 A) -> C is 360-periodic
  * the molecule maps fragment 1<->4, 2<->3              -> (L,C,R) == (R,C,L)
  * mirror image                                          -> (L,C,R) == (-L,-C,-R)
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_angle_scan import (  # noqa: E402
    FINAL_E_RE,
    HARTREE_TO_MEV,
    build_adjacency,
    check_chemical_integrity,
    fit_plane,
    outer_terminal_fragment,
    parse_out_file,
    reference_bonds,
    split_fragment_at_bond,
)
from analyze_free_relaxation import kabsch_rmsd, plane_angle, signed  # noqa: E402

# reference palette (light mode), see dataviz skill
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK_2, INK_3, GRID, MUTED = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c9c8c2"

CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
RADICALS = (41, 28)
RINGS = {  # flanking rings of each linker bond (same as analyze_free_relaxation / exchange analysis)
    "L": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]),
    "C": ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25]),
    "R": ([23, 21, 25, 24, 26, 27], [28, 29, 30, 31, 32]),
}

FAMILIES = {  # key: (label, description)
    "T++": ("co-twisted", "both outer rings twisted with the handedness favoured at that C"),
    "T--": ("counter-twisted", "both outer rings twisted with the disfavoured handedness"),
    "T+-": ("mixed twist", "one outer ring of each handedness"),
    "T90": ("twisted, C = 90", "both outer rings twisted, C = 90 (handedness reference undefined)"),
    "P": ("near-planar outer", ">= 1 outer ring within 22.5 deg of coplanar, none orthogonal"),
    "O1": ("one outer orthogonal", "exactly one outer ring within 22.5 deg of 90"),
    "O2": ("both outer orthogonal", "both outer rings within 22.5 deg of 90"),
}
FAMILY_ORDER = list(FAMILIES)
FAMILY_STYLE = {
    "T++": (SERIES[0], "o"), "T--": (SERIES[1], "s"), "T+-": (SERIES[2], "D"), "T90": (SERIES[6], "P"),
    "P": (SERIES[3], "^"), "O1": (SERIES[4], "v"), "O2": (SERIES[7], "X"),
}

# Zuo et al. SI Fig. S1(a): isolated 2-OS, B3LYP/6-31G(d,p), E_HS. Read by eye from the
# rendered figure (pixel positions, ~±2 meV); relative to the theta = 90 deg point.
ZUO_SI_THETA = [0, 15, 25, 30, 35, 40, 45, 50, 60, 65, 70, 80, 90]
ZUO_SI_E_HARTREE = [-2077.08895, -2077.09265, -2077.09347, -2077.09384, -2077.09394, -2077.09416,
                    -2077.09416, -2077.09406, -2077.09374, -2077.09352, -2077.09350, -2077.09379, -2077.09459]
# SI Fig. S1(b): J_Zuo = -(E_HS - E_BS)/(<S2>_HS - <S2>_BS), meV, read by eye (~±0.05 meV). Zuo uses
# H = -2 J_Zuo S1.S2; in the repository convention H = -J S1.S2 (exchange_convention.py) J = 2 J_Zuo.
ZUO_SI_J_ZUO_CONVENTION = [-2.13, -1.02, -0.33, -0.20, -0.12, 0.02, -0.02, -0.05, -0.03, 0.05, 0.45, 0.28, -0.06]
ZUO_SI_J = [2.0 * j for j in ZUO_SI_J_ZUO_CONVENTION]          # repository convention, meV


# ============================================================
# GEOMETRY DESCRIPTORS
# ============================================================
def fold180(x):
    """Outer torsion modulo the fluorenyl C2 flip, in [-90, 90)."""
    return (x + 90.0) % 180.0 - 90.0


def fluorene_descriptors(pos, flu1, flu4):
    c1, n1 = fit_plane(pos[flu1])
    c4, n4 = fit_plane(pos[flu4])
    rms1 = float(np.sqrt(np.mean(((pos[flu1] - c1) @ n1) ** 2)))
    rms4 = float(np.sqrt(np.mean(((pos[flu4] - c4) @ n4) ** 2)))
    if np.dot(n1, n4) < 0:  # planes are unoriented: take the acute pair
        n4 = -n4
    theta = float(np.degrees(np.arccos(np.clip(np.dot(n1, n4), -1.0, 1.0))))
    # sign: sense of the rotation n1 -> n4 about the radical-radical axis 41 -> 28.
    # Invariant to flipping either normal and to relabelling 1 <-> 4; flips under mirroring.
    axis = pos[RADICALS[1]] - pos[RADICALS[0]]
    axis /= np.linalg.norm(axis)
    m = np.cross(n1, n4)
    mnorm = np.linalg.norm(m)
    align = float(abs(np.dot(m / mnorm, axis))) if mnorm > 1e-8 else 0.0
    sgn = float(np.sign(np.dot(m, axis))) if mnorm > 1e-8 else 0.0
    return {"theta_FF": theta, "theta_FF_signed": sgn * theta, "sign_axis_alignment": align,
            "flu1_plane_rms": rms1, "flu4_plane_rms": rms4,
            "d_radicals": float(np.linalg.norm(pos[RADICALS[1]] - pos[RADICALS[0]]))}


def outer_state(x_folded, c):
    m = abs(x_folded)
    if m <= 22.5:
        return "P"
    if m >= 67.5:
        return "O"
    if abs(c - 90.0) < 1.0:
        return "t"
    return "+" if np.sign(x_folded) * np.sign(c - 90.0) > 0 else "-"


def family_of(lf, c, rf):
    s = {outer_state(lf, c), outer_state(rf, c)}
    pair = sorted([outer_state(lf, c), outer_state(rf, c)])
    n_o = pair.count("O")
    if n_o == 2:
        return "O2"
    if n_o == 1:
        return "O1"
    if "P" in s:
        return "P"
    if "t" in s:
        return "T90"
    return {("+", "+"): "T++", ("-", "-"): "T--"}.get(tuple(pair), "T+-")


def flip_rmsd(pos, symbols, frag, b, c):
    """RMSD between a fragment and its own 180-deg rotation about bond b-c (element-wise Hungarian match)."""
    p = pos[frag]
    ax = (pos[c] - pos[b]) / np.linalg.norm(pos[c] - pos[b])
    v = p - pos[b]
    q = pos[b] - v + 2.0 * np.outer(v @ ax, ax)  # rotation by pi about ax through b
    sym = np.array(symbols)[frag]
    tot = 0.0
    for el in set(sym):
        m = sym == el
        d = np.linalg.norm(p[m][:, None] - q[m][None], axis=-1)
        r, cc = linear_sum_assignment(d)
        tot += float((d[r, cc] ** 2).sum())
    return float(np.sqrt(tot / len(frag)))


def eta2(df, value, group_cols, bin_col):
    """Pooled fraction of within-bin variance of `value` explained by grouping `group_cols` inside each bin."""
    ss_tot = ss_within = 0.0
    for _, g in df.groupby(bin_col):
        if len(g) < 2:
            continue
        ss_tot += float(((g[value] - g[value].mean()) ** 2).sum())
        ss_within += float(g.groupby(group_cols)[value].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum())
    return 1.0 - ss_within / ss_tot if ss_tot > 0 else np.nan


def global_eta2(df, value, group_col):
    ss_tot = float(((df[value] - df[value].mean()) ** 2).sum())
    ss_within = float(df.groupby(group_col)[value].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum())
    return 1.0 - ss_within / ss_tot


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


# ============================================================
# DATA COLLECTION
# ============================================================
def check_input(inp_path):
    text = inp_path.read_text()
    head = text.splitlines()[0].upper()
    ok = "R2SCAN-3C" in head and "UKS" in head and "* XYZFILE 0 3" in text.upper()
    if not ok:
        raise RuntimeError(f"Incompatible level of theory / state in {inp_path}")


def collect(scans, ctx):
    rows = []
    # 1) coarse grid: reuse the curated angle_scan.csv (convergence, integrity, constraint checks)
    grid_root = scans / "angle_scan_relaxation"
    grid = pd.read_csv(grid_root / "angle_scan.csv")
    for _, g in grid[grid["usable"]].iterrows():
        d = grid_root / g["label"]
        check_input(d / "high_spin.inp")
        rows.append({"label": g["label"], "source": "grid", "kind": "constrained",
                     "L_nom": g["L"], "C_nom": g["C"], "R_nom": g["R"], "E_final": g["final_energy"],
                     "xyz": d / "high_spin.xyz"})
    # 2) constrained sets parsed here
    for src, root in (("C100_LR", scans / "angle_scan_C100_outer_relaxation"),
                      ("C100_L90", scans / "angle_scan_C100_L090_relaxation"),
                      ("C090", scans / "angle_scan_central_approx_90" / "fixed_outer")):
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("L")):
            out = d / "high_spin.out"
            if not out.exists():
                ctx["skipped"].append((src, d.name, "no output"))
                continue
            check_input(d / "high_spin.inp")
            rec = parse_out_file(out.read_text(errors="ignore"), out)
            if not rec["opt_converged"]:
                ctx["skipped"].append((src, d.name, "not converged"))
                continue
            L0, C0, R0 = (int(d.name[i:i + 3]) for i in (1, 6, 11))
            rows.append({"label": d.name, "source": src, "kind": "constrained", "L_nom": L0, "C_nom": C0,
                         "R_nom": R0, "E_final": rec["final_energy"], "xyz": d / "high_spin.xyz"})
    # 3) free HS minima
    free_root = scans / "angle_scan_free_relaxation"
    free = pd.read_csv(free_root / "free_relaxation.csv")
    for _, f in free[free["state"] == "high_spin"].iterrows():
        d = free_root / f["start"] / "high_spin"
        if not (f["converged"] and f["geometry_valid"]):
            ctx["skipped"].append(("free", f["start"], "walltime / not converged"))
            continue
        check_input(d / "high_spin.inp")
        rows.append({"label": f"free:{f['start']}", "source": "free", "kind": "free", "L_nom": np.nan,
                     "C_nom": np.nan, "R_nom": np.nan, "E_final": f["E_final"], "xyz": d / "high_spin.xyz",
                     "minimum": f["minimum"], "start": f["start"]})
    return rows


def main():
    # ---- USER SETTINGS ----
    repo = Path(__file__).resolve().parent.parent
    scans = repo / "runs" / "orca" / "short-2OS" / "angle_scans"
    out_dir = scans / "fluorene_angle_analysis"
    baseline_xyz = repo / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    DEGEN_TOL = 2.0          # deg, matched terminal-plane angle
    DEGEN_BIN = 4.0          # deg, interval width for the degeneracy table (= +/- 2 deg)
    ENVELOPE_BIN = 5.0       # deg
    DUPLICATE_RMSD = 0.01    # A, identical geometry in two constrained sets
    SIGN_MIN_ALIGNMENT = 0.3  # |axis . rotation axis| below this -> sign of theta_FF not meaningful
    out_dir.mkdir(exist_ok=True)

    # ---- fragments ----
    ref = read(baseline_xyz)
    adj = build_adjacency(ref)
    n_atoms = len(ref)
    bonded_pairs = {(i, j) for i in range(n_atoms) for j in adj[i] if j > i}
    bonds = reference_bonds(adj, ref)
    f1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    f4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    centre = sorted(set(range(n_atoms)) - set(f1) - set(f4))
    f2, f3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), centre)
    sym = ref.get_chemical_symbols()
    flu1 = [i for i in f1 if sym[i] == "C"]  # 13 rigid fluorenyl carbons
    flu4 = [i for i in f4 if sym[i] == "C"]
    assert len(flu1) == len(flu4) == 13, "terminal fragments are not C13 fluorenyls"
    ctx = {"skipped": [], "fragments": (f1, f2, f3, f4), "flu": (flu1, flu4)}

    rows = collect(scans, ctx)
    recs, positions = [], {}
    for r in rows:
        atoms = read(r["xyz"])
        assert len(atoms) == n_atoms
        integ = check_chemical_integrity(atoms, bonds, bonded_pairs, 0.30, 1.9, 1.5)
        if not integ["valid"]:
            ctx["skipped"].append((r["source"], r["label"], "chemically broken"))
            continue
        pos = atoms.get_positions()
        L, C, R = (signed(atoms.get_dihedral(*q)) for q in (LEFT, CENTER, RIGHT))
        if r["kind"] == "constrained":
            off = max(abs(signed(a - b)) for a, b in ((L, r["L_nom"]), (C, r["C_nom"]), (R, r["R_nom"])))
            if off > 1.0:
                ctx["skipped"].append((r["source"], r["label"], f"constraint off by {off:.1f} deg"))
                continue
        if C < 0:  # mirror image -> C > 0 representative (energy unchanged)
            L, C, R = -L, -C, -R
        rec = {k: v for k, v in r.items() if k != "xyz"}
        rec.update({"L": L, "C": C, "R": R, "L_fold": fold180(L), "R_fold": fold180(R)})
        rec.update({f"theta_{a}": plane_angle(pos, *RINGS[a]) for a in "LCR"})
        rec.update(fluorene_descriptors(pos, flu1, flu4))
        rec["family"] = family_of(rec["L_fold"], C, rec["R_fold"])
        rec["P_hop"] = float(np.prod([np.cos(np.radians(rec[f"theta_{a}"])) ** 2 for a in "LCR"]))
        rec["geometry_file"] = str(r["xyz"].relative_to(scans))
        recs.append(rec)
        positions[(r["source"], r["label"])] = pos
    df = pd.DataFrame(recs)

    # ---- duplicates between constrained sets (keep the grid copy) ----
    dup = []
    grid_keys = {lab for src, lab in positions if src == "grid"}
    for i, rr in df.iterrows():
        if rr["source"] in ("C100_LR", "C100_L90") and rr["label"] in grid_keys:
            if kabsch_rmsd(positions[(rr["source"], rr["label"])], positions[("grid", rr["label"])]) < DUPLICATE_RMSD:
                dup.append(i)
    ctx["duplicates"] = df.loc[dup, ["source", "label"]].values.tolist()
    df = df.drop(index=dup).reset_index(drop=True)

    # ---- energy reference: lowest converged HS energy of free minimum M1 ----
    m1 = df[(df["kind"] == "free") & (df["minimum"] == "M1")]
    e_ref = float(m1["E_final"].min())
    ctx["ref_label"] = m1.loc[m1["E_final"].idxmin(), "label"]
    df["E_meV"] = (df["E_final"] - e_ref) * HARTREE_TO_MEV
    df["sign_reliable"] = (df["sign_axis_alignment"] >= SIGN_MIN_ALIGNMENT) & df["theta_FF"].between(5, 85)
    df["bin4"] = np.minimum((df["theta_FF"] // DEGEN_BIN).astype(int), int(np.ceil(90 / DEGEN_BIN)) - 1)
    df["bin5"] = np.minimum((df["theta_FF"] // ENVELOPE_BIN).astype(int), int(np.ceil(90 / ENVELOPE_BIN)) - 1)

    # ---- symmetry checks on one relaxed geometry ----
    pos_m1 = positions[("free", ctx["ref_label"])]
    ctx["flip"] = {
        "fluorenyl 1 about 41-0": flip_rmsd(pos_m1, sym, f1, 41, 0),
        "fluorenyl 4 about 28-23": flip_rmsd(pos_m1, sym, f4, 28, 23),
        "backbone 2 about 9-17": flip_rmsd(pos_m1, sym, f2, 9, 17),
        "backbone 3 about 17-9": flip_rmsd(pos_m1, sym, f3, 17, 9),
    }

    # ---- empirical noise floor: symmetry-equivalent constrained points ----
    # (L,C,R) == (R,C,L) (fragment 1<->4) and L == L+180, R == R+180 (fluorenyl flip)
    con = df[df["kind"] == "constrained"].copy()
    con["eq_class"] = [(int(r.C_nom), tuple(sorted((int(r.L_nom) % 180, int(r.R_nom) % 180))))
                       for r in con.itertuples()]
    eq_pairs = []
    for _, g in con.groupby("eq_class"):
        for r1, r2 in combinations(g.itertuples(), 2):
            eq_pairs.append((r1.label, r1.source, r2.label, r2.source, abs(r1.E_meV - r2.E_meV),
                             abs(r1.theta_FF - r2.theta_FF)))
    ctx["noise"] = pd.DataFrame(eq_pairs, columns=["a", "src_a", "b", "src_b", "dE", "dtheta_FF"])

    ctx["positions"] = positions
    ctx["e_ref"] = e_ref
    for f in out_dir.glob("_*.pkl"):
        f.unlink()
    df = attach_exchange(df, scans, positions)
    res = analyse(df, ctx, DEGEN_TOL, DEGEN_BIN, ENVELOPE_BIN)
    make_plots(df, ctx, res, out_dir)
    export_csv(df, out_dir / "fluorene_angle_analysis.csv")
    write_report(df, ctx, res, out_dir / "fluorene_angle_analysis.md", DEGEN_TOL, DEGEN_BIN, ENVELOPE_BIN)
    print(f"Wrote {out_dir}")


# ============================================================
# PBE0 EXCHANGE (existing HS/BS single points at the same geometries)
# ============================================================
def same_geometry(xyz, pos, tol=2e-3):
    return xyz.exists() and kabsch_rmsd(read(xyz).get_positions(), pos) < tol


def attach_exchange(df, scans, positions):
    df = df.copy()
    df["dE_PBE0_meV"] = np.nan
    df["dE_PBE0_source"] = ""
    grid_ex = pd.read_csv(scans / "angle_scan_exchange_analysis" / "exchange_analysis.csv").set_index("label")
    outer_ex = pd.read_csv(scans / "outer_scan_analysis" / "outer_scans_exchange_points.csv")
    outer_ex = {("C100_LR" if r.scan == "A" else "C100_L90", r.label): r.dE for r in outer_ex.itertuples()}
    ex_dirs = {"C100_LR": scans / "angle_scan_C100_outer_exchange", "C100_L90": scans / "angle_scan_C100_L090_exchange"}
    for i, r in df.iterrows():
        pos = positions[(r["source"], r["label"])]
        if r["source"] == "grid" and r["label"] in grid_ex.index:
            val = grid_ex.loc[r["label"], "delta_E_raw_meV"]
            xyz = scans / "angle_scan_exchange_analysis" / r["label"] / "high_spin" / "geometry.xyz"
            if pd.notna(val) and same_geometry(xyz, pos):
                df.loc[i, ["dE_PBE0_meV", "dE_PBE0_source"]] = [val, "angle_scan_exchange_analysis"]
        elif r["source"] in ex_dirs and (r["source"], r["label"]) in outer_ex:
            xyz = ex_dirs[r["source"]] / r["label"] / "high_spin" / "geometry.xyz"
            if same_geometry(xyz, pos):
                df.loc[i, ["dE_PBE0_meV", "dE_PBE0_source"]] = [outer_ex[(r["source"], r["label"])],
                                                                 ex_dirs[r["source"]].name]
        elif r["kind"] == "free" and r["minimum"] == "M1":
            base = scans / "visualization_orbitals" / "M1"
            e, ok = {}, True
            for st in ("high_spin", "broken_symmetry"):
                d = base / st / "F0"
                out = d / f"{st}.out"
                # the PBE0 pair was run at the BS-relaxed M1 geometry (0.003 A from the HS one)
                if not (out.exists() and same_geometry(d / "geometry.xyz", pos, tol=5e-3)):
                    ok = False
                    break
                text = out.read_text(errors="ignore")
                ok &= "ORCA TERMINATED NORMALLY" in text
                e[st] = float(FINAL_E_RE.findall(text)[-1])
            if ok:
                df.loc[i, ["dE_PBE0_meV", "dE_PBE0_source"]] = [
                    (e["high_spin"] - e["broken_symmetry"]) * HARTREE_TO_MEV, "visualization_orbitals/M1/F0"]
    return df


# ============================================================
# ANALYSIS
# ============================================================
def analyse(df, ctx, tol, bin4, bin5):
    res = {}
    # --- 3a. +/-2 deg interval table ---
    rows = []
    for b, g in df.groupby("bin4"):
        rows.append({
            "interval": f"{b * bin4:.0f}–{min((b + 1) * bin4, 90):.0f}", "n": len(g),
            "E_min": g["E_meV"].min(), "E_median": g["E_meV"].median(), "E_max": g["E_meV"].max(),
            "spread": g["E_meV"].max() - g["E_meV"].min(),
            "L_fold": f"{g['L_fold'].min():.0f} to {g['L_fold'].max():.0f}",
            "C": f"{g['C'].min():.0f} to {g['C'].max():.0f}",
            "R_fold": f"{g['R_fold'].min():.0f} to {g['R_fold'].max():.0f}",
            "families": ", ".join(f for f in FAMILY_ORDER if f in set(g["family"])),
            "lowest": g.loc[g["E_meV"].idxmin(), "label"],
        })
    res["intervals"] = pd.DataFrame(rows)

    # --- 3b. pairs with matched theta_FF ---
    pr = []
    recs = df.to_dict("records")
    for a, b in combinations(range(len(recs)), 2):
        x, y = recs[a], recs[b]
        if abs(x["theta_FF"] - y["theta_FF"]) > tol:
            continue
        lo, hi = (x, y) if x["E_meV"] <= y["E_meV"] else (y, x)
        same_c = abs(x["C"] - y["C"]) < 2.0
        outer_x = sorted([round(x["L_fold"]), round(x["R_fold"])])
        outer_y = sorted([round(y["L_fold"]), round(y["R_fold"])])
        same_outer = all(abs(signed(2 * (u - v))) / 2 < 3.0 for u, v in zip(outer_x, outer_y))
        if same_c and same_outer:
            kind = "same nominal torsions"
        elif same_c:
            kind = "same C, different outer"
        elif same_outer:
            kind = "same outer, different C"
        else:
            kind = "all differ"
        pr.append({"low": lo["label"], "high": hi["label"], "E_low": lo["E_meV"], "E_high": hi["E_meV"],
                   "dE": hi["E_meV"] - lo["E_meV"], "theta_low": lo["theta_FF"], "theta_high": hi["theta_FF"],
                   "fam_low": lo["family"], "fam_high": hi["family"], "kind": kind,
                   "same_family": lo["family"] == hi["family"]})
    pairs = pd.DataFrame(pr)
    res["pairs"] = pairs
    noise = ctx["noise"]
    E = dict(zip(df["label"], df["E_meV"]))
    low_noise = noise[[min(E.get(a, 1e9), E.get(b, 1e9)) < 300 for a, b in zip(noise["a"], noise["b"])]]
    planar = lambda lab: any(lab[i:i + 3] in ("000", "180") for i in (1, 11))
    low_noise = low_noise.assign(planar_outer=[planar(a) or planar(b) for a, b in zip(low_noise["a"], low_noise["b"])])
    res["noise_low"] = low_noise
    res["sig_dE"] = 100.0  # > every symmetry-equivalent discrepancy below 300 meV (see report)
    sig = pairs[pairs["dE"] > res["sig_dE"]]
    res["sig_by_kind"] = sig.groupby("kind")["dE"].agg(["size", "median", "max"]) if not sig.empty else pd.DataFrame()
    res["sig_low_by_kind"] = (sig[sig["E_low"] < 150].groupby("kind")["dE"].agg(["size", "median", "max"])
                              if not sig.empty else pd.DataFrame())

    # --- 3c. what explains the energy at matched theta_FF? (within-bin eta^2) ---
    d = df.copy()
    d["C5"] = (d["C"] / 5).round() * 5
    d["outer"] = [tuple(sorted((int(round(l / 5) * 5), int(round(r / 5) * 5)))) for l, r in zip(d["L_fold"], d["R_fold"])]
    d["outer_mag"] = [tuple(sorted((int(round(abs(l) / 5) * 5), int(round(abs(r) / 5) * 5))))
                      for l, r in zip(d["L_fold"], d["R_fold"])]
    res["eta_theta"] = global_eta2(d, "E_meV", "bin4")
    res["eta_within"] = {
        "central torsion C": eta2(d, "E_meV", ["C5"], "bin4"),
        "outer magnitudes |L|,|R| (no handedness)": eta2(d, "E_meV", ["outer_mag"], "bin4"),
        "outer torsions with handedness": eta2(d, "E_meV", ["outer"], "bin4"),
        "family": eta2(d, "E_meV", ["family"], "bin4"),
        "family x C": eta2(d, "E_meV", ["family", "C5"], "bin4"),
    }

    # --- representative examples ---
    ex = []
    m1 = df[df["label"] == ctx["ref_label"]].iloc[0]
    m2 = df[(df["kind"] == "free") & (df["minimum"] == "M2")].sort_values("E_meV").iloc[0]
    ex.append(("pseudo-mirror minima: same local plane angles, different θ_FF", m1["label"], m2["label"]))
    near = pairs[((pairs["low"] == m1["label"]) | (pairs["high"] == m1["label"])) & (pairs["E_high"] < 400)]
    other = near[near["fam_high"] == "T--"]
    if not other.empty:
        r = other.loc[other["E_high"].idxmin()]
        ex.append(("matched θ_FF, similar outer twist, C on the other side of 90° (handedness flip)", r["low"], r["high"]))
    s = pairs[(pairs["kind"] == "same C, different outer") & (pairs["E_low"] < 150) & (pairs["E_high"] < 400)]
    if not s.empty:
        r = s.loc[s["dE"].idxmax()]
        ex.append(("matched θ_FF, same C, different outer torsions", r["low"], r["high"]))
    cmap = dict(zip(df["label"], df["C"]))
    s = near[(near["fam_high"] == near["fam_low"]).values &
             np.array([abs(cmap[u] - cmap[v]) > 10 for u, v in zip(near["low"], near["high"])])]
    if not s.empty:
        r = s.loc[s["dE"].idxmax()]
        ex.append(("matched θ_FF, same family, different central torsion", r["low"], r["high"]))
    res["examples"] = ex

    # --- 5. lower envelope (5 deg bins) ---
    env = []
    for b in range(int(90 // bin5)):
        g = df[df["bin5"] == b]
        gc = g[g["kind"] == "constrained"]
        row = {"bin": b, "lo": b * bin5, "hi": (b + 1) * bin5, "n": len(g)}
        if len(g):
            k = g["E_meV"].idxmin()
            row.update({"E_min": g.loc[k, "E_meV"], "label": g.loc[k, "label"], "family": g.loc[k, "family"],
                        "theta_FF": g.loc[k, "theta_FF"], "kind": g.loc[k, "kind"]})
        if len(gc):
            row["E_min_constrained"] = gc["E_meV"].min()
            row["label_constrained"] = gc.loc[gc["E_meV"].idxmin(), "label"]
        env.append(row)
    env = pd.DataFrame(env)
    e = env["E_min"].values
    loc = []
    for i in range(len(e)):
        if np.isnan(e[i]):
            continue
        left = e[i - 1] if i > 0 else np.inf
        right = e[i + 1] if i < len(e) - 1 else np.inf
        if (np.isnan(left) or e[i] < left) and (np.isnan(right) or e[i] < right):
            loc.append(i)
    env["envelope_local_min"] = env.index.isin(loc)
    env["edge_bin"] = (env.index == 0) | (env.index == len(env) - 1)
    res["envelope"] = env

    # --- 7. exchange ---
    x = df.dropna(subset=["dE_PBE0_meV"])
    res["n_exchange"] = len(x)
    if len(x) > 5:
        res["rho_theta"] = spearmanr(x["theta_FF"], x["dE_PBE0_meV"])
        res["rho_P"] = spearmanr(x["P_hop"], x["dE_PBE0_meV"])
        res["eta_ex_theta"] = global_eta2(x, "dE_PBE0_meV", "bin4")
        res["eta_ex_family_within"] = eta2(x, "dE_PBE0_meV", ["family"], "bin4")
        xp = []
        xr = x.to_dict("records")
        for a, b in combinations(range(len(xr)), 2):
            u, v = xr[a], xr[b]
            if abs(u["theta_FF"] - v["theta_FF"]) <= tol:
                lo, hi = (u, v) if u["dE_PBE0_meV"] <= v["dE_PBE0_meV"] else (v, u)
                xp.append({"low": lo["label"], "high": hi["label"], "dE_low": lo["dE_PBE0_meV"],
                           "dE_high": hi["dE_PBE0_meV"], "ddE": hi["dE_PBE0_meV"] - lo["dE_PBE0_meV"],
                           "theta_low": lo["theta_FF"], "theta_high": hi["theta_FF"],
                           "E_low": lo["E_meV"], "E_high": hi["E_meV"], "fam_low": lo["family"],
                           "fam_high": hi["family"]})
        res["ex_pairs"] = pd.DataFrame(xp)
    return res


# ============================================================
# PLOTS
# ============================================================
# diverging around C = 90 (the handedness branch point): blue arm C < 90, red arm C > 90,
# darker = farther from 90; neutral at 90
DIV_BLUE = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]
DIV_RED = ["#f0a3a2", "#e86b6a", "#c93a39", "#952625"]
DIV_MID = "#9a9993"


def c_color(c):
    d = c - 90.0
    if abs(d) < 5:
        return DIV_MID
    k = min(int((abs(d) - 5) // 20), 3)
    return (DIV_BLUE if d < 0 else DIV_RED)[k]


def free_minima(df):
    f = df[df["kind"] == "free"]
    return f.loc[f.groupby("minimum")["E_meV"].idxmin()].sort_values("minimum")


def mark_minima(ax, df, ymax=None, color=INK):
    for r in free_minima(df).itertuples():
        if ymax is not None and r.E_meV > ymax:
            continue
        ax.scatter(r.theta_FF, r.E_meV, marker="*", s=230, color="white", edgecolor=color, linewidth=1.4, zorder=6)
        ax.annotate(r.minimum, (r.theta_FF, r.E_meV), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold", zorder=7)


def make_plots(df, ctx, res, out_dir):
    con = df[df["kind"] == "constrained"]
    xlab = "angle between terminal fluorene planes θ$_{FF}$ (°)"
    ylab = "R2SCAN-3C HS energy rel. to M1 (meV)"

    # ---- 1. colored by C ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ymax in zip(axes, (None, 300)):
        for src, mk in (("grid", "o"), ("C100_LR", "^"), ("C100_L90", "v"), ("C090", "s")):
            g = con[con["source"] == src]
            ax.scatter(g["theta_FF"], g["E_meV"], c=[c_color(c) for c in g["C"]], marker=mk, s=34,
                       edgecolor="white", linewidth=0.6, zorder=3)
        mark_minima(ax, df, ymax)
        ax.set_xlim(-2, 92)
        ax.set_xticks(range(0, 91, 15))
        if ymax:
            ax.set_ylim(-15, ymax)
            ax.set_title("(b) low-energy region (0–300 meV)", loc="left", fontsize=10)
        else:
            ax.set_title("(a) all converged, chemically intact HS geometries", loc="left", fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        style_axes(ax)
    cs = [20, 40, 60, 80, 90, 100, 120, 140, 160]
    h = [Line2D([], [], marker="o", ls="", color=c_color(c), label=f"C = {c}°") for c in cs]
    h += [Line2D([], [], marker=m, ls="", color=INK_3, label=l) for m, l in
          (("o", "L×C×R grid"), ("^", "C=100, L=R scan"), ("v", "C=100, L=90 scan"), ("s", "C=90 grid"))]
    h.append(Line2D([], [], marker="*", ls="", ms=13, mfc="white", mec=INK, label="free minima M1–M5"))
    fig.legend(handles=h, loc="center right", fontsize=8, frameon=False)
    fig.suptitle("Conformational energy vs terminal fluorene-plane angle, colored by central torsion C "
                 "(blue C < 90°, red C > 90°)", x=0.02,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(out_dir / "energy_vs_fluorene_angle_C.png", dpi=200)
    plt.close(fig)

    # ---- 2. families ----
    YMAX = 700
    fams = [f for f in FAMILY_ORDER if f in set(df["family"])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
    axes = axes.ravel()
    ax = axes[0]
    for f in fams:
        g = df[(df["family"] == f) & (df["kind"] == "constrained")]
        col, mk = FAMILY_STYLE[f]
        ax.scatter(g["theta_FF"], g["E_meV"], color=col, marker=mk, s=26, edgecolor="white", linewidth=0.5,
                   label=f"{FAMILIES[f][0]} ({f})", zorder=3)
    mark_minima(ax, df, YMAX)
    ax.set_title("all families", loc="left", fontsize=10)
    for ax, f in zip(axes[1:], fams):
        ax.scatter(con["theta_FF"], con["E_meV"], color=GRID, s=14, zorder=2)
        g = df[(df["family"] == f) & (df["kind"] == "constrained")]
        col, mk = FAMILY_STYLE[f]
        ax.scatter(g["theta_FF"], g["E_meV"], color=col, marker=mk, s=32, edgecolor="white", linewidth=0.5, zorder=3)
        fm = free_minima(df)
        for r in fm[fm["family"] == f].itertuples():
            ax.scatter(r.theta_FF, r.E_meV, marker="*", s=230, color="white", edgecolor=INK, linewidth=1.4, zorder=6)
            ax.annotate(r.minimum, (r.theta_FF, r.E_meV), xytext=(6, 6), textcoords="offset points", fontsize=9,
                        fontweight="bold", color=INK)
        n_hidden = int((g["E_meV"] > YMAX).sum())
        ax.set_title(f"{FAMILIES[f][0]} ({f}), n = {len(g)}" + (f", {n_hidden} > {YMAX} meV" if n_hidden else ""),
                     loc="left", fontsize=9)
    for ax in axes[len(fams) + 1:]:
        ax.axis("off")
    for ax in axes:
        if ax.axison:
            ax.set_xlim(-2, 92)
            ax.set_ylim(-20, YMAX)
            ax.set_xticks(range(0, 91, 15))
            style_axes(ax)
    for ax in axes[4:]:
        ax.set_xlabel(xlab)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Same data colored by conformational family (handedness of L, R relative to C; L, R modulo the "
                 "fluorenyl 180° flip)", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "energy_vs_fluorene_angle_families.png", dpi=200)
    plt.close(fig)

    # ---- 3. lower envelope ----
    env = res["envelope"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ax.scatter(con["theta_FF"], con["E_meV"], color=MUTED, s=18, zorder=2, label="constrained geometries")
    ok = env.dropna(subset=["E_min"])
    edges = np.r_[ok["lo"].values, ok["hi"].values[-1]]
    ax.stairs(ok["E_min"].values, edges, color=INK, linewidth=2, zorder=4, baseline=None,
              label="sampled lower envelope (min per 5° bin, all data)")
    okc = env.dropna(subset=["E_min_constrained"])
    ax.stairs(okc["E_min_constrained"].values, np.r_[okc["lo"].values, okc["hi"].values[-1]], color=SERIES[0],
              linewidth=1.3, linestyle="--", zorder=4, baseline=None, label="same, constrained geometries only")
    for r in env[env["envelope_local_min"] & ~env["edge_bin"] & (env["n"] >= 3)].itertuples():
        ax.annotate("envelope dip", (r.theta_FF, r.E_min), xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.6))
    mark_minima(ax, df, 500)
    ax.set_ylim(-30, 500)
    ax.set_xlim(-2, 92)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("(a) sampled minimum per bin (not a relaxed profile); ★ = unconstrained minima", loc="left",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style_axes(ax)
    ax = axes[1]
    zuo = (np.array(ZUO_SI_E_HARTREE) - min(ZUO_SI_E_HARTREE)) * HARTREE_TO_MEV
    ax.stairs(ok["E_min"].values, edges, color=INK, linewidth=2, baseline=None,
              label="this work: sampled envelope, R2SCAN-3C HS")
    ax.plot(ZUO_SI_THETA, zuo, color=SERIES[1], marker="o", ms=5, lw=1.6,
            label="Zuo SI Fig. S1a: isolated, B3LYP/6-31G(d,p) HS\n(read from figure, ±2 meV)")
    for t in (20, 90):
        ax.axvline(t, color=SERIES[6], lw=1, ls=":", zorder=1)
    ax.text(21, 3, "Zuo junction minima (Au/2-OS/Au, SCAN) at 20° and 90°", fontsize=7.5, color=SERIES[6])
    mark_minima(ax, df, 200)
    ax.set_xlim(-2, 92)
    ax.set_ylim(-10, 200)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel(xlab)
    ax.set_ylabel("energy relative to each curve's minimum (meV)")
    ax.set_title("(b) shape comparison with Zuo et al. (different methods and protocols)", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_lower_envelope.png", dpi=200)
    plt.close(fig)

    # ---- 4. degeneracy examples (molecular projections) ----
    draw_examples(df, ctx, res, out_dir / "internal_geometry_degeneracy.png")

    # ---- 5. exchange ----
    x = df.dropna(subset=["dE_PBE0_meV"])
    if len(x) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axes[0]
        for f in [f for f in FAMILY_ORDER if f in set(x["family"])]:
            g = x[x["family"] == f]
            col, mk = FAMILY_STYLE[f]
            ax.scatter(g["theta_FF"], g["dE_PBE0_meV"], color=col, marker=mk, s=30, edgecolor="white",
                       linewidth=0.5, label=f"{FAMILIES[f][0]} ({f})", zorder=3)
        fx = x[x["kind"] == "free"]
        for r in fx.itertuples():
            ax.scatter(r.theta_FF, r.dE_PBE0_meV, marker="*", s=230, color="white", edgecolor=INK, lw=1.4, zorder=6)
            ax.annotate(f"{r.minimum} (free minimum)", (r.theta_FF, r.dE_PBE0_meV), xytext=(-40, -40),
                        textcoords="offset points", fontsize=8, color=INK)
        ax.plot(ZUO_SI_THETA, -np.array(ZUO_SI_J) / 2, color=INK_3, ls="--", lw=1.2,
                label="Zuo SI Fig. S1b, isolated B3LYP: −J/2 ≈ ΔE = E$_{HS}$−E$_{BS}$")
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlim(-2, 92)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xlabel(xlab)
        ax.set_ylabel("PBE0 ΔE = E$_{HS}$ − E$_{BS}$ (meV, symlog)")
        ax.set_title("(a) exchange splitting vs θ$_{FF}$ (> 0: BS lower)", loc="left", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
        style_axes(ax)
        ax = axes[1]
        ax.scatter(x["P_hop"], x["dE_PBE0_meV"], c=[FAMILY_STYLE[f][0] for f in x["family"]], s=22,
                   edgecolor="white", linewidth=0.4)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(0, 1.1 * x["P_hop"].max())
        ax.set_xlabel("P = cos²θ$_L$ cos²θ$_C$ cos²θ$_R$ (actual plane angles)")
        ax.set_title(f"(b) same data vs three-bottleneck proxy\nSpearman ρ(ΔE, P) = {res['rho_P'][0]:.2f} vs "
                     f"ρ(ΔE, θ$_{{FF}}$) = {res['rho_theta'][0]:.2f}", loc="left", fontsize=10)
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "exchange_vs_fluorene_angle.png", dpi=200)
        plt.close(fig)


def molecule_frame(pos, flu1):
    z = pos[RADICALS[1]] - pos[RADICALS[0]]
    z /= np.linalg.norm(z)
    _, n1 = fit_plane(pos[flu1])
    y = n1 - np.dot(n1, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    centre = 0.5 * (pos[RADICALS[0]] + pos[RADICALS[1]])
    return (pos - centre) @ np.vstack([x, y, z]).T


def draw_mol(ax, xy, depth, adj, frags, view):
    f1, f2, f3, f4 = frags
    col = {}
    for i in f1:
        col[i] = SERIES[0]
    for i in f4:
        col[i] = SERIES[1]
    for i in f2:
        col[i] = INK_3
    for i in f3:
        col[i] = MUTED
    heavy = [i for i in range(len(xy)) if i in col and not np.isnan(depth[i])]
    segs = sorted(((depth[i] + depth[j]) / 2, i, j) for i in heavy for j in adj[i] if j > i and j in col)
    for _, i, j in segs:
        c = col[i] if col[i] == col[j] else INK
        ax.plot(*np.c_[xy[i], xy[j]], color=c, lw=1.6, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.04, view, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK_2)


def draw_examples(df, ctx, res, path):
    ref = read(Path(__file__).resolve().parent.parent / "scripts" / "orca" / "orca_functional_scan_setup" /
               "geometries" / "relaxed_main_conf_4.xyz")
    adj = build_adjacency(ref)
    sym = ref.get_chemical_symbols()
    heavy = np.array([s != "H" for s in sym])
    frags = ctx["fragments"]
    flu1 = ctx["flu"][0]
    ex = res["examples"]
    fig, axes = plt.subplots(len(ex), 4, figsize=(15, 4.2 * len(ex)),
                             gridspec_kw={"width_ratios": [1.2, 1, 1.2, 1]})
    axes = np.atleast_2d(axes)
    by_label = df.set_index("label")
    for row, (title, la, lb) in enumerate(ex):
        for k, lab in enumerate((la, lb)):
            r = by_label.loc[lab]
            pos = ctx["positions"][(r["source"], lab)]
            p = molecule_frame(pos, flu1)
            p[~heavy] = np.nan
            draw_mol(axes[row, 2 * k], p[:, :2], p[:, 2], adj, frags, "end-on (along C41→C28)")
            draw_mol(axes[row, 2 * k + 1], p[:, [2, 1]], p[:, 0], adj, frags, "side view (fluorenyl 1 edge-on)")
            name = r["minimum"] + " (free)" if r["kind"] == "free" else lab
            txt = (f"{name}   E = {r['E_meV']:.0f} meV   θ$_{{FF}}$ = {r['theta_FF']:.1f}°\n"
                   f"L,C,R = {r['L']:.0f}, {r['C']:.0f}, {r['R']:.0f}°   "
                   f"θ$_L$,θ$_C$,θ$_R$ = {r['theta_L']:.0f}, {r['theta_C']:.0f}, {r['theta_R']:.0f}°   [{r['family']}]")
            axes[row, 2 * k].text(0.0, 1.10, txt, transform=axes[row, 2 * k].transAxes, fontsize=8.5, color=INK,
                                  va="bottom")
        axes[row, 0].text(0.0, 1.30, f"({chr(97 + row)}) {title}", transform=axes[row, 0].transAxes, fontsize=10,
                          fontweight="bold", color=INK, va="bottom")
    h = [Line2D([], [], color=SERIES[0], lw=2, label="fluorenyl 1 (radical C41)"),
         Line2D([], [], color=SERIES[1], lw=2, label="fluorenyl 4 (radical C28)"),
         Line2D([], [], color=INK_3, lw=2, label="backbone fragment 2"),
         Line2D([], [], color=MUTED, lw=2, label="backbone fragment 3")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, hspace=0.75, wspace=0.08)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_csv(df, path):
    cols = ["label", "source", "kind", "minimum", "start", "L_nom", "C_nom", "R_nom", "L", "C", "R", "L_fold",
            "R_fold", "theta_L", "theta_C", "theta_R", "theta_FF", "theta_FF_signed", "sign_axis_alignment",
            "sign_reliable", "flu1_plane_rms", "flu4_plane_rms", "d_radicals", "family", "P_hop", "E_final",
            "E_meV", "dE_PBE0_meV", "dE_PBE0_source", "geometry_file"]
    df[cols].sort_values(["kind", "theta_FF"]).to_csv(path, index=False, float_format="%.6f")


def md_table(df, floatfmt="{:.1f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else
                         ("–" if (isinstance(v, float) and pd.isna(v)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(df, ctx, res, path, tol, bin4, bin5):
    con = df[df["kind"] == "constrained"]
    fm = free_minima(df)
    by = df.set_index("label")
    n_src = df.groupby("source").size().to_dict()
    starts = {r.start: (r.minimum, r.theta_FF) for r in df[df["kind"] == "free"].itertuples()}
    L = []
    w = L.append

    # ---------------- summary ----------------
    m = {r.minimum: r for r in fm.itertuples()}
    env = res["envelope"]
    band = lambda lo, hi: env[(env["lo"] >= lo) & (env["hi"] <= hi)]["E_min"].min()
    x = df.dropna(subset=["dE_PBE0_meV"])
    ew = res["eta_within"]
    w("# Energy landscape vs the angle between the terminal fluorene planes\n")
    w("Collective coordinate of Zuo et al., *J. Phys. Chem. Lett.* **15**, 5761 (2024): θ_FF, the angle between "
      "the planes of the two terminal radical-bearing fluorenyl groups (fragments 1 and 4). Generated by "
      "`visualisation/analyze_fluorene_angle.py`.\n")
    w("## Summary\n")
    w(f"1. **The global free minimum M1 has nearly perpendicular terminal fluorenes** (θ_FF = {m['M1'].theta_FF:.1f}°), "
      f"although each individual link is only moderately twisted (θ_L, θ_C, θ_R = {m['M1'].theta_L:.0f}, "
      f"{m['M1'].theta_C:.0f}, {m['M1'].theta_R:.0f}°). The ~90° terminal angle is the *composition* of three "
      "rotations about non-parallel bonds, not a 90° twist anywhere in the chain.")
    w(f"2. **θ_FF is a poor descriptor of the conformational energy.** Binning on θ_FF ({bin4:.0f}° windows) explains "
      f"only {100 * res['eta_theta']:.0f}% of the energy variance of the {len(df)} geometries. Inside a window, the "
      f"central torsion C alone explains {100 * ew['central torsion C']:.0f}% of the remaining spread, the outer "
      f"torsions {100 * ew['outer torsions with handedness']:.0f}%, and family × C {100 * ew['family x C']:.0f}%. "
      f"The 88–90° window alone spans {res['intervals'].iloc[-1]['E_min']:.0f}–{res['intervals'].iloc[-1]['E_max']:.0f} meV.")
    w(f"3. **Several distinct low-energy conformations, not one curve.** The five free minima lie at θ_FF = "
      + ", ".join(f"{k} {v.theta_FF:.0f}° ({v.E_meV:.0f} meV)" for k, v in m.items())
      + f". M1 and M2 have the same three local plane angles (to within 2°) but θ_FF = {m['M1'].theta_FF:.0f}° vs "
      f"{m['M2'].theta_FF:.0f}°: the handedness of the twists relative to C, not their magnitude, sets the terminal angle.")
    w(f"4. **Low-energy structures near 90° — yes; near 20° — only at high energy.** The lowest sampled energy in "
      f"15–25° is {band(15, 25):.0f} meV and the only free minimum below 30° is M5 (θ_FF = {m['M5'].theta_FF:.0f}°, "
      f"{m['M5'].E_meV:.0f} meV, counter-twisted family). No competitive 20° conformer was found in the sampled "
      "isolated-molecule data at this level of theory.")
    w(f"5. **Shape vs Zuo's isolated-molecule curve (SI Fig. S1a, B3LYP):** both have the global minimum at θ ≈ 90° and "
      f"a secondary low-energy region at intermediate angle (Zuo: ~40–45°, +12 meV; here: M2 at "
      f"{m['M2'].theta_FF:.0f}°, +{m['M2'].E_meV:.0f} meV). Zuo's junction minimum at 20° has no low-energy "
      "counterpart here. The level of theory, the relaxation protocol and the electrodes all differ, so the electrodes "
      "cannot be singled out as the cause.")
    if len(x) > 5:
        w(f"6. **Exchange is not a function of θ_FF.** PBE0 ΔE = E_HS − E_BS vs θ_FF: Spearman ρ = "
          f"{res['rho_theta'][0]:.2f} (n = {len(x)}); vs the three-bottleneck proxy P = Πcos²θ_i: ρ = {res['rho_P'][0]:.2f}. "
          f"At matched θ_FF (±{tol:.0f}°), {int((res['ex_pairs']['ddE'] > 2).sum())} of {len(res['ex_pairs'])} pairs "
          "differ by more than 2 meV. The internal geometry carries the exchange information that a single collective "
          "angle discards.")
    w("")

    # ---------------- data ----------------
    w("## Data, quality control and conventions\n")
    w("| set | geometries used |\n|---|---|")
    names = {"grid": "`angle_scan_relaxation` (L×C×R grid, all constrained)",
             "C100_LR": "`angle_scan_C100_outer_relaxation` (C = 100, L = R, constrained)",
             "C100_L90": "`angle_scan_C100_L090_relaxation` (C = 100, L = 90, constrained)",
             "C090": "`angle_scan_central_approx_90/fixed_outer` (C = 90, L ≤ R, constrained)",
             "free": "`angle_scan_free_relaxation` (unconstrained HS runs)"}
    for k, v in names.items():
        w(f"| {v} | {n_src.get(k, 0)} |")
    w("")
    w("- Only ORCA-converged, chemically intact geometries (no broken bonds, no contacts < 1.5 Å, constrained torsions "
      "within 1° of target) are used. The grid uses the curated `usable` flag of `angle_scan.csv`.")
    w(f"- Excluded: " + "; ".join(f"{s} {l} ({why})" for s, l, why in ctx["skipped"]) + ". "
      f"Removed as exact duplicates of grid points (RMSD < 0.01 Å): "
      + ", ".join(f"{l} ({s})" for s, l in ctx["duplicates"]) + ".")
    w("- `angle_scan_central_approx_90/free_outer` has no outputs yet and is not included.")
    w("- **Energies**: one dataset only — R2SCAN-3C (def2-mTZVPP, D4, gCP), UKS, multiplicity 3, same 86-atom "
      "molecule; every input file was checked for `R2SCAN-3C`, `UKS` and `xyzfile 0 3`. Reference = lowest converged HS "
      f"energy of M1 ({ctx['ref_label']}, E = {ctx['e_ref']:.8f} Eh). R2SCAN-3C broken-symmetry energies exist only at "
      "the five free minima, so no BS landscape is built; at those minima BS lies 1.2–3.8 meV below HS with the same "
      "ordering (`free_relaxation.md`), far below the conformational differences discussed here.")
    w("- **θ_FF**: least-squares plane through the 13 carbon atoms of each fluorenyl (same atoms for every geometry); "
      f"angle between the unoriented normals, 0–90°. Fluorenyl planarity: RMS deviation ≤ "
      f"{max(df['flu1_plane_rms'].max(), df['flu4_plane_rms'].max()):.3f} Å over all geometries.")
    w("- θ_L, θ_C, θ_R: actual angles between the rings flanking each linker bond (as in `exchange_analysis.md`).")
    flip = ctx["flip"]
    w(f"- **Symmetry used to define handedness** (180° flip of a fragment about its linker bond, RMSD to itself, "
      f"at M1): fluorenyl 1 {flip['fluorenyl 1 about 41-0']:.2f} Å, fluorenyl 4 {flip['fluorenyl 4 about 28-23']:.2f} Å "
      f"(locally C2-symmetric), backbone 2 {flip['backbone 2 about 9-17']:.2f} Å, backbone 3 "
      f"{flip['backbone 3 about 17-9']:.2f} Å (not symmetric). Hence L ≡ L+180° and R ≡ R+180° (an outer torsion "
      "of 135° is a −45° twist), while C is a genuine 360° coordinate and C = 80° and 100° are *different* "
      "structures. Mirror images (L,C,R) → (−L,−C,−R) are folded onto C > 0.")
    w("")

    # ---------------- signed descriptor ----------------
    rel = df["sign_reliable"]
    w("### Signed relative-orientation descriptor\n")
    w("A fluorenyl plane has no intrinsic orientation: its local C2 axis (through C9 and the linker bond) lies in the "
      "plane and reverses the normal, so any sign built from fluorenyl atom order alone changes under a symmetry "
      "relabelling and is not physical. The descriptor used instead is the sense of the rotation that takes plane 1 "
      "into plane 4, measured about the radical–radical axis C41→C28: θ_FF,signed = sign[(n₁ × n₄)·â]·θ_FF with n₄ "
      "chosen so that n₁·n₄ ≥ 0. It is invariant to flipping either normal and to swapping the two ends, and changes sign "
      "under mirroring (it is a chirality label in the C > 0 convention).")
    w(f"It is only defined when the planes' intersection line is not perpendicular to the radical axis and θ_FF is "
      f"away from 0° and 90°; with |cos| ≥ 0.3 and 5° < θ_FF < 85° this holds for {int(rel.sum())} of {len(df)} "
      "geometries (`sign_reliable` in the CSV). It does not separate the families: the sign is mixed inside most "
      "families. Among the minima, M2 (+55°) and M5 (−11°) have opposite senses; M1 and M3 sit at θ_FF ≈ 85–89°, "
      "where the sign is undefined. It is therefore reported but not used to define families.\n")

    # ---------------- free minima ----------------
    w("## The free minima on this coordinate\n")
    t = fm[["minimum", "start", "L", "C", "R", "theta_L", "theta_C", "theta_R", "theta_FF", "theta_FF_signed",
            "family", "E_meV"]].copy()
    t.columns = ["minimum", "lowest-E start", "L", "C", "R", "θ_L", "θ_C", "θ_R", "θ_FF", "θ_FF signed", "family",
                 "E (meV)"]
    w(md_table(t))
    w("\nOuter torsions are listed as relaxed signed dihedrals; modulo the fluorenyl flip, M2's 145° is a −35° twist.\n")

    # ---------------- main figure ----------------
    w("## Energy vs θ_FF (HS, colored by C)\n")
    w("![Energy vs fluorene angle, colored by C](energy_vs_fluorene_angle_C.png)\n")
    w("Blue = C < 90°, red = C > 90° (darker = farther from 90°), gray = C = 90°. The low-energy points at "
      "θ_FF ≳ 80° are almost all C ≈ 100° with outer twists of 30–50° in the favoured handedness. The scatter forms no "
      "single curve: every θ_FF window contains geometries several hundred meV apart.\n")

    # ---------------- degeneracy ----------------
    w(f"## Degeneracy at matched θ_FF (±{tol:.0f}° windows)\n")
    it = res["intervals"].copy()
    it.columns = ["θ_FF (°)", "n", "E min", "E median", "E max", "spread", "L (mod 180)", "C", "R (mod 180)",
                  "families", "lowest"]
    w(md_table(it, "{:.0f}"))
    w("\nEnergies in meV relative to M1. L and R are folded to [−90°, 90°) (the fluorenyl flip).\n")
    nl = res["noise_low"]
    w(f"**How large must an energy difference be to count?** Symmetry-equivalent constrained points "
      f"((L,C,R) ≡ (R,C,L), L ≡ L+180°) should have identical energies. Below 300 meV they differ by at most "
      f"{nl.loc[~nl['planar_outer'], 'dE'].max():.0f} meV when no outer torsion is held at 0/180° "
      f"(n = {int((~nl['planar_outer']).sum())}), but by up to {nl.loc[nl['planar_outer'], 'dE'].max():.0f} meV and "
      f"{nl.loc[nl['planar_outer'], 'dtheta_FF'].max():.0f}° in θ_FF when one is "
      f"(n = {int(nl['planar_outer'].sum())}). A planar outer constraint is relieved by out-of-plane bending, which can "
      "go either way; the constrained relaxation picks one branch. So a nominal (L,C,R) does not fully determine the "
      f"structure near planarity, and only differences > {res['sig_dE']:.0f} meV are called significant below.\n")
    sk = res["sig_by_kind"].reset_index()
    sk.columns = ["what differs between the two geometries", "pairs", "median ΔE", "max ΔE"]
    w(f"Pairs with |Δθ_FF| ≤ {tol:.0f}° and ΔE > {res['sig_dE']:.0f} meV, by what differs between them (all pairs; "
      f"{len(res['pairs'])} matched pairs in total):\n")
    w(md_table(sk, "{:.0f}"))
    w("")
    w("**What sets the energy at fixed θ_FF** (fraction of the within-window energy variance explained by each "
      "grouping, pooled over the 4° windows):\n")
    w("| grouping | η² within windows |\n|---|---|")
    for k, v in ew.items():
        w(f"| {k} | {v:.2f} |")
    w(f"| *(for comparison) θ_FF window itself, over all data* | {res['eta_theta']:.2f} |")
    w("")
    w("The degeneracy is mainly a **central-torsion** effect, then **outer** torsions, with handedness adding to both. "
      "C alone accounts for most of the spread because C shifts the energy by hundreds of meV "
      "(`angle_scan.md`: E_min(C) rises from 0 at C = 100° to ~1 eV at C = 20° and 160°) while compensating outer "
      "torsions can keep θ_FF fixed. The outer torsions with their sign explain more than their magnitudes alone "
      f"({ew['outer torsions with handedness']:.2f} vs {ew['outer magnitudes |L|,|R| (no handedness)']:.2f}), and "
      f"family × C reaches {ew['family x C']:.2f}: handedness does not change the energy by itself as much as C does, "
      "but it selects which (L,C,R) combinations land on a given θ_FF.\n")
    w("### Representative examples\n")
    w("![Internal geometry degeneracy](internal_geometry_degeneracy.png)\n")
    rows = []
    for title, a, b in res["examples"]:
        for lab in (a, b):
            r = by.loc[lab]
            rows.append({"example": title, "geometry": (r["minimum"] + " (free)") if r["kind"] == "free" else lab,
                         "L": r["L"], "C": r["C"], "R": r["R"], "θ_L": r["theta_L"], "θ_C": r["theta_C"],
                         "θ_R": r["theta_R"], "θ_FF": r["theta_FF"], "family": r["family"], "E (meV)": r["E_meV"]})
    w(md_table(pd.DataFrame(rows), "{:.0f}"))
    w("")
    w("- (a) M1 vs M2: identical local plane angles, opposite sense of all three twists relative to the asymmetric "
      "backbone (C = 104° vs 76°) → θ_FF 89° vs 55°, 38 meV apart. Same magnitudes, different terminal angle.")
    w("- (b) same θ_FF (88.7–88.8°), outer twists of the same sign, C on the other side of 90°: the outer handedness "
      "becomes the disfavoured one (T−−) and the energy rises by >100 meV. This is the 45/45 ↔ 135/135 crossover of "
      "`angle_scan.md` seen at fixed θ_FF.")
    w("- (c) same C = 100°, outer rings 30/30° vs 45/90°: θ_FF differs by only ~1° while the energy rises by "
      "~275 meV; different outer torsions compensate each other on θ_FF.")
    w("- (d) same co-twisted family, C = 104° vs 140°: identical θ_FF (89°), ~360 meV apart — central strain is "
      "invisible on this coordinate.\n")

    # ---------------- families ----------------
    w("## Conformational families\n")
    w("Families are built from the internal torsions, never from θ_FF. Each outer torsion, folded modulo the "
      "fluorenyl flip to [−90°, 90°), is *near-planar* (|·| ≤ 22.5°), *orthogonal* (|·| ≥ 67.5°) or *twisted*; a "
      "twisted ring is `+` if its sign matches the side of C relative to 90° (sign(L_fold)·sign(C − 90°) > 0) — the "
      "handedness that `angle_scan.md` found lower in energy on both sides of 90° (45/45 for C > 90°, 135/135 for "
      "C < 90°) — and `−` otherwise. The orthogonal-outer families are the n_outer_90 classes of "
      "`exchange_analysis.md`.\n")
    rows = []
    for f in FAMILY_ORDER:
        g = df[df["family"] == f]
        if g.empty:
            continue
        rows.append({"family": f, "name": FAMILIES[f][0], "definition": FAMILIES[f][1], "n": len(g),
                     "E min": g["E_meV"].min(), "E median": g["E_meV"].median(),
                     "θ_FF range": f"{g['theta_FF'].min():.0f}–{g['theta_FF'].max():.0f}",
                     "free minima": ", ".join(sorted(set(g["minimum"].dropna()))) or "–"})
    w(md_table(pd.DataFrame(rows), "{:.0f}"))
    w("\n![Energy vs fluorene angle by family](energy_vs_fluorene_angle_families.png)\n")
    w("- **M1 and M2 are both co-twisted (T++)** but on opposite sides of C = 90°; they are pseudo-mirror images that "
      "the asymmetric backbone makes inequivalent (38 meV). M3 and M4 are mixed-twist (one ring of each handedness), "
      "M5 is counter-twisted.")
    w("- The families overlap almost completely in θ_FF (most span 40–90°), so θ_FF cannot identify a family. At their "
      "lowest points the energies order as T++ < T90 ≈ T+− < near-planar ≈ T−− < one-orthogonal < both-orthogonal.")
    w("- Within the constrained data no family is a continuous branch along θ_FF: neighbouring points in θ_FF belong "
      "to different C and different outer torsions, and only the free relaxations establish where minima are.\n")

    # ---------------- envelope ----------------
    w(f"## Lower envelope ({bin5:.0f}° bins)\n")
    w("![Lower envelope](energy_lower_envelope.png)\n")
    w("Three different quantities, not to be confused:")
    w("1. **Sampled lower envelope** — the lowest energy *among the geometries that happen to exist* in each bin. "
      "It depends on where the grid points fall; it is not a potential-energy curve and says nothing about barriers.")
    w("2. **Unconstrained local minima** — M1–M5 (stars), converged downhill optimizations (no Hessian check).")
    w("3. **A θ_FF-constrained relaxed profile** — energy minimized over all other coordinates at each fixed θ_FF. "
      "*Not computed here*; it would need a constraint on θ_FF itself (or a dense L,C,R search per θ_FF).\n")
    t = env[["lo", "hi", "n", "E_min", "label", "family", "E_min_constrained", "envelope_local_min"]].copy()
    t["bin"] = [f"{a:.0f}–{b:.0f}" for a, b in zip(t["lo"], t["hi"])]
    t["lowest geometry"] = t["label"].fillna("–")
    t["dip"] = ["yes" if d else "" for d in t["envelope_local_min"]]
    t = t[["bin", "n", "E_min", "lowest geometry", "family", "E_min_constrained", "dip"]]
    t.columns = ["θ_FF bin", "n", "E min, all (meV)", "lowest geometry", "family", "E min, constrained only", "dip"]
    w(md_table(t, "{:.0f}"))
    w("")
    w("Dips of the envelope (bins lower than both neighbours) and what supports them:\n")
    for r in env[env["envelope_local_min"]].itertuples():
        lab = r.label
        if r.n < 3 and r.edge_bin and r.bin == 0:
            w(f"- {r.lo:.0f}–{r.hi:.0f}°: {lab} ({r.E_min:.0f} meV) — edge bin with {r.n} points, strained "
              "near-planar geometry; not a minimum candidate.")
            continue
        if lab.startswith("free:"):
            w(f"- {r.lo:.0f}–{r.hi:.0f}°: free minimum {by.loc[lab, 'minimum']} itself ({r.E_min:.0f} meV) — an "
              "actual unconstrained local minimum.")
        elif lab in starts:
            mm, th = starts[lab]
            w(f"- {r.lo:.0f}–{r.hi:.0f}°: constrained {lab} ({r.E_min:.0f} meV). Released freely, it relaxed to {mm} "
              f"at θ_FF = {th:.0f}°: the dip is **not** a stable structure at this angle.")
        else:
            w(f"- {r.lo:.0f}–{r.hi:.0f}°: constrained {lab} ({r.E_min:.0f} meV); no free relaxation started here, "
              "stability unknown.")
    w("")
    w("The envelope has its lowest values at 85–90° (M1) and 50–60° (M2), with further dips from M4 (65–70°) and "
      "M5 (10–15°). The constrained-only envelope (dashed) misses the M2 dip entirely: in 50–60° the best constrained "
      "point is >200 meV, while the constrained point at 42° (L135_C080_R135, 76 meV) relaxes into M2 at 55°. "
      "So the sampled envelope's dip positions are partly an artefact of the grid. The two favourable orientations "
      "supported by unconstrained optimization are ~89° (M1) and ~55° (M2, +38 meV), not 20° and 90°.\n")

    # ---------------- Zuo ----------------
    w("## Comparison with Zuo et al.\n")
    w("| | Zuo, junction (Fig. 2a) | Zuo, isolated (SI Fig. S1a) | this work, isolated |")
    w("|---|---|---|---|")
    w("| method | VASP, PBE-D3 relaxation, SCAN energies and J, Au electrodes | B3LYP/6-31G(d,p), Gaussian | R2SCAN-3C (ORCA) |")
    w("| sampling | θ scan, one structure per θ (protocol not given in detail) | θ scan | constrained L,C,R grid + "
      "free relaxations |")
    w(f"| minima | θ = 20° (AFM) and 90° (FM) | global at 90°; shallow secondary at ~40–45° (+12 meV); ~30 meV hump at "
      f"65–70° | M1 at {m['M1'].theta_FF:.0f}° (global); M2 at {m['M2'].theta_FF:.0f}° (+{m['M2'].E_meV:.0f}); "
      f"M3/M4 at {m['M3'].theta_FF:.0f}/{m['M4'].theta_FF:.0f}° (+{m['M3'].E_meV:.0f}/+{m['M4'].E_meV:.0f}); "
      f"M5 at {m['M5'].theta_FF:.0f}° (+{m['M5'].E_meV:.0f}) |")
    w(f"| θ = 0° | — | +153 meV | lowest sampled < 5°: +{env.iloc[0]['E_min']:.0f} meV (strained) |")
    w("")
    w("Zuo's isolated values were read by eye from the rendered SI figure (±2 meV); the junction values are only "
      "described qualitatively (the panel's axis values were not extracted).\n")
    w("- **One or several low-energy families?** Several. Within 100 meV of M1 there are four distinct free minima "
      "(M1–M4) in two families (co-twisted, mixed twist), on both sides of C = 90°.")
    w(f"- **Low-energy structures near 20° and 90°?** Near 90°: yes, the global minimum. Near 20°: no — the best sampled "
      f"structure in 15–25° is {band(15, 25):.0f} meV above M1 (counter-twisted, L = R ≈ 130° at C = 100°), and the "
      f"nearest true minimum is M5 at {m['M5'].theta_FF:.0f}° (+{m['M5'].E_meV:.0f} meV, about 6.5 k_BT at 300 K).")
    w("- **What generates small θ_FF?** Counter-twisted outer rings (both in the disfavoured handedness, e.g. "
      "L = R ≈ 120–150° at C = 100°, or M5 = (135°, 111°, 135°)) and near-planar outer rings at large C. Both are "
      "high-energy families in the isolated molecule.")
    w("- **Do the free minima support a second family?** They support a second *orientation* (M2, ~55°, +38 meV) that "
      "is the pseudo-mirror of M1, i.e. the same co-twisted family on the other side of C = 90°. This matches the shape "
      "of Zuo's isolated curve (90° global, secondary near 40–55°) better than their junction curve.")
    w("- **Does the projection hide energetic differences?** Yes, strongly: the 88–90° bin alone spans "
      f"{res['intervals'].iloc[-1]['E_min']:.0f}–{res['intervals'].iloc[-1]['E_max']:.0f} meV, and θ_FF windows "
      f"explain only {100 * res['eta_theta']:.0f}% of the energy variance. A single θ value does not define a "
      "structure for this molecule.")
    w("- **Cause of the difference with the junction:** not attributable to the electrodes alone. Functional (SCAN/PBE "
      "vs B3LYP vs R2SCAN-3C), basis (plane waves vs Gaussian), dispersion, the scan protocol (one structure per θ vs "
      "all L,C,R combinations) and the Au contacts all differ. Zuo's own isolated-molecule curve already differs "
      "from their junction curve; ours is closer to the former.\n")

    # ---------------- exchange ----------------
    if len(x) > 5:
        xp = res["ex_pairs"]
        low, used = [], set()  # largest differences, each geometry used once
        for r in xp[(xp["E_low"] < 300) & (xp["E_high"] < 300)].sort_values("ddE", ascending=False).itertuples():
            if r.low not in used and r.high not in used:
                low.append(r._asdict())
                used |= {r.low, r.high}
            if len(low) == 5:
                break
        low = pd.DataFrame(low)
        orth = x[x["family"].isin(["O1", "O2"])]
        rest = x[~x["family"].isin(["O1", "O2"])]
        neg = x[x["dE_PBE0_meV"] < 0]
        w("## Exchange splitting vs θ_FF (PBE0)\n")
        w("![Exchange vs fluorene angle](exchange_vs_fluorene_angle.png)\n")
        w(f"- {len(x)} geometries have PBE0/def2-TZVP HS and BS single points at exactly the same structure (coordinates "
          "checked against the R2SCAN-3C geometry, RMSD < 0.002 Å; M1 uses the BS-relaxed M1 geometry, 0.003 Å from the "
          "HS one). ΔE = E_HS − E_BS > 0 means BS lower.")
        w(f"- ρ(ΔE, θ_FF) = {res['rho_theta'][0]:.2f} (p = {res['rho_theta'][1]:.2f}): no monotonic relation. "
          f"ρ(ΔE, P) = {res['rho_P'][0]:.2f}. θ_FF windows explain {100 * res['eta_ex_theta']:.0f}% of the ΔE variance; "
          f"within a window the family explains {100 * res['eta_ex_family_within']:.0f}%.")
        w(f"- {int((xp['ddE'] > 2).sum())} of {len(xp)} pairs with |Δθ_FF| ≤ {tol:.0f}° differ in ΔE by more than 2 meV. "
          "Largest differences between two geometries that are both below 300 meV:\n")
        t = low[["low", "high", "theta_low", "theta_high", "dE_low", "dE_high", "E_low", "E_high", "fam_low",
                 "fam_high"]].copy()
        t.columns = ["geometry A", "geometry B", "θ_FF A", "θ_FF B", "ΔE A (meV)", "ΔE B (meV)", "E A (meV)",
                     "E B (meV)", "family A", "family B"]
        w(md_table(t, "{:.2f}"))
        w("")
        w(f"- Geometries with an orthogonal outer ring (O1/O2, n = {len(orth)}) have |ΔE| < 0.5 meV in "
          f"{100 * (orth['dE_PBE0_meV'].abs() < 0.5).mean():.0f}% of cases (median {orth['dE_PBE0_meV'].median():.2f} meV); "
          f"the others (n = {len(rest)}) have median {rest['dE_PBE0_meV'].median():.1f} meV over the same θ_FF range. "
          "The exchange is controlled by whether each *link* is conjugated (P), not by the angle between the end "
          "planes. This is the information Zuo's single collective angle cannot capture; their junction J(θ) of "
          "−5 → +13 meV can only be read along their particular (θ1, θ2, θ3) path.")
        w("- Zuo's isolated B3LYP J (SI Fig. S1b, dashed, plotted as −J ≈ ΔE since ⟨S²⟩_HS − ⟨S²⟩_BS ≈ 1) stays within "
          "±2.2 meV and turns positive (FM) near 65–80°. Our PBE0 values are mostly positive (BS lower); the "
          f"{len(neg)} negative ones are ≥ {neg['dE_PBE0_meV'].min():.2f} meV and "
          f"{100 * neg['family'].isin(['O1', 'O2']).mean():.0f}% of them are O1/O2.\n")

    w("## Files\n")
    w("- `fluorene_angle_analysis.csv` — one row per geometry: source, nominal and relaxed L, C, R, folded outer "
      "torsions, θ_L/θ_C/θ_R, θ_FF, signed θ_FF and its reliability, fluorenyl planarity, family, P, R2SCAN-3C energy "
      "(Eh and meV rel. to M1), PBE0 ΔE where available, geometry file.")
    w("- `energy_vs_fluorene_angle_C.png`, `energy_vs_fluorene_angle_families.png`, `energy_lower_envelope.png`, "
      "`internal_geometry_degeneracy.png`, `exchange_vs_fluorene_angle.png`.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
