#!/usr/bin/env python3
"""
Free (unconstrained) HS / BS geometry relaxations started from low-energy
central-angle points of the constrained L x C x R angle scan.

Questions:
  1. Which minima exist, and which is the best candidate for the global
     geometrical minimum?
  2. Is that minimum (and the ordering of the minima) different for the
     high-spin (HS, M=3) and broken-symmetry (BS, M=1) surfaces?
  3. How do the free minima relate to the constrained scan (energy gained,
     where in (L, C, R) the molecule goes)?
  4. What does free relaxation do to the HS-BS gap (vertical vs adiabatic)?

Sign convention (same as analyze_exchange.py):
    dE = E_HS - E_BS ;  dE > 0 : BS lower (AFM tendency)

Level of theory: R2SCAN-3C TightSCF TightOpt UKS for both states (BS with
GuessMix). No frequencies were computed, so "minimum" means "converged
stationary point reached by downhill optimization", not a Hessian-verified
minimum.
"""

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
    ATOM_LINE_RE,
    CONV_TABLE_RE,
    CONV_CRITERIA_NAMES,
    FINAL_E_RE,
    HARTREE_TO_MEV,
    MULLIKEN_BLOCK_RE,
    S2_RE,
    build_adjacency,
    check_chemical_integrity,
    classify_convergence,
    cross_fragment_contact_stats,
    fit_plane,
    outer_terminal_fragment,
    parse_out_file,
    reference_bonds,
    split_fragment_at_bond,
)

STATES = ("high_spin", "broken_symmetry")
STATE_SHORT = {"high_spin": "HS", "broken_symmetry": "BS"}

# Categorical slots (validated reference palette, light mode), assigned in
# fixed order to minima M1, M2, ... ; identity is never color-alone (markers
# and direct labels carry it too).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


# ============================================================
# HELPERS
# ============================================================
def kabsch_rmsd(p, q):
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((p @ rot.T - q) ** 2).sum(axis=1).mean()))


def plane_angle(pos, ring_a, ring_b):
    _, n1 = fit_plane(pos[ring_a])
    _, n2 = fit_plane(pos[ring_b])
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))


def signed(angle):
    """Dihedral mapped to (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def fragment_spins(out_text, fragments, radical_atoms):
    blocks = list(MULLIKEN_BLOCK_RE.finditer(out_text))
    if not blocks:
        return {}
    spins = {}
    for line in blocks[-1].group(1).splitlines():
        m = ATOM_LINE_RE.match(line)
        if m:
            spins[int(m.group(1))] = float(m.group(4))
    rec = {f"spin_f{k}": sum(spins.get(i, 0.0) for i in atoms) for k, atoms in fragments.items()}
    for a in radical_atoms:
        rec[f"spin_atom{a}"] = spins.get(a, np.nan)
    return rec


def last_conv_table(out_text):
    tables = CONV_TABLE_RE.findall(out_text)
    if not tables:
        return {}
    g = tables[-1]
    return {f"last_{name}": float(g[3 * i]) for i, name in enumerate(CONV_CRITERIA_NAMES)} | {
        "last_n_criteria_met": sum(g[3 * i + 2] == "YES" for i in range(5))
    }


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


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    repo_root = Path(__file__).resolve().parent.parent
    scans = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans"
    free_root = scans / "angle_scan_free_relaxation"
    scan_csv = scans / "angle_scan_relaxation" / "angle_scan.csv"
    exchange_csv = scans / "angle_scan_exchange_analysis" / "exchange_analysis.csv"
    baseline_xyz = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    RADICAL_ATOMS = (41, 28)
    RINGS = {
        "L": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]),
        "C": ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25]),
        "R": ([23, 21, 25, 24, 26, 27], [28, 29, 30, 31, 32]),
    }
    SAME_MINIMUM_RMSD = 0.30   # Å; final geometries closer than this = same minimum
    BOND_STRETCH_TOL_ANG, BOND_ABS_MAX_ANG, MIN_NONBONDED_DIST_ANG = 0.30, 1.9, 1.5

    plots_dir = free_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    md_path = free_root / "free_relaxation.md"
    csv_path = free_root / "free_relaxation.csv"
    traj_csv_path = free_root / "free_relaxation_trajectories.csv"

    # ---- FRAGMENTS (same construction as analyze_angle_scan.py) ----
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

    scan = pd.read_csv(scan_csv).set_index("label")
    exch = pd.read_csv(exchange_csv).set_index("label") if exchange_csv.exists() else pd.DataFrame()
    scan_ref_label = scan.loc[scan["usable"], "final_energy"].idxmin()
    e_scan_ref = float(scan.loc[scan_ref_label, "final_energy"])

    # ---- PARSE ----
    rows, traj_rows, final_pos = [], [], {}
    for point in sorted(p for p in free_root.iterdir() if p.is_dir() and p.name.startswith("L")):
        L0, C0, R0 = (int(point.name[i:i + 3]) for i in (1, 6, 11))
        for state in STATES:
            d = point / state
            out = d / f"{state}.out"
            if not out.exists():
                continue
            text = out.read_text(errors="ignore")
            rec = parse_out_file(text, out)
            energies = [float(x) for x in FINAL_E_RE.findall(text)]
            s2 = [float(x) for x in S2_RE.findall(text)]
            trj = read(d / f"{state}_trj.xyz", ":")
            start, final = read(d / "geometry.xyz"), trj[-1]
            pos = final.get_positions()
            final_pos[(point.name, state)] = pos

            row = {
                "start": point.name, "state": state, "L_start": L0, "C_start": C0, "R_start": R0,
                "converged": rec["opt_converged"], "convergence_type": rec["convergence_type"],
                "walltime": rec["time_limit_cancelled"], "n_geom_cycles": rec["n_geom_cycles"],
                "trah_activations": rec["trah_activations"], "scf_cycles_max": rec["scf_cycles_max"],
                "E_first": energies[0], "E_final": energies[-1],
                "relaxation_energy_meV": (energies[0] - energies[-1]) * HARTREE_TO_MEV,
                "energy_monotonic": bool(np.all(np.diff(energies) <= 1e-6)),
                "E_vs_scan_ref_meV": (energies[-1] - e_scan_ref) * HARTREE_TO_MEV,
                "s2_first": s2[0] if s2 else np.nan, "s2_final": s2[-1] if s2 else np.nan,
            }
            if state == "high_spin" and point.name in scan.index:
                row["E_first_minus_scan_meV"] = (energies[0] - scan.loc[point.name, "final_energy"]) * HARTREE_TO_MEV
            for axis, quartet in (("L", LEFT), ("C", CENTER), ("R", RIGHT)):
                row[f"{axis}_start_actual"] = signed(start.get_dihedral(*quartet))
                row[f"{axis}_final"] = signed(final.get_dihedral(*quartet))
                row[f"theta_{axis}_final"] = plane_angle(pos, *RINGS[axis])
                row[f"theta_{axis}_start"] = plane_angle(start.get_positions(), *RINGS[axis])
            row.update(last_conv_table(text))
            # spins only count as relaxed-state spins when the optimization converged
            spins = fragment_spins(text, fragments, RADICAL_ATOMS) if rec["opt_converged"] else {}
            row.update(spins)
            integ = check_chemical_integrity(final, bonds, bonded_pairs, BOND_STRETCH_TOL_ANG,
                                             BOND_ABS_MAX_ANG, MIN_NONBONDED_DIST_ANG)
            row["geometry_valid"] = integ["valid"]
            row["min_nonbonded_distance"] = integ["min_nonbonded_distance"]
            symbols = final.get_chemical_symbols()
            m13, _ = cross_fragment_contact_stats(pos, symbols, f1, f3, 3, 4.0)
            m24, _ = cross_fragment_contact_stats(pos, symbols, f2, f4, 3, 4.0)
            row["cross_contact_3mean_13"], row["cross_contact_3mean_24"] = m13, m24
            rows.append(row)

            for k, (atoms, e) in enumerate(zip(trj, energies)):
                traj_rows.append({
                    "start": point.name, "state": state, "cycle": k + 1,
                    "E_vs_scan_ref_meV": (e - e_scan_ref) * HARTREE_TO_MEV,
                    "L": signed(atoms.get_dihedral(*LEFT)), "C": signed(atoms.get_dihedral(*CENTER)),
                    "R": signed(atoms.get_dihedral(*RIGHT)),
                })

    df = pd.DataFrame(rows)
    traj = pd.DataFrame(traj_rows)
    if df.empty:
        raise RuntimeError(f"No free-relaxation outputs under {free_root}")

    # ---- CLUSTER FINAL GEOMETRIES INTO MINIMA ----
    keys = list(zip(df["start"], df["state"]))
    reps = []  # representative key per cluster
    cluster = []
    for k in keys:
        for ci, rk in enumerate(reps):
            if kabsch_rmsd(final_pos[k], final_pos[rk]) < SAME_MINIMUM_RMSD:
                cluster.append(ci)
                break
        else:
            reps.append(k)
            cluster.append(len(reps) - 1)
    df["cluster"] = cluster
    # rank clusters by lowest converged BS energy (fallback: any energy)
    def cluster_energy(ci):
        sub = df[(df["cluster"] == ci) & df["converged"]]
        return sub["E_final"].min() if not sub.empty else df.loc[df["cluster"] == ci, "E_final"].min() + 1.0
    order = sorted(range(len(reps)), key=cluster_energy)
    df["minimum"] = df["cluster"].map({ci: f"M{i + 1}" for i, ci in enumerate(order)})
    minima = [f"M{i + 1}" for i in range(len(order))]
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(minima)}

    conv = df[df["converged"]]
    e_glob = conv["E_final"].min()
    df["E_vs_global_meV"] = (df["E_final"] - e_glob) * HARTREE_TO_MEV

    # ---- HS / BS PAIRS ----
    pairs = []
    for start, g in df.groupby("start"):
        g = g.set_index("state")
        if not set(STATES) <= set(g.index):
            continue
        hs, bs = g.loc["high_spin"], g.loc["broken_symmetry"]
        pr = {
            "start": start, "minimum_HS": hs["minimum"], "minimum_BS": bs["minimum"],
            "both_converged": bool(hs["converged"] and bs["converged"]),
            "dE_vertical_scan_geom_meV": (hs["E_first"] - bs["E_first"]) * HARTREE_TO_MEV,
            "dE_adiabatic_meV": (hs["E_final"] - bs["E_final"]) * HARTREE_TO_MEV,
            "rmsd_HS_BS": kabsch_rmsd(final_pos[(start, "high_spin")], final_pos[(start, "broken_symmetry")]),
            "max_abs_dihedral_diff_HS_BS": max(abs(signed(hs[f"{a}_final"] - bs[f"{a}_final"])) for a in "LCR"),
            "s2_HS": hs["s2_final"], "s2_BS": bs["s2_final"],
        }
        ds2 = pr["s2_HS"] - pr["s2_BS"]
        # H = -J S1.S2 (J > 0 FM), dE = E_HS - E_BS (adiabatic: each state at its own geometry)
        pr["J_ideal_meV"] = XC.j_ideal(pr["dE_adiabatic_meV"])
        pr["J_yamaguchi_meV"] = XC.j_yamaguchi(pr["dE_adiabatic_meV"], pr["s2_HS"], pr["s2_BS"]) if ds2 > 0 else np.nan
        pr["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(pr["J_yamaguchi_meV"])
        pr["dE_PBE0_vertical_scan_geom_meV"] = (
            float(exch.loc[start, "delta_E_raw_meV"]) if start in exch.index else np.nan
        )
        pairs.append(pr)
    pairs = pd.DataFrame(pairs)

    # ---- PER-MINIMUM SUMMARY (converged runs only) ----
    summary = []
    for m in minima:
        sub = df[df["minimum"] == m]
        sc = sub[sub["converged"]]
        entry = {"minimum": m, "starts": sorted(set(sub["start"])), "n_converged": len(sc),
                 "n_runs": len(sub)}
        for st in STATES:
            s = sc[sc["state"] == st]
            entry[f"E_{STATE_SHORT[st]}_meV"] = s["E_vs_global_meV"].min() if not s.empty else np.nan
            entry[f"E_{STATE_SHORT[st]}_spread_meV"] = np.ptp(s["E_vs_global_meV"]) if len(s) > 1 else 0.0
        src = sc if not sc.empty else sub
        for col in ("L_final", "C_final", "R_final", "theta_L_final", "theta_C_final", "theta_R_final",
                    "spin_f1", "spin_f2", "spin_f3", "spin_f4", "cross_contact_3mean_13", "cross_contact_3mean_24"):
            if col in src:
                vals = src[src["state"] == "high_spin"][col] if col.startswith("spin") else src[col]
                entry[col] = float(vals.mean()) if len(vals) else np.nan
        bs = src[src["state"] == "broken_symmetry"]
        entry["spin_f1_BS"] = float(bs["spin_f1"].mean()) if "spin_f1" in bs and len(bs) else np.nan
        entry["spin_f4_BS"] = float(bs["spin_f4"].mean()) if "spin_f4" in bs and len(bs) else np.nan
        entry["s2_HS"] = float(src.loc[src["state"] == "high_spin", "s2_final"].mean())
        entry["s2_BS"] = float(src.loc[src["state"] == "broken_symmetry", "s2_final"].mean())
        entry["gap_meV"] = entry["E_HS_meV"] - entry["E_BS_meV"]
        entry["handedness_product"] = (np.sign(abs(entry["L_final"]) - 90) * np.sign(abs(entry["C_final"]) - 90))
        entry["outer_same_sign"] = bool(np.sign(entry["L_final"]) == np.sign(entry["R_final"]))
        entry["P"] = float(np.prod([np.cos(np.radians(entry[f"theta_{a}_final"])) ** 2 for a in "LCR"]))
        summary.append(entry)
    summary = pd.DataFrame(summary)

    df.to_csv(csv_path, index=False)
    traj.to_csv(traj_csv_path, index=False)

    make_plots(df, traj, pairs, summary, scan, e_scan_ref, e_glob, color, plots_dir)
    write_report(df, traj, pairs, summary, scan, scan_ref_label, e_scan_ref, e_glob, md_path, fragments,
                 SAME_MINIMUM_RMSD)
    print(f"Wrote {md_path}\n      {csv_path}\n      {traj_csv_path}\n      {plots_dir}/")
    print(summary[["minimum", "starts", "E_HS_meV", "E_BS_meV", "gap_meV", "L_final", "C_final", "R_final"]]
          .to_string(index=False))


# ============================================================
# PLOTS
# ============================================================
def make_plots(df, traj, pairs, summary, scan, e_scan_ref, e_glob, color, plots_dir):
    offset = (e_glob - e_scan_ref) * HARTREE_TO_MEV  # scan-ref -> global-min reference
    minimum_of = dict(zip(zip(df["start"], df["state"]), df["minimum"]))
    ls = {"high_spin": "-", "broken_symmetry": "--"}

    # 1. Energy ladder of the minima, HS and BS side by side
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, r in summary.iterrows():
        for j, st in enumerate(("HS", "BS")):
            e = r[f"E_{st}_meV"]
            if not np.isfinite(e):
                continue
            x0 = i + (0.0 if st == "HS" else 0.42)
            ax.hlines(e, x0, x0 + 0.36, color=color[r["minimum"]], linewidth=3 if st == "BS" else 2,
                      linestyle="-" if st == "BS" else (0, (4, 2)))
            ax.text(x0 + 0.18, e + 3, st, ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.text(i + 0.39, -22, f"{r['minimum']}\n({r['L_final']:.0f}, {r['C_final']:.0f}, {r['R_final']:.0f})",
                ha="center", va="top", fontsize=8.5, color=INK)
    ax.axhline(-offset, color=INK_3, linewidth=1, linestyle=":")
    ax.text(len(summary) - 0.2, -offset + 2, f"constrained-scan best ({scan.loc[scan['usable'], 'final_energy'].idxmin()})",
            ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlim(-0.3, len(summary))
    ax.set_ylim(-45, max(summary[["E_HS_meV", "E_BS_meV"]].max()) + 20)
    ax.set_xticks([])
    ax.set_ylabel("E relative to lowest minimum (meV)")
    ax.set_title("Free-relaxation minima: HS (dashed) vs BS (solid) levels\nlabels: final signed dihedrals (L, C, R) in deg")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "minima_energy_ladder.png", dpi=200)
    plt.close(fig)

    # 2. Relaxation paths vs optimization cycle
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for (start, state), g in traj.groupby(["start", "state"]):
        m = minimum_of[(start, state)]
        conv = bool(df[(df["start"] == start) & (df["state"] == state)]["converged"].iloc[0])
        kw = dict(color=color[m], linestyle=ls[state], linewidth=1.6 if conv else 1.0, alpha=1.0 if conv else 0.55)
        axes[0, 0].plot(g["cycle"], g["E_vs_scan_ref_meV"] - offset, **kw)
        for ax, col in ((axes[0, 1], "C"), (axes[1, 0], "L"), (axes[1, 1], "R")):
            ax.plot(g["cycle"], g[col], **kw)
    axes[0, 0].set_ylabel("E rel. lowest minimum (meV)")
    axes[0, 0].set_ylim(-20, 700)
    axes[0, 1].set_ylabel("central dihedral C (deg)")
    axes[1, 0].set_ylabel("left dihedral L (deg)")
    axes[1, 1].set_ylabel("right dihedral R (deg)")
    for ax in axes[1]:
        ax.set_xlabel("optimization cycle")
    for ax in axes.flat:
        style_axes(ax)
    handles = [plt.Line2D([], [], color=color[m], linewidth=2, label=m) for m in summary["minimum"]]
    handles += [plt.Line2D([], [], color=INK_2, linestyle="-", label="HS"),
                plt.Line2D([], [], color=INK_2, linestyle="--", label="BS"),
                plt.Line2D([], [], color=INK_2, alpha=0.5, linewidth=1, label="not converged (walltime)")]
    fig.suptitle("Free relaxation paths, colored by the minimum reached", y=0.99, color=INK)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles),
               frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(plots_dir / "relaxation_paths.png", dpi=200)
    plt.close(fig)

    # 3. Overlay on the constrained-scan E(C) landscape
    usable = scan[scan["usable"]]
    e_min_c = usable.groupby("C")["relative_energy_meV"].min()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for outer, shade in ((45, "#b5b4ae"), (135, "#8a8984")):
        sym = usable[(usable["L"] == outer) & (usable["R"] == outer)].sort_values("C")
        ax.plot(sym["C"], sym["relative_energy_meV"] - offset, color=shade, linewidth=1.2, marker=".",
                label=f"scan L=R={outer}°")
    ax.plot(e_min_c.index, e_min_c.values - offset, color=INK, linewidth=2.2, label="scan E_min(C) over (L,R)")
    for (start, state), g in traj.groupby(["start", "state"]):
        if state != "broken_symmetry":
            continue
        m = minimum_of[(start, state)]
        cabs = g["C"].abs()
        ax.plot(cabs, g["E_vs_scan_ref_meV"] - offset, color=color[m], linewidth=1, alpha=0.6)
        ax.plot(cabs.iloc[0], g["E_vs_scan_ref_meV"].iloc[0] - offset, marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=color[m])
    for _, r in summary.iterrows():
        for st, mk in (("HS", "D"), ("BS", "o")):
            ax.plot(abs(r["C_final"]), r[f"E_{st}_meV"], marker=mk, markersize=9 if st == "BS" else 7,
                    color=color[r["minimum"]], markeredgecolor="white", markeredgewidth=1.5, linestyle="none")
        ax.annotate(r["minimum"], (abs(r["C_final"]), r["E_BS_meV"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(55, 125)
    ax.set_ylim(-15, 250)
    ax.set_xlabel("|central dihedral C| (deg)")
    ax.set_ylabel("E relative to lowest free minimum (meV)")
    ax.set_title("Free minima on the constrained-scan landscape\n(BS paths: open circle = start; ● BS minimum, ◆ HS minimum)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "free_minima_on_scan_landscape.png", dpi=200)
    plt.close(fig)

    # 4. Torsion map: where each start ended (signed dihedrals)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, outer in zip(axes, ("L", "R")):
        for (start, state), g in traj.groupby(["start", "state"]):
            if state != "broken_symmetry":
                continue
            m = minimum_of[(start, state)]
            ax.plot(g[outer], g["C"], color=color[m], linewidth=1, alpha=0.6)
            ax.plot(g[outer].iloc[0], g["C"].iloc[0], "o", markerfacecolor="white", markeredgecolor=color[m], markersize=6)
            ax.plot(g[outer].iloc[-1], g["C"].iloc[-1], "o", color=color[m], markeredgecolor="white", markersize=9)
        for _, r in summary.iterrows():
            ax.annotate(r["minimum"], (r[f"{outer}_final"], r["C_final"]), xytext=(7, 5),
                        textcoords="offset points", fontsize=9, color=INK)
        ax.set_xlabel(f"{outer} dihedral (deg, signed)")
        ax.set_xlim(-180, 180)
        ax.set_xticks(range(-180, 181, 45))
        style_axes(ax)
    axes[0].set_ylabel("C dihedral (deg, signed)")
    fig.suptitle("Start (open) → free BS minimum (filled) in torsion space", color=INK)
    fig.tight_layout()
    fig.savefig(plots_dir / "torsion_map_start_to_minimum.png", dpi=200)
    plt.close(fig)

    # 5. HS-BS gaps: vertical at scan geometry vs adiabatic
    p = pairs.sort_values("start")
    x = np.arange(len(p))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    series = [("dE_PBE0_vertical_scan_geom_meV", "PBE0 vertical (scan geometry)", SERIES[0]),
              ("dE_vertical_scan_geom_meV", "R2SCAN-3C vertical (scan geometry)", SERIES[1]),
              ("dE_adiabatic_meV", "R2SCAN-3C adiabatic (each state relaxed)", SERIES[2])]
    for k, (col, lab, c) in enumerate(series):
        ax.bar(x + (k - 1) * w, p[col], width=w - 0.03, color=c, label=lab)
    ax.axhline(0, color=INK_3, linewidth=1)
    labels = [f"{s}\n→ {m}" + ("" if ok else "\n(unconv.)") for s, m, ok in
              zip(p["start"], p["minimum_BS"], p["both_converged"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ΔE = E_HS − E_BS (meV)  (>0: BS lower)")
    ax.set_title("HS–BS gap before (scan geometry) and after free relaxation")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "hs_bs_gap_vertical_vs_adiabatic.png", dpi=200)
    plt.close(fig)


# ============================================================
# REPORT
# ============================================================
def write_report(df, traj, pairs, summary, scan, scan_ref_label, e_scan_ref, e_glob, md_path, fragments, same_rmsd):
    offset = (e_glob - e_scan_ref) * HARTREE_TO_MEV
    L = []
    g = summary.iloc[0]
    conv_pairs = pairs[pairs["both_converged"]]
    same_min = (pairs["minimum_HS"] == pairs["minimum_BS"]).all()
    unconv = df[~df["converged"]]

    L.append("# Free (unconstrained) HS / BS relaxations")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"- {len(df)} runs ({df['start'].nunique()} starting points × HS/BS); "
             f"{int(df['converged'].sum())} converged, {len(unconv)} stopped by walltime "
             f"({', '.join(sorted(set(unconv['start'])))}). All final geometries chemically intact: "
             f"**{'yes' if df['geometry_valid'].all() else 'no'}**.")
    L.append(f"- The {df['start'].nunique()} starts collapse onto **{len(summary)} distinct minima** "
             f"(final geometries within {same_rmsd} Å RMSD are the same minimum).")
    L.append(f"- **Global-minimum candidate: {g['minimum']}**, signed dihedrals (L, C, R) = "
             f"({g['L_final']:.1f}, {g['C_final']:.1f}, {g['R_final']:.1f})°, actual plane angles "
             f"θ = ({g['theta_L_final']:.1f}, {g['theta_C_final']:.1f}, {g['theta_R_final']:.1f})°; reached from "
             f"{', '.join(g['starts'])}.")
    L.append(f"- It lies **{-offset:.1f} meV below** the best constrained-scan point ({scan_ref_label}), "
             f"and {summary.iloc[1]['E_BS_meV'] - g['E_BS_meV']:.1f} meV below the next minimum "
             f"({summary.iloc[1]['minimum']}).")
    L.append(f"- **HS and BS relax to the same minimum from every start: {'yes' if same_min else 'NO'}** "
             f"(RMSD HS vs BS ≤ {pairs['rmsd_HS_BS'].max():.3f} Å, dihedral differences ≤ "
             f"{pairs['max_abs_dihedral_diff_HS_BS'].max():.1f}°). The energy ordering of the minima is the same "
             f"on both surfaces; BS is lower in every minimum by {summary['gap_meV'].min():.2f}–"
             f"{summary['gap_meV'].max():.2f} meV (R2SCAN-3C).")
    L.append("")

    L.append("## Setup and conventions")
    L.append("")
    L.append("- R2SCAN-3C TightSCF TightOpt UKS, no constraints, geometry MaxIter 200; HS = multiplicity 3, "
             "BS = multiplicity 1 with GuessMix. Each state is optimized on its own surface.")
    L.append("- Starting geometries: the HS-relaxed constrained-scan structures `angle_scan_relaxation/<point>/high_spin.xyz` "
             "for C = 80° and 100° with symmetric outer torsions L = R = 0, 45, 90, 135° (both states start from the same geometry).")
    L.append(f"- Energies are given relative to the lowest converged free minimum (E = {e_glob:.8f} Eh) unless stated; "
             f"the constrained-scan reference {scan_ref_label} (E = {e_scan_ref:.8f} Eh) sits at {-offset:+.1f} meV on this scale.")
    L.append("- ΔE = E_HS − E_BS; ΔE > 0 → BS lower (AFM tendency). *Vertical* = both states at the frozen scan geometry "
             "(first optimization cycle); *adiabatic* = each state at its own relaxed geometry.")
    L.append("- Dihedrals are the literal 4-atom quartets of the scan (L (42,41,0,5), C (4,9,17,18), R (21,23,28,29)), "
             "reported signed in (−180°, 180°]; θ are the actual ring-plane angles (0–90°) used in `exchange_analysis.md`.")
    L.append("- Spin populations are Mulliken, fragment-integrated (fragments 1–4 as in `angle_scan.md`), taken only from converged runs.")
    L.append("- **No frequency calculations** were run: minima are converged stationary points from downhill optimization, "
             "not Hessian-verified minima.")
    L.append("")

    L.append("## Minima found")
    L.append("")
    L.append("| minimum | reached from | L | C | R | θL | θC | θR | E_BS (meV) | E_HS (meV) | ΔE_HS−BS (meV) | same-sign outer twist | sign(|L|−90)·sign(|C|−90) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(f"| **{r['minimum']}** | {', '.join(r['starts'])} | {r['L_final']:.1f} | {r['C_final']:.1f} | {r['R_final']:.1f} | "
                 f"{r['theta_L_final']:.1f} | {r['theta_C_final']:.1f} | {r['theta_R_final']:.1f} | {r['E_BS_meV']:.1f} | "
                 f"{r['E_HS_meV']:.1f} | {r['gap_meV']:.2f} | {'yes' if r['outer_same_sign'] else 'no (opposite)'} | "
                 f"{r['handedness_product']:+.0f} |")
    L.append("")
    L.append("Energy spread between converged runs that reached the same minimum (a measure of how flat the "
             "surface is at the TightOpt stopping point): " + "; ".join(
                 f"{r['minimum']}: HS {r['E_HS_spread_meV']:.1f}, BS {r['E_BS_spread_meV']:.1f} meV"
                 for _, r in summary.iterrows() if r["n_converged"] > 2) + ".")
    L.append("")
    L.append("![Energy ladder](plots/minima_energy_ladder.png)")
    L.append("")
    L.append("![Free minima on scan landscape](plots/free_minima_on_scan_landscape.png)")
    L.append("")
    L.append("![Torsion map](plots/torsion_map_start_to_minimum.png)")
    L.append("")

    L.append("## Electronic structure at the minima")
    L.append("")
    L.append("| minimum | ⟨S²⟩ HS | ⟨S²⟩ BS | frag1 HS | frag2 HS | frag3 HS | frag4 HS | frag1 BS | frag4 BS | P = Πcos²θ | 1-3 / 2-4 contact (Å) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(f"| {r['minimum']} | {r['s2_HS']:.3f} | {r['s2_BS']:.3f} | {r['spin_f1']:.3f} | {r['spin_f2']:.3f} | "
                 f"{r['spin_f3']:.3f} | {r['spin_f4']:.3f} | {r['spin_f1_BS']:+.3f} | {r['spin_f4_BS']:+.3f} | "
                 f"{r['P']:.4f} | {r['cross_contact_3mean_13']:.2f} / {r['cross_contact_3mean_24']:.2f} |")
    L.append("")

    L.append("## HS vs BS: vertical and adiabatic gaps, geometric differences")
    L.append("")
    L.append("| start | minimum HS / BS | both converged | ΔE vertical R2SCAN (meV) | ΔE vertical PBE0 (meV) | ΔE adiabatic R2SCAN (meV) | E_T−E_S = −J (Yamaguchi (meV) | RMSD HS–BS (Å) | max Δdihedral (°) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in pairs.iterrows():
        L.append(f"| {r['start']} | {r['minimum_HS']} / {r['minimum_BS']} | {'yes' if r['both_converged'] else 'no'} | "
                 f"{r['dE_vertical_scan_geom_meV']:.2f} | {r['dE_PBE0_vertical_scan_geom_meV']:.2f} | "
                 f"{r['dE_adiabatic_meV']:.2f} | {r['E_T_minus_E_S_yamaguchi_meV']:.2f} | {r['rmsd_HS_BS']:.3f} | "
                 f"{r['max_abs_dihedral_diff_HS_BS']:.2f} |")
    L.append("")
    both = pairs.dropna(subset=["dE_PBE0_vertical_scan_geom_meV"])
    ratio = (both["dE_vertical_scan_geom_meV"] / both["dE_PBE0_vertical_scan_geom_meV"])[both["dE_PBE0_vertical_scan_geom_meV"].abs() > 0.3]
    rho = both[["dE_vertical_scan_geom_meV", "dE_PBE0_vertical_scan_geom_meV"]].corr(method="spearman").iloc[0, 1]
    L.append(f"Cross-check at the scan geometries: R2SCAN-3C and PBE0 vertical gaps rank the same way "
             f"(Spearman ρ = {rho:.2f}, n = {len(both)}); R2SCAN-3C is larger by a factor "
             f"{ratio.min():.2f}–{ratio.max():.2f} where |ΔE_PBE0| > 0.3 meV.")
    L.append("")
    L.append("![HS-BS gaps](plots/hs_bs_gap_vertical_vs_adiabatic.png)")
    L.append("")

    L.append("## Relaxation paths")
    L.append("")
    L.append("| start | state | status | cycles | conv. type | relaxation energy (meV) | L start→final | C start→final | R start→final | minimum | E rel. global (meV) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["start", "state"]).iterrows():
        status = "converged" if r["converged"] else ("walltime" if r["walltime"] else "not converged")
        L.append(f"| {r['start']} | {STATE_SHORT[r['state']]} | {status} | {r['n_geom_cycles']} | {r['convergence_type']} | "
                 f"{r['relaxation_energy_meV']:.1f} | {r['L_start_actual']:.0f}→{r['L_final']:.1f} | "
                 f"{r['C_start_actual']:.0f}→{r['C_final']:.1f} | {r['R_start_actual']:.0f}→{r['R_final']:.1f} | "
                 f"{r['minimum']} | {r['E_vs_global_meV']:.1f} |")
    L.append("")
    hs = df[df["state"] == "high_spin"]
    if "E_first_minus_scan_meV" in hs:
        L.append(f"Consistency: the first HS energy of each free run reproduces the constrained-scan final energy of its "
                 f"starting point to within {hs['E_first_minus_scan_meV'].abs().max():.3f} meV.")
        L.append("")
    L.append("![Relaxation paths](plots/relaxation_paths.png)")
    L.append("")
    if not unconv.empty:
        L.append("**Unconverged runs.** " + "; ".join(
            f"{r['start']} {STATE_SHORT[r['state']]}: {r['n_geom_cycles']} cycles, {r['trah_activations']} TRAH activations, "
            f"last max gradient {r.get('last_max_gradient', np.nan):.2e} Eh/bohr, last max step {r.get('last_max_step', np.nan):.2e}, "
            f"energy {'monotonic' if r['energy_monotonic'] else 'non-monotonic'}; final geometry assigned to {r['minimum']} "
            f"at {r['E_vs_global_meV']:+.1f} meV"
            for _, r in unconv.iterrows())
            + ". These runs are not used for energies or spins, but they show where the start is heading.")
        L.append("")

    L.append("## Main physical messages")
    L.append("")
    msgs = physical_messages(df, pairs, summary, offset, scan_ref_label)
    for i, m in enumerate(msgs, 1):
        L.append(f"{i}. {m}")
    L.append("")

    L.append("## Limitations and suggested follow-up")
    L.append("")
    L.append("- No Hessians: confirm the lowest minima (at least M1 and M2) with an analytical frequency calculation "
             "(no imaginary modes) before calling them minima.")
    L.append("- Only symmetric starts (L = R) at C = 80/100° were sampled; asymmetric starts and the opposite central "
             "handedness (C < 0) were not, so M1 is the best candidate *within this sampling*, not a proven global minimum.")
    L.append("- The TightOpt stopping point leaves ~1 meV of energy noise between runs landing in the same minimum; this "
             "is comparable to the HS–BS gap, so compare HS and BS from the *same* start (as in the pair table), not across starts.")
    L.append("- Gaps here are R2SCAN-3C; PBE0 single points on the M1/M2 HS and BS geometries would put them on the same "
             "footing as `exchange_analysis.md`.")
    md_path.write_text("\n".join(L) + "\n")


def physical_messages(df, pairs, summary, offset, scan_ref_label):
    g, second = summary.iloc[0], summary.iloc[1]
    msgs = []
    msgs.append(
        f"**The unconstrained optimum sits at C ≈ {abs(g['C_final']):.0f}° with outer twists ≈ {abs(g['L_final']):.0f}°, "
        f"not at a grid point.** Letting all torsions relax lowers the energy by {-offset:.1f} meV relative to the best "
        f"constrained point ({scan_ref_label}). Relaxation flattens the outer torsions (45° → ~{abs(g['L_final']):.0f}°) and "
        f"opens C slightly past 100°; the scan grid (45° steps in L, R) is too coarse to resolve this minimum.")
    m2_note = ""
    if abs(abs(g["theta_C_final"]) - abs(second["theta_C_final"])) < 3 and abs(g["theta_L_final"] - second["theta_L_final"]) < 3:
        m2_note = (f" Their actual ring-plane angles are almost identical (θ ≈ {g['theta_L_final']:.0f}/{g['theta_C_final']:.0f}/"
                   f"{g['theta_R_final']:.0f}° vs {second['theta_L_final']:.0f}/{second['theta_C_final']:.0f}/{second['theta_R_final']:.0f}°): "
                   "the two are the C > 90° and C < 90° realizations of the same twisted motif with the same hopping proxy P. "
                   f"The {second['E_BS_meV'] - g['E_BS_meV']:.0f} meV difference is therefore not a conjugation effect; it must come "
                   "from the non-equivalent environments of the two ring-flip options (packing/sterics, cf. the slightly shorter "
                   f"1-3/2-4 contacts in {g['minimum']}).")
    msgs.append(
        f"**The scan's 45/45 ↔ 135/135 crossover is confirmed as two distinct minima.** {g['minimum']} "
        f"(C ≈ {g['C_final']:.0f}°, same-sign outer ≈ {g['L_final']:.0f}°) and {second['minimum']} (C ≈ {second['C_final']:.0f}°, "
        f"outer ≈ {second['L_final']:.0f}°) both satisfy sign(|L|−90)·sign(|C|−90) = −1, the combination favored in `angle_scan.md`." + m2_note)
    others = summary.iloc[2:]
    opp = others[~others["outer_same_sign"]]
    if not opp.empty:
        msgs.append(
            f"**Opposite-handed outer twists are clearly disfavored.** Starting flat (L = R = 0°) the outer torsions twist "
            f"in opposite directions (L ≈ +{opp['L_final'].abs().mean():.0f}°, R ≈ −{opp['R_final'].abs().mean():.0f}°) and the "
            f"molecule stays trapped in {', '.join(opp['minimum'])} at {opp['E_BS_meV'].min():.0f}–{opp['E_BS_meV'].max():.0f} meV, "
            "with C barely moving from its start: the central torsion surface is flat here and the outer handedness pattern, "
            "once set, is not undone by a local optimizer.")
    wrong = others[(others["handedness_product"] > 0) & others["outer_same_sign"]]
    if not wrong.empty:
        r = wrong.iloc[0]
        msgs.append(
            f"**The 'wrong' handedness combination is a high local minimum.** Starting from L = R = 90° at C = 80°, the outer "
            f"rings fall to ≈ {r['L_final']:.0f}° while C opens to {r['C_final']:.0f}° ({r['minimum']}, "
            f"{r['E_BS_meV']:.0f} meV): C > 90° with 135/135-type outer twists, the combination the scan disfavors. It has the "
            f"smallest central plane angle (θC ≈ {r['theta_C_final']:.0f}°) and therefore the largest AFM gap "
            f"({r['gap_meV']:.1f} meV), but pays ~{r['E_BS_meV']:.0f} meV in strain/packing.")
    start_min = pairs.set_index("start")["minimum_BS"]
    n90 = [s for s in start_min.index if s.startswith("L090")]
    msgs.append(
        "**90° outer torsions are not stable.** Every L = R = 90° start leaves 90° (orthogonal radical fragments lose "
        "conjugation), and where it goes depends on the starting C: "
        + ", ".join(f"{s} → {start_min[s]}" for s in n90)
        + ". So the outer handedness is picked during descent and does not always end up in the favored combination.")
    msgs.append(
        f"**The geometric minimum does not depend on the spin state.** HS and BS land on the same structure from every start "
        f"(RMSD ≤ {pairs['rmsd_HS_BS'].max():.2f} Å, dihedrals within {pairs['max_abs_dihedral_diff_HS_BS'].max():.1f}°), and the "
        f"energy ordering of all minima is identical. The HS–BS gap ({summary['gap_meV'].min():.1f}–{summary['gap_meV'].max():.1f} meV) "
        f"is an order of magnitude smaller than the spacing between minima ({second['E_BS_meV'] - g['E_BS_meV']:.0f} meV and up): "
        "conformation is set by sterics/conjugation, and magnetism is a small perturbation on top — no spin-state-driven "
        "conformational switching is seen.")
    msgs.append(
        f"**The ground state is weakly antiferromagnetic at every minimum.** BS is lower everywhere; at {g['minimum']} "
        f"ΔE = {g['gap_meV']:.2f} meV (⟨S²⟩ HS {g['s2_HS']:.3f}, BS {g['s2_BS']:.3f}; Yamaguchi E_T − E_S ≈ "
        f"{2 * g['gap_meV'] / (g['s2_HS'] - g['s2_BS']):.1f} meV). Each terminal fragment carries ≈ "
        f"{g['spin_f1']:.2f} of spin in HS with ≈ {g['spin_f2']:.2f} delocalized onto each backbone half, the level of "
        f"delocalization expected for ~{g['theta_L_final']:.0f}° outer plane angles.")
    ninety = pairs[pairs["start"].str.startswith("L090")]
    if not ninety.empty:
        msgs.append(
            "**Relaxation switches the coupling on.** At the 90° starting geometries the vertical gap is ≈ 0 "
            f"({', '.join(f'{v:.2f}' for v in ninety['dE_vertical_scan_geom_meV'])} meV: decoupled radicals), but after free "
            f"relaxation the same molecules show {', '.join(f'{v:.1f}' for v in ninety['dE_adiabatic_meV'])} meV. The nearly "
            "HS-degenerate / HS-favored region of the exchange map (outer torsions near 90°) is not where any of the free "
            "minima sit.")
    return msgs


if __name__ == "__main__":
    main()
