#!/usr/bin/env python3
"""
Fragment-overlap plots for ONE dimer configuration (same definitions and code as analyze_dimer_overlap.py).

Set KEYS (and optionally PAIRS_SEL, ALL_PAIRS, OUT_ROOT) in the `if __name__ == "__main__":` block, then
    python3 plot_dimer_overlap_conf.py

Figures (visualisation/output/orca/dimers_run2/overlap/conf/<run>_<dimer>/):
    overview.png      whole dimer seen along its two shortest principal axes, halves A-D coloured, fluorenyl
                      rings filled, C9 stars, J_ij drawn between the radical carbons (width ~ |J|, red FM / blue AFM)
    projections.png   face view on the mean plane and edge view of every selected radical pair (default: the 4
                      intermolecular pairs, ordered by |J|), titles with J, d_perp, slip, tilt, phi, overlap, |S|
    pair_summary.png  J_ij, S_ij^2 and the main geometric descriptors for the 6 pairs side by side
    spins.png         Mulliken spin of fluorenyl / anthryl / H of every radical in every finished BS configuration
CSV: pairs.csv, internal_angles.csv, spins.csv (same columns as the overview script).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dimer_overlap as OV  # noqa: E402
from analyze_dimer_overlap import COLORS, INK, INK_2, INK_3, PL, PAIRS, INTER, RADICALS, style  # noqa: E402

RAD_COLOR = {"A": COLORS[0], "B": COLORS[2], "C": COLORS[1], "D": COLORS[3]}


def overview(geo, pairs, key, path):
    pos, z, units, rc = geo["pos"], geo["z"], geo["units"], geo["rc"]
    c = pos.mean(0)
    _, _, Vt = np.linalg.svd(pos - c)
    views = [((0, 1), "top view (along the shortest principal axis)"), ((0, 2), "side view (along the middle axis)")]
    J = {r["pair"]: r["J_meV"] for r in pairs}
    jmax = max(1.0, np.nanmax(np.abs(list(J.values()))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, ((i, j), ttl) in zip(axes, views):
        P = (pos - c) @ Vt[[i, j]].T
        for lab in RADICALS:
            u = units[lab]
            col = RAD_COLOR[lab]
            for r in u["rings_fl"]:
                ax.fill(*P[r].T, color=col, alpha=0.35, lw=0)
            for r in u["rings_fl"] + u["rings_an"]:
                ax.plot(*np.vstack([P[r], P[r][:1]]).T, color=col, lw=0.9)
            ax.scatter(*P[u["H"]].T, s=4, color=col, alpha=0.4)
            ax.scatter(*P[rc[lab]], marker="*", s=200, color=col, edgecolor=INK, lw=0.7, zorder=6)
            ax.annotate(lab, P[rc[lab]], xytext=(6, 6), textcoords="offset points", fontsize=12, weight="bold", color=col)
        for p in PAIRS:
            a, b = PL[p]
            if not np.isfinite(J[p]):
                continue
            xy = np.vstack([P[rc[a]], P[rc[b]]])
            ax.plot(*xy.T, color=OV.FM_C if J[p] > 0 else OV.AFM_C, lw=0.6 + 5 * abs(J[p]) / jmax, alpha=0.75, zorder=5,
                    ls="-" if p in INTER else "--")
            ax.text(*xy.mean(0), f"{J[p]:+.2f}", fontsize=7.5, color=INK, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("Å")
        ax.set_ylabel("Å")
        style(ax)
    fig.suptitle(f"{key}: radical halves (fluorenyl filled, anthryl outline, star = C9); lines = J_ij in meV "
                 "(red FM, blue AFM, dashed intramolecular, width ∝ |J|)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def projections(geo, pairs, sel, key, path):
    rows = {r["pair"]: r for r in pairs}
    fig, axes = plt.subplots(2, len(sel), figsize=(4.0 * len(sel), 7.2), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    for k, p in enumerate(sel):
        r = rows[p]
        warn = f"\n(tilt > {OV.TILT_WARN:.0f}°: φ and slip split indicative)" if r["tilt_deg"] > OV.TILT_WARN else ""
        ttl = (f"J_{p} ({PL[p]}) = {r['J_meV']:+.2f} meV  |S| {r['S_mag']:.3f}\n"
               f"d⊥ {r['sep_normal_A']:.2f} Å  slip {r['slip_A']:.2f} Å (v {r['slip_short_A']:+.2f}, u {r['slip_long_A']:.2f})\n"
               f"tilt {r['tilt_deg']:.0f}°  φ {r['phi_C2_deg']:.0f}°  overlap {r['overlap_area_A2']:.1f} Å²{warn}")
        OV.projection_panel(axes[0, k], axes[1, k], geo, p, ttl, colors=(RAD_COLOR[PL[p][0]], RAD_COLOR[PL[p][1]]))
    fig.suptitle(f"{key}: radical pairs projected on their mean plane (frame of the first radical). Filled = fluorenyl, "
                 "dotted = anthryl, star = C9, arrow = C2 axis,\ndots = HS Mulliken spin (black > 0, white < 0, size ∝ |ρ|)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pair_summary(pairs, key, path):
    T = pd.DataFrame(pairs)
    lab = [f"{r['radicals']}\n{r['relation']}" for _, r in T.iterrows()]
    x = np.arange(len(T))
    panels = [("J_meV", "J_ij (meV, > 0 FM)"), ("S2_mag", "magnetic-orbital overlap S²"),
              ("sep_normal_A", "separation along mean normal (Å)"), ("slip_A", "lateral slip (Å)"),
              ("phi_C2_deg", "in-plane rotation φ (°)"), ("tilt_deg", "tilt (°)"),
              ("overlap_area_A2", "projected ring overlap (Å²)"), (f"n_CC_{OV.CONTACT_CUT:.1f}", f"C…C < {OV.CONTACT_CUT} Å")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, (c, t) in zip(axes.ravel(), panels):
        v = T[c].astype(float)
        col = [OV.FM_C if j > 0 else OV.AFM_C for j in T["J_meV"]] if c == "J_meV" else \
            [INK_2 if rel == "inter" else INK_3 for rel in T["relation"]]
        ax.bar(x, v.fillna(0), color=col)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, vi, f"{vi:.3g}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=7, color=INK)
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_title(t, fontsize=9)
        style(ax)
    fig.suptitle(f"{key}: exchange, magnetic-orbital overlap and fluorenyl–fluorenyl geometry of the 6 radical pairs "
                 "(intra pairs: contacts not defined, bonded halves)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def spin_plot(spins, key, path):
    S = pd.DataFrame(spins)
    configs = [c for c in OV.CONFIGS if c in S["config"].unique()]
    fig, axes = plt.subplots(1, len(RADICALS), figsize=(14, 3.8), sharey=True)
    for ax, lab in zip(axes, RADICALS):
        s = S[S["radical"] == lab].set_index("config").reindex(configs)
        x = np.arange(len(configs))
        ax.bar(x - 0.27, s["spin_fluorenyl"], 0.27, color=RAD_COLOR[lab], label="fluorenyl")
        ax.bar(x, s["spin_anthryl"], 0.27, color=RAD_COLOR[lab], alpha=0.45, label="anthryl")
        ax.bar(x + 0.27, s["spin_H"], 0.27, color=INK_3, label="H")
        ax.scatter(x, s["spin_half"], color=INK, marker="_", s=200, zorder=4, label="half total")
        for xi, ok in zip(x, s["used_in_fit"]):
            if not ok:
                ax.text(xi, 1.25, "excl.", ha="center", fontsize=6.5, color=OV.FM_C)
        ax.axhline(0, color=INK_3, lw=0.8)
        for y in (-OV.XC.MIN_LOCAL_MOMENT, OV.XC.MIN_LOCAL_MOMENT):
            ax.axhline(y, color=INK_3, lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=90, fontsize=8)
        ax.set_title(f"radical {lab}", fontsize=9, color=RAD_COLOR[lab])
        style(ax)
    axes[0].set_ylabel("Mulliken spin")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{key}: spin on each unit in every finished BS configuration (dotted ±{OV.XC.MIN_LOCAL_MOMENT}: "
                 "local-moment threshold; excl. = not used in the J fit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(key, pairs_sel=None, all_pairs=False, out_root=None):
    """key = "<run>/<dimer>" (e.g. run2/conf_7, run4/free/docked_1); pairs_sel = pair labels for the
    projection figure (None = the intermolecular pairs, or all 6 with all_pairs, ordered by |J|)."""
    run, dimer = key.split("/", 1)

    J, E, F = OV.load_inputs()
    j = J[(J["run"] == run) & (J["dimer"] == dimer)]
    if j.empty:
        raise SystemExit(f"{key} not found in the exchange_couplings.csv of {OV.RUNS}")
    j = j.iloc[0]
    e = E[(E["run"] == run) & (E["dimer"] == dimer)]
    pairs, internal, spins, geo = OV.analyse_dimer(run, dimer, OV.Reference(OV.REF_XYZ), e)
    for r in pairs:
        r["J_meV"] = j[f"J_{r['pair']}_meV"]

    out = Path(out_root or OV.OUT / "conf") / key.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(internal).to_csv(out / "internal_angles.csv", index=False)
    pd.DataFrame(spins).to_csv(out / "spins.csv", index=False)

    if pairs_sel:
        unknown = set(pairs_sel) - set(PAIRS)
        if unknown:
            raise SystemExit(f"unknown pairs {sorted(unknown)}; use {PAIRS}")
        sel = list(pairs_sel)
    else:
        cand = PAIRS if all_pairs else INTER
        sel = sorted(cand, key=lambda p: -abs(j[f"J_{p}_meV"]) if np.isfinite(j[f"J_{p}_meV"]) else 0)

    overview(geo, pairs, key, out / "overview.png")
    projections(geo, pairs, sel, key, out / "projections.png")
    pair_summary(pairs, key, out / "pair_summary.png")
    spin_plot(spins, key, out / "spins.png")

    fam = F[F["key"] == key]
    print(f"{key}: fit {j['fit']}" + (f", {fam.iloc[0]['mag_family']}" if len(fam) else ""))
    cols = ["radicals", "J_meV", "S2_mag", "sep_normal_A", "slip_A", "slip_short_A", "slip_long_A", "tilt_deg",
            "phi_C2_deg", "overlap_area_A2", "contact_half_A", "contact_half_atoms"]
    print(pd.DataFrame(pairs)[cols].round({"S2_mag": 5}).round(3).to_string(index=False))
    print(pd.DataFrame(internal).round(1).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    # dimers to plot, "<run>/<dimer>" (run4: "<run>/<set>/<dimer>", e.g. "run4/free/docked_1")
    KEYS = ["run2/conf_27"]
    # pair labels for projections.png ("12" AB, "13" AC, "14" AD, "23" BC, "24" BD, "34" CD);
    # None = the 4 intermolecular pairs ordered by |J|
    PAIRS_SEL = None
    ALL_PAIRS = False          # with PAIRS_SEL = None: also project the intramolecular pairs AB, CD
    # output folder (one subfolder per dimer); None = visualisation/output/orca/dimers_run2/overlap/conf
    # input paths (dimer trees, CSVs, reference monomer) are set in analyze_dimer_overlap.py USER SETTINGS
    OUT_ROOT = None

    for key in KEYS:
        main(key, PAIRS_SEL, ALL_PAIRS, OUT_ROOT)
