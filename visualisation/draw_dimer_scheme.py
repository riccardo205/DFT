#!/usr/bin/env python3
"""
Schematics of the four-radical dimers (style of draw_torsion_scheme.py), for one representative dimer
(default run2/conf_7, structural family S01: strong FM coupling between radical A of molecule 1 and
radical C of molecule 2, stacked fluorenyls).

1. <run>_<dimer>/dimer_scheme.png/.pdf  readable 2D drawing, not the real 3D geometry. Every molecule is the RDKit 2D
   depiction of the reference monomer, mapped atom by atom (graph isomorphism) onto the dimer, so the
   radical halves A-D are exactly those of radical_fragments.json. The two molecules are placed so that
   the stacked fluorenyls of the dominant intermolecular pair sit on top of each other.
     halo colour     radical half (A, B: molecule 1; C, D: molecule 2); strong = fluorenyl (radical unit),
                     light = anthryl (linker)
     dots            spin sublattices: sign of the HS (uuuu) Mulliken spin, filled = majority (rho > 0),
                     open = minority (rho < 0), area ~ |rho|. The fluorenyl 5-ring is an odd ring, so the
                     carbon network is not strictly bipartite: the sublattices are taken from the computed
                     spin density, not from a graph 2-colouring.
     gold ring       radical carbon C9
2. <run>_<dimer>/spin_model.png/.pdf    four-spin Heisenberg model: one arrow per radical, grouped by molecule, the six
   couplings J_ij drawn between them (red FM, blue AFM, width ~ sqrt|J|), values of the chosen dimer.
   H = - sum_{i<j} J_ij S_i . S_j, sites 1-4 = A-D.

Settings in the `if __name__ == "__main__":` block.
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dimer_overlap as OV  # noqa: E402
from draw_torsion_scheme import build_adjacency, depict_2d  # noqa: E402

RADICALS = "ABCD"
RAD_COLOR = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eb6834", "D": "#eda100"}
FM_C, AFM_C = "#e34948", "#2a78d6"
BOND_C = "0.3"


def load_dimer(key):
    run, dimer = key.split("/", 1)
    J, E, _ = OV.load_inputs()
    j = J[(J["run"] == run) & (J["dimer"] == dimer)]
    if j.empty:
        raise SystemExit(f"{key} not found in exchange_couplings.csv")
    j = j.iloc[0]
    e = E[(E["run"] == run) & (E["dimer"] == dimer)]
    ref = OV.Reference(OV.REF_XYZ)
    pairs, _, _, geo = OV.analyse_dimer(run, dimer, ref, e)
    Jv = {p: float(j[f"J_{p}_meV"]) for p in OV.PAIRS}
    for r in pairs:
        r["J_meV"] = Jv[r["pair"]]
    return ref, geo, {r["pair"]: r for r in pairs}, Jv


# =========================
# FIGURE 1: 2D DIMER SCHEME
# =========================
def layout_2d(ref, geo, stack_pair):
    """2D coordinates for the heavy atoms of the dimer: reference depiction mapped onto each molecule,
    molecule 1 above molecule 2, the fluorenyls of stack_pair (e.g. 'AC') aligned vertically."""
    ref_atoms = read(OV.REF_XYZ)
    R = depict_2d(ref_atoms, build_adjacency(ref_atoms))
    xy = np.array(list(R.values()))
    _, _, vt = np.linalg.svd(xy - xy.mean(0))
    R = {k: (v - xy.mean(0)) @ vt.T for k, v in R.items()}
    bond = np.median([np.linalg.norm(R[a] - R[b]) for a, b in ref.graph.edges if a in R and b in R])

    pos, z, units, n = geo["pos"], geo["z"], geo["units"], ref.n
    X = {}
    for m in (0, 1):
        mono = Atoms(numbers=z[m * n:(m + 1) * n], positions=pos[m * n:(m + 1) * n])
        mp = ref.map_monomer(mono)
        for r, i in mp.items():
            if r in R:
                X[i + m * n] = R[r].copy()

    p_top, p_bot = stack_pair                      # radical on molecule 1, radical on molecule 2
    other = {"A": "B", "B": "A", "C": "D", "D": "C"}

    def cen(atoms):
        return np.mean([X[i] for i in atoms if i in X], axis=0)

    def flip(mol, axis):
        for i in range(mol * n, (mol + 1) * n):
            if i in X:
                X[i][axis] *= -1

    # molecule 1: stacked radical on the right and at the bottom; molecule 2: stacked radical on the left, top
    if cen(units[p_top]["fluorenyl"])[0] < cen(units[other[p_top]]["fluorenyl"])[0]:
        flip(0, 0)
    if cen(units[p_top]["fluorenyl"])[1] > cen(range(0, n))[1]:
        flip(0, 1)
    if cen(units[p_bot]["fluorenyl"])[0] > cen(units[other[p_bot]]["fluorenyl"])[0]:
        flip(1, 0)
    if cen(units[p_bot]["fluorenyl"])[1] < cen(range(n, 2 * n))[1]:
        flip(1, 1)
    # align the stacked fluorenyls (small slip to the right for readability) and open a vertical gap
    shift = cen(units[p_top]["fluorenyl"]) - cen(units[p_bot]["fluorenyl"]) + np.array([0.35 * bond, 0])
    for i in range(n, 2 * n):
        if i in X:
            X[i] = X[i] + shift
    m1 = np.array([X[i] for i in range(n) if i in X])
    m2_idx = [i for i in range(n, 2 * n) if i in X]
    dy = 0.0
    while True:
        m2 = np.array([X[i] for i in m2_idx]) - [0, dy]
        if np.linalg.norm(m1[:, None] - m2[None], axis=-1).min() > 2.4 * bond:
            break
        dy += 0.1 * bond
    for i in m2_idx:
        X[i] = X[i] - [0, dy]
    return X, bond


def dimer_scheme(ref, geo, pairs, stack_pair, key, out):
    X, bond = layout_2d(ref, geo, stack_pair)
    units, rc, rho, halves = geo["units"], geo["rc"], geo["rho"], geo["halves"]
    heavy = sorted(X)
    g = ref.graph
    n = ref.n
    rmax = np.nanmax(np.abs(rho[heavy]))

    fig = plt.figure(figsize=(12, 9.5))
    ax = fig.add_axes([0.0, 0.07, 1.0, 0.86])
    ax.set_aspect("equal")
    ax.axis("off")

    # stacking region between the two fluorenyls
    pa, pb = stack_pair
    P = np.array([X[i] for i in units[pa]["fluorenyl"] + units[pb]["fluorenyl"]])
    lo, hi = P.min(0) - 0.9 * bond, P.max(0) + 0.9 * bond
    ax.add_patch(FancyBboxPatch(lo, *(hi - lo), boxstyle=f"round,pad=0,rounding_size={bond}",
                                fc="0.93", ec="0.6", lw=1.2, ls="--", zorder=0))

    # halos: radical half colour, strong on the fluorenyl, light on the anthryl
    for lab in RADICALS:
        for i in units[lab]["fluorenyl"]:
            ax.scatter(*X[i], s=950, color=RAD_COLOR[lab], alpha=0.55, lw=0, zorder=1)
        for i in units[lab]["anthryl"]:
            ax.scatter(*X[i], s=950, color=RAD_COLOR[lab], alpha=0.18, lw=0, zorder=1)

    # bonds (within each molecule, reference graph mapped through the coordinates)
    for i in heavy:
        for j in heavy:
            if j > i and (i < n) == (j < n) and np.linalg.norm(geo["pos"][i] - geo["pos"][j]) < 1.75:
                ax.plot(*zip(X[i], X[j]), color=BOND_C, lw=1.6, zorder=2)

    # sublattice dots: sign of the HS spin density
    for i in heavy:
        s = rho[i]
        size = 18 + 260 * abs(s) / rmax
        ax.scatter(*X[i], s=size, color="0.1" if s > 0 else "white", edgecolor="0.1", lw=1.1, zorder=4)

    # radical carbons and labels
    for lab in RADICALS:
        c9 = X[rc[lab]]
        ax.scatter(*c9, s=520, facecolors="none", edgecolors="gold", linewidths=3.2, zorder=5)
        f = np.mean([X[i] for i in units[lab]["fluorenyl"]], axis=0)
        mol_c = np.mean([X[i] for i in (range(0, n) if lab in "AB" else range(n, 2 * n)) if i in X], axis=0)
        d = f - mol_c
        d = d / np.linalg.norm(d)
        lp = f + d * 3.4 * bond + np.array([0, 1.4 * bond if lab in "AB" else -1.4 * bond])
        if lab == pa:
            lp = f + np.array([2.2 * bond, 3.0 * bond])
        if lab == pb:
            lp = f + np.array([2.6 * bond, -3.0 * bond])
        ax.annotate(f"{lab}", xy=c9, xytext=lp, ha="center", va="center", fontsize=24, fontweight="bold",
                    color=RAD_COLOR[lab], zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=RAD_COLOR[lab], lw=2),
                    arrowprops=dict(arrowstyle="-", color=RAD_COLOR[lab], lw=1.5, ls="--", shrinkA=14, shrinkB=8))

    # molecule labels
    for m, (a, b) in enumerate((("A", "B"), ("C", "D"))):
        pts = np.array([X[i] for i in range(m * n, (m + 1) * n) if i in X])
        y0 = pts[:, 1].max() + 1.2 * bond if m == 0 else pts[:, 1].min() - 1.2 * bond
        ax.text(pts[:, 0].mean(), y0, f"molecule {m + 1}  ({a} + {b})", ha="center",
                va="bottom" if m == 0 else "top", fontsize=13, color="0.25")

    # stack annotation
    r = pairs[OV.PAIRS[[OV.PL[p] for p in OV.PAIRS].index("".join(sorted(stack_pair)))]]
    col = FM_C if r["J_meV"] > 0 else AFM_C
    txt = (f"π-stack {pa}–{pb}:  J = {r['J_meV']:+.1f} meV ({'FM' if r['J_meV'] > 0 else 'AFM'})\n"
           f"d⊥ {r['sep_normal_A']:.2f} Å, slip {r['slip_A']:.1f} Å, tilt {r['tilt_deg']:.0f}°, "
           f"φ {r['phi_C2_deg']:.0f}°\nS² = {r['S2_mag']:.4f}")
    ax.annotate(txt, xy=(hi[0], 0.5 * (lo[1] + hi[1])), xytext=(hi[0] + 3.5 * bond, 0.5 * (lo[1] + hi[1]) + 4.2 * bond),
                fontsize=11.5, color=col, ha="left", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.6),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.4, ls="--"))

    allp = np.array(list(X.values()))
    ax.set_xlim(allp[:, 0].min() - 4.0 * bond, allp[:, 0].max() + 5.0 * bond)
    ax.set_ylim(allp[:, 1].min() - 3.0 * bond, allp[:, 1].max() + 3.0 * bond)

    handles = [Patch(facecolor=RAD_COLOR[l], alpha=0.55, label=f"radical {l} (fluorenyl)") for l in RADICALS]
    handles += [Patch(facecolor="0.5", alpha=0.25, label="anthryl linker (light)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="0.1", markeredgecolor="0.1", markersize=10,
                       label="majority sublattice (ρ > 0)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="0.1", markersize=10,
                       label="minority sublattice (ρ < 0)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="gold",
                       markeredgewidth=3, markersize=14, label="radical carbon C9"),
                Patch(facecolor="0.93", edgecolor="0.6", ls="--", label="stacked fluorenyls")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5, frameon=False)
    fig.suptitle(f"Four-radical dimer, schematic 2D layout ({key}); dot area ∝ |HS Mulliken spin|", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


# =========================
# FIGURE 2: FOUR-SPIN MODEL
# =========================
def spin_model(Jv, key, config, out, show_values=True):
    # molecule 1 on top (B left, A right), molecule 2 below shifted right (C under A, D right)
    P = {"B": np.array([0.0, 2.4]), "A": np.array([3.0, 2.4]), "C": np.array([3.0, 0.0]), "D": np.array([6.0, 0.0])}
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_aspect("equal")
    ax.axis("off")
    for m, (a, b) in enumerate((("B", "A"), ("C", "D"))):
        x0, x1 = P[a][0] - 0.75, P[b][0] + 0.75
        y = P[a][1]
        ax.add_patch(FancyBboxPatch((x0, y - 0.5), x1 - x0, 1.0, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc="0.95", ec="0.55", lw=1.3, zorder=0))
        ax.text(x0 + 0.1, y + 0.6 if m == 0 else y - 0.6, f"molecule {m + 1}", fontsize=12, color="0.3", ha="left",
                va="bottom" if m == 0 else "top")
    jmax = max(abs(v) for v in Jv.values() if np.isfinite(v)) if show_values else 1
    curved = {}                                 # optional arc3 curvature per pair, e.g. {"BD": -0.15}
    label_frac = {"AC": 0.75, "BD": 0.25, "AD": 0.25}   # label position along each line (0 = first radical)
    for p in OV.PAIRS:
        a, b = OV.PL[p]
        v = Jv[p]
        col = (FM_C if v > 0 else AFM_C) if show_values and np.isfinite(v) else "0.35"
        lw = 1.2 + 7 * np.sqrt(abs(v) / jmax) if show_values and np.isfinite(v) else 2.0
        rad = curved.get(a + b, 0.0)
        ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                     color=col, lw=lw, alpha=0.85, zorder=1, shrinkA=22, shrinkB=22))
        f = label_frac.get(a + b, 0.5)
        d = P[b] - P[a]
        ctrl = P[a] + 0.5 * d + rad * np.array([d[1], -d[0]])     # control point of matplotlib's arc3 Bezier
        mid = (1 - f) ** 2 * P[a] + 2 * f * (1 - f) * ctrl + f ** 2 * P[b]
        lab = f"$J_{{{p}}}$ ({a}{b})" + (f"\n{v:+.2f} meV" if show_values and np.isfinite(v) else "")
        ax.text(*mid, lab, ha="center", va="center", fontsize=10.5, color=col if show_values else "0.15", zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.0))
    for k, lab in enumerate(RADICALS):
        x, y = P[lab]
        up = config[k] == "u"
        ax.scatter(x, y, s=1500, color=RAD_COLOR[lab], alpha=0.3, lw=0, zorder=2)
        ax.add_patch(FancyArrowPatch((x, y - 0.38 if up else y + 0.38), (x, y + 0.38 if up else y - 0.38),
                                     arrowstyle="-|>", mutation_scale=28, color=RAD_COLOR[lab], lw=3.2, zorder=4))
        ax.text(x - 0.32, y, lab, fontsize=18, fontweight="bold", color=RAD_COLOR[lab], ha="right", va="center", zorder=5)
        ax.text(x + 0.3, y - 0.3, str(k + 1), fontsize=11, color="0.35", ha="left", va="center", zorder=5)
    ax.set_xlim(-1.3, 7.3)
    ax.set_ylim(-1.1, 3.5)
    sub = f"{key}, spins drawn in configuration {config}" if show_values else f"spins drawn in configuration {config}"
    ax.set_title("Four-spin model  H = −Σ$_{i<j}$ J$_{ij}$ S$_i$·S$_j$   (J > 0 FM, red;  J < 0 AFM, blue;  "
                 "width ∝ √|J|)\n" + sub, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(key, stack_pair, config, out_dir, show_values):
    out = out_dir / key.replace("/", "_")
    ref, geo, pairs, Jv = load_dimer(key)
    dimer_scheme(ref, geo, pairs, stack_pair, key, out / "dimer_scheme.png")
    spin_model(Jv, key, config, out / "spin_model.png", show_values)
    print(f"wrote {out / 'dimer_scheme.png'} and {out / 'spin_model.png'} (+ .pdf)")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    # (dimer, radical on molecule 1 + radical on molecule 2 drawn on top of each other)
    # S01 family: strong FM A-C stack
    DIMERS = [("run2/conf_7", "AC"),
              ("run2/conf_28", "AC")]
    CONFIG = "uuuu"                   # spin arrows in spin_model.png (u/d for A B C D)
    SHOW_VALUES = True                # False: generic J_ij labels only (nomenclature figure)
    OUT_DIR = OV.CSV_ROOT / "schemes"  # one subfolder per dimer, e.g. schemes/run2_conf_7/

    for key, stack_pair in DIMERS:
        main(key, stack_pair, CONFIG, OUT_DIR, SHOW_VALUES)
