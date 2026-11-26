#!/usr/bin/env python3
"""
Inspect ONE dimer configuration: the two fluorenyls joined by the strongest intermolecular J.

1. read J_12..J_34 of the dimer (exchange_couplings.csv, written by Read_output_bs_orca.py);
2. dominant pair = largest |J| among J_13, J_14, J_23, J_24 (radicals A, B on molecule 1; C, D on molecule 2);
3. fluorenyl of each radical = reference atoms 28-40 / 41-53 mapped onto the dimer (radical_fragments.json
   gives the halves A-D, graph isomorphism gives the atoms);
4. frame of the first fluorenyl P: origin = centroid, z = plane normal (sign: towards the second fluorenyl),
   y = centroid -> C9 projected in the plane, x = y x z;
5. rotate the whole dimer into this frame and report
   J, RMS planarity of both fluorenyls, d_perp (z of the second centroid), tilt between the planes,
   in-plane rotation phi (angle from y to the second C2 axis centroid -> C9, counter-clockwise seen from +z),
   lateral displacement (dx, dy) of the second centroid and the slip sqrt(dx^2 + dy^2).
Plots: top_view.png (looking down z), side_view.png (looking along SIDE_AXIS) and both stacked
vertically in top_side_view.png; heavy atoms drawn as
Mulliken-spin dots of SPIN_CONFIG (black rho > 0, white rho < 0, area ~ |rho|).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from networkx.algorithms.isomorphism import GraphMatcher

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "orca"))
from prepare_dimers_broken_symmetry import bond_graph, mulliken_spins  # noqa: E402

REF_XYZ = REPO / "runs/orca/short-2OS/angle_scans/angle_scan_relaxation/L135_C120_R090/high_spin.xyz"
FLUORENYL = {28: range(28, 41), 41: range(41, 54)}      # reference numbering, keyed by the radical carbon C9
RADICALS = "ABCD"
INTER = ["13", "14", "23", "24"]
COL = {"P": "#2a78d6", "Q": "#eb6834"}                   # first / second interacting fluorenyl
MOL_COL = ["#7fa8d9", "#f0a982"]                         # light colours of molecule 1 / 2 (context)


def unit(v):
    return v / np.linalg.norm(v)


def plane(x):
    c = x.mean(0)
    n = np.linalg.svd(x - c)[2][-1]
    return c, n, float(np.sqrt((((x - c) @ n) ** 2).mean()))


def fluorenyls(atoms, halves):
    """{radical: (fluorenyl atom indices, C9 index)} via the reference monomer mapped onto each molecule."""
    ref = read(REF_XYZ)
    g_ref, n = bond_graph(ref), len(ref)
    out = {}
    for m in (0, 1):
        gm = GraphMatcher(g_ref, bond_graph(atoms[m * n:(m + 1) * n]), node_match=lambda a, b: a["z"] == b["z"])
        mp = {r: i + m * n for r, i in next(gm.isomorphisms_iter()).items()}
        for lab in RADICALS[2 * m:2 * m + 2]:
            rc = next(r for r in FLUORENYL if mp[r] in halves[lab])
            out[lab] = ([mp[r] for r in FLUORENYL[rc]], mp[rc])
    return out


def analyse(key, dimers_root, csv_root, spin_config):
    run, dimer = key.split("/", 1)
    d = dimers_root / f"{run}_broken_symmetry" / "relaxed" / dimer
    J = pd.read_csv(csv_root / f"{run}_broken_symmetry" / "exchange_couplings.csv")
    J = J[(J["stage"] == "relaxed") & (J["dimer"] == dimer)].iloc[0]
    pair = max(INTER, key=lambda p: abs(J[f"J_{p}_meV"]))
    a, b = RADICALS[int(pair[0]) - 1], RADICALS[int(pair[1]) - 1]

    atoms = read(d / "uuuu" / "geometry.xyz")
    halves = {k: set(v) for k, v in json.loads((d / "radical_fragments.json").read_text())["fragments"].items()}
    fl = fluorenyls(atoms, halves)
    pos = atoms.positions

    # reference frame of fluorenyl P
    cP, nP, rmsP = plane(pos[fl[a][0]])
    cQ, nQ, rmsQ = plane(pos[fl[b][0]])
    z = nP if (cQ - cP) @ nP >= 0 else -nP
    y = unit((pos[fl[a][1]] - cP) - ((pos[fl[a][1]] - cP) @ z) * z)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])                            # rows = new axes
    X = (pos - cP) @ R.T                                # whole dimer in the new frame

    cQn = (cQ - cP) @ R.T
    vQ = (pos[fl[b][1]] - cQ) @ R.T                     # second C2 axis (centroid -> C9) in the new frame
    desc = {"dimer": key, "pair": f"J_{pair} ({a}{b})", "J_meV": float(J[f"J_{pair}_meV"]),
            "rms_P_A": rmsP, "rms_Q_A": rmsQ, "d_perp_A": cQn[2],
            "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(nP @ nQ))))),
            "phi_deg": float(np.degrees(np.arctan2(-vQ[0], vQ[1]))),     # 0 = parallel to y, ccw from +z
            "dx_A": cQn[0], "dy_A": cQn[1], "slip_A": float(np.hypot(cQn[0], cQn[1]))}
    sp = mulliken_spins(d / spin_config / f"{spin_config}.out")    # last Mulliken block, 0-based indices
    rho = np.array([sp.get(i, 0.0) for i in range(len(atoms))])
    return desc, X, atoms.numbers, fl, a, b, len(atoms) // 2, rho


def draw(ax, X, numbers, fl, a, b, n_mono, i, j, title, rho):
    """2D projection on axes (i, j): bonds (context molecules light, the two fluorenyls bold) and heavy atoms
    as Mulliken-spin dots (black rho > 0, white rho < 0, area ~ |rho|). No other symbols."""
    heavy = np.where(numbers > 1)[0]
    target = {k: lab for lab in (a, b) for k in fl[lab][0]}
    for p in range(len(heavy)):
        for q in range(p + 1, len(heavy)):
            u, v = heavy[p], heavy[q]
            if (u < n_mono) != (v < n_mono) or np.linalg.norm(X[u] - X[v]) > 1.75:
                continue
            if u in target and v in target:
                col = COL["P"] if target[u] == a else COL["Q"]
                ax.plot(*X[[u, v]][:, [i, j]].T, color=col, lw=2.6, zorder=3, solid_capstyle="round")
            else:
                ax.plot(*X[[u, v]][:, [i, j]].T, color=MOL_COL[int(u >= n_mono)], lw=1.2, alpha=0.45, zorder=1)
    smax = np.abs(rho[heavy]).max() or 1.0
    for on_target in (False, True):
        idx = np.array([k for k in heavy if (k in target) == on_target])
        size = (10 + 150 * np.abs(rho[idx]) / smax) * (1.0 if on_target else 0.7)
        ax.scatter(*X[idx][:, [i, j]].T, s=size, c=np.where(rho[idx] > 0, "k", "white"), edgecolors="k",
                   linewidths=0.8 if on_target else 0.5, alpha=1.0 if on_target else 0.45,
                   zorder=5 if on_target else 2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def top_view(ax, X, numbers, fl, a, b, n_mono, rho, desc, head):
    draw(ax, X, numbers, fl, a, b, n_mono, 0, 1,
         f"{head}\nslip {desc['slip_A']:.2f} Å (Δx {desc['dx_A']:+.2f}, Δy {desc['dy_A']:+.2f})   "
         f"φ {desc['phi_deg']:+.0f}°   RMS {desc['rms_P_A']:.3f} / {desc['rms_Q_A']:.3f} Å", rho)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel(f"y (Å), {a}: centroid → C9")


def side_view(ax, X, numbers, fl, a, b, n_mono, rho, desc, head, h, side_axis):
    draw(ax, X, numbers, fl, a, b, n_mono, h, 2,
         f"{head}\nd⊥ {desc['d_perp_A']:.2f} Å   tilt {desc['tilt_deg']:.1f}°", rho)
    ax.set_xlabel(f"{side_axis} (Å)")
    ax.set_ylabel(f"z (Å), normal of {a}")


def main(key, out_dir, side_axis, spin_config):
    desc, X, numbers, fl, a, b, n_mono, rho = analyse(key, DIMERS_ROOT, CSV_ROOT, spin_config)
    out = out_dir / key.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([desc]).to_csv(out / "descriptors.csv", index=False)
    for k, v in desc.items():
        print(f"  {k:10s} {v:.3f}" if isinstance(v, float) else f"  {k:10s} {v}")

    head = f"{key}   {desc['pair']} = {desc['J_meV']:+.2f} meV"
    args = (X, numbers, fl, a, b, n_mono, rho, desc, head)
    h = 0 if side_axis == "x" else 1
    heavy = X[numbers > 1]
    top_span = np.ptp(heavy[:, [0, 1]], axis=0) + 2.0           # figure sizes follow the data aspect ratios
    side_span = np.ptp(heavy[:, [h, 2]], axis=0) + 2.0
    top_h = 7 * top_span[1] / top_span[0]
    side_h = 7 * side_span[1] / side_span[0]

    fig, ax = plt.subplots(figsize=(7, top_h + 0.9))
    top_view(ax, *args)
    fig.tight_layout()
    fig.savefig(out / "top_view.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, side_h + 0.9))
    side_view(ax, *args, h, side_axis)
    fig.tight_layout()
    fig.savefig(out / "side_view.png", dpi=200)
    plt.close(fig)

    # both views stacked vertically (same x scale)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, top_h + side_h + 1.8),
                                   gridspec_kw={"height_ratios": [top_h, side_h]})
    top_view(ax1, *args)
    side_view(ax2, *args, h, side_axis)
    fig.tight_layout()
    fig.savefig(out / "top_side_view.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    KEYS = ["run2/conf_28"]                               # one or more "<run>/<dimer>"
    SIDE_AXIS = "x"                                      # horizontal axis of the side view: "x" or "y"
    SPIN_CONFIG = "uuuu"                                 # BS configuration whose Mulliken spins are drawn
    DIMERS_ROOT = REPO / "runs" / "orca" / "short-2OS" / "dimers_run2"
    CSV_ROOT = REPO / "visualisation" / "output" / "orca" / "dimers_run2"
    OUT_DIR = CSV_ROOT / "dimer_overlap"

    for k in KEYS:
        main(k, OUT_DIR, SIDE_AXIS, SPIN_CONFIG)
