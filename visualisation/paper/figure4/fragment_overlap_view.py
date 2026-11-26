#!/usr/bin/env python3
"""
Top view of the overlap between the two fluorenyls joined by the strongest intermolecular coupling (as
visualisation/dimer_overlap/inspect_dimer.py top_view.png), redrawn in the style of the other paper figures:
carbons only, thin black bonds, atoms red / blue by the sign of their Mulliken spin in the broken-symmetry ground
state (lowest-energy BS determinant, or SPIN_CONFIG), spin arrows at the radical carbons C9.

  main fragment   the two overlapping fluorenyls (dominant J pair): full colour
  rest            the other atoms / bonds of both molecules: same colours, transparent (CONTEXT_ALPHA)
Frame (from inspect_dimer.analyse): origin = centroid of the first fluorenyl, y = its centroid -> C9, z = its
normal towards the second fluorenyl; we look down z (second fluorenyl in front).
The view is cut at Y_MAX (Å) on top and fitted around the main fragment elsewhere (PAD), so the overlap is the
focus; the figure height follows from WIDTH_CM and the window aspect (1 Å = 1 Å).

Output: OUT_DIR/<run>_<dimer>_overlap.pdf / .png (Arial, transparent). Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "visualisation" / "dimer_overlap"))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from inspect_dimer import analyse  # noqa: E402


def ground_state(bs_csv, dimer):
    b = pd.read_csv(bs_csv)
    b = b[(b["stage"] == "relaxed") & (b["status"] == "done") & (b["dimer"] == dimer)]
    return b.loc[b["E_Eh"].idxmin(), "config"]


def scene(key, dimers_root, csv_root, spin_config, invert_spins, y_max, pad, st, window_a=None):
    """Dimer in the inspect_dimer frame, spins of the BS ground state (or spin_config) and the drawing window
    (x0, x1, y0, y1) in Å. Shared with fragment_orbital_view.py."""
    run, dimer = key.split("/", 1)
    cfg = spin_config or ground_state(csv_root / f"{run}_broken_symmetry" / "bs_energies.csv", dimer)
    desc, X, numbers, fl, a, b, n_mono, rho = analyse(key, dimers_root, csv_root, cfg)
    if invert_spins:
        rho = -rho
    main_atoms = set(fl[a][0]) | set(fl[b][0])
    radicals = {lab: fl[lab][1] for lab in fl}

    # window: top at y_max, the rest fitted to the main fragment
    M = X[sorted(main_atoms)]
    x0, x1 = M[:, 0].min() - pad, M[:, 0].max() + pad
    if isinstance(y_max, dict):              # per-dimer value, key "default" for the others
        y_max = y_max.get(key, y_max.get("default"))
    y0 = M[:, 1].min() - pad
    y1 = M[:, 1].max() + pad if y_max is None else y_max     # None: fitted to the main fragment as well
    for lab in (a, b):                                        # keep the main radical arrows inside
        r = radicals[lab]
        tip = X[r, 1] + (st["arrow_len"] if rho[r] > 0 else -st["arrow_len"])
        y0, y1 = min(y0, tip - 0.2), max(y1, tip + 0.2)
    print(f"  {key}: own window {x1 - x0:.2f} x {y1 - y0:.2f} Å")
    if window_a:                             # same window size for every dimer: same box and scale in all panels
        w_a, h_a = window_a
        if w_a < x1 - x0 or h_a < y1 - y0:
            print(f"  [!] WINDOW_A {w_a} x {h_a} Å smaller than needed: the main fragment may be cut")
        xc = 0.5 * (x0 + x1)                 # centred on the main fragment, bottom kept at the fragment
        x0, x1, y1 = xc - w_a / 2, xc + w_a / 2, y0 + h_a
    return {"key": key, "cfg": cfg, "desc": desc, "X": X, "numbers": numbers, "fl": fl, "a": a, "b": b,
            "n_mono": n_mono, "rho": rho, "main_atoms": main_atoms, "radicals": radicals,
            "window": (x0, x1, y0, y1)}


def new_axes(window, width_cm):
    x0, x1, y0, y1 = window
    h_cm = width_cm * (y1 - y0) / (x1 - x0)
    fig, ax = V.new_figure((width_cm, h_cm))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return fig, ax, h_cm


def draw_structure(ax, sc, st, context_alpha, arrows_context, z_base=2.0, arrow_outline=None):
    """Bonds (black), atoms (red / blue by spin sign), spin arrows at the main radicals; atoms outside the main
    fragment transparent. Depth order by z (second fluorenyl in front); everything at zorder >= z_base.
    arrow_outline: width (pt) of a white contour around the spin arrows (None: none)."""
    X, numbers, rho, n_mono = sc["X"], sc["numbers"], sc["rho"], sc["n_mono"]
    main_atoms, a, b = sc["main_atoms"], sc["a"], sc["b"]
    heavy = np.where(numbers > 1)[0]
    z0, zr = X[heavy, 2].min(), (np.ptp(X[heavy, 2]) or 1.0)
    depth = lambda z: z_base + 2 * (z - z0) / zr
    rgba = lambda col, on: (*to_rgb(col), 1.0 if on else context_alpha)
    for p in range(len(heavy)):                               # bonds
        for q in range(p + 1, len(heavy)):
            u, v = heavy[p], heavy[q]
            if (u < n_mono) != (v < n_mono) or np.linalg.norm(X[u] - X[v]) > 1.75:
                continue
            on = u in main_atoms and v in main_atoms
            ax.plot(*X[[u, v]][:, :2].T, color=rgba("black", on), lw=st["bond_lw"], solid_capstyle="round",
                    zorder=depth(min(X[u, 2], X[v, 2])) - 1e-3)
    for k in heavy:                                           # atoms
        on = k in main_atoms
        col = st["col_a"] if rho[k] >= 0 else st["col_b"]
        ax.scatter(*X[k, :2], s=st["atom_size"], color=rgba(col, on), edgecolor=rgba("black", on),
                   lw=st["atom_edge_lw"], zorder=depth(X[k, 2]))
    for lab, r in sc["radicals"].items():                     # spin arrows at the radical carbons
        on = lab in (a, b)
        if not on and not arrows_context:
            continue
        col = st["col_a"] if rho[r] >= 0 else st["col_b"]
        tip = X[r, :2] + [0, st["arrow_len"] if rho[r] >= 0 else -st["arrow_len"]]
        arr = ax.add_patch(FancyArrowPatch(X[r, :2], tip, arrowstyle="-|>", mutation_scale=st["arrow_head"],
                                           lw=st["arrow_lw"], color=rgba(col, on), shrinkA=0, shrinkB=0,
                                           zorder=z_base + 8))
        if arrow_outline:                    # white contour (stroke drawn under the arrow)
            import matplotlib.patheffects as pe
            arr.set_path_effects([pe.Stroke(linewidth=st["arrow_lw"] + 2 * arrow_outline, foreground="white",
                                            joinstyle="round"),
                                  pe.Normal()])


def main(key, dimers_root, csv_root, out_dir, spin_config, invert_spins, y_max, pad, width_cm, transparent, st,
         context_alpha, arrows_context, window_a=None):
    sc = scene(key, dimers_root, csv_root, spin_config, invert_spins, y_max, pad, st, window_a)
    fig, ax, h_cm = new_axes(sc["window"], width_cm)
    draw_structure(ax, sc, st, context_alpha, arrows_context)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{key.replace('/', '_')}_overlap"
    V.save(fig, stem, transparent)
    desc, (x0, x1, y0, y1) = sc["desc"], sc["window"]
    print(f"{key}: spins of {sc['cfg']}{' (inverted)' if invert_spins else ''}; main fragment {desc['pair']} "
          f"({desc['J_meV']:+.2f} meV), slip {desc['slip_A']:.2f} Å, d_perp {desc['d_perp_A']:.2f} Å; "
          f"window x {x0:.1f}..{x1:.1f}, y {y0:.1f}..{y1:.1f} Å; {width_cm:.2f} x {h_cm:.2f} cm")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    KEYS = ["run2/conf_28", "run2/conf_7", "run2/conf_11"]       # one figure each
    DIMERS_ROOT = REPO / "runs" / "orca" / "short-2OS" / "dimers_run2"
    CSV_ROOT = REPO / "visualisation" / "output" / "orca" / "dimers_run2"
    OUT_DIR = HERE / "outputs"

    SPIN_CONFIG = None                          # BS determinant for the colours; None = lowest-energy one
    INVERT_SPINS = True                         # flip all spins (as the stacked view: dudu drawn as udud)

    # Å, top of the window (y = 0 at the first fluorenyl centroid); None = fitted to the main fragment (+PAD).
    # A dict sets it per dimer ({"run2/conf_11": None, "default": 4.0}).
    Y_MAX = {"run2/conf_11": None, "default": 4.0}
    WINDOW_A = (9.0, 8.2)                       # (width, height) Å of the window, the same for all dimers, so every
                                                # panel has the same box and scale (None: own window per dimer, from
                                                # Y_MAX / PAD); bottom at the fragment, centred in x
    PAD = 0.6                                   # Å around the main fragment (left, right, bottom)
    WIDTH_CM = 4.0                              # figure width; height from the window aspect
    TRANSPARENT = True
    CONTEXT_ALPHA = 0.25                        # opacity of the atoms / bonds outside the main fragment
    ARROWS_CONTEXT = False                      # also draw (transparent) arrows at the other two radicals
    STYLE = {"atom_size": 9.0, "atom_edge_lw": 0.2, "bond_lw": 0.6,
             "arrow_len": 1.0, "arrow_lw": 0.9, "arrow_head": 4.5,
             "col_a": "#C0392B", "col_b": "#1F4E9A"}   # spin up / down (as the other figures)

    for k in KEYS:
        main(k, DIMERS_ROOT, CSV_ROOT, OUT_DIR, SPIN_CONFIG, INVERT_SPINS, Y_MAX, PAD, WIDTH_CM, TRANSPARENT, STYLE,
             CONTEXT_ALPHA, ARROWS_CONTEXT, WINDOW_A)
