#!/usr/bin/env python3
"""
2D variant of spin_model_scheme.py: same four-spin model of two stacked molecules, but each molecule is a flat
2D ellipse (no thickness, no perspective). Spins at the two ends of each molecule (red up, blue down), all six
exchange couplings as curved double arrows with their labels (J12, J34 intramolecular, J13, J14, J23, J24
intermolecular).

Output: OUT_DIR/spin_model_scheme_2d.pdf / .png at exactly SIZE_CM (Arial, transparent). Settings in the
__main__ block.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.spin_model_scheme import arc_point  # noqa: E402


def main(out_dir, size_cm, font, label_size, transparent, ell_w, ell_h, site_frac, sep, shift_x, spins, arrow_len,
         arrow_lw, arrow_head, col_up, col_down, mol_face, mol_edge, mol_lw, j_col, j_lw, j_head, j_style, j_rad,
         label_t, label_out, label_nudge, pad):
    V.set_fonts(font)
    centres = {1: np.array([0.0, 0.0]), 2: np.array([shift_x, sep])}           # lower, upper molecule
    dx = site_frac * ell_w / 2
    site = {1: centres[1] - [dx, 0], 2: centres[1] + [dx, 0], 3: centres[2] - [dx, 0], 4: centres[2] + [dx, 0]}

    fig, ax = V.new_figure(size_cm)
    for k in (1, 2):                                                           # molecules: flat 2D ellipses
        ax.add_patch(Ellipse(centres[k], ell_w, ell_h, facecolor=mol_face[k - 1], edgecolor=mol_edge[k - 1],
                             lw=mol_lw, zorder=1))

    pts = list(site.values())
    for p, (a, b) in {"12": (1, 2), "34": (3, 4), "13": (1, 3), "24": (2, 4), "14": (1, 4), "23": (2, 3)}.items():
        intra = p in ("12", "34")
        col = j_col["intra" if intra else "inter"]
        ax.add_patch(FancyArrowPatch(site[a], site[b], arrowstyle=j_style, mutation_scale=j_head, lw=j_lw, color=col,
                                     connectionstyle=f"arc3,rad={j_rad[p]}", shrinkA=3, shrinkB=3,
                                     zorder=3 if intra else 2, capstyle="round"))
        m, out = arc_point(site[a], site[b], j_rad[p], label_t.get(p, 0.5))
        pos = m + label_out * out + np.array(label_nudge.get(p, (0, 0)))
        ax.text(*pos, rf"$J_{{{p}}}$", fontsize=label_size, color=col, ha="center", va="center", zorder=6)
        pts += [m, pos]

    for i, sgn in spins.items():
        col = col_up if sgn > 0 else col_down
        tip = site[i] + [0, arrow_len if sgn > 0 else -arrow_len]
        ax.add_patch(FancyArrowPatch(site[i], tip, arrowstyle="-|>", mutation_scale=arrow_head, lw=arrow_lw,
                                     color=col, shrinkA=0, shrinkB=0, zorder=5, capstyle="round"))
        ax.scatter(*site[i], s=14, color=col, edgecolor="white", lw=0.4, zorder=5.5)
        pts.append(tip)

    pts = np.vstack(pts + [c + [sx * ell_w / 2, sy * ell_h / 2] for c in centres.values() for sx in (-1, 1)
                           for sy in (-1, 1)])
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    V.save(fig, out_dir / "spin_model_scheme_2d", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "molecule_torsions"
    SIZE_CM = (4.0, 4.0)
    FONT = "Arial"
    LABEL_SIZE = 7                              # pt, J labels
    TRANSPARENT = True
    PAD = 0.15                                  # margin (drawing units)

    # geometry (drawing units; ellipse width = 1 molecule)
    ELL_W, ELL_H = 4.0, 1.0                     # ellipse axes
    SITE_FRAC = 0.72                            # spin sites at +-SITE_FRAC * half width from the centre
    SEP = 2.6                                   # vertical distance between the two molecules
    SHIFT_X = 0.6                               # horizontal shift of the upper molecule

    SPINS = {1: +1, 2: -1, 3: +1, 4: -1}        # +1 up (red), -1 down (blue); 1, 2 lower molecule, 3, 4 upper
    ARROW_LEN = 0.85
    ARROW_LW = 1.1
    ARROW_HEAD = 5
    COL_UP, COL_DOWN = "#C0392B", "#1F4E9A"

    MOL_FACE = ("#E8EEF4", "#E8EEF4")           # lower / upper molecule
    MOL_EDGE = ("#8C9BAA", "#8C9BAA")
    MOL_LW = 0.5

    J_COL = {"intra": "0.15", "inter": "0.45"}  # J12, J34 / intermolecular couplings
    J_LW = 0.55
    J_HEAD = 3
    J_STYLE = "<|-|>"                           # "-" for plain curves
    J_RAD = {"12": -0.4, "34": -0.4, "13": -0.45, "24": 0.45, "14": -0.15, "23": 0.15}   # curvature and side
    LABEL_T = {"12": 0.5, "34": 0.5, "13": 0.5, "24": 0.5, "14": 0.3, "23": 0.3}   # label position along each arc
    LABEL_OUT = 0.22                            # label distance from the arc, on its convex side
    LABEL_NUDGE = {"12": (0, 0), "34": (0, 0), "13": (0, 0), "24": (0, 0), "14": (0, 0), "23": (0, 0)}

    main(OUT_DIR, SIZE_CM, FONT, LABEL_SIZE, TRANSPARENT, ELL_W, ELL_H, SITE_FRAC, SEP, SHIFT_X, SPINS, ARROW_LEN,
         ARROW_LW, ARROW_HEAD, COL_UP, COL_DOWN, MOL_FACE, MOL_EDGE, MOL_LW, J_COL, J_LW, J_HEAD, J_STYLE, J_RAD,
         LABEL_T, LABEL_OUT, LABEL_NUDGE, PAD)
