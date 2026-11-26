#!/usr/bin/env python3
"""
Schematic junction of two stacked molecules, combining figure1/spin_model_scheme.py (two discs, four spins and
the six exchange couplings J_ij) with figure2/spin_model_junction.py (gold triangular electrodes, Kondo rings,
screening electrons).

Only the CONTACTED molecule (the upper one by default, CONTACT = 2) touches the electrodes: their tips meet the
two ends of its disc. The other molecule hangs below it, free. Concentric Kondo rings around the screened spin
(top-left, site 3) extend over the tip of the nearest electrode, where small down spins are the screening
conduction electrons; J_K labels the rings.

The figure size follows the drawing at a fixed scale (SCALE_CM cm per drawing unit, as the figure-2 junction), so
the molecules have the same size in both figures.
Output: OUT_DIR/spin_model_stack_junction.pdf / .png (Arial, transparent). Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D, ScaledTranslation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.spin_model_scheme import arc_point, slab  # noqa: E402
from visualisation.paper.figure2.spin_model_junction import electrode, offset_label, rings  # noqa: E402


def main(out_dir, scale_cm, font, label_size, transparent, pad, ell_w, ell_h, thick, site_frac, sep, shift_x, spins,
         arrow_len, arrow_lw, arrow_head, col_up, col_down, mol_face, mol_side, mol_edge, mol_lw, j_col, j_lw, j_head,
         j_style, j_rad, label_t, label_out, label_nudge, contact, el_length, el_height, el_light, el_dark, el_edge,
         el_lw, el_gap, kondo_site, kondo_colour, kondo_rx, kondo_ry, kondo_alpha, kondo_rings, kondo_line, kondo_lw,
         kondo_shift, lead_spins, lead_spin_len, lead_spin_lw, lead_spin_head, j_k, width_cm=None, removed=None,
         removed_style=None, out_name="spin_model_stack_junction"):
    """removed: site (1-4) whose spin is taken away: an empty dashed circle marks the site and the couplings to it
    are drawn barred (removed_style: colour, lw, ls, bar length / angle / position along the arc, labels)."""
    V.set_fonts(font)
    centres = {1: np.array([0.0, 0.0]), 2: np.array([shift_x, sep])}           # lower, upper molecule
    dx = site_frac * ell_w / 2
    site = {1: centres[1] - [dx, 0], 2: centres[1] + [dx, 0], 3: centres[2] - [dx, 0], 4: centres[2] + [dx, 0]}

    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    for k in (1, 2):
        slab(ax, centres[k], ell_w, ell_h, thick, mol_face, mol_side, mol_edge, mol_lw, 1)

    pts = []
    # electrodes: tips touch the two ends of the contacted molecule only
    cc = centres[contact]
    tips = {}
    for side, name in ((-1, "left"), (1, "right")):
        tip = tips[name] = cc + np.array([side * (ell_w / 2 + el_gap), -thick / 2])
        pts.append(electrode(ax, tip, side, el_length, el_height, el_light, el_dark, el_edge, el_lw,
                             False, kondo_colour, 1, 0, 0.2, 0.5, 1, kondo_line, kondo_lw))

    # exchange couplings (as spin_model_scheme.py)
    labels, spin_arrows = [], []
    for p, (a, b) in {"12": (1, 2), "34": (3, 4), "13": (1, 3), "24": (2, 4), "14": (1, 4), "23": (2, 3)}.items():
        intra = p in ("12", "34")
        col = j_col["intra" if intra else "inter"]
        gone = removed in (a, b)
        if gone:                                 # barred coupling: grey dashed arc without heads + a slash across it
            rs = removed_style
            col = rs["colour"]
            ax.add_patch(FancyArrowPatch(site[a], site[b], arrowstyle="-", lw=rs["lw"], color=col, ls=rs["ls"],
                                         connectionstyle=f"arc3,rad={j_rad[p]}", shrinkA=3, shrinkB=3, zorder=2,
                                         capstyle="butt"))
            q, nq = arc_point(site[a], site[b], j_rad[p], rs["bar_t"].get(p, 0.5))
            tq = np.array([nq[1], -nq[0]])                           # tangent at the bar
            ang = np.radians(rs["bar_angle"])
            d = 0.5 * rs["bar_len"] * (np.cos(ang) * tq + np.sin(ang) * nq)   # half bar, in points
            pt = (Affine2D().scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation(*q, ax.transData))
            ax.plot([-d[0], d[0]], [-d[1], d[1]], color=rs["bar_colour"], lw=rs["bar_lw"], transform=pt,
                    solid_capstyle="round", zorder=6, clip_on=False)
            if not rs["labels"]:
                continue
        else:
            ax.add_patch(FancyArrowPatch(site[a], site[b], arrowstyle=j_style, mutation_scale=j_head, lw=j_lw,
                                         color=col, connectionstyle=f"arc3,rad={j_rad[p]}", shrinkA=3, shrinkB=3,
                                         zorder=3 if intra else 2, capstyle="round"))
        m, out = arc_point(site[a], site[b], j_rad[p], label_t.get(p, 0.5))
        ha = "left" if out[0] > 0.35 else "right" if out[0] < -0.35 else "center"
        va = "bottom" if out[1] > 0.35 else "top" if out[1] < -0.35 else "center"
        off = label_out * out + np.array(label_nudge.get(p, (0, 0)))
        labels.append(ax.annotate(rf"$J_{{{p}}}$", m, xytext=off, textcoords="offset points", fontsize=label_size,
                                  color=col, ha=ha, va=va, zorder=6))
        pts.append([m])

    # spins out of the plane; down spins come out of the lower side of their disc
    for i, sgn in spins.items():
        if i == removed:                         # empty site: dashed hollow circle, no spin
            ax.add_patch(Circle(site[i], removed_style["site_r"], facecolor="white", ls=(0, (1.5, 1)),
                                edgecolor=removed_style["colour"], lw=removed_style["lw"], zorder=5.5))
            continue
        col = col_up if sgn > 0 else col_down
        c = centres[1 if i in (1, 2) else 2]
        if sgn > 0:
            tip = site[i] + [0, arrow_len]
        else:
            rim = c[1] - thick - ell_h / 2 * np.sqrt(max(0.0, 1 - ((site[i][0] - c[0]) / (ell_w / 2)) ** 2))
            tip = np.array([site[i][0], rim - arrow_len])
        z = 5 if sgn > 0 else 0.5                                # down spins under their disc
        spin_arrows.append(ax.add_patch(FancyArrowPatch(site[i], tip, arrowstyle="-|>", mutation_scale=arrow_head,
                                                        lw=arrow_lw, color=col, shrinkA=0, shrinkB=0, zorder=z,
                                                        capstyle="round")))
        ax.scatter(*site[i], s=14, color=col, edgecolor="white", lw=0.4, zorder=5.5)
        pts.append([tip])

    # Kondo rings around the screened spin, shifted towards the nearest electrode tip
    side = -1 if site[kondo_site][0] < centres[contact][0] else 1
    centre = site[kondo_site] + np.array([side * -kondo_shift[0], kondo_shift[1]])
    rings(ax, centre, kondo_rx, kondo_ry, kondo_rings, kondo_colour, kondo_alpha, kondo_line, kondo_lw, 3.5)
    lead_tip = tips["left" if side < 0 else "right"]
    for px, py in lead_spins:                                       # screening electrons in that electrode
        mid = lead_tip + np.array([side * -px, py])
        ax.add_patch(FancyArrowPatch(mid + [0, lead_spin_len / 2], mid - [0, lead_spin_len / 2], arrowstyle="-|>",
                                     mutation_scale=lead_spin_head, lw=lead_spin_lw, color=col_down, shrinkA=0,
                                     shrinkB=0, zorder=4, capstyle="round"))
    if j_k:
        t = np.radians(j_k["angle"])
        on_ring = centre + [kondo_rx * np.cos(t), kondo_ry * np.sin(t)]
        normal = [np.cos(t) / kondo_rx, np.sin(t) / kondo_ry]
        labels.append(offset_label(ax, on_ring, normal, j_k["label"], j_k["label_out"], label_size, 6))
    pts.append([centre - [kondo_rx, kondo_ry], centre + [kondo_rx, kondo_ry]])

    pts = np.vstack(pts + [[c + [sx * ell_w / 2, sy * ell_h / 2 - (thick if sy < 0 else 0)]
                            for c in centres.values() for sx in (-1, 1) for sy in (-1, 1)]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for _ in range(30):                      # figure size from the extent at the scale; grow until labels fit
        x0, x1, y0, y1 = lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad
        if width_cm:                         # fixed width: scale from the current extent (labels are in pt,
            scale_cm = width_cm / (x1 - x0)  # so a smaller scale makes them larger in drawing units: iterate)
        fig.set_size_inches((x1 - x0) * scale_cm / 2.54, (y1 - y0) * scale_cm / 2.54)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(r) for t in labels]
        inv = ax.transData.inverted()
        corners = np.vstack([inv.transform([[b.x0, b.y0], [b.x1, b.y1]]) for b in boxes])
        new_lo, new_hi = np.minimum(lo, corners.min(axis=0)), np.maximum(hi, corners.max(axis=0))
        if np.allclose(new_lo, lo) and np.allclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    for t, b in zip(labels, boxes):
        if any(b.overlaps(a.get_window_extent(r)) for a in spin_arrows):
            print(f"  [!] {t.get_text()} overlaps a spin arrow: adjust LABEL_T / LABEL_NUDGE")
    w, h = fig.get_size_inches() * 2.54
    print(f"figure size {w:.2f} x {h:.2f} cm")
    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / out_name, transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    SCALE_CM = 0.58                             # cm per drawing unit (figure-2 junction: ~0.58); sets the size
    WIDTH_CM = 3.5                              # figure width (cm): overrides SCALE_CM (None: use SCALE_CM)
    FONT = "Arial"
    LABEL_SIZE = 7                              # pt, J labels
    TRANSPARENT = True
    PAD = 0.1                                   # margin (drawing units)

    # molecules and spins (as figure1/spin_model_scheme.py)
    ELL_W, ELL_H = 4.0, 1.0
    THICK = 0.12
    SITE_FRAC = 0.72
    SEP = 2.6                                   # vertical distance between the two molecules
    SHIFT_X = 0.6                               # horizontal shift of the upper molecule
    SPINS = {1: +1, 2: -1, 3: +1, 4: -1}        # +1 up (red), -1 down (blue); 1, 2 lower molecule, 3, 4 upper
    ARROW_LEN = 0.85
    ARROW_LW = 1.1
    ARROW_HEAD = 5
    COL_UP, COL_DOWN = "#C0392B", "#1F4E9A"
    MOL_FACE, MOL_SIDE, MOL_EDGE = "#E8EEF4", "#C5D0DB", "#8C9BAA"
    MOL_LW = 0.4

    J_COL = {"intra": "black", "inter": "black"}
    J_LW = 0.55
    J_HEAD = 3
    J_STYLE = "<|-|>"
    J_RAD = {"12": -0.4, "34": -0.4, "13": -0.45, "24": 0.45, "14": -0.15, "23": 0.15}
    LABEL_T = {"12": 0.5, "34": 0.5, "13": 0.5, "24": 0.5, "14": 0.3, "23": 0.3}
    LABEL_OUT = 1.5                             # pt between an arc and its label
    LABEL_NUDGE = {}                            # extra label shifts in pt, e.g. {"13": (-2, 0)}

    # electrodes (as figure2/spin_model_junction.py), touching only molecule CONTACT (1 lower, 2 upper)
    CONTACT = 2
    EL_LENGTH = 1.2                             # short electrodes: more of the 3 cm width for the molecules
    EL_HEIGHT = 1.5
    EL_LIGHT, EL_DARK = "#F3DE8A", "#D4AF37"
    EL_EDGE = "#A8862A"
    EL_LW = 0.4
    EL_GAP = 0.0

    # Kondo screening of the top-left spin (site 3) with the left electrode
    KONDO_SITE = 3
    KONDO_SHIFT = (-0.3, 0.0)                   # ring centre relative to the spin (negative x = towards the electrode)
    KONDO_COLOUR = "#8E7CC3"
    KONDO_RX, KONDO_RY = 1.0, 0.7
    KONDO_RINGS = 5
    KONDO_ALPHA = 0.45
    KONDO_LINE = "#5E4A9C"
    KONDO_LW = 0.5
    LEAD_SPINS = [(-0.35, 0.0), (-0.65, 0.15), (-0.7, -0.17), (-0.98, 0.0)]   # relative to the electrode tip
    LEAD_SPIN_LEN = 0.32
    LEAD_SPIN_LW = 0.7
    LEAD_SPIN_HEAD = 3.5
    J_K = {"angle": 145,                        # deg, position on the outer Kondo ring (0 = right, 90 = top)
           "label": r"$J_\mathrm{K}$", "label_out": 1.5}
    # J_K = None

    main(OUT_DIR, SCALE_CM, FONT, LABEL_SIZE, TRANSPARENT, PAD, ELL_W, ELL_H, THICK, SITE_FRAC, SEP, SHIFT_X, SPINS,
         ARROW_LEN, ARROW_LW, ARROW_HEAD, COL_UP, COL_DOWN, MOL_FACE, MOL_SIDE, MOL_EDGE, MOL_LW, J_COL, J_LW, J_HEAD,
         J_STYLE, J_RAD, LABEL_T, LABEL_OUT, LABEL_NUDGE, CONTACT, EL_LENGTH, EL_HEIGHT, EL_LIGHT, EL_DARK, EL_EDGE,
         EL_LW, EL_GAP, KONDO_SITE, KONDO_COLOUR, KONDO_RX, KONDO_RY, KONDO_ALPHA, KONDO_RINGS, KONDO_LINE, KONDO_LW,
         KONDO_SHIFT, LEAD_SPINS, LEAD_SPIN_LEN, LEAD_SPIN_LW, LEAD_SPIN_HEAD, J_K, WIDTH_CM)
